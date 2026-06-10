"""
ETF Net Flows Pipeline
========================
Tracks institutional money movement through pillar-mapped ETFs.
Uses yfinance (free) for price/volume data, calculates flow proxies,
stores in Supabase, pushes to Google Sheets.

WHAT IT MEASURES:
  ETF "net flow" proxy = Daily Dollar Volume * Price Direction
  Positive = money flowing IN (buying pressure)
  Negative = money flowing OUT (selling pressure)
  This is the standard On-Balance Volume Dollar variant used by
  institutional flow trackers when actual creation/redemption data
  isn't available in real-time.

PILLAR MAPPING:
  Infrastructure: SMH (0.4), SOXX (0.3), XLU (0.15), GRID (0.15)
  Enterprise:     IGV (0.5), WCLD (0.3), AIQ (0.2)
  Financial:      XLF (0.5), KRE (0.3), ARKF (0.2)
  Macro:          TLT (0.4), GLD (0.3), UUP (0.3)
  Productivity:   BOTZ (0.5), ROBO (0.3), ARKQ (0.2)
  Demand:         XLY (0.5), IBUY (0.25), ONLN (0.25)
  Benchmark:      SPY (0.5), QQQ (0.5)

SIGNALS:
  Flow_Z > 2.0   = Strong inflow (institutional accumulation)
  Flow_Z < -2.0  = Strong outflow (institutional distribution)
  Pillar divergence = one pillar inflow while others outflow = rotation

OUTPUT:
  Supabase: etf_flows_history (per-ETF daily) + etf_flows_pillar (pillar aggregates)
  Sheets:   Macro_and_Options_History -> ETF_NetFlows tab

Usage:
  python etf_flows_pipeline.py test              # Test with 3 ETFs
  python etf_flows_pipeline.py daily             # Fetch all ETFs, upload
  python etf_flows_pipeline.py backfill          # Backfill 2 years
  python etf_flows_pipeline.py backfill --days 90  # Backfill 90 days
  python etf_flows_pipeline.py check             # Show latest pillar flows
  python etf_flows_pipeline.py check SMH         # Show SMH history
  python etf_flows_pipeline.py push              # Push to Google Sheets

Requires: pip install yfinance python-dotenv supabase
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    import yfinance as yf
except ImportError:
    print("yfinance required. Install with: pip install yfinance")
    sys.exit(1)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
except Exception:
    supabase = None
    logger.warning("Supabase not configured.")

# ============================================================
# CONFIGURATION
# ============================================================

SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "ETF_NetFlows"

# ETF Universe mapped to ARGUS-1 pillars
# Weights within each pillar sum to 1.0
ETF_UNIVERSE = [
    # Infrastructure & Energy
    {"ticker": "SMH",  "pillar": "Infrastructure", "weight": 0.40, "name": "VanEck Semiconductor"},
    {"ticker": "SOXX", "pillar": "Infrastructure", "weight": 0.30, "name": "iShares Semiconductor"},
    {"ticker": "XLU",  "pillar": "Infrastructure", "weight": 0.15, "name": "Utilities Select"},
    {"ticker": "GRID", "pillar": "Infrastructure", "weight": 0.15, "name": "First Trust Clean Edge"},

    # Enterprise Adoption
    {"ticker": "IGV",  "pillar": "Enterprise",     "weight": 0.50, "name": "iShares Software"},
    {"ticker": "WCLD", "pillar": "Enterprise",     "weight": 0.30, "name": "WisdomTree Cloud"},
    {"ticker": "AIQ",  "pillar": "Enterprise",     "weight": 0.20, "name": "Global X AI & Tech"},

    # Financial & Market
    {"ticker": "XLF",  "pillar": "Financial",      "weight": 0.50, "name": "Financial Select"},
    {"ticker": "KRE",  "pillar": "Financial",      "weight": 0.30, "name": "Regional Banks"},
    {"ticker": "ARKF", "pillar": "Financial",      "weight": 0.20, "name": "ARK Fintech"},

    # Macro & Policy
    {"ticker": "TLT",  "pillar": "Macro",          "weight": 0.40, "name": "20+ Year Treasury"},
    {"ticker": "GLD",  "pillar": "Macro",          "weight": 0.30, "name": "Gold Trust"},
    {"ticker": "UUP",  "pillar": "Macro",          "weight": 0.30, "name": "US Dollar Index"},

    # Productivity & Labor
    {"ticker": "BOTZ", "pillar": "Productivity",   "weight": 0.50, "name": "Global Robotics & AI"},
    {"ticker": "ROBO", "pillar": "Productivity",   "weight": 0.30, "name": "Robo Global Robotics"},
    {"ticker": "ARKQ", "pillar": "Productivity",   "weight": 0.20, "name": "ARK Autonomous Tech"},

    # Demand Dynamics
    {"ticker": "XLY",  "pillar": "Demand",         "weight": 0.50, "name": "Consumer Discretionary"},
    {"ticker": "IBUY", "pillar": "Demand",         "weight": 0.25, "name": "Amplify Online Retail"},
    {"ticker": "ONLN", "pillar": "Demand",         "weight": 0.25, "name": "ProShares Online Retail"},

    # Benchmarks
    {"ticker": "SPY",  "pillar": "Benchmark",      "weight": 0.50, "name": "S&P 500"},
    {"ticker": "QQQ",  "pillar": "Benchmark",      "weight": 0.50, "name": "Nasdaq 100"},

    # Energy (tracked-only: not in PILLAR_WEIGHTS, excluded from regime score)
    {"ticker": "XOP",  "pillar": "Energy",         "weight": 0.30, "name": "SPDR S&P Oil & Gas E&P"},
    {"ticker": "XLE",  "pillar": "Energy",         "weight": 0.50, "name": "Energy Select Sector"},
    {"ticker": "CRAK", "pillar": "Energy",         "weight": 0.20, "name": "VanEck Oil Refiners"},

    # Real Estate (tracked-only: not in PILLAR_WEIGHTS, excluded from regime score)
    {"ticker": "ITB",  "pillar": "Real Estate",    "weight": 0.25, "name": "iShares U.S. Home Construction"},
    {"ticker": "XHB",  "pillar": "Real Estate",    "weight": 0.25, "name": "SPDR S&P Homebuilders"},
    {"ticker": "XLRE", "pillar": "Real Estate",    "weight": 0.50, "name": "Real Estate Select Sector"},
]

ETF_TICKERS = [e["ticker"] for e in ETF_UNIVERSE]
TICKER_META = {e["ticker"]: e for e in ETF_UNIVERSE}

# Pillar weights for overall regime score
PILLAR_WEIGHTS = {
    "Infrastructure": 0.30,
    "Enterprise":     0.25,
    "Financial":      0.15,
    "Macro":          0.10,
    "Productivity":   0.10,
    "Demand":         0.10,
}


# ============================================================
# DATA FETCHING (yfinance)
# ============================================================
def fetch_etf_data(tickers, period="5d"):
    """
    Fetch OHLCV data for ETF tickers via yfinance.
    Returns DataFrame with date, ticker, close, volume, pct_change.
    """
    all_rows = []

    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period=period)

            if hist.empty:
                logger.warning(f"  {ticker}: No data returned")
                continue

            for date, row in hist.iterrows():
                all_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                })

        except Exception as e:
            logger.warning(f"  {ticker}: Error - {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def fetch_etf_data_range(tickers, start_date, end_date):
    """
    Fetch OHLCV data for a specific date range.
    More efficient for backfill - downloads all tickers at once.
    """
    logger.info(f"  Downloading {len(tickers)} ETFs: {start_date} to {end_date}")

    try:
        # yfinance bulk download
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.error(f"  Bulk download failed: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    all_rows = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]

            if ticker_data.empty:
                continue

            ticker_data = ticker_data.dropna(subset=["Close"])

            for date, row in ticker_data.iterrows():
                all_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    "high": round(float(row["High"]), 2) if pd.notna(row["High"]) else None,
                    "low": round(float(row["Low"]), 2) if pd.notna(row["Low"]) else None,
                })
        except Exception as e:
            logger.warning(f"  {ticker}: Parse error - {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info(f"  Downloaded {len(df)} rows across {df['ticker'].nunique()} ETFs")
    return df


# ============================================================
# CALCULATIONS
# ============================================================
def calculate_flow_metrics(df):
    """
    Calculate flow metrics per ETF ticker.

    Dollar Flow = Volume * Close * sign(daily return)
      Positive close-to-close = inflow day
      Negative close-to-close = outflow day

    Also calculates:
      flow_20d_avg: 20-day rolling average of dollar flow
      flow_z_score: Z-score of today's flow vs 20-day window
      volume_ratio: today's volume / 20-day average volume
    """
    result_dfs = []

    for ticker in df["ticker"].unique():
        tdf = df[df["ticker"] == ticker].sort_values("date").copy()

        # Daily return
        tdf["pct_change"] = tdf["close"].pct_change()

        # Dollar volume
        tdf["dollar_volume"] = tdf["volume"] * tdf["close"]

        # Flow proxy: dollar volume * direction
        # Positive return = inflow, negative = outflow
        tdf["dollar_flow"] = tdf["dollar_volume"] * np.sign(tdf["pct_change"])
        tdf["dollar_flow"] = tdf["dollar_flow"].round(0)

        # Convert to millions for readability
        tdf["flow_mm"] = (tdf["dollar_flow"] / 1_000_000).round(2)

        # 20-day rolling metrics
        tdf["flow_20d_avg"] = tdf["flow_mm"].rolling(window=20, min_periods=10).mean().round(2)

        flow_std = tdf["flow_mm"].rolling(window=20, min_periods=10).std()
        tdf["flow_z_score"] = np.where(
            flow_std > 0,
            ((tdf["flow_mm"] - tdf["flow_20d_avg"]) / flow_std).round(4),
            0
        )

        # Volume ratio (today vs 20d avg)
        vol_20d = tdf["volume"].rolling(window=20, min_periods=10).mean()
        tdf["volume_ratio"] = np.where(
            vol_20d > 0,
            (tdf["volume"] / vol_20d).round(2),
            1.0
        )

        # Flow signal classification
        tdf["flow_signal"] = tdf.apply(
            lambda r: classify_flow(r.get("flow_z_score", 0), r.get("volume_ratio", 1)),
            axis=1
        )

        result_dfs.append(tdf)

    if not result_dfs:
        return df

    return pd.concat(result_dfs, ignore_index=True)


def classify_flow(z_score, vol_ratio):
    """Classify flow signal based on Z-score and volume ratio."""
    if pd.isna(z_score):
        return "Neutral"

    if z_score > 2.0 and vol_ratio > 1.3:
        return "Strong Inflow"
    elif z_score > 1.0:
        return "Inflow"
    elif z_score < -2.0 and vol_ratio > 1.3:
        return "Strong Outflow"
    elif z_score < -1.0:
        return "Outflow"
    else:
        return "Neutral"


def calculate_pillar_flows(df_latest):
    """
    Aggregate ETF flows into pillar-level signals.
    Uses weighted average of flow_z_score within each pillar.
    Returns dict of pillar -> {score, signal, etf_details}.
    """
    pillar_results = {}

    for pillar in PILLAR_WEIGHTS.keys():
        pillar_etfs = [e for e in ETF_UNIVERSE if e["pillar"] == pillar]
        pillar_tickers = [e["ticker"] for e in pillar_etfs]

        pillar_data = df_latest[df_latest["ticker"].isin(pillar_tickers)]

        if pillar_data.empty:
            pillar_results[pillar] = {
                "weighted_z": 0,
                "signal": "No Data",
                "total_flow_mm": 0,
                "etf_count": 0,
            }
            continue

        # Weighted Z-score
        weighted_z = 0
        total_weight = 0
        total_flow = 0

        for _, row in pillar_data.iterrows():
            meta = TICKER_META.get(row["ticker"], {})
            w = meta.get("weight", 0)
            z = row.get("flow_z_score", 0) or 0
            flow = row.get("flow_mm", 0) or 0

            weighted_z += z * w
            total_weight += w
            total_flow += flow

        if total_weight > 0:
            weighted_z = round(weighted_z / total_weight, 4)

        # Classify pillar
        if weighted_z > 1.5:
            signal = "Strong Inflow"
        elif weighted_z > 0.5:
            signal = "Inflow"
        elif weighted_z < -1.5:
            signal = "Strong Outflow"
        elif weighted_z < -0.5:
            signal = "Outflow"
        else:
            signal = "Neutral"

        pillar_results[pillar] = {
            "weighted_z": weighted_z,
            "signal": signal,
            "total_flow_mm": round(total_flow, 2),
            "etf_count": len(pillar_data),
        }

    return pillar_results


def determine_regime(pillar_flows):
    """
    Determine overall flow regime from pillar signals.
    Weighted by PILLAR_WEIGHTS.
    """
    weighted_score = 0
    for pillar, weight in PILLAR_WEIGHTS.items():
        pf = pillar_flows.get(pillar, {})
        weighted_score += pf.get("weighted_z", 0) * weight

    if weighted_score > 1.0:
        return "Risk-On"
    elif weighted_score > 0.3:
        return "Mild Risk-On"
    elif weighted_score < -1.0:
        return "Risk-Off"
    elif weighted_score < -0.3:
        return "Mild Risk-Off"
    else:
        return "Neutral"


# ============================================================
# SUPABASE
# ============================================================
def upload_to_supabase(df):
    """Upload ETF flow data to etf_flows_history table."""
    if supabase is None:
        logger.warning("Supabase not configured. Skipping upload.")
        return 0

    records = []
    for _, row in df.iterrows():
        record = {
            "date": str(row.get("date", "")),
            "ticker": str(row.get("ticker", "")),
            "close": float(row["close"]) if pd.notna(row.get("close")) else None,
            "volume": int(row["volume"]) if pd.notna(row.get("volume")) else None,
            "pct_change": round(float(row["pct_change"]), 6) if pd.notna(row.get("pct_change")) else None,
            "dollar_volume": round(float(row["dollar_volume"]), 0) if pd.notna(row.get("dollar_volume")) else None,
            "flow_mm": float(row["flow_mm"]) if pd.notna(row.get("flow_mm")) else None,
            "flow_20d_avg": float(row["flow_20d_avg"]) if pd.notna(row.get("flow_20d_avg")) else None,
            "flow_z_score": float(row["flow_z_score"]) if pd.notna(row.get("flow_z_score")) else None,
            "volume_ratio": float(row["volume_ratio"]) if pd.notna(row.get("volume_ratio")) else None,
            "flow_signal": str(row.get("flow_signal", "Neutral")),
            "pillar": TICKER_META.get(row["ticker"], {}).get("pillar", ""),
            "pillar_weight": TICKER_META.get(row["ticker"], {}).get("weight", 0),
        }
        records.append(record)

    if not records:
        return 0

    BATCH_SIZE = 500
    uploaded = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("etf_flows_history") \
                .upsert(batch, on_conflict="date,ticker") \
                .execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


def load_from_supabase(ticker=None, limit=None):
    """Load etf_flows_history from Supabase."""
    if supabase is None:
        return pd.DataFrame()

    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        query = supabase.table("etf_flows_history").select("*").order("date", desc=True)
        if ticker:
            query = query.eq("ticker", ticker)
        query = query.range(offset, offset + page_size - 1)
        result = query.execute()

        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        if limit and len(all_rows) >= limit:
            break
        offset += page_size

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets():
    """Push etf_flows_history to Google Sheets."""
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading etf_flows_history from Supabase...")
    df = load_from_supabase()

    if df.empty:
        logger.info("No data to push.")
        return

    logger.info(f"  {len(df)} rows loaded")

    headers = [
        "Date", "Ticker", "Pillar", "Close", "Volume",
        "Pct_Change", "Flow_MM", "Flow_20D_Avg",
        "Flow_Z_Score", "Volume_Ratio", "Flow_Signal"
    ]

    sheet_rows = []
    for _, row in df.iterrows():
        sheet_rows.append([
            "'" + str(row.get("date", "")),
            row.get("ticker", ""),
            row.get("pillar", ""),
            row.get("close", ""),
            row.get("volume", ""),
            row.get("pct_change", ""),
            row.get("flow_mm", ""),
            row.get("flow_20d_avg", ""),
            row.get("flow_z_score", ""),
            row.get("volume_ratio", ""),
            row.get("flow_signal", ""),
        ])

    # Clean NaN
    for row in sheet_rows:
        for j in range(len(row)):
            if row[j] is None or (isinstance(row[j], float) and np.isnan(row[j])):
                row[j] = ""

    creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(SHEETS_SPREADSHEET)
    except Exception:
        logger.error(f"Spreadsheet '{SHEETS_SPREADSHEET}' not found.")
        return

    # Get or create tab
    try:
        sheet = spreadsheet.worksheet(SHEETS_TAB)
        sheet.clear()
        if sheet.row_count < len(sheet_rows) + 10:
            sheet.resize(rows=len(sheet_rows) + 100, cols=len(headers))
    except Exception:
        sheet = spreadsheet.add_worksheet(
            title=SHEETS_TAB,
            rows=len(sheet_rows) + 100,
            cols=len(headers)
        )

    sheet.update(range_name="A1", values=[headers], value_input_option="USER_ENTERED")
    sheet.format("1:1", {"textFormat": {"bold": True}})

    BATCH_SIZE = 5000
    total = 0
    for i in range(0, len(sheet_rows), BATCH_SIZE):
        batch = sheet_rows[i:i + BATCH_SIZE]
        sheet.update(range_name=f"A{i+2}", values=batch, value_input_option="USER_ENTERED")
        total += len(batch)
        logger.info(f"  Written {total}/{len(sheet_rows)} rows")
        time.sleep(1)

    logger.info(f"Push complete: {total} rows")


# ============================================================
# COMMANDS
# ============================================================
def run_daily():
    """Fetch latest ETF data and calculate flows."""
    logger.info("=" * 60)
    logger.info(f"ETF NET FLOWS - DAILY ({len(ETF_TICKERS)} ETFs)")
    logger.info("=" * 60)

    # Fetch 30 days to ensure rolling calcs work
    df = fetch_etf_data(ETF_TICKERS, period="1mo")

    if df.empty:
        logger.error("No data fetched.")
        return

    logger.info(f"  Fetched {len(df)} rows, {df['ticker'].nunique()} ETFs")

    # Load existing data for rolling calculations
    existing = load_from_supabase()
    if not existing.empty:
        keep_cols = ["date", "ticker", "close", "volume"]
        keep_cols = [c for c in keep_cols if c in existing.columns]
        combined = pd.concat([existing[keep_cols], df[keep_cols]], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        combined = df

    # Calculate metrics
    combined = calculate_flow_metrics(combined)

    # Get latest date
    latest_date = combined["date"].max()
    today_data = combined[combined["date"] == latest_date]

    # Upload only today
    uploaded = upload_to_supabase(today_data)

    # Pillar analysis
    pillar_flows = calculate_pillar_flows(today_data)
    regime = determine_regime(pillar_flows)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY COMPLETE - {latest_date}")
    logger.info(f"  ETFs fetched: {today_data['ticker'].nunique()}/{len(ETF_TICKERS)}")
    logger.info(f"  Uploaded:     {uploaded} rows")
    logger.info(f"  Regime:       {regime}")
    logger.info(f"\n  PILLAR FLOWS:")

    for pillar in ["Infrastructure", "Enterprise", "Financial", "Macro", "Productivity", "Demand"]:
        pf = pillar_flows.get(pillar, {})
        z = pf.get("weighted_z", 0)
        sig = pf.get("signal", "?")
        flow = pf.get("total_flow_mm", 0)
        logger.info(f"    {pillar:<16} Z={z:+.2f}  Flow=${flow:>8.1f}M  {sig}")

    # Alerts
    alerts = today_data[today_data["flow_z_score"].abs() > 2.0]
    if not alerts.empty:
        logger.info(f"\n  ALERTS ({len(alerts)}):")
        for _, row in alerts.iterrows():
            direction = "INFLOW" if row["flow_z_score"] > 0 else "OUTFLOW"
            logger.info(f"    {direction}: {row['ticker']:<6} "
                        f"Z={row['flow_z_score']:+.2f}  "
                        f"Flow=${row['flow_mm']:>8.1f}M  "
                        f"VolRatio={row['volume_ratio']:.1f}x")


def run_backfill(days=730):
    """Backfill ETF flow history."""
    logger.info("=" * 60)
    logger.info(f"ETF NET FLOWS - BACKFILL ({days} days)")
    logger.info(f"ETFs: {len(ETF_TICKERS)}")
    logger.info("=" * 60)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = fetch_etf_data_range(ETF_TICKERS, start_date, end_date)

    if df.empty:
        logger.error("No data fetched.")
        return

    logger.info(f"  Raw data: {len(df)} rows")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  ETFs: {df['ticker'].nunique()}")

    # Calculate metrics
    df = calculate_flow_metrics(df)

    # Drop rows with no flow data (first row per ticker has NaN pct_change)
    df = df.dropna(subset=["pct_change"])

    logger.info(f"  After calculations: {len(df)} rows")

    # Upload to Supabase
    uploaded = upload_to_supabase(df)
    logger.info(f"  Uploaded: {uploaded} rows")

    # Summary by pillar
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    pillar_flows = calculate_pillar_flows(latest)

    logger.info(f"\n  Latest pillar flows ({latest_date}):")
    for pillar in ["Infrastructure", "Enterprise", "Financial", "Macro", "Productivity", "Demand"]:
        pf = pillar_flows.get(pillar, {})
        logger.info(f"    {pillar:<16} Z={pf.get('weighted_z', 0):+.2f}  {pf.get('signal', '?')}")

    logger.info(f"\nBACKFILL COMPLETE")


def run_test():
    """Test with 3 ETFs."""
    test_tickers = ["SMH", "XLF", "SPY"]

    logger.info("=" * 60)
    logger.info("ETF NET FLOWS - TEST (3 ETFs)")
    logger.info("=" * 60)

    df = fetch_etf_data(test_tickers, period="1mo")

    if df.empty:
        logger.error("No data fetched. Check internet connection.")
        return

    df = calculate_flow_metrics(df)
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    logger.info(f"\n  Latest: {latest_date}")
    logger.info(f"  {'Ticker':<8} {'Close':>8} {'Volume':>12} {'Flow $M':>10} {'Z-Score':>8} {'Signal'}")
    logger.info(f"  {'-'*60}")

    for _, row in latest.iterrows():
        logger.info(f"  {row['ticker']:<8} "
                     f"${row['close']:>7.2f} "
                     f"{row['volume']:>12,} "
                     f"${row.get('flow_mm', 0):>9.1f} "
                     f"{row.get('flow_z_score', 0):>+8.2f} "
                     f"{row.get('flow_signal', 'N/A')}")

    # Check Supabase
    logger.info(f"\n  Supabase:")
    if supabase:
        try:
            result = supabase.table("etf_flows_history") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if result.data:
                logger.info(f"    Table exists, latest: {result.data[0].get('date')}")
            else:
                logger.info(f"    Table exists but empty. Run 'backfill' to start.")
        except Exception as e:
            logger.info(f"    Table check: {e}")
            logger.info(f"    Run schema SQL first (see below).")
    else:
        logger.info(f"    Not configured.")

    logger.info("\nTEST COMPLETE")


def run_check(ticker=None):
    """Show flow data."""
    logger.info("=" * 60)
    logger.info("ETF NET FLOWS - CHECK")
    logger.info("=" * 60)

    if supabase is None:
        logger.error("Supabase not configured.")
        return

    if ticker:
        result = supabase.table("etf_flows_history") \
            .select("*") \
            .eq("ticker", ticker.upper()) \
            .order("date", desc=True) \
            .limit(20) \
            .execute()

        if not result.data:
            logger.info(f"No data for {ticker.upper()}.")
            return

        meta = TICKER_META.get(ticker.upper(), {})
        logger.info(f"\n  {ticker.upper()} - {meta.get('name', '')} ({meta.get('pillar', '')})")
        logger.info(f"  {'Date':<12} {'Close':>8} {'Flow $M':>10} {'Z-Score':>8} {'VolRatio':>9} {'Signal'}")
        logger.info(f"  {'-'*65}")

        for row in result.data:
            d = row.get("date", "?")
            close = row.get("close", 0)
            flow = row.get("flow_mm", 0)
            z = row.get("flow_z_score")
            vr = row.get("volume_ratio")
            sig = row.get("flow_signal", "?")
            z_s = f"{z:+.2f}" if z else "N/A"
            vr_s = f"{vr:.1f}x" if vr else "N/A"
            flag = " <<<" if z and abs(z) > 2.0 else ""
            logger.info(f"  {d:<12} ${close:>7.2f} ${flow:>9.1f} {z_s:>8} {vr_s:>9} {sig}{flag}")

    else:
        # Show latest pillar summary
        result = supabase.table("etf_flows_history") \
            .select("*") \
            .order("date", desc=True) \
            .limit(len(ETF_TICKERS) * 2) \
            .execute()

        if not result.data:
            logger.info("No data. Run 'backfill' or 'daily' first.")
            return

        latest_date = result.data[0].get("date")
        today_rows = [r for r in result.data if r.get("date") == latest_date]

        logger.info(f"\n  Latest: {latest_date}")
        logger.info(f"  {'Ticker':<6} {'Pillar':<16} {'Flow $M':>10} {'Z-Score':>8} {'Signal'}")
        logger.info(f"  {'-'*55}")

        for row in sorted(today_rows, key=lambda r: r.get("flow_z_score", 0) or 0, reverse=True):
            t = row.get("ticker", "?")
            p = row.get("pillar", "?")
            flow = row.get("flow_mm", 0)
            z = row.get("flow_z_score")
            sig = row.get("flow_signal", "?")
            z_s = f"{z:+.2f}" if z else "N/A"
            flag = " <<<" if z and abs(z) > 2.0 else ""
            logger.info(f"  {t:<6} {p:<16} ${flow:>9.1f} {z_s:>8} {sig}{flag}")

        # Pillar summary
        df_latest = pd.DataFrame(today_rows)
        if not df_latest.empty:
            pillar_flows = calculate_pillar_flows(df_latest)
            regime = determine_regime(pillar_flows)

            logger.info(f"\n  PILLAR SUMMARY:")
            for pillar in ["Infrastructure", "Enterprise", "Financial", "Macro", "Productivity", "Demand"]:
                pf = pillar_flows.get(pillar, {})
                logger.info(f"    {pillar:<16} Z={pf.get('weighted_z', 0):+.2f}  {pf.get('signal', '?')}")
            logger.info(f"\n  Overall Regime: {regime}")


def run_push():
    """Push to Google Sheets."""
    logger.info("=" * 60)
    logger.info("ETF NET FLOWS - PUSH TO SHEETS")
    logger.info("=" * 60)
    push_to_sheets()


def print_schema():
    """Print the Supabase schema SQL."""
    schema = """
-- ============================================================
-- ETF Flows History - Per-ETF daily flow data
-- ============================================================
CREATE TABLE IF NOT EXISTS etf_flows_history (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    close NUMERIC(10,2),
    volume BIGINT,
    pct_change NUMERIC(10,6),
    dollar_volume NUMERIC(18,0),
    flow_mm NUMERIC(12,2),
    flow_20d_avg NUMERIC(12,2),
    flow_z_score NUMERIC(8,4),
    volume_ratio NUMERIC(6,2),
    flow_signal TEXT,
    pillar TEXT,
    pillar_weight NUMERIC(4,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(date, ticker)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_etf_flows_date ON etf_flows_history(date DESC);
CREATE INDEX IF NOT EXISTS idx_etf_flows_ticker ON etf_flows_history(ticker);
CREATE INDEX IF NOT EXISTS idx_etf_flows_pillar ON etf_flows_history(pillar);
CREATE INDEX IF NOT EXISTS idx_etf_flows_signal ON etf_flows_history(flow_signal)
    WHERE flow_signal IN ('Strong Inflow', 'Strong Outflow');
"""
    print(schema)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ETF Net Flows Pipeline")
        print("=" * 50)
        print()
        print(f"ETFs:    {len(ETF_TICKERS)} across 6 pillars + benchmarks")
        print(f"Source:  yfinance (free)")
        print(f"Signal:  |Flow_Z_Score| > 2.0 = unusual flow")
        print()
        print("Usage:")
        print("  python etf_flows_pipeline.py test              # Test 3 ETFs")
        print("  python etf_flows_pipeline.py daily             # Daily update")
        print("  python etf_flows_pipeline.py backfill          # Backfill 2 years")
        print("  python etf_flows_pipeline.py backfill --days 90  # Backfill 90 days")
        print("  python etf_flows_pipeline.py check             # Pillar summary")
        print("  python etf_flows_pipeline.py check SMH         # ETF history")
        print("  python etf_flows_pipeline.py push              # Push to Sheets")
        print("  python etf_flows_pipeline.py schema            # Print SQL schema")
        print()
        print("ETF Universe:")
        current_pillar = ""
        for etf in ETF_UNIVERSE:
            if etf["pillar"] != current_pillar:
                current_pillar = etf["pillar"]
                print(f"\n  {current_pillar}:")
            print(f"    {etf['ticker']:<6} {etf['name']:<30} weight={etf['weight']}")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test":
        run_test()
    elif cmd == "daily":
        run_daily()
    elif cmd == "backfill":
        days = 730
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                days = int(sys.argv[idx + 1])
        run_backfill(days=days)
    elif cmd == "check":
        ticker_arg = sys.argv[2] if len(sys.argv) > 2 else None
        run_check(ticker_arg)
    elif cmd == "push":
        run_push()
    elif cmd == "schema":
        print_schema()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)