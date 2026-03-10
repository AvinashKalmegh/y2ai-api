"""
INTRADAY DM SCANNER
====================
Computes intraday Dark Matter (Capital Flow) scores using Polygon.io
live snapshot data. Supplements the EOD pipeline with mid-day reads.

Scheduled runs: 10:30 AM, 1:00 PM, 3:30 PM ET on trading days.

DM = (RelStr vs ETF * 0.50) + (RelStr vs SPY * 0.30) + (VolumeZ * 0.20)

Uses the SAME formula as Dm_historical_pipeline.py — only the price
and volume inputs change from EOD closes to live values.

Usage:
    python intraday_dm_scanner.py                  # Run scan (full priority universe)
    python intraday_dm_scanner.py --marketplace     # 18 Marketplace Watchlist only
    python intraday_dm_scanner.py --flowos          # 26 FlowOS universe only
    python intraday_dm_scanner.py --tickers NVDA,TSLA,UPWK  # Custom tickers
    python intraday_dm_scanner.py --dry-run         # Print results, don't write to DB

Environment Variables (.env):
    POLYGON_API_KEY   - Polygon.io API key (paid tier)
    SUPABASE_URL      - Supabase project URL
    SUPABASE_KEY      - Supabase service role key
"""

import os
import sys
import time
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# DM formula weights (FlowOS) — same as EOD pipeline
W_REL_STR_ETF = 0.50
W_REL_STR_SPY = 0.30
W_VOLUME_Z = 0.20

# Relative strength scaling
REL_STR_SCALE = 500

# EMA smoothing
EMA_PERIOD = 5
EMA_K = 2 / (EMA_PERIOD + 1)  # 0.333

# Return period (20-day = 19 intervals)
RETURN_PERIOD = 19

# Market hours (minutes in full trading day: 9:30 AM - 4:00 PM = 390 min)
MARKET_MINUTES_TOTAL = 390

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# SECTOR ETF MAPPING (same as EOD pipeline)
# ============================================================

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Semiconductors": "SMH",
    "Software": "IGV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Biotechnology": "XBI",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Energy": "XLE",
    "Communication Services": "XLC",
    "Nuclear": "URA",
    "Uranium": "URA",
    "Nuclear/Uranium": "URA",
    "Clean Energy": "TAN",
    "Cybersecurity": "HACK",
    "Aerospace & Defense": "ITA",
    "Defense": "ITA",
    "Defense/Aerospace": "ITA",
    "Transportation": "IYT",
}

ALL_ETFS = list(set(SECTOR_ETF_MAP.values())) + ["SPY"]
DEFAULT_ETF = "SPY"


# ============================================================
# TICKER UNIVERSES
# ============================================================

# 21 Marketplace Watchlist tickers
MARKETPLACE_TICKERS = [
    "UPWK", "GTLB", "ACN", "HUBS", "FVRR", "TWLO",
    "ZS", "CRWD", "NET", "DDOG", "MDB", "SNOW",
    "PLTR", "NOW", "CRM", "ADBE", "SHOP", "WDAY",
    "RBLX", "BILL", "DOCN",
]

# 26 FlowOS production universe
FLOWOS_TICKERS = [
    "NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "QCOM",
    "MSFT", "GOOGL", "AMZN", "META", "ORCL",
    "CEG", "VRT", "NRG",
    "TSLA", "APP", "UBER",
    "GS", "MS",
    "CRWD", "PANW",
    "FSLR",
    "DLR", "EQIX",
    "SMCI",
]

# Combined priority universe (44 unique tickers)
PRIORITY_TICKERS = sorted(set(MARKETPLACE_TICKERS + FLOWOS_TICKERS))


# ============================================================
# SUPABASE CLIENT
# ============================================================

_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ============================================================
# POLYGON SNAPSHOT API
# ============================================================

def fetch_snapshot_all():
    """
    Fetch snapshot for ALL US tickers in one API call.
    Returns dict: {ticker: {price, volume, updated}}
    """
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    params = {"apiKey": POLYGON_API_KEY}

    logger.info("Fetching Polygon snapshot (all US tickers)...")
    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if data.get("status") != "OK":
        raise Exception(f"Polygon snapshot failed: {data.get('status')} - {data.get('error', 'unknown')}")

    snapshot = {}
    for t in data.get("tickers", []):
        ticker = t.get("ticker", "")
        day_data = t.get("day", {})
        last_trade = t.get("lastTrade", {}) or t.get("lastQuote", {})

        price = day_data.get("c") or last_trade.get("p") or day_data.get("vw")
        volume = day_data.get("v", 0)

        if price and price > 0:
            snapshot[ticker] = {
                "price": float(price),
                "volume": int(volume) if volume else 0,
                "vwap": float(day_data.get("vw", 0)),
            }

    logger.info(f"  Snapshot received: {len(snapshot)} tickers")
    return snapshot


def build_snapshot_from_history(hist_df):
    """
    Fallback: build a snapshot from the latest historical closes.
    Used when markets are closed (weekends, after hours) for testing.
    """
    logger.info("  Building snapshot from latest historical closes (market closed fallback)")
    latest = hist_df.sort_values('date').groupby('ticker').tail(1)
    snapshot = {}
    for _, row in latest.iterrows():
        snapshot[row['ticker']] = {
            "price": float(row['close']),
            "volume": int(row['volume']) if pd.notna(row['volume']) else 0,
            "vwap": 0.0,
        }
    logger.info(f"  Fallback snapshot: {len(snapshot)} tickers")
    return snapshot


# ============================================================
# HISTORICAL DATA LOADERS
# ============================================================

def load_historical_prices(tickers, n_days=25):
    """
    Load recent historical closes from Supabase price_history.
    Returns DataFrame with columns: date, ticker, close, volume
    """
    supabase = get_supabase()
    cutoff = (datetime.now() - timedelta(days=n_days * 2)).strftime("%Y-%m-%d")

    all_rows = []
    # Batch tickers to avoid query size limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            result = supabase.table("price_history") \
                .select("date,ticker,close,volume") \
                .in_("ticker", batch) \
                .gte("date", cutoff) \
                .order("date") \
                .execute()
            if result.data:
                all_rows.extend(result.data)
        except Exception as e:
            logger.warning(f"  Error loading prices for batch {i}: {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['close'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    return df


def load_prior_dm(tickers, n_days=4):
    """
    Load last 4 days of EOD DM from dm_history for EMA5 anchoring.
    Returns dict: {ticker: [dm_day_minus_4, ..., dm_day_minus_1]}
    """
    supabase = get_supabase()
    cutoff = (datetime.now() - timedelta(days=n_days * 3)).strftime("%Y-%m-%d")

    dm_history = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            result = supabase.table("dm_history") \
                .select("ticker,date,dm_smoothed") \
                .in_("ticker", batch) \
                .gte("date", cutoff) \
                .order("date", desc=True) \
                .execute()
            if result.data:
                for row in result.data:
                    t = row['ticker']
                    if t not in dm_history:
                        dm_history[t] = []
                    if len(dm_history[t]) < n_days:
                        dm_history[t].append(float(row['dm_smoothed']))
        except Exception as e:
            logger.warning(f"  Error loading DM history for batch {i}: {e}")

    # Reverse so oldest first
    for t in dm_history:
        dm_history[t] = list(reversed(dm_history[t]))

    return dm_history


def load_ticker_sectors(tickers):
    """
    Load sector mappings from scanner_universe table.
    Returns dict: {ticker: sector}
    """
    supabase = get_supabase()
    sector_map = {}

    try:
        result = supabase.table("scanner_universe") \
            .select("ticker,sector") \
            .in_("ticker", tickers) \
            .execute()
        if result.data:
            for row in result.data:
                sector_map[row['ticker']] = row.get('sector', 'Technology')
    except Exception as e:
        logger.warning(f"  Could not load sectors: {e}")

    return sector_map


# ============================================================
# DM COMPUTATION (matches EOD formula exactly)
# ============================================================

def calculate_relative_strength(ticker_return, benchmark_return):
    """RelStr score 0-100. 50 = equal, >50 = outperformance."""
    diff = ticker_return - benchmark_return
    score = 50 + (diff * REL_STR_SCALE)
    return max(0.0, min(100.0, score))


def calculate_volume_z_intraday(volume_today, vol_20d_avg, vol_20d_std=None,
                                 minutes_elapsed=None):
    """
    Volume z-score for intraday.
    Adjusts today's volume for time-of-day using annualization.
    """
    if not vol_20d_avg or vol_20d_avg <= 0:
        return 50.0
    if not volume_today or volume_today <= 0:
        return 50.0

    # Annualize: scale partial-day volume to full-day estimate
    if minutes_elapsed and minutes_elapsed > 0 and minutes_elapsed < MARKET_MINUTES_TOTAL:
        projected_volume = volume_today * (MARKET_MINUTES_TOTAL / minutes_elapsed)
    else:
        projected_volume = volume_today

    # Use the GS-matching formula: ratio-based scoring
    ratio = projected_volume / vol_20d_avg
    score = (ratio - 0.5) * 66.67
    return max(0.0, min(100.0, score))


def compute_ema5(prior_dm_values, current_dm):
    """
    Compute EMA5 anchored by prior 4 days of EOD DM.
    prior_dm_values: list of up to 4 prior EOD DM values (oldest first)
    current_dm: today's intraday DM raw
    """
    values = list(prior_dm_values) + [current_dm]

    if len(values) == 1:
        return current_dm

    # Apply EMA from start
    ema = values[0]
    for v in values[1:]:
        ema = v * EMA_K + ema * (1 - EMA_K)

    return max(0.0, min(100.0, ema))


def get_minutes_since_open():
    """Calculate minutes since market open (9:30 AM ET)."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < market_open:
        return 0
    elapsed = (now - market_open).total_seconds() / 60
    return min(elapsed, MARKET_MINUTES_TOTAL)


# ============================================================
# MAIN SCAN
# ============================================================

def run_intraday_scan(tickers, dry_run=False):
    """
    Execute one intraday DM scan for the given tickers.

    1. Fetch Polygon snapshot (one API call)
    2. Load historical prices for 20-day returns
    3. Load prior 4 days of DM for EMA5
    4. Compute DM for each ticker
    5. Write to dm_intraday table
    """
    scan_time = datetime.now(timezone.utc)
    minutes_elapsed = get_minutes_since_open()

    logger.info("=" * 60)
    logger.info("INTRADAY DM SCAN")
    logger.info(f"  Time: {scan_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"  Minutes since open: {minutes_elapsed:.0f}")
    logger.info(f"  Tickers: {len(tickers)}")
    logger.info("=" * 60)

    # 1. Snapshot
    snapshot = fetch_snapshot_all()

    # Determine all needed tickers (our universe + their ETFs + SPY)
    sector_map = load_ticker_sectors(tickers)
    ticker_etf_map = {}
    needed_etfs = set(["SPY"])
    for t in tickers:
        sector = sector_map.get(t, "Technology")
        etf = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)
        ticker_etf_map[t] = etf
        needed_etfs.add(etf)

    all_needed = list(set(tickers) | needed_etfs)

    # 2. Historical prices
    logger.info("Loading historical prices...")
    hist_df = load_historical_prices(all_needed, n_days=30)
    if hist_df.empty:
        logger.error("No historical price data. Cannot compute returns.")
        return []

    # If snapshot is empty (weekend/after hours), use historical closes as fallback
    if not snapshot:
        logger.warning("Snapshot empty (market likely closed). Using historical close fallback.")
        snapshot = build_snapshot_from_history(hist_df)
        minutes_elapsed = MARKET_MINUTES_TOTAL  # treat as full day for volume calc

    # 3. Prior DM for EMA5
    logger.info("Loading prior DM history for EMA5...")
    prior_dm = load_prior_dm(tickers, n_days=4)

    # 4. Compute
    logger.info("Computing intraday DM scores...")
    results = []
    errors = []

    for ticker in tickers:
        try:
            # Get live data
            if ticker not in snapshot:
                logger.warning(f"  {ticker}: not in snapshot, skipping")
                continue

            live = snapshot[ticker]
            current_price = live["price"]
            volume_today = live["volume"]

            # Get ETF and SPY live data
            etf_symbol = ticker_etf_map.get(ticker, DEFAULT_ETF)
            spy_live = snapshot.get("SPY", {})
            etf_live = snapshot.get(etf_symbol, {})

            if not spy_live:
                logger.warning(f"  {ticker}: SPY not in snapshot, skipping")
                continue

            # Historical prices for this ticker
            t_hist = hist_df[hist_df['ticker'] == ticker].sort_values('date')
            spy_hist = hist_df[hist_df['ticker'] == 'SPY'].sort_values('date')
            etf_hist = hist_df[hist_df['ticker'] == etf_symbol].sort_values('date')

            if len(t_hist) < RETURN_PERIOD:
                logger.warning(f"  {ticker}: insufficient history ({len(t_hist)} days)")
                continue

            # 20-day-ago close (RETURN_PERIOD intervals back)
            price_20d_ago = t_hist.iloc[-(RETURN_PERIOD + 1)]['close'] if len(t_hist) > RETURN_PERIOD else t_hist.iloc[0]['close']
            ticker_return = (current_price - price_20d_ago) / price_20d_ago

            # SPY 20-day return
            if len(spy_hist) > RETURN_PERIOD:
                spy_price_20d_ago = spy_hist.iloc[-(RETURN_PERIOD + 1)]['close']
                spy_current = spy_live.get("price", spy_hist.iloc[-1]['close'])
                spy_return = (spy_current - spy_price_20d_ago) / spy_price_20d_ago
            else:
                spy_return = 0.0

            # ETF 20-day return
            if len(etf_hist) > RETURN_PERIOD:
                etf_price_20d_ago = etf_hist.iloc[-(RETURN_PERIOD + 1)]['close']
                etf_current = etf_live.get("price", etf_hist.iloc[-1]['close']) if etf_live else etf_hist.iloc[-1]['close']
                etf_return = (etf_current - etf_price_20d_ago) / etf_price_20d_ago
            else:
                etf_return = 0.0

            # Volume Z-score (using 20-day avg from history)
            vol_20d_avg = t_hist['volume'].tail(20).mean()
            vol_20d_std = t_hist['volume'].tail(20).std()
            volume_z = calculate_volume_z_intraday(
                volume_today, vol_20d_avg, vol_20d_std, minutes_elapsed
            )

            # Relative strength scores
            rel_str_etf = calculate_relative_strength(ticker_return, etf_return)
            rel_str_spy = calculate_relative_strength(ticker_return, spy_return)

            # DM Raw
            dm_raw = (rel_str_etf * W_REL_STR_ETF +
                      rel_str_spy * W_REL_STR_SPY +
                      volume_z * W_VOLUME_Z)
            dm_raw = max(0.0, min(100.0, dm_raw))

            # EMA5 (anchored by prior 4 EOD DM values)
            prior_values = prior_dm.get(ticker, [])
            dm_ema5 = compute_ema5(prior_values, dm_raw)

            row = {
                "scan_time": scan_time.isoformat(),
                "ticker": ticker,
                "dm_score": round(dm_raw, 1),
                "dm_ema5": round(dm_ema5, 1),
                "current_price": round(current_price, 2),
                "volume_today": volume_today,
                "volume_zscore": round(volume_z, 2),
                "return_20d": round(ticker_return, 4),
                "etf_return_20d": round(etf_return, 4),
                "spy_return_20d": round(spy_return, 4),
            }
            results.append(row)

        except Exception as e:
            logger.error(f"  {ticker}: ERROR - {e}")
            errors.append(ticker)

    logger.info(f"\nComputed DM for {len(results)}/{len(tickers)} tickers")
    if errors:
        logger.warning(f"Errors ({len(errors)}): {', '.join(errors)}")

    # Print results table
    if results:
        print_results_table(results)

    # 5. Write to Supabase + Google Sheets
    if not dry_run and results:
        write_results(results)
        push_results_to_sheets(results)
    elif dry_run:
        logger.info("DRY RUN — results NOT written to database")

    return results


def print_results_table(results):
    """Print a formatted results table to console."""
    sorted_results = sorted(results, key=lambda r: r['dm_ema5'], reverse=True)

    print("\n" + "=" * 95)
    print(f"{'Ticker':<8} {'Price':>9} {'DM Raw':>8} {'DM EMA5':>9} {'Vol Z':>7} "
          f"{'Ret 20d':>9} {'ETF Ret':>9} {'SPY Ret':>9} {'Volume':>12}")
    print("-" * 95)

    for r in sorted_results:
        print(f"{r['ticker']:<8} {r['current_price']:>9.2f} {r['dm_score']:>8.1f} "
              f"{r['dm_ema5']:>9.1f} {r['volume_zscore']:>7.2f} "
              f"{r['return_20d']:>9.4f} {r['etf_return_20d']:>9.4f} "
              f"{r['spy_return_20d']:>9.4f} {r['volume_today']:>12,}")

    print("=" * 95)

    # Highlight movers
    high_dm = [r for r in sorted_results if r['dm_ema5'] >= 70]
    low_dm = [r for r in sorted_results if r['dm_ema5'] <= 30]

    if high_dm:
        print(f"\n  STRONG FLOW (DM >= 70): {', '.join(r['ticker'] for r in high_dm)}")
    if low_dm:
        print(f"  WEAK FLOW (DM <= 30):   {', '.join(r['ticker'] for r in low_dm)}")
    print()


def write_results(results):
    """Write scan results to dm_intraday table in Supabase."""
    supabase = get_supabase()

    try:
        supabase.table("dm_intraday").upsert(
            results, on_conflict="scan_time,ticker"
        ).execute()
        logger.info(f"  Written {len(results)} rows to dm_intraday")
    except Exception as e:
        logger.error(f"  Failed to write to dm_intraday: {e}")
        logger.info("  Table may not exist. Create it in Supabase SQL Editor.")


def push_results_to_sheets(results):
    """Push intraday scan results to Google Sheets DM_Intraday tab."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        logger.error("  gspread/oauth2client not installed. Skipping GS push.")
        return

    GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
    SPREADSHEET_NAME = "copy-dm-history 2024-current"
    TAB_NAME = "DM_Intraday"

    HEADERS = [
        'Scan_Time', 'Ticker', 'DM_Raw', 'DM_EMA5', 'Price',
        'Volume', 'Vol_Z', 'Return_20d', 'ETF_Ret_20d', 'SPY_Ret_20d'
    ]

    if not os.path.exists(GOOGLE_SHEETS_CREDS_FILE):
        # GitHub Actions writes credentials.json from secrets
        creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            with open(GOOGLE_SHEETS_CREDS_FILE, 'w') as f:
                f.write(creds_json)
        else:
            logger.warning("  No Google credentials found. Skipping GS push.")
            return

    logger.info("Pushing intraday results to Google Sheets...")

    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SHEETS_CREDS_FILE, scope
    )
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(SPREADSHEET_NAME)
    except gspread.SpreadsheetNotFound:
        logger.error(f"  Spreadsheet '{SPREADSHEET_NAME}' not found.")
        return

    # Get or create tab
    try:
        sheet = spreadsheet.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=TAB_NAME, rows=2000, cols=len(HEADERS))
        logger.info(f"  Created new tab: {TAB_NAME}")

    # Sort by DM_EMA5 descending
    sorted_results = sorted(results, key=lambda r: r['dm_ema5'], reverse=True)

    # Build rows
    scan_time = sorted_results[0]['scan_time'] if sorted_results else ''
    # Format scan_time in ET for readability
    try:
        from datetime import datetime as dt
        from zoneinfo import ZoneInfo
        st = dt.fromisoformat(scan_time.replace('Z', '+00:00'))
        st_et = st.astimezone(ZoneInfo('America/New_York'))
        scan_label = st_et.strftime('%Y-%m-%d %H:%M ET')
    except Exception:
        scan_label = str(scan_time)

    sheet_rows = []
    for r in sorted_results:
        sheet_rows.append([
            scan_label,
            r['ticker'],
            r['dm_score'],
            r['dm_ema5'],
            r['current_price'],
            r['volume_today'],
            r['volume_zscore'],
            r['return_20d'],
            r['etf_return_20d'],
            r['spy_return_20d'],
        ])

    # Check if tab already has data — append new scan below existing
    existing_data = sheet.col_values(1)
    if len(existing_data) <= 1:
        # Empty tab — write header + data
        all_data = [HEADERS] + sheet_rows
        sheet.update(range_name='A1', values=all_data, value_input_option='USER_ENTERED')
        sheet.format('1:1', {'textFormat': {'bold': True}})
    else:
        # Append below existing data (add blank separator row)
        start_row = len(existing_data) + 2
        # Expand if needed
        rows_needed = start_row + len(sheet_rows)
        if rows_needed > sheet.row_count:
            sheet.resize(rows=rows_needed + 500)
        sheet.update(range_name=f'A{start_row}', values=sheet_rows, value_input_option='USER_ENTERED')

    logger.info(f"  Pushed {len(sheet_rows)} rows to GS {TAB_NAME} (scan: {scan_label})")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Intraday DM Scanner")
    parser.add_argument("--marketplace", action="store_true",
                        help="Scan 18 Marketplace Watchlist tickers only")
    parser.add_argument("--flowos", action="store_true",
                        help="Scan 26 FlowOS production tickers only")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated custom ticker list")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to database")
    args = parser.parse_args()

    # Validate config
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in environment")
        sys.exit(1)
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL/SUPABASE_KEY not set in environment")
        sys.exit(1)

    # Determine ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.marketplace:
        tickers = MARKETPLACE_TICKERS
    elif args.flowos:
        tickers = FLOWOS_TICKERS
    else:
        tickers = PRIORITY_TICKERS

    logger.info(f"Universe: {len(tickers)} tickers")
    run_intraday_scan(tickers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
