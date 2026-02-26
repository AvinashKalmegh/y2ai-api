"""
Options Put/Call Ratio Pipeline (Polygon Starter)
====================================================
Per-ticker put/call ratios for 55 priority names using Polygon Options Starter ($29/m).

WHAT IT DOES:
  Uses the Options Chain Snapshot endpoint to get all contracts for each underlying,
  then sums put volume vs call volume. This is the per-ticker leading indicator:
  a rising P/C ratio while DM is still high = institutions hedging 1-3 weeks early.

ENDPOINT:
  /v3/snapshot/options/{underlyingAsset}
  Returns all active contracts with day.volume and details.contract_type (put/call).
  Paginated (250 results per page). Starter tier: unlimited API calls.

SIGNALS:
  PC_Z_Score > 2.0  = Unusual put buying (institutions hedging, early warning)
  PC_Z_Score < -2.0 = Unusual call buying (accumulation or speculation)
  High P/C on high-DM names = distribution before price confirms (1-3 week lead)

OUTPUT COLUMNS:
  Date, Ticker, Put_Volume, Call_Volume, Put_Call_Ratio, PC_20D_Avg, PC_Z_Score

Usage:
  python options_pipeline.py test         # Test with 3 tickers
  python options_pipeline.py daily        # Fetch all 55 tickers, upload
  python options_pipeline.py backfill     # Same as daily (snapshot = current day only)
  python options_pipeline.py check NVDA   # Show history for specific ticker
  python options_pipeline.py push         # Push to Google Sheets

Rate limit: Starter tier = unlimited calls, but we add 0.5s delay for courtesy.
55 tickers with pagination = ~2-3 minutes.

BACKFILL NOTE:
  The snapshot endpoint returns current-day data only. There is no efficient way
  to reconstruct historical per-ticker P/C from Polygon (would require enumerating
  every individual option contract and pulling daily aggs for each). The practical
  approach: start collecting daily now, rolling metrics activate after 20 trading days.
  For ARGUS purposes, the forward-looking signal is what matters.

Requires: pip install requests python-dotenv supabase
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
except Exception:
    supabase = None
    logger.warning("Supabase not configured.")

# ============================================================
# CONFIGURATION
# ============================================================

# Delay between tickers (Starter = unlimited calls, but be polite)
DELAY_BETWEEN_TICKERS = 0.5  # seconds

# Google Sheets
SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "Options_PutCall"

# 55 priority tickers from spec: short portfolio + TOOLS + SERVICES layers
PRIORITY_TICKERS = [
    # Short portfolio (~15)
    "CRM", "ADBE", "NOW", "WDAY", "ACN", "OMC", "LPLA", "SCHW", "RJF",
    "UNH", "KTOS", "AVAV", "MRCY", "WDC",
    # TOOLS layer (~20)
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "TSM", "INTC",
    "SNOW", "PLTR", "DDOG", "NET", "CRWD", "ZS", "PANW", "FTNT", "ORCL", "SAP",
    # SERVICES layer (~15 incl global)
    "INFY", "ADP", "PAYX", "IT", "AON", "AJG", "MMC", "SPGI", "MCO",
    "WIT", "CTSH", "HDB", "IBM",
    # Infrastructure / Financial
    "GS", "MS", "JPM", "V", "MA", "BLK", "ICE",
    "EQIX", "DLR", "AMT", "CEG", "VST", "ANET", "DELL", "SMCI",
]
PRIORITY_TICKERS = list(dict.fromkeys(PRIORITY_TICKERS))  # dedupe


# ============================================================
# POLYGON OPTIONS SNAPSHOT
# ============================================================
def fetch_ticker_options(ticker):
    """
    Fetch all option contracts for a ticker via Polygon Options Chain Snapshot.
    Paginates through all results, sums put vs call volume.

    Endpoint: GET /v3/snapshot/options/{underlyingAsset}
    Returns: { put_volume, call_volume, put_open_interest, call_open_interest, contracts }
    """
    base_url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {
        "limit": 250,
        "apiKey": POLYGON_API_KEY,
    }

    put_volume = 0
    call_volume = 0
    put_oi = 0
    call_oi = 0
    total_contracts = 0
    page = 0

    while True:
        try:
            resp = requests.get(base_url, params=params, timeout=30)

            if resp.status_code == 403:
                logger.error(f"  {ticker}: 403 Forbidden - check Polygon plan (need Options Starter)")
                return None
            if resp.status_code == 429:
                logger.warning(f"  {ticker}: Rate limited, waiting 10s...")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                logger.warning(f"  {ticker}: HTTP {resp.status_code}")
                return None

            data = resp.json()

            if data.get("status") == "ERROR":
                msg = data.get("error", data.get("message", "Unknown"))
                logger.warning(f"  {ticker}: API error - {msg}")
                return None

            results = data.get("results", [])
            if not results:
                break

            for contract in results:
                day = contract.get("day", {})
                vol = day.get("volume", 0) or 0
                oi = contract.get("open_interest", 0) or 0

                details = contract.get("details", {})
                contract_type = details.get("contract_type", "").lower()

                if contract_type == "put":
                    put_volume += vol
                    put_oi += oi
                elif contract_type == "call":
                    call_volume += vol
                    call_oi += oi

                total_contracts += 1

            # Check for next page
            next_url = data.get("next_url")
            if next_url:
                # next_url is a full URL, just append API key
                base_url = next_url
                params = {"apiKey": POLYGON_API_KEY}
                page += 1
                time.sleep(0.2)  # Small delay between pages
            else:
                break

        except requests.exceptions.Timeout:
            logger.warning(f"  {ticker}: Timeout")
            return None
        except Exception as e:
            logger.warning(f"  {ticker}: Error - {e}")
            return None

    if total_contracts == 0:
        return None

    pc_ratio = round(put_volume / call_volume, 4) if call_volume > 0 else 0

    return {
        "put_volume": int(put_volume),
        "call_volume": int(call_volume),
        "put_open_interest": int(put_oi),
        "call_open_interest": int(call_oi),
        "put_call_ratio": pc_ratio,
        "contracts": total_contracts,
        "pages": page + 1,
    }


# ============================================================
# CALCULATIONS
# ============================================================
def calculate_pc_metrics(df):
    """Calculate PC_20D_Avg and PC_Z_Score per ticker."""
    result_dfs = []

    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker].sort_values('date').copy()

        # Put/Call Ratio (recalculate in case raw data was loaded)
        tdf['put_call_ratio'] = np.where(
            tdf['call_volume'] > 0,
            tdf['put_volume'] / tdf['call_volume'],
            0
        )
        tdf['put_call_ratio'] = tdf['put_call_ratio'].round(4)

        # 20-day rolling average
        tdf['pc_20d_avg'] = tdf['put_call_ratio'].rolling(
            window=20, min_periods=10
        ).mean().round(4)

        # Z-score
        pc_std = tdf['put_call_ratio'].rolling(window=20, min_periods=10).std()
        tdf['pc_z_score'] = np.where(
            pc_std > 0,
            ((tdf['put_call_ratio'] - tdf['pc_20d_avg']) / pc_std).round(4),
            0
        )

        result_dfs.append(tdf)

    if not result_dfs:
        return df
    return pd.concat(result_dfs, ignore_index=True)


# ============================================================
# SUPABASE
# ============================================================
def upload_to_supabase(df):
    """Upload to options_history table."""
    if supabase is None:
        logger.warning("Supabase not configured. Skipping upload.")
        return 0

    records = []
    for _, row in df.iterrows():
        record = {
            'date': str(row.get('date', '')),
            'ticker': str(row.get('ticker', '')),
            'put_volume': int(row['put_volume']) if pd.notna(row.get('put_volume')) else None,
            'call_volume': int(row['call_volume']) if pd.notna(row.get('call_volume')) else None,
            'put_call_ratio': float(row['put_call_ratio']) if pd.notna(row.get('put_call_ratio')) else None,
            'pc_20d_avg': float(row['pc_20d_avg']) if pd.notna(row.get('pc_20d_avg')) else None,
            'pc_z_score': float(row['pc_z_score']) if pd.notna(row.get('pc_z_score')) else None,
        }
        records.append(record)

    if not records:
        return 0

    BATCH_SIZE = 500
    uploaded = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("options_history") \
                .upsert(batch, on_conflict="date,ticker") \
                .execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


def load_from_supabase(ticker=None):
    """Load options_history from Supabase. Optionally filter by ticker."""
    if supabase is None:
        return pd.DataFrame()

    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        query = supabase.table("options_history").select("*").order("date", desc=True)
        if ticker:
            query = query.eq("ticker", ticker)
        query = query.range(offset, offset + page_size - 1)
        result = query.execute()

        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets():
    """Push options_history to Google Sheets."""
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading options_history from Supabase...")
    df = load_from_supabase()

    if df.empty:
        logger.info("No data to push.")
        return

    logger.info(f"  {len(df)} rows loaded")

    headers = [
        "Date", "Ticker", "Put_Volume", "Call_Volume",
        "Put_Call_Ratio", "PC_20D_Avg", "PC_Z_Score"
    ]

    sheet_rows = []
    for _, row in df.iterrows():
        sheet_rows.append([
            "'" + str(row.get('date', '')),
            row.get('ticker', ''),
            row.get('put_volume', ''),
            row.get('call_volume', ''),
            row.get('put_call_ratio', ''),
            row.get('pc_20d_avg', ''),
            row.get('pc_z_score', ''),
        ])

    for row in sheet_rows:
        for j in range(len(row)):
            if row[j] is None or (isinstance(row[j], float) and np.isnan(row[j])):
                row[j] = ''

    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
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

    sheet.update(range_name='A1', values=[headers], value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})

    BATCH_SIZE = 5000
    total = 0
    for i in range(0, len(sheet_rows), BATCH_SIZE):
        batch = sheet_rows[i:i + BATCH_SIZE]
        sheet.update(range_name=f'A{i+2}', values=batch, value_input_option='USER_ENTERED')
        total += len(batch)
        logger.info(f"  Written {total}/{len(sheet_rows)} rows")
        time.sleep(1)

    logger.info(f"Push complete: {total} rows")


# ============================================================
# COMMANDS
# ============================================================
def run_daily(tickers=None):
    """
    Fetch today's options data for priority tickers.
    Uses Polygon Options Chain Snapshot (Starter tier).
    """
    if tickers is None:
        tickers = PRIORITY_TICKERS

    logger.info("=" * 60)
    logger.info(f"OPTIONS P/C RATIO - DAILY ({len(tickers)} tickers)")
    logger.info("=" * 60)

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in .env")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    alerts = []

    for i, ticker in enumerate(tickers):
        logger.info(f"  [{i+1}/{len(tickers)}] {ticker}...")

        result = fetch_ticker_options(ticker)

        if result:
            rows.append({
                'date': today,
                'ticker': ticker,
                'put_volume': result['put_volume'],
                'call_volume': result['call_volume'],
                'put_call_ratio': result['put_call_ratio'],
            })
            logger.info(f"    Put={result['put_volume']:>10,}  Call={result['call_volume']:>10,}  "
                         f"P/C={result['put_call_ratio']:.4f}  "
                         f"({result['contracts']} contracts, {result['pages']} pages)")
        else:
            logger.warning(f"    No data")

        if i < len(tickers) - 1:
            time.sleep(DELAY_BETWEEN_TICKERS)

    if not rows:
        logger.error("No data fetched. Check API key and plan.")
        return

    df_today = pd.DataFrame(rows)

    # Load existing data for rolling calculations
    existing = load_from_supabase()
    if not existing.empty:
        keep_cols = ['date', 'ticker', 'put_volume', 'call_volume']
        keep_cols = [c for c in keep_cols if c in existing.columns]
        combined = pd.concat([existing[keep_cols], df_today], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
    else:
        combined = df_today

    # Calculate metrics across full history
    combined = calculate_pc_metrics(combined)

    # Extract today's rows with calculated metrics
    today_data = combined[combined['date'] == today]

    # Upload
    uploaded = upload_to_supabase(today_data)

    # Check for alerts (|Z| > 1.5)
    for _, row in today_data.iterrows():
        z = row.get('pc_z_score', 0)
        if pd.notna(z) and abs(z) > 1.5:
            direction = "PUT SPIKE" if z > 0 else "CALL SPIKE"
            alerts.append({
                'ticker': row['ticker'],
                'pc_ratio': row.get('put_call_ratio', 0),
                'z_score': z,
                'direction': direction,
            })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY COMPLETE - {today}")
    logger.info(f"  Tickers fetched: {len(rows)}/{len(tickers)}")
    logger.info(f"  Uploaded:        {uploaded} rows")

    if alerts:
        logger.info(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            logger.info(f"    {a['direction']}: {a['ticker']}  "
                         f"P/C={a['pc_ratio']:.4f}  Z={a['z_score']:+.2f}")
    else:
        days_collected = len(existing['date'].unique()) + 1 if not existing.empty else 1
        if days_collected < 20:
            logger.info(f"\n  No alerts (need {20 - days_collected} more days for Z-scores)")
        else:
            logger.info(f"\n  No alerts (all tickers within normal range)")


def run_test():
    """Test with 3 tickers to verify API access."""
    test_tickers = ["AAPL", "NVDA", "CRM"]

    logger.info("=" * 60)
    logger.info("OPTIONS P/C RATIO - TEST (3 tickers)")
    logger.info("=" * 60)

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in .env")
        logger.info("Add to your .env file: POLYGON_API_KEY=your_key_here")
        return

    logger.info(f"\nPolygon API key: {POLYGON_API_KEY[:8]}...{POLYGON_API_KEY[-4:]}")
    logger.info(f"Testing endpoint: /v3/snapshot/options/{{ticker}}")

    for ticker in test_tickers:
        logger.info(f"\n  {ticker}:")
        result = fetch_ticker_options(ticker)

        if result:
            logger.info(f"    Put volume:       {result['put_volume']:>12,}")
            logger.info(f"    Call volume:      {result['call_volume']:>12,}")
            logger.info(f"    P/C ratio:        {result['put_call_ratio']:.4f}")
            logger.info(f"    Put OI:           {result['put_open_interest']:>12,}")
            logger.info(f"    Call OI:          {result['call_open_interest']:>12,}")
            logger.info(f"    Total contracts:  {result['contracts']:>12,}")
            logger.info(f"    Pages fetched:    {result['pages']}")
        else:
            logger.error(f"    FAILED - check your Polygon plan")
            logger.info(f"    Options Starter ($29/m) required for snapshot endpoint")
            logger.info(f"    Activate at: https://polygon.io/dashboard")

        time.sleep(DELAY_BETWEEN_TICKERS)

    # Check Supabase
    logger.info(f"\n  Supabase:")
    if supabase:
        try:
            result = supabase.table("options_history") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if result.data:
                logger.info(f"    Table exists, latest: {result.data[0].get('date')}")
            else:
                logger.info(f"    Table exists but empty. Run 'daily' to start collecting.")
        except Exception as e:
            logger.info(f"    Table check: {e}")
            logger.info(f"    Run options_history schema SQL first.")
    else:
        logger.info(f"    Not configured.")

    logger.info("\nTEST COMPLETE")


def run_backfill():
    """
    Start collecting options P/C data.
    The snapshot endpoint only provides current-day data, so backfill = daily.
    After 20 trading days, rolling metrics (PC_20D_Avg, PC_Z_Score) activate.
    """
    logger.info("=" * 60)
    logger.info("OPTIONS P/C RATIO - BACKFILL")
    logger.info("=" * 60)
    logger.info("")
    logger.info("The Polygon snapshot endpoint provides current-day data only.")
    logger.info("Historical per-ticker P/C cannot be efficiently reconstructed.")
    logger.info("")
    logger.info("Collection plan:")
    logger.info("  Day 1-10:  Raw P/C ratios collected, no rolling metrics yet")
    logger.info("  Day 10-20: PC_20D_Avg starts (min_periods=10)")
    logger.info("  Day 20+:   Full Z-scores active, alerts fire on |Z| > 1.5")
    logger.info("")
    logger.info("Starting daily collection now...")
    logger.info("")
    run_daily()


def run_check(ticker=None):
    """Show history for a specific ticker or recent alerts."""
    logger.info("=" * 60)
    logger.info("OPTIONS P/C RATIO - CHECK")
    logger.info("=" * 60)

    if supabase is None:
        logger.error("Supabase not configured.")
        return

    if ticker:
        # Show history for specific ticker
        result = supabase.table("options_history") \
            .select("*") \
            .eq("ticker", ticker.upper()) \
            .order("date", desc=True) \
            .limit(20) \
            .execute()

        if not result.data:
            logger.info(f"No data for {ticker.upper()}. Run 'daily' first.")
            return

        logger.info(f"\n  {ticker.upper()} - Last 20 days:")
        logger.info(f"  {'Date':<12} {'Put Vol':>10} {'Call Vol':>10} {'P/C':>8} {'20D Avg':>8} {'Z-Score':>8}")
        logger.info(f"  {'-'*60}")
        for row in result.data:
            d = row.get('date', '?')
            pv = row.get('put_volume', 0)
            cv = row.get('call_volume', 0)
            pc = row.get('put_call_ratio', 0)
            avg = row.get('pc_20d_avg')
            z = row.get('pc_z_score')
            pv_s = f"{pv:>10,}" if pv else "?"
            cv_s = f"{cv:>10,}" if cv else "?"
            pc_s = f"{pc:.4f}" if pc else "?"
            avg_s = f"{avg:.4f}" if avg else "N/A"
            z_s = f"{z:+.2f}" if z else "N/A"
            flag = " <<<" if z and abs(z) > 1.5 else ""
            logger.info(f"  {d:<12} {pv_s} {cv_s} {pc_s:>8} {avg_s:>8} {z_s:>8}{flag}")
    else:
        # Show latest alerts across all tickers
        result = supabase.table("options_history") \
            .select("*") \
            .order("date", desc=True) \
            .limit(200) \
            .execute()

        if not result.data:
            logger.info("No data. Run 'daily' first.")
            return

        # Get latest date
        latest_date = result.data[0].get('date')
        logger.info(f"\n  Latest: {latest_date}")

        # Show today's data sorted by P/C ratio
        today_rows = [r for r in result.data if r.get('date') == latest_date]
        today_rows.sort(key=lambda r: r.get('put_call_ratio', 0) or 0, reverse=True)

        logger.info(f"  {'Ticker':<8} {'P/C':>8} {'20D Avg':>8} {'Z-Score':>8} {'Put Vol':>10} {'Call Vol':>10}")
        logger.info(f"  {'-'*60}")
        for row in today_rows:
            t = row.get('ticker', '?')
            pc = row.get('put_call_ratio', 0)
            avg = row.get('pc_20d_avg')
            z = row.get('pc_z_score')
            pv = row.get('put_volume', 0)
            cv = row.get('call_volume', 0)
            pc_s = f"{pc:.4f}" if pc else "?"
            avg_s = f"{avg:.4f}" if avg else "N/A"
            z_s = f"{z:+.2f}" if z else "N/A"
            flag = " <<<" if z and abs(z) > 1.5 else ""
            logger.info(f"  {t:<8} {pc_s:>8} {avg_s:>8} {z_s:>8} {pv:>10,} {cv:>10,}{flag}")

        # Count collection days
        dates = set(r.get('date') for r in result.data)
        logger.info(f"\n  Days collected: {len(dates)}")
        if len(dates) < 20:
            logger.info(f"  Need {20 - len(dates)} more days for full Z-scores")


def run_push():
    """Push to Google Sheets."""
    logger.info("=" * 60)
    logger.info("OPTIONS P/C RATIO - PUSH TO SHEETS")
    logger.info("=" * 60)
    push_to_sheets()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Options Put/Call Ratio Pipeline (Polygon Starter)")
        print("=" * 50)
        print()
        print(f"Tickers: {len(PRIORITY_TICKERS)} priority names")
        print(f"Source:  Polygon Options Chain Snapshot (Starter $29/m)")
        print(f"Signal:  |PC_Z_Score| > 1.5 = unusual options activity")
        print()
        print("Usage:")
        print("  python options_pipeline.py test           # Verify API with 3 tickers")
        print("  python options_pipeline.py daily          # Fetch all 55, upload to Supabase")
        print("  python options_pipeline.py backfill       # Start collection (= daily)")
        print("  python options_pipeline.py check          # Show latest, sorted by P/C")
        print("  python options_pipeline.py check NVDA     # Show NVDA history")
        print("  python options_pipeline.py push           # Push to Google Sheets")
        print()
        print(f"Tickers ({len(PRIORITY_TICKERS)}):")
        for i in range(0, len(PRIORITY_TICKERS), 10):
            print(f"  {', '.join(PRIORITY_TICKERS[i:i+10])}")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test":
        run_test()
    elif cmd == "daily":
        run_daily()
    elif cmd == "backfill":
        run_backfill()
    elif cmd == "check":
        ticker_arg = sys.argv[2] if len(sys.argv) > 2 else None
        run_check(ticker_arg)
    elif cmd == "push":
        run_push()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)