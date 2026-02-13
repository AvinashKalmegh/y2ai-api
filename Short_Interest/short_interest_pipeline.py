"""
Short Interest Pipeline
=======================
Fetches bi-monthly short interest data from Polygon/Massive API (FINRA-sourced).
Stores in Supabase short_interest table and pushes to Google Sheets.

Short interest is reported twice per month by FINRA (~24 data points/year per ticker).
Data includes: shares short, avg daily volume, days to cover.
Pipeline also fetches shares_outstanding to compute short_pct_outstanding.

Usage:
  python short_interest_pipeline.py test AAPL       # Test single ticker (run this FIRST!)
  python short_interest_pipeline.py backfill        # Full backfill 2020-present, all tickers
  python short_interest_pipeline.py daily           # Fetch latest settlement dates only
  python short_interest_pipeline.py push            # Push Supabase data to Sheets

Prerequisites:
  - Polygon/Massive Stocks plan with short interest access
    (Stocks Starter $29/mo should work; if not, Stocks Developer $79/mo)
  - POLYGON_API_KEY in .env (or MASSIVE_API_KEY — same key works for both)
  - Run short_interest_schema.sql in Supabase first

API Reference:
  Endpoint: GET /stocks/v1/short-interest?ticker=AAPL
  Docs: https://massive.com/docs/rest/stocks/fundamentals/short-interest
  Max limit: 50,000 per request
  Note: api.polygon.io and api.massive.com both work (same service, rebrand Oct 2025)
  Ticker is a QUERY PARAMETER, not a path segment.
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
from supabase import create_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY') or os.getenv('MASSIVE_API_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Both domains work; api.polygon.io is supported for backward compat
POLYGON_BASE = "https://api.polygon.io"

# Google Sheets config
SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "Short_Interest"

SHEETS_HEADERS = [
    "Settlement_Date", "Ticker", "Short_Interest", "Avg_Daily_Volume",
    "Days_To_Cover", "Shares_Outstanding", "Short_Pct_Outstanding"
]


# ============================================================
# TICKER UNIVERSE (from Supabase)
# ============================================================
def get_ticker_universe():
    """Load all active tickers from scanner_universe."""
    all_tickers = []
    offset = 0
    page_size = 1000

    while True:
        result = supabase.table("scanner_universe") \
            .select("ticker") \
            .order("ticker") \
            .range(offset, offset + page_size - 1) \
            .execute()

        if not result.data:
            break
        all_tickers.extend([r['ticker'] for r in result.data])
        if len(result.data) < page_size:
            break
        offset += page_size

    logger.info(f"Loaded {len(all_tickers)} active tickers from scanner_universe")
    return all_tickers


# ============================================================
# POLYGON API: SHORT INTEREST
# ============================================================
def fetch_short_interest(ticker, start_date="2020-01-01", end_date=None):
    """
    Fetch short interest data for a single ticker from Polygon/Massive.
    Uses /stocks/v1/short-interest endpoint with ticker as query param.
    Returns list of dicts: [{settlement_date, short_interest, avg_daily_volume, days_to_cover}]
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = f"{POLYGON_BASE}/stocks/v1/short-interest"
    all_results = []
    next_url = None

    params = {
        "ticker": ticker,
        "settlement_date.gte": start_date,
        "settlement_date.lte": end_date,
        "sort": "settlement_date",
        "order": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    while True:
        if next_url:
            # Pagination: next_url already has base params, just add API key
            sep = "&" if "?" in next_url else "?"
            fetch_url = f"{next_url}{sep}apiKey={POLYGON_API_KEY}"
            resp = requests.get(fetch_url)
        else:
            resp = requests.get(url, params=params)

        if resp.status_code == 403:
            logger.warning(f"  {ticker}: 403 Forbidden — short interest not available on your plan")
            return []
        elif resp.status_code == 404:
            logger.warning(f"  {ticker}: 404 Not Found — no short interest data")
            return []
        elif resp.status_code == 429:
            logger.warning(f"  {ticker}: Rate limited, waiting 60s...")
            time.sleep(60)
            continue
        elif resp.status_code != 200:
            logger.error(f"  {ticker}: HTTP {resp.status_code} — {resp.text[:300]}")
            return []

        data = resp.json()

        # Check for plan access errors in response body
        if data.get("status") == "ERROR":
            error_msg = data.get("error", data.get("message", "Unknown error"))
            logger.warning(f"  {ticker}: API error — {error_msg}")
            return []

        results = data.get("results", [])
        all_results.extend(results)

        # Check for pagination
        next_url = data.get("next_url")
        if not next_url:
            break

    return all_results


def fetch_shares_outstanding(ticker):
    """
    Fetch current shares outstanding from Polygon Ticker Details v3.
    Returns int or None.
    """
    url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
    params = {"apiKey": POLYGON_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        results = data.get("results", {})

        # Try weighted first (total across all share classes), then share_class
        shares = results.get("weighted_shares_outstanding")
        if not shares:
            shares = results.get("share_class_shares_outstanding")

        return shares
    except Exception as e:
        logger.warning(f"  {ticker}: Error fetching shares outstanding: {e}")
        return None


# ============================================================
# PROCESS AND STORE
# ============================================================
def process_ticker(ticker, start_date="2020-01-01", shares_cache=None):
    """
    Fetch and process short interest for a single ticker.
    Returns list of records ready for Supabase.
    """
    si_data = fetch_short_interest(ticker, start_date)

    if not si_data:
        return []

    # Get shares outstanding (use cache to avoid repeat calls)
    shares_out = None
    if shares_cache is not None and ticker in shares_cache:
        shares_out = shares_cache[ticker]
    else:
        shares_out = fetch_shares_outstanding(ticker)
        if shares_cache is not None:
            shares_cache[ticker] = shares_out
        time.sleep(0.15)

    records = []
    for entry in si_data:
        short_int = entry.get("short_interest")
        avg_vol = entry.get("avg_daily_volume")
        dtc = entry.get("days_to_cover")

        # Calculate short % of outstanding
        short_pct = None
        if short_int and shares_out and shares_out > 0:
            short_pct = round((short_int / shares_out) * 100, 4)

        records.append({
            "settlement_date": entry.get("settlement_date"),
            "ticker": ticker,
            "short_interest": short_int,
            "avg_daily_volume": avg_vol,
            "days_to_cover": float(dtc) if dtc is not None else None,
            "shares_outstanding": shares_out,
            "short_pct_outstanding": short_pct,
        })

    return records


def upload_to_supabase(records):
    """Upload short_interest records to Supabase with upsert."""
    if not records:
        return 0

    batch_size = 500
    uploaded = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("short_interest").upsert(
                batch, on_conflict="settlement_date,ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


def load_from_supabase(ticker=None, limit=None):
    """Load short_interest data from Supabase."""
    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        query = supabase.table("short_interest") \
            .select("*") \
            .order("settlement_date", desc=True)

        if ticker:
            query = query.eq("ticker", ticker)

        if limit:
            query = query.limit(limit)
            result = query.execute()
            return result.data

        result = query.range(offset, offset + page_size - 1).execute()

        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    return all_rows


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets():
    """Push short_interest data to Google Sheets."""
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading short_interest from Supabase...")
    all_rows = load_from_supabase()

    if not all_rows:
        logger.info("No data to push.")
        return

    logger.info(f"  {len(all_rows)} rows loaded")

    # Sort by date desc, then ticker
    all_rows.sort(key=lambda r: (r['settlement_date'], r['ticker']), reverse=True)

    # Build sheet rows
    sheet_rows = []
    for row in all_rows:
        sheet_rows.append([
            "'" + row.get('settlement_date', ''),
            row.get('ticker', ''),
            row.get('short_interest', ''),
            row.get('avg_daily_volume', ''),
            row.get('days_to_cover', ''),
            row.get('shares_outstanding', ''),
            row.get('short_pct_outstanding', ''),
        ])

    # Replace None with empty string
    for row in sheet_rows:
        for j in range(len(row)):
            if row[j] is None:
                row[j] = ''

    # Connect to Sheets
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(SHEETS_SPREADSHEET)
    except gspread.SpreadsheetNotFound:
        logger.error(f"Spreadsheet '{SHEETS_SPREADSHEET}' not found.")
        return

    # Delete and recreate tab
    try:
        old_sheet = spreadsheet.worksheet(SHEETS_TAB)
        spreadsheet.del_worksheet(old_sheet)
        logger.info(f"Deleted existing {SHEETS_TAB} tab")
    except gspread.WorksheetNotFound:
        pass

    sheet = spreadsheet.add_worksheet(
        title=SHEETS_TAB,
        rows=len(sheet_rows) + 100,
        cols=len(SHEETS_HEADERS)
    )

    # Write header
    sheet.update(range_name='A1', values=[SHEETS_HEADERS], value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})

    # Write data in batches
    BATCH_SIZE = 5000
    total_written = 0
    for i in range(0, len(sheet_rows), BATCH_SIZE):
        batch = sheet_rows[i:i + BATCH_SIZE]
        start_row = i + 2
        sheet.update(range_name=f'A{start_row}', values=batch, value_input_option='USER_ENTERED')
        total_written += len(batch)
        logger.info(f"  Written {total_written}/{len(sheet_rows)} rows")
        time.sleep(1)

    logger.info(f"Sheets push complete: {total_written} rows to {SHEETS_TAB}")


# ============================================================
# COMMANDS
# ============================================================
def run_backfill():
    """Full backfill from 2020 to present for all tickers."""
    logger.info("=" * 60)
    logger.info("SHORT INTEREST — FULL BACKFILL")
    logger.info("=" * 60)

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY (or MASSIVE_API_KEY) not set in .env")
        return

    tickers = get_ticker_universe()
    start_date = "2020-01-01"

    total_records = 0
    failed_tickers = []
    consecutive_403 = 0  # Track plan access failures
    shares_cache = {}

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] {ticker}...")

        # If first 5 tickers all return 403, stop early — plan issue
        if consecutive_403 >= 5:
            logger.error("\n" + "=" * 60)
            logger.error("STOPPING: First 5 tickers all returned 403 (plan access denied)")
            logger.error("Your Polygon/Massive plan likely does not include Short Interest.")
            logger.error("Options:")
            logger.error("  1. Upgrade to Stocks Starter ($29/mo) or higher")
            logger.error("  2. Check: https://massive.com/pricing")
            logger.error("  3. Contact support@massive.com to confirm plan access")
            logger.error("  4. Fallback: use short_volume_pipeline.py (free FINRA data)")
            logger.error("=" * 60)
            return

        try:
            records = process_ticker(ticker, start_date, shares_cache)

            if records:
                uploaded = upload_to_supabase(records)
                total_records += uploaded
                logger.info(f"  {ticker}: {len(records)} settlement dates, uploaded {uploaded}")
                consecutive_403 = 0  # Reset — got data
            else:
                logger.info(f"  {ticker}: no data")
                failed_tickers.append(ticker)

        except Exception as e:
            logger.error(f"  {ticker}: ERROR — {e}")
            failed_tickers.append(ticker)

        # Rate limit: be polite even on unlimited plans
        time.sleep(0.3)

        # Checkpoint every 50 tickers
        if (i + 1) % 50 == 0:
            logger.info(f"\n--- CHECKPOINT: {i+1}/{len(tickers)} tickers, {total_records} total rows ---\n")

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Total rows uploaded: {total_records}")
    logger.info(f"  Failed/empty tickers: {len(failed_tickers)}")
    if failed_tickers:
        logger.info(f"  Failed list: {', '.join(failed_tickers[:30])}")
        if len(failed_tickers) > 30:
            logger.info(f"  ... and {len(failed_tickers) - 30} more")
    logger.info("=" * 60)
    logger.info("Next: python short_interest_pipeline.py push  (to sync Google Sheets)")


def run_daily():
    """Fetch latest settlement dates for all tickers."""
    logger.info("=" * 60)
    logger.info("SHORT INTEREST — DAILY UPDATE")
    logger.info("=" * 60)

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY (or MASSIVE_API_KEY) not set in .env")
        return

    tickers = get_ticker_universe()

    # Fetch last 45 days to catch the most recent 1-2 settlement dates
    start_date = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")

    total_records = 0
    shares_cache = {}

    for i, ticker in enumerate(tickers):
        try:
            records = process_ticker(ticker, start_date, shares_cache)
            if records:
                uploaded = upload_to_supabase(records)
                total_records += uploaded
        except Exception as e:
            logger.error(f"  {ticker}: ERROR — {e}")

        time.sleep(0.2)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(tickers)} tickers, {total_records} rows")

    logger.info(f"\nDAILY UPDATE COMPLETE: {total_records} rows updated")
    logger.info("Next: python short_interest_pipeline.py push  (to sync Google Sheets)")


def run_test(ticker):
    """
    Test with a single ticker to verify API access.
    This is the FIRST command your developer should run.
    """
    logger.info("=" * 60)
    logger.info(f"SHORT INTEREST — API ACCESS TEST: {ticker}")
    logger.info("=" * 60)

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY (or MASSIVE_API_KEY) not set in .env")
        logger.error("Add one of these to your .env file.")
        return

    logger.info(f"Using API key: {POLYGON_API_KEY[:8]}...{POLYGON_API_KEY[-4:]}")
    logger.info(f"Base URL: {POLYGON_BASE}")

    # Quick curl example for manual testing
    logger.info(f"\nManual test (paste in browser or curl):")
    logger.info(f"  {POLYGON_BASE}/stocks/v1/short-interest?ticker={ticker}&settlement_date.gte=2024-01-01&limit=5&apiKey=YOUR_KEY")

    # Step 1: Test short interest endpoint
    logger.info(f"\n[STEP 1] Fetching short interest for {ticker} (2024-present)...")

    url = f"{POLYGON_BASE}/stocks/v1/short-interest"
    params = {
        "ticker": ticker,
        "settlement_date.gte": "2024-01-01",
        "sort": "settlement_date",
        "order": "asc",
        "limit": 10,
        "apiKey": POLYGON_API_KEY,
    }

    resp = requests.get(url, params=params)
    logger.info(f"  HTTP Status: {resp.status_code}")

    if resp.status_code == 403:
        logger.error(f"\n  === ACCESS DENIED (403) ===")
        logger.error(f"  The Short Interest endpoint is NOT available on your current plan.")
        logger.error(f"  Response: {resp.text[:300]}")
        logger.error(f"\n  WHAT TO DO:")
        logger.error(f"  1. Log in at https://massive.com/dashboard")
        logger.error(f"  2. Check your plan tier under Subscriptions")
        logger.error(f"  3. Short Interest may require Stocks Starter ($29/mo) or higher")
        logger.error(f"  4. Or contact support@massive.com to confirm")
        logger.error(f"\n  FALLBACK: Use short_volume_pipeline.py (free FINRA data, already built)")
        return

    if resp.status_code != 200:
        logger.error(f"  Unexpected status: {resp.status_code}")
        logger.error(f"  Response: {resp.text[:300]}")
        return

    data = resp.json()

    if data.get("status") == "ERROR":
        logger.error(f"  API returned error: {data.get('error', data.get('message', 'Unknown'))}")
        return

    results = data.get("results", [])
    logger.info(f"  SUCCESS! Got {len(results)} settlement dates")

    if not results:
        logger.warning(f"  No data for {ticker}. Trying AAPL as fallback...")
        if ticker != "AAPL":
            run_test("AAPL")
        return

    # Display results
    for entry in results:
        si = entry.get('short_interest', 0)
        avg_vol = entry.get('avg_daily_volume', 0)
        dtc = entry.get('days_to_cover', 'N/A')
        logger.info(f"  {entry['settlement_date']}: "
                     f"SI={si:>12,}  AvgVol={avg_vol:>12,}  DTC={dtc}")

    # Step 2: Test shares outstanding
    logger.info(f"\n[STEP 2] Fetching shares outstanding for {ticker}...")
    shares = fetch_shares_outstanding(ticker)
    if shares:
        logger.info(f"  Shares outstanding: {shares:,}")

        # Calculate short % for latest
        latest = results[-1]
        si_count = latest.get('short_interest', 0)
        if si_count and shares:
            pct = (si_count / shares) * 100
            logger.info(f"  Short % of outstanding: {pct:.2f}%")
            logger.info(f"  (This is what FlowOS/ARGUS will use for squeeze/sentiment signals)")
    else:
        logger.warning(f"  Could not fetch shares outstanding — short_pct will be NULL")
        logger.warning(f"  This is OK; the raw short interest count is still useful")

    # Step 3: Test Supabase upload
    logger.info(f"\n[STEP 3] Testing Supabase upload...")
    try:
        records = process_ticker(ticker, "2024-01-01", {})
        if records:
            uploaded = upload_to_supabase(records)
            logger.info(f"  Uploaded {uploaded} rows to Supabase short_interest table")
        else:
            logger.warning(f"  No records to upload")
    except Exception as e:
        logger.error(f"  Supabase upload failed: {e}")
        logger.error(f"  Did you run short_interest_schema.sql in the Supabase SQL Editor?")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TEST COMPLETE for {ticker}")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nAPI access: CONFIRMED")
    logger.info(f"Data points per ticker: ~{len(results) * 2} per year (bi-monthly)")
    logger.info(f"Estimated rows for full backfill (537 tickers x 5 years): ~{537 * 24 * 5:,}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. python short_interest_pipeline.py backfill    # Full 2020-present backfill")
    logger.info(f"  2. python short_interest_pipeline.py push        # Sync to Google Sheets")
    logger.info(f"  3. python short_interest_pipeline.py daily       # Add to cron for ongoing updates")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
Short Interest Pipeline — Polygon/Massive API
==============================================
Usage:
  python short_interest_pipeline.py test AAPL    # Test API access (RUN THIS FIRST!)
  python short_interest_pipeline.py backfill     # Full backfill 2020-present, all tickers
  python short_interest_pipeline.py daily        # Update latest settlement dates
  python short_interest_pipeline.py push         # Push to Google Sheets

Prerequisites:
  1. POLYGON_API_KEY in .env
  2. Run short_interest_schema.sql in Supabase SQL Editor
  3. Run 'test' command first to verify API access
""")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test":
        ticker = sys.argv[2].upper() if len(sys.argv) > 2 else "AAPL"
        run_test(ticker)
    elif cmd == "backfill":
        run_backfill()
    elif cmd == "daily":
        run_daily()
    elif cmd == "push":
        push_to_sheets()
    else:
        print(f"Unknown command: {cmd}")
        print("Valid commands: test, backfill, daily, push")
        sys.exit(1)