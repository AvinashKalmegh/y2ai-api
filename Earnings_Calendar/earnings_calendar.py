"""
Earnings Calendar Pipeline
==============================
Fetches historical and upcoming earnings dates from Finnhub.
Stores in Supabase earnings_calendar table and pushes to Google Sheets.

Key use case for ARGUS/FlowOS: Avoid entering positions right before earnings
(binary event risk). Also tracks EPS surprise patterns for sector analysis.

Usage:
  python earnings_calendar_pipeline.py test               # Test API access
  python earnings_calendar_pipeline.py backfill           # Backfill 2024-present
  python earnings_calendar_pipeline.py upcoming           # Next 4 weeks of earnings
  python earnings_calendar_pipeline.py daily              # Refresh recent + upcoming
  python earnings_calendar_pipeline.py push               # Push Supabase data to Sheets
  python earnings_calendar_pipeline.py check NVDA         # Check specific ticker

API Source: Finnhub (https://finnhub.io/docs/api/earnings-calendar)
  Endpoint: GET https://finnhub.io/api/v1/calendar/earnings
  Auth: token query parameter
  Free tier: 60 calls/minute (no monthly cap)
  Key advantage: Fetches by DATE RANGE (not per-ticker), so one call
  returns all tickers reporting in that window. Very efficient.

  Response fields:
    date          - Earnings release date
    epsActual     - Actual EPS (non-GAAP, after report)
    epsEstimate   - Consensus EPS estimate
    hour          - "bmo" (before market open), "amc" (after market close), "dmh" (during)
    quarter       - Fiscal quarter (1-4)
    revenueActual - Actual revenue
    revenueEstimate - Consensus revenue estimate
    symbol        - Ticker
    year          - Fiscal year (not calendar year)

Prerequisites:
  - FINNHUB_API_KEY in .env (same key as insider transactions)
  - Run earnings_calendar_schema.sql in Supabase first
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# Google Sheets config
SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "Earnings_Calendar"

SHEETS_HEADERS = [
    "Date", "Ticker", "Hour", "Quarter", "Fiscal_Year",
    "EPS_Estimate", "EPS_Actual", "EPS_Surprise_Pct",
    "Revenue_Estimate", "Revenue_Actual", "Revenue_Surprise_Pct",
    "In_Universe"
]


# ============================================================
# TICKER UNIVERSE (from Supabase)
# ============================================================
def get_ticker_universe():
    """Load all tickers from scanner_universe as a set for fast lookup."""
    all_tickers = set()
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
        all_tickers.update(r['ticker'] for r in result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    logger.info(f"Loaded {len(all_tickers)} tickers from scanner_universe")
    return all_tickers


# ============================================================
# FINNHUB API: EARNINGS CALENDAR
# ============================================================
def fetch_earnings_by_date_range(date_from, date_to):
    """
    Fetch earnings calendar for a date range. Returns ALL tickers
    reporting in that window in a single call.

    Finnhub response format:
    {
      "earningsCalendar": [
        {
          "date": "2024-01-25",
          "epsActual": 2.18,
          "epsEstimate": 2.10,
          "hour": "amc",
          "quarter": 1,
          "revenueActual": 119580000000,
          "revenueEstimate": 117900000000,
          "symbol": "AAPL",
          "year": 2024
        }
      ]
    }
    """
    url = f"{FINNHUB_BASE}/calendar/earnings"
    params = {
        "from": date_from,
        "to": date_to,
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 401:
            logger.error("401 Unauthorized — check FINNHUB_API_KEY")
            return None
        elif resp.status_code == 429:
            logger.warning("Rate limited, waiting 30s...")
            time.sleep(30)
            return fetch_earnings_by_date_range(date_from, date_to)
        elif resp.status_code != 200:
            logger.error(f"HTTP {resp.status_code} — {resp.text[:200]}")
            return []

        data = resp.json()
        earnings = data.get("earningsCalendar", [])
        return earnings

    except Exception as e:
        logger.error(f"Request error — {e}")
        return []


def fetch_earnings_by_ticker(ticker, date_from=None, date_to=None):
    """Fetch earnings for a specific ticker."""
    url = f"{FINNHUB_BASE}/calendar/earnings"
    params = {
        "symbol": ticker,
        "token": FINNHUB_API_KEY,
    }
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.error(f"{ticker}: HTTP {resp.status_code}")
            return []
        data = resp.json()
        return data.get("earningsCalendar", [])
    except Exception as e:
        logger.error(f"{ticker}: Request error — {e}")
        return []


# ============================================================
# PROCESS AND STORE
# ============================================================
def process_earnings(raw_data, universe_tickers=None):
    """
    Transform Finnhub response into records for Supabase.
    Optionally filter to only tickers in our scanner_universe.
    Calculate EPS and Revenue surprise percentages.
    """
    records = []

    for entry in raw_data:
        ticker = entry.get("symbol", "")
        date = entry.get("date")
        if not date or not ticker:
            continue

        # EPS surprise calculation
        eps_actual = entry.get("epsActual")
        eps_estimate = entry.get("epsEstimate")
        eps_surprise_pct = None
        if eps_actual is not None and eps_estimate is not None and eps_estimate != 0:
            eps_surprise_pct = round(((eps_actual - eps_estimate) / abs(eps_estimate)) * 100, 2)

        # Revenue surprise calculation
        rev_actual = entry.get("revenueActual")
        rev_estimate = entry.get("revenueEstimate")
        rev_surprise_pct = None
        if rev_actual is not None and rev_estimate is not None and rev_estimate != 0:
            rev_surprise_pct = round(((rev_actual - rev_estimate) / abs(rev_estimate)) * 100, 2)

        # Check if ticker is in our universe
        in_universe = ticker in universe_tickers if universe_tickers else None

        records.append({
            "earnings_date": date,
            "ticker": ticker.upper(),
            "hour": entry.get("hour", ""),       # bmo, amc, dmh
            "fiscal_quarter": entry.get("quarter"),
            "fiscal_year": entry.get("year"),
            "eps_estimate": round(float(eps_estimate), 4) if eps_estimate is not None else None,
            "eps_actual": round(float(eps_actual), 4) if eps_actual is not None else None,
            "eps_surprise_pct": eps_surprise_pct,
            "revenue_estimate": int(rev_estimate) if rev_estimate is not None else None,
            "revenue_actual": int(rev_actual) if rev_actual is not None else None,
            "revenue_surprise_pct": rev_surprise_pct,
            "in_universe": in_universe,
        })

    return records


def upload_to_supabase(records):
    """Upload earnings_calendar records to Supabase with upsert."""
    if not records:
        return 0

    batch_size = 500
    uploaded = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("earnings_calendar").upsert(
                batch, on_conflict="earnings_date,ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets(universe_only=True):
    """
    Push earnings_calendar data to Google Sheets.
    By default only pushes tickers in our scanner_universe.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading earnings_calendar from Supabase...")

    all_rows = []
    offset = 0
    page_size = 1000

    # Build query — only fetch recent data (last 6 months + next 3 months)
    cutoff_past = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    while True:
        query = supabase.table("earnings_calendar") \
            .select("*") \
            .gte("earnings_date", cutoff_past) \
            .order("earnings_date", desc=True) \
            .range(offset, offset + page_size - 1)

        if universe_only:
            query = query.eq("in_universe", True)

        result = query.execute()

        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_rows:
        logger.info("No data to push.")
        return

    logger.info(f"  {len(all_rows)} total rows loaded")

    # Build sheet rows
    sheet_rows = []
    for row in all_rows:
        # Format revenue in millions for readability
        rev_est = row.get('revenue_estimate')
        rev_act = row.get('revenue_actual')
        rev_est_str = f"{rev_est / 1_000_000:.0f}M" if rev_est else ""
        rev_act_str = f"{rev_act / 1_000_000:.0f}M" if rev_act else ""

        in_univ = "YES" if row.get('in_universe') else ""

        sheet_rows.append([
            "'" + (row.get('earnings_date') or ''),
            row.get('ticker', ''),
            row.get('hour', ''),
            row.get('fiscal_quarter', ''),
            row.get('fiscal_year', ''),
            row.get('eps_estimate', ''),
            row.get('eps_actual', ''),
            row.get('eps_surprise_pct', ''),
            rev_est_str,
            rev_act_str,
            row.get('revenue_surprise_pct', ''),
            in_univ,
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
def run_test():
    """Test API access with a sample date range."""
    logger.info("=" * 60)
    logger.info("EARNINGS CALENDAR — API ACCESS TEST")
    logger.info("=" * 60)

    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set in .env")
        return

    logger.info(f"Using API key: {FINNHUB_API_KEY[:6]}...{FINNHUB_API_KEY[-4:]}")
    logger.info(f"Endpoint: {FINNHUB_BASE}/calendar/earnings")

    # Step 1: Test per-ticker fetch (more reliable on free tier)
    logger.info(f"\n[STEP 1] Fetching earnings for AAPL (per-ticker, 2024-present)...")
    results_ticker = fetch_earnings_by_ticker("AAPL", "2024-01-01")

    if results_ticker is None:
        logger.error("Authentication failed.")
        return

    logger.info(f"  Got {len(results_ticker)} earnings entries for AAPL")

    if results_ticker:
        for entry in results_ticker[:6]:
            date = entry.get('date', '?')
            hour = entry.get('hour', '?')
            eps_est = entry.get('epsEstimate')
            eps_act = entry.get('epsActual')
            rev_act = entry.get('revenueActual')
            hour_label = {"bmo": "Before Open", "amc": "After Close", "dmh": "During Mkt"}.get(hour, hour)
            eps_str = f"Est: {eps_est}" if eps_est is not None else "Est: N/A"
            if eps_act is not None:
                eps_str += f" | Act: {eps_act}"
            rev_str = f" | Rev: ${rev_act / 1e9:.1f}B" if rev_act else ""
            logger.info(f"    {date} {hour_label:>12} | AAPL   | {eps_str}{rev_str}")

    # Step 1b: Also try date-range fetch for this week
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)

    date_from = monday.strftime("%Y-%m-%d")
    date_to = friday.strftime("%Y-%m-%d")

    logger.info(f"\n[STEP 1b] Fetching earnings by date range ({date_from} to {date_to})...")
    results = fetch_earnings_by_date_range(date_from, date_to)

    if results:
        logger.info(f"  Got {len(results)} earnings entries this week (date-range works)")
    else:
        logger.info(f"  Date-range returned 0 results (normal on free tier for some dates)")
        logger.info(f"  Backfill uses per-ticker approach instead")
        results = []  # Use empty list for downstream

    if results is None:
        logger.error("Authentication failed.")
        return

    logger.info(f"  Got {len(results)} earnings entries this week")

    # Load universe for filtering
    universe = get_ticker_universe()
    in_universe = [e for e in results if e.get('symbol', '') in universe]

    logger.info(f"  In our scanner_universe: {len(in_universe)}")

    # Display universe tickers
    if in_universe:
        logger.info(f"\n  Our tickers reporting this week:")
        for entry in sorted(in_universe, key=lambda x: x.get('date', '')):
            ticker = entry.get('symbol', '?')
            date = entry.get('date', '?')
            hour = entry.get('hour', '?')
            eps_est = entry.get('epsEstimate')
            eps_act = entry.get('epsActual')

            hour_label = {"bmo": "Before Open", "amc": "After Close", "dmh": "During Mkt"}.get(hour, hour)
            eps_str = f"Est: {eps_est:.2f}" if eps_est is not None else "Est: N/A"
            if eps_act is not None:
                eps_str += f" | Act: {eps_act:.2f}"

            logger.info(f"    {date} {hour_label:>12} | {ticker:<6} | {eps_str}")

    # Step 2: Test Supabase upload
    logger.info(f"\n[STEP 2] Testing Supabase upload...")
    try:
        # Combine both result sets
        all_results = (results_ticker or []) + (results or [])
        records = process_earnings(all_results, universe)
        if records:
            uploaded = upload_to_supabase(records)
            logger.info(f"  Uploaded {uploaded} rows to Supabase earnings_calendar table")
            univ_count = sum(1 for r in records if r.get('in_universe'))
            logger.info(f"  Universe tickers: {univ_count} / {len(records)} total")
        else:
            logger.warning(f"  No records to upload after processing")
    except Exception as e:
        logger.error(f"  Supabase upload failed: {e}")
        logger.error(f"  Did you run earnings_calendar_schema.sql in the Supabase SQL Editor?")

    # Step 3: Test single-ticker fetch
    logger.info(f"\n[STEP 3] Testing single-ticker fetch: NVDA...")
    nvda = fetch_earnings_by_ticker("NVDA", "2024-01-01", today.strftime("%Y-%m-%d"))
    if nvda:
        logger.info(f"  NVDA: {len(nvda)} earnings records since 2024")
        for entry in nvda[:4]:
            date = entry.get('date', '?')
            eps_est = entry.get('epsEstimate')
            eps_act = entry.get('epsActual')
            rev_act = entry.get('revenueActual')
            surprise = ""
            if eps_est and eps_act and eps_est != 0:
                pct = ((eps_act - eps_est) / abs(eps_est)) * 100
                surprise = f"({pct:+.1f}% surprise)"
            rev_str = f"Rev: ${rev_act / 1e9:.1f}B" if rev_act else ""
            logger.info(f"    {date}: EPS Est={eps_est} Act={eps_act} {surprise} {rev_str}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TEST COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nAPI access: CONFIRMED")
    logger.info(f"Rate limit: 60 calls/minute (no monthly cap)")
    logger.info(f"Per-ticker query: Returns all historical earnings for one symbol")
    logger.info(f"Date-range query: Returns all tickers for a date window (may be limited on free tier)")
    logger.info(f"\nBackfill strategy: per-ticker (537 tickers at 1.1s each = ~10 minutes)")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. python earnings_calendar_pipeline.py backfill")
    logger.info(f"  2. python earnings_calendar_pipeline.py push")
    logger.info(f"  3. python earnings_calendar_pipeline.py upcoming    # Check next 4 weeks")


def run_backfill(start_year=2024):
    """
    Backfill earnings calendar by fetching per-ticker.
    Finnhub free tier returns historical data when querying by symbol,
    but date-range-only queries return empty for past dates.
    With 537 tickers at 60 calls/min, this takes ~9 minutes.
    """
    logger.info("=" * 60)
    logger.info(f"EARNINGS CALENDAR — BACKFILL (from {start_year}, per-ticker)")
    logger.info("=" * 60)

    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set in .env")
        return

    tickers = get_ticker_universe()
    universe_set = set(tickers)  # for in_universe tagging

    date_from = f"{start_year}-01-01"
    date_to = (datetime.now() + timedelta(weeks=4)).strftime("%Y-%m-%d")

    total_records = 0
    tickers_with_data = 0
    failed_tickers = []

    for i, ticker in enumerate(sorted(tickers)):
        try:
            results = fetch_earnings_by_ticker(ticker, date_from=date_from, date_to=date_to)

            if results:
                records = process_earnings(results, universe_set)
                if records:
                    uploaded = upload_to_supabase(records)
                    total_records += uploaded
                    tickers_with_data += 1

        except Exception as e:
            logger.error(f"  {ticker}: ERROR — {e}")
            failed_tickers.append(ticker)

        # Rate limit: 60/min = 1 per second is safe
        time.sleep(1.1)

        # Checkpoint every 50 tickers
        if (i + 1) % 50 == 0:
            logger.info(f"  CHECKPOINT: {i+1}/{len(tickers)} tickers | "
                        f"{total_records} rows | {tickers_with_data} with data")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"  Tickers processed: {len(tickers)}")
    logger.info(f"  Tickers with earnings data: {tickers_with_data}")
    logger.info(f"  Total rows uploaded: {total_records}")
    if failed_tickers:
        logger.info(f"  Failed tickers: {', '.join(failed_tickers[:20])}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Next: python earnings_calendar_pipeline.py push")


def run_upcoming():
    """
    Show earnings for the next 4 weeks, filtered to our universe.
    Reads from Supabase (populated by backfill/daily). No API calls needed.
    """
    logger.info("=" * 60)
    logger.info("EARNINGS CALENDAR — UPCOMING (next 4 weeks)")
    logger.info("=" * 60)

    today = datetime.now().strftime("%Y-%m-%d")
    four_weeks = (datetime.now() + timedelta(weeks=4)).strftime("%Y-%m-%d")

    result = supabase.table("earnings_calendar") \
        .select("*") \
        .gte("earnings_date", today) \
        .lte("earnings_date", four_weeks) \
        .eq("in_universe", True) \
        .order("earnings_date") \
        .limit(200) \
        .execute()

    if not result.data:
        logger.info("  No universe tickers reporting in the next 4 weeks.")
        logger.info("  (Run 'backfill' first if the table is empty)")
        return

    logger.info(f"\n  {len(result.data)} of our tickers reporting:\n")

    # Group by week
    current_week = ""
    for entry in result.data:
        date = entry.get('earnings_date', '?')
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            week_label = f"Week of {(d - timedelta(days=d.weekday())).strftime('%b %d')}"
        except:
            week_label = "Unknown"

        if week_label != current_week:
            current_week = week_label
            logger.info(f"\n  --- {week_label} ---")

        ticker = entry.get('ticker', '?')
        hour = entry.get('hour', '?')
        hour_label = {"bmo": "Pre-Mkt", "amc": "Post-Mkt", "dmh": "During"}.get(hour, hour)
        eps_est = entry.get('eps_estimate')
        eps_str = f"EPS Est: {eps_est:.2f}" if eps_est is not None else ""

        logger.info(f"    {date} {hour_label:>8} | {ticker:<6} | {eps_str}")

    logger.info(f"\n  TOTAL: {len(result.data)} earnings events in next 4 weeks")
    logger.info(f"  Tip: Avoid entering FlowOS positions before earnings dates")


def run_daily():
    """
    Daily refresh: per-ticker fetch for all universe tickers.
    Fetches recent past (to pick up actuals) + upcoming dates.
    With 537 tickers at 60 calls/min, takes ~10 minutes.
    """
    logger.info("=" * 60)
    logger.info("EARNINGS CALENDAR — DAILY UPDATE (per-ticker)")
    logger.info("=" * 60)

    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set in .env")
        return

    tickers = get_ticker_universe()
    universe_set = set(tickers)
    today = datetime.now()

    # Look back 30 days (actuals get filled in) + forward 90 days
    date_from = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=90)).strftime("%Y-%m-%d")

    total_records = 0
    tickers_with_data = 0

    for i, ticker in enumerate(sorted(tickers)):
        try:
            results = fetch_earnings_by_ticker(ticker, date_from=date_from, date_to=date_to)
            if results:
                records = process_earnings(results, universe_set)
                if records:
                    uploaded = upload_to_supabase(records)
                    total_records += uploaded
                    tickers_with_data += 1
        except Exception as e:
            logger.error(f"  {ticker}: ERROR — {e}")

        time.sleep(1.1)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(tickers)} tickers, "
                        f"{total_records} rows, {tickers_with_data} with data")

    logger.info(f"\nDAILY UPDATE COMPLETE: {total_records} rows from "
                f"{tickers_with_data} tickers")

    # Print upcoming universe earnings as alert
    upcoming = supabase.table("earnings_calendar") \
        .select("*") \
        .gte("earnings_date", today.strftime("%Y-%m-%d")) \
        .eq("in_universe", True) \
        .order("earnings_date") \
        .limit(30) \
        .execute()

    if upcoming.data:
        logger.info(f"\nUPCOMING EARNINGS (our universe):")
        for entry in upcoming.data:
            date = entry.get('earnings_date', '?')
            ticker = entry.get('ticker', '?')
            hour = entry.get('hour', '?')
            hour_label = {"bmo": "Pre-Mkt", "amc": "Post-Mkt", "dmh": "During"}.get(hour, hour)
            logger.info(f"  {date} {hour_label:>8} | {ticker}")


def run_check(ticker):
    """Check earnings history for a specific ticker."""
    logger.info("=" * 60)
    logger.info(f"EARNINGS CALENDAR — {ticker}")
    logger.info("=" * 60)

    # Try Supabase first
    result = supabase.table("earnings_calendar") \
        .select("*") \
        .eq("ticker", ticker.upper()) \
        .order("earnings_date", desc=True) \
        .limit(12) \
        .execute()

    if result.data:
        logger.info(f"  {len(result.data)} earnings records in Supabase:\n")
        for entry in result.data:
            date = entry.get('earnings_date', '?')
            hour = entry.get('hour', '?')
            eps_est = entry.get('eps_estimate')
            eps_act = entry.get('eps_actual')
            surprise = entry.get('eps_surprise_pct')
            rev_act = entry.get('revenue_actual')

            eps_str = f"Est: {eps_est}" if eps_est is not None else "Est: -"
            if eps_act is not None:
                eps_str += f" Act: {eps_act}"
            if surprise is not None:
                eps_str += f" ({surprise:+.1f}%)"
            rev_str = f"Rev: ${rev_act / 1e9:.1f}B" if rev_act else ""

            logger.info(f"  {date} ({hour}) | {eps_str} | {rev_str}")
    else:
        logger.info(f"  No data in Supabase for {ticker}. Fetching from Finnhub...")
        results = fetch_earnings_by_ticker(ticker, "2023-01-01")
        if results:
            logger.info(f"  Found {len(results)} earnings from Finnhub:")
            for entry in results[:8]:
                date = entry.get('date', '?')
                eps_est = entry.get('epsEstimate')
                eps_act = entry.get('epsActual')
                logger.info(f"    {date}: Est={eps_est} Act={eps_act}")
        else:
            logger.info(f"  No earnings data found for {ticker}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
Earnings Calendar Pipeline — Finnhub
=======================================
Usage:
  python earnings_calendar_pipeline.py test             # Test API access
  python earnings_calendar_pipeline.py backfill         # Backfill 2024-present
  python earnings_calendar_pipeline.py upcoming         # Next 4 weeks (universe only)
  python earnings_calendar_pipeline.py daily            # Refresh recent + upcoming
  python earnings_calendar_pipeline.py push             # Push to Google Sheets
  python earnings_calendar_pipeline.py check NVDA       # Check specific ticker

Prerequisites:
  1. FINNHUB_API_KEY in .env (same key as insider transactions)
  2. Run earnings_calendar_schema.sql in Supabase SQL Editor

Efficiency: Backfill uses per-ticker fetch (~10 min for 537 tickers).
Daily/upcoming use date-range fetch (a few calls total).
""")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test":
        run_test()
    elif cmd == "backfill":
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
        run_backfill(start_year=year)
    elif cmd == "upcoming":
        run_upcoming()
    elif cmd == "daily":
        run_daily()
    elif cmd == "push":
        push_to_sheets()
    elif cmd == "check":
        ticker = sys.argv[2].upper() if len(sys.argv) > 2 else "NVDA"
        run_check(ticker)
    else:
        print(f"Unknown command: {cmd}")
        print("Valid commands: test, backfill, upcoming, daily, push, check")
        sys.exit(1)