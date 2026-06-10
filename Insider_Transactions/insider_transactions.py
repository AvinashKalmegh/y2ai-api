"""
Insider Transactions Pipeline
==============================
Fetches SEC Form 3/4/5 insider buy/sell data from Finnhub.
Stores in Supabase insider_transactions table and pushes to Google Sheets.

Insider transactions are event-driven (sparse), not continuous signals.
The spec says: "When a CFO sells $5M while DM is falling, that's triple confirmation."
Pipeline focuses on significant transactions ($500K+) for ARGUS/FlowOS signal use.

Usage:
  python insider_transactions_pipeline.py test AAPL         # Test API access
  python insider_transactions_pipeline.py backfill          # Backfill all tickers (2024-present)
  python insider_transactions_pipeline.py daily             # Daily update, all tickers
  python insider_transactions_pipeline.py push              # Push Supabase data to Sheets
  python insider_transactions_pipeline.py summary           # Print recent significant activity

API Source: Finnhub (https://finnhub.io/docs/api/insider-transactions)
  Endpoint: GET https://finnhub.io/api/v1/stock/insider-transactions
  Auth: token query parameter
  Free tier: 60 calls/minute (no monthly cap)
  Limit: 100 transactions per call
  Response: change (+buy/-sell), filingDate, name, share, transactionCode,
            transactionDate, transactionPrice, symbol

Why Finnhub over API Ninjas:
  - 60 calls/minute vs 3,000 calls/month (can sweep all 537 tickers daily)
  - 100 results per call vs 10
  - No monthly budget management needed
  - Also provides insider-sentiment (MSPR) as a bonus signal

Prerequisites:
  - FINNHUB_API_KEY in .env (free at https://finnhub.io/register)
  - Run insider_transactions_schema.sql in Supabase first
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
# Dedicated "Insider_Transactions" workbook (opened by ID). Insider history grew to
# ~500K+ rows after the 2018 backfill — far too large for the shared
# Macro_and_Options_History workbook's 10M-cell budget, so it lives on its own.
SHEETS_SPREADSHEET = "Macro_and_Options_History"  # legacy name, kept for reference
SHEETS_SPREADSHEET_ID = "1JjZ_KF7cQKAd_4u_q1bt-VWf2eTfClNYRBaPyfRUI2w"
SHEETS_TAB = "Insider_Transactions"

SHEETS_HEADERS = [
    "Filing_Date", "Transaction_Date", "Ticker", "Insider_Name",
    "Transaction_Type", "Transaction_Code", "Shares_Changed",
    "Price", "Value", "Shares_After", "Significant"
]

# Significant transaction threshold (from spec: "Flag transactions above $500K")
SIGNIFICANT_THRESHOLD = 500_000

# Transaction code mapping (SEC Form 4 codes)
TRANSACTION_CODES = {
    "P": "Purchase",
    "S": "Sale",
    "A": "Award",
    "D": "Disposition",
    "F": "Tax Payment",
    "G": "Gift",
    "M": "Exercise/Conversion",
    "C": "Conversion",
    "X": "Exercise",
    "J": "Other",
    "W": "Will/Inheritance",
    "K": "Equity Swap",
    "I": "Discretionary",
    "Z": "Trust",
    "U": "Tender",
    "L": "Small Acquisition",
    "H": "Expiration (short)",
    "E": "Expiration (long)",
}


# ============================================================
# TICKER UNIVERSE (from Supabase)
# ============================================================
def get_ticker_universe():
    """Load all tickers from scanner_universe."""
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

    logger.info(f"Loaded {len(all_tickers)} tickers from scanner_universe")
    return all_tickers


# ============================================================
# FINNHUB API: INSIDER TRANSACTIONS
# ============================================================
def fetch_insider_transactions(ticker, date_from=None, date_to=None):
    """
    Fetch insider transactions for a single ticker from Finnhub.
    Returns up to 100 transactions per call.

    Finnhub response format:
    {
      "data": [
        {
          "change": 5000,           # positive=buy, negative=sell
          "filingDate": "2024-01-15",
          "name": "John Smith",
          "share": 50000,           # shares held after transaction
          "symbol": "AAPL",
          "transactionCode": "P",
          "transactionDate": "2024-01-12",
          "transactionPrice": 185.50
        }
      ],
      "symbol": "AAPL"
    }
    """
    url = f"{FINNHUB_BASE}/stock/insider-transactions"
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

        if resp.status_code == 401:
            logger.error(f"  {ticker}: 401 Unauthorized — check FINNHUB_API_KEY")
            return None
        elif resp.status_code == 429:
            logger.warning(f"  {ticker}: Rate limited, waiting 30s...")
            time.sleep(30)
            return fetch_insider_transactions(ticker, date_from, date_to)
        elif resp.status_code != 200:
            logger.error(f"  {ticker}: HTTP {resp.status_code} — {resp.text[:200]}")
            return []

        data = resp.json()

        # Finnhub wraps results in "data" key
        transactions = data.get("data", [])
        return transactions

    except Exception as e:
        logger.error(f"  {ticker}: Request error — {e}")
        return []


# ============================================================
# PROCESS AND STORE
# ============================================================
def process_transactions(ticker, raw_data):
    """
    Transform Finnhub response into records for Supabase.
    Flag transactions above $500K as significant.
    """
    records = []

    for entry in raw_data:
        filing_date = entry.get("filingDate")
        txn_date = entry.get("transactionDate")
        if not filing_date:
            continue

        # "change" field: positive = buy, negative = sell
        change = entry.get("change", 0) or 0
        price = entry.get("transactionPrice", 0) or 0
        shares_after = entry.get("share")

        # Calculate transaction value
        value = abs(change * price) if change and price else 0

        # Determine transaction type from change direction and code
        txn_code = entry.get("transactionCode", "")
        if txn_code in TRANSACTION_CODES:
            txn_type = TRANSACTION_CODES[txn_code]
        elif change > 0:
            txn_type = "Purchase"
        elif change < 0:
            txn_type = "Sale"
        else:
            txn_type = "Other"

        # Flag significant transactions
        is_significant = value >= SIGNIFICANT_THRESHOLD if value else False

        records.append({
            "filing_date": filing_date,
            "transaction_date": txn_date,
            "ticker": entry.get("symbol", ticker).upper(),
            "insider_name": entry.get("name", ""),
            "transaction_type": txn_type,
            "transaction_code": txn_code,
            "shares_changed": int(change) if change else 0,
            "price_per_share": round(float(price), 2) if price else None,
            "transaction_value": round(float(value), 2) if value else None,
            "shares_after": int(shares_after) if shares_after else None,
            "is_significant": is_significant,
        })

    return records


def upload_to_supabase(records):
    """Upload insider_transactions records to Supabase with upsert."""
    if not records:
        return 0

    # Deduplicate on the conflict key to avoid "cannot affect row a second time"
    seen = {}
    for r in records:
        key = (r['filing_date'], r['ticker'], r['insider_name'], r['transaction_code'], r['shares_changed'])
        seen[key] = r  # last occurrence wins
    records = list(seen.values())

    batch_size = 200
    uploaded = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("insider_transactions").upsert(
                batch, on_conflict="filing_date,ticker,insider_name,transaction_code,shares_changed"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets():
    """Push insider_transactions data to Google Sheets."""
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading insider_transactions from Supabase...")

    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        result = supabase.table("insider_transactions") \
            .select("*") \
            .order("filing_date", desc=True) \
            .range(offset, offset + page_size - 1) \
            .execute()

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
        sig = "YES" if row.get('is_significant') else ""
        sheet_rows.append([
            "'" + (row.get('filing_date') or ''),
            "'" + (row.get('transaction_date') or ''),
            row.get('ticker', ''),
            row.get('insider_name', ''),
            row.get('transaction_type', ''),
            row.get('transaction_code', ''),
            row.get('shares_changed', ''),
            row.get('price_per_share', ''),
            row.get('transaction_value', ''),
            row.get('shares_after', ''),
            sig,
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
        spreadsheet = client.open_by_key(SHEETS_SPREADSHEET_ID)
    except Exception as e:
        logger.error(f"Spreadsheet ID '{SHEETS_SPREADSHEET_ID}' not accessible: {e}")
        return

    # Get or create tab
    try:
        sheet = spreadsheet.worksheet(SHEETS_TAB)
        sheet.clear()
        if sheet.row_count < len(sheet_rows) + 10:
            sheet.resize(rows=len(sheet_rows) + 100, cols=len(SHEETS_HEADERS))
    except gspread.WorksheetNotFound:
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
def run_test(ticker):
    """Test API access with a single ticker."""
    logger.info("=" * 60)
    logger.info(f"INSIDER TRANSACTIONS — API ACCESS TEST: {ticker}")
    logger.info("=" * 60)

    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set in .env")
        logger.error("Sign up free at https://finnhub.io/register")
        return

    logger.info(f"Using API key: {FINNHUB_API_KEY[:6]}...{FINNHUB_API_KEY[-4:]}")
    logger.info(f"Endpoint: {FINNHUB_BASE}/stock/insider-transactions")

    # Step 1: Fetch recent transactions
    logger.info(f"\n[STEP 1] Fetching insider transactions for {ticker} (last 90 days)...")

    date_from = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    results = fetch_insider_transactions(ticker, date_from=date_from)

    if results is None:
        logger.error("Authentication failed.")
        return

    if not results:
        logger.warning(f"  No insider transactions found for {ticker} in last 90 days")
        logger.info("  Trying without date filter...")
        results = fetch_insider_transactions(ticker)

    if not results:
        logger.warning(f"  Still no data. This can be normal — trying NVDA instead...")
        if ticker != "NVDA":
            run_test("NVDA")
        return

    logger.info(f"  Got {len(results)} transactions")

    # Display results
    for entry in results[:15]:
        name = entry.get('name', 'Unknown')
        change = entry.get('change', 0) or 0
        price = entry.get('transactionPrice', 0) or 0
        value = abs(change * price) if change and price else 0
        date = entry.get('filingDate', '?')
        code = entry.get('transactionCode', '?')
        txn_type = TRANSACTION_CODES.get(code, "Other")

        direction = "BUY" if change > 0 else "SELL" if change < 0 else "OTHER"
        value_str = f"${value:,.0f}" if value else "N/A"
        sig = ""
        if value >= SIGNIFICANT_THRESHOLD:
            sig = " *** SIGNIFICANT ***"

        logger.info(f"  {date}: {name} — {direction} ({code}={txn_type}) — "
                     f"{abs(change):,} shares @ ${price:.2f} = {value_str}{sig}")

    if len(results) > 15:
        logger.info(f"  ... and {len(results) - 15} more")

    # Step 2: Test Supabase upload
    logger.info(f"\n[STEP 2] Testing Supabase upload...")
    try:
        records = process_transactions(ticker, results)
        if records:
            uploaded = upload_to_supabase(records)
            logger.info(f"  Uploaded {uploaded} rows to Supabase insider_transactions table")
            sig_total = sum(1 for r in records if r.get('is_significant'))
            logger.info(f"  Significant transactions (>$500K): {sig_total}")
        else:
            logger.warning(f"  No records to upload after processing")
    except Exception as e:
        logger.error(f"  Supabase upload failed: {e}")
        logger.error(f"  Did you run insider_transactions_schema.sql in the Supabase SQL Editor?")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TEST COMPLETE for {ticker}")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nAPI access: CONFIRMED")
    logger.info(f"Rate limit: 60 calls/minute (no monthly cap)")
    logger.info(f"Results per call: up to 100")
    logger.info(f"\nWith 537 tickers at 60/min, full daily sweep takes ~9 minutes")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. python insider_transactions_pipeline.py backfill    # 2024-present, all tickers")
    logger.info(f"  2. python insider_transactions_pipeline.py push        # Sync to Google Sheets")
    logger.info(f"  3. python insider_transactions_pipeline.py daily       # Add to daily workflow")


def run_backfill(start_date="2024-01-01"):
    """
    Backfill insider transactions for all tickers.
    Finnhub has no monthly cap, so we can sweep the full universe.
    """
    logger.info("=" * 60)
    logger.info(f"INSIDER TRANSACTIONS — BACKFILL (from {start_date})")
    logger.info("=" * 60)

    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set in .env")
        return

    tickers = get_ticker_universe()
    date_to = datetime.now().strftime("%Y-%m-%d")

    total_records = 0
    total_significant = 0
    tickers_with_data = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        try:
            results = fetch_insider_transactions(ticker, date_from=start_date, date_to=date_to)

            if results is None:
                logger.error("Authentication failed — stopping.")
                break

            if results:
                records = process_transactions(ticker, results)
                if records:
                    uploaded = upload_to_supabase(records)
                    total_records += uploaded
                    sig_count = sum(1 for r in records if r.get('is_significant'))
                    total_significant += sig_count
                    tickers_with_data += 1

                    if sig_count > 0:
                        logger.info(f"[{i+1}/{len(tickers)}] {ticker}: "
                                     f"{len(records)} txns ({sig_count} significant)")

        except Exception as e:
            logger.error(f"  {ticker}: ERROR — {e}")
            failed_tickers.append(ticker)

        # Rate limit: 60/min = 1 per second is safe
        time.sleep(1.1)

        # Checkpoint every 50 tickers
        if (i + 1) % 50 == 0:
            logger.info(f"\n--- CHECKPOINT: {i+1}/{len(tickers)} tickers | "
                        f"{total_records} rows | {total_significant} significant | "
                        f"{tickers_with_data} tickers with data ---\n")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"  Tickers processed: {i+1}")
    logger.info(f"  Tickers with insider data: {tickers_with_data}")
    logger.info(f"  Total rows uploaded: {total_records}")
    logger.info(f"  Significant transactions (>$500K): {total_significant}")
    if failed_tickers:
        logger.info(f"  Failed tickers: {', '.join(failed_tickers[:20])}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Next: python insider_transactions_pipeline.py push")


def run_daily():
    """
    Daily update: fetch latest insider transactions for all tickers.
    With 60 calls/min and 537 tickers, takes ~9 minutes.
    """
    logger.info("=" * 60)
    logger.info("INSIDER TRANSACTIONS — DAILY UPDATE")
    logger.info("=" * 60)

    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not set in .env")
        return

    tickers = get_ticker_universe()

    # Look back 14 days to catch any we missed (filings can lag)
    date_from = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")

    total_records = 0
    total_significant = 0
    new_significant = []

    for i, ticker in enumerate(tickers):
        try:
            results = fetch_insider_transactions(ticker, date_from=date_from)

            if results is None:
                logger.error("Authentication failed — stopping.")
                break

            if results:
                records = process_transactions(ticker, results)
                if records:
                    uploaded = upload_to_supabase(records)
                    total_records += uploaded

                    for r in records:
                        if r.get('is_significant'):
                            total_significant += 1
                            new_significant.append(r)

        except Exception as e:
            logger.error(f"  {ticker}: ERROR — {e}")

        time.sleep(1.1)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(tickers)} tickers, "
                        f"{total_records} rows, {total_significant} significant")

    logger.info(f"\nDAILY UPDATE COMPLETE: {total_records} rows, {total_significant} significant")

    # Print significant transactions as alerts
    if new_significant:
        logger.info(f"\n{'=' * 60}")
        logger.info("SIGNIFICANT INSIDER ACTIVITY (>$500K):")
        logger.info("=" * 60)

        buys = [t for t in new_significant if t.get('shares_changed', 0) > 0]
        sells = [t for t in new_significant if t.get('shares_changed', 0) < 0]

        if buys:
            logger.info("\n  BUYS:")
            for txn in buys:
                val = txn.get('transaction_value', 0)
                logger.info(
                    f"    {txn['filing_date']} | {txn['ticker']} | "
                    f"{txn['insider_name']} | "
                    f"{txn['shares_changed']:,} shares | ${val:,.0f}"
                )

        if sells:
            logger.info("\n  SELLS:")
            for txn in sells:
                val = txn.get('transaction_value', 0)
                logger.info(
                    f"    {txn['filing_date']} | {txn['ticker']} | "
                    f"{txn['insider_name']} | "
                    f"{abs(txn['shares_changed']):,} shares | ${val:,.0f}"
                )
    else:
        logger.info("  No significant insider transactions in the last 14 days.")


def run_summary():
    """Print summary of recent significant insider activity from Supabase."""
    logger.info("=" * 60)
    logger.info("INSIDER TRANSACTIONS — RECENT SIGNIFICANT ACTIVITY")
    logger.info("=" * 60)

    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    try:
        result = supabase.table("insider_transactions") \
            .select("*") \
            .eq("is_significant", True) \
            .gte("filing_date", cutoff) \
            .order("filing_date", desc=True) \
            .limit(50) \
            .execute()

        if not result.data:
            logger.info("  No significant insider transactions in the last 30 days.")
            return

        logger.info(f"  Found {len(result.data)} significant transactions (last 30 days):\n")

        buys = [r for r in result.data if (r.get('shares_changed') or 0) > 0]
        sells = [r for r in result.data if (r.get('shares_changed') or 0) < 0]
        other = [r for r in result.data if r not in buys and r not in sells]

        if buys:
            logger.info("  INSIDER BUYS (contrarian signal — buying into weakness?):")
            for txn in buys:
                val = txn.get('transaction_value', 0)
                val_str = f"${val:,.0f}" if val else "N/A"
                logger.info(
                    f"    {txn['filing_date']} | {txn['ticker']} | "
                    f"{txn['insider_name']} | {txn.get('shares_changed', 0):,} shares | {val_str}"
                )

        if sells:
            logger.info("\n  INSIDER SELLS (confirmation signal — selling into strength?):")
            for txn in sells:
                val = txn.get('transaction_value', 0)
                val_str = f"${val:,.0f}" if val else "N/A"
                logger.info(
                    f"    {txn['filing_date']} | {txn['ticker']} | "
                    f"{txn['insider_name']} | {abs(txn.get('shares_changed', 0)):,} shares | {val_str}"
                )

        if other:
            logger.info("\n  OTHER (Awards, Exercises, etc.):")
            for txn in other:
                val = txn.get('transaction_value', 0)
                val_str = f"${val:,.0f}" if val else "N/A"
                logger.info(
                    f"    {txn['filing_date']} | {txn['ticker']} | "
                    f"{txn['insider_name']} | {txn.get('transaction_type', '')} | {val_str}"
                )

    except Exception as e:
        logger.error(f"  Error querying Supabase: {e}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
Insider Transactions Pipeline — Finnhub
=========================================
Usage:
  python insider_transactions_pipeline.py test AAPL       # Test API access
  python insider_transactions_pipeline.py backfill        # Backfill all tickers (2024-present)
  python insider_transactions_pipeline.py daily           # Daily update (all tickers)
  python insider_transactions_pipeline.py push            # Push to Google Sheets
  python insider_transactions_pipeline.py summary         # Recent significant activity

Prerequisites:
  1. FINNHUB_API_KEY in .env (free at https://finnhub.io/register)
  2. Run insider_transactions_schema.sql in Supabase SQL Editor
  3. Run 'test' command first to verify API access

Rate limit: 60 calls/min (no monthly cap)
Full sweep of 537 tickers takes ~9 minutes
""")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test":
        ticker = sys.argv[2].upper() if len(sys.argv) > 2 else "AAPL"
        run_test(ticker)
    elif cmd == "backfill":
        start = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
        run_backfill(start_date=start)
    elif cmd == "daily":
        run_daily()
    elif cmd == "push":
        push_to_sheets()
    elif cmd == "summary":
        run_summary()
    else:
        print(f"Unknown command: {cmd}")
        print("Valid commands: test, backfill, daily, push, summary")
        sys.exit(1)