"""
Short Sale Volume Pipeline
===========================
Fetches daily short sale volume from FINRA Reg SHO data (free, no auth).
Calculates Short_Ratio, Short_20D_Avg, Short_Z_Score per ticker.
Stores in Supabase short_volume table and pushes to Google Sheets.

Source: FINRA Consolidated NMS (all exchanges)
URL pattern: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
Available since: 2009-08-03, updated every trading day

Usage:
  python short_volume_pipeline.py backfill          # Full backfill 2020-present
  python short_volume_pipeline.py backfill 2024     # Backfill from specific year
  python short_volume_pipeline.py daily             # Fetch last 5 trading days
  python short_volume_pipeline.py calculate         # Recalculate rolling metrics
  python short_volume_pipeline.py push              # Push to Google Sheets

No API key needed. FINRA data is free and public.
"""

import os
import sys
import io
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

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FINRA Reg SHO daily short sale volume
FINRA_BASE_URL = "https://cdn.finra.org/equity/regsho/daily"
FINRA_PREFIX = "CNMSshvol"  # Consolidated NMS (all exchanges)

# Google Sheets config
SHEETS_SPREADSHEET = "short_volume"
SHEETS_TAB = "Short_Volume"

SHORT_HEADERS = [
    "Date", "Ticker", "Short_Volume", "Total_Volume",
    "Short_Ratio", "Short_20D_Avg", "Short_Z_Score"
]

# Delay between FINRA file downloads (be respectful)
DELAY_BETWEEN_DOWNLOADS = 0.5  # seconds

# Load ticker universe from Supabase scanner_universe
def load_ticker_universe():
    """Load all tickers from scanner_universe."""
    all_tickers = set()
    offset = 0
    while True:
        result = supabase.table("scanner_universe") \
            .select("ticker") \
            .range(offset, offset + 999) \
            .execute()
        if not result.data:
            break
        for row in result.data:
            all_tickers.add(row['ticker'])
        if len(result.data) < 1000:
            break
        offset += 1000
    return all_tickers


# ============================================================
# FINRA DATA FETCHING
# ============================================================
def fetch_finra_daily(date_str):
    """
    Fetch one day's CNMS short sale volume file from FINRA.
    date_str: 'YYYYMMDD' format
    Returns list of dicts or None if no data for that date.
    """
    url = f"{FINRA_BASE_URL}/{FINRA_PREFIX}{date_str}.txt"

    try:
        resp = requests.get(url, timeout=30)

        if resp.status_code == 403 or resp.status_code == 404:
            # No data for this date (weekend/holiday)
            return None

        if resp.status_code != 200:
            logger.warning(f"  {date_str}: HTTP {resp.status_code}")
            return None

        # Check for "Access Denied" response
        if "Access Denied" in resp.text[:100]:
            return None

        # Parse pipe-delimited data
        rows = []
        lines = resp.text.strip().split('\n')

        for line in lines[1:]:  # Skip header
            parts = line.strip().split('|')
            if len(parts) < 5:
                continue

            try:
                rows.append({
                    'date': parts[0][:4] + '-' + parts[0][4:6] + '-' + parts[0][6:8],
                    'ticker': parts[1],
                    'short_volume': int(float(parts[2])),
                    'short_exempt_volume': int(float(parts[3])),
                    'total_volume': int(float(parts[4])),
                })
            except (ValueError, IndexError):
                continue

        return rows

    except requests.exceptions.Timeout:
        logger.warning(f"  {date_str}: Timeout")
        return None
    except Exception as e:
        logger.warning(f"  {date_str}: Error - {e}")
        return None


def get_trading_dates(start_date, end_date):
    """Generate list of potential trading dates (weekdays only)."""
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    return dates


# ============================================================
# CALCULATIONS
# ============================================================
def calculate_short_metrics(df):
    """Calculate Short_Ratio, Short_20D_Avg, Short_Z_Score per ticker."""
    result_dfs = []

    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker].sort_values('date').copy()

        # Short Ratio
        tdf['short_ratio'] = np.where(
            tdf['total_volume'] > 0,
            (tdf['short_volume'] / tdf['total_volume']).round(4),
            0
        )

        # 20-day rolling average
        tdf['short_20d_avg'] = tdf['short_ratio'].rolling(
            window=20, min_periods=10
        ).mean().round(4)

        # Z-Score
        short_std = tdf['short_ratio'].rolling(
            window=20, min_periods=10
        ).std()
        tdf['short_z_score'] = np.where(
            short_std > 0,
            ((tdf['short_ratio'] - tdf['short_20d_avg']) / short_std).round(4),
            0
        )

        result_dfs.append(tdf)

    if not result_dfs:
        return pd.DataFrame()

    return pd.concat(result_dfs, ignore_index=True)


# ============================================================
# SUPABASE
# ============================================================
def upload_to_supabase(df):
    """Upload short_volume data to Supabase."""
    records = []
    for _, row in df.iterrows():
        record = {
            'date': str(row['date']),
            'ticker': row['ticker'],
            'short_volume': int(row['short_volume']),
            'total_volume': int(row['total_volume']),
            'short_ratio': float(row['short_ratio']) if not pd.isna(row.get('short_ratio', np.nan)) else None,
            'short_20d_avg': float(row['short_20d_avg']) if not pd.isna(row.get('short_20d_avg', np.nan)) else None,
            'short_z_score': float(row['short_z_score']) if not pd.isna(row.get('short_z_score', np.nan)) else None,
        }
        records.append(record)

    batch_size = 500
    uploaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("short_volume").upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


def load_from_supabase(ticker=None, limit=None):
    """Load short_volume from Supabase."""
    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        query = supabase.table("short_volume").select("*").order("date", desc=True)
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
def push_to_sheets(full=False):
    """Push short_volume data to Google Sheets.

    If full=False (default): check latest date in sheet, only push newer data.
    If full=True: clear sheet and rewrite everything.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

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
    except Exception:
        sheet = spreadsheet.add_worksheet(
            title=SHEETS_TAB, rows=1000, cols=len(SHORT_HEADERS)
        )
        full = True  # New sheet, must do full push

    # Determine latest date already in sheet
    latest_sheet_date = None
    if not full:
        try:
            date_col = sheet.col_values(1)  # Column A = dates
            # Skip header, find latest date (row 2 is newest since sorted desc)
            if len(date_col) > 1:
                latest_sheet_date = date_col[1].replace("'", "").strip()
                logger.info(f"Latest date in sheet: {latest_sheet_date}")
            else:
                full = True  # Empty sheet
        except Exception as e:
            logger.warning(f"Could not read sheet dates: {e}")
            full = True

    if full:
        logger.info("FULL PUSH — loading all data from Supabase...")
        df = load_from_supabase()
    else:
        # Load only data newer than latest sheet date
        logger.info(f"Loading data newer than {latest_sheet_date} from Supabase...")
        all_rows = []
        offset = 0
        page_size = 1000
        while True:
            result = supabase.table("short_volume").select("*") \
                .gt("date", latest_sheet_date) \
                .order("date", desc=True) \
                .range(offset, offset + page_size - 1) \
                .execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            if len(result.data) < page_size:
                break
            offset += page_size
        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if df.empty:
        logger.info("No new data to push.")
        return

    logger.info(f"  {len(df)} rows to push")

    # Build sheet rows
    sheet_rows = []
    for _, row in df.iterrows():
        date_str = str(row.get('date', ''))
        sheet_rows.append([
            "'" + date_str,
            row.get('ticker', ''),
            row.get('short_volume', ''),
            row.get('total_volume', ''),
            row.get('short_ratio', ''),
            row.get('short_20d_avg', ''),
            row.get('short_z_score', ''),
        ])

    # Sort newest-first
    sheet_rows.sort(key=lambda r: r[0], reverse=True)

    # Replace NaN
    for row in sheet_rows:
        for j in range(len(row)):
            if row[j] is None or (isinstance(row[j], float) and np.isnan(row[j])):
                row[j] = ''

    if full:
        # Full push: clear and rewrite
        sheet.clear()
        if sheet.row_count < len(sheet_rows) + 10:
            sheet.resize(rows=len(sheet_rows) + 100, cols=len(SHORT_HEADERS))
        sheet.update(range_name='A1', values=[SHORT_HEADERS], value_input_option='USER_ENTERED')
        sheet.format('1:1', {'textFormat': {'bold': True}})

        BATCH_SIZE = 10000
        total = 0
        for i in range(0, len(sheet_rows), BATCH_SIZE):
            batch = sheet_rows[i:i + BATCH_SIZE]
            sheet.update(range_name=f'A{i+2}', values=batch, value_input_option='USER_ENTERED')
            total += len(batch)
            logger.info(f"  Written {total}/{len(sheet_rows)} rows")
            time.sleep(5)
    else:
        # Incremental push: insert new rows at top (after header)
        current_rows = sheet.row_count
        needed = current_rows + len(sheet_rows)
        if needed > current_rows:
            sheet.resize(rows=needed + 100, cols=len(SHORT_HEADERS))

        # Insert new rows after header row
        sheet.insert_rows(sheet_rows, row=2, value_input_option='USER_ENTERED')
        total = len(sheet_rows)
        logger.info(f"  Inserted {total} new rows at top of sheet")

    logger.info(f"Push complete: {total} rows")


# ============================================================
# COMMANDS
# ============================================================
def run_backfill(start_year=2020):
    """
    Backfill short sale volume from FINRA.
    Downloads daily CNMS files, filters to scanner_universe tickers,
    calculates metrics, uploads to Supabase.

    Spec says 2020-present. ~1,250 trading days = ~10 min at 0.5s delay.
    """
    logger.info("=" * 60)
    logger.info("SHORT VOLUME - BACKFILL")
    logger.info(f"Source: FINRA Reg SHO (CNMS)")
    logger.info(f"Start: {start_year}-01-01")
    logger.info("=" * 60)

    # Load ticker universe
    logger.info("\nLoading scanner_universe...")
    universe = load_ticker_universe()
    logger.info(f"  {len(universe)} tickers in universe")

    # Generate date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime.now() - timedelta(days=1)  # yesterday
    trading_dates = get_trading_dates(start_date, end_date)
    logger.info(f"  {len(trading_dates)} potential trading days to process")

    # Process in batches to upload periodically
    all_rows = []
    dates_with_data = 0
    dates_no_data = 0
    total_uploaded = 0
    upload_batch_size = 50  # upload every 50 days of data

    for i, date_str in enumerate(trading_dates):
        rows = fetch_finra_daily(date_str)

        if rows is None:
            dates_no_data += 1
        else:
            dates_with_data += 1
            # Filter to our universe
            filtered = [r for r in rows if r['ticker'] in universe]
            all_rows.extend(filtered)

        # Progress logging every 50 dates
        if (i + 1) % 50 == 0:
            logger.info(
                f"  [{i+1}/{len(trading_dates)}] "
                f"Data: {dates_with_data} days, "
                f"Skip: {dates_no_data}, "
                f"Rows buffered: {len(all_rows)}"
            )

        # Periodic upload
        if len(all_rows) >= 10000:
            df = pd.DataFrame(all_rows)
            df = calculate_short_metrics(df)
            uploaded = upload_to_supabase(df)
            total_uploaded += uploaded
            logger.info(f"  ** Uploaded {uploaded} rows (total: {total_uploaded})")
            all_rows = []

        time.sleep(DELAY_BETWEEN_DOWNLOADS)

    # Final upload
    if all_rows:
        df = pd.DataFrame(all_rows)
        df = calculate_short_metrics(df)
        uploaded = upload_to_supabase(df)
        total_uploaded += uploaded
        logger.info(f"  ** Final upload: {uploaded} rows")

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Trading days with data: {dates_with_data}")
    logger.info(f"Days skipped (no data): {dates_no_data}")
    logger.info(f"Total rows uploaded: {total_uploaded}")
    logger.info(f"\nNext: python short_volume_pipeline.py push")


def run_daily():
    """Fetch last 5 trading days to catch any gaps."""
    logger.info("=" * 60)
    logger.info("SHORT VOLUME - DAILY UPDATE")
    logger.info("=" * 60)

    universe = load_ticker_universe()
    logger.info(f"  {len(universe)} tickers in universe")

    # Last 7 calendar days (covers 5 trading days + weekend)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    trading_dates = get_trading_dates(start_date, end_date)
    logger.info(f"  Checking {len(trading_dates)} dates")

    all_rows = []
    for date_str in trading_dates:
        rows = fetch_finra_daily(date_str)
        if rows:
            filtered = [r for r in rows if r['ticker'] in universe]
            all_rows.extend(filtered)
            formatted = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:]
            logger.info(f"  {formatted}: {len(filtered)} tickers")
        time.sleep(DELAY_BETWEEN_DOWNLOADS)

    if not all_rows:
        logger.info("No new data.")
        return

    # Need historical data for rolling calcs
    # Load last 30 days from Supabase
    logger.info("Loading recent history for rolling calculations...")
    existing = load_from_supabase()

    df_new = pd.DataFrame(all_rows)

    if not existing.empty:
        existing = existing[['date', 'ticker', 'short_volume', 'total_volume']]
        # Convert types
        for col in ['short_volume', 'total_volume']:
            existing[col] = pd.to_numeric(existing[col], errors='coerce').fillna(0).astype(int)
        combined = pd.concat([existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
    else:
        combined = df_new

    combined = calculate_short_metrics(combined)

    # Only upload the new dates
    new_dates = set(df_new['date'].unique())
    to_upload = combined[combined['date'].isin(new_dates)]

    uploaded = upload_to_supabase(to_upload)
    logger.info(f"\nUploaded {uploaded} rows")
    logger.info("DAILY UPDATE COMPLETE")


def run_calculate():
    """Recalculate rolling metrics on all existing data."""
    logger.info("=" * 60)
    logger.info("SHORT VOLUME - RECALCULATE METRICS")
    logger.info("=" * 60)

    logger.info("Loading all data from Supabase...")
    df = load_from_supabase()

    if df.empty:
        logger.info("No data found.")
        return

    logger.info(f"  {len(df)} rows loaded")

    # Convert types
    for col in ['short_volume', 'total_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df = calculate_short_metrics(df)
    uploaded = upload_to_supabase(df)
    logger.info(f"Recalculated and uploaded {uploaded} rows")


def run_push(full=False):
    """Push to Google Sheets."""
    logger.info("=" * 60)
    logger.info(f"SHORT VOLUME - PUSH TO SHEETS ({'FULL' if full else 'INCREMENTAL'})")
    logger.info("=" * 60)
    push_to_sheets(full=full)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python short_volume_pipeline.py backfill          # Full backfill 2020-present")
        print("  python short_volume_pipeline.py backfill 2024     # Backfill from specific year")
        print("  python short_volume_pipeline.py daily             # Fetch last 5 trading days")
        print("  python short_volume_pipeline.py calculate         # Recalculate rolling metrics")
        print("  python short_volume_pipeline.py push              # Incremental push (new dates only)")
        print("  python short_volume_pipeline.py push full         # Full rewrite of sheet")
        print("")
        print("Source: FINRA Reg SHO Daily Short Sale Volume (CNMS)")
        print("URL: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt")
        print("No API key required.")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "backfill":
        start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
        run_backfill(start_year)
    elif cmd == "daily":
        run_daily()
    elif cmd == "calculate":
        run_calculate()
    elif cmd == "push":
        full_flag = len(sys.argv) > 2 and sys.argv[2].lower() == "full"
        run_push(full=full_flag)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)