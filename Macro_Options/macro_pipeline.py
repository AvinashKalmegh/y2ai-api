"""
Macro History Pipeline
======================
Fetches VIX, Treasury yields, Fed Funds Rate, and Credit Spreads from FRED API.
Calculates VIX_20D_Avg, VIX_Z_Score, Yield_Spread, Credit_Z_Score.
Stores in Supabase macro_history table and pushes to Google Sheets.

Usage:
  python macro_pipeline.py backfill     # Full backfill 2016-present
  python macro_pipeline.py daily        # Update with latest data
  python macro_pipeline.py push         # Push Supabase data to Sheets

FRED API key: Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
Set in .env as FRED_API_KEY
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
FRED_API_KEY = os.getenv('FRED_API_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FRED series IDs
FRED_SERIES = {
    'vix_close': 'VIXCLS',
    'treasury_10y': 'DGS10',
    'treasury_2y': 'DGS2',
    'fed_funds_rate': 'DFF',
    'hy_spread': 'BAMLH0A0HYM2',      # ICE BofA US High Yield OAS
    'ig_spread': 'BAMLC0A0CM',         # ICE BofA US Corporate OAS
}

# Google Sheets config
SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "Macro_History"

MACRO_HEADERS = [
    "Date", "VIX_Close", "VIX_20D_Avg", "VIX_Z_Score",
    "Treasury_10Y", "Treasury_2Y", "Yield_Spread", "Fed_Funds_Rate",
    "HY_Spread", "IG_Spread", "Credit_Z_Score"
]


# ============================================================
# FRED API
# ============================================================
def fetch_fred_series(series_id, start_date, end_date):
    """Fetch a single FRED series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    resp = requests.get(url, params=params)
    data = resp.json()

    if "observations" not in data:
        logger.error(f"FRED error for {series_id}: {data.get('error_message', 'Unknown error')}")
        return pd.Series(dtype=float)

    observations = data["observations"]
    dates = []
    values = []

    for obs in observations:
        date = obs["date"]
        val = obs["value"]
        if val == ".":  # FRED uses "." for missing
            values.append(np.nan)
        else:
            values.append(float(val))
        dates.append(date)

    series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    return series


def fetch_all_fred_data(start_date, end_date):
    """Fetch all FRED series and merge into a single DataFrame."""
    logger.info(f"Fetching FRED data: {start_date} to {end_date}")

    all_series = {}
    for col_name, series_id in FRED_SERIES.items():
        logger.info(f"  Fetching {series_id} ({col_name})...")
        series = fetch_fred_series(series_id, start_date, end_date)
        all_series[col_name] = series
        logger.info(f"    {len(series)} observations")
        time.sleep(0.5)  # Be nice to FRED

    # Merge on date index
    df = pd.DataFrame(all_series)
    df.index.name = 'date'

    # Forward-fill gaps (holidays, weekends already excluded by FRED)
    df = df.ffill()

    # Drop rows where ALL values are NaN (before first data point)
    df = df.dropna(how='all')

    logger.info(f"  Merged: {len(df)} rows, {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    return df


# ============================================================
# CALCULATIONS
# ============================================================
def calculate_derived_fields(df):
    """Calculate VIX_20D_Avg, VIX_Z_Score, Yield_Spread."""
    # VIX 20-day moving average
    df['vix_20d_avg'] = df['vix_close'].rolling(window=20, min_periods=10).mean()

    # VIX Z-Score: (VIX - 20D_Avg) / 20D_StdDev
    vix_std = df['vix_close'].rolling(window=20, min_periods=10).std()
    df['vix_z_score'] = np.where(
        vix_std > 0,
        (df['vix_close'] - df['vix_20d_avg']) / vix_std,
        0
    )

    # Yield Spread: 10Y - 2Y (negative = inverted)
    df['yield_spread'] = df['treasury_10y'] - df['treasury_2y']

    # Credit Z-Score: based on HY spread (widening = stress)
    hy_avg = df['hy_spread'].rolling(window=20, min_periods=10).mean()
    hy_std = df['hy_spread'].rolling(window=20, min_periods=10).std()
    df['credit_z_score'] = np.where(
        hy_std > 0,
        (df['hy_spread'] - hy_avg) / hy_std,
        0
    )

    # Round everything
    for col in ['vix_close', 'vix_20d_avg', 'vix_z_score',
                'treasury_10y', 'treasury_2y', 'yield_spread', 'fed_funds_rate',
                'hy_spread', 'ig_spread', 'credit_z_score']:
        if col in df.columns:
            df[col] = df[col].round(4)

    return df


# ============================================================
# SUPABASE
# ============================================================
def upload_to_supabase(df):
    """Upload macro_history DataFrame to Supabase."""
    records = []
    for date, row in df.iterrows():
        record = {
            'date': date.strftime('%Y-%m-%d'),
            'vix_close': None if pd.isna(row.get('vix_close')) else float(row['vix_close']),
            'vix_20d_avg': None if pd.isna(row.get('vix_20d_avg')) else float(row['vix_20d_avg']),
            'vix_z_score': None if pd.isna(row.get('vix_z_score')) else float(row['vix_z_score']),
            'treasury_10y': None if pd.isna(row.get('treasury_10y')) else float(row['treasury_10y']),
            'treasury_2y': None if pd.isna(row.get('treasury_2y')) else float(row['treasury_2y']),
            'yield_spread': None if pd.isna(row.get('yield_spread')) else float(row['yield_spread']),
            'fed_funds_rate': None if pd.isna(row.get('fed_funds_rate')) else float(row['fed_funds_rate']),
            'hy_spread': None if pd.isna(row.get('hy_spread')) else float(row['hy_spread']),
            'ig_spread': None if pd.isna(row.get('ig_spread')) else float(row['ig_spread']),
            'credit_z_score': None if pd.isna(row.get('credit_z_score')) else float(row['credit_z_score']),
        }
        records.append(record)

    batch_size = 500
    uploaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("macro_history").upsert(
                batch, on_conflict="date"
            ).execute()
            uploaded += len(batch)
            logger.info(f"  Uploaded {uploaded}/{len(records)} rows")
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


def load_from_supabase():
    """Load all macro_history from Supabase."""
    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        result = supabase.table("macro_history") \
            .select("*") \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets():
    """Push macro_history to Google Sheets."""
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading macro_history from Supabase...")
    df = load_from_supabase()

    if df.empty:
        logger.info("No data to push.")
        return

    logger.info(f"  {len(df)} rows loaded")

    # Build sheet rows
    sheet_rows = []
    for date, row in df.iterrows():
        sheet_rows.append([
            "'" + date.strftime('%Y-%m-%d'),  # Force text date
            row.get('vix_close', ''),
            row.get('vix_20d_avg', ''),
            row.get('vix_z_score', ''),
            row.get('treasury_10y', ''),
            row.get('treasury_2y', ''),
            row.get('yield_spread', ''),
            row.get('fed_funds_rate', ''),
            row.get('hy_spread', ''),
            row.get('ig_spread', ''),
            row.get('credit_z_score', ''),
        ])

    # Replace NaN with empty string
    for row in sheet_rows:
        for j in range(len(row)):
            if row[j] is None or (isinstance(row[j], float) and np.isnan(row[j])):
                row[j] = ''

    # Connect to Sheets
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(SHEETS_SPREADSHEET)
    except gspread.SpreadsheetNotFound:
        logger.error(f"Spreadsheet '{SHEETS_SPREADSHEET}' not found.")
        logger.error(f"Create it and share with your service account first.")
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
        cols=len(MACRO_HEADERS)
    )

    # Write header
    sheet.update(range_name='A1', values=[MACRO_HEADERS], value_input_option='USER_ENTERED')
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
    """Full backfill from 2016 to present."""
    logger.info("=" * 60)
    logger.info("MACRO HISTORY - FULL BACKFILL")
    logger.info("=" * 60)

    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not set in .env")
        logger.error("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    start_date = "2016-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch from FRED
    df = fetch_all_fred_data(start_date, end_date)

    # Calculate derived fields
    df = calculate_derived_fields(df)

    # Drop rows before calculations are valid (need 20 days for moving avg)
    df = df.iloc[20:]

    logger.info(f"\nFinal dataset: {len(df)} rows")
    logger.info(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Sample output
    logger.info("\nSample (latest 5 rows):")
    for date, row in df.tail().iterrows():
        logger.info(f"  {date.strftime('%Y-%m-%d')}: VIX={row['vix_close']:.2f}, "
                     f"VIX_Z={row['vix_z_score']:.2f}, "
                     f"10Y={row['treasury_10y']:.2f}, "
                     f"Spread={row['yield_spread']:.2f}, "
                     f"FFR={row['fed_funds_rate']:.2f}, "
                     f"HY={row.get('hy_spread', 0):.2f}, "
                     f"Credit_Z={row.get('credit_z_score', 0):.2f}")

    # Upload to Supabase
    logger.info("\nUploading to Supabase...")
    uploaded = upload_to_supabase(df)
    logger.info(f"Uploaded {uploaded} rows to macro_history")

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info("Next: python macro_pipeline.py push  (to sync Google Sheets)")


def run_daily():
    """Fetch last 30 days to catch any gaps, recalculate, update."""
    logger.info("=" * 60)
    logger.info("MACRO HISTORY - DAILY UPDATE")
    logger.info("=" * 60)

    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not set in .env")
        return

    # Fetch last 60 days (need 20-day lookback for calculations)
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    df = fetch_all_fred_data(start_date, end_date)

    # For daily update, we need older data for the 20-day window
    # Load existing from Supabase to compute rolling averages correctly
    logger.info("Loading existing data for rolling calculations...")
    existing = load_from_supabase()

    if not existing.empty:
        # Merge: use existing for older dates, new data for recent
        combined = existing.copy()
        # Drop columns we'll recalculate
        for col in ['vix_20d_avg', 'vix_z_score', 'yield_spread', 'credit_z_score', 'created_at', 'updated_at']:
            if col in combined.columns:
                combined = combined.drop(columns=[col])

        # Update with fresh FRED data
        for date, row in df.iterrows():
            combined.loc[date] = row

        combined = combined.sort_index()
        df = calculate_derived_fields(combined)

        # Only upload the recent portion
        cutoff = pd.Timestamp(start_date)
        df = df[df.index >= cutoff]
    else:
        df = calculate_derived_fields(df)

    logger.info(f"Updating {len(df)} rows")
    uploaded = upload_to_supabase(df)
    logger.info(f"Uploaded {uploaded} rows")

    logger.info("DAILY UPDATE COMPLETE")


def run_push():
    """Push Supabase data to Google Sheets."""
    logger.info("=" * 60)
    logger.info("MACRO HISTORY - PUSH TO SHEETS")
    logger.info("=" * 60)
    push_to_sheets()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python macro_pipeline.py backfill    # Full backfill 2016-present")
        print("  python macro_pipeline.py daily       # Update with latest data")
        print("  python macro_pipeline.py push        # Push to Google Sheets")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "backfill":
        run_backfill()
    elif cmd == "daily":
        run_daily()
    elif cmd == "push":
        run_push()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)