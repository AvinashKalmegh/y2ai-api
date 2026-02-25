"""
dm_push_to_sheets.py
Push DM data from Supabase to Google Sheets with year-range partitioning.

Spreadsheet layout (10M cell limit per spreadsheet):
  - dm-history:           DM_Latest tab + DM_2024_2026 tab
  - dm-history-2020-2023: DM_2020_2023 tab
  - dm-history-2016-2019: DM_2016_2019 tab

All columns from dm_history are included (18 columns).
~130K rows/year x 18 cols = 2.34M cells/year, max ~4 years per spreadsheet.

Usage:
  python dm_push_to_sheets.py latest              # Push dm_latest only (~5 sec)
  python dm_push_to_sheets.py history              # Push all history partitions
  python dm_push_to_sheets.py history 2024_2026    # Push single partition
  python dm_push_to_sheets.py all                  # Push latest + all history
  python dm_push_to_sheets.py daily                # Push latest + append new rows

Location: y2ai/dm_push_to_sheets.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from supabase import create_client

# ============================================================
# CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'

# Spreadsheet -> partition mapping
PARTITIONS = [
    {
        "spreadsheet": "copy-dm-history-2016-2019",
        "tab": "DM_2016_2019",
        "start": "2016-01-01",
        "end": "2019-12-31",
        "key": "2016_2019",
    },
    {
        "spreadsheet": "copy-dm-history-2020-2023",
        "tab": "DM_2020_2023",
        "start": "2020-01-01",
        "end": "2023-12-31",
        "key": "2020_2023",
    },
    {
        "spreadsheet": "copy-dm-history 2024-current",
        "tab": "DM_2024_2026",
        "start": "2024-01-01",
        "end": "2026-12-31",
        "key": "2024_2026",
    },
]

# DM_Latest goes in main spreadsheet
LATEST_SPREADSHEET = "copy-dm-history 2024-current"
TAB_LATEST = "DM_Latest"

LATEST_HEADERS = [
    'Date', 'Ticker', 'DM', 'Phase', 'Lead', 'ETF_Flow',
    'DM_7d_Ago', 'DM_Change', 'Volume_Z', 'RelStr_ETF', 'RelStr_SPY', 'Close', 'ETF'
]

# All dm_history columns
HISTORY_HEADERS = [
    'Date', 'Ticker', 'Close', 'Volume', 'Return_20d', 'ETF_Return_20d',
    'SPY_Return_20d', 'Vol_20d_Avg', 'Vol_Ratio', 'RelStr_ETF', 'RelStr_SPY',
    'Volume_Z', 'DM_Raw', 'DM', 'ETF', 'ETF_DM', 'Lead', 'Phase'
]

# Column mapping from Supabase field names
HISTORY_FIELDS = [
    'date', 'ticker', 'close', 'volume', 'return_20d', 'etf_return_20d',
    'spy_return_20d', 'vol_20d_avg', 'vol_ratio', 'rel_str_etf', 'rel_str_spy',
    'volume_z', 'dm_raw', 'dm_smoothed', 'etf', 'etf_dm', 'lead', 'phase'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CLIENTS
# ============================================================

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_sheets_client():
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SHEETS_CREDS_FILE, scope
    )
    return gspread.authorize(creds)


# ============================================================
# PUSH DM_LATEST
# ============================================================

def push_latest():
    """Push dm_latest from Supabase to GS DM_Latest tab."""
    logger.info("=" * 60)
    logger.info("PUSHING DM_LATEST")
    logger.info("=" * 60)

    supabase = get_supabase()

    all_rows = []
    offset = 0
    page_size = 1000
    while True:
        result = supabase.table("dm_latest") \
            .select("date,ticker,dm_smoothed,phase,lead,etf_dm,dm_7d_ago,dm_change,volume_z,rel_str_etf,rel_str_spy,close,etf") \
            .order("dm_smoothed", desc=True) \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    logger.info(f"Fetched {len(all_rows)} rows from dm_latest")

    sheet_rows = []
    for r in all_rows:
        sheet_rows.append([
            "'" + str(r.get('date', '')),  # Force text to prevent date auto-parse
            r.get('ticker', ''),
            r.get('dm_smoothed', ''),
            r.get('phase', ''),
            r.get('lead', ''),
            r.get('etf_dm', ''),
            r.get('dm_7d_ago', ''),
            r.get('dm_change', ''),
            r.get('volume_z', ''),
            r.get('rel_str_etf', ''),
            r.get('rel_str_spy', ''),
            r.get('close', ''),
            r.get('etf', ''),
        ])

    client = get_sheets_client()
    spreadsheet = client.open(LATEST_SPREADSHEET)

    try:
        sheet = spreadsheet.worksheet(TAB_LATEST)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=TAB_LATEST, rows=1000, cols=len(LATEST_HEADERS))

    sheet.clear()
    all_data = [LATEST_HEADERS] + sheet_rows
    sheet.update(range_name='A1', values=all_data, value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})

    logger.info(f"DM_Latest pushed: {len(sheet_rows)} rows")


# ============================================================
# PUSH HISTORY PARTITION
# ============================================================

def fetch_history_partition(start_date, end_date):
    """Fetch dm_history rows for a date range from Supabase."""
    supabase = get_supabase()
    select_fields = ",".join(HISTORY_FIELDS)

    all_rows = []
    offset = 0
    page_size = 1000

    while True:
        result = supabase.table("dm_history") \
            .select(select_fields) \
            .gte("date", start_date) \
            .lte("date", end_date) \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()

        if not result.data:
            break

        all_rows.extend(result.data)

        if len(result.data) < page_size:
            break

        offset += page_size

        if offset % 50000 == 0:
            logger.info(f"  Fetched {offset} rows...")

    return all_rows


def rows_to_sheet_data(rows):
    """Convert Supabase rows to sheet row arrays."""
    sheet_rows = []
    for r in rows:
        row = [r.get(f, '') for f in HISTORY_FIELDS]
        row[0] = "'" + str(row[0])  # Force date as text to prevent auto-parse
        sheet_rows.append(row)
    return sheet_rows


def push_partition(partition):
    """Push a single history partition to its spreadsheet tab."""
    key = partition["key"]
    spreadsheet_name = partition["spreadsheet"]
    tab_name = partition["tab"]
    start_date = partition["start"]
    end_date = partition["end"]

    logger.info("=" * 60)
    logger.info(f"PUSHING PARTITION: {key} ({start_date} to {end_date})")
    logger.info(f"  Spreadsheet: {spreadsheet_name}")
    logger.info(f"  Tab: {tab_name}")
    logger.info("=" * 60)

    # Fetch from Supabase
    logger.info("Fetching from Supabase...")
    rows = fetch_history_partition(start_date, end_date)
    logger.info(f"Fetched {len(rows)} rows")

    if not rows:
        logger.info("No rows for this partition. Skipping.")
        return

    sheet_rows = rows_to_sheet_data(rows)
    cells = len(sheet_rows) * len(HISTORY_HEADERS)
    logger.info(f"Cell count: {cells:,} ({len(sheet_rows)} rows x {len(HISTORY_HEADERS)} cols)")

    # Open spreadsheet
    client = get_sheets_client()
    try:
        spreadsheet = client.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        logger.error(f"Spreadsheet '{spreadsheet_name}' not found.")
        logger.error(f"Create it and share with service account first.")
        return

    # Get or create tab
    total_rows_needed = len(sheet_rows) + 1
    try:
        sheet = spreadsheet.worksheet(tab_name)
        sheet.clear()
        if sheet.row_count < total_rows_needed:
            sheet.resize(rows=total_rows_needed, cols=len(HISTORY_HEADERS))
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(
            title=tab_name,
            rows=total_rows_needed,
            cols=len(HISTORY_HEADERS)
        )

    # Write header
    sheet.update(range_name='A1', values=[HISTORY_HEADERS], value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})

    # Write data in batches
    BATCH_SIZE = 5000
    total_written = 0

    for i in range(0, len(sheet_rows), BATCH_SIZE):
        batch = sheet_rows[i:i + BATCH_SIZE]
        start_row = i + 2
        range_name = f'A{start_row}'

        try:
            sheet.update(range_name=range_name, values=batch, value_input_option='USER_ENTERED')
            total_written += len(batch)
            logger.info(f"  Written {total_written}/{len(sheet_rows)} rows")
        except Exception as e:
            logger.error(f"  Batch error at row {start_row}: {e}")
            RETRY_SIZE = 1000
            for j in range(0, len(batch), RETRY_SIZE):
                mini_batch = batch[j:j + RETRY_SIZE]
                retry_row = start_row + j
                try:
                    sheet.update(range_name=f'A{retry_row}', values=mini_batch, value_input_option='USER_ENTERED')
                    total_written += len(mini_batch)
                except Exception as e2:
                    logger.error(f"  Mini-batch error at row {retry_row}: {e2}")

        time.sleep(1)

    logger.info(f"Partition {key} complete: {total_written} rows written")


def push_all_history():
    """Push all history partitions."""
    for partition in PARTITIONS:
        push_partition(partition)


# ============================================================
# PUSH DAILY INCREMENT
# ============================================================

def push_daily():
    """Push dm_latest + append new rows to the most recent history partition."""
    logger.info("=" * 60)
    logger.info("DAILY PUSH TO GOOGLE SHEETS")
    logger.info("=" * 60)

    # Step 1: Push latest
    push_latest()

    # Step 2: Find the active partition (most recent)
    active = PARTITIONS[-1]  # 2024_2026
    spreadsheet_name = active["spreadsheet"]
    tab_name = active["tab"]

    client = get_sheets_client()
    try:
        spreadsheet = client.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        logger.error(f"Spreadsheet '{spreadsheet_name}' not found.")
        return

    try:
        history_sheet = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        logger.info(f"{tab_name} tab not found. Running full partition push.")
        push_partition(active)
        return

    # Get last date from GS
    all_dates = history_sheet.col_values(1)
    if len(all_dates) > 1:
        last_gs_date = all_dates[-1]
        logger.info(f"Last date in GS {tab_name}: {last_gs_date}")
    else:
        logger.info(f"{tab_name} is empty. Running full partition push.")
        push_partition(active)
        return

    # Fetch new rows from Supabase
    supabase = get_supabase()
    select_fields = ",".join(HISTORY_FIELDS)

    new_rows = []
    offset = 0
    page_size = 1000
    while True:
        result = supabase.table("dm_history") \
            .select(select_fields) \
            .gt("date", last_gs_date) \
            .lte("date", active["end"]) \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not result.data:
            break
        new_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not new_rows:
        logger.info("No new history rows to append.")
        return

    logger.info(f"New history rows to append: {len(new_rows)}")

    sheet_rows = rows_to_sheet_data(new_rows)

    BATCH_SIZE = 5000
    current_row = len(all_dates) + 1

    # Expand sheet if needed
    rows_needed = current_row + len(sheet_rows) - 1
    if rows_needed > history_sheet.row_count:
        new_size = rows_needed + 5000  # Add buffer for future appends
        history_sheet.resize(rows=new_size)
        logger.info(f"  Expanded sheet from {history_sheet.row_count} to {new_size} rows")

    for i in range(0, len(sheet_rows), BATCH_SIZE):
        batch = sheet_rows[i:i + BATCH_SIZE]
        start_row = current_row + i
        try:
            history_sheet.update(range_name=f'A{start_row}', values=batch, value_input_option='USER_ENTERED')
            logger.info(f"  Appended {len(batch)} rows (row {start_row})")
        except Exception as e:
            logger.error(f"  Append error at row {start_row}: {e}")
        time.sleep(1)

    logger.info(f"Daily push complete. Appended {len(new_rows)} history rows.")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dm_push_to_sheets.py latest              # Push dm_latest snapshot")
        print("  python dm_push_to_sheets.py history              # Push all history partitions")
        print("  python dm_push_to_sheets.py history 2016_2019    # Push single partition")
        print("  python dm_push_to_sheets.py history 2020_2023")
        print("  python dm_push_to_sheets.py history 2024_2026")
        print("  python dm_push_to_sheets.py all                  # Push latest + all history")
        print("  python dm_push_to_sheets.py daily                # Push latest + append new rows")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'latest':
        push_latest()

    elif command == 'history':
        if len(sys.argv) > 2:
            target_key = sys.argv[2]
            partition = next((p for p in PARTITIONS if p["key"] == target_key), None)
            if partition:
                push_partition(partition)
            else:
                print(f"Unknown partition: {target_key}")
                print(f"Available: {', '.join(p['key'] for p in PARTITIONS)}")
                sys.exit(1)
        else:
            push_all_history()

    elif command == 'all':
        push_latest()
        push_all_history()

    elif command == 'daily':
        push_daily()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)