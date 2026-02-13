"""
Append new tickers' historical DM data to all 3 partition sheets.
Only pushes data for specified tickers - does not touch existing rows.

Usage: python push_new_tickers.py
"""
import os
import time
import logging
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

NEW_TICKERS = ["INFY", "WIT", "HDB", "SAP"]

PARTITIONS = [
    {
        "key": "2024_2026",
        "spreadsheet": "dm-history 2024-current",
        "tab": "DM_2024_2026",
        "start": "2024-01-01",
        "end": "2026-12-31",
    },
    {
        "key": "2020_2023",
        "spreadsheet": "dm-history-2020-2023",
        "tab": "DM_2020_2023",
        "start": "2020-01-01",
        "end": "2023-12-31",
    },
    {
        "key": "2016_2019",
        "spreadsheet": "dm-history-2016-2019",
        "tab": "DM_2016_2019",
        "start": "2016-01-01",
        "end": "2019-12-31",
    },
]

HISTORY_FIELDS = [
    'date', 'ticker', 'close', 'volume', 'return_20d',
    'etf_return_20d', 'spy_return_20d', 'vol_20d_avg',
    'vol_ratio', 'rel_str_etf', 'rel_str_spy', 'volume_z',
    'dm_raw', 'dm_smoothed', 'etf', 'etf_dm', 'lead', 'phase'
]


def get_sheets_client():
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    return gspread.authorize(creds)


def fetch_ticker_dm(supabase, ticker, start_date, end_date):
    """Fetch DM history for a single ticker in a date range."""
    all_rows = []
    offset = 0
    page_size = 1000
    select_fields = ",".join(HISTORY_FIELDS)

    while True:
        result = supabase.table("dm_history") \
            .select(select_fields) \
            .eq("ticker", ticker) \
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

    return all_rows


def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    client = get_sheets_client()

    logger.info("=" * 60)
    logger.info("PUSH NEW TICKERS TO ALL PARTITIONS")
    logger.info(f"Tickers: {', '.join(NEW_TICKERS)}")
    logger.info("=" * 60)

    for partition in PARTITIONS:
        key = partition["key"]
        spreadsheet_name = partition["spreadsheet"]
        tab_name = partition["tab"]
        start_date = partition["start"]
        end_date = partition["end"]

        logger.info("")
        logger.info("-" * 60)
        logger.info(f"PARTITION: {key} ({start_date} to {end_date})")
        logger.info(f"  Spreadsheet: {spreadsheet_name}")
        logger.info("-" * 60)

        # Fetch all new ticker data for this partition
        all_rows = []
        for ticker in NEW_TICKERS:
            rows = fetch_ticker_dm(supabase, ticker, start_date, end_date)
            logger.info(f"  {ticker}: {len(rows)} rows from Supabase")
            all_rows.extend(rows)

        if not all_rows:
            logger.info(f"  No data to append. Skipping.")
            continue

        # Convert to sheet format with apostrophe date prefix
        sheet_rows = []
        for r in all_rows:
            row = [r.get(f, '') for f in HISTORY_FIELDS]
            row[0] = "'" + str(row[0])  # Force date as text
            sheet_rows.append(row)

        logger.info(f"  Total rows to append: {len(sheet_rows)}")

        # Open spreadsheet and find last row
        try:
            spreadsheet = client.open(spreadsheet_name)
        except gspread.SpreadsheetNotFound:
            logger.error(f"  Spreadsheet '{spreadsheet_name}' not found. Skipping.")
            continue

        sheet = spreadsheet.worksheet(tab_name)
        last_row = sheet.row_count
        current_data_rows = len(sheet.col_values(1))  # includes header

        # Expand sheet if needed
        rows_needed = current_data_rows + len(sheet_rows)
        if rows_needed > last_row:
            new_size = rows_needed + 5000
            sheet.resize(rows=new_size)
            logger.info(f"  Expanded sheet to {new_size} rows")

        # Append in batches
        BATCH_SIZE = 5000
        start_row = current_data_rows + 1
        total_written = 0

        for i in range(0, len(sheet_rows), BATCH_SIZE):
            batch = sheet_rows[i:i + BATCH_SIZE]
            write_row = start_row + i
            try:
                sheet.update(
                    range_name=f'A{write_row}',
                    values=batch,
                    value_input_option='USER_ENTERED'
                )
                total_written += len(batch)
                logger.info(f"  Written {total_written}/{len(sheet_rows)} rows")
            except Exception as e:
                logger.error(f"  Write error at row {write_row}: {e}")
            time.sleep(1)

        logger.info(f"  ✓ Partition {key} done: {total_written} rows appended")

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL PARTITIONS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()