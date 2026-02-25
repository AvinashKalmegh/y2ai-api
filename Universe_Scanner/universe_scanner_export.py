"""
universe_scanner_export.py
FlowOS Universe Scanner - Export crossings data to CSV for Google Sheets

Joins universe_crossings with universe_tickers (market_cap, sector)
and scanner_universe (curated sector) for full pivot-ready dataset.

Usage:
  python universe_scanner_export.py                # Export to CSV
  python universe_scanner_export.py --sheets       # Also push to Google Sheets (requires gspread)

Output: Universe_Scanner/reports/universe_crossings_export_{date}.csv
"""

import os
import sys
import csv
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

PREFERRED_28 = [
    "CEG", "CRWD", "TSM", "APP", "VRT", "MU", "NVDA", "CCJ",
    "DNN", "PLTR", "TSLA", "TTD", "MSTR", "UEC", "LEU", "HAL",
    "WDC", "ENPH", "UUUU", "PDD", "NCLH", "FCX", "RCL", "LVS",
    "PSKY", "MRNA", "WYNN", "SMR"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def paginated_fetch(supabase, table, select, filters=None, order=None, page_size=1000):
    """Fetch all rows from a table, paginating past the 1000 limit."""
    all_rows = []
    offset = 0
    while True:
        query = supabase.table(table).select(select)
        if filters:
            for method, args in filters:
                query = getattr(query, method)(*args)
        if order:
            query = query.order(order)
        query = query.range(offset, offset + page_size - 1)
        result = query.execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
    return all_rows


def export_csv():
    """Export universe_crossings joined with ticker metadata to CSV."""
    supabase = get_supabase()
    
    # ---- Load crossings (all 10,512) ----
    logger.info("Loading crossings...")
    crossings = paginated_fetch(
        supabase, "universe_crossings",
        "id,ticker,cross_date,entry_dm,entry_price,max_price_90d,max_return_30d,max_return_60d,max_return_90d,is_hit,evaluated"
    )
    logger.info(f"  {len(crossings)} crossings loaded")
    
    # ---- Load ticker metadata ----
    logger.info("Loading ticker metadata...")
    tickers_raw = paginated_fetch(
        supabase, "universe_tickers",
        "ticker,name,exchange,market_cap,avg_dollar_volume,sic_code,sic_description"
    )
    ticker_meta = {t["ticker"]: t for t in tickers_raw}
    logger.info(f"  {len(ticker_meta)} tickers from universe_tickers")
    
    # ---- Load curated sectors ----
    logger.info("Loading curated sectors...")
    scanner = supabase.table("scanner_universe").select("ticker,sector").execute()
    curated_sectors = {}
    if scanner.data:
        for r in scanner.data:
            if r.get("ticker") and r.get("sector"):
                curated_sectors[r["ticker"]] = r["sector"]
    logger.info(f"  {len(curated_sectors)} curated sectors from scanner_universe")
    
    # ---- Build joined rows ----
    logger.info("Joining data...")
    rows = []
    for c in crossings:
        ticker = c["ticker"]
        meta = ticker_meta.get(ticker, {})
        
        # Sector: curated > sic_description > empty
        sector = curated_sectors.get(ticker, "")
        if not sector:
            sector = meta.get("sic_description", "") or ""
        
        row = {
            "ticker": ticker,
            "cross_date": c["cross_date"],
            "entry_dm": c.get("entry_dm", ""),
            "entry_price": c.get("entry_price", ""),
            "max_price_90d": c.get("max_price_90d", ""),
            "max_return_30d": c.get("max_return_30d", ""),
            "max_return_60d": c.get("max_return_60d", ""),
            "max_return_90d": c.get("max_return_90d", ""),
            "is_hit": c.get("is_hit", ""),
            "evaluated": c.get("evaluated", False),
            "name": meta.get("name", ""),
            "exchange": meta.get("exchange", ""),
            "market_cap": meta.get("market_cap", ""),
            "avg_dollar_volume": meta.get("avg_dollar_volume", ""),
            "sector": sector,
            "sic_code": meta.get("sic_code", ""),
            "in_preferred_28": ticker in PREFERRED_28,
        }
        rows.append(row)
    
    # ---- Write CSV ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"universe_crossings_export_{today}.csv"
    filepath = os.path.join(reports_dir, filename)
    
    fieldnames = [
        "ticker", "cross_date", "entry_dm", "entry_price",
        "max_price_90d", "max_return_30d", "max_return_60d", "max_return_90d",
        "is_hit", "evaluated",
        "name", "exchange", "market_cap", "avg_dollar_volume",
        "sector", "sic_code", "in_preferred_28"
    ]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"  Exported {len(rows)} rows to {filepath}")
    
    # ---- Quick stats ----
    evaluated_rows = [r for r in rows if r["evaluated"]]
    hit_rows = [r for r in rows if r["is_hit"]]
    unique_tickers = len(set(r["ticker"] for r in rows))
    
    logger.info("")
    logger.info("EXPORT SUMMARY")
    logger.info(f"  Total crossings: {len(rows)}")
    logger.info(f"  Unique tickers: {unique_tickers}")
    logger.info(f"  Evaluated: {len(evaluated_rows)}")
    logger.info(f"  Hits: {len(hit_rows)}")
    logger.info(f"  With market cap: {sum(1 for r in rows if r['market_cap'])}")
    logger.info(f"  With sector: {sum(1 for r in rows if r['sector'])}")
    logger.info(f"  Preferred 28 crossings: {sum(1 for r in rows if r['in_preferred_28'])}")
    logger.info("")
    logger.info(f"  File: {filepath}")
    logger.info(f"  Import to Google Sheets: File > Import > Upload > {filename}")
    
    return filepath


def push_to_sheets(csv_path):
    """Push crossings data to Google Sheets using project's established pattern."""
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    
    GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
    SPREADSHEET_NAME = 'Universe_Scanner_Import'
    TAB_NAME = 'Crossings'
    
    logger.info(f"  Pushing to Google Sheets: {SPREADSHEET_NAME} / {TAB_NAME}")
    
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SHEETS_CREDS_FILE, scope
    )
    client = gspread.authorize(creds)
    
    # Open spreadsheet by name
    spreadsheet = client.open(SPREADSHEET_NAME)
    
    # Get or create the tab
    try:
        sheet = spreadsheet.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=TAB_NAME, rows=12000, cols=20)
    
    # Read CSV into row arrays
    import csv as csv_mod
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.reader(f)
        all_data = list(reader)
    
    logger.info(f"  {len(all_data)} rows (including header)")
    
    # Clear and push — gspread handles large updates in chunks
    sheet.clear()
    
    # Push in batches of 5000 to avoid API limits
    BATCH = 5000
    for i in range(0, len(all_data), BATCH):
        batch = all_data[i:i + BATCH]
        start_row = i + 1
        sheet.update(range_name=f'A{start_row}', values=batch, value_input_option='USER_ENTERED')
        logger.info(f"  Pushed rows {start_row}-{start_row + len(batch) - 1}")
    
    # Bold header
    sheet.format('1:1', {'textFormat': {'bold': True}})
    
    logger.info(f"  Done: {len(all_data) - 1} data rows pushed to {SPREADSHEET_NAME} / {TAB_NAME}")
    return spreadsheet.url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export universe crossings to CSV/Sheets")
    parser.add_argument("--sheets", action="store_true", help="Also push to Google Sheets")
    args = parser.parse_args()
    
    csv_path = export_csv()
    
    if args.sheets:
        push_to_sheets(csv_path)