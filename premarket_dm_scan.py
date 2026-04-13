"""
premarket_dm_scan.py
====================
Pre-Market DM Scan -- Morning Intelligence Layer
Run time: 7:30 AM ET on trading days (Mon-Fri)

Scans pre-market prices for V4 universe + cluster key names.
Compares against previous close from Supabase dm_latest.
Writes gap analysis to Google Sheets (DM_PreMarket tab).

No DM calculation -- price and gap only.
Appends each scan (does not overwrite) to preserve history.
"""

import os
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import requests
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# CONFIGURATION
# ============================================================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")

STAGING_SHEET_ID = '1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA'
TAB_NAME         = 'DM_PreMarket'

# Cluster key names (always included)
CLUSTER_KEYS = ['NVDA', 'MRVL', 'VRT', 'EQIX', 'DELL', 'CRWD']

# Gap direction thresholds
GAP_THRESHOLD = 0.5  # +/- 0.5% = FLAT

POLYGON_SNAPSHOT_URL = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
POLYGON_BATCH_SIZE   = 150

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# SUPABASE -- load prev close + DM from dm_latest
# ============================================================

def load_dm_latest() -> dict:
    """Return {ticker: {close, dm}} from Supabase dm_latest."""
    from supabase import create_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY not set.")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    result = {}
    offset, page = 0, 5000
    while True:
        r = (sb.table('dm_latest')
             .select('ticker,close,dm_smoothed')
             .range(offset, offset + page - 1)
             .execute())
        for row in r.data:
            tk = str(row['ticker']).strip().upper()
            result[tk] = {
                'close': float(row['close']) if row.get('close') is not None else None,
                'dm':    float(row['dm_smoothed']) if row.get('dm_smoothed') is not None else None,
            }
        if len(r.data) < page:
            break
        offset += page

    logger.info(f"[DM Latest] Loaded {len(result)} tickers")
    return result

# ============================================================
# V4 UNIVERSE LOADER
# ============================================================

def load_v4_tickers() -> list:
    """Load V4 universe tickers from CSV."""
    import csv
    path = os.path.join(os.path.dirname(__file__), 'files_3', 'v4_universe_candidates.csv')
    tickers = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tk = row.get('ticker', '').strip().upper()
            if tk:
                tickers.append(tk)
    logger.info(f"[V4 Universe] Loaded {len(tickers)} tickers from CSV")
    return tickers

# ============================================================
# POLYGON -- fetch pre-market snapshots
# ============================================================

def fetch_snapshots(tickers: list) -> dict:
    """Fetch pre-market snapshots from Polygon. Returns {ticker: price}."""
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY not set.")

    import math
    result = {}
    n_batches = math.ceil(len(tickers) / POLYGON_BATCH_SIZE)

    logger.info(f"[Polygon] Fetching snapshots for {len(tickers)} tickers in {n_batches} batch(es)...")

    for batch_idx in range(n_batches):
        batch = tickers[batch_idx * POLYGON_BATCH_SIZE : (batch_idx + 1) * POLYGON_BATCH_SIZE]
        joined = ",".join(batch)

        try:
            resp = requests.get(
                POLYGON_SNAPSHOT_URL,
                params={"tickers": joined, "apiKey": POLYGON_API_KEY},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            snaps = data.get("tickers") or data.get("results") or []

            for snap in snaps:
                tk = snap.get("ticker", "").upper()
                # Use last trade price (most recent pre-market trade)
                last_trade = snap.get("lastTrade", {})
                day = snap.get("day", {}) or {}
                price = last_trade.get("p") or day.get("c")
                if price is not None:
                    result[tk] = float(price)

        except Exception as e:
            logger.warning(f"[Polygon] Batch {batch_idx + 1} error: {e}")

        if batch_idx < n_batches - 1:
            import time
            time.sleep(0.25)

    logger.info(f"[Polygon] Got prices for {len(result)} of {len(tickers)} tickers")
    return result

# ============================================================
# GOOGLE SHEETS -- append to DM_PreMarket
# ============================================================

def get_sheet():
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials.json')
    creds = Credentials.from_service_account_file(creds_file, scopes=scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(STAGING_SHEET_ID)

    try:
        ws = spreadsheet.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=TAB_NAME, rows=1000, cols=7)
        ws.update(values=[['Scan_Time', 'Ticker', 'PreMarket_Price',
                              'Prev_Close', 'Gap_Pct', 'Direction', 'DM_EOD']], range_name='A1:G1')
        logger.info(f"  Created new tab: {TAB_NAME}")

    return ws


def append_rows(ws, rows: list):
    """Append rows to the sheet (does not overwrite)."""
    if not rows:
        return
    # Find the next empty row
    existing = ws.get_all_values()
    next_row = len(existing) + 1

    # Add header if sheet is empty
    if next_row == 1:
        ws.update(values=[['Scan_Time', 'Ticker', 'PreMarket_Price',
                              'Prev_Close', 'Gap_Pct', 'Direction', 'DM_EOD']], range_name='A1:G1')
        next_row = 2

    end_row = next_row + len(rows) - 1
    ws.update(values=rows, range_name=f'A{next_row}:G{end_row}', value_input_option='USER_ENTERED')
    logger.info(f"  Appended {len(rows)} rows to {TAB_NAME} (rows {next_row}-{end_row})")

# ============================================================
# MAIN
# ============================================================

def run_scan():
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("  PRE-MARKET DM SCAN")
    logger.info(f"  {scan_time}")
    logger.info("=" * 60)

    # 1. Load V4 universe + cluster keys
    v4_tickers = load_v4_tickers()
    all_tickers = sorted(set(v4_tickers + CLUSTER_KEYS))
    logger.info(f"  Total tickers to scan: {len(all_tickers)}")

    # 2. Load prev close + DM from Supabase
    dm_data = load_dm_latest()

    # 3. Fetch pre-market prices from Polygon
    snapshots = fetch_snapshots(all_tickers)

    # 4. Build output rows
    rows = []
    gaps_up = 0
    gaps_down = 0

    for ticker in all_tickers:
        prev = dm_data.get(ticker, {})
        prev_close = prev.get('close')
        dm_eod = prev.get('dm')
        premarket_price = snapshots.get(ticker)

        if prev_close and premarket_price:
            gap_pct = round((premarket_price - prev_close) / prev_close * 100, 2)
        else:
            gap_pct = 0.0
            premarket_price = premarket_price or prev_close or 0

        if gap_pct > GAP_THRESHOLD:
            direction = 'UP'
            gaps_up += 1
        elif gap_pct < -GAP_THRESHOLD:
            direction = 'DOWN'
            gaps_down += 1
        else:
            direction = 'FLAT'

        rows.append([
            scan_time,
            ticker,
            round(premarket_price, 2) if premarket_price else '',
            round(prev_close, 2) if prev_close else '',
            gap_pct,
            direction,
            round(dm_eod, 1) if dm_eod is not None else '',
        ])

    # 5. Print summary
    flats = len(rows) - gaps_up - gaps_down
    logger.info("")
    logger.info(f"  GAPS: {gaps_up} UP | {gaps_down} DOWN | {flats} FLAT")

    # Show significant gaps (>2%)
    significant = [r for r in rows if abs(r[4]) >= 2.0]
    if significant:
        significant.sort(key=lambda r: r[4])
        logger.info(f"  SIGNIFICANT GAPS (>= 2%):")
        for r in significant:
            logger.info(f"    {r[1]:<8} {r[4]:+.2f}%  DM={r[6]}  "
                        f"Pre=${r[2]}  PrevClose=${r[3]}")

    # 6. Push to Google Sheets
    ws = get_sheet()
    append_rows(ws, rows)

    logger.info("")
    logger.info("  Done.")
    logger.info("=" * 60)


if __name__ == '__main__':
    run_scan()
