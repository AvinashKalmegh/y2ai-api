"""
INTRADAY DM HISTORY BACKFILL
==============================
Computes daily DM scores (using EOD close prices) for the full scanner
universe (~580 tickers) from 2016 to current.

Same formula as the intraday scanner, but uses historical EOD closes
instead of live Polygon snapshots. One reading per day (market close).

Stores results in Supabase `dm_intraday_history` table and pushes
to a dedicated Google Sheet.

Usage:
    python intraday_dm_history.py                    # Full backfill 2016-current
    python intraday_dm_history.py --start 2024-01-01 # From specific date
    python intraday_dm_history.py --dry-run          # Compute only, no writes
    python intraday_dm_history.py --sheets-only      # Push existing Supabase data to Sheets

Environment Variables (.env):
    SUPABASE_URL, SUPABASE_KEY, POLYGON_API_KEY
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# DM formula weights (same as intraday scanner)
W_REL_STR_ETF = 0.50
W_REL_STR_SPY = 0.30
W_VOLUME_Z = 0.20

# Relative strength scaling
REL_STR_SCALE = 500

# EMA smoothing
EMA_PERIOD = 5

# Return period (20-day = 19 intervals)
RETURN_PERIOD = 19

# Supabase table
SUPA_TABLE = "dm_intraday_history"

# Google Sheets — open by key (spreadsheet ID)
SPREADSHEET_KEY = "15iT_ubT8xZWts6H4s5PsLY_AYP_txcQ7Uedx2OTR_tQ"

# Split into two tabs to stay under 10M cell limit
SHEETS_PARTITIONS = [
    {"tab": "DM_Intraday_2016_2019", "start": "2016-01-01", "end": "2019-12-31"},
    {"tab": "DM_Intraday_2020_Current", "start": "2020-01-01", "end": "2099-12-31"},
]

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Semiconductors": "SMH",
    "Software": "IGV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Biotechnology": "XBI",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Energy": "XLE",
    "Communication Services": "XLC",
    "Nuclear": "URA",
    "Uranium": "URA",
    "Nuclear/Uranium": "URA",
    "Clean Energy": "TAN",
    "Cybersecurity": "HACK",
    "Aerospace & Defense": "ITA",
    "Defense": "ITA",
    "Defense/Aerospace": "ITA",
    "Transportation": "IYT",
}
DEFAULT_ETF = "SPY"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# SUPABASE CLIENT
# ============================================================

_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ============================================================
# DATA LOADERS
# ============================================================

def load_full_universe():
    """Load all tickers + sectors from scanner_universe."""
    supabase = get_supabase()
    logger.info("Loading full universe from scanner_universe...")

    all_rows = []
    offset = 0
    while True:
        result = supabase.table("scanner_universe") \
            .select("ticker,sector") \
            .range(offset, offset + 999) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000

    tickers = []
    sector_map = {}
    for row in all_rows:
        t = row.get('ticker', '').strip()
        if t:
            tickers.append(t)
            sector_map[t] = row.get('sector', 'Technology')

    logger.info(f"  Loaded {len(tickers)} tickers")
    return tickers, sector_map


def load_all_price_history(tickers, start_date="2015-06-01"):
    """
    Load price history from Supabase for all tickers.
    Starts 6 months before target to have lookback for 20-day returns.
    """
    supabase = get_supabase()
    logger.info(f"Loading price history from {start_date}...")

    all_rows = []
    batch_size = 50
    t0 = time.time()

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        offset = 0
        while True:
            result = supabase.table("price_history") \
                .select("date,ticker,close,volume") \
                .in_("ticker", batch) \
                .gte("date", start_date) \
                .order("date") \
                .range(offset, offset + 999) \
                .execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            if len(result.data) < 1000:
                break
            offset += 1000

        if (i // batch_size) % 4 == 0 and i > 0:
            logger.info(f"  Loaded {len(all_rows)} rows ({i + batch_size}/{len(tickers)} tickers)...")

    elapsed = time.time() - t0
    logger.info(f"  Total: {len(all_rows)} rows in {elapsed:.1f}s")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['close'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    return df


# ============================================================
# DM COMPUTATION (vectorized for speed)
# ============================================================

def calculate_relative_strength(ticker_return, benchmark_return):
    diff = ticker_return - benchmark_return
    score = 50 + (diff * REL_STR_SCALE)
    return np.clip(score, 0, 100)


def compute_dm_history(price_df, sector_map, start_output="2016-01-01"):
    """
    Compute daily DM scores for all tickers using vectorized operations.
    Returns DataFrame with columns matching intraday scanner output.
    """
    logger.info("Computing DM scores (vectorized)...")
    t0 = time.time()

    # Build ETF mapping
    ticker_etf = {}
    for ticker in price_df['ticker'].unique():
        sector = sector_map.get(ticker, 'Technology')
        ticker_etf[ticker] = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)

    # Pivot to wide format for vectorized computation
    close_wide = price_df.pivot_table(index='date', columns='ticker', values='close')
    volume_wide = price_df.pivot_table(index='date', columns='ticker', values='volume')

    # 20-day returns
    returns_20d = close_wide.pct_change(RETURN_PERIOD)

    # Volume stats (20-day rolling)
    vol_20d_avg = volume_wide.rolling(20, min_periods=10).mean()

    # Volume z-score (ratio-based, same as intraday scanner)
    vol_ratio = volume_wide / vol_20d_avg
    volume_z = (vol_ratio - 0.5) * 66.67
    volume_z = volume_z.clip(0, 100)

    # SPY returns
    if 'SPY' not in returns_20d.columns:
        logger.error("SPY not in price history!")
        return pd.DataFrame()
    spy_ret = returns_20d['SPY']

    # Get all unique ETFs needed
    all_etfs = set(ticker_etf.values())

    # Compute DM for each ticker
    results = []
    tickers = [t for t in close_wide.columns if t not in all_etfs and t != 'SPY']

    for ticker in tickers:
        if ticker not in returns_20d.columns:
            continue

        etf = ticker_etf.get(ticker, DEFAULT_ETF)
        t_ret = returns_20d[ticker]

        # ETF return
        if etf in returns_20d.columns:
            etf_ret = returns_20d[etf]
        else:
            etf_ret = spy_ret  # fallback

        # Relative strength scores
        rel_str_etf = calculate_relative_strength(t_ret, etf_ret)
        rel_str_spy = calculate_relative_strength(t_ret, spy_ret)

        # Volume Z for this ticker
        vol_z = volume_z[ticker] if ticker in volume_z.columns else pd.Series(50.0, index=close_wide.index)

        # DM Raw
        dm_raw = (rel_str_etf * W_REL_STR_ETF +
                  rel_str_spy * W_REL_STR_SPY +
                  vol_z * W_VOLUME_Z)
        dm_raw = dm_raw.clip(0, 100)

        # EMA5
        dm_ema5 = dm_raw.ewm(span=EMA_PERIOD, adjust=False).mean().clip(0, 100)

        # Build per-ticker DataFrame
        t_df = pd.DataFrame({
            'date': close_wide.index,
            'ticker': ticker,
            'dm_score': dm_raw.round(1),
            'dm_ema5': dm_ema5.round(1),
            'current_price': close_wide[ticker].round(2),
            'volume_today': volume_wide[ticker].fillna(0).astype(int) if ticker in volume_wide.columns else 0,
            'volume_zscore': vol_z.round(2),
            'return_20d': t_ret.round(4),
            'etf_return_20d': etf_ret.round(4),
            'spy_return_20d': spy_ret.round(4),
        })

        # Filter to output range and drop NaN rows
        t_df = t_df[t_df['date'] >= start_output].dropna(subset=['dm_score'])
        results.append(t_df)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values(['date', 'dm_ema5'], ascending=[True, False]).reset_index(drop=True)

    elapsed = time.time() - t0
    logger.info(f"  Computed {len(combined)} rows for {len(tickers)} tickers in {elapsed:.1f}s")

    return combined


# ============================================================
# SUPABASE WRITE
# ============================================================

def write_to_supabase(df):
    """Write results to dm_intraday_history table."""
    supabase = get_supabase()
    logger.info(f"Writing {len(df)} rows to {SUPA_TABLE}...")

    # Convert to records
    records = []
    for _, r in df.iterrows():
        records.append({
            "date": r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date']),
            "ticker": r['ticker'],
            "dm_score": float(r['dm_score']),
            "dm_ema5": float(r['dm_ema5']),
            "current_price": float(r['current_price']),
            "volume_today": int(r['volume_today']),
            "volume_zscore": float(r['volume_zscore']),
            "return_20d": float(r['return_20d']),
            "etf_return_20d": float(r['etf_return_20d']),
            "spy_return_20d": float(r['spy_return_20d']),
        })

    # Batch upsert
    batch_size = 500
    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table(SUPA_TABLE).upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            written += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

        if (i // batch_size) % 100 == 0 and i > 0:
            logger.info(f"  Written {written}/{len(records)} rows...")

    logger.info(f"  Written {written} rows to {SUPA_TABLE}")


# ============================================================
# GOOGLE SHEETS PUSH
# ============================================================

def push_to_sheets(df=None):
    """
    Push dm_intraday_history to Google Sheets, split into partition tabs.
    If df is None, loads from Supabase first.
    """
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        logger.error("gspread not installed. Skipping.")
        return

    CREDS_FILE = 'credentials.json'
    if not os.path.exists(CREDS_FILE):
        creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            with open(CREDS_FILE, 'w') as f:
                f.write(creds_json)
        else:
            logger.warning("No Google credentials. Skipping.")
            return

    # If no DataFrame provided, load from Supabase
    if df is None:
        logger.info("Loading from Supabase for Sheets push...")
        df = load_from_supabase()
        if df.empty:
            logger.error("No data to push.")
            return

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open_by_key(SPREADSHEET_KEY)
    except Exception as e:
        logger.error(f"  Cannot open spreadsheet: {e}")
        return

    HEADERS = ['Date', 'Ticker', 'DM_Raw', 'DM_EMA5', 'Price',
               'Volume', 'Vol_Z', 'Return_20d', 'ETF_Ret_20d', 'SPY_Ret_20d']

    for partition in SHEETS_PARTITIONS:
        tab_name = partition["tab"]
        start = partition["start"]
        end = partition["end"]

        # Filter data for this partition
        part_df = df[(df['date'] >= start) & (df['date'] <= end)]
        if part_df.empty:
            logger.info(f"  No data for {tab_name}, skipping.")
            continue

        # Sort by date desc, then DM desc
        part_df = part_df.sort_values(['date', 'dm_ema5'], ascending=[False, False])

        logger.info(f"Pushing {len(part_df)} rows to {tab_name}...")

        sheet_rows = []
        for _, r in part_df.iterrows():
            date_str = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
            sheet_rows.append([
                date_str,
                r['ticker'],
                r['dm_score'],
                r['dm_ema5'],
                r['current_price'],
                int(r['volume_today']),
                r['volume_zscore'],
                r['return_20d'],
                r['etf_return_20d'],
                r['spy_return_20d'],
            ])

        total_cells = (len(sheet_rows) + 1) * len(HEADERS)
        logger.info(f"  Cells: {total_cells:,} ({len(sheet_rows)} rows x {len(HEADERS)} cols)")

        # Get or create tab
        total_rows_needed = len(sheet_rows) + 10
        try:
            sheet = spreadsheet.worksheet(tab_name)
            sheet.clear()
            if sheet.row_count < total_rows_needed:
                sheet.resize(rows=total_rows_needed, cols=len(HEADERS))
        except Exception:
            sheet = spreadsheet.add_worksheet(title=tab_name, rows=total_rows_needed, cols=len(HEADERS))

        # Write header
        sheet.update(range_name='A1', values=[HEADERS], value_input_option='USER_ENTERED')
        sheet.format('1:1', {'textFormat': {'bold': True}})
        sheet.freeze(rows=1)

        # Write data in batches
        BATCH_SIZE = 5000
        total_written = 0

        for i in range(0, len(sheet_rows), BATCH_SIZE):
            batch = sheet_rows[i:i + BATCH_SIZE]
            start_row = i + 2  # row 1 is header

            try:
                sheet.update(range_name=f'A{start_row}', values=batch, value_input_option='USER_ENTERED')
                total_written += len(batch)
                logger.info(f"  Written {total_written}/{len(sheet_rows)} rows to {tab_name}")
            except Exception as e:
                logger.error(f"  Batch error at row {start_row}: {e}")
                for j in range(0, len(batch), 1000):
                    mini = batch[j:j + 1000]
                    try:
                        sheet.update(range_name=f'A{start_row + j}', values=mini, value_input_option='USER_ENTERED')
                        total_written += len(mini)
                    except Exception as e2:
                        logger.error(f"  Mini-batch error at row {start_row + j}: {e2}")

            time.sleep(1)

        logger.info(f"  Pushed {total_written} rows to GS {tab_name}")


def load_from_supabase():
    """Load all data from dm_intraday_history table."""
    supabase = get_supabase()
    all_rows = []
    offset = 0

    while True:
        result = supabase.table(SUPA_TABLE) \
            .select("*") \
            .order("date", desc=True) \
            .range(offset, offset + 999) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000
        if offset % 50000 == 0:
            logger.info(f"  Loaded {offset} rows...")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Intraday DM History Backfill")
    parser.add_argument("--start", type=str, default="2016-01-01",
                        help="Start date for backfill (default: 2016-01-01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute only, don't write to DB or Sheets")
    parser.add_argument("--sheets-only", action="store_true",
                        help="Push existing Supabase data to Sheets (no recompute)")
    parser.add_argument("--no-sheets", action="store_true",
                        help="Write to Supabase only, skip Sheets push")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL/SUPABASE_KEY not set")
        sys.exit(1)

    if args.sheets_only:
        push_to_sheets()
        return

    # Load universe
    tickers, sector_map = load_full_universe()

    # Need ETFs + SPY in the price data too
    all_etfs = set(SECTOR_ETF_MAP.values()) | {"SPY"}
    all_tickers = list(set(tickers) | all_etfs)

    # Load price history (start 6 months earlier for lookback)
    from datetime import datetime as dt
    start_dt = dt.strptime(args.start, "%Y-%m-%d")
    lookback_start = (start_dt - timedelta(days=180)).strftime("%Y-%m-%d")

    price_df = load_all_price_history(all_tickers, start_date=lookback_start)
    if price_df.empty:
        logger.error("No price data loaded.")
        return

    logger.info(f"Price data: {price_df['ticker'].nunique()} tickers, "
                f"{price_df['date'].min().strftime('%Y-%m-%d')} to {price_df['date'].max().strftime('%Y-%m-%d')}")

    # Compute DM
    df = compute_dm_history(price_df, sector_map, start_output=args.start)
    if df.empty:
        logger.error("No DM results computed.")
        return

    # Summary
    logger.info(f"\nResults: {len(df)} rows, {df['ticker'].nunique()} tickers")
    logger.info(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    if args.dry_run:
        logger.info("DRY RUN - not writing to DB or Sheets")
        # Print sample
        latest = df[df['date'] == df['date'].max()].sort_values('dm_ema5', ascending=False).head(20)
        print(f"\nTop 20 by DM EMA5 on {df['date'].max().strftime('%Y-%m-%d')}:")
        for _, r in latest.iterrows():
            print(f"  {r['ticker']:<8} DM={r['dm_ema5']:>6.1f}  Price={r['current_price']:>9.2f}  Vol_Z={r['volume_zscore']:>6.2f}")
        return

    # Write to Supabase
    write_to_supabase(df)

    # Push to Sheets
    if not args.no_sheets:
        push_to_sheets(df)


if __name__ == "__main__":
    main()
