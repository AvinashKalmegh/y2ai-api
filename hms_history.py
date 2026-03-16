"""
HMS HISTORY BACKFILL
====================
Computes HMS v1 (3-component) for the full scanner universe (~580 tickers)
from 2016 to current. Uses the same formula as hms_calculator.py but
without Component 4 (Trade Fragmentation) since that requires too many
Polygon API calls for historical data.

Stores results in Supabase `hms_history` table and pushes to 3 Google Sheets.

HMS v1 formula:
    C1 = Price Compression (volume / price_range, 10d rolling)
    C2 = Volume Absorption (volume / price_drop on down days, 10d rolling)
    C3 = Persistent Order Flow (up_vol - down_vol blend, 5d/10d)

    Per-ticker z-score normalization for C1, C2 (60d rolling)
    Cross-sectional normalization per day:
        C1, C2 = percentile rank
        C3 = min-max

    HMS = C3 * 0.3529 + C2 * 0.2941 + C1 * 0.2353

Usage:
    python hms_history.py                      # Full backfill 2016-current
    python hms_history.py --start 2024-01-01   # From specific date
    python hms_history.py --dry-run            # Compute only, no writes
    python hms_history.py --sheets-only        # Push existing Supabase data to Sheets
    python hms_history.py --no-sheets          # Write to Supabase only

Environment Variables (.env):
    SUPABASE_URL, SUPABASE_KEY
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

# HMS v1 weights (3-component, C4 unavailable — rescaled from v2)
W_V1_FLOW = 0.3529
W_V1_ABSORPTION = 0.2941
W_V1_COMPRESSION = 0.2353

# Rolling windows (trading days)
COMPRESSION_WINDOW = 10
ABSORPTION_WINDOW = 10
FLOW_SHORT_WINDOW = 5
FLOW_LONG_WINDOW = 10

# Flow persistence blend
FLOW_SHORT_WEIGHT = 0.6
FLOW_LONG_WEIGHT = 0.4

# Warm-up period (rows flagged as hms_valid=FALSE)
WARMUP_DAYS = 10

# Supabase table
SUPA_TABLE = "hms_history"

# Google Sheets — each partition in its own spreadsheet (10M cell limit)
SHEETS_PARTITIONS = [
    {"tab": "HMS_2016_2019", "start": "2016-01-01", "end": "2019-12-31",
     "key": "1N5QD4tYRV6XFPJ56_MmPWMdzJyNt3l6_0AkSA2al40c"},
    {"tab": "HMS_2020_2022", "start": "2020-01-01", "end": "2022-12-31",
     "key": "1Zeix17K_A-s6nh2F6L9sCjARglvn8PjaTFBcQa7drDw"},
    {"tab": "HMS_2023_Current", "start": "2023-01-01", "end": "2099-12-31",
     "key": "1yeM58E4KE3JVIV99xAkKBR_Up-FZFyLz9ATQe-Iwur8"},
]

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
    """Load all tickers from scanner_universe."""
    supabase = get_supabase()
    logger.info("Loading full universe from scanner_universe...")

    all_rows = []
    offset = 0
    while True:
        result = supabase.table("scanner_universe") \
            .select("ticker") \
            .range(offset, offset + 999) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000

    tickers = [row['ticker'].strip() for row in all_rows if row.get('ticker', '').strip()]
    logger.info(f"  Loaded {len(tickers)} tickers")
    return tickers


def load_all_price_history(tickers, start_date="2015-06-01"):
    """
    Load OHLCV from Supabase price_history for all tickers.
    Starts 6 months before target for rolling window lookback.
    """
    supabase = get_supabase()
    logger.info(f"Loading price history from {start_date} ({len(tickers)} tickers)...")

    all_rows = []
    batch_size = 50
    t0 = time.time()

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        offset = 0
        while True:
            result = supabase.table("price_history") \
                .select("date,ticker,open,high,low,close,volume") \
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
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close', 'volume'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    return df


# ============================================================
# HMS COMPONENTS (same as hms_calculator.py)
# ============================================================

def compute_price_compression(df):
    """
    Component 1: Price Compression Score
    compression = rolling_volume(10d) / rolling_price_range(10d)
    """
    df = df.copy()
    df['price_range'] = df['high'].rolling(COMPRESSION_WINDOW).max() - df['low'].rolling(COMPRESSION_WINDOW).min()
    df['rolling_vol'] = df['volume'].rolling(COMPRESSION_WINDOW).mean()
    df['price_range'] = df['price_range'].replace(0, np.nan)
    df['compression_raw'] = df['rolling_vol'] / df['price_range']
    return df['compression_raw']


def compute_volume_absorption(df):
    """
    Component 2: Volume Absorption Ratio
    On down-days: absorption = volume / abs(price_change)
    """
    df = df.copy()
    df['price_change'] = (df['close'] - df['open']).abs()
    df['is_down'] = df['close'] < df['open']
    df['absorption'] = np.where(
        df['is_down'] & (df['price_change'] > 0),
        df['volume'] / df['price_change'],
        np.nan
    )
    df['absorption_raw'] = df['absorption'].rolling(ABSORPTION_WINDOW, min_periods=3).mean()
    return df['absorption_raw']


def compute_persistent_flow(df):
    """
    Component 3: Persistent Order Flow
    daily_imbalance = (up_vol - down_vol) / total_vol
    Blend of 5d and 10d averages.
    """
    df = df.copy()
    df['up_volume'] = np.where(df['close'] >= df['open'], df['volume'], 0)
    df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['daily_imbalance'] = (df['up_volume'] - df['down_volume']) / df['volume']
    df['flow_5d'] = df['daily_imbalance'].rolling(FLOW_SHORT_WINDOW).mean()
    df['flow_10d'] = df['daily_imbalance'].rolling(FLOW_LONG_WINDOW).mean()
    df['flow_raw'] = df['flow_5d'] * FLOW_SHORT_WEIGHT + df['flow_10d'] * FLOW_LONG_WEIGHT
    return df['flow_raw']


# ============================================================
# NORMALIZATION (same as hms_calculator.py)
# ============================================================

def normalize_cross_sectional_minmax(series):
    """Min-max normalization across the universe for that day."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def normalize_cross_sectional_percentile(series):
    """Percentile rank normalization (outlier-resistant)."""
    if len(series.dropna()) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, na_option='bottom')


# ============================================================
# HMS COMPUTATION
# ============================================================

def compute_hms_history(price_df, start_output="2016-01-01"):
    """
    Compute HMS v1 (3-component) for all tickers using the same logic
    as hms_calculator.py but for the full history.

    Steps:
    1. Per-ticker: compute raw C1, C2, C3 using rolling windows
    2. Per-ticker: z-score normalization for C1, C2 (60d rolling)
    3. Per-day: cross-sectional normalization (percentile for C1/C2, min-max for C3)
    4. Combine with v1 weights
    5. Flag warm-up rows
    """
    logger.info("Computing HMS v1 scores...")
    t0 = time.time()

    # 1. Compute raw components per ticker
    logger.info("  Step 1: Raw component computation...")
    all_results = []
    tickers = sorted(price_df['ticker'].unique())

    for idx, ticker in enumerate(tickers):
        t_df = price_df[price_df['ticker'] == ticker].copy().reset_index(drop=True)
        if len(t_df) < FLOW_LONG_WINDOW + 5:
            continue

        t_df['comp1_raw'] = compute_price_compression(t_df)
        t_df['comp2_raw'] = compute_volume_absorption(t_df)
        t_df['comp3_raw'] = compute_persistent_flow(t_df)
        t_df['ticker'] = ticker

        all_results.append(t_df[['date', 'ticker', 'close', 'volume',
                                  'comp1_raw', 'comp2_raw', 'comp3_raw']])

        if (idx + 1) % 100 == 0:
            logger.info(f"    Computed {idx + 1}/{len(tickers)} tickers...")

    if not all_results:
        logger.error("No results computed.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"  Raw components: {len(combined)} rows, {combined['ticker'].nunique()} tickers")

    # 2. Per-ticker z-score normalization for C1 and C2 (same as hms_calculator.py)
    logger.info("  Step 2: Per-ticker z-score normalization...")
    for comp in ['comp1_raw', 'comp2_raw']:
        combined[comp] = combined.groupby('ticker')[comp].transform(
            lambda x: (x - x.rolling(60, min_periods=10).mean()) /
                      x.rolling(60, min_periods=10).std().replace(0, np.nan)
        )

    # 3. Cross-sectional normalization per day
    logger.info("  Step 3: Cross-sectional normalization per day...")

    # Filter to output date range before normalization
    # (We need lookback data but only normalize/output from start_output)
    combined_output = combined[combined['date'] >= start_output].copy()

    def normalize_day(group):
        group = group.copy()
        group['comp1_norm'] = normalize_cross_sectional_percentile(group['comp1_raw'].fillna(0))
        group['comp2_norm'] = normalize_cross_sectional_percentile(group['comp2_raw'].fillna(0))
        group['comp3_norm'] = normalize_cross_sectional_minmax(group['comp3_raw'].fillna(0))
        return group

    # Process in chunks of dates to manage memory and show progress
    dates = sorted(combined_output['date'].unique())
    logger.info(f"    Normalizing {len(dates)} trading days...")

    chunk_size = 250
    normalized_chunks = []
    for i in range(0, len(dates), chunk_size):
        chunk_dates = dates[i:i + chunk_size]
        chunk_df = combined_output[combined_output['date'].isin(chunk_dates)]
        chunk_normalized = chunk_df.groupby('date', group_keys=False).apply(normalize_day)
        normalized_chunks.append(chunk_normalized)
        if (i + chunk_size) % 500 == 0 or i + chunk_size >= len(dates):
            logger.info(f"    Normalized {min(i + chunk_size, len(dates))}/{len(dates)} days...")

    combined_output = pd.concat(normalized_chunks, ignore_index=True)

    # 4. Compute final HMS v1 score
    logger.info("  Step 4: Computing HMS v1 scores...")
    combined_output['hms_score'] = (
        W_V1_FLOW * combined_output['comp3_norm'] +
        W_V1_ABSORPTION * combined_output['comp2_norm'] +
        W_V1_COMPRESSION * combined_output['comp1_norm']
    )
    combined_output['hms_score'] = combined_output['hms_score'].clip(0, 1).round(4)

    # 5. Flag warm-up period
    combined_output = combined_output.sort_values(['ticker', 'date']).reset_index(drop=True)
    combined_output['hms_valid'] = combined_output.groupby('ticker').cumcount() >= WARMUP_DAYS

    # Drop NaN HMS rows
    combined_output = combined_output.dropna(subset=['hms_score'])

    # Sort by date, then HMS desc
    combined_output = combined_output.sort_values(['date', 'hms_score'], ascending=[True, False]).reset_index(drop=True)

    elapsed = time.time() - t0
    invalid_count = (~combined_output['hms_valid']).sum()
    logger.info(f"  Done: {len(combined_output)} rows, {combined_output['ticker'].nunique()} tickers in {elapsed:.1f}s")
    logger.info(f"  Warm-up rows flagged: {invalid_count}")
    logger.info(f"  Date range: {combined_output['date'].min().strftime('%Y-%m-%d')} to {combined_output['date'].max().strftime('%Y-%m-%d')}")

    return combined_output


# ============================================================
# SUPABASE WRITE
# ============================================================

def write_to_supabase(df):
    """Write results to hms_history table."""
    supabase = get_supabase()
    logger.info(f"Writing {len(df)} rows to {SUPA_TABLE}...")

    records = []
    for _, r in df.iterrows():
        records.append({
            "date": r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date']),
            "ticker": r['ticker'],
            "hms_score": round(float(r['hms_score']), 4),
            "hms_c1_compression": round(float(r['comp1_norm']), 4),
            "hms_c2_absorption": round(float(r['comp2_norm']), 4),
            "hms_c3_flow": round(float(r['comp3_norm']), 4),
            "close": round(float(r['close']), 2),
            "volume": int(r['volume']) if pd.notna(r['volume']) else None,
            "hms_version": "v1",
            "hms_valid": bool(r['hms_valid']),
        })

    batch_size = 500
    written = 0
    t0 = time.time()
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table(SUPA_TABLE).upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            written += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

        if (i // batch_size) % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            logger.info(f"  Written {written}/{len(records)} rows ({rate:.0f} rows/sec)...")

    elapsed = time.time() - t0
    logger.info(f"  Written {written} rows to {SUPA_TABLE} in {elapsed:.1f}s")


# ============================================================
# GOOGLE SHEETS PUSH
# ============================================================

def push_to_sheets(df=None):
    """
    Push hms_history to Google Sheets, split into partition tabs.
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

    if df is None:
        logger.info("Loading from Supabase for Sheets push...")
        df = load_from_supabase()
        if df.empty:
            logger.error("No data to push.")
            return

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)

    HEADERS = ['Date', 'Ticker', 'HMS_Score', 'C1_Compression', 'C2_Absorption',
               'C3_Flow', 'Close', 'Volume', 'HMS_Version', 'HMS_Valid']

    for partition in SHEETS_PARTITIONS:
        tab_name = partition["tab"]
        start = partition["start"]
        end = partition["end"]
        sheet_key = partition["key"]

        # Filter data for this partition
        part_df = df[(df['date'] >= start) & (df['date'] <= end)]
        if part_df.empty:
            logger.info(f"  No data for {tab_name}, skipping.")
            continue

        # Sort by date desc, then HMS desc
        part_df = part_df.sort_values(['date', 'hms_score'], ascending=[False, False])

        logger.info(f"Pushing {len(part_df)} rows to {tab_name}...")

        try:
            spreadsheet = client.open_by_key(sheet_key)
        except Exception as e:
            logger.error(f"  Cannot open spreadsheet {sheet_key}: {e}")
            continue

        sheet_rows = []
        for _, r in part_df.iterrows():
            date_str = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
            sheet_rows.append([
                date_str,
                r['ticker'],
                round(float(r['hms_score']), 4) if 'hms_score' in r.index else r.get('comp1_norm', ''),
                round(float(r['comp1_norm']), 4),
                round(float(r['comp2_norm']), 4),
                round(float(r['comp3_norm']), 4),
                round(float(r['close']), 2),
                int(r['volume']) if pd.notna(r['volume']) else 0,
                r.get('hms_version', 'v1') if 'hms_version' in r.index else 'v1',
                bool(r['hms_valid']) if 'hms_valid' in r.index else True,
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
            start_row = i + 2

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
    """Load all data from hms_history table."""
    supabase = get_supabase()
    all_rows = []
    offset = 0

    logger.info("Loading hms_history from Supabase...")
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

    logger.info(f"  Loaded {len(all_rows)} rows from Supabase")
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])

    # Map column names to match computation output
    if 'hms_c1_compression' in df.columns:
        df.rename(columns={
            'hms_c1_compression': 'comp1_norm',
            'hms_c2_absorption': 'comp2_norm',
            'hms_c3_flow': 'comp3_norm',
        }, inplace=True)

    return df


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HMS History Backfill")
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
    tickers = load_full_universe()

    # Load price history (start 6 months earlier for rolling window lookback)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    lookback_start = (start_dt - timedelta(days=180)).strftime("%Y-%m-%d")

    price_df = load_all_price_history(tickers, start_date=lookback_start)
    if price_df.empty:
        logger.error("No price data loaded.")
        return

    logger.info(f"Price data: {price_df['ticker'].nunique()} tickers, "
                f"{price_df['date'].min().strftime('%Y-%m-%d')} to {price_df['date'].max().strftime('%Y-%m-%d')}")

    # Compute HMS
    df = compute_hms_history(price_df, start_output=args.start)
    if df.empty:
        logger.error("No HMS results computed.")
        return

    # Summary
    logger.info(f"\nResults: {len(df)} rows, {df['ticker'].nunique()} tickers")
    logger.info(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    if args.dry_run:
        logger.info("DRY RUN - not writing to DB or Sheets")
        latest = df[df['date'] == df['date'].max()].sort_values('hms_score', ascending=False).head(20)
        print(f"\nTop 20 by HMS on {df['date'].max().strftime('%Y-%m-%d')}:")
        for _, r in latest.iterrows():
            print(f"  {r['ticker']:<8} HMS={r['hms_score']:>6.4f}  "
                  f"C1={r['comp1_norm']:>5.3f}  C2={r['comp2_norm']:>5.3f}  C3={r['comp3_norm']:>5.3f}  "
                  f"Close={r['close']:>9.2f}")
        return

    # Write to Supabase
    write_to_supabase(df)

    # Push to Sheets
    if not args.no_sheets:
        push_to_sheets(df)


if __name__ == "__main__":
    main()
