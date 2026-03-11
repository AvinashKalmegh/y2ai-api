"""
HIDDEN MONEY SCORE (HMS) CALCULATOR
====================================
Detects institutional accumulation before DM registers movement.
HMS fires 2-4 weeks before DM crosses meaningful thresholds.

HMS = (0.30 * Persistent Order Flow)
    + (0.25 * Volume Absorption Ratio)
    + (0.25 * Trade Fragmentation Index)
    + (0.20 * Price Compression Score)

All components normalized 0-1 across universe daily.
Final HMS_Score range: 0.0 to 1.0.

Usage:
    python hms_calculator.py                    # Run daily HMS for priority universe
    python hms_calculator.py --validate         # Validation run (NVDA, AMAT, VST)
    python hms_calculator.py --backtest         # Full backtest 2020-2025
    python hms_calculator.py --dry-run          # Print results, don't write to DB
    python hms_calculator.py --tickers NVDA,TSLA  # Custom tickers

Environment Variables (.env):
    POLYGON_API_KEY   - Polygon.io API key
    SUPABASE_URL      - Supabase project URL
    SUPABASE_KEY      - Supabase service role key
"""

import os
import sys
import time
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# HMS component weights
W_PERSISTENT_FLOW = 0.30
W_VOLUME_ABSORPTION = 0.25
W_TRADE_FRAGMENTATION = 0.25
W_PRICE_COMPRESSION = 0.20

# Rolling windows (trading days)
COMPRESSION_WINDOW = 10
ABSORPTION_WINDOW = 10
FLOW_SHORT_WINDOW = 5
FLOW_LONG_WINDOW = 10
FRAGMENTATION_WINDOW = 10

# Flow persistence blend
FLOW_SHORT_WEIGHT = 0.6
FLOW_LONG_WEIGHT = 0.4

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Priority universe (same as intraday scanner)
PRIORITY_TICKERS = [
    "ACN", "ADBE", "AMD", "AMZN", "APP", "ASML", "AVGO", "BILL",
    "CEG", "CRM", "CRWD", "DDOG", "DLR", "DOCN", "EQIX", "FSLR",
    "FVRR", "GOOGL", "GS", "GTLB", "HUBS", "IBM", "MDB", "META",
    "MNDY", "MS", "MSFT", "MU", "NET", "NOW", "NRG", "NVDA",
    "OKTA", "ORCL", "PANW", "PLTR", "QCOM", "RBLX", "S", "SHOP",
    "SMCI", "SNOW", "TEAM", "TSLA", "TSM", "TWLO", "UBER", "UPWK",
    "VRT", "WDAY", "ZS",
]

VALIDATION_TICKERS = ["NVDA", "AMAT", "VST"]


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
# DATA LOADING
# ============================================================

def load_price_history(tickers, start_date="2019-06-01"):
    """
    Load OHLCV from Supabase price_history.
    Returns DataFrame: date, ticker, open, high, low, close, volume
    """
    supabase = get_supabase()
    all_rows = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        offset = 0
        page_size = 1000
        while True:
            result = supabase.table("price_history") \
                .select("date,ticker,open,high,low,close,volume") \
                .in_("ticker", batch) \
                .gte("date", start_date) \
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
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close', 'volume'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    return df


def fetch_trade_counts_polygon(tickers, start_date, end_date):
    """
    Fetch daily trade counts from Polygon for Component 4.
    Returns dict: {(ticker, date_str): trade_count}
    """
    trade_counts = {}

    for ticker in tickers:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}

        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            for r in data.get("results", []):
                date_str = datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
                if "n" in r:
                    trade_counts[(ticker, date_str)] = int(r["n"])
        except Exception as e:
            logger.warning(f"  Trade count fetch failed for {ticker}: {e}")

        time.sleep(0.2)  # courtesy delay

    logger.info(f"  Fetched trade counts: {len(trade_counts)} data points")
    return trade_counts


# ============================================================
# HMS COMPONENTS
# ============================================================

def compute_price_compression(df):
    """
    Component 1: Price Compression Score
    compression = rolling_volume(10d) / rolling_price_range(10d)
    High volume + tight range = accumulation signature.
    """
    df = df.copy()
    df['price_range'] = df['high'].rolling(COMPRESSION_WINDOW).max() - df['low'].rolling(COMPRESSION_WINDOW).min()
    df['rolling_vol'] = df['volume'].rolling(COMPRESSION_WINDOW).mean()

    # Avoid division by zero
    df['price_range'] = df['price_range'].replace(0, np.nan)
    df['compression_raw'] = df['rolling_vol'] / df['price_range']

    return df['compression_raw']


def compute_volume_absorption(df):
    """
    Component 2: Volume Absorption Ratio
    On down-days (close < open): absorption = volume / abs(price_change)
    High ratio = large volume, small price drop = supply being absorbed.
    """
    df = df.copy()
    df['price_change'] = (df['close'] - df['open']).abs()
    df['is_down'] = df['close'] < df['open']

    # Only compute on down days
    df['absorption'] = np.where(
        df['is_down'] & (df['price_change'] > 0),
        df['volume'] / df['price_change'],
        np.nan
    )

    # Rolling mean over down-days within window
    df['absorption_raw'] = df['absorption'].rolling(ABSORPTION_WINDOW, min_periods=3).mean()

    return df['absorption_raw']


def compute_persistent_flow(df):
    """
    Component 3: Persistent Order Flow (highest weight)
    daily_imbalance = (up_vol - down_vol) / total_vol
    Using daily bars: close > open = up day.
    Blend of 5d and 10d averages.
    """
    df = df.copy()

    # Daily imbalance: +1 if up day, -1 if down day, weighted by volume
    df['up_volume'] = np.where(df['close'] >= df['open'], df['volume'], 0)
    df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['daily_imbalance'] = (df['up_volume'] - df['down_volume']) / df['volume']

    # Short and long averages
    df['flow_5d'] = df['daily_imbalance'].rolling(FLOW_SHORT_WINDOW).mean()
    df['flow_10d'] = df['daily_imbalance'].rolling(FLOW_LONG_WINDOW).mean()

    df['flow_raw'] = df['flow_5d'] * FLOW_SHORT_WEIGHT + df['flow_10d'] * FLOW_LONG_WEIGHT

    return df['flow_raw']


def compute_trade_fragmentation(df, trade_counts):
    """
    Component 4: Trade Fragmentation Index
    fragmentation = trade_count / volume
    High ratio = many small trades = algorithmic accumulation.
    """
    df = df.copy()

    df['trade_count'] = df.apply(
        lambda r: trade_counts.get((r.name[0] if isinstance(r.name, tuple) else '', r['date'].strftime('%Y-%m-%d')), np.nan)
        if 'date' in r.index else np.nan,
        axis=1
    )

    # Try mapping from the trade_counts dict
    tc_series = []
    for _, row in df.iterrows():
        ticker = row.get('ticker', '')
        date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else ''
        tc_series.append(trade_counts.get((ticker, date_str), np.nan))

    df['trade_count'] = tc_series
    df['trade_count'] = pd.to_numeric(df['trade_count'], errors='coerce')

    df['frag_daily'] = np.where(
        df['volume'] > 0,
        df['trade_count'] / df['volume'],
        np.nan
    )
    df['fragmentation_raw'] = df['frag_daily'].rolling(FRAGMENTATION_WINDOW, min_periods=3).mean()

    return df['fragmentation_raw']


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_cross_sectional(series):
    """Normalize a series 0-1 using min-max across the universe for that day."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


# ============================================================
# MAIN HMS COMPUTATION
# ============================================================

def compute_hms(tickers, start_date="2019-06-01", fetch_trade_data=True, dry_run=False):
    """
    Compute HMS for all tickers.

    1. Load price history
    2. Compute raw components per ticker
    3. Normalize cross-sectionally per day
    4. Combine into final HMS score
    """
    logger.info("=" * 60)
    logger.info("HIDDEN MONEY SCORE (HMS) CALCULATION")
    logger.info(f"  Tickers: {len(tickers)}")
    logger.info(f"  Start date: {start_date}")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("Loading price history...")
    price_df = load_price_history(tickers, start_date)
    if price_df.empty:
        logger.error("No price data found.")
        return pd.DataFrame()

    logger.info(f"  Loaded {len(price_df)} rows for {price_df['ticker'].nunique()} tickers")

    # 2. Fetch trade counts for Component 4
    trade_counts = {}
    has_fragmentation = False
    if fetch_trade_data and POLYGON_API_KEY:
        logger.info("Fetching trade counts from Polygon...")
        tc_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        tc_end = datetime.now().strftime("%Y-%m-%d")
        trade_counts = fetch_trade_counts_polygon(tickers, tc_start, tc_end)
        has_fragmentation = len(trade_counts) > 0

    if not has_fragmentation:
        logger.warning("  Trade count data unavailable. Using HMS_v1 (3-component).")

    # 3. Compute raw components per ticker
    logger.info("Computing raw components...")
    all_results = []

    for ticker in tickers:
        t_df = price_df[price_df['ticker'] == ticker].copy().reset_index(drop=True)
        if len(t_df) < FLOW_LONG_WINDOW + 5:
            logger.warning(f"  {ticker}: insufficient data ({len(t_df)} rows), skipping")
            continue

        t_df['comp1_raw'] = compute_price_compression(t_df)
        t_df['comp2_raw'] = compute_volume_absorption(t_df)
        t_df['comp3_raw'] = compute_persistent_flow(t_df)

        if has_fragmentation:
            t_df['comp4_raw'] = compute_trade_fragmentation(t_df, trade_counts)
        else:
            t_df['comp4_raw'] = np.nan

        t_df['ticker'] = ticker
        all_results.append(t_df[['date', 'ticker', 'close', 'volume',
                                  'comp1_raw', 'comp2_raw', 'comp3_raw', 'comp4_raw']])

    if not all_results:
        logger.error("No results computed.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # 4. Cross-sectional normalization per day
    logger.info("Normalizing across universe per day...")

    def normalize_day(group):
        group = group.copy()
        group['comp1_norm'] = normalize_cross_sectional(group['comp1_raw'])
        group['comp2_norm'] = normalize_cross_sectional(group['comp2_raw'])
        group['comp3_norm'] = normalize_cross_sectional(group['comp3_raw'])
        if has_fragmentation:
            group['comp4_norm'] = normalize_cross_sectional(group['comp4_raw'])
        else:
            group['comp4_norm'] = 0.0
        return group

    combined = combined.groupby('date', group_keys=False).apply(normalize_day)

    # 5. Compute final HMS
    if has_fragmentation:
        combined['hms_score'] = (
            W_PERSISTENT_FLOW * combined['comp3_norm'] +
            W_VOLUME_ABSORPTION * combined['comp2_norm'] +
            W_TRADE_FRAGMENTATION * combined['comp4_norm'] +
            W_PRICE_COMPRESSION * combined['comp1_norm']
        )
    else:
        # Reweight without Component 4
        total_w = W_PERSISTENT_FLOW + W_VOLUME_ABSORPTION + W_PRICE_COMPRESSION
        combined['hms_score'] = (
            (W_PERSISTENT_FLOW / total_w) * combined['comp3_norm'] +
            (W_VOLUME_ABSORPTION / total_w) * combined['comp2_norm'] +
            (W_PRICE_COMPRESSION / total_w) * combined['comp1_norm']
        )
        combined['comp4_norm'] = 0.0

    combined['hms_score'] = combined['hms_score'].clip(0, 1).round(4)

    # Drop rows with NaN HMS
    combined = combined.dropna(subset=['hms_score'])

    logger.info(f"  Computed HMS for {combined['ticker'].nunique()} tickers, {len(combined)} total rows")

    return combined


# ============================================================
# OUTPUT
# ============================================================

def print_latest_hms(df):
    """Print the most recent HMS scores sorted by score."""
    latest_date = df['date'].max()
    latest = df[df['date'] == latest_date].sort_values('hms_score', ascending=False)

    print(f"\n{'='*85}")
    print(f"HMS SCORES — {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*85}")
    print(f"{'Ticker':<8} {'HMS':>6} {'Compress':>9} {'Absorb':>8} {'Flow':>8} {'Frag':>8} {'Close':>10}")
    print(f"{'-'*85}")

    for _, r in latest.iterrows():
        print(f"{r['ticker']:<8} {r['hms_score']:>6.3f} "
              f"{r['comp1_norm']:>9.3f} {r['comp2_norm']:>8.3f} "
              f"{r['comp3_norm']:>8.3f} {r['comp4_norm']:>8.3f} "
              f"{r['close']:>10.2f}")

    print(f"{'='*85}")

    # Top signals
    top = latest[latest['hms_score'] >= 0.7]
    if not top.empty:
        tickers_str = ', '.join(top['ticker'].tolist())
        print(f"\n  HIGH HMS (>= 0.7): {tickers_str}")

    print()


def write_hms_to_supabase(df):
    """Write HMS results to hms_daily table."""
    supabase = get_supabase()

    records = []
    for _, r in df.iterrows():
        records.append({
            "date": r['date'].strftime('%Y-%m-%d'),
            "ticker": r['ticker'],
            "hms_score": round(float(r['hms_score']), 4),
            "comp1_compression": round(float(r['comp1_norm']), 4),
            "comp2_absorption": round(float(r['comp2_norm']), 4),
            "comp3_flow": round(float(r['comp3_norm']), 4),
            "comp4_fragmentation": round(float(r['comp4_norm']), 4),
            "close": round(float(r['close']), 2),
            "volume": int(r['volume']) if pd.notna(r['volume']) else None,
        })

    # Batch upsert
    batch_size = 500
    uploaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("hms_daily").upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    logger.info(f"  Written {uploaded} rows to hms_daily")
    return uploaded


def write_hms_to_sheets(df):
    """Push latest HMS scores to Google Sheets."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        logger.warning("  gspread not installed. Skipping GS push.")
        return

    CREDS_FILE = 'credentials.json'
    SPREADSHEET = "copy-dm-history 2024-current"
    TAB = "HMS_Daily"

    if not os.path.exists(CREDS_FILE):
        creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            with open(CREDS_FILE, 'w') as f:
                f.write(creds_json)
        else:
            logger.warning("  No Google credentials. Skipping GS push.")
            return

    latest_date = df['date'].max()
    latest = df[df['date'] == latest_date].sort_values('hms_score', ascending=False)

    HEADERS = ['Date', 'Ticker', 'HMS_Score', 'Compression', 'Absorption',
               'Flow', 'Fragmentation', 'Close', 'Volume']

    rows = [HEADERS]
    for _, r in latest.iterrows():
        rows.append([
            latest_date.strftime('%Y-%m-%d'),
            r['ticker'],
            round(r['hms_score'], 4),
            round(r['comp1_norm'], 4),
            round(r['comp2_norm'], 4),
            round(r['comp3_norm'], 4),
            round(r['comp4_norm'], 4),
            round(r['close'], 2),
            int(r['volume']) if pd.notna(r['volume']) else 0,
        ])

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(SPREADSHEET)
    except Exception as e:
        logger.error(f"  Spreadsheet not found: {e}")
        return

    try:
        sheet = spreadsheet.worksheet(TAB)
        sheet.clear()
    except Exception:
        sheet = spreadsheet.add_worksheet(title=TAB, rows=200, cols=len(HEADERS))

    sheet.update(range_name='A1', values=rows, value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})
    sheet.freeze(rows=1)

    logger.info(f"  Pushed {len(rows)-1} rows to GS {TAB}")


def export_backtest_csv(df, filename="hms_backtest.csv"):
    """Export backtest data as CSV."""
    # Load DM history for comparison
    supabase = get_supabase()
    tickers = df['ticker'].unique().tolist()

    logger.info("Loading DM history for backtest comparison...")
    dm_rows = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        offset = 0
        while True:
            result = supabase.table("dm_history") \
                .select("date,ticker,dm_smoothed") \
                .in_("ticker", batch) \
                .gte("date", df['date'].min().strftime('%Y-%m-%d')) \
                .order("date") \
                .range(offset, offset + 1000 - 1) \
                .execute()
            if not result.data:
                break
            dm_rows.extend(result.data)
            if len(result.data) < 1000:
                break
            offset += 1000

    dm_df = pd.DataFrame(dm_rows)
    if not dm_df.empty:
        dm_df['date'] = pd.to_datetime(dm_df['date'])
        dm_df.rename(columns={'dm_smoothed': 'dm_score'}, inplace=True)
        df = df.merge(dm_df[['date', 'ticker', 'dm_score']], on=['date', 'ticker'], how='left')
    else:
        df['dm_score'] = np.nan

    # Export
    export = df[['ticker', 'date', 'hms_score', 'comp1_norm', 'comp2_norm',
                  'comp3_norm', 'comp4_norm', 'dm_score', 'close', 'volume']].copy()
    export.columns = ['Ticker', 'Date', 'HMS_Score', 'HMS_Component1', 'HMS_Component2',
                       'HMS_Component3', 'HMS_Component4', 'DM_Score', 'Close_Price', 'Volume']
    export['Date'] = export['Date'].dt.strftime('%Y-%m-%d')
    export = export.sort_values(['Ticker', 'Date'])
    export.to_csv(filename, index=False)
    logger.info(f"  Exported {len(export)} rows to {filename}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hidden Money Score (HMS) Calculator")
    parser.add_argument("--validate", action="store_true",
                        help="Validation run: NVDA, AMAT, VST only")
    parser.add_argument("--backtest", action="store_true",
                        help="Full backtest 2020-2025, export CSV")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated custom ticker list")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to DB")
    parser.add_argument("--no-fragmentation", action="store_true",
                        help="Skip Component 4 (trade count)")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL/SUPABASE_KEY not set")
        sys.exit(1)

    # Determine tickers and date range
    if args.validate:
        tickers = VALIDATION_TICKERS
        start_date = "2023-06-01"
        fetch_trades = not args.no_fragmentation
    elif args.backtest:
        tickers = PRIORITY_TICKERS
        start_date = "2019-06-01"  # extra lookback for 2020 start
        fetch_trades = False  # too many API calls for full backtest
        logger.info("Backtest mode: Component 4 disabled (too many API calls)")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        start_date = "2024-01-01"
        fetch_trades = not args.no_fragmentation
    else:
        tickers = PRIORITY_TICKERS
        start_date = "2024-01-01"
        fetch_trades = not args.no_fragmentation

    # Compute
    df = compute_hms(tickers, start_date=start_date, fetch_trade_data=fetch_trades)

    if df.empty:
        logger.error("No HMS results.")
        return

    # Output
    print_latest_hms(df)

    if args.backtest:
        export_backtest_csv(df, "hms_backtest.csv")

    if not args.dry_run and not args.validate and not args.backtest:
        write_hms_to_supabase(df)
        write_hms_to_sheets(df)
    elif args.dry_run:
        logger.info("DRY RUN - results NOT written to database")


if __name__ == "__main__":
    main()
