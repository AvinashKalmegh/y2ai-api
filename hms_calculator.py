"""
HIDDEN MONEY SCORE (HMS) CALCULATOR
====================================
Detects institutional accumulation before DM registers movement.
HMS fires 2-4 weeks before DM crosses meaningful thresholds.

HMS_v1 (3-component, C4 unavailable):
    Reweighted from original 4-component formula.
    C1 and C2 use percentile rank normalization (outlier-resistant).
    C3 uses min-max normalization.

HMS_v2 (4-component, when trade count data available):
    HMS = (0.30 * Persistent Order Flow)
        + (0.25 * Volume Absorption Ratio)
        + (0.25 * Trade Fragmentation Index)
        + (0.20 * Price Compression Score)

All components normalized 0-1 across universe daily.
Final HMS_Score range: 0.0 to 1.0.
HMS_Valid = FALSE for warm-up period rows (first 10 trading days per ticker).

Usage:
    python hms_calculator.py                    # Run daily HMS for priority universe
    python hms_calculator.py --validate         # Validation run (NVDA, CEG, VST — full-universe normalization)
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

# HMS component weights (v2 — full 4-component)
W_PERSISTENT_FLOW = 0.30
W_VOLUME_ABSORPTION = 0.25
W_TRADE_FRAGMENTATION = 0.25
W_PRICE_COMPRESSION = 0.20

# HMS v1 weights (3-component, C4 unavailable — per brief spec)
W_V1_FLOW = 0.3529
W_V1_ABSORPTION = 0.2941
W_V1_COMPRESSION = 0.2353

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

VALIDATION_TICKERS = ["NVDA", "CEG", "VST"]


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

def normalize_cross_sectional_minmax(series):
    """Normalize a series 0-1 using min-max across the universe for that day."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def normalize_cross_sectional_percentile(series):
    """Normalize a series 0-1 using percentile rank across the universe for that day.
    Immune to single-outlier domination (Finding 1 fix)."""
    if len(series.dropna()) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, na_option='bottom')


# ============================================================
# MAIN HMS COMPUTATION
# ============================================================

def compute_hms(tickers, start_date="2019-06-01", fetch_trade_data=True, dry_run=False,
                output_tickers=None):
    """
    Compute HMS for all tickers.

    1. Load price history
    2. Compute raw components per ticker
    3. Normalize cross-sectionally per day
    4. Combine into final HMS score

    Args:
        tickers: Full universe for cross-sectional normalization.
        output_tickers: If set, filter final output to only these tickers.
                        Normalization still uses full universe.
    """
    logger.info("=" * 60)
    logger.info("HIDDEN MONEY SCORE (HMS) CALCULATION")
    logger.info(f"  Universe: {len(tickers)} tickers")
    if output_tickers:
        logger.info(f"  Output filter: {output_tickers}")
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

    # 3b. Per-ticker z-score normalization BEFORE cross-sectional ranking
    # This converts raw scores to "how unusual is this for THIS stock" so that
    # high-volume stocks like NVDA don't permanently dominate percentile ranks.
    logger.info("Applying per-ticker z-score normalization...")
    for comp in ['comp1_raw', 'comp2_raw']:
        combined[comp] = combined.groupby('ticker')[comp].transform(
            lambda x: (x - x.rolling(60, min_periods=10).mean()) /
                      x.rolling(60, min_periods=10).std().replace(0, np.nan)
        )

    # 4. Cross-sectional normalization per day (cross-universe, not per-ticker)
    logger.info("Normalizing across universe per day...")

    # Use transform instead of apply to avoid pandas index/column issues across versions
    by_date = combined.groupby('date')
    combined['comp1_norm'] = by_date['comp1_raw'].transform(
        lambda x: normalize_cross_sectional_percentile(x.fillna(0))
    )
    combined['comp2_norm'] = by_date['comp2_raw'].transform(
        lambda x: normalize_cross_sectional_percentile(x.fillna(0))
    )
    combined['comp3_norm'] = by_date['comp3_raw'].transform(
        lambda x: normalize_cross_sectional_minmax(x.fillna(0))
    )
    if has_fragmentation:
        combined['comp4_raw'] = combined['comp4_raw'].fillna(0)
        combined['comp4_norm'] = by_date['comp4_raw'].transform(
            lambda x: normalize_cross_sectional_minmax(x)
        )
    else:
        combined['comp4_norm'] = 0.0

    # 5. Compute final HMS
    if has_fragmentation:
        combined['hms_score'] = (
            W_PERSISTENT_FLOW * combined['comp3_norm'] +
            W_VOLUME_ABSORPTION * combined['comp2_norm'] +
            W_TRADE_FRAGMENTATION * combined['comp4_norm'] +
            W_PRICE_COMPRESSION * combined['comp1_norm']
        )
    else:
        # v1 weights from brief (NOT renormalized — intentionally < 1.0 to reflect missing C4)
        combined['hms_score'] = (
            W_V1_FLOW * combined['comp3_norm'] +
            W_V1_ABSORPTION * combined['comp2_norm'] +
            W_V1_COMPRESSION * combined['comp1_norm']
        )
        combined['comp4_norm'] = 0.0

    combined['hms_score'] = combined['hms_score'].clip(0, 1).round(4)

    # Finding 2: Label HMS version
    combined['hms_version'] = 'v1' if not has_fragmentation else 'v2'

    # Finding 4: Flag warm-up period rows where rolling windows have insufficient history
    # Sort by ticker+date so cumcount reflects chronological order within each ticker
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    combined['hms_valid'] = combined.groupby('ticker').cumcount() >= FLOW_LONG_WINDOW

    # Persistence-based ALERT: HMS >= 0.55 for 3+ consecutive days
    # WATCH = single day >= 0.55, ALERT = sustained accumulation (3+ days)
    WATCH_THRESHOLD = 0.55
    ALERT_CONSECUTIVE = 3

    def compute_signal(group):
        group = group.sort_values('date')
        is_watch = group['hms_score'] >= WATCH_THRESHOLD
        # Count consecutive WATCH days using cumsum trick
        consecutive = is_watch.groupby((~is_watch).cumsum()).cumcount() + 1
        consecutive = consecutive.where(is_watch, 0)
        group['hms_signal'] = 'NONE'
        group.loc[is_watch, 'hms_signal'] = 'WATCH'
        group.loc[consecutive >= ALERT_CONSECUTIVE, 'hms_signal'] = 'ALERT'
        group['hms_consecutive_days'] = consecutive.astype(int)
        return group

    combined = combined.groupby('ticker', group_keys=False).apply(compute_signal)

    # Drop rows with NaN HMS
    combined = combined.dropna(subset=['hms_score'])

    invalid_count = (~combined['hms_valid']).sum()
    if invalid_count > 0:
        logger.info(f"  Flagged {invalid_count} warm-up rows as HMS_Valid=FALSE")

    logger.info(f"  Computed HMS for {combined['ticker'].nunique()} tickers, {len(combined)} total rows")

    # Filter to output tickers if requested (normalization already used full universe)
    if output_tickers:
        combined = combined[combined['ticker'].isin(output_tickers)].copy()
        logger.info(f"  Filtered to {len(output_tickers)} output tickers: {len(combined)} rows")

    return combined


# ============================================================
# OUTPUT
# ============================================================

def print_latest_hms(df):
    """Print the most recent HMS scores sorted by score."""
    latest_date = df['date'].max()
    latest = df[df['date'] == latest_date].sort_values('hms_score', ascending=False)
    version = latest['hms_version'].iloc[0] if 'hms_version' in latest.columns else '??'

    print(f"\n{'='*100}")
    print(f"HMS SCORES ({version}) — {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*100}")
    print(f"{'Ticker':<8} {'HMS':>6} {'Signal':<7} {'Days':>4} {'Compress':>9} {'Absorb':>8} {'Flow':>8} {'Frag':>8} {'Close':>10}")
    print(f"{'-'*100}")

    for _, r in latest.iterrows():
        signal = r.get('hms_signal', 'NONE')
        consec = int(r.get('hms_consecutive_days', 0))
        print(f"{r['ticker']:<8} {r['hms_score']:>6.3f} {signal:<7} {consec:>4} "
              f"{r['comp1_norm']:>9.3f} {r['comp2_norm']:>8.3f} "
              f"{r['comp3_norm']:>8.3f} {r['comp4_norm']:>8.3f} "
              f"{r['close']:>10.2f}")

    print(f"{'='*100}")

    # Signal summary
    alerts = latest[latest.get('hms_signal', '') == 'ALERT']
    watches = latest[latest.get('hms_signal', '') == 'WATCH']
    if not alerts.empty:
        print(f"\n  ALERT (sustained >= 0.55): {', '.join(alerts['ticker'].tolist())}")
    if not watches.empty:
        print(f"  WATCH (>= 0.55): {', '.join(watches['ticker'].tolist())}")

    print()


def write_hms_to_supabase(df):
    """Write HMS results to hms_daily table."""
    supabase = get_supabase()

    records = []
    for _, r in df.iterrows():
        record = {
            "date": r['date'].strftime('%Y-%m-%d'),
            "ticker": r['ticker'],
            "hms_score": round(float(r['hms_score']), 4),
            "comp1_compression": round(float(r['comp1_norm']), 4),
            "comp2_absorption": round(float(r['comp2_norm']), 4),
            "comp3_flow": round(float(r['comp3_norm']), 4),
            "comp4_fragmentation": round(float(r['comp4_norm']), 4),
            "close": round(float(r['close']), 2),
            "volume": int(r['volume']) if pd.notna(r['volume']) else None,
            "hms_version": r.get('hms_version', 'v1'),
            "hms_valid": bool(r.get('hms_valid', True)),
            "hms_signal": r.get('hms_signal', 'NONE'),
            "hms_consecutive_days": int(r.get('hms_consecutive_days', 0)),
        }
        if 'dm_score' in r.index and pd.notna(r.get('dm_score')):
            record["dm_score"] = round(float(r['dm_score']), 2)
        records.append(record)

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

    HEADERS = ['Ticker', 'Date', 'HMS_Score', 'HMS_C1_Compression', 'HMS_C2_Absorption',
               'HMS_C3_Flow', 'HMS_C4_Fragmentation', 'Close', 'Volume',
               'HMS_Version', 'HMS_Valid', 'HMS_Signal', 'HMS_Consecutive_Days', 'DM_Score']

    # Load DM scores for the output dates to include in the sheet
    dm_scores = {}
    try:
        supabase = get_supabase()
        latest_date = df['date'].max().strftime('%Y-%m-%d')
        result = supabase.table("dm_latest") \
            .select("ticker,dm_smoothed") \
            .execute()
        for row in result.data or []:
            dm_scores[row['ticker']] = float(row['dm_smoothed']) if row.get('dm_smoothed') else None
    except Exception as e:
        logger.warning(f"  Could not load DM scores: {e}")

    # Push full history sorted by date desc, then by HMS desc within each date
    sorted_df = df.sort_values(['date', 'hms_score'], ascending=[False, False])

    rows = [HEADERS]
    for _, r in sorted_df.iterrows():
        dm_val = r.get('dm_score') if 'dm_score' in r.index and pd.notna(r.get('dm_score')) else dm_scores.get(r['ticker'])
        rows.append([
            r['ticker'],
            r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date']),
            round(r['hms_score'], 4),
            round(r['comp1_norm'], 4),
            round(r['comp2_norm'], 4),
            round(r['comp3_norm'], 4),
            round(r['comp4_norm'], 4),
            round(r['close'], 2),
            int(r['volume']) if pd.notna(r['volume']) else 0,
            r.get('hms_version', 'v1'),
            bool(r.get('hms_valid', True)),
            r.get('hms_signal', 'NONE'),
            int(r.get('hms_consecutive_days', 0)),
            round(dm_val, 2) if dm_val is not None else '',
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
        sheet = spreadsheet.add_worksheet(title=TAB, rows=max(len(rows)+10, 200), cols=len(HEADERS))

    sheet.update(range_name='A1', values=rows, value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})
    sheet.freeze(rows=1)

    logger.info(f"  Pushed {len(rows)-1} rows to GS {TAB}")


def _get_sheets_client():
    """Get authenticated gspread client and spreadsheet."""
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    CREDS_FILE = 'credentials.json'
    SPREADSHEET = "copy-dm-history 2024-current"

    if not os.path.exists(CREDS_FILE):
        creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            with open(CREDS_FILE, 'w') as f:
                f.write(creds_json)
        else:
            return None, None

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open(SPREADSHEET)
    return client, spreadsheet


def _write_tab(spreadsheet, tab_name, rows):
    """Write rows (including header) to a sheet tab."""
    try:
        sheet = spreadsheet.worksheet(tab_name)
        sheet.clear()
    except Exception:
        sheet = spreadsheet.add_worksheet(title=tab_name, rows=max(len(rows)+10, 200), cols=len(rows[0]) if rows else 10)

    sheet.update(range_name='A1', values=rows, value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})
    sheet.freeze(rows=1)
    logger.info(f"  Pushed {len(rows)-1} rows to GS {tab_name}")


def write_backtest_results_to_sheets(df, spreadsheet):
    """HMS_Backtest_Results — summary per ticker: avg HMS, signal counts, precision, lead time."""
    TAB = "HMS_Backtest_Results"

    # Load DM history for lead-time calculation
    supabase = get_supabase()
    tickers = df['ticker'].unique().tolist()
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

    # Build DM breakout dates (first crossing above 70)
    dm_breakouts = {}
    if dm_rows:
        dm_df = pd.DataFrame(dm_rows)
        dm_df['date'] = pd.to_datetime(dm_df['date'])
        dm_df['dm_smoothed'] = pd.to_numeric(dm_df['dm_smoothed'], errors='coerce')
        for ticker in tickers:
            t_dm = dm_df[dm_df['ticker'] == ticker].sort_values('date')
            if t_dm.empty:
                continue
            prev = t_dm['dm_smoothed'].shift(1)
            crossings = t_dm[(prev < 70) & (t_dm['dm_smoothed'] >= 70)]
            if not crossings.empty:
                dm_breakouts[ticker] = crossings['date'].iloc[-1]

    HEADERS = ['Ticker', 'Avg_HMS', 'Max_HMS', 'Total_Days', 'ALERT_Days', 'WATCH_Days',
               'First_ALERT_Date', 'DM_Breakout_Date', 'Lead_Days', 'HMS_Valid_Pct']

    rows = [HEADERS]
    for ticker in sorted(tickers):
        t_df = df[df['ticker'] == ticker].sort_values('date')
        if t_df.empty:
            continue

        valid_df = t_df[t_df['hms_valid'] == True]
        alert_days = len(t_df[t_df['hms_signal'] == 'ALERT'])
        watch_days = len(t_df[t_df['hms_signal'] == 'WATCH'])

        # First ALERT date
        alerts = t_df[t_df['hms_signal'] == 'ALERT']
        first_alert = alerts['date'].min() if not alerts.empty else None

        # Lead time
        dm_breakout = dm_breakouts.get(ticker)
        lead_days = ''
        if first_alert is not None and dm_breakout is not None and first_alert < dm_breakout:
            lead_days = (dm_breakout - first_alert).days

        rows.append([
            ticker,
            round(valid_df['hms_score'].mean(), 4) if not valid_df.empty else '',
            round(t_df['hms_score'].max(), 4),
            len(t_df),
            alert_days,
            watch_days,
            first_alert.strftime('%Y-%m-%d') if first_alert is not None else '',
            dm_breakout.strftime('%Y-%m-%d') if dm_breakout is not None else '',
            lead_days,
            round(len(valid_df) / len(t_df) * 100, 1) if len(t_df) > 0 else '',
        ])

    _write_tab(spreadsheet, TAB, rows)


def write_backtest_detail_to_sheets(df, spreadsheet):
    """HMS_Backtest_Detail — full row-level data (ticker, date, scores, components, DM)."""
    TAB = "HMS_Backtest_Detail"

    # Load DM scores
    supabase = get_supabase()
    tickers = df['ticker'].unique().tolist()
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

    dm_map = {}
    if dm_rows:
        for r in dm_rows:
            dm_map[(r['ticker'], r['date'])] = float(r['dm_smoothed']) if r.get('dm_smoothed') else None

    HEADERS = ['Ticker', 'Date', 'HMS_Score', 'C1_Compression', 'C2_Absorption',
               'C3_Flow', 'C4_Fragmentation', 'Close', 'Volume',
               'HMS_Signal', 'Consecutive_Days', 'HMS_Valid', 'DM_Score']

    sorted_df = df.sort_values(['ticker', 'date'])
    rows = [HEADERS]
    for _, r in sorted_df.iterrows():
        date_str = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
        dm_val = dm_map.get((r['ticker'], date_str))
        rows.append([
            r['ticker'],
            date_str,
            round(r['hms_score'], 4),
            round(r['comp1_norm'], 4),
            round(r['comp2_norm'], 4),
            round(r['comp3_norm'], 4),
            round(r['comp4_norm'], 4),
            round(r['close'], 2),
            int(r['volume']) if pd.notna(r['volume']) else 0,
            r.get('hms_signal', 'NONE'),
            int(r.get('hms_consecutive_days', 0)),
            bool(r.get('hms_valid', True)),
            round(dm_val, 2) if dm_val is not None else '',
        ])

    _write_tab(spreadsheet, TAB, rows)


def write_dm_correlation_to_sheets(df, spreadsheet):
    """HMS_DM_Correlation — per-ticker correlation between HMS and DM scores."""
    TAB = "HMS_DM_Correlation"

    # Load DM history
    supabase = get_supabase()
    tickers = df['ticker'].unique().tolist()
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

    if not dm_rows:
        logger.warning("  No DM data for correlation. Skipping HMS_DM_Correlation tab.")
        return

    dm_df = pd.DataFrame(dm_rows)
    dm_df['date'] = pd.to_datetime(dm_df['date'])
    dm_df['dm_smoothed'] = pd.to_numeric(dm_df['dm_smoothed'], errors='coerce')

    merged = df.merge(dm_df[['date', 'ticker', 'dm_smoothed']], on=['date', 'ticker'], how='inner')

    HEADERS = ['Ticker', 'Correlation', 'Avg_HMS', 'Avg_DM', 'HMS_Leads_DM',
               'Overlap_Days', 'HMS_Above_055_Days', 'DM_Above_70_Days']

    rows = [HEADERS]
    for ticker in sorted(tickers):
        t = merged[merged['ticker'] == ticker].dropna(subset=['hms_score', 'dm_smoothed'])
        if len(t) < 10:
            continue

        corr = t['hms_score'].corr(t['dm_smoothed'])
        avg_hms = t['hms_score'].mean()
        avg_dm = t['dm_smoothed'].mean()
        hms_above = len(t[t['hms_score'] >= 0.55])
        dm_above = len(t[t['dm_smoothed'] >= 70])

        # Check if HMS signal tends to lead DM: cross-correlation at lag 1-20 days
        # Find peak lag where HMS best predicts future DM
        best_lag = 0
        best_corr = corr
        t_sorted = t.sort_values('date').reset_index(drop=True)
        for lag in range(1, 21):
            if len(t_sorted) <= lag:
                break
            shifted_corr = t_sorted['hms_score'].iloc[:-lag].reset_index(drop=True).corr(
                t_sorted['dm_smoothed'].iloc[lag:].reset_index(drop=True)
            )
            if pd.notna(shifted_corr) and shifted_corr > best_corr:
                best_corr = shifted_corr
                best_lag = lag

        rows.append([
            ticker,
            round(corr, 4) if pd.notna(corr) else '',
            round(avg_hms, 4),
            round(avg_dm, 2),
            f"{best_lag}d (r={round(best_corr, 3)})" if best_lag > 0 else 'No lead',
            len(t),
            hms_above,
            dm_above,
        ])

    # Sort by correlation descending
    header = rows[0]
    data_rows = sorted(rows[1:], key=lambda x: x[1] if isinstance(x[1], float) else -999, reverse=True)
    rows = [header] + data_rows

    _write_tab(spreadsheet, TAB, rows)


def write_all_sheets(df):
    """Push data to all 4 HMS tabs in Google Sheets."""
    try:
        import gspread
    except ImportError:
        logger.warning("  gspread not installed. Skipping GS push.")
        return

    try:
        client, spreadsheet = _get_sheets_client()
        if spreadsheet is None:
            logger.warning("  No Google credentials. Skipping GS push.")
            return
    except Exception as e:
        logger.error(f"  Spreadsheet error: {e}")
        return

    # Tab 1: HMS_Daily (reuse existing logic but with shared client)
    write_hms_to_sheets(df)

    # Tab 2: HMS_Backtest_Results
    try:
        write_backtest_results_to_sheets(df, spreadsheet)
    except Exception as e:
        logger.error(f"  HMS_Backtest_Results error: {e}")

    # Tab 3: HMS_Backtest_Detail
    try:
        write_backtest_detail_to_sheets(df, spreadsheet)
    except Exception as e:
        logger.error(f"  HMS_Backtest_Detail error: {e}")

    # Tab 4: HMS_DM_Correlation
    try:
        write_dm_correlation_to_sheets(df, spreadsheet)
    except Exception as e:
        logger.error(f"  HMS_DM_Correlation error: {e}")


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
                  'comp3_norm', 'comp4_norm', 'dm_score', 'close', 'volume',
                  'hms_version', 'hms_valid']].copy()
    export.columns = ['Ticker', 'Date', 'HMS_Score', 'HMS_Component1', 'HMS_Component2',
                       'HMS_Component3', 'HMS_Component4', 'DM_Score', 'Close_Price', 'Volume',
                       'HMS_Version', 'HMS_Valid']
    export['Date'] = export['Date'].dt.strftime('%Y-%m-%d')
    export = export.sort_values(['Ticker', 'Date'])
    export.to_csv(filename, index=False)
    logger.info(f"  Exported {len(export)} rows to {filename}")


def export_validation_csv(df, validation_tickers, last_n_days=60):
    """Export validation CSV for specific tickers with DM breakout dates."""
    supabase = get_supabase()

    # Load DM history for breakout detection
    logger.info("Loading DM history for breakout comparison...")
    dm_rows = []
    offset = 0
    while True:
        result = supabase.table("dm_history") \
            .select("date,ticker,dm_smoothed") \
            .in_("ticker", validation_tickers) \
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

    # Find most recent DM crossing above 70 per ticker (below→above transition)
    breakout_dates = {}
    if dm_rows:
        dm_df = pd.DataFrame(dm_rows)
        dm_df['date'] = pd.to_datetime(dm_df['date'])
        dm_df['dm_smoothed'] = pd.to_numeric(dm_df['dm_smoothed'], errors='coerce')
        for ticker in validation_tickers:
            t_dm = dm_df[dm_df['ticker'] == ticker].sort_values('date')
            if t_dm.empty:
                continue
            # Find crossings: previous day < 70, current day >= 70
            prev = t_dm['dm_smoothed'].shift(1)
            crossings = t_dm[(prev < 70) & (t_dm['dm_smoothed'] >= 70)]
            if not crossings.empty:
                # Use most recent crossing
                breakout_dates[ticker] = crossings['date'].iloc[-1].strftime('%Y-%m-%d')

    # Take last N trading days BEFORE each ticker's breakout date
    filtered = []
    for ticker in validation_tickers:
        t_df = df[df['ticker'] == ticker].sort_values('date')
        if ticker in breakout_dates:
            # Filter to only data on or before the breakout date
            t_df = t_df[t_df['date'] <= breakout_dates[ticker]]
        filtered.append(t_df.tail(last_n_days))

    if not filtered:
        logger.warning("No validation data to export.")
        return

    export_df = pd.concat(filtered, ignore_index=True)

    # Build export
    rows = []
    for _, r in export_df.iterrows():
        rows.append({
            'Ticker': r['ticker'],
            'Date': r['date'].strftime('%Y-%m-%d'),
            'HMS_Score': round(r['hms_score'], 4),
            'HMS_C1_Compression': round(r['comp1_norm'], 4),
            'HMS_C2_Absorption': round(r['comp2_norm'], 4),
            'HMS_C3_Flow': round(r['comp3_norm'], 4),
            'HMS_C4_Fragmentation': round(r['comp4_norm'], 4),
            'Close': round(r['close'], 2),
            'Volume': int(r['volume']) if pd.notna(r['volume']) else 0,
            'HMS_Version': r.get('hms_version', 'v1'),
            'HMS_Valid': bool(r.get('hms_valid', True)),
            'DM_Breakout_Date': breakout_dates.get(r['ticker'], ''),
        })

    out_df = pd.DataFrame(rows)
    tickers_str = '_'.join(validation_tickers)
    filename = f"hms_validation_{tickers_str}.csv"
    out_df.to_csv(filename, index=False)
    logger.info(f"  Exported {len(out_df)} rows to {filename}")
    logger.info(f"  DM breakout dates: {breakout_dates}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hidden Money Score (HMS) Calculator")
    parser.add_argument("--validate", action="store_true",
                        help="Validation run: NVDA, CEG, VST (full-universe normalization)")
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
        # Use FULL universe for cross-sectional normalization, filter output to validation tickers
        # Ensure validation tickers are included in universe even if not in PRIORITY_TICKERS
        tickers = list(set(PRIORITY_TICKERS + VALIDATION_TICKERS))
        output_tickers = VALIDATION_TICKERS
        start_date = "2023-06-01"
        fetch_trades = not args.no_fragmentation
    elif args.backtest:
        tickers = PRIORITY_TICKERS
        output_tickers = None
        start_date = "2019-06-01"  # extra lookback for 2020 start
        fetch_trades = False  # too many API calls for full backtest
        logger.info("Backtest mode: Component 4 disabled (too many API calls)")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        output_tickers = None
        start_date = "2023-01-01"
        fetch_trades = not args.no_fragmentation
    else:
        tickers = PRIORITY_TICKERS
        output_tickers = None
        start_date = "2023-01-01"
        fetch_trades = not args.no_fragmentation

    # Compute (output_tickers filters after full-universe normalization)
    df = compute_hms(tickers, start_date=start_date, fetch_trade_data=fetch_trades,
                     output_tickers=output_tickers)

    if df.empty:
        logger.error("No HMS results.")
        return

    # Output
    print_latest_hms(df)

    if args.backtest:
        export_backtest_csv(df, "hms_backtest_v2.csv")

    if args.validate:
        # Export validation CSV with DM breakout comparison
        export_validation_csv(df, VALIDATION_TICKERS)

    if not args.dry_run and not args.validate and not args.backtest:
        write_hms_to_supabase(df)
        write_all_sheets(df)
    elif args.dry_run:
        logger.info("DRY RUN - results NOT written to database")


if __name__ == "__main__":
    main()
