"""
universe_scanner_dm.py
FlowOS Universe Scanner - Phase 2: DM Calculation

Calculates Dark Matter (DM) for all tickers in universe_tickers.
Uses the exact same formula as the daily DM pipeline.

Usage:
  python universe_scanner_dm.py backfill          # Initial: calculate 260 days of DM history
  python universe_scanner_dm.py weekly             # Weekly: calculate DM for new trading days only
  python universe_scanner_dm.py status             # Show universe_dm_daily stats
  python universe_scanner_dm.py validate           # Compare DM values with dm_history for Preferred 28

Pipeline: Weekly run (Saturday/Sunday) after liquidity filter.
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

POLYGON_DELAY = 0.2

# DM Formula constants (identical to Dm_historical_pipeline.py)
W_REL_STR_ETF = 0.50
W_REL_STR_SPY = 0.30
W_VOLUME_Z = 0.20
REL_STR_SCALE = 500
EMA_PERIOD = 5
RETURN_PERIOD = 19       # 20-day returns (19 intervals)
VOLUME_AVG_PERIOD = 20

# Universe scanner specific
HISTORY_DAYS = 280       # Trading days to fetch (260 target + 20 buffer for lookback)
RETAIN_DAYS = 260        # Days to keep in universe_dm_daily

# Sector ETF mapping (same as daily pipeline)
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
    "Clean Energy": "TAN",
    "Cybersecurity": "HACK",
    "Aerospace & Defense": "ITA",
    "Defense": "ITA",
    "Transportation": "IYT",
}
DEFAULT_ETF = "SPY"

# SIC code to sector mapping (for universe tickers without explicit sector)
SIC_TO_SECTOR = {}
def _build_sic_map():
    """Build SIC code range to sector mapping."""
    ranges = [
        (100, 999, "Materials"),              # Agriculture/Forestry
        (1000, 1399, "Energy"),               # Mining (oil, gas, metals)
        (1400, 1499, "Materials"),             # Nonmetallic minerals
        (1500, 1799, "Industrials"),           # Construction
        (2000, 2199, "Consumer Staples"),      # Food, Tobacco
        (2200, 2399, "Consumer Discretionary"),# Textiles, Apparel
        (2400, 2799, "Materials"),             # Lumber, Paper, Printing
        (2800, 2829, "Materials"),             # Chemicals
        (2830, 2869, "Health Care"),           # Pharma, Biotech
        (2870, 2899, "Materials"),             # Ag chemicals
        (2900, 2999, "Energy"),               # Petroleum refining
        (3000, 3499, "Industrials"),           # Rubber, Metals
        (3500, 3599, "Industrials"),           # Machinery
        (3600, 3669, "Technology"),            # Electronics
        (3670, 3679, "Semiconductors"),        # Semiconductors
        (3680, 3699, "Technology"),            # Other electronics
        (3700, 3799, "Industrials"),           # Transportation equipment
        (3800, 3899, "Technology"),            # Instruments
        (3900, 3999, "Industrials"),           # Misc manufacturing
        (4000, 4799, "Industrials"),           # Transportation
        (4800, 4899, "Communication Services"),# Communications
        (4900, 4999, "Utilities"),             # Electric, Gas, Water
        (5000, 5199, "Industrials"),           # Wholesale - durable
        (5200, 5399, "Consumer Staples"),      # Retail - food, general
        (5400, 5499, "Consumer Staples"),      # Food stores
        (5500, 5999, "Consumer Discretionary"),# Retail - auto, apparel, etc.
        (6000, 6199, "Financials"),            # Banks
        (6200, 6299, "Financials"),            # Securities
        (6300, 6499, "Financials"),            # Insurance
        (6500, 6599, "Real Estate"),           # Real estate
        (6600, 6799, "Financials"),            # Other finance
        (7000, 7299, "Consumer Discretionary"),# Hotels, services
        (7300, 7399, "Technology"),            # Business services (many SaaS)
        (7400, 7999, "Consumer Discretionary"),# Entertainment, recreation
        (8000, 8099, "Health Care"),           # Health services
        (8100, 8499, "Industrials"),           # Legal, education
        (8700, 8799, "Technology"),            # Engineering, research
        (8800, 8999, "Consumer Discretionary"),# Other services
        (9100, 9999, "Industrials"),           # Public admin
    ]
    for start, end, sector in ranges:
        for code in range(start, end + 1):
            SIC_TO_SECTOR[str(code)] = sector

_build_sic_map()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# SUPABASE CLIENT
# ============================================================

def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# STEP 1: Load universe tickers
# ============================================================

def load_universe_tickers():
    """Load all tickers from universe_tickers with sector mapping.
    
    Priority: scanner_universe sector (curated) > SIC code mapping > SPY default.
    """
    logger.info("STEP 1: Loading universe tickers from Supabase...")
    supabase = get_supabase()
    
    # Load universe tickers
    result = supabase.table("universe_tickers").select("ticker,sic_code").execute()
    
    # Load curated sectors from scanner_universe
    scanner = supabase.table("scanner_universe").select("ticker,sector").execute()
    curated_sectors = {}
    if scanner.data:
        for r in scanner.data:
            curated_sectors[r["ticker"]] = r.get("sector", "")
    logger.info(f"  Loaded {len(curated_sectors)} curated sector mappings from scanner_universe")
    
    tickers = []
    curated_used = 0
    sic_used = 0
    unmapped = 0
    
    for r in result.data:
        ticker = r["ticker"]
        sic = r.get("sic_code", "") or ""
        
        # Priority 1: curated sector from scanner_universe
        if ticker in curated_sectors and curated_sectors[ticker]:
            sector = curated_sectors[ticker]
            curated_used += 1
        # Priority 2: SIC code mapping
        elif sic and sic in SIC_TO_SECTOR:
            sector = SIC_TO_SECTOR[sic]
            sic_used += 1
        else:
            sector = ""
            unmapped += 1
        
        etf = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)
        tickers.append({"ticker": ticker, "sector": sector, "etf": etf})
    
    # Count sector distribution
    sectors = {}
    for t in tickers:
        s = t["sector"] or "Unmapped"
        sectors[s] = sectors.get(s, 0) + 1
    
    logger.info(f"  Loaded {len(tickers)} tickers")
    logger.info(f"  Sector sources: {curated_used} curated, {sic_used} SIC, {unmapped} unmapped")
    logger.info(f"  Sector distribution:")
    for s, c in sorted(sectors.items(), key=lambda x: -x[1]):
        etf = SECTOR_ETF_MAP.get(s, DEFAULT_ETF)
        logger.info(f"    {s}: {c} -> {etf}")
    
    return tickers


# ============================================================
# STEP 2: Fetch price history from Polygon grouped daily
# ============================================================

def get_trading_days(n_days):
    """Get last n trading days (Mon-Fri)."""
    dates = []
    d = datetime.now()
    while len(dates) < n_days:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
    return sorted(dates)


def fetch_grouped_daily(date_str):
    """Fetch all US stock prices for a single date."""
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    params = {"adjusted": "true", "apiKey": POLYGON_API_KEY}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("resultsCount", 0) == 0:
        return {}
    
    result = {}
    for r in data.get("results", []):
        ticker = r.get("T", "")
        if ticker:
            result[ticker] = {
                "close": float(r.get("c", 0)),
                "volume": int(r.get("v", 0)),
            }
    return result


def fetch_price_history(universe_tickers, n_days=HISTORY_DAYS):
    """
    Fetch n_days of price history for all universe tickers using grouped daily.
    Returns dict of ticker -> DataFrame(date, close, volume).
    
    This is extremely efficient: n_days API calls regardless of ticker count.
    """
    logger.info(f"STEP 2: Fetching {n_days} trading days of price data...")
    
    ticker_set = {t["ticker"] for t in universe_tickers}
    # Also need SPY and all sector ETFs
    all_etfs = set(SECTOR_ETF_MAP.values()) | {"SPY"}
    need_tickers = ticker_set | all_etfs
    
    dates = get_trading_days(n_days)
    logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
    logger.info(f"  Tracking {len(need_tickers)} tickers (universe + ETFs)")
    
    # Accumulate: ticker -> list of (date, close, volume)
    price_data = {t: [] for t in need_tickers}
    skipped_dates = 0
    
    for i, date_str in enumerate(dates):
        daily = fetch_grouped_daily(date_str)
        
        if not daily:
            skipped_dates += 1
            continue
        
        for ticker in need_tickers:
            if ticker in daily:
                price_data[ticker].append({
                    "date": date_str,
                    "close": daily[ticker]["close"],
                    "volume": daily[ticker]["volume"],
                })
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Fetched {i + 1}/{n_days} dates")
        
        time.sleep(POLYGON_DELAY)
    
    if skipped_dates:
        logger.info(f"  Skipped {skipped_dates} dates (holidays/weekends)")
    
    # Convert to DataFrames
    price_dfs = {}
    for ticker in need_tickers:
        if price_data[ticker]:
            df = pd.DataFrame(price_data[ticker])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            price_dfs[ticker] = df
    
    logger.info(f"  Built price DataFrames for {len(price_dfs)} tickers")
    
    # Report coverage
    have_data = sum(1 for t in ticker_set if t in price_dfs and len(price_dfs[t]) >= RETURN_PERIOD + 5)
    logger.info(f"  {have_data}/{len(ticker_set)} universe tickers have enough data for DM calculation")
    
    return price_dfs


# ============================================================
# STEP 3: DM Calculation (identical formula to daily pipeline)
# ============================================================

def calculate_relative_strength(ticker_return, benchmark_return):
    """Calculate relative strength score (0-100)."""
    diff = ticker_return - benchmark_return
    score = 50 + (diff * REL_STR_SCALE)
    return max(0, min(100, score))


def calculate_volume_z(vol_5d_avg, vol_baseline_avg):
    """Calculate volume z-score scaled to 0-100."""
    if vol_baseline_avg is None or vol_baseline_avg <= 0 or pd.isna(vol_baseline_avg):
        return 50.0
    if vol_5d_avg is None or pd.isna(vol_5d_avg):
        return 50.0
    ratio = vol_5d_avg / vol_baseline_avg
    score = (ratio - 0.5) * 66.67
    return max(0, min(100, score))


def calculate_dm_for_ticker(ticker_df, spy_df, etf_df):
    """
    Calculate DM time series for a single ticker.
    Identical formula to Dm_historical_pipeline.py calculate_dm_for_ticker().
    
    Returns DataFrame with date, dm_raw, dm_smoothed, close, volume.
    """
    if len(ticker_df) < RETURN_PERIOD + 5:
        return pd.DataFrame()
    
    # Set index to date
    ticker_df = ticker_df.set_index("date").sort_index()
    spy_df = spy_df.set_index("date").sort_index()
    etf_df = etf_df.set_index("date").sort_index()
    
    # Ticker metrics
    ticker_df["return_20d"] = ticker_df["close"].pct_change(RETURN_PERIOD)
    ticker_df["vol_5d_avg"] = ticker_df["volume"].rolling(5).mean()
    
    # 60-calendar-day volume baseline (matches GS)
    dates = ticker_df.index
    cutoffs = dates - pd.Timedelta(days=60)
    window_starts = dates.searchsorted(cutoffs, side="left")
    indices = np.arange(len(dates))
    total_in_window = indices + 1 - window_starts
    baseline_counts = np.maximum(total_in_window - 5, 1)
    vol_arr = ticker_df["volume"].values.astype(float)
    cs = np.concatenate(([0.0], np.cumsum(vol_arr)))
    window_sums = cs[indices + 1] - cs[window_starts]
    recent_sums = cs[indices + 1] - cs[np.maximum(indices - 4, 0)]
    baseline_sums = window_sums - recent_sums
    baseline_avg_arr = baseline_sums / baseline_counts
    baseline_avg_arr[:5] = np.nan
    baseline_avg_arr[total_in_window < 10] = np.nan
    ticker_df["vol_baseline_avg"] = baseline_avg_arr
    
    # SPY return
    spy_df["spy_return_20d"] = spy_df["close"].pct_change(RETURN_PERIOD)
    
    # ETF return
    etf_df["etf_return_20d"] = etf_df["close"].pct_change(RETURN_PERIOD)
    
    # Merge
    merged = ticker_df[["close", "volume", "return_20d", "vol_5d_avg", "vol_baseline_avg"]].copy()
    merged = merged.join(spy_df[["spy_return_20d"]], how="left")
    merged = merged.join(etf_df[["etf_return_20d"]], how="left")
    
    merged["spy_return_20d"] = merged["spy_return_20d"].ffill()
    merged["etf_return_20d"] = merged["etf_return_20d"].ffill()
    
    merged = merged.dropna(subset=["return_20d"])
    
    if merged.empty:
        return pd.DataFrame()
    
    # DM components
    merged["rel_str_etf"] = merged.apply(
        lambda r: calculate_relative_strength(r["return_20d"], r["etf_return_20d"])
        if pd.notna(r["etf_return_20d"]) else 50.0, axis=1
    )
    merged["rel_str_spy"] = merged.apply(
        lambda r: calculate_relative_strength(r["return_20d"], r["spy_return_20d"])
        if pd.notna(r["spy_return_20d"]) else 50.0, axis=1
    )
    merged["volume_z"] = merged.apply(
        lambda r: calculate_volume_z(r["vol_5d_avg"], r["vol_baseline_avg"]), axis=1
    )
    
    # DM Raw
    merged["dm_raw"] = (
        merged["rel_str_etf"] * W_REL_STR_ETF +
        merged["rel_str_spy"] * W_REL_STR_SPY +
        merged["volume_z"] * W_VOLUME_Z
    ).clip(0, 100)
    
    # DM Smoothed (EMA-5)
    merged["dm_smoothed"] = merged["dm_raw"].ewm(span=EMA_PERIOD, adjust=False).mean().clip(0, 100)
    
    # Return only needed columns
    result = merged[["close", "volume", "dm_raw", "dm_smoothed"]].copy()
    result = result.reset_index()
    result["dm_raw"] = result["dm_raw"].round(2)
    result["dm_smoothed"] = result["dm_smoothed"].round(2)
    result["close"] = result["close"].round(2)
    
    return result


def calculate_all_dm(universe_tickers, price_dfs, start_date=None):
    """
    Calculate DM for all universe tickers.
    
    Args:
        universe_tickers: list of {ticker, sector, etf}
        price_dfs: dict of ticker -> DataFrame(date, close, volume)
        start_date: only return DM rows after this date (for weekly updates)
    
    Returns:
        list of dicts ready for Supabase upsert
    """
    logger.info("STEP 3: Calculating DM for all universe tickers...")
    
    spy_df = price_dfs.get("SPY")
    if spy_df is None or spy_df.empty:
        logger.error("  No SPY data! Cannot calculate DM.")
        return []
    
    all_rows = []
    processed = 0
    skipped = 0
    
    for i, t in enumerate(universe_tickers):
        ticker = t["ticker"]
        etf = t["etf"]
        
        ticker_df = price_dfs.get(ticker)
        if ticker_df is None or len(ticker_df) < RETURN_PERIOD + 5:
            skipped += 1
            continue
        
        etf_df = price_dfs.get(etf)
        if etf_df is None:
            etf_df = spy_df.copy()  # Fallback to SPY
        
        try:
            dm_df = calculate_dm_for_ticker(
                ticker_df.copy(), spy_df.copy(), etf_df.copy()
            )
            
            if dm_df.empty:
                skipped += 1
                continue
            
            # Filter to start_date if specified
            if start_date:
                dm_df = dm_df[dm_df["date"] >= pd.Timestamp(start_date)]
            
            for _, row in dm_df.iterrows():
                all_rows.append({
                    "ticker": ticker,
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "dm_raw": float(row["dm_raw"]),
                    "dm_smoothed": float(row["dm_smoothed"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                })
            
            processed += 1
            
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                logger.warning(f"  Error calculating DM for {ticker}: {e}")
        
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i + 1}/{len(universe_tickers)} "
                        f"({processed} calculated, {skipped} skipped, {len(all_rows)} rows)")
    
    logger.info(f"  Done: {processed} tickers calculated, {skipped} skipped")
    logger.info(f"  Total DM rows: {len(all_rows)}")
    
    return all_rows


# ============================================================
# STEP 4: Write to Supabase
# ============================================================

def write_dm_to_supabase(rows):
    """Write DM rows to universe_dm_daily table."""
    logger.info(f"STEP 4: Writing {len(rows)} rows to universe_dm_daily...")
    
    if not rows:
        logger.info("  No rows to write.")
        return
    
    supabase = get_supabase()
    
    BATCH = 5000
    uploaded = 0
    errors = 0
    
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        try:
            supabase.table("universe_dm_daily").upsert(
                batch, on_conflict="ticker,date"
            ).execute()
            uploaded += len(batch)
            
            if uploaded % 50000 == 0 or uploaded == len(rows):
                logger.info(f"  Upserted {uploaded}/{len(rows)}")
                
        except Exception as e:
            errors += 1
            logger.error(f"  Upsert error at batch {i}: {e}")
            # Try smaller batches on error
            for row in batch:
                try:
                    supabase.table("universe_dm_daily").upsert(
                        [row], on_conflict="ticker,date"
                    ).execute()
                    uploaded += 1
                except:
                    pass
            time.sleep(1)
    
    logger.info(f"  Done: {uploaded} rows written ({errors} batch errors)")


def cleanup_old_data():
    """Remove data older than RETAIN_DAYS trading days."""
    logger.info(f"Cleaning up data older than {RETAIN_DAYS} trading days...")
    supabase = get_supabase()
    
    # Calculate cutoff date (~260 trading days ≈ 365 calendar days)
    cutoff = (datetime.now() - timedelta(days=int(RETAIN_DAYS * 1.5))).strftime("%Y-%m-%d")
    
    try:
        result = supabase.table("universe_dm_daily").delete().lt("date", cutoff).execute()
        logger.info(f"  Deleted rows before {cutoff}")
    except Exception as e:
        logger.error(f"  Cleanup error: {e}")


# ============================================================
# STATUS COMMAND
# ============================================================

def show_status():
    """Show universe_dm_daily stats."""
    supabase = get_supabase()
    
    try:
        # Total rows
        result = supabase.table("universe_dm_daily").select("ticker", count="exact").limit(1).execute()
        logger.info(f"universe_dm_daily: {result.count} total rows")
        
        # Unique tickers
        tickers = supabase.table("universe_dm_daily").select("ticker").execute()
        unique = len(set(r["ticker"] for r in tickers.data)) if tickers.data else 0
        logger.info(f"  Unique tickers: {unique}")
        
        # Date range
        first = supabase.table("universe_dm_daily").select("date").order("date").limit(1).execute()
        last = supabase.table("universe_dm_daily").select("date").order("date", desc=True).limit(1).execute()
        if first.data and last.data:
            logger.info(f"  Date range: {first.data[0]['date']} to {last.data[0]['date']}")
        
        # Rows per date (latest)
        if last.data:
            latest_date = last.data[0]["date"]
            latest = supabase.table("universe_dm_daily").select("ticker", count="exact").eq("date", latest_date).execute()
            logger.info(f"  Tickers on {latest_date}: {latest.count}")
            
    except Exception as e:
        logger.error(f"Error: {e}")


# ============================================================
# VALIDATE COMMAND
# ============================================================

def validate_vs_daily_pipeline():
    """Compare DM values with dm_history for Preferred 28 tickers."""
    PREFERRED_28 = [
        "CEG", "CRWD", "TSM", "APP", "VRT", "MU", "NVDA", "CCJ",
        "DNN", "PLTR", "TSLA", "TTD", "MSTR", "UEC", "LEU", "HAL",
        "WDC", "ENPH", "UUUU", "PDD", "NCLH", "FCX", "RCL", "LVS",
        "PSKY", "MRNA", "WYNN", "SMR"
    ]
    
    supabase = get_supabase()
    logger.info("Validating universe DM vs daily pipeline DM for Preferred 28...")
    
    # Get latest date from both tables
    uni_latest = supabase.table("universe_dm_daily").select("date").order("date", desc=True).limit(1).execute()
    dm_latest = supabase.table("dm_history").select("date").order("date", desc=True).limit(1).execute()
    
    if not uni_latest.data or not dm_latest.data:
        logger.error("  One or both tables are empty.")
        return
    
    # Use the earlier of the two latest dates
    uni_date = uni_latest.data[0]["date"]
    dm_date = dm_latest.data[0]["date"]
    compare_date = min(uni_date, dm_date)
    logger.info(f"  Comparing on date: {compare_date}")
    
    matches = 0
    mismatches = 0
    missing = 0
    
    for ticker in PREFERRED_28:
        uni = supabase.table("universe_dm_daily").select("dm_smoothed").eq("ticker", ticker).eq("date", compare_date).execute()
        dm = supabase.table("dm_history").select("dm_smoothed").eq("ticker", ticker).eq("date", compare_date).execute()
        
        if not uni.data or not dm.data:
            missing += 1
            logger.info(f"  {ticker}: MISSING (uni={bool(uni.data)}, dm={bool(dm.data)})")
            continue
        
        uni_val = float(uni.data[0]["dm_smoothed"])
        dm_val = float(dm.data[0]["dm_smoothed"])
        diff = abs(uni_val - dm_val)
        
        if diff <= 2.0:
            matches += 1
        else:
            mismatches += 1
            logger.info(f"  {ticker}: MISMATCH universe={uni_val:.1f} vs pipeline={dm_val:.1f} (diff={diff:.1f})")
    
    logger.info(f"\n  Results: {matches} match, {mismatches} mismatch, {missing} missing")
    if mismatches > 0:
        logger.info("  Note: Small differences are expected due to price data timing and rounding.")


# ============================================================
# MAIN COMMANDS
# ============================================================

def run_backfill():
    """Initial backfill: calculate full 260-day DM history."""
    logger.info("=" * 60)
    logger.info("UNIVERSE SCANNER - PHASE 2: DM BACKFILL")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)
    
    # Step 1: Load universe
    universe = load_universe_tickers()
    if not universe:
        logger.error("No tickers in universe_tickers. Run Phase 1 first.")
        return
    
    # Step 2: Fetch prices
    price_dfs = fetch_price_history(universe, n_days=HISTORY_DAYS)
    
    # Step 3: Calculate DM
    rows = calculate_all_dm(universe, price_dfs)
    
    # Step 4: Write to Supabase
    write_dm_to_supabase(rows)
    
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)


def run_weekly():
    """Weekly update: calculate DM for new trading days only."""
    logger.info("=" * 60)
    logger.info("UNIVERSE SCANNER - PHASE 2: WEEKLY DM UPDATE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    
    # Find last date in universe_dm_daily
    last = supabase.table("universe_dm_daily").select("date").order("date", desc=True).limit(1).execute()
    
    if not last.data:
        logger.info("No existing data. Running full backfill instead.")
        run_backfill()
        return
    
    last_date = last.data[0]["date"]
    logger.info(f"  Last DM date: {last_date}")
    
    # Still need full price history for DM lookback windows
    universe = load_universe_tickers()
    price_dfs = fetch_price_history(universe, n_days=HISTORY_DAYS)
    
    # Only calculate rows after last_date
    start_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"  Calculating DM from {start_date} onwards")
    
    rows = calculate_all_dm(universe, price_dfs, start_date=start_date)
    write_dm_to_supabase(rows)
    
    # Cleanup old data
    cleanup_old_data()
    
    logger.info("=" * 60)
    logger.info("WEEKLY UPDATE COMPLETE")
    logger.info("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowOS Universe Scanner - DM Calculation")
    parser.add_argument("command", choices=["backfill", "weekly", "status", "validate"],
                        help="backfill: initial 260-day history | weekly: update new days | status: show stats | validate: compare with daily pipeline")
    
    args = parser.parse_args()
    
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in .env")
        sys.exit(1)
    
    if args.command == "backfill":
        run_backfill()
    elif args.command == "weekly":
        run_weekly()
    elif args.command == "status":
        show_status()
    elif args.command == "validate":
        validate_vs_daily_pipeline()