"""
DM HISTORICAL PIPELINE
=======================
Calculates Dark Matter scores for the full Scanner Universe (533 tickers)
using the FlowOS 3-component formula and stores in Supabase.

DM = (RelStr vs ETF * 0.50) + (RelStr vs SPY * 0.30) + (VolumeZ * 0.20)

Usage:
    python dm_historical_pipeline.py fetch        # Fetch prices (resumable)
    python dm_historical_pipeline.py calculate    # Calculate DM scores
    python dm_historical_pipeline.py summary      # Build dm_latest
    python dm_historical_pipeline.py validate     # Spot-check results
    python dm_historical_pipeline.py all          # Run full pipeline

Requirements:
    pip install supabase pandas requests python-dotenv

Environment Variables (.env):
    TWELVEDATA_API_KEY=your_key
    SUPABASE_URL=https://jfdihmlxzemvdytrdcpw.supabase.co
    SUPABASE_KEY=your_key
"""

import os
import sys
import time
import json
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

TWELVEDATA_API_KEY = os.getenv("TWELVE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Twelve Data rate limits (free tier)
CALLS_PER_MINUTE = 8
DELAY_BETWEEN_CALLS = 60 / CALLS_PER_MINUTE  # 7.5 seconds
DAILY_CALL_LIMIT = 800

# Price history range
START_DATE = "2020-01-02"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# DM formula weights (FlowOS)
W_REL_STR_ETF = 0.50
W_REL_STR_SPY = 0.30
W_VOLUME_Z = 0.20

# Relative strength scaling factor
# Maps return differences to 0-100 scale centered at 50
# 10% outperformance -> score of 75
REL_STR_SCALE = 500  # Matches GS calcRelativeStrength: 50 + (diff * 500)

# EMA smoothing period
EMA_PERIOD = 5
EMA_K = 2 / (EMA_PERIOD + 1)  # 0.333

# Phase classification uses FlowOS determinePhase(dm, etfFlow) logic

# Lookback for returns
RETURN_PERIOD = 19  # GS: prices.slice(-20) then (last - first)/first = 19 intervals
VOLUME_AVG_PERIOD = 20

# Batch sizes
SUPABASE_BATCH_SIZE = 500
PRICE_FETCH_BATCH_SIZE = 50  # tickers per checkpoint save

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# SECTOR ETF MAPPING
# ============================================================

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

# All unique ETFs needed (plus SPY as benchmark)
ALL_ETFS = list(set(SECTOR_ETF_MAP.values())) + ["SPY"]

# Default ETF for unmapped sectors
DEFAULT_ETF = "SPY"


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
# TICKER UNIVERSE LOADING
# ============================================================

def load_universe(csv_path="universe.csv"):
    """
    Load ticker universe with sector mappings.
    
    Tries in order:
    1. Local CSV file (ticker, sector columns)
    2. Supabase scanner_universe table (if exists)
    3. Fallback to ARGUS-1 core tickers
    
    Returns: list of dicts [{ticker, sector, etf}, ...]
    """
    # Try CSV first
    if os.path.exists(csv_path):
        logger.info(f"Loading universe from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        if 'ticker' not in df.columns:
            # Try first column as ticker
            df.columns = ['ticker'] + list(df.columns[1:])
        
        if 'sector' not in df.columns:
            if len(df.columns) > 1:
                df.columns = ['ticker', 'sector'] + list(df.columns[2:])
            else:
                df['sector'] = 'Technology'  # fallback
        
        universe = []
        for _, row in df.iterrows():
            ticker = str(row['ticker']).strip()
            sector = str(row['sector']).strip()
            etf = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)
            universe.append({"ticker": ticker, "sector": sector, "etf": etf})
        
        logger.info(f"Loaded {len(universe)} tickers from CSV")
        return universe
    
    # Try Supabase
    try:
        supabase = get_supabase()
        result = supabase.table("scanner_universe").select("*").execute()
        if result.data and len(result.data) > 0:
            logger.info(f"Loaded {len(result.data)} tickers from Supabase scanner_universe")
            universe = []
            for row in result.data:
                ticker = row.get('ticker', '')
                sector = row.get('sector', 'Technology')
                etf = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)
                universe.append({"ticker": ticker, "sector": sector, "etf": etf})
            return universe
    except Exception as e:
        logger.warning(f"Could not load from Supabase: {e}")
    
    # Fallback: ARGUS core tickers
    logger.warning("Using fallback ARGUS-1 core tickers (53)")
    fallback = [
        ("NVDA", "Semiconductors"), ("AMD", "Semiconductors"), ("AVGO", "Semiconductors"),
        ("TSM", "Semiconductors"), ("ASML", "Semiconductors"), ("INTC", "Semiconductors"),
        ("MU", "Semiconductors"), ("QCOM", "Semiconductors"), ("NXPI", "Semiconductors"),
        ("ON", "Semiconductors"), ("KLAC", "Semiconductors"), ("LRCX", "Semiconductors"),
        ("AMAT", "Semiconductors"), ("ARM", "Semiconductors"),
        ("MSFT", "Technology"), ("GOOGL", "Technology"), ("AMZN", "Technology"),
        ("META", "Technology"), ("ORCL", "Technology"),
        ("CRM", "Software"), ("NOW", "Software"), ("ADBE", "Software"),
        ("PLTR", "Software"), ("SHOP", "Software"), ("SNOW", "Software"),
        ("MDB", "Software"), ("NET", "Software"), ("DDOG", "Software"),
        ("ZS", "Software"), ("CRWD", "Software"), ("PANW", "Software"),
        ("CEG", "Utilities"), ("VRT", "Utilities"), ("NRG", "Utilities"),
        ("FSLR", "Clean Energy"), ("JKS", "Clean Energy"),
        ("GS", "Financials"), ("MS", "Financials"),
        ("DLR", "Real Estate"), ("EQIX", "Real Estate"),
        ("TSLA", "Consumer Discretionary"), ("UBER", "Consumer Discretionary"),
        ("SMCI", "Technology"),
    ]
    return [{"ticker": t, "sector": s, "etf": SECTOR_ETF_MAP.get(s, DEFAULT_ETF)} for t, s in fallback]


# ============================================================
# POPULATE: Push universe CSV to Supabase
# ============================================================

def run_populate(csv_path=None):
    """
    Load universe into Supabase scanner_universe table.
    
    Tries in order:
    1. Direct pull from BubbleBackTest Google Sheet (no download needed)
    2. Local CSV file if provided via --csv
    """
    logger.info("=" * 60)
    logger.info("POPULATE SCANNER UNIVERSE")
    logger.info("=" * 60)
    
    df = None
    
    # Option 1: Pull directly from Google Sheets
    if df is None:
        SPREADSHEET_ID = "1WF-AHuukqkzMMLcm9RlRwW4pNrxSOPeckW_n305KPdk"
        SHEET_NAME = "Scanner_Universe"
        export_url = (
            f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
            f"/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"
        )
        
        logger.info(f"Pulling from BubbleBackTest > Scanner_Universe...")
        try:
            response = requests.get(export_url, timeout=30)
            if response.status_code == 200 and len(response.text) > 50:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"  Fetched {len(df)} rows from Google Sheets")
            else:
                logger.warning(f"  Google Sheets returned status {response.status_code}")
                logger.warning("  Sheet may not be publicly accessible.")
                logger.warning("  Fix: Open BubbleBackTest > Share > 'Anyone with the link' > Viewer")
        except Exception as e:
            logger.warning(f"  Could not fetch from Google Sheets: {e}")
    
    # Option 2: Local CSV
    if df is None and csv_path and os.path.exists(csv_path):
        logger.info(f"Falling back to local CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    
    if df is None:
        logger.error("Could not load universe data.")
        logger.error("")
        logger.error("Two options:")
        logger.error("  1. Make BubbleBackTest viewable: Share > 'Anyone with the link' > Viewer")
        logger.error("     Then re-run: python dm_historical_pipeline.py populate")
        logger.error("")
        logger.error("  2. Download Scanner_Universe as CSV and provide it:")
        logger.error("     python dm_historical_pipeline.py populate --csv path/to/Scanner_Universe.csv")
        return
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'ticker' not in df.columns:
        df.columns = ['ticker'] + list(df.columns[1:])
    
    if 'sector' not in df.columns:
        if len(df.columns) > 1:
            col_names = list(df.columns)
            col_names[1] = 'sector'
            df.columns = col_names
        else:
            logger.error("Data must have at least 2 columns: ticker, sector")
            return
    
    # Clean and deduplicate
    df['ticker'] = df['ticker'].astype(str).str.strip()
    df['sector'] = df['sector'].astype(str).str.strip()
    df = df[df['ticker'].str.len() > 0]
    df = df[df['ticker'] != 'nan']
    df = df.drop_duplicates(subset='ticker', keep='first')
    
    logger.info(f"Unique tickers: {len(df)}")
    
    # Show sector distribution
    sector_counts = df['sector'].value_counts()
    logger.info("")
    logger.info("Sector distribution:")
    for sector, count in sector_counts.items():
        etf = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)
        mapped = "" if etf != DEFAULT_ETF else " (unmapped, using SPY)"
        logger.info(f"  {sector}: {count} tickers -> {etf}{mapped}")
    
    # Check for unmapped sectors
    unmapped = [s for s in sector_counts.index if s not in SECTOR_ETF_MAP]
    if unmapped:
        logger.warning(f"\nUnmapped sectors (will use SPY as ETF): {', '.join(unmapped)}")
        logger.warning("Consider adding these to SECTOR_ETF_MAP in the script")
    
    # Push to Supabase
    supabase = get_supabase()
    
    rows = [
        {"ticker": row['ticker'], "sector": row['sector']}
        for _, row in df.iterrows()
    ]
    
    logger.info(f"\nPushing {len(rows)} tickers to Supabase scanner_universe...")
    
    batch_size = 100
    uploaded = 0
    errors = 0
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            supabase.table("scanner_universe").upsert(
                batch, on_conflict="ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.warning(f"Batch error at {i}: {e}")
            # Try one-by-one
            for row in batch:
                try:
                    supabase.table("scanner_universe").upsert(
                        [row], on_conflict="ticker"
                    ).execute()
                    uploaded += 1
                except Exception as e2:
                    logger.warning(f"  Skip {row['ticker']}: {e2}")
                    errors += 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("POPULATE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Uploaded: {uploaded} tickers")
    if errors:
        logger.warning(f"Errors: {errors}")
    logger.info("")
    logger.info("Next step: python dm_historical_pipeline.py fetch")


# ============================================================
# PHASE 1: PRICE FETCHING (RESUMABLE)
# ============================================================

def fetch_ticker_prices(ticker, start_date=START_DATE, end_date=END_DATE):
    """Fetch daily prices from Twelve Data API."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": 5000
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("status") == "error":
        raise Exception(f"API error for {ticker}: {data.get('message')}")
    
    if "values" not in data:
        raise Exception(f"No data returned for {ticker}")
    
    rows = []
    for v in data["values"]:
        try:
            rows.append({
                "date": v["datetime"],
                "ticker": ticker,
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": int(float(v["volume"]))
            })
        except (ValueError, KeyError) as e:
            continue  # skip malformed rows
    
    return rows


def upload_prices_to_supabase(rows):
    """Batch upsert price rows to Supabase."""
    supabase = get_supabase()
    uploaded = 0
    
    for i in range(0, len(rows), SUPABASE_BATCH_SIZE):
        batch = rows[i:i + SUPABASE_BATCH_SIZE]
        try:
            supabase.table("price_history").upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.warning(f"Batch upload error, trying one-by-one: {e}")
            for row in batch:
                try:
                    supabase.table("price_history").upsert(
                        [row], on_conflict="date,ticker"
                    ).execute()
                    uploaded += 1
                except Exception as e2:
                    logger.warning(f"Skip {row['ticker']} {row['date']}: {e2}")
    
    return uploaded


def save_checkpoint(ticker, status, rows_fetched=0, error_msg=None):
    """Save fetch progress to Supabase checkpoint table."""
    supabase = get_supabase()
    try:
        supabase.table("dm_pipeline_checkpoint").upsert({
            "phase": "price_fetch",
            "ticker": ticker,
            "status": status,
            "rows_fetched": rows_fetched,
            "error_msg": error_msg,
            "updated_at": datetime.utcnow().isoformat()
        }, on_conflict="phase,ticker").execute()
    except Exception as e:
        logger.warning(f"Checkpoint save failed for {ticker}: {e}")


def get_completed_tickers():
    """Get list of tickers already fetched (from checkpoint table)."""
    supabase = get_supabase()
    try:
        result = supabase.table("dm_pipeline_checkpoint") \
            .select("ticker") \
            .eq("phase", "price_fetch") \
            .eq("status", "complete") \
            .execute()
        return set(row["ticker"] for row in result.data) if result.data else set()
    except:
        return set()


def get_existing_price_tickers():
    """Get tickers that already have price data in Supabase."""
    supabase = get_supabase()
    try:
        # Use RPC or raw query to get distinct tickers
        result = supabase.rpc("get_distinct_price_tickers").execute()
        if result.data:
            return set(row["ticker"] for row in result.data)
    except:
        pass
    
    # Fallback: check a few known tickers
    try:
        result = supabase.table("price_history") \
            .select("ticker") \
            .limit(1000) \
            .execute()
        return set(row["ticker"] for row in result.data) if result.data else set()
    except:
        return set()


def run_price_fetch(universe, skip_existing=True):
    """
    Fetch prices for all tickers. Resumable via checkpoint table.
    
    Args:
        universe: list of {ticker, sector, etf} dicts
        skip_existing: if True, skip tickers already in price_history
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: PRICE FETCH")
    logger.info("=" * 60)
    
    # Build full ticker list (universe + ETFs + SPY)
    universe_tickers = [u["ticker"] for u in universe]
    all_tickers = list(set(universe_tickers + ALL_ETFS))
    
    # Check what's already done
    completed = get_completed_tickers()
    
    # Optionally check existing price data
    if skip_existing:
        existing = get_existing_price_tickers()
        logger.info(f"Already in price_history: {len(existing)} tickers")
    else:
        existing = set()
    
    # Filter to pending tickers
    pending = [t for t in all_tickers if t not in completed and t not in existing]
    
    logger.info(f"Total tickers: {len(all_tickers)}")
    logger.info(f"Already completed: {len(completed)}")
    logger.info(f"Already in price_history: {len(existing)}")
    logger.info(f"Pending: {len(pending)}")
    logger.info(f"Estimated time: ~{len(pending) * DELAY_BETWEEN_CALLS / 60:.0f} minutes")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info("=" * 60)
    
    if not pending:
        logger.info("All tickers already fetched. Nothing to do.")
        return
    
    total_rows = 0
    errors = []
    api_calls = 0
    
    for i, ticker in enumerate(pending, 1):
        logger.info(f"[{i}/{len(pending)}] Fetching {ticker}...")
        
        try:
            rows = fetch_ticker_prices(ticker)
            logger.info(f"  Got {len(rows)} rows")
            
            uploaded = upload_prices_to_supabase(rows)
            logger.info(f"  Uploaded {uploaded} rows")
            
            save_checkpoint(ticker, "complete", rows_fetched=uploaded)
            total_rows += uploaded
            api_calls += 1
            
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            save_checkpoint(ticker, "error", error_msg=str(e))
            errors.append(ticker)
        
        # Rate limiting
        if i < len(pending):
            time.sleep(DELAY_BETWEEN_CALLS)
        
        # Daily limit check
        if api_calls >= DAILY_CALL_LIMIT - 10:
            logger.warning(f"Approaching daily API limit ({api_calls} calls). Stopping.")
            logger.warning(f"Run again tomorrow to continue from checkpoint.")
            break
        
        # Progress checkpoint every N tickers
        if i % PRICE_FETCH_BATCH_SIZE == 0:
            logger.info(f"--- Checkpoint: {i}/{len(pending)} done, {total_rows} rows total ---")
    
    logger.info("=" * 60)
    logger.info("PRICE FETCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rows uploaded: {total_rows}")
    logger.info(f"API calls used: {api_calls}")
    if errors:
        logger.warning(f"Failed tickers ({len(errors)}): {', '.join(errors)}")
    else:
        logger.info("All tickers successful.")


# ============================================================
# PHASE 2: DM CALCULATION
# ============================================================

def load_all_prices(cache_csv="price_cache.csv"):
    """
    Load all price data into a pandas DataFrame.
    
    Strategy:
    1. If local CSV cache exists and is recent, load from there (fast)
    2. Otherwise, load from Supabase with aggressive pagination
    3. Save to local cache for next time
    
    With 533 tickers x 1260 days = ~670K rows, Supabase pagination
    at 1000 rows/page would need 670 API calls. The CSV cache avoids
    this on subsequent runs.
    """
    # Check for local cache
    if os.path.exists(cache_csv):
        cache_age = time.time() - os.path.getmtime(cache_csv)
        if cache_age < 86400:  # less than 24 hours old
            logger.info(f"Loading from local cache ({cache_csv}, {cache_age/3600:.1f}h old)")
            df = pd.read_csv(cache_csv, parse_dates=['date'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            logger.info(f"  {len(df)} rows, {df['ticker'].nunique()} tickers")
            return df
        else:
            logger.info(f"Local cache is {cache_age/3600:.0f}h old, refreshing from Supabase")
    
    # Load from Supabase
    supabase = get_supabase()
    logger.info("Loading price data from Supabase (this may take a few minutes)...")
    
    all_rows = []
    page_size = 1000
    offset = 0
    
    while True:
        result = supabase.table("price_history") \
            .select("date,ticker,close,volume") \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        
        if not result.data:
            break
        
        all_rows.extend(result.data)
        offset += page_size
        
        if len(result.data) < page_size:
            break
        
        if offset % 50000 == 0:
            logger.info(f"  Loaded {offset} rows...")
    
    logger.info(f"  Total: {len(all_rows)} price rows loaded from Supabase")
    
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['close'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Save local cache for next run
    logger.info(f"  Saving local cache to {cache_csv}")
    df.to_csv(cache_csv, index=False)
    
    return df


def calculate_relative_strength(ticker_return, benchmark_return):
    """
    Calculate relative strength score (0-100).
    50 = equal performance, >50 = outperformance, <50 = underperformance.
    """
    diff = ticker_return - benchmark_return
    score = 50 + (diff * REL_STR_SCALE)
    return max(0, min(100, score))


def calculate_volume_z(vol_5d_avg, vol_baseline_avg):
    """
    Calculate volume z-score scaled to 0-100.
    Matches GS: compares 5-day recent avg to longer baseline.
    Scale: 0.5x = 0, ~1.25x = 50, 2.0x = 100
    """
    if vol_baseline_avg is None or vol_baseline_avg <= 0 or pd.isna(vol_baseline_avg):
        return 50.0  # neutral if no data
    if vol_5d_avg is None or pd.isna(vol_5d_avg):
        return 50.0
    
    ratio = vol_5d_avg / vol_baseline_avg
    # GS formula: (ratio - 0.5) * 66.67
    score = (ratio - 0.5) * 66.67
    return max(0, min(100, score))


def classify_phase(dm_smoothed, etf_dm):
    """Classify phase using FlowOS v3 determinePhase logic.
    
    Matches Google Sheets FlowOS v3:
      STRONG:  dm >= 60 AND etfFlow >= 60
      EARLY:   dm >= 40 AND etfFlow < 40
      BUILD:   dm >= 40 AND etfFlow >= 40 AND etfFlow < 60
      EXHAUST: dm < 40 AND etfFlow >= 60
      WEAK:    dm < 40 (and etfFlow < 60)
      TRANS:   everything else
    """
    if pd.isna(dm_smoothed) or pd.isna(etf_dm):
        return "TRANS"
    if dm_smoothed >= 60 and etf_dm >= 60:
        return "STRONG"
    if dm_smoothed >= 40 and etf_dm < 40:
        return "EARLY"
    if dm_smoothed >= 40 and etf_dm >= 40 and etf_dm < 60:
        return "BUILD"
    if dm_smoothed < 40 and etf_dm >= 60:
        return "EXHAUST"
    if dm_smoothed < 40:
        return "WEAK"
    return "TRANS"


def calculate_dm_for_ticker(ticker_df, spy_df, etf_df, etf_symbol):
    """
    Calculate DM time series for a single ticker.
    
    Args:
        ticker_df: DataFrame with date, close, volume for this ticker
        spy_df: DataFrame with date, close for SPY
        etf_df: DataFrame with date, close, volume for sector ETF
        etf_symbol: ETF symbol string
    
    Returns:
        DataFrame with DM calculations for each date
    """
    if len(ticker_df) < RETURN_PERIOD + 5:
        return pd.DataFrame()  # not enough data
    
    # Align dates
    ticker_df = ticker_df.set_index('date').sort_index()
    spy_df = spy_df.set_index('date').sort_index()
    etf_df = etf_df.set_index('date').sort_index()
    
    # Calculate ticker metrics
    ticker_df['return_20d'] = ticker_df['close'].pct_change(RETURN_PERIOD)
    ticker_df['vol_20d_avg'] = ticker_df['volume'].rolling(VOLUME_AVG_PERIOD).mean()
    ticker_df['vol_ratio'] = ticker_df['volume'] / ticker_df['vol_20d_avg']
    # GS-matching volume: 60-calendar-day window, recent 5 trading days vs rest
    # GS fetches 60 calendar days of volume via GOOGLEFINANCE, splits into:
    #   recent = last 5 trading days, baseline = all remaining (~34 trading days)
    # Vectorized: use searchsorted to find 60-cal-day window start for all dates at once
    ticker_df['vol_5d_avg'] = ticker_df['volume'].rolling(5).mean()
    dates = ticker_df.index
    cutoffs = dates - pd.Timedelta(days=60)
    window_starts = dates.searchsorted(cutoffs, side='left')
    indices = np.arange(len(dates))
    total_in_window = indices + 1 - window_starts
    baseline_counts = np.maximum(total_in_window - 5, 1)
    vol_arr = ticker_df['volume'].values
    vol_cumsum = np.cumsum(vol_arr)
    # Prepend 0 for easy window sum: sum[a:b] = cs[b] - cs[a-1]
    cs = np.concatenate(([0.0], vol_cumsum))
    window_sums = cs[indices + 1] - cs[window_starts]
    recent_sums = cs[indices + 1] - cs[np.maximum(indices - 4, 0)]  # last 5 elements
    baseline_sums = window_sums - recent_sums
    baseline_avg_arr = baseline_sums / baseline_counts
    baseline_avg_arr[:5] = np.nan  # need at least 5 days for recent
    baseline_avg_arr[total_in_window < 10] = np.nan  # need enough baseline
    ticker_df['vol_baseline_avg'] = baseline_avg_arr
    
    # Calculate SPY return
    spy_df['spy_return_20d'] = spy_df['close'].pct_change(RETURN_PERIOD)
    
    # Calculate ETF return and its own DM components
    etf_df['etf_return_20d'] = etf_df['close'].pct_change(RETURN_PERIOD)
    etf_df['etf_vol_20d_avg'] = etf_df['volume'].rolling(VOLUME_AVG_PERIOD).mean()
    etf_df['etf_vol_ratio'] = etf_df['volume'] / etf_df['etf_vol_20d_avg']
    
    # Merge on date
    merged = ticker_df[['close', 'volume', 'return_20d', 'vol_20d_avg', 'vol_ratio', 'vol_5d_avg', 'vol_baseline_avg']].copy()
    merged = merged.join(spy_df[['spy_return_20d']], how='left')
    merged = merged.join(
        etf_df[['etf_return_20d', 'etf_vol_20d_avg', 'etf_vol_ratio']],
        how='left'
    )
    
    # Forward-fill benchmark data for missing dates
    merged['spy_return_20d'] = merged['spy_return_20d'].ffill()
    merged['etf_return_20d'] = merged['etf_return_20d'].ffill()
    merged['etf_vol_ratio'] = merged['etf_vol_ratio'].ffill()
    
    # Drop rows without enough data
    merged = merged.dropna(subset=['return_20d'])
    
    if merged.empty:
        return pd.DataFrame()
    
    # Calculate DM components
    merged['rel_str_etf'] = merged.apply(
        lambda r: calculate_relative_strength(r['return_20d'], r['etf_return_20d'])
        if pd.notna(r['etf_return_20d']) else 50.0,
        axis=1
    )
    
    merged['rel_str_spy'] = merged.apply(
        lambda r: calculate_relative_strength(r['return_20d'], r['spy_return_20d'])
        if pd.notna(r['spy_return_20d']) else 50.0,
        axis=1
    )
    
    merged['volume_z'] = merged.apply(
        lambda r: calculate_volume_z(r['vol_5d_avg'], r['vol_baseline_avg']),
        axis=1
    )
    
    # DM Raw
    merged['dm_raw'] = (
        merged['rel_str_etf'] * W_REL_STR_ETF +
        merged['rel_str_spy'] * W_REL_STR_SPY +
        merged['volume_z'] * W_VOLUME_Z
    )
    merged['dm_raw'] = merged['dm_raw'].clip(0, 100)
    
    # DM Smoothed - NO EMA, matches GS which stores raw DM directly
    merged['dm_smoothed'] = merged['dm_raw'].copy()
    merged['dm_smoothed'] = merged['dm_smoothed'].clip(0, 100)
    
    # ETF's own DM (simplified: use its rel_str vs SPY + volume)
    # ETF doesn't have a "sector ETF" so we use SPY as its benchmark
    etf_merged = etf_df.copy()
    etf_merged = etf_merged.join(spy_df[['spy_return_20d']], how='left')
    etf_merged['spy_return_20d'] = etf_merged['spy_return_20d'].ffill()
    
    if 'etf_return_20d' in etf_merged.columns:
        etf_merged['etf_rel_str_spy'] = etf_merged.apply(
            lambda r: calculate_relative_strength(r['etf_return_20d'], r['spy_return_20d'])
            if pd.notna(r['spy_return_20d']) and pd.notna(r['etf_return_20d']) else 50.0,
            axis=1
        )
        etf_merged['etf_vol_5d_avg'] = etf_merged['volume'].rolling(5).mean()
        etf_dates = etf_merged.index
        etf_cutoffs = etf_dates - pd.Timedelta(days=60)
        etf_window_starts = etf_dates.searchsorted(etf_cutoffs, side='left')
        etf_indices = np.arange(len(etf_dates))
        etf_total_in_window = etf_indices + 1 - etf_window_starts
        etf_baseline_counts = np.maximum(etf_total_in_window - 5, 1)
        etf_vol_arr = etf_merged['volume'].values
        etf_cs = np.concatenate(([0.0], np.cumsum(etf_vol_arr)))
        etf_window_sums = etf_cs[etf_indices + 1] - etf_cs[etf_window_starts]
        etf_recent_sums = etf_cs[etf_indices + 1] - etf_cs[np.maximum(etf_indices - 4, 0)]
        etf_baseline_sums = etf_window_sums - etf_recent_sums
        etf_baseline_avg_arr = etf_baseline_sums / etf_baseline_counts
        etf_baseline_avg_arr[:5] = np.nan
        etf_baseline_avg_arr[etf_total_in_window < 10] = np.nan
        etf_merged['etf_vol_baseline_avg'] = etf_baseline_avg_arr
        etf_merged['etf_vol_z'] = etf_merged.apply(
            lambda r: calculate_volume_z(r['etf_vol_5d_avg'], r['etf_vol_baseline_avg']),
            axis=1
        )
        etf_merged['etf_dm_raw'] = (
            etf_merged['etf_rel_str_spy'] * 0.70 +
            etf_merged['etf_vol_z'] * 0.30
        ).clip(0, 100)
        etf_merged['etf_dm'] = etf_merged['etf_dm_raw'].clip(0, 100)
    else:
        etf_merged['etf_dm'] = 50.0
    
    # Join ETF DM back
    merged = merged.join(etf_merged[['etf_dm']], how='left')
    merged['etf_dm'] = merged['etf_dm'].ffill().fillna(50.0)
    
    # Lead
    merged['lead'] = merged['dm_smoothed'] - merged['etf_dm']
    
    # Phase (FlowOS v3 logic using both DM and ETF flow)
    merged['phase'] = merged.apply(
        lambda r: classify_phase(r['dm_smoothed'], r['etf_dm']), axis=1
    )
    
    # Reset index to get date as column
    merged = merged.reset_index()
    merged.rename(columns={'index': 'date'}, inplace=True)
    
    return merged


def run_dm_calculation(universe):
    """
    Calculate DM for all tickers and store in dm_history.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: DM CALCULATION")
    logger.info("=" * 60)
    
    # Load all price data
    prices_df = load_all_prices()
    
    if prices_df.empty:
        logger.error("No price data found. Run 'fetch' phase first.")
        return
    
    # Separate SPY and ETF data
    spy_df = prices_df[prices_df['ticker'] == 'SPY'].copy()
    logger.info(f"SPY data: {len(spy_df)} rows")
    
    # Build ETF lookup
    etf_data = {}
    for etf in ALL_ETFS:
        etf_df = prices_df[prices_df['ticker'] == etf].copy()
        if not etf_df.empty:
            etf_data[etf] = etf_df
            logger.info(f"  {etf}: {len(etf_df)} rows")
    
    # Build ticker-to-ETF mapping
    ticker_etf_map = {}
    for u in universe:
        ticker_etf_map[u['ticker']] = u['etf']
    
    # Get unique tickers (exclude ETFs and SPY from DM calc)
    etf_set = set(ALL_ETFS)
    ticker_list = [t for t in prices_df['ticker'].unique() if t not in etf_set]
    logger.info(f"Tickers to process: {len(ticker_list)}")
    
    supabase = get_supabase()
    total_rows = 0
    errors = []
    
    for i, ticker in enumerate(ticker_list, 1):
        ticker_df = prices_df[prices_df['ticker'] == ticker].copy()
        
        if len(ticker_df) < RETURN_PERIOD + 10:
            logger.warning(f"  [{i}/{len(ticker_list)}] {ticker}: insufficient data ({len(ticker_df)} rows)")
            continue
        
        # Get sector ETF
        etf_symbol = ticker_etf_map.get(ticker, DEFAULT_ETF)
        etf_df = etf_data.get(etf_symbol, etf_data.get(DEFAULT_ETF, spy_df))
        
        try:
            dm_df = calculate_dm_for_ticker(ticker_df, spy_df, etf_df, etf_symbol)
            
            if dm_df.empty:
                logger.warning(f"  [{i}/{len(ticker_list)}] {ticker}: no DM calculated")
                continue
            
            # Prepare rows for Supabase
            rows = []
            for _, r in dm_df.iterrows():
                rows.append({
                    "ticker": ticker,
                    "date": r['date'].strftime('%Y-%m-%d'),
                    "close": round(float(r['close']), 2) if pd.notna(r['close']) else None,
                    "volume": int(r['volume']) if pd.notna(r['volume']) else None,
                    "return_20d": round(float(r['return_20d']), 6) if pd.notna(r['return_20d']) else None,
                    "etf_return_20d": round(float(r['etf_return_20d']), 6) if pd.notna(r.get('etf_return_20d')) else None,
                    "spy_return_20d": round(float(r['spy_return_20d']), 6) if pd.notna(r.get('spy_return_20d')) else None,
                    "vol_20d_avg": round(float(r['vol_20d_avg']), 2) if pd.notna(r['vol_20d_avg']) else None,
                    "vol_ratio": round(float(r['vol_ratio']), 4) if pd.notna(r['vol_ratio']) else None,
                    "rel_str_etf": round(float(r['rel_str_etf']), 2),
                    "rel_str_spy": round(float(r['rel_str_spy']), 2),
                    "volume_z": round(float(r['volume_z']), 2),
                    "dm_raw": round(float(r['dm_raw']), 2),
                    "dm_smoothed": round(float(r['dm_smoothed']), 2),
                    "etf": etf_symbol,
                    "etf_dm": round(float(r['etf_dm']), 2),
                    "lead": round(float(r['lead']), 2),
                    "phase": r['phase'],
                })
            
            # Batch upload
            uploaded = 0
            for j in range(0, len(rows), SUPABASE_BATCH_SIZE):
                batch = rows[j:j + SUPABASE_BATCH_SIZE]
                try:
                    supabase.table("dm_history").upsert(
                        batch, on_conflict="ticker,date"
                    ).execute()
                    uploaded += len(batch)
                except Exception as e:
                    logger.warning(f"  Batch error for {ticker}: {e}")
                    # Try one-by-one
                    for row in batch:
                        try:
                            supabase.table("dm_history").upsert(
                                [row], on_conflict="ticker,date"
                            ).execute()
                            uploaded += 1
                        except:
                            pass
            
            total_rows += uploaded
            
            if i % 25 == 0 or i == len(ticker_list):
                logger.info(f"  [{i}/{len(ticker_list)}] {ticker}: {uploaded} rows | Total: {total_rows}")
            
        except Exception as e:
            logger.error(f"  [{i}/{len(ticker_list)}] {ticker}: ERROR - {e}")
            errors.append(ticker)
    
    logger.info("=" * 60)
    logger.info("DM CALCULATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rows: {total_rows}")
    if errors:
        logger.warning(f"Errors ({len(errors)}): {', '.join(errors[:20])}")


# ============================================================
# PHASE 3: SUMMARY TABLE (dm_latest)
# ============================================================

def run_summary():
    """Build dm_latest table from dm_history."""
    logger.info("=" * 60)
    logger.info("PHASE 3: BUILD dm_latest SUMMARY")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    
    # Get all unique tickers by paginating
    all_tickers = set()
    offset = 0
    page_size = 1000
    while True:
        result = supabase.table("dm_history") \
            .select("ticker") \
            .order("ticker") \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not result.data:
            break
        for row in result.data:
            all_tickers.add(row['ticker'])
        if len(result.data) < page_size:
            break
        offset += page_size
    
    if not all_tickers:
        logger.error("No dm_history data found.")
        return
    
    tickers = sorted(all_tickers)
    logger.info(f"Building summary for {len(tickers)} tickers")
    
    summaries = []
    
    for ticker in tickers:
        try:
            # Get latest row
            latest = supabase.table("dm_history") \
                .select("*") \
                .eq("ticker", ticker) \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if not latest.data:
                continue
            
            row = latest.data[0]
            
            # Get DM from 7 days ago
            seven_days_ago = (datetime.strptime(row['date'], '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
            hist = supabase.table("dm_history") \
                .select("dm_smoothed,date") \
                .eq("ticker", ticker) \
                .gte("date", seven_days_ago) \
                .order("date") \
                .limit(1) \
                .execute()
            
            dm_7d_ago = hist.data[0]['dm_smoothed'] if hist.data else None
            dm_change = (row['dm_smoothed'] - dm_7d_ago) if dm_7d_ago else None
            
            summaries.append({
                "ticker": ticker,
                "date": row['date'],
                "dm_smoothed": row['dm_smoothed'],
                "dm_raw": row['dm_raw'],
                "lead": row['lead'],
                "phase": row['phase'],
                "etf": row['etf'],
                "etf_dm": row['etf_dm'],
                "rel_str_etf": row['rel_str_etf'],
                "rel_str_spy": row['rel_str_spy'],
                "volume_z": row['volume_z'],
                "close": row['close'],
                "dm_7d_ago": dm_7d_ago,
                "dm_change": round(dm_change, 2) if dm_change else None,
                "updated_at": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.warning(f"  {ticker}: {e}")
    
    # Batch upsert summaries
    if summaries:
        for i in range(0, len(summaries), SUPABASE_BATCH_SIZE):
            batch = summaries[i:i + SUPABASE_BATCH_SIZE]
            try:
                supabase.table("dm_latest").upsert(
                    batch, on_conflict="ticker"
                ).execute()
            except Exception as e:
                logger.error(f"Summary upsert error: {e}")
    
    logger.info(f"dm_latest updated: {len(summaries)} tickers")
    
    # Print top scores
    top = sorted(summaries, key=lambda x: x['dm_smoothed'] or 0, reverse=True)[:15]
    logger.info("")
    logger.info("TOP 15 BY DM:")
    logger.info(f"{'Ticker':<8} {'DM':>6} {'Lead':>7} {'Phase':<10} {'ETF':<5} {'7d Chg':>7}")
    logger.info("-" * 50)
    for s in top:
        chg = f"{s['dm_change']:+.1f}" if s['dm_change'] else "  n/a"
        logger.info(
            f"{s['ticker']:<8} {s['dm_smoothed']:6.1f} {s['lead']:+7.1f} "
            f"{s['phase']:<10} {s['etf']:<5} {chg:>7}"
        )


# ============================================================
# PHASE 4: VALIDATION
# ============================================================

def run_validation():
    """Spot-check DM calculations against known values."""
    logger.info("=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    
    # Check row counts
    result = supabase.table("dm_history").select("ticker", count="exact").execute()
    total_rows = result.count if result.count else 0
    logger.info(f"Total dm_history rows: {total_rows}")
    
    # Count per ticker
    result = supabase.table("dm_history") \
        .select("ticker") \
        .limit(10000) \
        .execute()
    
    if result.data:
        from collections import Counter
        counts = Counter(row['ticker'] for row in result.data)
        logger.info(f"Unique tickers: {len(counts)}")
        
        # Min/max rows per ticker
        min_ticker = min(counts, key=counts.get)
        max_ticker = max(counts, key=counts.get)
        logger.info(f"Min rows: {min_ticker} ({counts[min_ticker]})")
        logger.info(f"Max rows: {max_ticker} ({counts[max_ticker]})")
        avg_rows = sum(counts.values()) / len(counts)
        logger.info(f"Avg rows per ticker: {avg_rows:.0f}")
    
    # Check date coverage
    result = supabase.table("dm_history") \
        .select("date") \
        .order("date") \
        .limit(1) \
        .execute()
    if result.data:
        logger.info(f"Earliest date: {result.data[0]['date']}")
    
    result = supabase.table("dm_history") \
        .select("date") \
        .order("date", desc=True) \
        .limit(1) \
        .execute()
    if result.data:
        logger.info(f"Latest date: {result.data[0]['date']}")
    
    # Phase distribution (latest date)
    if result.data:
        latest_date = result.data[0]['date']
        phases = supabase.table("dm_history") \
            .select("phase") \
            .eq("date", latest_date) \
            .execute()
        
        if phases.data:
            from collections import Counter
            phase_counts = Counter(row['phase'] for row in phases.data)
            logger.info(f"\nPhase distribution on {latest_date}:")
            for phase, count in sorted(phase_counts.items()):
                logger.info(f"  {phase}: {count}")
    
    # Spot check known tickers
    spot_checks = ["NVDA", "POWL", "APP", "CCJ", "RCL"]
    logger.info("\nSpot Checks:")
    logger.info(f"{'Ticker':<8} {'Date':>12} {'DM':>6} {'Lead':>7} {'Phase':<10}")
    logger.info("-" * 50)
    
    for ticker in spot_checks:
        result = supabase.table("dm_history") \
            .select("*") \
            .eq("ticker", ticker) \
            .order("date", desc=True) \
            .limit(1) \
            .execute()
        
        if result.data:
            r = result.data[0]
            logger.info(
                f"{r['ticker']:<8} {r['date']:>12} {r['dm_smoothed']:6.1f} "
                f"{r['lead']:+7.1f} {r['phase']:<10}"
            )
        else:
            logger.info(f"{ticker:<8} NO DATA")


# ============================================================
# DAILY UPDATE (for ongoing use)
# ============================================================

def load_recent_prices(lookback_days=120, cache_csv="price_cache.csv"):
    """
    Load only recent price data for daily calculation.
    Uses local cache if available, otherwise fetches from Supabase.
    Only loads last ~120 days (enough for 60-cal-day volume + 20-day returns).
    """
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Try local cache first — filter to recent dates
    if os.path.exists(cache_csv):
        cache_age = time.time() - os.path.getmtime(cache_csv)
        if cache_age < 86400:
            logger.info(f"Loading from local cache (filtering to >={start_date})")
            df = pd.read_csv(cache_csv, parse_dates=['date'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df[df['date'] >= start_date]
            logger.info(f"  {len(df)} rows, {df['ticker'].nunique()} tickers (from cache)")
            return df
    
    # Load from Supabase with date filter
    supabase = get_supabase()
    logger.info(f"Loading recent prices from Supabase (>= {start_date})...")
    
    all_rows = []
    page_size = 1000
    offset = 0
    
    while True:
        result = supabase.table("price_history") \
            .select("date,ticker,close,volume") \
            .gte("date", start_date) \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        
        if not result.data:
            break
        
        all_rows.extend(result.data)
        offset += page_size
        
        if len(result.data) < page_size:
            break
    
    logger.info(f"  {len(all_rows)} rows loaded from Supabase")
    
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['close'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    return df


def get_latest_dm_date():
    """Get the most recent date in dm_history."""
    supabase = get_supabase()
    result = supabase.table("dm_history") \
        .select("date") \
        .order("date", desc=True) \
        .limit(1) \
        .execute()
    
    if result.data:
        return result.data[0]['date']
    return None


def run_daily_calculation(universe):
    """
    Optimized daily DM calculation.
    Only computes DM for NEW dates not already in dm_history.
    Loads only ~120 days of prices instead of full 6-year history.
    """
    logger.info("=" * 60)
    logger.info("DAILY DM CALCULATION (INCREMENTAL)")
    logger.info("=" * 60)
    
    # Find latest date already calculated
    latest_date = get_latest_dm_date()
    logger.info(f"Latest date in dm_history: {latest_date}")
    
    # Load only recent prices (120 days = enough for lookback)
    prices_df = load_recent_prices(lookback_days=120)
    
    if prices_df.empty:
        logger.error("No recent price data found.")
        return
    
    # Find new dates to calculate
    all_price_dates = sorted(prices_df['date'].dt.strftime('%Y-%m-%d').unique())
    if latest_date:
        new_dates = [d for d in all_price_dates if d > latest_date]
    else:
        new_dates = all_price_dates
    
    if not new_dates:
        logger.info("No new dates to calculate. dm_history is up to date.")
        return
    
    logger.info(f"New dates to calculate: {len(new_dates)} ({new_dates[0]} to {new_dates[-1]})")
    
    # Separate SPY and ETF data
    spy_df = prices_df[prices_df['ticker'] == 'SPY'].copy()
    logger.info(f"SPY data: {len(spy_df)} rows")
    
    etf_data = {}
    for etf in ALL_ETFS:
        edf = prices_df[prices_df['ticker'] == etf].copy()
        if not edf.empty:
            etf_data[etf] = edf
    
    # Build ticker-to-ETF mapping
    ticker_etf_map = {}
    for u in universe:
        ticker_etf_map[u['ticker']] = u['etf']
    
    # Get unique tickers (exclude ETFs)
    etf_set = set(ALL_ETFS)
    ticker_list = [t for t in prices_df['ticker'].unique() if t not in etf_set]
    logger.info(f"Tickers to process: {len(ticker_list)}")
    
    supabase = get_supabase()
    total_rows = 0
    errors = []
    
    for i, ticker in enumerate(ticker_list, 1):
        ticker_df = prices_df[prices_df['ticker'] == ticker].copy()
        
        if len(ticker_df) < RETURN_PERIOD + 10:
            continue
        
        etf_symbol = ticker_etf_map.get(ticker, DEFAULT_ETF)
        etf_df = etf_data.get(etf_symbol, etf_data.get(DEFAULT_ETF, spy_df))
        
        try:
            dm_df = calculate_dm_for_ticker(ticker_df, spy_df, etf_df, etf_symbol)
            
            if dm_df.empty:
                continue
            
            # Filter to only NEW dates
            dm_df['date_str'] = dm_df['date'].dt.strftime('%Y-%m-%d')
            new_dm_df = dm_df[dm_df['date_str'].isin(new_dates)]
            
            if new_dm_df.empty:
                continue
            
            # Prepare rows for upload
            rows = []
            for _, r in new_dm_df.iterrows():
                rows.append({
                    "ticker": ticker,
                    "date": r['date'].strftime('%Y-%m-%d'),
                    "close": round(float(r['close']), 2) if pd.notna(r['close']) else None,
                    "volume": int(r['volume']) if pd.notna(r['volume']) else None,
                    "return_20d": round(float(r['return_20d']), 6) if pd.notna(r['return_20d']) else None,
                    "etf_return_20d": round(float(r['etf_return_20d']), 6) if pd.notna(r.get('etf_return_20d')) else None,
                    "spy_return_20d": round(float(r['spy_return_20d']), 6) if pd.notna(r.get('spy_return_20d')) else None,
                    "vol_20d_avg": round(float(r['vol_20d_avg']), 2) if pd.notna(r['vol_20d_avg']) else None,
                    "vol_ratio": round(float(r['vol_ratio']), 4) if pd.notna(r['vol_ratio']) else None,
                    "rel_str_etf": round(float(r['rel_str_etf']), 2),
                    "rel_str_spy": round(float(r['rel_str_spy']), 2),
                    "volume_z": round(float(r['volume_z']), 2),
                    "dm_raw": round(float(r['dm_raw']), 2),
                    "dm_smoothed": round(float(r['dm_smoothed']), 2),
                    "etf": etf_symbol,
                    "etf_dm": round(float(r['etf_dm']), 2),
                    "lead": round(float(r['lead']), 2),
                    "phase": r['phase'],
                })
            
            # Batch upload only new rows
            uploaded = 0
            for j in range(0, len(rows), SUPABASE_BATCH_SIZE):
                batch = rows[j:j + SUPABASE_BATCH_SIZE]
                try:
                    supabase.table("dm_history").upsert(
                        batch, on_conflict="ticker,date"
                    ).execute()
                    uploaded += len(batch)
                except Exception as e:
                    logger.warning(f"  Batch error for {ticker}: {e}")
                    for row in batch:
                        try:
                            supabase.table("dm_history").upsert(
                                [row], on_conflict="ticker,date"
                            ).execute()
                            uploaded += 1
                        except:
                            pass
            
            total_rows += uploaded
            
            if i % 50 == 0 or i == len(ticker_list):
                logger.info(f"  [{i}/{len(ticker_list)}] Total new rows: {total_rows}")
            
        except Exception as e:
            logger.error(f"  [{i}/{len(ticker_list)}] {ticker}: ERROR - {e}")
            errors.append(ticker)
    
    logger.info("=" * 60)
    logger.info("DAILY DM CALCULATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"New rows uploaded: {total_rows}")
    logger.info(f"Dates added: {', '.join(new_dates)}")
    if errors:
        logger.warning(f"Errors ({len(errors)}): {', '.join(errors[:20])}")


def run_daily_summary():
    """
    Optimized dm_latest update.
    Uses batch queries instead of 1 query per ticker.
    """
    logger.info("=" * 60)
    logger.info("DAILY dm_latest UPDATE (OPTIMIZED)")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    
    # Get latest date in dm_history
    latest_date = get_latest_dm_date()
    if not latest_date:
        logger.error("No dm_history data found.")
        return
    
    logger.info(f"Latest date: {latest_date}")
    
    # Get 7d-ago date (use 10 calendar days to account for weekends)
    date_obj = datetime.strptime(latest_date, '%Y-%m-%d')
    date_7d_ago = (date_obj - timedelta(days=10)).strftime('%Y-%m-%d')
    
    # Batch fetch: all rows for the latest date
    latest_rows = []
    offset = 0
    page_size = 1000
    while True:
        result = supabase.table("dm_history") \
            .select("*") \
            .eq("date", latest_date) \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not result.data:
            break
        latest_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
    
    logger.info(f"Latest date rows: {len(latest_rows)}")
    
    # Batch fetch: all rows for the 7d-ago date window
    # Get the closest trading date on or after date_7d_ago
    hist_rows = []
    offset = 0
    while True:
        result = supabase.table("dm_history") \
            .select("ticker,dm_smoothed,date") \
            .gte("date", date_7d_ago) \
            .lt("date", latest_date) \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not result.data:
            break
        hist_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
    
    # Build lookup: ticker -> earliest DM in the 7d window (closest to 7d ago)
    dm_7d_lookup = {}
    for row in hist_rows:
        ticker = row['ticker']
        if ticker not in dm_7d_lookup:
            dm_7d_lookup[ticker] = float(row['dm_smoothed'])
    
    logger.info(f"7d-ago lookup: {len(dm_7d_lookup)} tickers")
    
    # Build summaries
    summaries = []
    for row in latest_rows:
        ticker = row['ticker']
        dm_7d_ago = dm_7d_lookup.get(ticker)
        dm_change = round(float(row['dm_smoothed']) - dm_7d_ago, 2) if dm_7d_ago is not None else None
        
        summaries.append({
            "ticker": ticker,
            "date": row['date'],
            "dm_smoothed": row['dm_smoothed'],
            "dm_raw": row['dm_raw'],
            "lead": row['lead'],
            "phase": row['phase'],
            "etf": row['etf'],
            "etf_dm": row['etf_dm'],
            "rel_str_etf": row['rel_str_etf'],
            "rel_str_spy": row['rel_str_spy'],
            "volume_z": row['volume_z'],
            "close": row['close'],
            "dm_7d_ago": dm_7d_ago,
            "dm_change": dm_change,
            "updated_at": datetime.utcnow().isoformat()
        })
    
    # Batch upsert
    if summaries:
        for i in range(0, len(summaries), SUPABASE_BATCH_SIZE):
            batch = summaries[i:i + SUPABASE_BATCH_SIZE]
            try:
                supabase.table("dm_latest").upsert(
                    batch, on_conflict="ticker"
                ).execute()
            except Exception as e:
                logger.error(f"Summary upsert error: {e}")
    
    logger.info(f"dm_latest updated: {len(summaries)} tickers")
    
    # Print top scores
    top = sorted(summaries, key=lambda x: x['dm_smoothed'], reverse=True)[:15]
    logger.info("")
    logger.info("TOP 15 BY DM:")
    logger.info(f"{'Ticker':<8} {'DM':>7} {'Lead':>7} {'Phase':<10} {'ETF':<6} {'7d Chg':>7}")
    logger.info("-" * 50)
    for r in top:
        chg = f"{r['dm_change']:+.1f}" if r['dm_change'] is not None else "N/A"
        logger.info(
            f"{r['ticker']:<8} {r['dm_smoothed']:>7.1f} "
            f"{r['lead']:+7.1f} {r['phase']:<10} {r['etf']:<6} {chg:>7}"
        )


def run_daily_update(universe):
    """
    Optimized daily update pipeline.
    Phase 1: Fetch last 5 days of prices (~69 min, rate-limited)
    Phase 2: Calculate DM for NEW dates only (~30 sec vs 15 min)
    Phase 3: Update dm_latest with batch queries (~5 sec vs 10 min)
    """
    logger.info("=" * 60)
    logger.info("DAILY UPDATE (OPTIMIZED)")
    logger.info("=" * 60)
    
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    
    # Phase 1: Fetch fresh prices
    all_tickers = list(set([u['ticker'] for u in universe] + ALL_ETFS))
    
    logger.info(f"Phase 1: Fetching prices for {len(all_tickers)} tickers ({yesterday} to {today})")
    
    api_calls = 0
    for i, ticker in enumerate(all_tickers, 1):
        try:
            rows = fetch_ticker_prices(ticker, start_date=yesterday, end_date=today)
            if rows:
                upload_prices_to_supabase(rows)
            api_calls += 1
        except Exception as e:
            logger.warning(f"  {ticker}: {e}")
        
        if i < len(all_tickers):
            time.sleep(DELAY_BETWEEN_CALLS)
        
        if api_calls >= DAILY_CALL_LIMIT - 10:
            logger.warning("Approaching daily API limit. Stopping price fetch.")
            break
    
    logger.info(f"Phase 1 complete: {api_calls} API calls")
    
    # Invalidate local cache so Phase 2 picks up new prices
    cache_csv = "price_cache.csv"
    if os.path.exists(cache_csv):
        os.remove(cache_csv)
        logger.info("Local price cache invalidated")
    
    # Phase 2: Incremental DM calculation
    run_daily_calculation(universe)
    
    # Phase 3: Update dm_latest
    run_daily_summary()
    
    logger.info("=" * 60)
    logger.info("DAILY UPDATE COMPLETE")
    logger.info("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DM Historical Pipeline")
    parser.add_argument("command", choices=[
        "populate", "fetch", "calculate", "summary", "validate", "daily", "all"
    ], help="Pipeline phase to run")
    parser.add_argument("--csv", default="universe.csv",
                       help="Path to universe CSV (ticker,sector)")
    parser.add_argument("--no-skip", action="store_true",
                       help="Don't skip tickers already in price_history")
    parser.add_argument("--refresh-cache", action="store_true",
                       help="Force reload prices from Supabase (ignore local cache)")
    
    args = parser.parse_args()
    
    # Validate environment
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    
    if args.command in ("fetch", "daily", "all") and not TWELVEDATA_API_KEY:
        logger.error("TWELVEDATA_API_KEY must be set in .env for price fetching")
        sys.exit(1)
    
    # Populate doesn't need universe loaded (it creates the universe)
    if args.command == "populate":
        run_populate(args.csv)
        return
    
    # Load universe
    universe = load_universe(args.csv)
    logger.info(f"Universe: {len(universe)} tickers")
    
    # Run requested phase
    if args.command == "fetch":
        run_price_fetch(universe, skip_existing=not args.no_skip)
    
    elif args.command == "calculate":
        if args.refresh_cache and os.path.exists("price_cache.csv"):
            os.remove("price_cache.csv")
        run_dm_calculation(universe)
    
    elif args.command == "summary":
        run_daily_summary()
    
    elif args.command == "validate":
        run_validation()
    
    elif args.command == "daily":
        run_daily_update(universe)
    
    elif args.command == "all":
        logger.info("Running full pipeline...")
        run_price_fetch(universe, skip_existing=not args.no_skip)
        run_dm_calculation(universe)
        run_daily_summary()
        run_validation()


if __name__ == "__main__":
    main()