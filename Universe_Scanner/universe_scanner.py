"""
universe_scanner.py
FlowOS Universe Scanner - Phase 1: Liquidity Filter

Discovers all US equities meeting liquidity criteria:
  - Common stocks (CS) and ADRs (ADRC)
  - Market cap > $1B
  - Average daily dollar volume > $5M (20-day)
  - Listed on NYSE, NASDAQ, or AMEX (no OTC/pink sheets)

Uses Polygon API for all data. Writes results to Supabase universe_tickers table.

Usage:
  python universe_scanner.py filter           # Run liquidity filter, update universe_tickers
  python universe_scanner.py filter --dry-run  # Preview results without writing to Supabase
  python universe_scanner.py status            # Show current universe_tickers stats
  python universe_scanner.py compare           # Compare universe_tickers vs scanner_universe

Pipeline: Weekly run (Saturday/Sunday) before DM calculation.
"""

import os
import sys
import time
import logging
import argparse
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Liquidity filter thresholds (from spec)
MIN_MARKET_CAP = 1_000_000_000       # $1B
MIN_AVG_DOLLAR_VOLUME = 5_000_000    # $5M daily
VOLUME_LOOKBACK_DAYS = 20            # 20 trading days

# Valid exchanges (no OTC/pink sheets)
VALID_EXCHANGES = {"XNYS", "XNAS", "XASE", "XNMS", "XNCM", "XNGS"}
# XNYS = NYSE, XNAS/XNMS/XNCM/XNGS = NASDAQ variants, XASE = AMEX

POLYGON_DELAY = 0.2   # 200ms courtesy delay

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# SUPABASE CLIENT
# ============================================================

def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# STEP 1: Get all active US common stock tickers from Polygon
# ============================================================

def fetch_all_tickers():
    """
    Fetch all active US stock tickers from Polygon reference endpoint.
    Includes both Common Stock (CS) and ADRs (ADRC) since several
    Preferred Universe tickers are ADRs (TSM, PDD).
    Paginates through all results. Returns list of dicts with ticker info.
    """
    logger.info("STEP 1: Fetching all active US stock tickers from Polygon...")
    
    TICKER_TYPES = ["CS", "ADRC"]   # Common Stock + ADRs
    all_tickers = []
    
    for ticker_type in TICKER_TYPES:
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "market": "stocks",
            "type": ticker_type,
            "active": "true",
            "limit": 1000,
            "apiKey": POLYGON_API_KEY
        }
        
        type_count = 0
        page = 1
        
        while True:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") == "ERROR" or "results" not in data:
                logger.error(f"Polygon tickers API error ({ticker_type}): {data.get('error', data.get('message', 'unknown'))}")
                break
            
            results = data["results"]
            all_tickers.extend(results)
            type_count += len(results)
            
            if page % 5 == 0:
                logger.info(f"  [{ticker_type}] Page {page}: {type_count} tickers so far...")
            
            next_url = data.get("next_url")
            if not next_url:
                break
            
            url = next_url
            params = {"apiKey": POLYGON_API_KEY}
            page += 1
            time.sleep(POLYGON_DELAY)
        
        logger.info(f"  {ticker_type}: {type_count} tickers")
    
    logger.info(f"  Total active tickers (CS + ADRC): {len(all_tickers)}")
    return all_tickers


# ============================================================
# STEP 2: Filter by exchange
# ============================================================

def filter_by_exchange(tickers):
    """Filter to NYSE, NASDAQ, AMEX only."""
    logger.info("STEP 2: Filtering by exchange (NYSE/NASDAQ/AMEX)...")
    
    filtered = []
    for t in tickers:
        exchange = t.get("primary_exchange", "")
        if exchange in VALID_EXCHANGES:
            filtered.append(t)
    
    logger.info(f"  {len(tickers)} -> {len(filtered)} (removed {len(tickers) - len(filtered)} non-major exchange)")
    return filtered


# ============================================================
# STEP 3: Calculate average dollar volume using grouped daily
# ============================================================

def fetch_grouped_daily(date_str):
    """Fetch all US stock prices for a single date from Polygon."""
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    params = {"adjusted": "true", "apiKey": POLYGON_API_KEY}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("resultsCount", 0) == 0:
        return {}
    
    # Return dict of ticker -> {close, volume}
    result = {}
    for r in data.get("results", []):
        ticker = r.get("T", "")
        if ticker:
            result[ticker] = {
                "close": float(r.get("c", 0)),
                "volume": int(r.get("v", 0))
            }
    return result


def get_recent_trading_days(n_days=20):
    """Get last n trading days (Mon-Fri, skip today)."""
    dates = []
    d = datetime.now()
    while len(dates) < n_days:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
    return sorted(dates)


def calculate_dollar_volumes(ticker_set):
    """
    Calculate 20-day average dollar volume for all tickers.
    Uses grouped daily endpoint (1 API call per date = 20 calls total).
    Returns dict of ticker -> avg_dollar_volume.
    """
    logger.info("STEP 3: Calculating 20-day average dollar volume...")
    
    dates = get_recent_trading_days(VOLUME_LOOKBACK_DAYS)
    logger.info(f"  Fetching {len(dates)} trading days: {dates[0]} to {dates[-1]}")
    
    # Accumulate dollar volume per ticker across days
    ticker_dv_sum = {}    # ticker -> total dollar volume
    ticker_dv_count = {}  # ticker -> number of days with data
    
    for i, date_str in enumerate(dates):
        daily = fetch_grouped_daily(date_str)
        
        for ticker in ticker_set:
            if ticker in daily:
                dv = daily[ticker]["close"] * daily[ticker]["volume"]
                ticker_dv_sum[ticker] = ticker_dv_sum.get(ticker, 0) + dv
                ticker_dv_count[ticker] = ticker_dv_count.get(ticker, 0) + 1
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {i + 1}/{len(dates)} dates")
        
        time.sleep(POLYGON_DELAY)
    
    # Calculate averages
    avg_dv = {}
    for ticker in ticker_dv_sum:
        count = ticker_dv_count.get(ticker, 1)
        if count >= 10:  # Need at least 10 of 20 days
            avg_dv[ticker] = ticker_dv_sum[ticker] / count
    
    logger.info(f"  Calculated dollar volume for {len(avg_dv)} tickers")
    return avg_dv


def filter_by_dollar_volume(tickers, avg_dv):
    """Filter tickers by minimum average daily dollar volume."""
    logger.info(f"  Filtering by avg dollar volume >= ${MIN_AVG_DOLLAR_VOLUME / 1e6:.0f}M...")
    
    filtered = []
    for t in tickers:
        ticker = t.get("ticker", "")
        dv = avg_dv.get(ticker, 0)
        if dv >= MIN_AVG_DOLLAR_VOLUME:
            t["avg_dollar_volume"] = round(dv, 2)
            filtered.append(t)
    
    logger.info(f"  {len(tickers)} -> {len(filtered)} (removed {len(tickers) - len(filtered)} low volume)")
    return filtered


# ============================================================
# STEP 4: Get market cap from ticker details
# ============================================================

def fetch_market_caps(tickers):
    """
    Fetch market cap for each ticker via Polygon ticker details endpoint.
    This is the expensive step - one API call per ticker.
    """
    logger.info(f"STEP 4: Fetching market cap for {len(tickers)} tickers...")
    
    enriched = []
    skipped = 0
    
    for i, t in enumerate(tickers):
        ticker = t.get("ticker", "")
        
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        params = {"apiKey": POLYGON_API_KEY}
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") == "OK" and "results" in data:
                details = data["results"]
                market_cap = details.get("market_cap")
                
                if market_cap and market_cap >= MIN_MARKET_CAP:
                    t["market_cap"] = market_cap
                    t["sic_code"] = details.get("sic_code", "")
                    t["sic_description"] = details.get("sic_description", "")
                    enriched.append(t)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(tickers)} ({len(enriched)} passed so far, {skipped} skipped)")
        
        time.sleep(POLYGON_DELAY)
    
    logger.info(f"  {len(tickers)} -> {len(enriched)} (market cap >= ${MIN_MARKET_CAP / 1e9:.0f}B)")
    return enriched


# ============================================================
# STEP 5: Write to Supabase universe_tickers
# ============================================================

def write_to_supabase(tickers, dry_run=False):
    """Write filtered tickers to universe_tickers table in Supabase."""
    logger.info(f"STEP 5: Writing {len(tickers)} tickers to Supabase...")
    
    if dry_run:
        logger.info("  DRY RUN - skipping Supabase write")
        return
    
    supabase = get_supabase()
    today = datetime.now().strftime("%Y-%m-%d")
    
    rows = []
    for t in tickers:
        rows.append({
            "ticker": t.get("ticker", ""),
            "name": t.get("name", ""),
            "exchange": t.get("primary_exchange", ""),
            "market_cap": int(t.get("market_cap", 0)),
            "avg_dollar_volume": round(t.get("avg_dollar_volume", 0), 2),
            "sic_code": t.get("sic_code", ""),
            "sic_description": t.get("sic_description", ""),
            "last_seen": today,
            "first_seen": today,   # Will be preserved on conflict
        })
    
    # Upsert in batches
    BATCH = 500
    uploaded = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        try:
            supabase.table("universe_tickers").upsert(
                batch,
                on_conflict="ticker"
            ).execute()
            uploaded += len(batch)
            logger.info(f"  Upserted {uploaded}/{len(rows)}")
        except Exception as e:
            logger.error(f"  Upsert error at batch {i}: {e}")
    
    logger.info(f"  Done: {uploaded} tickers written to universe_tickers")
    return uploaded


# ============================================================
# STEP 6: Summary and validation
# ============================================================

def print_summary(tickers):
    """Print summary stats for the filtered universe."""
    logger.info("=" * 60)
    logger.info("UNIVERSE SCANNER RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Total tickers passing filter: {len(tickers)}")
    
    # Exchange breakdown
    exchanges = {}
    for t in tickers:
        ex = t.get("primary_exchange", "unknown")
        exchanges[ex] = exchanges.get(ex, 0) + 1
    logger.info("Exchange breakdown:")
    for ex, count in sorted(exchanges.items(), key=lambda x: -x[1]):
        logger.info(f"  {ex}: {count}")
    
    # Market cap ranges
    caps = [t.get("market_cap", 0) for t in tickers]
    if caps:
        logger.info(f"Market cap range: ${min(caps)/1e9:.1f}B - ${max(caps)/1e9:.1f}B")
        mega = sum(1 for c in caps if c >= 200e9)
        large = sum(1 for c in caps if 10e9 <= c < 200e9)
        mid = sum(1 for c in caps if 1e9 <= c < 10e9)
        logger.info(f"  Mega (>$200B): {mega}")
        logger.info(f"  Large ($10-200B): {large}")
        logger.info(f"  Mid ($1-10B): {mid}")
    
    # Dollar volume ranges
    dvols = [t.get("avg_dollar_volume", 0) for t in tickers]
    if dvols:
        logger.info(f"Avg dollar volume range: ${min(dvols)/1e6:.1f}M - ${max(dvols)/1e6:.0f}M")


def check_preferred_overlap(tickers):
    """Check how many of the 28 Preferred tickers are in the universe."""
    PREFERRED_28 = [
        "CEG", "CRWD", "TSM", "APP", "VRT", "MU", "NVDA", "CCJ",
        "DNN", "PLTR", "TSLA", "TTD", "MSTR", "UEC", "LEU", "HAL",
        "WDC", "ENPH", "UUUU", "PDD", "NCLH", "FCX", "RCL", "LVS",
        "PSKY", "MRNA", "WYNN", "SMR"
    ]
    
    universe_tickers = {t.get("ticker", "") for t in tickers}
    
    found = [t for t in PREFERRED_28 if t in universe_tickers]
    missing = [t for t in PREFERRED_28 if t not in universe_tickers]
    
    logger.info(f"\nPreferred 28 overlap: {len(found)}/28 found in universe")
    if missing:
        logger.info(f"  Missing: {', '.join(missing)}")
        logger.info(f"  (These may fail liquidity filter - small cap or low volume)")


# ============================================================
# STATUS COMMAND
# ============================================================

def show_status():
    """Show current universe_tickers stats from Supabase."""
    supabase = get_supabase()
    
    try:
        result = supabase.table("universe_tickers").select("ticker,exchange,market_cap,avg_dollar_volume,last_seen", count="exact").execute()
        
        if not result.data:
            logger.info("universe_tickers table is empty or doesn't exist.")
            return
        
        logger.info(f"universe_tickers: {result.count} tickers")
        
        # Last seen date
        dates = set(r.get("last_seen", "") for r in result.data)
        logger.info(f"Last seen dates: {sorted(dates)}")
        
        # Exchange breakdown
        exchanges = {}
        for r in result.data:
            ex = r.get("exchange", "")
            exchanges[ex] = exchanges.get(ex, 0) + 1
        for ex, count in sorted(exchanges.items(), key=lambda x: -x[1]):
            logger.info(f"  {ex}: {count}")
            
    except Exception as e:
        logger.error(f"Error reading universe_tickers: {e}")


# ============================================================
# COMPARE COMMAND
# ============================================================

def compare_universes():
    """Compare universe_tickers with scanner_universe."""
    supabase = get_supabase()
    
    try:
        uni = supabase.table("universe_tickers").select("ticker").execute()
        scan = supabase.table("scanner_universe").select("ticker").execute()
        
        uni_set = {r["ticker"] for r in uni.data} if uni.data else set()
        scan_set = {r["ticker"] for r in scan.data} if scan.data else set()
        
        logger.info(f"universe_tickers: {len(uni_set)} tickers")
        logger.info(f"scanner_universe: {len(scan_set)} tickers")
        logger.info(f"In both: {len(uni_set & scan_set)}")
        logger.info(f"Only in universe_tickers: {len(uni_set - scan_set)}")
        logger.info(f"Only in scanner_universe: {len(scan_set - uni_set)}")
        
        only_scanner = sorted(scan_set - uni_set)
        if only_scanner:
            logger.info(f"\nIn scanner_universe but NOT in universe_tickers:")
            logger.info(f"  {', '.join(only_scanner[:30])}")
            if len(only_scanner) > 30:
                logger.info(f"  ...and {len(only_scanner) - 30} more")
                
    except Exception as e:
        logger.error(f"Error: {e}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_filter(dry_run=False):
    """Run the full liquidity filter pipeline."""
    logger.info("=" * 60)
    logger.info("FLOWOS UNIVERSE SCANNER - PHASE 1: LIQUIDITY FILTER")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Thresholds: Market cap >= ${MIN_MARKET_CAP/1e9:.0f}B, "
                f"Avg dollar volume >= ${MIN_AVG_DOLLAR_VOLUME/1e6:.0f}M")
    logger.info("=" * 60)
    
    # Step 1: Get all active common stocks
    all_tickers = fetch_all_tickers()
    if not all_tickers:
        logger.error("No tickers returned from Polygon. Check API key.")
        return
    
    # Step 2: Filter by exchange
    exchange_filtered = filter_by_exchange(all_tickers)
    
    # Step 3: Calculate dollar volume (cheap - 20 API calls)
    ticker_set = {t.get("ticker", "") for t in exchange_filtered}
    avg_dv = calculate_dollar_volumes(ticker_set)
    volume_filtered = filter_by_dollar_volume(exchange_filtered, avg_dv)
    
    # Step 4: Get market cap for volume-filtered tickers (expensive but reduced set)
    final_tickers = fetch_market_caps(volume_filtered)
    
    # Step 5: Summary
    print_summary(final_tickers)
    check_preferred_overlap(final_tickers)
    
    # Step 6: Write to Supabase
    if final_tickers:
        write_to_supabase(final_tickers, dry_run=dry_run)
    
    return final_tickers


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowOS Universe Scanner")
    parser.add_argument("command", choices=["filter", "status", "compare"],
                        help="filter: run liquidity filter | status: show current universe | compare: vs scanner_universe")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview results without writing to Supabase")
    
    args = parser.parse_args()
    
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in .env")
        sys.exit(1)
    
    if args.command == "filter":
        run_filter(dry_run=args.dry_run)
    elif args.command == "status":
        show_status()
    elif args.command == "compare":
        compare_universes()