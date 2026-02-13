"""
dm_backfill_2016.py
Backfill price_history with 2016-01-02 to 2019-12-31 data.

Resumable: tracks completed tickers so you can restart if rate-limited.
Uses the same Twelve Data API + Supabase upload from dm_historical_pipeline.

Usage:
    python dm_backfill_2016.py           # Run backfill (resumable)
    python dm_backfill_2016.py --status  # Check progress
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from Dm_historical_pipeline import (
    fetch_ticker_prices,
    upload_prices_to_supabase,
    load_universe,
    get_supabase,
    ALL_ETFS,
    DELAY_BETWEEN_CALLS,
    DAILY_CALL_LIMIT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BACKFILL_START = "2016-01-02"
BACKFILL_END = "2019-12-31"
CHECKPOINT_PHASE = "backfill_2016"


def get_backfill_completed():
    """Get tickers already backfilled."""
    supabase = get_supabase()
    try:
        result = supabase.table("dm_pipeline_checkpoint") \
            .select("ticker") \
            .eq("phase", CHECKPOINT_PHASE) \
            .eq("status", "complete") \
            .execute()
        return set(row["ticker"] for row in result.data) if result.data else set()
    except:
        return set()


def save_backfill_checkpoint(ticker, status, rows_fetched=0, error_msg=None):
    """Save backfill progress."""
    supabase = get_supabase()
    try:
        supabase.table("dm_pipeline_checkpoint").upsert({
            "phase": CHECKPOINT_PHASE,
            "ticker": ticker,
            "status": status,
            "rows_fetched": rows_fetched,
            "error_msg": error_msg,
            "updated_at": datetime.utcnow().isoformat()
        }, on_conflict="phase,ticker").execute()
    except Exception as e:
        logger.warning(f"Checkpoint save failed: {e}")


def run_backfill():
    """Fetch 2016-2019 prices for all tickers."""
    logger.info("=" * 60)
    logger.info(f"BACKFILL: {BACKFILL_START} to {BACKFILL_END}")
    logger.info("=" * 60)

    # Build ticker list
    universe = load_universe()
    all_tickers = list(set([u["ticker"] for u in universe] + ALL_ETFS))
    
    # Check what's done
    completed = get_backfill_completed()
    pending = [t for t in all_tickers if t not in completed]
    
    logger.info(f"Total tickers: {len(all_tickers)}")
    logger.info(f"Already backfilled: {len(completed)}")
    logger.info(f"Pending: {len(pending)}")
    logger.info(f"Estimated time: ~{len(pending) * DELAY_BETWEEN_CALLS / 60:.0f} minutes")
    logger.info("=" * 60)
    
    if not pending:
        logger.info("All tickers already backfilled.")
        return
    
    total_rows = 0
    errors = []
    api_calls = 0
    
    for i, ticker in enumerate(pending, 1):
        logger.info(f"[{i}/{len(pending)}] Fetching {ticker} ({BACKFILL_START} to {BACKFILL_END})...")
        
        try:
            rows = fetch_ticker_prices(ticker, start_date=BACKFILL_START, end_date=BACKFILL_END)
            logger.info(f"  Got {len(rows)} rows")
            
            if rows:
                uploaded = upload_prices_to_supabase(rows)
                logger.info(f"  Uploaded {uploaded} rows")
                total_rows += uploaded
            else:
                logger.info(f"  No data (ticker may not exist before 2020)")
            
            save_backfill_checkpoint(ticker, "complete", rows_fetched=len(rows))
            api_calls += 1
            
        except Exception as e:
            error_str = str(e)
            if "not found" in error_str.lower() or "no data" in error_str.lower():
                logger.info(f"  No data available (likely IPO after 2019)")
                save_backfill_checkpoint(ticker, "complete", rows_fetched=0, error_msg="no_data_pre2020")
            else:
                logger.error(f"  FAILED: {e}")
                save_backfill_checkpoint(ticker, "error", error_msg=error_str)
                errors.append(ticker)
        
        # Rate limiting
        if i < len(pending):
            time.sleep(DELAY_BETWEEN_CALLS)
        
        # Daily limit check
        if api_calls >= DAILY_CALL_LIMIT - 10:
            logger.warning(f"Approaching daily limit ({api_calls} calls). Stopping.")
            logger.warning(f"Run again to continue from checkpoint.")
            break
        
        # Progress every 25
        if i % 25 == 0:
            logger.info(f"--- Progress: {i}/{len(pending)}, {total_rows} rows, {api_calls} API calls ---")
    
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"Rows uploaded: {total_rows}")
    logger.info(f"API calls: {api_calls}")
    if errors:
        logger.warning(f"Errors ({len(errors)}): {', '.join(errors[:20])}")
    logger.info("=" * 60)


def show_status():
    """Show backfill progress."""
    supabase = get_supabase()
    
    result = supabase.table("dm_pipeline_checkpoint") \
        .select("status,ticker,rows_fetched,error_msg") \
        .eq("phase", CHECKPOINT_PHASE) \
        .execute()
    
    if not result.data:
        print("No backfill progress yet.")
        return
    
    complete = [r for r in result.data if r["status"] == "complete"]
    errors = [r for r in result.data if r["status"] == "error"]
    no_data = [r for r in complete if r.get("error_msg") == "no_data_pre2020"]
    with_data = [r for r in complete if r.get("error_msg") != "no_data_pre2020"]
    
    total_rows = sum(r.get("rows_fetched", 0) for r in complete)
    
    print(f"Backfill Status: {BACKFILL_START} to {BACKFILL_END}")
    print(f"  Completed: {len(complete)} tickers")
    print(f"    With data: {len(with_data)} ({total_rows} rows)")
    print(f"    No pre-2020 data: {len(no_data)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for r in errors[:10]:
            print(f"    {r['ticker']}: {r.get('error_msg', 'unknown')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_status()
    else:
        run_backfill()