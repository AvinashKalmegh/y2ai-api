"""
ARGUS-1 SCHEDULER
Automated collection runs - replaces manual Google Alerts checking

Schedule:
- Weekdays: 4x daily (6am, 12pm, 6pm, 10pm)
- Weekends: 1x daily (10am)

Can be run as:
- Standalone script with APScheduler
- Cron job
- Cloud Functions / AWS Lambda
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# COLLECTION RUNNER
# =============================================================================

def run_collection(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    hours_back: Optional[int] = None,
    process_limit: int = 30,
    forced_date: Optional[datetime.date] = None,
    aggregate_daily_signals: bool = True,  # NEW: Enable daily signal aggregation
) -> dict:
    """
    Run a complete collection cycle.

    New: accepts start_time/end_time for explicit windows (historical backfill).
    Backwards-compatible: if start_time/end_time not provided, uses hours_back (last N hours).

    forced_date: if provided, the storage-insert step will override any article timestamps
                 to this date (useful if aggregator cannot natively fetch historical windows).
    aggregate_daily_signals: if True, calculate and store daily signal aggregates after processing
    """
    from .aggregator import NewsAggregator
    from .processor import ClaudeProcessor, DailySignalAggregator
    from .storage import get_storage

    start_time = start_time or (datetime.utcnow() - timedelta(hours=hours_back)) if hours_back else None
    end_time = end_time or datetime.utcnow()

    start_time = start_time or (datetime.utcnow() - timedelta(hours=24))
    end_time = end_time or datetime.utcnow()

    start_time_utc = start_time
    end_time_utc = end_time

    start_time_iso = start_time_utc.isoformat()
    end_time_iso = end_time_utc.isoformat()

    start_time_log = start_time_utc.strftime("%Y-%m-%d %H:%M:%S")
    end_time_log = end_time_utc.strftime("%Y-%m-%d %H:%M:%S")

    start_time_run = datetime.utcnow()
    result = {
        "started_at": start_time_run.isoformat(),
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "raw_count": 0,
        "filtered_count": 0,
        "processed_count": 0,
        "daily_signals_updated": False,
        "errors": [],
        "status": "running"
    }

    try:
        storage = get_storage()
        run_id = -1

        # Start run in database if supported
        if hasattr(storage, 'start_collection_run'):
            run_id = storage.start_collection_run(["all"])
            result["run_id"] = run_id
            logger.info("Started collection run id=%s", run_id)

        # Phase 1: Collect raw articles using explicit time window (preferred)
        logger.info(f"Starting collection window: {start_time_log} → {end_time_log}")
        aggregator = NewsAggregator()

        # Prefer explicit window if aggregator supports it
        try:
            raw_articles = aggregator.collect_all(start_time=start_time_utc, end_time=end_time_utc)
        except TypeError:
            delta = end_time_utc - start_time_utc
            fallback_hours = int(delta.total_seconds() // 3600) or 24
            logger.warning(
                "NewsAggregator.collect_all(start_time, end_time) not supported; falling back to hours_back=%s",
                fallback_hours
            )
            raw_articles = aggregator.collect_all(hours_back=fallback_hours)

        result["raw_count"] = len(raw_articles)
        logger.info(f"Collected {len(raw_articles)} raw articles")

        # Store raw articles
        if raw_articles:
            raw_dicts = [a.to_dict() for a in raw_articles]

            if forced_date is not None:
                forced_iso = forced_date.isoformat() if hasattr(forced_date, "isoformat") else str(forced_date)
                for r in raw_dicts:
                    if 'published_at' in r:
                        r['published_at'] = f"{forced_iso}T00:00:00Z"
                    else:
                        r['published_date'] = forced_iso

            inserted = storage.insert_raw_articles(raw_dicts, collection_run_id=run_id)
            logger.info(f"Stored {inserted} raw articles")

        # Phase 2: Filter for relevance
        processor = ClaudeProcessor()
        filtered = processor.quick_relevance_filter(raw_articles)
        result["filtered_count"] = len(filtered)
        logger.info(f"Filtered to {len(filtered)} relevant articles")

        # Phase 3: Process through Claude
        processed_dates = set()  # Track which dates we processed
        if filtered:
            processed = processor.process_batch(filtered, max_batch=process_limit)
            result["processed_count"] = len(processed)
            logger.info(f"Processed {len(processed)} articles through Claude")

            # Store processed articles
            processed_dicts = [p.to_dict() for p in processed]

            if forced_date is not None:
                forced_iso = forced_date.isoformat() if hasattr(forced_date, "isoformat") else str(forced_date)
                for r in processed_dicts:
                    if 'published_at' in r:
                        r['published_at'] = f"{forced_iso}T00:00:00Z"
                    else:
                        r['published_date'] = forced_iso

            storage.insert_processed_articles(processed_dicts)
            
            # Collect unique dates from processed articles
            for p in processed:
                if p.published_at:
                    try:
                        if isinstance(p.published_at, str):
                            date_str = p.published_at[:10]  # YYYY-MM-DD
                        else:
                            date_str = p.published_at.strftime("%Y-%m-%d")
                        processed_dates.add(date_str)
                    except Exception:
                        pass

        # =====================================================================
        # Phase 4: AGGREGATE DAILY SIGNALS (NEW!)
        # =====================================================================
        if aggregate_daily_signals and processed_dates:
            logger.info("=" * 60)
            logger.info("AGGREGATING DAILY SIGNALS")
            logger.info("=" * 60)
            
            try:
                signal_aggregator = DailySignalAggregator()
                
                for date_str in sorted(processed_dates):
                    logger.info(f"Calculating daily signals for {date_str}...")
                    success = signal_aggregator.store_daily_signals(date_str)
                    if success:
                        logger.info(f"✅ Daily signals stored for {date_str}")
                    else:
                        logger.warning(f"⚠️ Failed to store daily signals for {date_str}")
                
                result["daily_signals_updated"] = True
                result["daily_signals_dates"] = list(processed_dates)
                logger.info(f"Daily signals updated for {len(processed_dates)} dates")
                
            except Exception as e:
                logger.error(f"Daily signal aggregation error: {e}")
                result["errors"].append(f"Daily signal aggregation: {str(e)}")
        
        # If forced_date was used, also aggregate for that specific date
        if aggregate_daily_signals and forced_date is not None:
            try:
                forced_date_str = forced_date.isoformat() if hasattr(forced_date, "isoformat") else str(forced_date)
                if forced_date_str not in processed_dates:
                    logger.info(f"Calculating daily signals for forced_date {forced_date_str}...")
                    signal_aggregator = DailySignalAggregator()
                    signal_aggregator.store_daily_signals(forced_date_str)
            except Exception as e:
                logger.error(f"Forced date signal aggregation error: {e}")

        # Complete run
        result["status"] = "completed"
        result["duration_seconds"] = (datetime.utcnow() - start_time_run).total_seconds()

        if hasattr(storage, 'complete_collection_run') and run_id > 0:
            storage.complete_collection_run(
                run_id,
                raw_count=result["raw_count"],
                processed_count=result["processed_count"],
                errors=result["errors"]
            )

        logger.info(f"Collection complete: {result['raw_count']} raw, {result['processed_count']} processed")

    except Exception as e:
        logger.exception("Collection failed: %s", e)
        result["status"] = "failed"
        result["errors"].append(str(e))

    return result


# =============================================================================
# DAILY SIGNAL AGGREGATION (standalone function)
# =============================================================================

def aggregate_signals_for_date(date_str: str) -> bool:
    """
    Aggregate daily signals for a specific date.
    Can be called independently of collection.
    
    Usage:
        from argus1.scheduler import aggregate_signals_for_date
        aggregate_signals_for_date("2025-12-10")
    """
    from .processor import DailySignalAggregator
    
    logger.info(f"Aggregating signals for {date_str}")
    aggregator = DailySignalAggregator()
    return aggregator.store_daily_signals(date_str)


def aggregate_signals_for_date_range(start_date: str, end_date: str) -> dict:
    """
    Aggregate daily signals for a date range.
    
    Usage:
        from argus1.scheduler import aggregate_signals_for_date_range
        aggregate_signals_for_date_range("2025-12-01", "2025-12-10")
    """
    from .processor import DailySignalAggregator
    
    logger.info(f"Aggregating signals for {start_date} to {end_date}")
    aggregator = DailySignalAggregator()
    return aggregator.backfill_daily_signals(start_date, end_date)


# =============================================================================
# APSCHEDULER IMPLEMENTATION
# =============================================================================

def start_scheduler():
    """
    Start the background scheduler.
    
    Schedule:
    - Weekdays (Mon-Fri): 6am, 12pm, 6pm, 10pm
    - Weekends (Sat-Sun): 10am only
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return None
    
    scheduler = BackgroundScheduler()
    
    # Weekday schedule (4x daily)
    weekday_hours = [6, 12, 18, 22]
    for hour in weekday_hours:
        scheduler.add_job(
            run_collection,
            CronTrigger(day_of_week='mon-fri', hour=hour, minute=0),
            kwargs={"hours_back": 6, "process_limit": 30, "aggregate_daily_signals": True},
            id=f"weekday_{hour}",
            name=f"Weekday collection at {hour}:00"
        )
    
    # Weekend schedule (1x daily)
    scheduler.add_job(
        run_collection,
        CronTrigger(day_of_week='sat,sun', hour=10, minute=0),
        kwargs={"hours_back": 24, "process_limit": 20, "aggregate_daily_signals": True},
        id="weekend",
        name="Weekend collection at 10:00"
    )
    
    scheduler.start()
    logger.info("Scheduler started")
    logger.info(f"Jobs: {[job.name for job in scheduler.get_jobs()]}")
    
    return scheduler


# =============================================================================
# CRON-COMPATIBLE RUNNER
# =============================================================================

def cron_entry():
    """
    Entry point for cron jobs.
    
    Add to crontab:
    # Weekdays: 6am, 12pm, 6pm, 10pm
    0 6,12,18,22 * * 1-5 cd /path/to/argus1 && python -m argus1.scheduler --cron
    
    # Weekends: 10am
    0 10 * * 0,6 cd /path/to/argus1 && python -m argus1.scheduler --cron --weekend
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cron', action='store_true', help='Run as cron job')
    parser.add_argument('--weekend', action='store_true', help='Weekend mode (longer lookback)')
    parser.add_argument('--hours', type=int, default=None, help='Override hours_back')
    parser.add_argument('--limit', type=int, default=30, help='Max articles to process')
    parser.add_argument('--no-signals', action='store_true', help='Skip daily signal aggregation')
    
    args = parser.parse_args()
    
    if args.cron:
        hours_back = args.hours or (24 if args.weekend else 6)
        result = run_collection(
            hours_back=hours_back, 
            process_limit=args.limit,
            aggregate_daily_signals=not args.no_signals
        )
        print(f"Collection complete: {result}")
        return result
    else:
        # Start background scheduler
        scheduler = start_scheduler()
        if scheduler:
            try:
                while True:
                    import time
                    time.sleep(60)
            except KeyboardInterrupt:
                scheduler.shutdown()
                print("Scheduler stopped")


# =============================================================================
# CLOUD FUNCTION HANDLER
# =============================================================================

def cloud_function_handler(event, context) -> dict:
    """
    Handler for AWS Lambda / Google Cloud Functions.
    """
    hours_back = event.get("hours_back", 6)
    process_limit = event.get("process_limit", 30)
    aggregate_signals = event.get("aggregate_daily_signals", True)

    start_time = None
    end_time = None
    forced_date = None

    if event.get("start_time"):
        try:
            start_time = datetime.fromisoformat(event["start_time"])
        except Exception:
            logger.warning("Invalid start_time format: %s", event.get("start_time"))

    if event.get("end_time"):
        try:
            end_time = datetime.fromisoformat(event["end_time"])
        except Exception:
            logger.warning("Invalid end_time format: %s", event.get("end_time"))

    if event.get("forced_date"):
        try:
            fd = datetime.fromisoformat(event["forced_date"])
            forced_date = fd.date()
        except Exception:
            try:
                forced_date = datetime.strptime(event["forced_date"], "%Y-%m-%d").date()
            except Exception:
                logger.warning("Invalid forced_date format: %s", event.get("forced_date"))

    logger.info(f"Cloud function triggered: hours_back={hours_back}, limit={process_limit}")

    result = run_collection(
        start_time=start_time, 
        end_time=end_time, 
        hours_back=hours_back, 
        process_limit=process_limit, 
        forced_date=forced_date,
        aggregate_daily_signals=aggregate_signals
    )

    return {
        "statusCode": 200 if result["status"] == "completed" else 500,
        "body": result
    }


# =============================================================================
# MANUAL TRIGGER
# =============================================================================

def manual_run(hours_back: int = 24, process_limit: int = 50, aggregate_signals: bool = True):
    """
    Run collection manually from command line.
    
    Usage:
        python -c "from argus1.scheduler import manual_run; manual_run(24, 50)"
    """
    print(f"\n{'='*60}")
    print("ARGUS-1 Manual Collection Run")
    print(f"{'='*60}")
    print(f"Hours back: {hours_back}")
    print(f"Process limit: {process_limit}")
    print(f"Aggregate daily signals: {aggregate_signals}")
    print(f"{'='*60}\n")
    
    result = run_collection(
        hours_back=hours_back, 
        process_limit=process_limit,
        aggregate_daily_signals=aggregate_signals
    )
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Status: {result['status']}")
    print(f"Raw articles: {result['raw_count']}")
    print(f"Filtered: {result['filtered_count']}")
    print(f"Processed: {result['processed_count']}")
    print(f"Daily signals updated: {result.get('daily_signals_updated', False)}")
    print(f"Duration: {result.get('duration_seconds', 0):.1f}s")
    
    if result['errors']:
        print(f"Errors: {result['errors']}")
    
    return result


# =============================================================================
# BACKFILL FUNCTION
# =============================================================================

def backfill(start_date: datetime, end_date: datetime, process_limit: int = 500):
    """
    Backfill articles between start_date and end_date (inclusive).
    Iterates day-by-day and calls run_collection with an explicit start_time/end_time.
    """
    if isinstance(start_date, datetime):
        start_dt = start_date
    else:
        start_dt = datetime.combine(start_date, datetime.min.time())

    if isinstance(end_date, datetime):
        end_dt = end_date
    else:
        end_dt = datetime.combine(end_date, datetime.max.time())

    day = start_dt.date()
    end_date_only = end_dt.date()

    total_raw = 0
    total_processed = 0
    details = []

    logger.info("Starting backfill: %s -> %s", start_dt.date(), end_date_only)

    while day <= end_date_only:
        start_ts = datetime.combine(day, datetime.min.time())
        end_ts = datetime.combine(day, datetime.max.time())
        try:
            logger.info("Backfilling day: %s (window: %s → %s)", day.isoformat(), start_ts, end_ts)

            result = run_collection(
                start_time=start_ts,
                end_time=end_ts,
                process_limit=process_limit,
                forced_date=day,
                aggregate_daily_signals=True  # Aggregate signals for each day
            )

            raw = int(result.get("raw_count", 0))
            processed = int(result.get("processed_count", 0))
            total_raw += raw
            total_processed += processed
            details.append({
                "day": day.isoformat(), 
                "raw": raw, 
                "processed": processed, 
                "status": result.get("status"),
                "daily_signals": result.get("daily_signals_updated", False)
            })
            logger.info("Day %s -> raw=%s processed=%s signals=%s", 
                       day.isoformat(), raw, processed, result.get("daily_signals_updated", False))
        except Exception as e:
            logger.exception("run_collection failed for day %s: %s", day.isoformat(), e)
            details.append({"day": day.isoformat(), "error": str(e)})
        day = day + timedelta(days=1)

    status = "completed" if total_processed > 0 else "partial" if total_raw > 0 else "failed"
    summary = {
        "status": status,
        "raw_count": total_raw,
        "processed_count": total_processed,
        "details": details
    }

    logger.info("Backfill finished: %s", summary)
    return summary


# =============================================================================
# REPROCESS AND AGGREGATE (for existing articles)
# =============================================================================

def reprocess_and_aggregate(start_date: str, end_date: str, batch_size: int = 50):
    """
    Reprocess existing articles with enhanced signal detection and aggregate daily signals.
    
    Use this when you've already collected articles but need to:
    1. Re-extract signals with the enhanced prompt
    2. Populate daily_signals table
    
    Usage:
        from argus1.scheduler import reprocess_and_aggregate
        reprocess_and_aggregate("2025-12-01", "2025-12-10")
    """
    from .processor import ArticleReprocessor, DailySignalAggregator
    
    logger.info("=" * 60)
    logger.info("REPROCESSING ARTICLES AND AGGREGATING SIGNALS")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("=" * 60)
    
    # Step 1: Reprocess articles
    reprocessor = ArticleReprocessor()
    reprocess_stats = reprocessor.reprocess_all(
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size
    )
    
    logger.info(f"Reprocessing complete: {reprocess_stats}")
    
    # Step 2: Aggregate daily signals
    aggregator = DailySignalAggregator()
    signal_stats = aggregator.backfill_daily_signals(start_date, end_date)
    
    logger.info(f"Daily signal aggregation complete: {signal_stats}")
    
    return {
        "reprocess": reprocess_stats,
        "daily_signals": signal_stats
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if "--aggregate-only" in sys.argv:
            # Just aggregate signals for today
            today = datetime.utcnow().strftime("%Y-%m-%d")
            aggregate_signals_for_date(today)
        elif "--aggregate-range" in sys.argv:
            # Aggregate for a date range
            idx = sys.argv.index("--aggregate-range")
            if len(sys.argv) > idx + 2:
                start = sys.argv[idx + 1]
                end = sys.argv[idx + 2]
                aggregate_signals_for_date_range(start, end)
            else:
                print("Usage: --aggregate-range START_DATE END_DATE")
        else:
            cron_entry()
    else:
        # Default: run once manually
        manual_run(hours_back=24, process_limit=30, aggregate_signals=True)