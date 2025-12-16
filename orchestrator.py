"""
Y2AI ORCHESTRATOR
Main scheduler that coordinates all Y2AI services

Schedule:
- 4:30 PM ET: Daily bubble index and stock tracker update
- 4:45 PM ET: Social media daily update post
- 6:00 AM, 12:00 PM, 6:00 PM, 10:00 PM ET: ARGUS-1 news collection (weekdays)
- 10:00 AM ET: Weekend news collection
- Sunday 6:00 PM ET: Newsletter generation
- Monday 8:30 AM ET: Newsletter social post

This ties together:
- ARGUS-1 (news collection)
- Bubble Index Service
- Stock Tracker Service
- Social Publisher
- Supabase Storage
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional
import logging
import pprint
from urgency_mode import get_system_mode, check_nlp_urgency_triggers, is_urgency_mode


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULED TASKS
# =============================================================================



# inside y2ai/orchestrator.py (or a new utils/backfill.py)

from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

# def backfill_news_and_generate_newsletter(start_date_str: str, end_date_str: str, process_limit: int = 500):
#     """
#     Backfill ARGUS-1 articles between start_date and end_date (inclusive)
#     and then run newsletter generation.

#     start_date_str, end_date_str: 'YYYY-MM-DD'
#     """
#     # parse dates (explicit)
#     start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
#     end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

#     if end_date < start_date:
#         raise ValueError("end_date must be >= start_date")

#     # import argus1 backfill API
#     sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
#     try:
#         from argus1.scheduler import backfill
#     except Exception as e:
#         logger.error(f"argus1 backfill API not available: {e}")
#         raise

#     logger.info(f"Starting backfill: {start_date_str} -> {end_date_str}")
#     # Ensure we pass datetime objects covering the entire day
#     start_dt = datetime.combine(start_date, datetime.min.time())
#     end_dt = datetime.combine(end_date, datetime.max.time())

#     status = backfill(start_date=start_dt, end_date=end_dt, process_limit=process_limit)
#     if not status.get("status") == "completed":
#         logger.warning(f"Backfill reported non-completed status: {status}")
#     else:
#         logger.info(f"Backfill complete: raw={status.get('raw_count')}, processed={status.get('processed_count')}")

#     # Now generate newsletter from that data
#     from .storage import get_storage
#     storage = get_storage()

#     # Use newsletter generation (this uses storage.get_newsletter_ready_articles(days_back=...))
#     # but we'll pass days_back = number of days between inclusive range
#     days_back = (end_date - start_date).days + 1
#     articles = storage.get_newsletter_ready_articles(days_back=days_back)

#     if not articles:
#         logger.warning("No articles found for newsletter after backfill")
#         return False

#     # run existing generator
#     return run_newsletter_generation()


def backfill_news_and_generate_newsletter(start_date_str: str, end_date_str: str, process_limit: int = 500):
    """
    Backfill ARGUS-1 articles between start_date and end_date (inclusive)
    and STOP after storing raw + processed articles.
    
    No newsletter generation.
    No deletion.
    """

    # parse dates
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    # import argus backfill API
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from argus1.scheduler import backfill
    except Exception as e:
        logger.error(f"argus1 backfill API not available: {e}")
        raise

    logger.info(f"Starting backfill ONLY (no newsletter): {start_date_str} -> {end_date_str}")

    # Full date-time boundaries
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # Run backfill (this stores raw + processed!)
    status = backfill(start_date=start_dt, end_date=end_dt, process_limit=process_limit)

    if status.get("status") != "completed":
        logger.warning(f"Backfill finished with warnings: {status}")
    else:
        logger.info(
            f"Backfill done. Raw={status.get('raw_count')}, Processed={status.get('processed_count')}"
        )

    logger.info("Backfill completed. Skipping newsletter generation completely.")
    return True

def run_pipeline_with_mode_check(start_date_str: str, end_date_str: str):
    """
    Run backfill pipeline with urgency mode awareness.
    In urgency mode, logs extra info and checks NLP triggers after.
    """
    mode = get_system_mode()
    
    if mode["mode"] == "urgency":
        logger.info("=" * 50)
        logger.info("🚨 RUNNING IN URGENCY MODE")
        logger.info(f"   Triggered by: {mode['triggered_by']}")
        logger.info(f"   Reason: {mode['reason']}")
        logger.info("=" * 50)
    
    # Run normal backfill
    backfill_news_and_generate_newsletter(start_date_str, end_date_str)
    
    # After NLP processing, check if we should trigger urgency
    check_nlp_urgency_triggers()
    

def run_daily_indicators():
    """
    Run at 4:30 PM ET daily after market close.
    
    Updates:
    - Bubble Index (VIX, CAPE, Credit Spreads)
    - Stock Tracker (18 stocks, 3 pillars)
    """
    logger.info("=" * 60)
    logger.info("RUNNING DAILY INDICATORS UPDATE")
    logger.info("=" * 60)
    
    from .bubble_index import BubbleIndexCalculator
    from .stock_tracker import StockTracker
    from .storage import get_storage
    
    storage = get_storage()
    
    # 1. Calculate Bubble Index
    logger.info("\n--- BUBBLE INDEX ---")
    try:
        calculator = BubbleIndexCalculator()
        reading = calculator.calculate()
        
        # Store in database
        storage.store_bubble_reading(reading.to_dict())
        logger.info(f"✅ Bubble Index: {reading.bubble_index} | Regime: {reading.regime}")
    except Exception as e:
        logger.error(f"❌ Bubble Index error: {e}")
    
    # 2. Calculate Stock Tracker
    logger.info("\n--- STOCK TRACKER ---")
    try:
        tracker = StockTracker()
        report = tracker.generate_daily_report()
        
        # Store in database
        storage.store_stock_report(report.to_dict())
        logger.info(f"✅ Y2AI Index: {report.y2ai_index_today:+.2f}% | Status: {report.status}")
    except Exception as e:
        logger.error(f"❌ Stock Tracker error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DAILY INDICATORS COMPLETE")
    logger.info("=" * 60)
    
    return True


def run_daily_social_post():
    """
    Run at 4:45 PM ET daily after indicators update.
    Posts daily stock tracker update to all platforms.
    """
    logger.info("=" * 60)
    logger.info("RUNNING DAILY SOCIAL POST")
    logger.info("=" * 60)

    from .stock_tracker import StockTracker, DailyReport
    from types import SimpleNamespace
    from .social_publisher import SocialPublisher
    from .storage import get_storage
    from datetime import datetime

    storage = get_storage()
    report_data = storage.get_latest_stock_report()

    if not report_data:
        logger.error("No stock report available for social post")
        return False

    # Build a DailyReport object
    report = DailyReport(
        date=report_data.get('date'),
        stocks=report_data.get('stocks', []),
        pillars=report_data.get('pillars', []),
        y2ai_index_today=report_data.get('y2ai_index_today'),
        y2ai_index_5day=report_data.get('y2ai_index_5day'),
        y2ai_index_ytd=report_data.get('y2ai_index_ytd'),
        spy_today=report_data.get('spy_today'),
        spy_5day=report_data.get('spy_5day'),
        spy_ytd=report_data.get('spy_ytd'),
        qqq_today=report_data.get('qqq_today'),
        qqq_5day=report_data.get('qqq_5day'),
        qqq_ytd=report_data.get('qqq_ytd'),
        status=report_data.get('status'),
        best_stock=report_data.get('best_stock'),
        worst_stock=report_data.get('worst_stock'),
        best_pillar=report_data.get('best_pillar'),
        worst_pillar=report_data.get('worst_pillar'),
        calculated_at=report_data.get('calculated_at')
    )

    # Convert dicts → objects that support attribute access
    if isinstance(report.stocks, list):
        report.stocks = [
            SimpleNamespace(**s) if isinstance(s, dict) else s
            for s in report.stocks
        ]

    if isinstance(report.pillars, list):
        report.pillars = [
            SimpleNamespace(**p) if isinstance(p, dict) else p
            for p in report.pillars
        ]


    # 🔥 Now generate social post
    tracker = StockTracker()
    post_content = tracker.format_for_social(report)

    publisher = SocialPublisher()
    results = publisher.publish_daily_tracker(post_content)

    for platform, url in results.items():
        if url:
            logger.info(f"✅ Posted to {platform}: {url}")
        else:
            logger.warning(f"❌ Failed to post to {platform}")

    return True



def run_news_collection(hours_back: int = 6, process_limit: int = 30):
    """
    Run ARGUS-1 news collection.
    
    Weekdays: 4x daily (6am, 12pm, 6pm, 10pm)
    Weekends: 1x daily (10am)
    """
    logger.info("=" * 60)
    logger.info("RUNNING NEWS COLLECTION (ARGUS-1)")
    logger.info("=" * 60)
    
    # Import from argus1 package (sibling directory)
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from argus1.scheduler import run_collection
        result = run_collection(hours_back=hours_back, process_limit=process_limit)
        
        logger.info(f"Raw articles: {result.get('raw_count', 0)}")
        logger.info(f"Processed: {result.get('processed_count', 0)}")
        logger.info(f"Status: {result.get('status', 'unknown')}")
        
        return result.get('status') == 'completed'
    except ImportError as e:
        logger.error(f"ARGUS-1 not available: {e}")
        return False
    except Exception as e:
        logger.error(f"News collection error: {e}")
        return False
    
    


def format_long_date(date_str):
    """Convert YYYY-MM-DD → Month DD, YYYY"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%B %d, %Y")


def generate_newsletter_from_context(context):
    import httpx
    import json
    import os
    
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment variables")
    
    prompt = f"""
You are Y2AI, an AI macro + infrastructure research analyst.

You are writing **Edition #{context['edition_number']}** of a weekly newsletter.

If the JSON context includes a 'previous_newsletter' field, that text is the
full content of the previous edition (Edition #{context.get('previous_edition_number')}).  
Use it to maintain narrative and analytical continuity:
- Keep the same overall voice and tone
- Treat this edition as a continuation: update the thesis, call back to prior points
- Refer to prior edition when helpful (e.g., “Last week we argued that…”)
- Do NOT repeat the exact same text; build on it and extend the argument

Rewrite the weekly newsletter using the EXACT SAME THEME, STRUCTURE, AND STYLE as the following model newsletter:

================= STYLE TEMPLATE (COPY EXACTLY) =================
# Y2AI WEEKLY EDITION #[edition_number] - PREVIEW DRAFT
## "[Dynamic Title Based on Context]"
**Status:** Preview (Full publication Monday, {context['week_ending']})
**Generated:** {context['week_ending']}
**Edition Number:** {context['edition_number']}
**Word Count:** ~1200 words
**Framework:** One-sentence framework summarizing the week's analytical lens

---

## EXECUTIVE SUMMARY
(3–4 strong paragraphs narratively summarizing the week's structural insight, written in a serious forward-looking analyst tone.)

---

## CORE FRAMEWORK
(Define the analytical lens of the week, comparing two opposing forces or concepts.)

---

## EVIDENCE PILLAR #1: (Auto-generate a pillar name)
(2–3 paragraphs written like the example: company data → implications → structural meaning.)

## EVIDENCE PILLAR #2: (Auto-generate a pillar name)
(same structure)

## EVIDENCE PILLAR #3: (Auto-generate a pillar name)
(same structure)

(Optional: Add Pillar #4 or #5 depending on context richness)

---

## THE IMPLICATIONS
(Forward-looking consequences for 2025–2026. Connect signals into macro meaning.)

---

## THE THESIS
(1 paragraph summarizing the “big insight” of the week.)

---

## WHAT TO WATCH NEXT WEEK
- 4–6 bullet points tightly tied to companies, capex, constraints, cloud, AI infra

---

## OPENING HOOK FOR LINKEDIN
(A 1–2 sentence punchy LinkedIn-style hook summarizing the framework.)

---

Follow this exact structure.  
Follow the tone, pacing, paragraph style, and transitions from the sample newsletter.

================= END STYLE TEMPLATE =================

Below is the JSON context to use for all factual references.  
If 'previous_newsletter' is present, treat it as prior narrative context, not as data to copy verbatim.  
DO NOT include the JSON itself in the final output.

{json.dumps(context, indent=2)}

Now generate the fully formatted newsletter using the theme template above.
"""

    headers = {
        "x-api-key": claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 5000,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=80
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data["content"][0]["text"]
        
    except Exception as e:
        logger.error(f"Newsletter generation error: {e}")
        raise


def run_newsletter_generation():
    logger.info("=" * 60)
    logger.info("RUNNING NEWSLETTER GENERATION")
    logger.info("=" * 60)
    
    from .storage import get_storage
    storage = get_storage()

    # --- 1) Determine edition numbers and load previous newsletter (if any) ---
    latest_edition = 0
    previous_newsletter = None

    try:
        latest_edition = storage.get_latest_edition_number()
    except Exception as e:
        logger.error(f"Error getting latest edition number: {e}")

    next_edition = latest_edition + 1
    logger.info(f"Next newsletter edition will be #{next_edition}")

    if latest_edition > 0 and hasattr(storage, "get_newsletter_by_edition"):
        try:
            previous_newsletter = storage.get_newsletter_by_edition(latest_edition)
            if previous_newsletter:
                logger.info(f"Loaded previous newsletter edition #{latest_edition} for continuity")
        except Exception as e:
            logger.error(f"Error loading previous newsletter edition {latest_edition}: {e}")

    # --- 2) Fetch articles + bubble for this edition ---
    articles = storage.get_newsletter_ready_articles(days_back=1)
    if not articles:
        logger.warning("No articles found for newsletter")
        return False

    bubble = storage.get_latest_bubble_reading()
    category_counts = {cat: len(arts) for cat, arts in articles.items()}
    logger.info(f"Articles by category: {category_counts}")

    # Flatten articles to capture their IDs
    all_articles = []
    for cat_articles in articles.values():
        all_articles.extend(cat_articles)
    article_ids = [a.get("id") for a in all_articles if a.get("id") is not None]

    # --- 3) Build context (with continuity info) ---
    newsletter_context = {
        "edition_number": next_edition,
        "week_ending": format_long_date(datetime.now().strftime("%Y-%m-%d")),
        "bubble_index": bubble,
        "articles_by_category": articles,
        "total_articles": sum(category_counts.values()),
        "previous_edition_number": latest_edition if latest_edition > 0 else None,
        "previous_newsletter": (
            previous_newsletter.get("content_markdown")
            if previous_newsletter
            else None
        ),
    }

    logger.info(f"Prepared context for Edition #{next_edition}")
    logger.info(f"Total articles: {newsletter_context['total_articles']}")
    pprint.pprint(newsletter_context)

    # --- 4) Generate & save newsletter, then persist edition + delete articles ---
    try:
        newsletter_md = generate_newsletter_from_context(newsletter_context)
        
        # File names
        md_path = f"newsletter_edition_{next_edition}.md"
        txt_path = f"newsletter_edition_{next_edition}.txt"

        # Save Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(newsletter_md)

        # Convert MD → TXT (remove markdown symbols)
        newsletter_txt = (
            newsletter_md.replace("#", "")
                         .replace("*", "")
                         .replace("•", "-")
                         .strip()
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(newsletter_txt)

        logger.info(f"Newsletter generated and saved:")
        logger.info(f" - Markdown: {md_path}")
        logger.info(f" - Text: {txt_path}")

        # 4a) Save newsletter edition in Supabase
        if hasattr(storage, "save_newsletter_edition"):
            try:
                storage.save_newsletter_edition(
                    edition_number=next_edition,
                    title=f"Y2AI Weekly Edition #{next_edition}",
                    content_markdown=newsletter_md,
                    status="published",
                    article_ids=article_ids,
                )
                logger.info(f"Saved newsletter edition #{next_edition} in storage")
            except Exception as e:
                logger.error(f"Error saving newsletter edition to storage: {e}")
        else:
            logger.warning("Storage backend does not support save_newsletter_edition")

        

        return True

    except Exception as e:
        logger.error(f"Error saving newsletter files: {e}")
        return False

def run_newsletter_social():
    """
    Run Monday 8:30 AM ET to post newsletter announcement.
    """
    logger.info("=" * 60)
    logger.info("RUNNING NEWSLETTER SOCIAL POST")
    logger.info("=" * 60)
    
    from .social_publisher import SocialPublisher, PostTemplates
    from .storage import get_storage
    
    storage = get_storage()
    
    # Get latest newsletter
    latest_edition = storage.get_latest_edition_number()
    
    if latest_edition == 0:
        logger.warning("No newsletter editions found")
        return False
    
    # Generate social posts
    # TODO: Pull actual newsletter data for customization
    linkedin_post = PostTemplates.newsletter_linkedin(
        edition_number=latest_edition,
        title="Infrastructure vs Bubble Analysis",
        hook="The AI spending continues...",
        key_findings=["Key finding 1", "Key finding 2"],
        prediction="Q1 2026 prediction",
        link="https://y2ai.us/newsletter"
    )
    
    twitter_thread = PostTemplates.newsletter_twitter_thread(
        edition_number=latest_edition,
        title="Infrastructure vs Bubble Analysis",
        hook="The AI spending continues...",
        key_points=["Point 1", "Point 2", "Point 3"],
        prediction="Q1 2026 prediction",
        link="https://y2ai.us/newsletter"
    )
    
    # Publish
    publisher = SocialPublisher()
    results = publisher.publish_newsletter(linkedin_post, twitter_thread)
    
    for platform, url in results.items():
        if url:
            logger.info(f"✅ Posted to {platform}: {url}")
    
    return True


# =============================================================================
# SCHEDULER
# =============================================================================

def start_scheduler():
    """
    Start the APScheduler background scheduler.
    
    Schedule:
    - 4:30 PM ET: Daily indicators
    - 4:45 PM ET: Daily social post
    - 6:00, 12:00, 18:00, 22:00 ET: News collection (weekdays)
    - 10:00 ET: News collection (weekends)
    - Sunday 18:00 ET: Newsletter generation
    - Monday 8:30 ET: Newsletter social
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return None
    
    scheduler = BackgroundScheduler(timezone='US/Eastern')
    
    # Daily indicators (4:30 PM ET)
    scheduler.add_job(
        run_daily_indicators,
        CronTrigger(day_of_week='mon-fri', hour=16, minute=30),
        id='daily_indicators',
        name='Daily Indicators Update'
    )
    
    # Daily social post (4:45 PM ET)
    scheduler.add_job(
        run_daily_social_post,
        CronTrigger(day_of_week='mon-fri', hour=16, minute=45),
        id='daily_social',
        name='Daily Social Post'
    )
    
    # News collection - weekdays (4x daily)
    for hour in [6, 12, 18, 22]:
        scheduler.add_job(
            run_news_collection,
            CronTrigger(day_of_week='mon-fri', hour=hour, minute=0),
            id=f'news_collection_{hour}',
            name=f'News Collection {hour}:00',
            kwargs={'hours_back': 6, 'process_limit': 30}
        )
    
    # News collection - weekends (1x daily at 10am)
    scheduler.add_job(
        run_news_collection,
        CronTrigger(day_of_week='sat,sun', hour=10, minute=0),
        id='news_collection_weekend',
        name='Weekend News Collection',
        kwargs={'hours_back': 24, 'process_limit': 20}
    )
    
    # Newsletter generation (Sunday 6pm)
    scheduler.add_job(
        run_newsletter_generation,
        CronTrigger(day_of_week='sun', hour=18, minute=0),
        id='newsletter_gen',
        name='Newsletter Generation'
    )
    
    # Newsletter social (Monday 8:30am)
    scheduler.add_job(
        run_newsletter_social,
        CronTrigger(day_of_week='mon', hour=8, minute=30),
        id='newsletter_social',
        name='Newsletter Social Post'
    )
    
    scheduler.start()
    
    logger.info("Y2AI Scheduler started")
    logger.info("Jobs scheduled:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")
    
    return scheduler


# =============================================================================
# MANUAL TRIGGERS
# =============================================================================

def run_all_now():
    """Run all daily tasks immediately (for testing)"""
    logger.info("Running all tasks now...")
    
    run_news_collection(hours_back=24, process_limit=50)
    run_daily_indicators()
    run_daily_social_post()
    
    


def run_weekly_now():
    """Run weekly tasks immediately (for testing)"""
    logger.info("Running weekly tasks now...")
    
    run_newsletter_generation()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Y2AI Orchestrator")
    parser.add_argument('--daemon', action='store_true', help='Run as background scheduler')
    parser.add_argument('--indicators', action='store_true', help='Run daily indicators now')
    parser.add_argument('--social', action='store_true', help='Run daily social post now')
    parser.add_argument('--news', action='store_true', help='Run news collection now')
    parser.add_argument('--newsletter', action='store_true', help='Run newsletter generation now')
    parser.add_argument('--all', action='store_true', help='Run all daily tasks now')
    parser.add_argument(
    '--backfill',
    nargs=2,
    metavar=('START_DATE', 'END_DATE'),
    help='Backfill articles and generate newsletter (YYYY-MM-DD YYYY-MM-DD)'
)
    parser.add_argument(
    '--wipe-supabase',
    action='store_true',
    help='Delete ALL data from Supabase storage (DANGEROUS)'
)


    
    args = parser.parse_args()
    
    if args.daemon:
        scheduler = start_scheduler()
        if scheduler:
            try:
                print("Y2AI Orchestrator running. Press Ctrl+C to stop.")
                while True:
                    import time
                    time.sleep(60)
            except KeyboardInterrupt:
                scheduler.shutdown()
                print("\nScheduler stopped.")
    elif args.indicators:
        run_daily_indicators()
    elif args.social:
        run_daily_social_post()
    elif args.news:
        run_news_collection()
    elif args.newsletter:
        run_newsletter_generation()
    elif args.all:
        run_all_now()
    elif args.backfill:
        start_date, end_date = args.backfill
        run_pipeline_with_mode_check(start_date, end_date) 
        
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m y2ai.orchestrator --daemon     # Run scheduler")
        print("  python -m y2ai.orchestrator --indicators # Run bubble index + stocks")
        print("  python -m y2ai.orchestrator --social     # Post to social media")
        print("  python -m y2ai.orchestrator --all        # Run everything now")
        
       
