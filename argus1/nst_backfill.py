"""
NST Backfill Script
====================
Populates nst_mentions from existing processed_articles table.

Usage:
    python -m argus1.nst_backfill --days 30
    python -m argus1.nst_backfill --days 7 --bubble "AI/Compute"
    python -m argus1.nst_backfill --days 14 --dry-run
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from nst import (
    NSTStorage,
    NSTMention,
    NST_KEYWORDS,
    ATTRACTOR_TICKERS,
    classify_mention,
    infer_bubble_type_from_alert,
    run_daily_aggregation,
)


def get_supabase_client():
    """Get Supabase client."""
    from supabase import create_client
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def fetch_processed_articles(
    client,
    days_back: int = 30,
    bubble_type: Optional[str] = None,
    limit: int = 1000
) -> List[Dict]:
    """
    Fetch processed articles from Supabase.
    
    Args:
        client: Supabase client
        days_back: How many days back to fetch
        bubble_type: Optional filter (not used yet, for future)
        limit: Max articles to fetch
    
    Returns:
        List of article dicts
    """
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    try:
        response = client.table('processed_articles').select(
            'id, title, url, published_at, y2ai_category, source_name, '
            'extracted_facts, companies_mentioned, sentiment'
        ).gte('published_at', cutoff).order(
            'published_at', desc=True
        ).limit(limit).execute()
        
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch articles: {e}")
        return []


def fetch_raw_articles(
    client,
    days_back: int = 30,
    limit: int = 1000
) -> List[Dict]:
    """
    Fetch raw articles from Supabase (alternative source).
    """
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    try:
        response = client.table('raw_articles').select(
            'id, title, url, published_at, source_name, content'
        ).gte('published_at', cutoff).order(
            'published_at', desc=True
        ).limit(limit).execute()
        
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch raw articles: {e}")
        return []


def backfill_nst(
    days_back: int = 30,
    bubble_filter: Optional[str] = None,
    use_raw: bool = False,
    dry_run: bool = False,
    limit: int = 1000
) -> Dict:
    """
    Backfill nst_mentions from existing articles.
    
    Args:
        days_back: How many days back to process
        bubble_filter: Only process articles matching this bubble type
        use_raw: Use raw_articles instead of processed_articles
        dry_run: Don't actually save, just show what would happen
        limit: Max articles to process
    
    Returns:
        Summary dict with counts
    """
    logger.info(f"Starting NST backfill: {days_back} days, limit={limit}, dry_run={dry_run}")
    
    client = get_supabase_client()
    storage = NSTStorage(client)
    
    # Fetch articles
    if use_raw:
        articles = fetch_raw_articles(client, days_back, limit)
        logger.info(f"Fetched {len(articles)} raw articles")
    else:
        articles = fetch_processed_articles(client, days_back, limit=limit)
        logger.info(f"Fetched {len(articles)} processed articles")
    
    if not articles:
        logger.warning("No articles found to backfill")
        return {"processed": 0, "saved": 0, "skipped": 0, "by_bubble": {}}
    
    # Track stats
    stats = {
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "by_bubble": defaultdict(int),
        "by_category": defaultdict(int),
        "dates_processed": set()
    }
    
    # Process each article
    for article in articles:
        title = article.get('title', '')
        content = article.get('content') or article.get('extracted_facts') or ''
        if isinstance(content, list):
            content = ' '.join(str(c) for c in content)
        
        source_name = article.get('source_name', 'unknown')
        url = article.get('url', '')
        published_at = article.get('published_at', '')
        
        # Infer bubble type
        bubble_type = infer_bubble_type_from_alert(
            alert_name=source_name,
            headline=title,
            snippet=content
        )
        
        if not bubble_type:
            stats["skipped"] += 1
            continue
        
        # Apply filter if specified
        if bubble_filter and bubble_type != bubble_filter:
            stats["skipped"] += 1
            continue
        
        # Classify the mention
        mention = classify_mention(
            headline=title,
            snippet=content,
            source=source_name,
            url=url,
            published_at=published_at,
            bubble_type=bubble_type
        )
        
        stats["processed"] += 1
        stats["by_bubble"][bubble_type] += 1
        stats["by_category"][mention.alert_category] += 1
        
        # Track date for aggregation
        if published_at:
            try:
                date_str = published_at[:10]  # YYYY-MM-DD
                stats["dates_processed"].add(date_str)
            except:
                pass
        
        # Save if not dry run
        if not dry_run:
            result = storage.save_mention(mention)
            if result:
                stats["saved"] += 1
        else:
            # Log what would be saved
            logger.debug(f"[DRY RUN] Would save: {bubble_type} / {mention.alert_category} / {title[:50]}...")
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKFILL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles: {len(articles)}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Saved: {stats['saved']}")
    logger.info(f"Skipped (no bubble match): {stats['skipped']}")
    
    logger.info(f"\nBy Bubble Type:")
    for bt, count in sorted(stats["by_bubble"].items()):
        logger.info(f"  {bt}: {count}")
    
    logger.info(f"\nBy Category:")
    for cat, count in sorted(stats["by_category"].items()):
        logger.info(f"  {cat}: {count}")
    
    # Run daily aggregation for all dates we processed
    if not dry_run and stats["dates_processed"]:
        logger.info(f"\nRunning daily aggregation for {len(stats['dates_processed'])} dates...")
        for date_str in sorted(stats["dates_processed"]):
            for bubble_type in ATTRACTOR_TICKERS.keys():
                from nst import calculate_daily_summary
                summary = calculate_daily_summary(storage, date_str, bubble_type)
                if summary.total_mentions > 0:
                    logger.info(f"  {date_str} / {bubble_type}: {summary.total_mentions} mentions, "
                               f"density={summary.narrative_density:.3f}")
    
    # Convert set to list for JSON serialization
    stats["dates_processed"] = list(stats["dates_processed"])
    stats["by_bubble"] = dict(stats["by_bubble"])
    stats["by_category"] = dict(stats["by_category"])
    
    return stats


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill NST from existing articles")
    parser.add_argument('--days', type=int, default=30,
                       help='Days back to process (default: 30)')
    parser.add_argument('--bubble', type=str, default=None,
                       help='Only process this bubble type')
    parser.add_argument('--raw', action='store_true',
                       help='Use raw_articles instead of processed_articles')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would happen without saving')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Max articles to process (default: 1000)')
    
    args = parser.parse_args()
    
    stats = backfill_nst(
        days_back=args.days,
        bubble_filter=args.bubble,
        use_raw=args.raw,
        dry_run=args.dry_run,
        limit=args.limit
    )
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {stats['processed']}")
    print(f"Saved: {stats['saved']}")
    
    if not args.dry_run:
        print("\nTest with:")
        print('  python -m argus1.nst status --bubble "AI/Compute"')
        print('  python -m argus1nst density --bubble "AI/Compute" --days 7')


if __name__ == "__main__":
    main()