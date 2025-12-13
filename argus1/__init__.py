"""
ARGUS-1 News Intelligence Layer
Replaces Google Alerts + Google Sheets with direct API access

Modules:
- aggregator: Multi-source collection (NewsAPI, Alpha Vantage, SEC EDGAR, RSS)
- processor: Claude extraction and Y2AI categorization  
- storage: PostgreSQL/Supabase storage (replaces Google Sheets)
- api: FastAPI endpoints for collection and querying
- scheduler: Automated collection runs

Quick Start:
    from argus1 import NewsAggregator, ClaudeProcessor, get_storage
    
    # Collect articles
    aggregator = NewsAggregator()
    articles = aggregator.collect_all(hours_back=24)
    
    # Process through Claude
    processor = ClaudeProcessor()
    processed = processor.process_batch(articles)
    
    # Store in database
    storage = get_storage()
    storage.insert_processed_articles([p.to_dict() for p in processed])

Environment Variables:
    NEWSAPI_KEY         - NewsAPI.org API key (optional, $449/mo for production)
    ALPHAVANTAGE_KEY    - Alpha Vantage API key (free, 25 req/day)
    ANTHROPIC_API_KEY   - Claude API key (required for processing)
    SUPABASE_URL        - Supabase project URL
    SUPABASE_KEY        - Supabase anon/service key
"""

__version__ = "1.0.0"
__author__ = "Y2AI"

# Core imports
from .aggregator import (
    NewsAggregator,
    RawArticle,
    ProcessedArticle,
    NewsAPIAdapter,
    AlphaVantageAdapter,
    SECEdgarAdapter,
    RSSAdapter
)

from .processor import (
    ClaudeProcessor,
    NewsletterProcessor,
    Y2AI_CATEGORIES
)

from .storage import (
    SupabaseStorage,
    SQLiteStorage,
    get_storage,
    SCHEMA_SQL
)

from .scheduler import (
    run_collection,
    start_scheduler,
    manual_run
)

__all__ = [
    # Aggregator
    "NewsAggregator",
    "RawArticle", 
    "ProcessedArticle",
    "NewsAPIAdapter",
    "AlphaVantageAdapter",
    "SECEdgarAdapter",
    "RSSAdapter",
    
    # Processor
    "ClaudeProcessor",
    "NewsletterProcessor",
    "Y2AI_CATEGORIES",
    
    # Storage
    "SupabaseStorage",
    "SQLiteStorage",
    "get_storage",
    "SCHEMA_SQL",
    
    # Scheduler
    "run_collection",
    "start_scheduler",
    "manual_run"
]
