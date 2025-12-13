"""
ARGUS-1 STORAGE LAYER
PostgreSQL/Supabase storage replacing Google Sheets

Tables:
- raw_articles: Unprocessed articles from all sources
- processed_articles: Claude-enriched articles with Y2AI categorization
- collection_runs: Metadata about each collection run
- newsletter_editions: Published newsletter content
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SQL SCHEMA DEFINITIONS
# =============================================================================

SCHEMA_SQL = """
-- Raw articles table (replaces Google Alerts email parsing)
CREATE TABLE IF NOT EXISTS raw_articles (
    id SERIAL PRIMARY KEY,
    article_hash VARCHAR(32) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_name VARCHAR(200),
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    content TEXT,
    author VARCHAR(200),
    ticker VARCHAR(10),
    relevance_signals JSONB,
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    collection_run_id INTEGER
);

-- Processed articles table (replaces Google Sheets data entry)
CREATE TABLE IF NOT EXISTS processed_articles (
    id SERIAL PRIMARY KEY,
    article_hash VARCHAR(32) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_name VARCHAR(200),
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    y2ai_category VARCHAR(50) NOT NULL,
    extracted_facts JSONB,
    impact_score FLOAT,
    sentiment VARCHAR(20),
    companies_mentioned JSONB,
    dollar_amounts JSONB,
    key_quotes JSONB,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    used_in_newsletter INTEGER,
    CONSTRAINT valid_category CHECK (y2ai_category IN (
        'spending', 'constraints', 'data', 'policy', 
        'skepticism', 'smartmoney', 'china', 'energy', 'adoption'
    )),
    CONSTRAINT valid_sentiment CHECK (sentiment IN ('bullish', 'bearish', 'neutral'))
);

-- Collection runs metadata
CREATE TABLE IF NOT EXISTS collection_runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    sources_queried JSONB,
    raw_count INTEGER,
    processed_count INTEGER,
    errors JSONB,
    status VARCHAR(50) DEFAULT 'running'
);

-- Newsletter editions
CREATE TABLE IF NOT EXISTS newsletter_editions (
    id SERIAL PRIMARY KEY,
    edition_number INTEGER UNIQUE NOT NULL,
    title VARCHAR(500),
    published_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'draft',
    content_markdown TEXT,
    content_html TEXT,
    article_ids JSONB,
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_processed_category ON processed_articles(y2ai_category);
CREATE INDEX IF NOT EXISTS idx_processed_published ON processed_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_processed_impact ON processed_articles(impact_score DESC);
CREATE INDEX IF NOT EXISTS idx_raw_collected ON raw_articles(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_articles(source_type);
"""


# =============================================================================
# SUPABASE CLIENT
# =============================================================================

class SupabaseStorage:
    """Storage operations using Supabase/PostgreSQL"""
    
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.client = None
        
        if self.url and self.key:
            try:
                from supabase import create_client
                self.client = create_client(self.url, self.key)
                logger.info("Supabase client initialized")
            except ImportError:
                logger.warning("supabase-py not installed")
            except Exception as e:
                logger.error(f"Supabase connection error: {e}")
    
    def is_connected(self) -> bool:
        return self.client is not None
    
    # -------------------------------------------------------------------------
    # RAW ARTICLES
    # -------------------------------------------------------------------------
    
    """
    FINAL FIX: Handles missing article_hash AND fixes Supabase query error
    Replace insert_raw_articles and insert_processed_articles in argus1/storage.py
    """

    def insert_raw_articles(self, articles: List[Dict], collection_run_id: int = None) -> int:
        """Insert raw articles, using composite uniqueness (hash + date)"""
        if not self.is_connected():
            return 0

        inserted = 0
        for article in articles:
            try:
                if collection_run_id:
                    article["collection_run_id"] = collection_run_id

                # 🔥 FIX 1: Generate article_hash if missing
                if "article_hash" not in article or not article["article_hash"]:
                    import hashlib
                    url = article.get("url", "")
                    if url:
                        article["article_hash"] = hashlib.sha256(url.encode()).hexdigest()[:16]
                    else:
                        logger.warning(f"Article has no URL, skipping: {article.get('title', 'Unknown')[:50]}")
                        continue

                # Normalize keywords_used
                kw = article.get("keywords_used", None)
                if isinstance(kw, str):
                    if "," in kw:
                        normalized_kw = [k.strip() for k in kw.split(",") if k.strip()]
                    else:
                        normalized_kw = [kw] if kw.strip() else []
                elif isinstance(kw, list):
                    normalized_kw = [str(k).strip() for k in kw if k is not None and str(k).strip()]
                elif kw is None:
                    normalized_kw = None
                else:
                    normalized_kw = [str(kw)]

                if normalized_kw == []:
                    normalized_kw = None

                article["keywords_used"] = normalized_kw

                # 🔥 FIX 2: Use filter().gte().lte() instead of LIKE for date matching
                published_at = article.get("published_at", "")
                if not published_at:
                    logger.warning(f"Article has no published_at date, using current date: {article.get('title', 'Unknown')[:50]}")
                    from datetime import datetime
                    published_date = datetime.utcnow().strftime("%Y-%m-%d")
                    article["published_at"] = f"{published_date}T00:00:00Z"
                    published_at = article["published_at"]
                
                # Extract date and create date range
                if isinstance(published_at, str) and len(published_at) >= 10:
                    date_str = published_at[:10]  # YYYY-MM-DD
                    start_of_day = f"{date_str}T00:00:00"
                    end_of_day = f"{date_str}T23:59:59"
                else:
                    from datetime import datetime
                    date_str = datetime.utcnow().strftime("%Y-%m-%d")
                    start_of_day = f"{date_str}T00:00:00"
                    end_of_day = f"{date_str}T23:59:59"
                
                # Check if (hash + date) exists using date range
                existing = (
                    self.client.table("raw_articles")
                    .select("id")
                    .eq("article_hash", article["article_hash"])
                    .gte("published_at", start_of_day)
                    .lte("published_at", end_of_day)
                    .limit(1)
                    .execute()
                )
                
                # Only insert if this (hash + date) combination doesn't exist
                if not existing.data:
                    self.client.table("raw_articles").insert(article).execute()
                    inserted += 1
                    logger.debug(f"✅ Inserted raw: {article.get('title', 'Unknown')[:50]} ({date_str})")
                else:
                    logger.debug(f"⏭️  Skipping duplicate raw: {article.get('title', 'Unknown')[:50]} ({date_str})")
                    
            except Exception as e:
                logger.error(f"❌ Article insert error: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        return inserted


    # def insert_processed_articles(self, articles: List[Dict]) -> int:
    #     """Insert processed articles, using composite uniqueness (hash + date)"""
    #     if not self.is_connected():
    #         return 0

    #     inserted = 0
    #     for article in articles:
    #         try:
    #             # 🔥 FIX 1: Generate article_hash if missing
    #             if "article_hash" not in article or not article["article_hash"]:
    #                 import hashlib
    #                 url = article.get("url", "")
    #                 if url:
    #                     article["article_hash"] = hashlib.sha256(url.encode()).hexdigest()[:16]
    #                 else:
    #                     logger.warning(f"Processed article has no URL, skipping: {article.get('title', 'Unknown')[:50]}")
    #                     continue

    #             # Normalize keywords_used
    #             kw = article.get("keywords_used", None)

    #             if isinstance(kw, str):
    #                 if "," in kw:
    #                     normalized_kw = [k.strip() for k in kw.split(",") if k.strip()]
    #                 else:
    #                     normalized_kw = [kw] if kw.strip() else []
    #             elif isinstance(kw, list):
    #                 normalized_kw = [str(k).strip() for k in kw if k is not None and str(k).strip()]
    #             elif kw is None:
    #                 normalized_kw = None
    #             else:
    #                 normalized_kw = [str(kw)]

    #             if normalized_kw == []:
    #                 normalized_kw = None

    #             article["keywords_used"] = normalized_kw

    #             # 🔥 FIX 2: Use filter().gte().lte() instead of LIKE for date matching
    #             published_at = article.get("published_at", "")
    #             if not published_at:
    #                 logger.warning(f"Article has no published_at date, using current date: {article.get('title', 'Unknown')[:50]}")
    #                 from datetime import datetime
    #                 published_date = datetime.utcnow().strftime("%Y-%m-%d")
    #                 article["published_at"] = f"{published_date}T00:00:00Z"
    #                 published_at = article["published_at"]
                
    #             # Extract date and create date range
    #             if isinstance(published_at, str) and len(published_at) >= 10:
    #                 date_str = published_at[:10]  # YYYY-MM-DD
    #                 start_of_day = f"{date_str}T00:00:00"
    #                 end_of_day = f"{date_str}T23:59:59"
    #             else:
    #                 from datetime import datetime
    #                 date_str = datetime.utcnow().strftime("%Y-%m-%d")
    #                 start_of_day = f"{date_str}T00:00:00"
    #                 end_of_day = f"{date_str}T23:59:59"
                
    #             # Check if (hash + date) exists using date range
    #             existing = (
    #                 self.client.table("processed_articles")
    #                 .select("id")
    #                 .eq("article_hash", article["article_hash"])
    #                 .gte("published_at", start_of_day)
    #                 .lte("published_at", end_of_day)
    #                 .limit(1)
    #                 .execute()
    #             )
                
    #             # Only insert if this (hash + date) combination doesn't exist
    #             if not existing.data:
    #                 self.client.table("processed_articles").insert(article).execute()
    #                 inserted += 1
    #                 logger.debug(f"✅ Inserted processed: {article.get('title', 'Unknown')[:50]} ({date_str})")
    #             else:
    #                 logger.debug(f"⏭️  Skipping duplicate processed: {article.get('title', 'Unknown')[:50]} ({date_str})")

    #         except Exception as e:
    #             logger.error(f"❌ Processed article insert error (article_hash={article.get('article_hash')}): {e}")
    #             import traceback
    #             logger.debug(traceback.format_exc())

    #     return inserted
    
    
    def insert_processed_articles(self, articles: List[Dict]) -> int:
        """
        Insert processed articles, using composite uniqueness (hash + date).

        Keeps original logic: check exists by article_hash + published_at date-range,
        and INSERT only when not present (i.e. do NOT upsert/overwrite).

        Enhancements added:
        - Generate article_hash from URL when missing (skip if no URL)
        - Normalize keywords_used (string or list -> list or None)
        - Set processed_at to utcnow if missing
        - Defensive logging and per-row error handling
        """
        if not self.is_connected():
            return 0

        import hashlib
        from datetime import datetime
        inserted = 0

        for article in articles:
            try:
                # Defensive copy
                a = dict(article)

                # ------------------------------
                # 1) Ensure article_hash (generate from URL if missing)
                # ------------------------------
                if "article_hash" not in a or not a.get("article_hash"):
                    url = a.get("url", "") or ""
                    if url:
                        a["article_hash"] = hashlib.sha256(url.encode()).hexdigest()[:16]
                    else:
                        logger.warning(
                            "Processed article skipped (no article_hash and no url): %s",
                            a.get("title", "")[:120]
                        )
                        continue  # skip this row

                # ------------------------------
                # 2) Normalize keywords_used
                # ------------------------------
                kw = a.get("keywords_used", None)
                if isinstance(kw, str):
                    if "," in kw:
                        normalized_kw = [k.strip() for k in kw.split(",") if k.strip()]
                    else:
                        normalized_kw = [kw.strip()] if kw.strip() else []
                elif isinstance(kw, (list, tuple)):
                    normalized_kw = [str(k).strip() for k in kw if k is not None and str(k).strip()]
                elif kw is None:
                    normalized_kw = None
                else:
                    normalized_kw = [str(kw)]

                if normalized_kw == []:
                    # keep None (NULL in DB) when there are no keywords
                    normalized_kw = None

                a["keywords_used"] = normalized_kw

                # ------------------------------
                # 3) Ensure published_at is present (fallback to today midnight UTC)
                # ------------------------------
                published_at = a.get("published_at", "")
                if not published_at:
                    logger.warning(
                        "Processed article missing published_at, defaulting to today midnight UTC: %s",
                        a.get("title", "")[:120]
                    )
                    pub_date = datetime.utcnow().strftime("%Y-%m-%d")
                    a["published_at"] = f"{pub_date}T00:00:00Z"
                    published_at = a["published_at"]

                # ------------------------------
                # 4) Ensure processed_at is present (set to now if missing)
                # ------------------------------
                if not a.get("processed_at"):
                    a["processed_at"] = datetime.utcnow().isoformat()

                # ------------------------------
                # 5) Create start/end of day strings for published_at date-range check
                # ------------------------------
                if isinstance(published_at, str) and len(published_at) >= 10:
                    date_str = published_at[:10]  # YYYY-MM-DD
                    start_of_day = f"{date_str}T00:00:00"
                    end_of_day = f"{date_str}T23:59:59"
                else:
                    # fallback to today
                    date_str = datetime.utcnow().strftime("%Y-%m-%d")
                    start_of_day = f"{date_str}T00:00:00"
                    end_of_day = f"{date_str}T23:59:59"

                # ------------------------------
                # 6) Duplicate check: same article_hash and published_at within the same day
                #    (preserves your original behavior)
                # ------------------------------
                try:
                    existing = (
                        self.client.table("processed_articles")
                        .select("id")
                        .eq("article_hash", a["article_hash"])
                        .gte("published_at", start_of_day)
                        .lte("published_at", end_of_day)
                        .limit(1)
                        .execute()
                    )
                except Exception as e:
                    # Defensive: if the query fails, log and continue to next article
                    logger.error("Processed article select error (article_hash=%s): %s", a.get("article_hash"), e)
                    continue

                if not existing.data:
                    # Insert the article as-is (we normalized keywords & timestamps above)
                    try:
                        self.client.table("processed_articles").insert(a).execute()
                        inserted += 1
                        logger.debug("✅ Inserted processed: %s (%s)", a.get("title", "")[:80], date_str)
                    except Exception as e:
                        logger.error(
                            "Processed article insert error (article_hash=%s): %s",
                            a.get("article_hash"),
                            e
                        )
                else:
                    logger.debug("⏭️  Skipping duplicate processed: %s (%s)", a.get("title", "")[:80], date_str)

            except Exception as e:
                # Catch-all for per-row errors, continue with next article
                logger.exception("Unexpected error while inserting processed article: %s", e)

        return inserted


    
    def get_processed_articles(
        self,
        category: str = None,
        min_impact: float = None,
        sentiment: str = None,
        hours_back: int = 168,  # Default: 1 week
        limit: int = 100
    ) -> List[Dict]:
        """Retrieve processed articles with filters"""
        if not self.is_connected():
            return []
        
        query = self.client.table("processed_articles").select("*")
        
        if hours_back:
            cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            query = query.gte("processed_at", cutoff)
        
        if category:
            query = query.eq("y2ai_category", category)
        
        if min_impact:
            query = query.gte("impact_score", min_impact)
        
        if sentiment:
            query = query.eq("sentiment", sentiment)
        
        result = query.order("impact_score", desc=True).limit(limit).execute()
        return result.data
    
    def get_newsletter_ready_articles(self, days_back: int = 7) -> Dict[str, List[Dict]]:
        """Get articles organized for newsletter generation"""
        articles = self.get_processed_articles(hours_back=days_back * 24, limit=200)
        
        # Group by category
        by_category = {}
        for article in articles:
            cat = article.get("y2ai_category", "data")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(article)
        
        # Sort each category by impact
        for cat in by_category:
            by_category[cat].sort(key=lambda x: x.get("impact_score", 0), reverse=True)
        
        return by_category
    
    # -------------------------------------------------------------------------
    # COLLECTION RUNS
    # -------------------------------------------------------------------------
    
    def start_collection_run(self, sources: List[str]) -> int:
        """Start a new collection run, return run ID"""
        if not self.is_connected():
            return -1
        
        result = self.client.table("collection_runs").insert({
            "sources_queried": sources,
            "status": "running"
        }).execute()
        
        return result.data[0]["id"] if result.data else -1
    
    def complete_collection_run(
        self, 
        run_id: int, 
        raw_count: int, 
        processed_count: int,
        errors: List[str] = None
    ):
        """Mark collection run as complete"""
        if not self.is_connected():
            return
        
        self.client.table("collection_runs").update({
            "completed_at": datetime.utcnow().isoformat(),
            "raw_count": raw_count,
            "processed_count": processed_count,
            "errors": errors or [],
            "status": "completed"
        }).eq("id", run_id).execute()
    
    # -------------------------------------------------------------------------
    # NEWSLETTER EDITIONS
    # -------------------------------------------------------------------------
    
    def create_newsletter_draft(
        self, 
        edition_number: int, 
        title: str,
        content_markdown: str,
        article_ids: List[int]
    ) -> int:
        """Create a new newsletter draft"""
        if not self.is_connected():
            return -1
        
        result = self.client.table("newsletter_editions").insert({
            "edition_number": edition_number,
            "title": title,
            "content_markdown": content_markdown,
            "article_ids": article_ids,
            "status": "draft"
        }).execute()
        
        return result.data[0]["id"] if result.data else -1
    
    def publish_newsletter(self, edition_id: int, content_html: str):
        """Mark newsletter as published"""
        if not self.is_connected():
            return
        
        self.client.table("newsletter_editions").update({
            "published_at": datetime.utcnow().isoformat(),
            "content_html": content_html,
            "status": "published"
        }).eq("id", edition_id).execute()
    
    def get_latest_edition_number(self) -> int:
        """Get the latest newsletter edition number"""
        if not self.is_connected():
            return 0
        
        result = self.client.table("newsletter_editions")\
            .select("edition_number")\
            .order("edition_number", desc=True)\
            .limit(1)\
            .execute()
        
        return result.data[0]["edition_number"] if result.data else 0
    
    # -------------------------------------------------------------------------
    # ANALYTICS
    # -------------------------------------------------------------------------
    
    def get_collection_stats(self, days_back: int = 30) -> Dict:
        """Get collection statistics"""
        if not self.is_connected():
            return {}
        
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        
        # Count by source
        raw = self.client.table("raw_articles")\
            .select("source_type")\
            .gte("collected_at", cutoff)\
            .execute()
        
        source_counts = {}
        for r in raw.data:
            src = r["source_type"]
            source_counts[src] = source_counts.get(src, 0) + 1
        
        # Count by category
        processed = self.client.table("processed_articles")\
            .select("y2ai_category")\
            .gte("processed_at", cutoff)\
            .execute()
        
        category_counts = {}
        for p in processed.data:
            cat = p["y2ai_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "period_days": days_back,
            "total_raw": len(raw.data),
            "total_processed": len(processed.data),
            "by_source": source_counts,
            "by_category": category_counts
        }


# =============================================================================
# LOCAL SQLITE FALLBACK (for development/testing)
# =============================================================================

class SQLiteStorage:
    """SQLite fallback for local development"""
    
    def __init__(self, db_path: str = "argus1.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_hash TEXT UNIQUE,
                source_type TEXT,
                source_name TEXT,
                title TEXT,
                url TEXT,
                published_at TEXT,
                content TEXT,
                collected_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_hash TEXT UNIQUE,
                source_type TEXT,
                source_name TEXT,
                title TEXT,
                url TEXT,
                published_at TEXT,
                y2ai_category TEXT,
                extracted_facts TEXT,
                impact_score REAL,
                sentiment TEXT,
                companies_mentioned TEXT,
                dollar_amounts TEXT,
                key_quotes TEXT,
                processed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def insert_raw_articles(self, articles: List[Dict]) -> int:
        cursor = self.conn.cursor()
        inserted = 0
        
        for article in articles:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO raw_articles 
                    (article_hash, source_type, source_name, title, url, published_at, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.get("article_hash"),
                    article.get("source_type"),
                    article.get("source_name"),
                    article.get("title"),
                    article.get("url"),
                    article.get("published_at"),
                    article.get("content")
                ))
                inserted += cursor.rowcount
            except Exception as e:
                logger.error(f"SQLite insert error: {e}")
        
        self.conn.commit()
        return inserted
    
    def get_raw_articles(self, limit: int = 100) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM raw_articles 
            ORDER BY collected_at DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# STORAGE FACTORY
# =============================================================================

def get_storage():
    """Get appropriate storage backend based on environment"""
    supabase_url = os.getenv('SUPABASE_URL')
    
    if supabase_url:
        storage = SupabaseStorage()
        if storage.is_connected():
            return storage
    
    logger.info("Using SQLite fallback storage")
    return SQLiteStorage()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    storage = get_storage()
    
    print(f"\n{'='*60}")
    print("ARGUS-1 Storage Status")
    print(f"{'='*60}")
    
    if isinstance(storage, SupabaseStorage):
        print("Backend: Supabase/PostgreSQL")
        if storage.is_connected():
            stats = storage.get_collection_stats(days_back=7)
            print(f"Last 7 days: {stats.get('total_raw', 0)} raw, {stats.get('total_processed', 0)} processed")
            print(f"By source: {stats.get('by_source', {})}")
            print(f"By category: {stats.get('by_category', {})}")
    else:
        print("Backend: SQLite (local)")
        articles = storage.get_raw_articles(limit=5)
        print(f"Recent articles: {len(articles)}")
