"""
Y2AI STORAGE LAYER
PostgreSQL/Supabase schema and integration for the complete Y2AI system

Tables:
- bubble_index_daily: Daily VIX, CAPE, Credit Spread readings
- stock_tracker_daily: Daily stock and pillar performance
- social_posts: Published and scheduled social media posts
- Plus ARGUS-1 tables: raw_articles, processed_articles, newsletter_editions
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# COMPLETE DATABASE SCHEMA
# =============================================================================

SCHEMA_SQL = """
-- ============================================================================
-- Y2AI COMPLETE DATABASE SCHEMA
-- Replaces Google Sheets entirely
-- ============================================================================

-- ============================================================================
-- ARGUS-1 TABLES (News Intelligence)
-- ============================================================================

-- Raw articles from news sources
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
    collection_run_id INTEGER,
    keywords_used TEXT
);

-- Processed articles with Y2AI classification
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
    keywords_used TEXT,
    CONSTRAINT valid_category CHECK (y2ai_category IN (
        'spending', 'constraints', 'data', 'policy', 
        'skepticism', 'smartmoney', 'china', 'energy', 'adoption'
    )),
    CONSTRAINT valid_sentiment CHECK (sentiment IN ('bullish', 'bearish', 'neutral'))
);

-- Collection run metadata
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

-- ============================================================================
-- BUBBLE INDEX TABLES (Replaces Google Sheets formulas)
-- ============================================================================

-- Daily bubble index readings
CREATE TABLE IF NOT EXISTS bubble_index_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    
    -- Raw values
    vix FLOAT NOT NULL,
    cape FLOAT NOT NULL,
    credit_spread_ig FLOAT,
    credit_spread_hy FLOAT,
    
    -- Z-scores
    vix_zscore FLOAT,
    cape_zscore FLOAT,
    credit_zscore FLOAT,
    
    -- Calculated indices
    bubble_index FLOAT NOT NULL,  -- 0-100 scale
    bifurcation_score FLOAT NOT NULL,  -- -1 to +1 typically
    
    -- Regime
    regime VARCHAR(50) NOT NULL,  -- INFRASTRUCTURE, ADOPTION, TRANSITION, BUBBLE_WARNING
    
    -- Metadata
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- STOCK TRACKER TABLES (Replaces GOOGLEFINANCE)
-- ============================================================================

-- Daily stock readings
CREATE TABLE IF NOT EXISTS stock_readings (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    name VARCHAR(100),
    pillar VARCHAR(50),
    price FLOAT NOT NULL,
    change_today FLOAT,
    change_5day FLOAT,
    change_ytd FLOAT,
    retrieved_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

-- Daily stock tracker reports
CREATE TABLE IF NOT EXISTS stock_tracker_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    
    -- Y2AI Index performance
    y2ai_index_today FLOAT,
    y2ai_index_5day FLOAT,
    y2ai_index_ytd FLOAT,
    
    -- Benchmarks
    spy_today FLOAT,
    spy_5day FLOAT,
    spy_ytd FLOAT,
    qqq_today FLOAT,
    qqq_5day FLOAT,
    qqq_ytd FLOAT,
    
    -- Signals
    status VARCHAR(50),  -- VALIDATING, NEUTRAL, CONTRADICTING
    best_stock VARCHAR(10),
    worst_stock VARCHAR(10),
    best_pillar VARCHAR(50),
    worst_pillar VARCHAR(50),
    
    -- Full data
    stocks JSONB,  -- Array of stock readings
    pillars JSONB,  -- Array of pillar performance
    
    -- Metadata
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- SOCIAL PUBLISHING TABLES
-- ============================================================================

-- Social media posts
CREATE TABLE IF NOT EXISTS social_posts (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,  -- twitter, linkedin, bluesky
    content TEXT NOT NULL,
    thread JSONB,  -- Array of tweet texts for threads
    image_path TEXT,
    scheduled_at TIMESTAMPTZ,
    posted_at TIMESTAMPTZ,
    post_url TEXT,
    status VARCHAR(20) DEFAULT 'draft',  -- draft, scheduled, posted, failed
    post_type VARCHAR(50),  -- daily_tracker, newsletter, breaking_news
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_processed_category ON processed_articles(y2ai_category);
CREATE INDEX IF NOT EXISTS idx_processed_published ON processed_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_processed_impact ON processed_articles(impact_score DESC);
CREATE INDEX IF NOT EXISTS idx_raw_collected ON raw_articles(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_articles(source_type);
CREATE INDEX IF NOT EXISTS idx_bubble_date ON bubble_index_daily(date DESC);
CREATE INDEX IF NOT EXISTS idx_stocks_date ON stock_readings(date DESC);
CREATE INDEX IF NOT EXISTS idx_tracker_date ON stock_tracker_daily(date DESC);
CREATE INDEX IF NOT EXISTS idx_social_status ON social_posts(status);
CREATE INDEX IF NOT EXISTS idx_social_scheduled ON social_posts(scheduled_at);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Latest bubble index reading
CREATE OR REPLACE VIEW v_latest_bubble_index AS
SELECT * FROM bubble_index_daily 
ORDER BY date DESC 
LIMIT 1;

-- Latest stock tracker report
CREATE OR REPLACE VIEW v_latest_stock_tracker AS
SELECT * FROM stock_tracker_daily 
ORDER BY date DESC 
LIMIT 1;

-- This week's high-impact articles
CREATE OR REPLACE VIEW v_weekly_highlights AS
SELECT * FROM processed_articles
WHERE processed_at > NOW() - INTERVAL '7 days'
AND impact_score >= 0.7
ORDER BY impact_score DESC;

-- Pending social posts
CREATE OR REPLACE VIEW v_pending_posts AS
SELECT * FROM social_posts
WHERE status = 'scheduled'
AND scheduled_at <= NOW()
ORDER BY scheduled_at;
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
    
    
    
        
        
    def get_newsletter_by_edition(self, edition_number: int) -> Optional[Dict]:
        """
        Fetch a single newsletter edition by its edition_number.
        """
        if not self.is_connected():
            return None

        try:
            result = (
                self.client
                    .table("newsletter_editions")
                    .select("*")
                    .eq("edition_number", edition_number)
                    .limit(1)
                    .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching newsletter edition {edition_number}: {e}")
            return None
        
        
    def save_newsletter_edition(
        self,
        edition_number: int,
        title: str,
        content_markdown: str,
        status: str = "published",
        article_ids: List[int] = None
    ) -> Optional[int]:
        """
        Upsert a newsletter edition row by edition_number.
        """
        if not self.is_connected():
            return None

        payload = {
            "edition_number": edition_number,
            "title": title,
            "content_markdown": content_markdown,
            "status": status,
            "article_ids": article_ids or [],
        }

        try:
            result = (
                self.client
                    .table("newsletter_editions")
                    .upsert(payload)
                    .execute()
            )
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Error saving newsletter edition {edition_number}: {e}")
            return None
    
    

    
    # -------------------------------------------------------------------------
    # BUBBLE INDEX OPERATIONS
    # -------------------------------------------------------------------------
    
    def store_bubble_reading(self, reading: Dict) -> bool:
        """Store a bubble index reading"""
        if not self.is_connected():
            return False
        
        try:
            self.client.table("bubble_index_daily").upsert(
                reading,
                on_conflict="date"
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Bubble index store error: {e}")
            return False
    
    def get_latest_bubble_reading(self) -> Optional[Dict]:
        """Get most recent bubble index reading"""
        if not self.is_connected():
            return None
        
        result = self.client.table("bubble_index_daily")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        return result.data[0] if result.data else None
    
    def get_bubble_history(self, days: int = 30) -> List[Dict]:
        """Get bubble index history"""
        if not self.is_connected():
            return []
        
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        result = self.client.table("bubble_index_daily")\
            .select("*")\
            .gte("date", cutoff)\
            .order("date", desc=True)\
            .execute()
        
        return result.data
    
    # -------------------------------------------------------------------------
    # STOCK TRACKER OPERATIONS
    # -------------------------------------------------------------------------
    
    def store_stock_report(self, report: Dict) -> bool:
        """Store a stock tracker report"""
        if not self.is_connected():
            return False
        
        try:
            self.client.table("stock_tracker_daily").upsert(
                report,
                on_conflict="date"
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Stock report store error: {e}")
            return False
    
    def get_latest_stock_report(self) -> Optional[Dict]:
        """Get most recent stock tracker report"""
        if not self.is_connected():
            return None
        
        result = self.client.table("stock_tracker_daily")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        return result.data[0] if result.data else None
    
    # -------------------------------------------------------------------------
    # SOCIAL POSTS OPERATIONS
    # -------------------------------------------------------------------------
    
    def schedule_post(self, post: Dict) -> Optional[int]:
        """Schedule a social media post"""
        if not self.is_connected():
            return None
        
        try:
            result = self.client.table("social_posts").insert(post).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Post schedule error: {e}")
            return None
    
    def get_pending_posts(self) -> List[Dict]:
        """Get posts ready to be published"""
        if not self.is_connected():
            return []
        
        result = self.client.table("social_posts")\
            .select("*")\
            .eq("status", "scheduled")\
            .lte("scheduled_at", datetime.utcnow().isoformat())\
            .execute()
        
        return result.data
    
    def mark_post_published(self, post_id: int, post_url: str) -> bool:
        """Mark a post as published"""
        if not self.is_connected():
            return False
        
        try:
            self.client.table("social_posts").update({
                "status": "posted",
                "posted_at": datetime.utcnow().isoformat(),
                "post_url": post_url
            }).eq("id", post_id).execute()
            return True
        except Exception as e:
            logger.error(f"Post update error: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # ARTICLE OPERATIONS (for ARGUS-1 integration)
    # -------------------------------------------------------------------------
    
    def insert_raw_articles(self, articles: List[Dict], collection_run_id: int = None) -> int:
        """Insert raw articles, using composite uniqueness (hash + date)"""
        if not self.is_connected():
            return 0

        inserted = 0
        for article in articles:
            try:
                if collection_run_id:
                    article["collection_run_id"] = collection_run_id

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

                # 🔥 FIX: Check uniqueness based on BOTH hash AND published date
                # This allows same article on different dates, but prevents duplicates on same date
                published_date = article.get("published_at", "")[:10] if article.get("published_at") else ""
                
                if not published_date:
                    logger.warning(f"Article has no published_at date, using current date: {article.get('title', 'Unknown')[:50]}")
                    from datetime import datetime
                    published_date = datetime.utcnow().strftime("%Y-%m-%d")
                    article["published_at"] = f"{published_date}T00:00:00Z"
                
                existing = (
                    self.client.table("raw_articles")
                    .select("id")
                    .eq("article_hash", article["article_hash"])
                    .like("published_at", f"{published_date}%")  # Same day
                    .limit(1)
                    .execute()
                )
                
                # Only insert if this (hash + date) combination doesn't exist
                if not existing.data:
                    self.client.table("raw_articles").insert(article).execute()
                    inserted += 1
                    logger.debug(f"✅ Inserted: {article.get('title', 'Unknown')[:50]} ({published_date})")
                else:
                    logger.debug(f"⏭️  Skipping duplicate: {article.get('title', 'Unknown')[:50]} ({published_date})")
                    
            except Exception as e:
                logger.error(f"❌ Article insert error: {e}")

        return inserted
    
    
    def insert_processed_articles(self, articles: List[Dict]) -> int:
        """Insert processed articles, using composite uniqueness (hash + date)"""
        if not self.is_connected():
            return 0

        inserted = 0
        for article in articles:
            try:
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

                # 🔥 FIX: Check uniqueness based on BOTH hash AND published date
                # This allows same article on different dates, but prevents duplicates on same date
                published_date = article.get("published_at", "")[:10] if article.get("published_at") else ""
                
                if not published_date:
                    logger.warning(f"Article has no published_at date, using current date: {article.get('title', 'Unknown')[:50]}")
                    from datetime import datetime
                    published_date = datetime.utcnow().strftime("%Y-%m-%d")
                    article["published_at"] = f"{published_date}T00:00:00Z"
                
                existing = (
                    self.client.table("processed_articles")
                    .select("id")
                    .eq("article_hash", article["article_hash"])
                    .like("published_at", f"{published_date}%")  # Same day
                    .limit(1)
                    .execute()
                )
                
                # Only insert if this (hash + date) combination doesn't exist
                if not existing.data:
                    self.client.table("processed_articles").insert(article).execute()
                    inserted += 1
                    logger.debug(f"✅ Inserted: {article.get('title', 'Unknown')[:50]} ({published_date})")
                else:
                    logger.debug(f"⏭️  Skipping duplicate: {article.get('title', 'Unknown')[:50]} ({published_date})")

            except Exception as e:
                logger.error(f"❌ Processed article insert error (article_hash={article.get('article_hash')}): {e}")

        return inserted

    
    
    def get_newsletter_ready_articles(self, days_back: int = 7) -> Dict[str, List[Dict]]:
        """Get articles organized for newsletter generation"""
        if not self.is_connected():
            return {}
        
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        
        result = self.client.table("processed_articles")\
            .select("*")\
            .gte("processed_at", cutoff)\
            .order("impact_score", desc=True)\
            .limit(200)\
            .execute()
        
        # Group by category
        by_category = {}
        for article in result.data:
            cat = article.get("y2ai_category", "data")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(article)
        
        return by_category
    
    # -------------------------------------------------------------------------
    # NEWSLETTER OPERATIONS
    # -------------------------------------------------------------------------
    
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
    
    def create_newsletter_draft(
        self,
        edition_number: int,
        title: str,
        content_markdown: str,
        article_ids: List[int] = None
    ) -> Optional[int]:
        """Create a new newsletter draft"""
        if not self.is_connected():
            return None
        
        try:
            result = self.client.table("newsletter_editions").insert({
                "edition_number": edition_number,
                "title": title,
                "content_markdown": content_markdown,
                "article_ids": article_ids or [],
                "status": "draft"
            }).execute()
            
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Newsletter create error: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # DASHBOARD DATA (for website)
    # -------------------------------------------------------------------------
    
    def get_dashboard_data(self) -> Dict:
        """
        Get all data needed for the Y2AI website dashboard
        
        This replaces the Google Apps Script API endpoint.
        """
        data = {
            "bubble_index": None,
            "stock_tracker": None,
            "recent_articles": [],
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Get latest bubble index
        bubble = self.get_latest_bubble_reading()
        if bubble:
            data["bubble_index"] = {
                "value": bubble.get("bubble_index"),
                "vix": bubble.get("vix"),
                "credit_spread": bubble.get("credit_spread_ig"),
                "bifurcation_score": bubble.get("bifurcation_score"),
                "regime": bubble.get("regime"),
                "date": bubble.get("date")
            }
        
        # Get latest stock tracker
        tracker = self.get_latest_stock_report()
        if tracker:
            data["stock_tracker"] = {
                "y2ai_index": tracker.get("y2ai_index_today"),
                "spy": tracker.get("spy_today"),
                "status": tracker.get("status"),
                "date": tracker.get("date")
            }
        
        # Get recent high-impact articles
        articles = self.get_newsletter_ready_articles(days_back=7)
        high_impact = []
        for cat_articles in articles.values():
            high_impact.extend([a for a in cat_articles if a.get("impact_score", 0) >= 0.7][:3])
        data["recent_articles"] = sorted(
            high_impact,
            key=lambda x: x.get("impact_score", 0),
            reverse=True
        )[:5]
        
        return data


# =============================================================================
# SQLITE FALLBACK (for local development)
# =============================================================================

class SQLiteStorage:
    """SQLite fallback for local development"""
    
    def __init__(self, db_path: str = "y2ai.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bubble_index_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                vix REAL, cape REAL,
                credit_spread_ig REAL, credit_spread_hy REAL,
                vix_zscore REAL, cape_zscore REAL, credit_zscore REAL,
                bubble_index REAL, bifurcation_score REAL,
                regime TEXT,
                calculated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_tracker_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                y2ai_index_today REAL, y2ai_index_5day REAL, y2ai_index_ytd REAL,
                spy_today REAL, spy_5day REAL, spy_ytd REAL,
                qqq_today REAL, qqq_5day REAL, qqq_ytd REAL,
                status TEXT,
                best_stock TEXT, worst_stock TEXT,
                best_pillar TEXT, worst_pillar TEXT,
                stocks TEXT, pillars TEXT,
                calculated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def is_connected(self) -> bool:
        return True
    
    def store_bubble_reading(self, reading: Dict) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO bubble_index_daily 
            (date, vix, cape, credit_spread_ig, credit_spread_hy,
             vix_zscore, cape_zscore, credit_zscore,
             bubble_index, bifurcation_score, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reading.get("date"),
            reading.get("vix"),
            reading.get("cape"),
            reading.get("credit_spread_ig"),
            reading.get("credit_spread_hy"),
            reading.get("vix_zscore"),
            reading.get("cape_zscore"),
            reading.get("credit_zscore"),
            reading.get("bubble_index"),
            reading.get("bifurcation_score"),
            reading.get("regime")
        ))
        self.conn.commit()
        return True
    
    def get_latest_bubble_reading(self) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM bubble_index_daily ORDER BY date DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else None


# =============================================================================
# FACTORY FUNCTION
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
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--schema":
        print(SCHEMA_SQL)
    else:
        storage = get_storage()
        
        print(f"\n{'='*60}")
        print("Y2AI Storage Status")
        print(f"{'='*60}")
        
        if isinstance(storage, SupabaseStorage):
            print("Backend: Supabase/PostgreSQL")
            if storage.is_connected():
                # Check latest data
                bubble = storage.get_latest_bubble_reading()
                tracker = storage.get_latest_stock_report()
                
                print(f"\nLatest Bubble Index: {bubble.get('date') if bubble else 'None'}")
                print(f"Latest Stock Tracker: {tracker.get('date') if tracker else 'None'}")
        else:
            print("Backend: SQLite (local)")
