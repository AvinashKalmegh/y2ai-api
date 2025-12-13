"""
ARGUS-1 NEWS INTELLIGENCE LAYER (ENHANCED)
With robust error handling, retry logic, circuit breakers, and rate limiting

Sources:
- NewsAPI (optional, requires paid key for production)
- Alpha Vantage (free, 25 req/day) 
- SEC EDGAR (free, unlimited)
- RSS Feeds (free, unlimited) - 30+ curated sources
"""

import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

# Import resilience module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.resilience import (
    resilient_call,
    with_fallback,
    aggregate_with_partial_failure,
    get_http_session,
    get_circuit_breaker,
    get_health_tracker,
    get_system_status,
    CircuitOpenError,
    RateLimitError,
    RetryExhaustedError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS (unchanged)
# =============================================================================

@dataclass
class RawArticle:
    """Raw article from any source before processing"""
    source_type: str
    source_name: str
    title: str
    url: str
    published_at: str
    content: str
    author: Optional[str] = None
    ticker: Optional[str] = None
    relevance_signals: Optional[dict] = None
    keywords_used: Optional[List[str]] = None
   
    @property
    def article_hash(self) -> str:
        """Unique hash for deduplication"""
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class ProcessedArticle:
    """Article after Claude extraction and Y2AI classification"""
    article_hash: str
    source_type: str
    source_name: str
    title: str
    url: str
    published_at: str
    y2ai_category: str
    extracted_facts: List[str]
    impact_score: float
    sentiment: str
    companies_mentioned: List[str]
    dollar_amounts: List[str]
    key_quotes: List[str]
    processed_at: str
    keywords_used: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# SOURCE ADAPTERS (Enhanced with resilience)
# =============================================================================

class SourceAdapter(ABC):
    """Base class for all news source adapters"""
    
    def __init__(self):
        self._session = get_http_session()
    
    @abstractmethod
    def fetch(self, hours_back: int = 24) -> List[RawArticle]:
        """Fetch articles from source"""
        pass
    
    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source"""
        pass
    
    def is_available(self) -> bool:
        """Check if this adapter can currently accept requests"""
        circuit = get_circuit_breaker(self.source_id)
        return circuit.can_execute()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health metrics for this adapter"""
        tracker = get_health_tracker(self.source_id)
        return tracker.to_dict()


class NewsAPIAdapter(SourceAdapter):
    """NewsAPI.org adapter - 80,000+ sources"""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('NEWSAPI_KEY')
        self.base_url = "https://newsapi.org/v2"
        
        # Y2AI-specific queries
        self.queries = [
            '("AI" OR "artificial intelligence") AND ("capex" OR "capital expenditure" OR "spending")',
            '("data center" OR "datacenter") AND ("construction" OR "investment" OR "billion")',
            '("GPU" OR "NVIDIA" OR "H100") AND ("shortage" OR "supply" OR "demand")',
            '"hyperscaler" AND ("infrastructure" OR "spending" OR "investment")',
            '("Microsoft" OR "Google" OR "Amazon" OR "Meta") AND ("AI infrastructure" OR "AI spending")',
            '"earnings" AND ("AI" OR "artificial intelligence") AND ("capex" OR "guidance")',
        ]
    
    @property
    def source_id(self) -> str:
        return "newsapi"
    
    @resilient_call(
        service_name="newsapi",
        max_retries=3,
        base_delay=2.0,
        use_circuit_breaker=True,
        use_rate_limiter=True,
    )
    def _fetch_query(self, query: str, from_time: str) -> List[Dict]:
        """Fetch articles for a single query with resilience"""
        response = self._session.get(
            f"{self.base_url}/everything",
            params={
                "q": query,
                "from": from_time,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": 'a3db44b650b14919aceab6f0be1458b3'
            },
            timeout=30
        )
        
        # Check for rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)
            raise RateLimitError("newsapi", int(retry_after))
        
        response.raise_for_status()
        return response.json().get("articles", [])
    
    def fetch(self, hours_back: int = 24) -> List[RawArticle]:
        # if not self.api_key:
        #     logger.warning("NewsAPI key not set, skipping")
        #     return []
        
        if not self.is_available():
            logger.warning("NewsAPI circuit breaker is open, skipping")
            return []
        
        articles = []
        from_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        
        for query in self.queries:
            try:
                items = self._fetch_query(query, from_time)
                for item in items:
                    articles.append(RawArticle(
                        source_type="newsapi",
                        source_name=item.get("source", {}).get("name", "Unknown"),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        published_at=item.get("publishedAt", ""),
                        content=item.get("content", "") or item.get("description", ""),
                        author=item.get("author")
                    ))
            except CircuitOpenError:
                logger.warning(f"NewsAPI circuit open, stopping queries")
                break
            except RateLimitError as e:
                logger.warning(f"NewsAPI rate limited: {e}")
                break
            except RetryExhaustedError as e:
                logger.error(f"NewsAPI query failed after retries: {e}")
                continue
            except Exception as e:
                logger.error(f"NewsAPI unexpected error: {e}")
                continue
        
        logger.info(f"NewsAPI fetched {len(articles)} articles")
        return articles


class AlphaVantageAdapter(SourceAdapter):
    """Alpha Vantage News Sentiment API - Financial news with sentiment"""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('ALPHAVANTAGE_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        
        # Y2AI-relevant tickers
        self.tickers = [
            "MSFT", "GOOGL", "AMZN", "META", "NVDA",
            "AMD", "INTC", "TSM", "AVGO",
            "ORCL", "IBM", "CRM", "NOW",
            "EQIX", "DLR", "AMT",
        ]
    
    @property
    def source_id(self) -> str:
        return "alphavantage"
    
    @resilient_call(
        service_name="alphavantage",
        max_retries=2,
        base_delay=5.0,  # Alpha Vantage is slow
        use_circuit_breaker=True,
        use_rate_limiter=True,
    )
    def _fetch_batch(self, ticker_batch: str, time_from: str) -> List[Dict]:
        """Fetch news for a batch of tickers with resilience"""
        response = self._session.get(
            self.base_url,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker_batch,
                "time_from": time_from,
                "limit": 50,
                "apikey": 'OWB57KP9OQHRJLXR'
            },
            timeout=60  # Alpha Vantage can be slow
        )
        
        # Alpha Vantage returns 200 with error in body when rate limited
        data = response.json()
        if "Note" in data or "Information" in data:
            error_msg = data.get("Note") or data.get("Information")
            if "call frequency" in error_msg.lower() or "limit" in error_msg.lower():
                raise RateLimitError("alphavantage")
            raise Exception(error_msg)
        
        response.raise_for_status()
        return data.get("feed", [])
    
    def fetch(self, hours_back: int = 24) -> List[RawArticle]:
        # if not self.api_key:
        #     logger.warning("Alpha Vantage key not set, skipping")
        #     return []
        
        if not self.is_available():
            logger.warning("Alpha Vantage circuit breaker is open, skipping")
            return []
        
        articles = []
        time_from = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M")
        
        # Process tickers in batches of 5 (API limit)
        for i in range(0, len(self.tickers), 5):
            ticker_batch = ",".join(self.tickers[i:i+5])
            
            try:
                items = self._fetch_batch(ticker_batch, time_from)
                for item in items:
                    ticker_sentiment = item.get("ticker_sentiment", [])
                    primary_ticker = ticker_sentiment[0].get("ticker") if ticker_sentiment else None
                    
                    articles.append(RawArticle(
                        source_type="alphavantage",
                        source_name=item.get("source", "Unknown"),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        published_at=item.get("time_published", ""),
                        content=item.get("summary", ""),
                        author=", ".join(item.get("authors", [])),
                        ticker=primary_ticker,
                        relevance_signals={
                            "overall_sentiment": item.get("overall_sentiment_label"),
                            "sentiment_score": item.get("overall_sentiment_score"),
                            "ticker_sentiment": ticker_sentiment
                        }
                    ))
            except CircuitOpenError:
                logger.warning("Alpha Vantage circuit open, stopping")
                break
            except RateLimitError:
                logger.warning("Alpha Vantage rate limited, stopping")
                break
            except RetryExhaustedError as e:
                logger.error(f"Alpha Vantage batch failed after retries: {e}")
                continue
            except Exception as e:
                logger.error(f"Alpha Vantage unexpected error: {e}")
                continue
        
        logger.info(f"Alpha Vantage fetched {len(articles)} articles")
        return articles


class SECEdgarAdapter(SourceAdapter):
    """SEC EDGAR filings - 8-K, 10-K, 10-Q for AI infrastructure companies"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://efts.sec.gov/LATEST/search-index"
        
        self.company_ciks = {
            "0000789019": "Microsoft",
            "0001652044": "Alphabet/Google",
            "0001018724": "Amazon",
            "0001326801": "Meta",
            "0001045810": "NVIDIA",
            "0000002488": "AMD",
            "0000050863": "Intel",
            "0001101239": "Oracle",
        }
        
        self.form_types = ["8-K", "10-K", "10-Q"]
        
        self.search_terms = [
            '"capital expenditure" AND "artificial intelligence"',
            '"data center" AND "investment"',
            '"AI infrastructure"',
            '"GPU" OR "computing infrastructure"'
        ]
    
    @property
    def source_id(self) -> str:
        return "sec_edgar"
    
    @resilient_call(
        service_name="sec_edgar",
        max_retries=3,
        base_delay=1.0,
        use_circuit_breaker=True,
        use_rate_limiter=False,  # SEC has generous limits
    )
    def _search_filings(self, term: str, start_date: str, end_date: str) -> List[Dict]:
        """Search SEC filings with resilience"""
        response = self._session.get(
            self.base_url,
            params={
                "q": term,
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "forms": ",".join(self.form_types)
            },
            headers={"User-Agent": "Y2AI Research contact@y2ai.us"},
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        return data.get("hits", {}).get("hits", [])
    
    def fetch(self, hours_back: int = 24) -> List[RawArticle]:
        if not self.is_available():
            logger.warning("SEC EDGAR circuit breaker is open, skipping")
            return []
        
        articles = []
        start_date = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        for term in self.search_terms:
            try:
                hits = self._search_filings(term, start_date, end_date)
                for hit in hits:
                    source = hit.get("_source", {})
                    articles.append(RawArticle(
                        source_type="sec_edgar",
                        source_name=f"SEC EDGAR - {source.get('form', 'Filing')}",
                        title=f"{source.get('display_names', ['Unknown'])[0]} - {source.get('form', 'Filing')}",
                        url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={source.get('ciks', [''])[0]}",
                        published_at=source.get("file_date", ""),
                        content=source.get("file_description", ""),
                        ticker=None,
                        relevance_signals={
                            "form_type": source.get("form"),
                            "company_names": source.get("display_names", [])
                        }
                    ))
            except RetryExhaustedError as e:
                logger.error(f"SEC EDGAR search failed after retries: {e}")
                continue
            except Exception as e:
                logger.error(f"SEC EDGAR unexpected error: {e}")
                continue
        
        logger.info(f"SEC EDGAR fetched {len(articles)} filings")
        return articles


class RSSAdapter(SourceAdapter):
    """RSS feed aggregator for curated premium sources"""
    
    def __init__(self):
        super().__init__()
        
        # Curated RSS feeds with health tracking per feed
        self.feeds = {
            # Technology
            "reuters_tech": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best&best-topics=tech",
            "ars_tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
            "verge_tech": "https://www.theverge.com/rss/index.xml",
            
            # Business/Finance
            "fortune": "https://fortune.com/feed/",
            "bloomberg_tech": "https://feeds.bloomberg.com/technology/news.rss",
            
            # AI Specific
            "mit_ai": "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
            "ai_news": "https://www.artificialintelligence-news.com/feed/",
            
            # Data Centers/Infrastructure
            "datacenter_knowledge": "https://www.datacenterknowledge.com/rss.xml",
            "datacenter_dynamics": "https://www.datacenterdynamics.com/en/rss/",
            
            # Energy
            "utility_dive": "https://www.utilitydive.com/feeds/news/",
            "power_magazine": "https://www.powermag.com/feed/",
            
            # Government/Policy
            "route_fifty": "https://www.route-fifty.com/rss/technology/",
            "nextgov": "https://www.nextgov.com/rss/all/",
        }
        
        # Track failed feeds
        self._failed_feeds: Dict[str, datetime] = {}
        self._feed_failure_threshold = 3
        self._feed_cooldown_minutes = 60
        
        # Keywords for relevance filtering
        self.keywords = [
            "ai", "artificial intelligence", "data center", "datacenter",
            "gpu", "nvidia", "infrastructure", "capex", "capital expenditure",
            "hyperscaler", "cloud", "computing", "semiconductor", "chip",
            "microsoft", "google", "amazon", "meta", "openai"
        ]
    
    @property
    def source_id(self) -> str:
        return "rss"
    
    def _is_feed_available(self, feed_name: str) -> bool:
        """Check if a feed is available (not in cooldown)"""
        if feed_name in self._failed_feeds:
            failed_time = self._failed_feeds[feed_name]
            if (datetime.utcnow() - failed_time).total_seconds() < self._feed_cooldown_minutes * 60:
                return False
            # Cooldown expired, remove from failed list
            del self._failed_feeds[feed_name]
        return True
    
    def _mark_feed_failed(self, feed_name: str):
        """Mark a feed as failed"""
        self._failed_feeds[feed_name] = datetime.utcnow()
        logger.warning(f"RSS feed {feed_name} marked as failed, cooldown {self._feed_cooldown_minutes}min")
    
    @resilient_call(
        service_name="rss",
        max_retries=2,
        base_delay=1.0,
        use_circuit_breaker=False,  # Per-feed tracking instead
        use_rate_limiter=False,
    )
    def _fetch_feed(self, feed_name: str, feed_url: str) -> List[Dict]:
        """Fetch a single RSS feed with resilience"""
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser not installed")
        
        # feedparser doesn't raise on HTTP errors, so we do a HEAD check first
        head_response = self._session.head(feed_url, timeout=10, allow_redirects=True)
        if head_response.status_code >= 400:
            raise Exception(f"Feed returned HTTP {head_response.status_code}")
        
        feed = feedparser.parse(feed_url)
        
        # Check for parse errors
        if feed.bozo and feed.bozo_exception:
            raise Exception(f"Feed parse error: {feed.bozo_exception}")
        
        return feed.entries
    
    def fetch(self, hours_back: int = 24) -> List[RawArticle]:
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed, skipping RSS")
            return []
        
        articles = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        successful_feeds = 0
        failed_feeds = 0
        
        for feed_name, feed_url in self.feeds.items():
            # Skip feeds in cooldown
            if not self._is_feed_available(feed_name):
                logger.debug(f"RSS feed {feed_name} in cooldown, skipping")
                continue
            
            try:
                entries = self._fetch_feed(feed_name, feed_url)
                feed_articles = 0
                
                for entry in entries:
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Skip if older than cutoff
                    if pub_date and pub_date < cutoff_time:
                        continue
                    
                    # Check relevance via keywords
                    text_to_check = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                    if not any(kw in text_to_check for kw in self.keywords):
                        continue
                    
                    articles.append(RawArticle(
                        source_type="rss",
                        source_name=feed_name,
                        title=entry.get("title", ""),
                        url=entry.get("link", ""),
                        published_at=pub_date.isoformat() if pub_date else "",
                        content=entry.get("summary", ""),
                        author=entry.get("author")
                    ))
                    feed_articles += 1
                
                successful_feeds += 1
                if feed_articles > 0:
                    logger.debug(f"RSS feed {feed_name}: {feed_articles} relevant articles")
                    
            except RetryExhaustedError:
                self._mark_feed_failed(feed_name)
                failed_feeds += 1
            except Exception as e:
                logger.warning(f"RSS feed {feed_name} error: {e}")
                self._mark_feed_failed(feed_name)
                failed_feeds += 1
        
        logger.info(
            f"RSS fetched {len(articles)} articles from {successful_feeds} feeds "
            f"({failed_feeds} failed)"
        )
        return articles
    
    def get_feed_status(self) -> Dict[str, Any]:
        """Get status of all RSS feeds"""
        status = {}
        for feed_name in self.feeds:
            if feed_name in self._failed_feeds:
                failed_time = self._failed_feeds[feed_name]
                cooldown_remaining = self._feed_cooldown_minutes * 60 - \
                    (datetime.utcnow() - failed_time).total_seconds()
                status[feed_name] = {
                    "available": False,
                    "cooldown_remaining_seconds": max(0, int(cooldown_remaining))
                }
            else:
                status[feed_name] = {"available": True}
        return status


# =============================================================================
# AGGREGATOR ORCHESTRATOR (Enhanced)
# =============================================================================

class NewsAggregator:
    """Main orchestrator that coordinates all source adapters with resilience"""
    
    def __init__(self, adapters: List[SourceAdapter] = None):
        if adapters is None:
            self.adapters = [
                NewsAPIAdapter(),
                AlphaVantageAdapter(),
                SECEdgarAdapter(),
                RSSAdapter(),
            ]
        else:
            self.adapters = adapters
        
        self.seen_hashes = set()
    
    def collect_all(self, hours_back: int = 24) -> List[RawArticle]:
        """
        Collect from all sources with graceful degradation.
        
        If some adapters fail, we still return results from successful ones.
        """
        all_articles = []
        adapter_results = {}
        
        for adapter in self.adapters:
            logger.info(f"Fetching from {adapter.source_id}...")
            
            try:
                articles = adapter.fetch(hours_back=hours_back)
                adapter_results[adapter.source_id] = {
                    "success": True,
                    "count": len(articles)
                }
                logger.info(f"  Got {len(articles)} articles from {adapter.source_id}")
                all_articles.extend(articles)
                
            except CircuitOpenError as e:
                adapter_results[adapter.source_id] = {
                    "success": False,
                    "error": "circuit_open",
                    "reset_time": e.reset_time.isoformat()
                }
                logger.warning(f"  {adapter.source_id} circuit open, skipping")
                
            except RateLimitError as e:
                adapter_results[adapter.source_id] = {
                    "success": False,
                    "error": "rate_limited"
                }
                logger.warning(f"  {adapter.source_id} rate limited, skipping")
                
            except Exception as e:
                adapter_results[adapter.source_id] = {
                    "success": False,
                    "error": str(e)[:100]
                }
                logger.error(f"  Adapter {adapter.source_id} failed: {e}")
        
        # Deduplicate by URL hash
        unique_articles = []
        for article in all_articles:
            if article.article_hash not in self.seen_hashes:
                self.seen_hashes.add(article.article_hash)
                unique_articles.append(article)
        
        # Log summary
        successful = sum(1 for r in adapter_results.values() if r.get("success"))
        logger.info(
            f"Collection complete: {len(unique_articles)} unique articles "
            f"(from {successful}/{len(self.adapters)} adapters)"
        )
        
        return unique_articles
    
    def collect_by_source(self, source_id: str, hours_back: int = 24) -> List[RawArticle]:
        """Collect from a specific source only"""
        for adapter in self.adapters:
            if adapter.source_id == source_id:
                return adapter.fetch(hours_back=hours_back)
        
        logger.warning(f"No adapter found for source: {source_id}")
        return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all adapters"""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "adapters": {}
        }
        
        for adapter in self.adapters:
            adapter_status = adapter.get_health()
            adapter_status["available"] = adapter.is_available()
            
            # Add RSS-specific feed status
            if isinstance(adapter, RSSAdapter):
                adapter_status["feeds"] = adapter.get_feed_status()
            
            status["adapters"][adapter.source_id] = adapter_status
        
        return status


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import json
    
    aggregator = NewsAggregator()
    
    # Show health status first
    print(f"\n{'='*60}")
    print("ADAPTER HEALTH STATUS")
    print(f"{'='*60}")
    health = aggregator.get_health_status()
    for adapter_id, status in health["adapters"].items():
        available = "✓" if status.get("available", False) else "✗"
        print(f"  {available} {adapter_id}: {status.get('success_rate', 100):.1f}% success rate")
    
    # Collect articles
    print(f"\n{'='*60}")
    print("COLLECTING ARTICLES")
    print(f"{'='*60}")
    articles = aggregator.collect_all(hours_back=24)
    
    print(f"\n{'='*60}")
    print("ARGUS-1 Collection Complete (Enhanced)")
    print(f"{'='*60}")
    print(f"Total articles: {len(articles)}")
    
    # Group by source
    by_source = {}
    for a in articles:
        by_source.setdefault(a.source_type, []).append(a)
    
    for source, items in by_source.items():
        print(f"\n{source.upper()}: {len(items)} articles")
        for item in items[:3]:
            print(f"  - {item.title[:60]}...")
    
    # Show system resilience status
    print(f"\n{'='*60}")
    print("SYSTEM RESILIENCE STATUS")
    print(f"{'='*60}")
    system_status = get_system_status()
    print(json.dumps(system_status, indent=2, default=str))
