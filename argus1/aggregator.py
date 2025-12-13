"""
ARGUS-1 NEWS INTELLIGENCE LAYER
Replaces Google Alerts with direct API access to news sources

Sources:
- NewsAPI (optional, requires paid key for production)
- Alpha Vantage (free, 25 req/day) 
- SEC EDGAR (free, unlimited)
- RSS Feeds (free, unlimited) - 30+ curated sources
"""

import os
import hashlib
import requests
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RawArticle:
    """Raw article from any source before processing"""
    source_type: str         # newsapi, alphavantage, edgar, rss
    source_name: str         # "Reuters", "WSJ", "SEC EDGAR"
    title: str
    url: str
    published_at: str        # ISO format
    content: str             # Full text or summary
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
    y2ai_category: str       # spending, constraints, energy, skepticism, etc.
    extracted_facts: List[str]
    impact_score: float      # 0-1 relevance to Y2AI thesis
    sentiment: str           # bullish, bearish, neutral
    companies_mentioned: List[str]
    dollar_amounts: List[str]
    key_quotes: List[str]
    processed_at: str
    keywords_used: Optional[List[str]] = None
    
    tickers_mentioned: Optional[List[str]] = None
    
    # Capex signals
    capex_detected: Optional[bool] = False
    capex_direction: Optional[str] = None
    capex_magnitude: Optional[str] = None
    capex_company: Optional[str] = None
    capex_amount: Optional[str] = None
    capex_context: Optional[str] = None

    # Energy signals
    energy_detected: Optional[bool] = False
    energy_event_type: Optional[str] = None
    energy_direction: Optional[str] = None
    energy_region: Optional[str] = None
    energy_context: Optional[str] = None

    # Compute signals
    compute_detected: Optional[bool] = False
    compute_event_type: Optional[str] = None
    compute_direction: Optional[str] = None
    compute_companies_affected: Optional[List[str]] = None
    compute_context: Optional[str] = None

    # Depreciation signals
    depreciation_detected: Optional[bool] = False
    depreciation_event_type: Optional[str] = None
    depreciation_amount: Optional[str] = None
    depreciation_company: Optional[str] = None
    depreciation_context: Optional[str] = None

    # Veto signals
    veto_detected: Optional[bool] = False
    veto_trigger_type: Optional[str] = None
    veto_severity: Optional[str] = None
    veto_context: Optional[str] = None

    # Newsletter hints
    include_in_weekly: Optional[bool] = False
    suggested_pillar: Optional[str] = None
    one_line_summary: Optional[str] = None
    
    # Thesis relevance (NEW - for enhanced signal detection)
    thesis_infrastructure_support: Optional[bool] = False
    thesis_bubble_warning: Optional[bool] = False
    thesis_constraint_evidence: Optional[bool] = False
    thesis_demand_validation: Optional[bool] = False
    thesis_explanation: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# SOURCE ADAPTERS (Abstract base + implementations)
# =============================================================================

class SourceAdapter(ABC):
    """Base class for all news source adapters.

    fetch should prefer an explicit start_time/end_time window for historical backfills.
    If those are not provided, implementations may fall back to hours_back (last N hours).
    """

    @abstractmethod
    def fetch(
        self,
        hours_back: int = 24,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[RawArticle]:
        """Fetch articles from source.

        Prefer start_time/end_time (explicit window). If not provided, fall back to hours_back.
        """
        pass

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source"""
        pass


# =============================================================================
# COMPOUND KEYWORD RULES FOR RELEVANCE FILTERING
# =============================================================================

# Tier 1: High-value terms that indicate direct relevance (pass alone)
HIGH_VALUE_KEYWORDS = [
    "capex", "capital expenditure",
    "data center construction", "datacenter construction",
    "gpu shortage", "chip shortage",
    "ai infrastructure", "ai spending", "ai investment",
    "hyperscaler spending", "hyperscaler capex",
    "nvidia h100", "nvidia h200", "nvidia blackwell",
    "ai data center", "ai datacenter",
]

# Tier 2: AI/Tech terms (need pairing with Tier 3)
AI_TERMS = [
    "artificial intelligence", "ai ", " ai,", " ai.", "(ai)",
    "machine learning", "deep learning", "large language model", "llm",
    "generative ai", "genai", "foundation model",
]

# Tier 3: Infrastructure/Investment context terms
INFRA_TERMS = [
    "data center", "datacenter", "data centre",
    "infrastructure",
    "capex", "capital expenditure", "capital spending",
    "$1 billion", "$2 billion", "$5 billion", "$10 billion", "$50 billion", "$80 billion", "$100 billion",
    "billion investment", "billion spending", "billion deal",
    "construction", "expansion",
    "gpu", "gpus", "chip shortage", "semiconductor",
    "power grid", "power demand", "energy consumption", "megawatt", "gigawatt",
    "cloud infrastructure", "compute capacity", "computing infrastructure",
]

# Tier 4: Company names (need pairing with AI_TERMS or INFRA_TERMS)
COMPANY_TERMS = [
    "nvidia", "amd", "intel", "tsmc", "asml",
    "microsoft", "google", "amazon", "meta", "openai", "anthropic",
    "aws", "azure", "gcp",
    "oracle", "ibm", "salesforce", "snowflake",
    "equinix", "digital realty",
]

# Negative keywords - skip articles containing these
NEGATIVE_KEYWORDS = [
    "recipe", "cooking", "restaurant review",
    "sports score", "game recap",
    "celebrity gossip", "entertainment news",
    "horoscope", "weather forecast",
    "obituary",
]


def check_article_relevance(title: str, content: str) -> tuple:
    """
    Check if article is relevant to AI infrastructure investment.
    Returns: (is_relevant, matched_keywords, relevance_score)
    
    Relevance scores:
        1.0 = High-value keyword match (definitely relevant)
        0.8 = AI term + Infrastructure term
        0.6 = AI term + Company term
        0.5 = Company term + Infrastructure term
        0.0 = No meaningful match
    """
    text = f"{title} {content}".lower()
    matched = []
    
    # Check for negative keywords first
    for neg in NEGATIVE_KEYWORDS:
        if neg in text:
            return False, [], 0.0
    
    # Tier 1: High-value keywords (instant pass)
    for kw in HIGH_VALUE_KEYWORDS:
        if kw in text:
            matched.append(kw)
            return True, matched, 1.0
    
    # Check which tiers are present
    has_ai_term = False
    has_infra_term = False
    has_company_term = False
    
    for kw in AI_TERMS:
        if kw in text:
            has_ai_term = True
            matched.append(kw.strip())
            break
    
    for kw in INFRA_TERMS:
        if kw in text:
            has_infra_term = True
            matched.append(kw)
            break
    
    for kw in COMPANY_TERMS:
        if kw in text:
            has_company_term = True
            matched.append(kw)
            break
    
    # Compound matching rules
    if has_ai_term and has_infra_term:
        return True, matched, 0.8
    
    if has_ai_term and has_company_term:
        return True, matched, 0.6
    
    if has_company_term and has_infra_term:
        return True, matched, 0.5
    
    # Single matches don't pass
    return False, matched, 0.0


# =============================================================================
# SOURCE ADAPTER IMPLEMENTATIONS
# =============================================================================

class NewsAPIAdapter(SourceAdapter):
    """NewsAPI.org adapter - 80,000+ sources with compound relevance filtering"""
    
    def __init__(self):
        self.api_key = os.getenv('NEWSAPI_KEY')
        self.base_url = "https://newsapi.org/v2"
        
        # Y2AI-specific queries (similar to your Google Alerts)
        self.queries = [
            '("AI" OR "artificial intelligence") AND ("capex" OR "capital expenditure" OR "spending")',
            '("data center" OR "datacenter") AND ("construction" OR "investment" OR "billion")',
            '("GPU" OR "NVIDIA" OR "H100") AND ("shortage" OR "supply" OR "demand")',
            '"hyperscaler" AND ("infrastructure" OR "spending" OR "investment")',
            '("Microsoft" OR "Google" OR "Amazon" OR "Meta") AND ("AI infrastructure" OR "AI spending")',
            '"earnings" AND ("AI" OR "artificial intelligence") AND ("capex" OR "guidance")',
        ]
        
        # Minimum relevance score for filtering
        self.min_relevance_score = 0.5
    
    @property
    def source_id(self) -> str:
        return "newsapi"
    
    def fetch(self, hours_back: int = 24, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[RawArticle]:
        raw_articles = []
        filtered_articles = []

        # If explicit window provided, use it. Otherwise compute from_time using hours_back.
        if start_time is not None and end_time is not None:
            from_time = start_time.isoformat()
            to_time = end_time.isoformat()
        else:
            from_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            to_time = datetime.utcnow().isoformat()

        for query in self.queries:
            try:
                response = requests.get(
                    f"{self.base_url}/everything",
                    params={
                        "q": query,
                        "from": from_time,
                        "to": to_time,
                        "sortBy": "publishedAt",
                        "language": "en",
                        "apiKey": self.api_key or 'a3db44b650b14919aceab6f0be1458b3'
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("articles", []):
                        pub = item.get("publishedAt") or item.get("published_at") or ""
                        title = item.get("title", "")
                        content = item.get("content", "") or item.get("description", "")
                        
                        # Apply compound relevance filter
                        is_relevant, matched_keywords, relevance_score = check_article_relevance(title, content)
                        
                        if not is_relevant or relevance_score < self.min_relevance_score:
                            continue
                        
                        # Add query to keywords for tracking
                        if query not in matched_keywords:
                            matched_keywords.append(query)
                        
                        filtered_articles.append(RawArticle(
                            source_type="newsapi",
                            source_name=item.get("source", {}).get("name", "Unknown"),
                            title=title,
                            url=item.get("url", ""),
                            published_at=pub,
                            content=content,
                            author=item.get("author"),
                            keywords_used=matched_keywords
                        ))
                else:
                    logger.warning(f"NewsAPI returned {response.status_code}")

            except Exception as e:
                logger.error(f"NewsAPI fetch error: {e}")

        logger.info(f"NewsAPI adapter: {len(filtered_articles)} articles passed compound relevance filter")
        return filtered_articles



class AlphaVantageAdapter(SourceAdapter):
    """Alpha Vantage News Sentiment API - Financial news with sentiment"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHAVANTAGE_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        
        # Y2AI-relevant tickers
        self.tickers = [
            "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # Big Tech + AI
            "AMD", "INTC", "TSM", "AVGO",              # Semiconductors
            "ORCL", "IBM", "CRM", "NOW",               # Enterprise AI
            "EQIX", "DLR", "AMT",                      # Data center REITs
        ]
        
        # Y2AI-relevant topics
        self.topics = [
            "technology", "earnings", "financial_markets",
            "manufacturing", "real_estate"
        ]
        
        # Minimum relevance score for filtering
        self.min_relevance_score = 0.5
    
    @property
    def source_id(self) -> str:
        return "alphavantage"
    
    def fetch(self, hours_back: int = 24, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[RawArticle]:
        if not self.api_key:
            logger.warning("Alpha Vantage key not set, skipping")
            return []

        raw_articles = []

        # If explicit window provided, convert to the time_from string format AlphaVantage expects.
        if start_time is not None:
            time_from = start_time.strftime("%Y%m%dT%H%M")
        else:
            time_from = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M")

        ticker_batch = ",".join(self.tickers[:5])  # First batch

        try:
            response = requests.get(
                self.base_url,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker_batch,
                    "time_from": time_from,
                    "limit": 50,
                    "apikey": self.api_key
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                for item in data.get("feed", []):
                    ticker_sentiment = item.get("ticker_sentiment", [])
                    primary_ticker = ticker_sentiment[0].get("ticker") if ticker_sentiment else None
                    
                    title = item.get("title", "")
                    content = item.get("summary", "")
                    
                    # Apply compound relevance filter
                    is_relevant, matched_keywords, relevance_score = check_article_relevance(title, content)
                    
                    if not is_relevant or relevance_score < self.min_relevance_score:
                        continue
                    
                    # Add ticker to matched keywords if present
                    if primary_ticker and primary_ticker not in matched_keywords:
                        matched_keywords.append(primary_ticker)

                    raw_articles.append(RawArticle(
                        source_type="alphavantage",
                        source_name=item.get("source", "Unknown"),
                        title=title,
                        url=item.get("url", ""),
                        published_at=item.get("time_published", ""),
                        content=content,
                        author=", ".join(item.get("authors", [])),
                        ticker=primary_ticker,
                        keywords_used=matched_keywords,
                        relevance_signals={
                            "overall_sentiment": item.get("overall_sentiment_label"),
                            "sentiment_score": item.get("overall_sentiment_score"),
                            "ticker_sentiment": ticker_sentiment,
                            "relevance_score": relevance_score
                        }
                    ))
            else:
                logger.warning(f"Alpha Vantage returned {response.status_code}")

        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")

        logger.info(f"AlphaVantage adapter: {len(raw_articles)} articles passed compound relevance filter")
        return raw_articles


class SECEdgarAdapter(SourceAdapter):
    """SEC EDGAR filings - 8-K, 10-K, 10-Q for AI infrastructure companies
    
    Note: SEC filings use compound search queries and target specific companies,
    so additional keyword filtering is not applied (filing descriptions use
    formal legal language that may not match news-style keywords).
    """
    
    def __init__(self):
        self.base_url = "https://efts.sec.gov/LATEST/search-index"
        
        # CIK numbers for key companies (Central Index Key)
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
        
        # Filing types we care about
        self.form_types = ["8-K", "10-K", "10-Q"]
    
    @property
    def source_id(self) -> str:
        return "sec_edgar"
    
    def fetch(self, hours_back: int = 24, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[RawArticle]:
        articles = []

        # SEC expects dates in YYYY-MM-DD; use explicit window if provided
        if start_time is not None and end_time is not None:
            start_str = start_time.strftime("%Y-%m-%d")
            end_str = end_time.strftime("%Y-%m-%d")
        else:
            start_str = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
            end_str = datetime.utcnow().strftime("%Y-%m-%d")

        search_terms = [
            '"capital expenditure" AND "artificial intelligence"',
            '"data center" AND "investment"',
            '"AI infrastructure"',
            '"GPU" OR "computing infrastructure"'
        ]

        for term in search_terms:
            try:
                response = requests.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q": term,
                        "dateRange": "custom",
                        "startdt": start_str,
                        "enddt": end_str,
                        "forms": ",".join(self.form_types)
                    },
                    headers={"User-Agent": "Y2AI Research contact@y2ai.us"},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    for hit in data.get("hits", {}).get("hits", []):
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
                            },
                            keywords_used=[term] 
                        ))

            except Exception as e:
                logger.error(f"SEC EDGAR fetch error: {e}")

        logger.info(f"SEC EDGAR adapter: {len(articles)} filings found (compound search, no post-filter)")
        return articles


class RSSAdapter(SourceAdapter):
    """
    Advanced RSS feed adapter with:
      - Publisher weighting
      - Multi-stage compound relevance scoring
      - Intelligent time-window filtering
      - Optional full-article enrichment
      - Hard deduping via URL + title hashing
      - Integrated CNBC high-signal feeds
    """

    def __init__(self, fetch_full_text: bool = False):
        try:
            import feedparser
        except ImportError:
            raise RuntimeError("feedparser is required for RSSAdapter")

        self.fetch_full_text = fetch_full_text

        # 🚀 Curated, high-signal, AI-relevant feeds (merged with your original set)
        self.feeds = {
            # --- CNBC High-Value ---
            "cnbc_tech": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "cnbc_ai": "https://www.cnbc.com/id/105142363/device/rss/rss.html",
            "cnbc_cloud": "https://www.cnbc.com/id/105784618/device/rss/rss.html",
            "cnbc_earnings": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
            "cnbc_markets": "https://www.cnbc.com/id/100003114/device/rss/rss.html",

            # --- Your original feeds ---
            "reuters_tech": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best&best-topics=tech",
            "ars_tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
            "verge_tech": "https://www.theverge.com/rss/index.xml",
            "fortune": "https://fortune.com/feed/",
            "bloomberg_tech": "https://feeds.bloomberg.com/technology/news.rss",
            "mit_ai": "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
            "ai_news": "https://www.artificialintelligence-news.com/feed/",
            "datacenter_knowledge": "https://www.datacenterknowledge.com/rss.xml",
            "datacenter_dynamics": "https://www.datacenterdynamics.com/en/rss/",
            "utility_dive": "https://www.utilitydive.com/feeds/news/",
            "power_magazine": "https://www.powermag.com/feed/",
            "route_fifty": "https://www.route-fifty.com/rss/technology/",
            "nextgov": "https://www.nextgov.com/rss/all/",
        }

        # 🚦 Publisher weighting: high-trust sources get boosted
        self.publisher_weights = {
            "cnbc": 1.15,
            "reuters": 1.20,
            "bloomberg": 1.20,
            "fortune": 1.05,
            "arstechnica": 1.05,
            # Neutral weight is 1.0
        }

        # Minimum score after weighting required to pass
        self.min_relevance_score = 0.50

    @property
    def source_id(self) -> str:
        return "rss"

    def _publisher_weight(self, feed_name: str) -> float:
        for key, w in self.publisher_weights.items():
            if key in feed_name.lower():
                return w
        return 1.0

    def _extract_pub_date(self, entry):
        """Robust date extraction from RSS entries."""
        dt = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6])
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            dt = datetime(*entry.updated_parsed[:6])
        return dt

    def _fetch_full_article(self, url: str) -> str:
        """Optional full-article fetch for thin RSS summaries."""
        if not self.fetch_full_text:
            return ""
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                return r.text
        except Exception:
            return ""
        return ""

    def fetch(self, hours_back: int = 24,
              start_time: Optional[datetime] = None,
              end_time: Optional[datetime] = None) -> List[RawArticle]:

        import feedparser

        articles = []
        window_start = start_time or (datetime.utcnow() - timedelta(hours=hours_back))
        window_end = end_time or datetime.utcnow()

        for feed_name, feed_url in self.feeds.items():
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    pub_date = self._extract_pub_date(entry)

                    # Filter by date window
                    if pub_date:
                        if pub_date < window_start or pub_date > window_end:
                            continue

                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    link = entry.get("link", "")

                    # Combined text for relevance scoring
                    combined_text = f"{title}\n{summary}"
                    is_rel, matched, score = check_article_relevance(title, combined_text)

                    if not is_rel:
                        continue

                    # Publisher weighting
                    score *= self._publisher_weight(feed_name)

                    if score < self.min_relevance_score:
                        continue

                    # Optional full-text enrichment
                    full_text = self._fetch_full_article(link)
                    content = full_text if full_text else summary

                    articles.append(
                        RawArticle(
                            source_type="rss",
                            source_name=feed_name,
                            title=title,
                            url=link,
                            published_at=pub_date.isoformat() if pub_date else "",
                            content=content,
                            author=entry.get("author", None),
                            keywords_used=matched,
                            relevance_signals={
                                "raw_score": score,
                                "publisher_weight": self._publisher_weight(feed_name),
                            }
                        )
                    )

            except Exception as e:
                logger.error(f"RSS fetch error for {feed_name}: {e}")

        logger.info(f"Advanced RSS adapter: {len(articles)} relevant articles")
        return articles


class GoogleAlertsAdapter(SourceAdapter):
    """
    Google Alerts RSS feed adapter.
    
    These are pre-filtered by Google based on your alert queries,
    but we still apply compound keyword filtering to ensure relevance.
    """
    
    def __init__(self):
        # Google Alerts RSS feeds (29 unique feeds after removing duplicates)
        # Feed IDs correspond to your configured alerts
        self.feeds = {
            "alert_01": "https://www.google.com/alerts/feeds/11684174711489635674/4365419633177123664",
            "alert_02": "https://www.google.com/alerts/feeds/11684174711489635674/5184195265249208169",
            "alert_03": "https://www.google.com/alerts/feeds/11684174711489635674/2753479771155929540",
            "alert_04": "https://www.google.com/alerts/feeds/11684174711489635674/2039993254286218360",
            "alert_05": "https://www.google.com/alerts/feeds/11684174711489635674/15932431990654056342",
            "alert_06": "https://www.google.com/alerts/feeds/11684174711489635674/14529729975935794949",
            "alert_07": "https://www.google.com/alerts/feeds/11684174711489635674/282298084729478997",
            "alert_08": "https://www.google.com/alerts/feeds/11684174711489635674/17206405693888883541",
            "alert_09": "https://www.google.com/alerts/feeds/11684174711489635674/5062454546241190321",
            "alert_10": "https://www.google.com/alerts/feeds/11684174711489635674/4250126265034307408",
            "alert_11": "https://www.google.com/alerts/feeds/11684174711489635674/17712857490264567381",
            "alert_12": "https://www.google.com/alerts/feeds/11684174711489635674/2823826983640805720",
            "alert_13": "https://www.google.com/alerts/feeds/11684174711489635674/12293189059655535840",
            "alert_14": "https://www.google.com/alerts/feeds/11684174711489635674/4256031800154310924",
            "alert_15": "https://www.google.com/alerts/feeds/11684174711489635674/15680614048743180250",
            "alert_16": "https://www.google.com/alerts/feeds/11684174711489635674/9400295411401056476",
            "alert_17": "https://www.google.com/alerts/feeds/11684174711489635674/9677814869234886239",
            "alert_18": "https://www.google.com/alerts/feeds/11684174711489635674/15720907603979165915",
            "alert_19": "https://www.google.com/alerts/feeds/11684174711489635674/572747977406752442",
            "alert_20": "https://www.google.com/alerts/feeds/11684174711489635674/15306364356586741862",
            "alert_21": "https://www.google.com/alerts/feeds/11684174711489635674/17206405693888880873",
            "alert_22": "https://www.google.com/alerts/feeds/11684174711489635674/10628118736661352058",
            "alert_23": "https://www.google.com/alerts/feeds/11684174711489635674/13812936224103732642",
            "alert_24": "https://www.google.com/alerts/feeds/11684174711489635674/16787346356724528745",
            "alert_25": "https://www.google.com/alerts/feeds/11684174711489635674/3783018569209270398",
            "alert_26": "https://www.google.com/alerts/feeds/11684174711489635674/3783018569209273501",
            "alert_27": "https://www.google.com/alerts/feeds/11684174711489635674/9400295411401059970",
            "alert_28": "https://www.google.com/alerts/feeds/11684174711489635674/2039993254286219220",
            "alert_29": "https://www.google.com/alerts/feeds/11684174711489635674/5746720537511748554",
            "alert_30": "https://www.google.com/alerts/feeds/11684174711489635674/12293189059655533862",
        }
        
        # Google Alerts are pre-filtered, so we can be less strict
        # Set to 0.0 to accept all, or 0.5 for additional filtering
        self.min_relevance_score = 0.0  # Trust Google's filtering
    
    @property
    def source_id(self) -> str:
        return "google_alerts"
    
    def fetch(self, hours_back: int = 24, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[RawArticle]:
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed, skipping Google Alerts")
            return []

        articles = []

        if start_time is not None:
            cutoff_time = start_time
        else:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        for alert_name, feed_url in self.feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                # Extract alert query from feed title if available
                alert_query = feed.feed.get("title", alert_name).replace("Google Alert - ", "")

                for entry in feed.entries:
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])

                    # Time filtering
                    if pub_date:
                        if start_time is not None and pub_date < start_time:
                            continue
                        if end_time is not None and pub_date > end_time:
                            continue
                        # Default hours_back filter
                        if start_time is None and pub_date < cutoff_time:
                            continue

                    title = entry.get("title", "")
                    # Google Alerts content is in 'content' or 'summary'
                    content = ""
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].get('value', '')
                    else:
                        content = entry.get("summary", "")
                    
                    # Optional: Apply compound relevance filter
                    if self.min_relevance_score > 0:
                        is_relevant, matched_keywords, relevance_score = check_article_relevance(title, content)
                        if not is_relevant or relevance_score < self.min_relevance_score:
                            continue
                    else:
                        # Trust Google's filtering, just track the alert query
                        matched_keywords = [alert_query]
                    
                    articles.append(RawArticle(
                        source_type="google_alerts",
                        source_name=f"google_alert_{alert_query[:30]}",
                        title=title,
                        url=entry.get("link", ""),
                        published_at=pub_date.isoformat() if pub_date else "",
                        content=content,
                        author=None,
                        keywords_used=matched_keywords
                    ))

            except Exception as e:
                logger.error(f"Google Alerts fetch error for {alert_name}: {e}")

        logger.info(f"Google Alerts adapter: {len(articles)} articles fetched from {len(self.feeds)} alerts")
        return articles


# =============================================================================
# AGGREGATOR ORCHESTRATOR
# =============================================================================

class NewsAggregator:
    """Main orchestrator that coordinates all source adapters"""
    
    def __init__(self, adapters: List[SourceAdapter] = None):
        if adapters is None:
            # Default: use all available adapters
            self.adapters = [
                NewsAPIAdapter(),
                AlphaVantageAdapter(),
                SECEdgarAdapter(),
                RSSAdapter(),
                GoogleAlertsAdapter(),
            ]
        else:
            self.adapters = adapters
        
        self.seen_hashes = set()
    
    def collect_all(self, hours_back: int = 24, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[RawArticle]:
        """Collect from all sources and deduplicate.
        Prefer using start_time/end_time for explicit historical windows. Falls back to hours_back if window not given.
        """
        all_articles = []

        for adapter in self.adapters:
            logger.info(f"Fetching from {adapter.source_id}...")
            try:
                # Call adapter.fetch with the same parameters, adapters handle fallback.
                articles = adapter.fetch(hours_back=hours_back, start_time=start_time, end_time=end_time)
                logger.info(f"  Got {len(articles)} articles from {adapter.source_id}")
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Adapter {adapter.source_id} failed: {e}")

        # Deduplicate by URL hash
        unique_articles = []
        for article in all_articles:
            if article.article_hash not in self.seen_hashes:
                self.seen_hashes.add(article.article_hash)
                unique_articles.append(article)

        logger.info(f"Total unique articles: {len(unique_articles)} (deduplicated from {len(all_articles)})")
        return unique_articles

    
    def collect_by_source(self, source_id: str, hours_back: int = 24, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[RawArticle]:
        for adapter in self.adapters:
            if adapter.source_id == source_id:
                return adapter.fetch(hours_back=hours_back, start_time=start_time, end_time=end_time)

        logger.warning(f"No adapter found for source: {source_id}")
        return []



# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    aggregator = NewsAggregator()
    articles = aggregator.collect_all(hours_back=24)
    
    print(f"\n{'='*60}")
    print(f"ARGUS-1 Collection Complete")
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