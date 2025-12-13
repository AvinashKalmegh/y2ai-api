"""
ARGUS-1 RSS FEED HEALTH CHECKER
Validates feed availability, content quality, and tracks health over time

Features:
- Live availability checks for all feeds
- Response time measurement
- Content validation (not just HTTP status)
- URL redirect detection
- Historical health tracking
- Alternative feed suggestions
- CLI and programmatic interface
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.resilience import get_http_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FEED REGISTRY - All known RSS feeds for Y2AI
# =============================================================================

# Complete feed registry with metadata and alternatives
FEED_REGISTRY = {
    # === TECHNOLOGY NEWS ===
    "reuters_tech": {
        "url": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best&best-topics=tech",
        "name": "Reuters Technology",
        "category": "technology",
        "priority": "high",
        "alternatives": [
            "https://feeds.reuters.com/reuters/technologyNews",
        ],
    },
    "ars_tech": {
        "url": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "name": "Ars Technica",
        "category": "technology",
        "priority": "high",
        "alternatives": [
            "https://feeds.arstechnica.com/arstechnica/index",
        ],
    },
    "verge_tech": {
        "url": "https://www.theverge.com/rss/index.xml",
        "name": "The Verge",
        "category": "technology",
        "priority": "medium",
        "alternatives": [
            "https://www.theverge.com/tech/rss/index.xml",
        ],
    },
    "wired": {
        "url": "https://www.wired.com/feed/rss",
        "name": "Wired",
        "category": "technology",
        "priority": "medium",
        "alternatives": [],
    },
    "techcrunch": {
        "url": "https://techcrunch.com/feed/",
        "name": "TechCrunch",
        "category": "technology",
        "priority": "high",
        "alternatives": [],
    },
    "zdnet": {
        "url": "https://www.zdnet.com/news/rss.xml",
        "name": "ZDNet",
        "category": "technology",
        "priority": "medium",
        "alternatives": [],
    },
    
    # === BUSINESS/FINANCE ===
    "fortune": {
        "url": "https://fortune.com/feed/",
        "name": "Fortune",
        "category": "business",
        "priority": "high",
        "alternatives": [
            "https://fortune.com/feed/fortune-feeds/?id=3230629",
        ],
    },
    "bloomberg_tech": {
        "url": "https://feeds.bloomberg.com/technology/news.rss",
        "name": "Bloomberg Technology",
        "category": "business",
        "priority": "high",
        "alternatives": [
            "https://feeds.bloomberg.com/markets/news.rss",
        ],
    },
    "wsj_tech": {
        "url": "https://feeds.a.dj.com/rss/RSSWSJD.xml",
        "name": "WSJ Tech",
        "category": "business",
        "priority": "high",
        "alternatives": [],
    },
    "ft_tech": {
        "url": "https://www.ft.com/technology?format=rss",
        "name": "Financial Times Tech",
        "category": "business",
        "priority": "high",
        "alternatives": [],
    },
    "cnbc_tech": {
        "url": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "name": "CNBC Technology",
        "category": "business",
        "priority": "medium",
        "alternatives": [],
    },
    
    # === AI SPECIFIC ===
    "mit_ai": {
        "url": "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
        "name": "MIT AI News",
        "category": "ai",
        "priority": "high",
        "alternatives": [
            "https://news.mit.edu/rss/topic/artificial-intelligence2",
        ],
    },
    "ai_news": {
        "url": "https://www.artificialintelligence-news.com/feed/",
        "name": "AI News",
        "category": "ai",
        "priority": "medium",
        "alternatives": [],
    },
    "openai_blog": {
        "url": "https://openai.com/blog/rss.xml",
        "name": "OpenAI Blog",
        "category": "ai",
        "priority": "high",
        "alternatives": [
            "https://openai.com/blog/rss/",
        ],
    },
    "google_ai_blog": {
        "url": "https://ai.googleblog.com/feeds/posts/default",
        "name": "Google AI Blog",
        "category": "ai",
        "priority": "high",
        "alternatives": [],
    },
    "deepmind_blog": {
        "url": "https://deepmind.com/blog/feed/basic/",
        "name": "DeepMind Blog",
        "category": "ai",
        "priority": "high",
        "alternatives": [],
    },
    "anthropic_news": {
        "url": "https://www.anthropic.com/news/rss.xml",
        "name": "Anthropic News",
        "category": "ai",
        "priority": "high",
        "alternatives": [],
    },
    
    # === DATA CENTERS/INFRASTRUCTURE ===
    "datacenter_knowledge": {
        "url": "https://www.datacenterknowledge.com/rss.xml",
        "name": "Data Center Knowledge",
        "category": "infrastructure",
        "priority": "high",
        "alternatives": [],
    },
    "datacenter_dynamics": {
        "url": "https://www.datacenterdynamics.com/en/rss/",
        "name": "Data Center Dynamics",
        "category": "infrastructure",
        "priority": "high",
        "alternatives": [
            "https://www.datacenterdynamics.com/en/feed/",
        ],
    },
    "datacenter_frontier": {
        "url": "https://www.datacenterfrontier.com/feed/",
        "name": "Data Center Frontier",
        "category": "infrastructure",
        "priority": "medium",
        "alternatives": [],
    },
    
    # === SEMICONDUCTORS ===
    "semiconductor_engineering": {
        "url": "https://semiengineering.com/feed/",
        "name": "Semiconductor Engineering",
        "category": "semiconductors",
        "priority": "high",
        "alternatives": [],
    },
    "eetimes": {
        "url": "https://www.eetimes.com/feed/",
        "name": "EE Times",
        "category": "semiconductors",
        "priority": "medium",
        "alternatives": [],
    },
    "anandtech": {
        "url": "https://www.anandtech.com/rss/",
        "name": "AnandTech",
        "category": "semiconductors",
        "priority": "medium",
        "alternatives": [],
    },
    
    # === CLOUD/ENTERPRISE ===
    "aws_news": {
        "url": "https://aws.amazon.com/blogs/aws/feed/",
        "name": "AWS News Blog",
        "category": "cloud",
        "priority": "high",
        "alternatives": [],
    },
    "azure_updates": {
        "url": "https://azure.microsoft.com/en-us/blog/feed/",
        "name": "Azure Blog",
        "category": "cloud",
        "priority": "high",
        "alternatives": [],
    },
    "gcp_blog": {
        "url": "https://cloud.google.com/blog/rss",
        "name": "Google Cloud Blog",
        "category": "cloud",
        "priority": "high",
        "alternatives": [],
    },
    
    # === ENERGY ===
    "utility_dive": {
        "url": "https://www.utilitydive.com/feeds/news/",
        "name": "Utility Dive",
        "category": "energy",
        "priority": "high",
        "alternatives": [],
    },
    "power_magazine": {
        "url": "https://www.powermag.com/feed/",
        "name": "Power Magazine",
        "category": "energy",
        "priority": "medium",
        "alternatives": [],
    },
    "energy_central": {
        "url": "https://energycentral.com/c/ec/rss.xml",
        "name": "Energy Central",
        "category": "energy",
        "priority": "medium",
        "alternatives": [],
    },
    
    # === GOVERNMENT/POLICY ===
    "route_fifty": {
        "url": "https://www.route-fifty.com/rss/technology/",
        "name": "Route Fifty Tech",
        "category": "government",
        "priority": "low",
        "alternatives": [],
    },
    "nextgov": {
        "url": "https://www.nextgov.com/rss/all/",
        "name": "Nextgov",
        "category": "government",
        "priority": "low",
        "alternatives": [
            "https://www.nextgov.com/rss/emerging-tech/",
        ],
    },
    "fedscoop": {
        "url": "https://fedscoop.com/feed/",
        "name": "FedScoop",
        "category": "government",
        "priority": "low",
        "alternatives": [],
    },
    
    # === RESEARCH/ACADEMIC ===
    "arxiv_cs_ai": {
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "name": "arXiv CS.AI",
        "category": "research",
        "priority": "medium",
        "alternatives": [],
    },
    "nature_ai": {
        "url": "https://www.nature.com/natmachintell.rss",
        "name": "Nature Machine Intelligence",
        "category": "research",
        "priority": "high",
        "alternatives": [],
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================

class FeedStatus(Enum):
    """Feed health status"""
    HEALTHY = "healthy"           # Feed working normally
    DEGRADED = "degraded"         # Feed works but has issues (slow, few items)
    REDIRECTED = "redirected"     # Feed URL has changed
    UNAVAILABLE = "unavailable"   # Feed temporarily down
    DEAD = "dead"                 # Feed permanently gone (404, domain expired)
    UNKNOWN = "unknown"           # Not yet checked


@dataclass
class FeedHealthCheck:
    """Result of a single feed health check"""
    feed_id: str
    feed_name: str
    url: str
    category: str
    status: FeedStatus
    checked_at: str
    
    # Response metrics
    response_time_ms: Optional[float] = None
    http_status: Optional[int] = None
    
    # Content metrics
    item_count: Optional[int] = None
    latest_item_age_hours: Optional[float] = None
    
    # Issues detected
    error_message: Optional[str] = None
    redirect_url: Optional[str] = None
    
    # Recommendations
    alternative_url: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @property
    def is_usable(self) -> bool:
        """Can this feed still be used for collection?"""
        return self.status in (FeedStatus.HEALTHY, FeedStatus.DEGRADED, FeedStatus.REDIRECTED)


@dataclass
class FeedHealthReport:
    """Complete health report for all feeds"""
    generated_at: str
    total_feeds: int
    healthy_count: int
    degraded_count: int
    unavailable_count: int
    dead_count: int
    
    checks: List[FeedHealthCheck] = field(default_factory=list)
    
    # Summary by category
    category_health: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Recommendations
    feeds_to_remove: List[str] = field(default_factory=list)
    feeds_to_update: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total_feeds": self.total_feeds,
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "unavailable": self.unavailable_count,
                "dead": self.dead_count,
            },
            "category_health": self.category_health,
            "recommendations": {
                "feeds_to_remove": self.feeds_to_remove,
                "feeds_to_update": self.feeds_to_update,
            },
            "checks": [c.to_dict() for c in self.checks],
        }
    
    def print_summary(self):
        """Print a human-readable summary"""
        print(f"\n{'='*60}")
        print("RSS FEED HEALTH REPORT")
        print(f"{'='*60}")
        print(f"Generated: {self.generated_at}")
        print(f"\nOverall Status:")
        print(f"  ✓ Healthy:     {self.healthy_count}")
        print(f"  ~ Degraded:    {self.degraded_count}")
        print(f"  ! Unavailable: {self.unavailable_count}")
        print(f"  ✗ Dead:        {self.dead_count}")
        print(f"  Total:         {self.total_feeds}")
        
        print(f"\nBy Category:")
        for category, counts in sorted(self.category_health.items()):
            healthy = counts.get("healthy", 0)
            total = sum(counts.values())
            print(f"  {category}: {healthy}/{total} healthy")
        
        if self.feeds_to_remove:
            print(f"\n⚠ Feeds to Remove ({len(self.feeds_to_remove)}):")
            for feed_id in self.feeds_to_remove:
                print(f"  - {feed_id}")
        
        if self.feeds_to_update:
            print(f"\n↻ Feeds to Update ({len(self.feeds_to_update)}):")
            for update in self.feeds_to_update:
                print(f"  - {update['feed_id']}: {update['old_url']}")
                print(f"    → {update['new_url']}")
        
        print(f"{'='*60}")


# =============================================================================
# FEED HEALTH CHECKER
# =============================================================================

class FeedHealthChecker:
    """
    Checks health of RSS feeds and generates reports.
    
    Usage:
        checker = FeedHealthChecker()
        report = checker.check_all()
        report.print_summary()
    """
    
    def __init__(
        self,
        feeds: Optional[Dict[str, Dict]] = None,
        timeout: int = 15,
        max_workers: int = 10,
    ):
        """
        Initialize the health checker.
        
        Args:
            feeds: Feed registry dict, defaults to FEED_REGISTRY
            timeout: Request timeout in seconds
            max_workers: Max concurrent checks
        """
        self.feeds = feeds or FEED_REGISTRY
        self.timeout = timeout
        self.max_workers = max_workers
        self._session = get_http_session()
        
        # Cache for feedparser import
        self._feedparser = None
    
    def _get_feedparser(self):
        """Lazy import feedparser"""
        if self._feedparser is None:
            try:
                import feedparser
                self._feedparser = feedparser
            except ImportError:
                raise ImportError(
                    "feedparser is required for RSS health checks. "
                    "Install with: pip install feedparser"
                )
        return self._feedparser
    
    def check_feed(self, feed_id: str, feed_info: Dict) -> FeedHealthCheck:
        """
        Check health of a single feed.
        
        Returns:
            FeedHealthCheck with status and metrics
        """
        url = feed_info["url"]
        name = feed_info.get("name", feed_id)
        category = feed_info.get("category", "unknown")
        alternatives = feed_info.get("alternatives", [])
        
        check = FeedHealthCheck(
            feed_id=feed_id,
            feed_name=name,
            url=url,
            category=category,
            status=FeedStatus.UNKNOWN,
            checked_at=datetime.utcnow().isoformat(),
        )
        
        start_time = time.time()
        
        try:
            # Step 1: HTTP check
            response = self._session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                headers={"User-Agent": "Y2AI-FeedHealthChecker/1.0"}
            )
            
            check.response_time_ms = (time.time() - start_time) * 1000
            check.http_status = response.status_code
            
            # Check for redirects
            if response.history:
                final_url = response.url
                if final_url != url:
                    check.redirect_url = final_url
            
            # Handle HTTP errors
            if response.status_code == 404:
                check.status = FeedStatus.DEAD
                check.error_message = "Feed not found (404)"
                check.alternative_url = self._find_working_alternative(alternatives)
                return check
            
            if response.status_code == 403:
                check.status = FeedStatus.UNAVAILABLE
                check.error_message = "Access forbidden (403)"
                return check
            
            if response.status_code >= 500:
                check.status = FeedStatus.UNAVAILABLE
                check.error_message = f"Server error ({response.status_code})"
                return check
            
            if response.status_code >= 400:
                check.status = FeedStatus.UNAVAILABLE
                check.error_message = f"HTTP error ({response.status_code})"
                return check
            
            # Step 2: Parse feed content
            feedparser = self._get_feedparser()
            feed = feedparser.parse(response.text)
            
            # Check for parse errors
            if feed.bozo:
                if isinstance(feed.bozo_exception, Exception):
                    # Some bozo exceptions are warnings, not fatal
                    error_type = type(feed.bozo_exception).__name__
                    if "CharacterEncodingOverride" in error_type:
                        pass  # Encoding warning, feed still usable
                    elif len(feed.entries) == 0:
                        check.status = FeedStatus.DEAD
                        check.error_message = f"Parse error: {feed.bozo_exception}"
                        return check
            
            # Step 3: Validate content
            check.item_count = len(feed.entries)
            
            if check.item_count == 0:
                check.status = FeedStatus.DEGRADED
                check.error_message = "Feed returned no items"
                return check
            
            # Check freshness of latest item
            if feed.entries:
                latest = feed.entries[0]
                pub_date = None
                
                if hasattr(latest, 'published_parsed') and latest.published_parsed:
                    pub_date = datetime(*latest.published_parsed[:6])
                elif hasattr(latest, 'updated_parsed') and latest.updated_parsed:
                    pub_date = datetime(*latest.updated_parsed[:6])
                
                if pub_date:
                    age = datetime.utcnow() - pub_date
                    check.latest_item_age_hours = age.total_seconds() / 3600
                    
                    # Flag if feed hasn't updated in 7+ days
                    if check.latest_item_age_hours > 168:
                        check.status = FeedStatus.DEGRADED
                        check.error_message = f"Feed stale ({check.latest_item_age_hours:.0f}h since last item)"
                        return check
            
            # Step 4: Determine final status
            if check.redirect_url and check.redirect_url != url:
                check.status = FeedStatus.REDIRECTED
                check.error_message = f"Feed has moved to {check.redirect_url}"
            elif check.response_time_ms > 5000:
                check.status = FeedStatus.DEGRADED
                check.error_message = f"Slow response ({check.response_time_ms:.0f}ms)"
            else:
                check.status = FeedStatus.HEALTHY
            
            return check
            
        except Exception as e:
            check.response_time_ms = (time.time() - start_time) * 1000
            
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                check.status = FeedStatus.UNAVAILABLE
                check.error_message = f"Request timed out after {self.timeout}s"
            elif "connection" in error_str or "refused" in error_str:
                check.status = FeedStatus.DEAD
                check.error_message = f"Connection failed: {e}"
            elif "name or service not known" in error_str or "nodename" in error_str:
                check.status = FeedStatus.DEAD
                check.error_message = "Domain not found (DNS failure)"
            elif "ssl" in error_str or "certificate" in error_str:
                check.status = FeedStatus.UNAVAILABLE
                check.error_message = f"SSL/TLS error: {e}"
            else:
                check.status = FeedStatus.UNAVAILABLE
                check.error_message = str(e)[:200]
            
            # Try to find working alternative
            if check.status == FeedStatus.DEAD:
                check.alternative_url = self._find_working_alternative(alternatives)
            
            return check
    
    def _find_working_alternative(self, alternatives: List[str]) -> Optional[str]:
        """Try alternative URLs and return first working one"""
        for alt_url in alternatives:
            try:
                response = self._session.head(
                    alt_url,
                    timeout=5,
                    allow_redirects=True
                )
                if response.status_code < 400:
                    return alt_url
            except Exception:
                continue
        return None
    
    def check_all(
        self,
        categories: Optional[List[str]] = None,
        priorities: Optional[List[str]] = None,
    ) -> FeedHealthReport:
        """
        Check all feeds and generate a health report.
        
        Args:
            categories: Filter to specific categories
            priorities: Filter to specific priorities
        
        Returns:
            FeedHealthReport with all check results
        """
        # Filter feeds
        feeds_to_check = {}
        for feed_id, info in self.feeds.items():
            if categories and info.get("category") not in categories:
                continue
            if priorities and info.get("priority") not in priorities:
                continue
            feeds_to_check[feed_id] = info
        
        logger.info(f"Checking {len(feeds_to_check)} feeds...")
        
        # Run checks in parallel
        checks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_feed = {
                executor.submit(self.check_feed, feed_id, info): feed_id
                for feed_id, info in feeds_to_check.items()
            }
            
            for future in as_completed(future_to_feed):
                feed_id = future_to_feed[future]
                try:
                    check = future.result()
                    checks.append(check)
                    
                    # Log status
                    status_icon = {
                        FeedStatus.HEALTHY: "✓",
                        FeedStatus.DEGRADED: "~",
                        FeedStatus.REDIRECTED: "↻",
                        FeedStatus.UNAVAILABLE: "!",
                        FeedStatus.DEAD: "✗",
                        FeedStatus.UNKNOWN: "?",
                    }.get(check.status, "?")
                    
                    logger.info(f"  {status_icon} {feed_id}: {check.status.value}")
                    
                except Exception as e:
                    logger.error(f"  ✗ {feed_id}: Check failed - {e}")
                    checks.append(FeedHealthCheck(
                        feed_id=feed_id,
                        feed_name=feeds_to_check[feed_id].get("name", feed_id),
                        url=feeds_to_check[feed_id]["url"],
                        category=feeds_to_check[feed_id].get("category", "unknown"),
                        status=FeedStatus.UNKNOWN,
                        checked_at=datetime.utcnow().isoformat(),
                        error_message=str(e),
                    ))
        
        # Sort by status (worst first) then by name
        status_order = {
            FeedStatus.DEAD: 0,
            FeedStatus.UNAVAILABLE: 1,
            FeedStatus.DEGRADED: 2,
            FeedStatus.REDIRECTED: 3,
            FeedStatus.HEALTHY: 4,
            FeedStatus.UNKNOWN: 5,
        }
        checks.sort(key=lambda c: (status_order.get(c.status, 99), c.feed_name))
        
        # Calculate statistics
        status_counts = {s: 0 for s in FeedStatus}
        category_health = {}
        feeds_to_remove = []
        feeds_to_update = []
        
        for check in checks:
            status_counts[check.status] += 1
            
            # Category stats
            cat = check.category
            if cat not in category_health:
                category_health[cat] = {}
            status_name = check.status.value
            category_health[cat][status_name] = category_health[cat].get(status_name, 0) + 1
            
            # Recommendations
            if check.status == FeedStatus.DEAD:
                if check.alternative_url:
                    feeds_to_update.append({
                        "feed_id": check.feed_id,
                        "old_url": check.url,
                        "new_url": check.alternative_url,
                        "reason": check.error_message,
                    })
                else:
                    feeds_to_remove.append(check.feed_id)
            elif check.status == FeedStatus.REDIRECTED and check.redirect_url:
                feeds_to_update.append({
                    "feed_id": check.feed_id,
                    "old_url": check.url,
                    "new_url": check.redirect_url,
                    "reason": "Feed has permanently moved",
                })
        
        report = FeedHealthReport(
            generated_at=datetime.utcnow().isoformat(),
            total_feeds=len(checks),
            healthy_count=status_counts[FeedStatus.HEALTHY],
            degraded_count=status_counts[FeedStatus.DEGRADED] + status_counts[FeedStatus.REDIRECTED],
            unavailable_count=status_counts[FeedStatus.UNAVAILABLE],
            dead_count=status_counts[FeedStatus.DEAD],
            checks=checks,
            category_health=category_health,
            feeds_to_remove=feeds_to_remove,
            feeds_to_update=feeds_to_update,
        )
        
        return report
    
    def check_single(self, feed_id: str) -> FeedHealthCheck:
        """Check a single feed by ID"""
        if feed_id not in self.feeds:
            raise ValueError(f"Unknown feed: {feed_id}")
        return self.check_feed(feed_id, self.feeds[feed_id])
    
    def get_working_feeds(self) -> Dict[str, str]:
        """
        Return dict of feed_id -> url for all working feeds.
        
        Useful for updating the aggregator's feed list.
        """
        report = self.check_all()
        working = {}
        
        for check in report.checks:
            if check.is_usable:
                # Use redirect URL if available
                url = check.redirect_url if check.redirect_url else check.url
                working[check.feed_id] = url
        
        return working
    
    def export_working_feeds(self, output_path: str = None) -> str:
        """
        Export working feeds as Python dict for use in aggregator.
        
        Args:
            output_path: Optional file path to write to
        
        Returns:
            Python code string defining the feeds dict
        """
        working = self.get_working_feeds()
        
        lines = ["# Auto-generated by FeedHealthChecker"]
        lines.append(f"# Generated: {datetime.utcnow().isoformat()}")
        lines.append("# Working feeds only\n")
        lines.append("FEEDS = {")
        
        for feed_id, url in sorted(working.items()):
            name = self.feeds[feed_id].get("name", feed_id)
            lines.append(f'    "{feed_id}": "{url}",  # {name}')
        
        lines.append("}")
        
        code = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(code)
            logger.info(f"Exported {len(working)} working feeds to {output_path}")
        
        return code


# =============================================================================
# HISTORICAL TRACKING
# =============================================================================

class FeedHealthHistory:
    """
    Track feed health over time and detect trends.
    
    Stores history in a JSON file for persistence.
    """
    
    def __init__(self, history_file: str = "feed_health_history.json"):
        self.history_file = Path(history_file)
        self._history: Dict[str, List[Dict]] = {}
        self._load()
    
    def _load(self):
        """Load history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = {}
    
    def _save(self):
        """Save history to file"""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def record(self, report: FeedHealthReport):
        """Record a health report to history"""
        for check in report.checks:
            feed_id = check.feed_id
            if feed_id not in self._history:
                self._history[feed_id] = []
            
            # Keep last 30 days of history
            entry = {
                "timestamp": check.checked_at,
                "status": check.status.value,
                "response_time_ms": check.response_time_ms,
                "item_count": check.item_count,
                "error": check.error_message,
            }
            
            self._history[feed_id].append(entry)
            
            # Trim old entries
            cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
            self._history[feed_id] = [
                e for e in self._history[feed_id]
                if e["timestamp"] > cutoff
            ]
        
        self._save()
    
    def get_trend(self, feed_id: str) -> Dict[str, Any]:
        """
        Get health trend for a feed.
        
        Returns dict with:
        - current_status
        - checks_last_7d
        - healthy_pct
        - avg_response_time
        - trend (improving, stable, degrading)
        """
        if feed_id not in self._history:
            return {"error": "No history for feed"}
        
        history = self._history[feed_id]
        if not history:
            return {"error": "Empty history"}
        
        # Last 7 days
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent = [e for e in history if e["timestamp"] > week_ago]
        
        if not recent:
            return {"error": "No recent checks"}
        
        healthy_count = sum(1 for e in recent if e["status"] == "healthy")
        healthy_pct = (healthy_count / len(recent)) * 100
        
        response_times = [e["response_time_ms"] for e in recent if e["response_time_ms"]]
        avg_response = sum(response_times) / len(response_times) if response_times else None
        
        # Determine trend (compare first half to second half)
        mid = len(recent) // 2
        if mid > 0:
            first_half_healthy = sum(1 for e in recent[:mid] if e["status"] == "healthy")
            second_half_healthy = sum(1 for e in recent[mid:] if e["status"] == "healthy")
            
            if second_half_healthy > first_half_healthy + 1:
                trend = "improving"
            elif second_half_healthy < first_half_healthy - 1:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "current_status": recent[-1]["status"],
            "checks_last_7d": len(recent),
            "healthy_pct": round(healthy_pct, 1),
            "avg_response_time_ms": round(avg_response) if avg_response else None,
            "trend": trend,
        }
    
    def get_problem_feeds(self) -> List[str]:
        """Get feeds that have been consistently problematic"""
        problems = []
        
        for feed_id, history in self._history.items():
            if not history:
                continue
            
            # Check last 5 entries
            recent = history[-5:]
            unhealthy = sum(1 for e in recent if e["status"] != "healthy")
            
            if unhealthy >= 4:  # 80%+ unhealthy
                problems.append(feed_id)
        
        return problems


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check health of RSS feeds for Y2AI"
    )
    parser.add_argument(
        "--category", "-c",
        help="Filter by category (ai, technology, business, etc.)"
    )
    parser.add_argument(
        "--priority", "-p",
        choices=["high", "medium", "low"],
        help="Filter by priority"
    )
    parser.add_argument(
        "--feed", "-f",
        help="Check single feed by ID"
    )
    parser.add_argument(
        "--export",
        help="Export working feeds to Python file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=15,
        help="Request timeout in seconds (default: 15)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Max concurrent checks (default: 10)"
    )
    parser.add_argument(
        "--list-feeds",
        action="store_true",
        help="List all registered feeds"
    )
    
    args = parser.parse_args()
    
    checker = FeedHealthChecker(timeout=args.timeout, max_workers=args.workers)
    
    # List feeds
    if args.list_feeds:
        print(f"\nRegistered Feeds ({len(FEED_REGISTRY)}):")
        by_category = {}
        for feed_id, info in FEED_REGISTRY.items():
            cat = info.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((feed_id, info))
        
        for cat, feeds in sorted(by_category.items()):
            print(f"\n{cat.upper()}:")
            for feed_id, info in sorted(feeds):
                priority = info.get("priority", "?")
                print(f"  [{priority[0].upper()}] {feed_id}: {info.get('name', feed_id)}")
        return
    
    # Single feed check
    if args.feed:
        check = checker.check_single(args.feed)
        if args.json:
            print(json.dumps(check.to_dict(), indent=2))
        else:
            print(f"\n{check.feed_name} ({check.feed_id})")
            print(f"  URL: {check.url}")
            print(f"  Status: {check.status.value}")
            if check.response_time_ms:
                print(f"  Response Time: {check.response_time_ms:.0f}ms")
            if check.item_count is not None:
                print(f"  Items: {check.item_count}")
            if check.error_message:
                print(f"  Error: {check.error_message}")
            if check.redirect_url:
                print(f"  Redirected to: {check.redirect_url}")
            if check.alternative_url:
                print(f"  Alternative: {check.alternative_url}")
        return
    
    # Full check
    categories = [args.category] if args.category else None
    priorities = [args.priority] if args.priority else None
    
    report = checker.check_all(categories=categories, priorities=priorities)
    
    # Export working feeds
    if args.export:
        checker.export_working_feeds(args.export)
    
    # Output
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()
        
        # Show problematic feeds
        print("\nProblematic Feeds:")
        for check in report.checks:
            if not check.is_usable:
                print(f"  ✗ {check.feed_id}: {check.error_message or check.status.value}")


if __name__ == "__main__":
    main()
