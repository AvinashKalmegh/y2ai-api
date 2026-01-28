"""
Private Capital Tracker for ARGUS-1
====================================

Tracks AI venture capital and private equity funding as a leading indicator
for the Infrastructure pillar. When private capital surges into AI while
public sentiment is weak, that's institutional money voting differently
than retail - a classic Y2AI signal.

Data Sources (All Free):
- RSS Feeds: TechCrunch, Crunchbase News, VentureBeat, Reuters
- Google Alerts: 7 configured alerts for AI funding news

Output:
- Private_Capital_Dial in Supabase
- Intensity Score (0-100)
- Regime: FROZEN / WEAK / NORMAL / STRONG / SURGE

Integration:
- Runs weekly (Monday, Part 2 workflow)
- Feeds into Infrastructure & Energy pillar signal
- Cross-referenced with Bubble Index for cycle positioning

Author: ARGUS-1 Development Team
Created: January 2026
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FundingEntry:
    """Single funding announcement."""
    date: str
    company: str
    amount_m: float  # Amount in millions USD
    round_type: str  # Seed/A/B/C/D/E/PE/Fund_Launch/IPO_Filing
    category: str    # Foundation_Model/Infrastructure/Application/Chips/Energy
    lead_investor: str
    source: str      # RSS_techcrunch, RSS_crunchbase, Alert, Manual
    url: str
    processed: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Supabase."""
        return {
            'date': self.date,
            'company': self.company,
            'amount_m': self.amount_m,
            'round_type': self.round_type,
            'category': self.category,
            'lead_investor': self.lead_investor,
            'source': self.source,
            'url': self.url,
            'processed': self.processed
        }


@dataclass
class IntensityResult:
    """Private Capital Intensity calculation result."""
    score: int           # 0-100
    regime: str          # FROZEN/WEAK/NORMAL/STRONG/SURGE
    vol_30d_m: float     # 30-day volume in millions
    vol_90d_m: float     # 90-day volume in millions
    megarounds_30d: int  # Count of >$500M deals in 30 days
    fund_launches_30d: int
    categories: List[str]  # Active categories in 30 days
    detail: str          # Human-readable summary


# =============================================================================
# CONFIGURATION
# =============================================================================

class PrivateCapitalConfig:
    """Configuration constants for Private Capital Tracker."""
    
    # RSS Feed URLs
    RSS_FEEDS = {
        'techcrunch': 'https://techcrunch.com/category/fundings-exits/feed/',
        'crunchbase': 'https://news.crunchbase.com/feed/',
        # 'venturebeat': 'https://venturebeat.com/category/ai/feed/',  # Often slow/timeout
        'reuters': 'https://www.reuters.com/technology/feed/',
        'fortune_termsheet': 'https://fortune.com/section/term-sheet/feed/',
    }
    
    # Google Alert Queries (for reference - set up in google.com/alerts as RSS)
    GOOGLE_ALERT_QUERIES = {
        'ai_funding_rounds': '"AI startup" AND ("raised" OR "funding round" OR "Series")',
        'ai_late_stage': '"artificial intelligence" AND ("Series C" OR "Series D" OR "Series E")',
        'ai_early_stage': '"AI startup" AND ("seed round" OR "Series A" OR "Series B")',
        'down_rounds': '"AI" AND ("down round" OR "valuation cut" OR "flat round")',
        'ai_exits': '"AI startup" AND ("acquired" OR "acquisition" OR "IPO")',
        'tier1_vc': '("Sequoia" OR "Andreessen" OR "a16z") AND "artificial intelligence"',
        'investment_pullback': '"AI" AND ("VC pullback" OR "funding slowdown")',
    }
    
    # Keywords for AI relevance filtering
    AI_KEYWORDS = [
        'ai', 'artificial intelligence', 'machine learning', 'llm',
        'foundation model', 'data center', 'gpu', 'chips', 'nvidia',
        'openai', 'anthropic', 'mistral', 'cohere', 'xai',
        'cloud', 'inference', 'training', 'compute', 'generative'
    ]
    
    # Regex patterns for extracting funding amounts
    FUNDING_PATTERNS = [
        r'\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
        r'raises?\s*\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
        r'(\d+(?:\.\d+)?)\s*(billion|million|B|M)\s*(?:funding|round|investment)',
        r'secured\s*\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
        r'valued\s*at\s*\$(\d+(?:\.\d+)?)\s*(billion|million|B|M)',
    ]
    
    # Category classification keywords
    CATEGORY_MAP = {
        'foundation_model': [
            'openai', 'anthropic', 'mistral', 'cohere', 'xai', 'ai21',
            'llm', 'foundation model', 'large language model', 'gpt',
            'claude', 'gemini', 'llama'
        ],
        'infrastructure': [
            'data center', 'datacenter', 'coreweave', 'lambda labs', 'crusoe',
            'cloud', 'inference', 'colocation', 'hyperscale', 'server'
        ],
        'chips': [
            'nvidia', 'amd', 'groq', 'cerebras', 'sambanova', 'chip',
            'semiconductor', 'gpu', 'tpu', 'asic', 'silicon'
        ],
        'application': [
            'cursor', 'jasper', 'runway', 'midjourney', 'agent', 'copilot',
            'chatbot', 'saas', 'enterprise ai', 'ai assistant'
        ],
        'energy': [
            'power', 'nuclear', 'grid', 'energy', 'electricity', 'utility',
            'solar', 'renewable', 'smr', 'small modular reactor'
        ]
    }
    
    # Round type detection keywords
    ROUND_TYPE_KEYWORDS = {
        'Seed': ['seed', 'pre-seed', 'angel'],
        'A': ['series a'],
        'B': ['series b'],
        'C': ['series c'],
        'D': ['series d'],
        'E+': ['series e', 'series f', 'series g', 'late stage', 'growth round'],
        'PE': ['private equity', 'buyout', 'acquisition financing'],
        'Fund_Launch': ['fund launch', 'fund close', 'new fund', 'raised fund'],
        'IPO_Filing': ['ipo', 's-1', 'going public', 'direct listing', 'spac'],
    }
    
    # Thresholds
    MIN_AMOUNT_M = 50  # Minimum $50M to track (filter noise)
    MEGAROUND_THRESHOLD_M = 500  # $500M+ = megaround
    
    # Intensity score weights
    WEIGHT_VOLUME = 0.40
    WEIGHT_MEGAROUNDS = 0.30
    WEIGHT_BREADTH = 0.20
    WEIGHT_MOMENTUM = 0.10


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================

class PrivateCapitalTracker:
    """
    Free-tier private capital tracking for ARGUS-1.
    
    Collects AI funding news from RSS feeds,
    extracts structured data, calculates intensity score,
    and writes to Supabase.
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize tracker.
        
        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self.config = PrivateCapitalConfig()
        
    # -------------------------------------------------------------------------
    # RSS FEED COLLECTION
    # -------------------------------------------------------------------------
    
    def fetch_rss_feeds(self, hours_back: int = 168) -> List[FundingEntry]:
        """
        Fetch and parse all RSS feeds for AI funding news.
        
        Args:
            hours_back: How far back to look (default 168 = 1 week)
            
        Returns:
            List of FundingEntry objects
        """
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed. Run: pip install feedparser")
            return []
        
        entries = []
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        
        for source, url in self.config.RSS_FEEDS.items():
            try:
                logger.info(f"Fetching RSS: {source}")
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:30]:  # Last 30 per feed
                    parsed = self._parse_rss_entry(entry, source, cutoff)
                    if parsed:
                        entries.append(parsed)
                        
            except Exception as e:
                logger.error(f"Error fetching {source}: {e}")
        
        # Deduplicate
        unique = self._dedupe_entries(entries)
        logger.info(f"RSS collection: {len(unique)} unique entries from {len(entries)} total")
        
        return unique
    
    def _parse_rss_entry(self, entry: dict, source: str, cutoff: datetime) -> Optional[FundingEntry]:
        """Parse single RSS entry for funding data."""
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        link = entry.get('link', '')
        text = f"{title} {summary}".lower()
        
        # Check date
        pub_date = self._parse_entry_date(entry)
        if pub_date and pub_date < cutoff:
            return None
        
        # Check AI relevance
        if not self._is_ai_related(text):
            return None
        
        # Extract funding amount
        amount_m = self._extract_amount(text)
        if not amount_m or amount_m < self.config.MIN_AMOUNT_M:
            return None
        
        # Extract other fields
        company = self._extract_company(title)
        round_type = self._detect_round_type(text)
        category = self._classify_category(text)
        lead_investor = self._extract_investor(text)
        
        return FundingEntry(
            date=pub_date.strftime('%Y-%m-%d') if pub_date else datetime.now().strftime('%Y-%m-%d'),
            company=company,
            amount_m=amount_m,
            round_type=round_type,
            category=category,
            lead_investor=lead_investor,
            source=f'RSS_{source}',
            url=link,
            processed=False
        )
    
    def _parse_entry_date(self, entry: dict) -> Optional[datetime]:
        """Parse publication date from RSS entry."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        return None
    
    def _is_ai_related(self, text: str) -> bool:
        """Check if text is AI-related based on keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.config.AI_KEYWORDS)
    
    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract funding amount in millions USD."""
        text_lower = text.lower()
        
        for pattern in self.config.FUNDING_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    if unit in ['billion', 'b']:
                        return amount * 1000
                    elif unit in ['million', 'm']:
                        return amount
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_company(self, title: str) -> str:
        """Extract company name from headline."""
        patterns = [
            r'^([A-Z][A-Za-z0-9\.\s]+?)(?:\s+raises|\s+secures|\s+announces|\s+closes)',
            r'^([A-Z][A-Za-z0-9\.\s]+?),',
            r'^([A-Z][A-Za-z0-9\.]+)\s',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                company = match.group(1).strip()
                if len(company) > 2 and len(company) < 50:
                    return company
        
        # Fallback: first 3 words
        words = title.split()[:3]
        return ' '.join(words) if words else 'Unknown'
    
    def _detect_round_type(self, text: str) -> str:
        """Detect funding round type from text."""
        text_lower = text.lower()
        
        for round_type, keywords in self.config.ROUND_TYPE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return round_type
        
        return 'Unknown'
    
    def _classify_category(self, text: str) -> str:
        """Classify into ARGUS-1 pillar categories."""
        text_lower = text.lower()
        
        for category, keywords in self.config.CATEGORY_MAP.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return 'application'  # Default category
    
    def _extract_investor(self, text: str) -> str:
        """Extract lead investor name if mentioned."""
        patterns = [
            r'led by ([A-Z][A-Za-z\s&]+)',
            r'from ([A-Z][A-Za-z\s&]+)',
            r'backed by ([A-Z][A-Za-z\s&]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                investor = match.group(1).strip()[:50]
                if investor.lower() not in ['the', 'a', 'an', 'and']:
                    return investor
        
        return ''
    
    def _dedupe_entries(self, entries: List[FundingEntry]) -> List[FundingEntry]:
        """Remove duplicate entries by company + date."""
        seen = set()
        unique = []
        
        for entry in entries:
            key = f"{entry.company.lower()}_{entry.date}"
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        
        return unique
    
    # -------------------------------------------------------------------------
    # INTENSITY CALCULATION
    # -------------------------------------------------------------------------
    
    def calculate_intensity(self, entries: List[FundingEntry] = None) -> IntensityResult:
        """
        Calculate Private Capital Intensity Score (0-100).
        
        Components:
        - 30D volume vs baseline (40%)
        - Megaround concentration (30%)
        - Category breadth (20%)
        - Momentum vs prior 30D (10%)
        """
        if entries is None:
            entries = self._load_entries_from_storage(days=90)
        
        # 30-day metrics
        entries_30d = [e for e in entries if self._days_old(e.date) <= 30]
        vol_30d = sum(e.amount_m for e in entries_30d)
        megarounds_30d = len([e for e in entries_30d if e.amount_m >= self.config.MEGAROUND_THRESHOLD_M])
        fund_launches_30d = len([e for e in entries_30d if e.round_type == 'Fund_Launch'])
        categories_30d = list(set(e.category for e in entries_30d))
        
        # 90-day metrics for baseline
        entries_90d = [e for e in entries if self._days_old(e.date) <= 90]
        vol_90d = sum(e.amount_m for e in entries_90d)
        baseline_monthly = vol_90d / 3 if vol_90d > 0 else 5000  # $5B/month default
        
        # Prior 30-day metrics for momentum
        entries_prior = [e for e in entries if 30 < self._days_old(e.date) <= 60]
        vol_prior = sum(e.amount_m for e in entries_prior)
        
        # Component scores
        volume_score = min(100, (vol_30d / baseline_monthly) * 50) if baseline_monthly > 0 else 50
        megaround_score = min(100, megarounds_30d * 25)
        breadth_score = min(100, len(categories_30d) * 20)
        
        # Momentum score
        if vol_prior > 0:
            momentum_ratio = (vol_30d - vol_prior) / vol_prior
            momentum_score = 50 + (momentum_ratio * 50)
            momentum_score = max(0, min(100, momentum_score))
        else:
            momentum_score = 50
        
        # Weighted composite
        intensity = (
            volume_score * self.config.WEIGHT_VOLUME +
            megaround_score * self.config.WEIGHT_MEGAROUNDS +
            breadth_score * self.config.WEIGHT_BREADTH +
            momentum_score * self.config.WEIGHT_MOMENTUM
        )
        
        return IntensityResult(
            score=round(intensity),
            regime=self._score_to_regime(intensity),
            vol_30d_m=round(vol_30d, 1),
            vol_90d_m=round(vol_90d, 1),
            megarounds_30d=megarounds_30d,
            fund_launches_30d=fund_launches_30d,
            categories=categories_30d,
            detail=f"30D: ${vol_30d/1000:.1f}B, Megarounds: {megarounds_30d}, Categories: {len(categories_30d)}"
        )
    
    def _days_old(self, date_str: str) -> int:
        """Calculate days since date string."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return (datetime.now() - date).days
        except:
            return 999
    
    def _score_to_regime(self, score: float) -> str:
        """Convert intensity score to regime label."""
        if score >= 80:
            return 'SURGE'
        elif score >= 60:
            return 'STRONG'
        elif score >= 40:
            return 'NORMAL'
        elif score >= 20:
            return 'WEAK'
        else:
            return 'FROZEN'
    
    # -------------------------------------------------------------------------
    # STORAGE OPERATIONS
    # -------------------------------------------------------------------------
    
    def _load_entries_from_storage(self, days: int = 90) -> List[FundingEntry]:
        """Load entries from Supabase."""
        if not self.supabase:
            logger.warning("No Supabase client configured")
            return []
        
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            response = self.supabase.table('private_capital_entries')\
                .select('*')\
                .gte('date', cutoff)\
                .order('date', desc=True)\
                .execute()
            
            entries = []
            for row in response.data:
                entries.append(FundingEntry(
                    date=row['date'],
                    company=row['company'],
                    amount_m=row['amount_m'],
                    round_type=row['round_type'],
                    category=row['category'],
                    lead_investor=row.get('lead_investor', ''),
                    source=row['source'],
                    url=row.get('url', ''),
                    processed=row.get('processed', False)
                ))
            
            return entries
            
        except Exception as e:
            logger.error(f"Error loading from Supabase: {e}")
            return []
    
    def save_entries_to_supabase(self, entries: List[FundingEntry]) -> int:
        """Save new entries to Supabase. Returns count saved."""
        if not self.supabase:
            logger.warning("No Supabase client configured")
            return 0
        
        if not entries:
            return 0
        
        try:
            rows = []
            for e in entries:
                row = e.to_dict()
                row['created_at'] = datetime.utcnow().isoformat()
                rows.append(row)
            
            self.supabase.table('private_capital_entries')\
                .upsert(rows, on_conflict='date,company')\
                .execute()
            
            logger.info(f"Saved {len(rows)} entries to Supabase")
            return len(rows)
            
        except Exception as e:
            logger.error(f"Error saving to Supabase: {e}")
            return 0
    
    def save_intensity_to_supabase(self, result: IntensityResult) -> bool:
        """Save intensity calculation to Supabase."""
        if not self.supabase:
            logger.warning("No Supabase client configured")
            return False
        
        try:
            row = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'score': result.score,
                'regime': result.regime,
                'vol_30d_m': result.vol_30d_m,
                'vol_90d_m': result.vol_90d_m,
                'megarounds_30d': result.megarounds_30d,
                'fund_launches_30d': result.fund_launches_30d,
                'categories': result.categories,
                'detail': result.detail,
                'calculated_at': datetime.utcnow().isoformat()
            }
            
            self.supabase.table('private_capital_daily')\
                .upsert(row, on_conflict='date')\
                .execute()
            
            logger.info(f"Saved intensity to Supabase: {result.score} ({result.regime})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving intensity to Supabase: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # MAIN UPDATE CYCLE
    # -------------------------------------------------------------------------
    
    def update_dial(self, include_google_alerts: bool = True) -> dict:
        """
        Full update cycle: fetch -> parse -> dedupe -> save -> calculate.
        
        This is the main entry point for the weekly workflow.
        
        Args:
            include_google_alerts: Whether to also fetch from Google Alerts (default True)
        """
        logger.info("=== PRIVATE CAPITAL DIAL UPDATE ===")
        
        # 1. Fetch new entries from RSS
        new_entries = self.fetch_rss_feeds(hours_back=168)
        logger.info(f"Fetched {len(new_entries)} entries from RSS feeds")
        
        # 2. Fetch from Google Alerts if enabled
        if include_google_alerts:
            try:
                from .google_alerts_adapter import GoogleAlertsAdapter
                
                adapter = GoogleAlertsAdapter()
                configured = adapter.get_configured_feeds()
                
                if configured:
                    alerts = adapter.fetch_alerts(hours_back=168)
                    alert_entries = adapter.convert_to_funding_entries(alerts)
                    logger.info(f"Fetched {len(alert_entries)} entries from Google Alerts")
                    
                    # Combine and dedupe
                    combined = new_entries + alert_entries
                    new_entries = self._dedupe_entries(combined)
                    logger.info(f"Combined total: {len(new_entries)} unique entries")
                else:
                    logger.info("No Google Alerts configured, using RSS only")
            except Exception as e:
                logger.warning(f"Could not fetch Google Alerts: {e}")
        
        # 3. Save to Supabase
        saved_count = self.save_entries_to_supabase(new_entries)
        
        # 4. Load all recent entries for calculation
        all_entries = self._load_entries_from_storage(days=90)
        
        # 5. Calculate intensity
        intensity = self.calculate_intensity(all_entries)
        
        # 6. Save intensity result
        self.save_intensity_to_supabase(intensity)
        
        # 7. Log summary
        logger.info(f"Private Capital Update Complete:")
        logger.info(f"  New entries: {len(new_entries)}")
        logger.info(f"  Saved: {saved_count}")
        logger.info(f"  Intensity: {intensity.score} ({intensity.regime})")
        logger.info(f"  30D Volume: ${intensity.vol_30d_m/1000:.1f}B")
        logger.info(f"  Megarounds: {intensity.megarounds_30d}")
        
        return {
            'new_entries': len(new_entries),
            'saved': saved_count,
            'intensity': intensity,
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# SUPABASE TABLE SCHEMA
# =============================================================================

SUPABASE_SCHEMA = """
-- Private Capital Entries (raw data)
CREATE TABLE IF NOT EXISTS private_capital_entries (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    company VARCHAR(100) NOT NULL,
    amount_m FLOAT NOT NULL,
    round_type VARCHAR(20),
    category VARCHAR(30),
    lead_investor VARCHAR(100),
    source VARCHAR(50),
    url TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(date, company)
);

CREATE INDEX IF NOT EXISTS idx_pce_date ON private_capital_entries(date DESC);
CREATE INDEX IF NOT EXISTS idx_pce_category ON private_capital_entries(category);

-- Private Capital Daily (aggregated metrics)
CREATE TABLE IF NOT EXISTS private_capital_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    score INTEGER NOT NULL,
    regime VARCHAR(20) NOT NULL,
    vol_30d_m FLOAT,
    vol_90d_m FLOAT,
    megarounds_30d INTEGER,
    fund_launches_30d INTEGER,
    categories TEXT[],
    detail TEXT,
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pcd_date ON private_capital_daily(date DESC);
"""


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line entry point for testing."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Private Capital Tracker for ARGUS-1')
    parser.add_argument('--test-rss', action='store_true', help='Test RSS feed collection')
    parser.add_argument('--update', action='store_true', help='Run full update cycle')
    parser.add_argument('--print-schema', action='store_true', help='Print Supabase schema')
    
    args = parser.parse_args()
    
    if args.print_schema:
        print(SUPABASE_SCHEMA)
        return
    
    tracker = PrivateCapitalTracker()
    
    if args.test_rss:
        print("\n=== Testing RSS Feed Collection ===\n")
        entries = tracker.fetch_rss_feeds(hours_back=168)
        
        print(f"Found {len(entries)} entries:\n")
        for entry in entries[:10]:
            print(f"  [{entry.date}] {entry.company}")
            print(f"    Amount: ${entry.amount_m}M | Round: {entry.round_type} | Category: {entry.category}")
            print(f"    Source: {entry.source}")
            print()
        
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more\n")
        
        # Calculate intensity without saving
        intensity = tracker.calculate_intensity(entries)
        print(f"\n=== Intensity Calculation ===")
        print(f"Score: {intensity.score}")
        print(f"Regime: {intensity.regime}")
        print(f"30D Volume: ${intensity.vol_30d_m/1000:.1f}B")
        print(f"Megarounds: {intensity.megarounds_30d}")
        print(f"Categories: {intensity.categories}")
        
    elif args.update:
        print("\n=== Running Full Update ===\n")
        print("Note: Requires Supabase client to be configured")
        result = tracker.update_dial()
        print(f"\nResult: {result}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()