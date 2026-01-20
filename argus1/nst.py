"""
NST Integration — NarrativeDensity for FlowOS
==============================================
Extends Google Alerts pipeline to feed two FlowOS components:
1. NarrativeDensity — 30% weight in Attractor Mass formula
2. Exit Rule Catalyst Detection — modifies exit signals based on news density

Data Flow:
    Google Alerts → Ingest/Classify → nst_mentions → Daily Summary → API endpoints

Tables:
    - nst_mentions: Raw mentions with classification
    - nst_daily_summary: Pre-aggregated daily stats per bubble_type

API Endpoints:
    - get_narrative_density(bubble_type, days) → feeds Attractor Mass calculator
    - get_nst_status(bubble_type) → feeds exit rule logic
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Attractor tickers per bubble type
ATTRACTOR_TICKERS = {
    'AI/Compute': 'NVDA',
    'Energy/Grid': 'CEG',
    'Crypto': 'COIN',
    'Clean Energy': 'ENPH',
}

# Keywords for classification
NST_KEYWORDS = {
    'AI/Compute': {
        'attractor': ['NVDA', 'NVIDIA', 'Jensen', 'Jensen Huang'],
        'theme': ['AI chip', 'GPU', 'datacenter', 'hyperscaler', 'AI infrastructure', 
                  'data center', 'artificial intelligence', 'machine learning'],
        'stress': ['AI bubble', 'overvalued', 'correction', 'AI winter', 'bubble burst'],
        'catalyst': ['earnings', 'guidance', 'export controls', 'antitrust', 'blackwell']
    },
    'Energy/Grid': {
        'attractor': ['CEG', 'Constellation', 'Constellation Energy'],
        'theme': ['nuclear power', 'AI power demand', 'datacenter power', 'grid', 
                  'SMR', 'NRC', 'nuclear energy', 'power infrastructure'],
        'stress': ['nuclear bubble', 'rate caps', 'utility overvalued', 'NRC rejection'],
        'catalyst': ['NRC approval', 'PPA', 'rate case', 'Microsoft', 'Amazon datacenter',
                    'power purchase agreement']
    },
    'Crypto': {
        'attractor': ['COIN', 'Coinbase', 'Bitcoin', 'BTC'],
        'theme': ['crypto', 'cryptocurrency', 'Bitcoin ETF', 'halving', 'blockchain'],
        'stress': ['crypto crash', 'bubble', 'SEC enforcement', 'FTX', 'fraud'],
        'catalyst': ['ETF approval', 'regulation', 'Binance', 'Tether', 'stablecoin']
    },
    'Clean Energy': {
        'attractor': ['ENPH', 'Enphase'],
        'theme': ['solar', 'clean energy', 'renewable', 'IRA', 'green energy'],
        'stress': ['solar bubble', 'overcapacity', 'tariff', 'subsidy cut'],
        'catalyst': ['IRA', 'tax credit', 'tariff', 'interest rate', 'net metering']
    }
}

# Catalyst type mapping (keyword -> type)
CATALYST_TYPE_MAP = {
    # AI/Compute
    'earnings': 'POSITIVE_PENDING',
    'guidance': 'POSITIVE_PENDING', 
    'blackwell': 'POSITIVE_PENDING',
    'export controls': 'NEGATIVE',
    'antitrust': 'NEGATIVE',
    
    # Energy/Grid
    'NRC approval': 'POSITIVE_PENDING',
    'PPA': 'POSITIVE_PENDING',
    'power purchase agreement': 'POSITIVE_PENDING',
    'Microsoft': 'POSITIVE_PENDING',
    'Amazon datacenter': 'POSITIVE_PENDING',
    'rate case': 'NEGATIVE',
    
    # Crypto
    'ETF approval': 'POSITIVE_PENDING',
    'regulation': 'NEGATIVE',
    'Binance': 'NEGATIVE',
    'Tether': 'NEGATIVE',
    'stablecoin': 'NEGATIVE',
    
    # Clean Energy
    'IRA': 'POSITIVE_PENDING',
    'tax credit': 'POSITIVE_PENDING',
    'tariff': 'NEGATIVE',
    'interest rate': 'NEGATIVE',
    'net metering': 'NEGATIVE',
}

# Density level thresholds (based on total mentions per day)
DENSITY_THRESHOLDS = {
    'LOW': 2,       # 0-2 mentions
    'MODERATE': 5,  # 3-5 mentions
    'HIGH': 10,     # 6-10 mentions
    # > 10 = EXTREME
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NSTMention:
    """Single classified mention from Google Alerts."""
    source: str
    headline: str
    snippet: Optional[str]
    url: str
    published_at: Optional[str]
    bubble_type: str
    attractor_mentioned: bool
    attractor_ticker: Optional[str]
    alert_category: str  # 'attractor', 'theme', 'stress', 'catalyst'
    keywords_matched: List[str]
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    catalyst_type: Optional[str] = None


@dataclass
class NSTDailySummary:
    """Aggregated daily summary per bubble type."""
    date: str
    bubble_type: str
    total_mentions: int
    attractor_mentions: int
    stress_mentions: int
    catalyst_mentions: int
    narrative_density: float
    density_level: str


@dataclass
class NarrativeDensityResponse:
    """Response for Attractor Mass calculator."""
    status: str
    narrative_density: float
    scaled: float
    total_mentions: int
    attractor_mentions: int


@dataclass
class NSTStatusResponse:
    """Response for exit rule logic."""
    density_level: str
    total_mentions: int
    catalyst_pending: bool
    catalyst_type: Optional[str]


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def get_attractor_ticker(bubble_type: str) -> Optional[str]:
    """Get the attractor ticker for a bubble type."""
    return ATTRACTOR_TICKERS.get(bubble_type)


def classify_mention(
    headline: str,
    snippet: str,
    source: str,
    url: str,
    published_at: str,
    bubble_type: str
) -> NSTMention:
    """
    Classify a Google Alert mention for NST tracking.
    
    Args:
        headline: Article headline
        snippet: Article snippet/summary
        source: Source name
        url: Article URL
        published_at: Publication timestamp
        bubble_type: Which bubble to classify for
    
    Returns:
        NSTMention with classification fields populated
    """
    keywords = NST_KEYWORDS.get(bubble_type, {})
    if not keywords:
        logger.warning(f"Unknown bubble_type: {bubble_type}")
        return NSTMention(
            source=source,
            headline=headline,
            snippet=snippet,
            url=url,
            published_at=published_at,
            bubble_type=bubble_type,
            attractor_mentioned=False,
            attractor_ticker=None,
            alert_category='theme',
            keywords_matched=[]
        )
    
    text = (headline + ' ' + (snippet or '')).lower()
    
    mention = NSTMention(
        source=source,
        headline=headline,
        snippet=snippet,
        url=url,
        published_at=published_at,
        bubble_type=bubble_type,
        attractor_mentioned=False,
        attractor_ticker=None,
        alert_category='theme',
        keywords_matched=[]
    )
    
    matched_keywords = []
    
    # Check attractor keywords first (highest priority)
    for kw in keywords.get('attractor', []):
        if kw.lower() in text:
            mention.attractor_mentioned = True
            mention.attractor_ticker = get_attractor_ticker(bubble_type)
            mention.alert_category = 'attractor'
            matched_keywords.append(kw)
            break
    
    # Check theme keywords
    for kw in keywords.get('theme', []):
        if kw.lower() in text:
            matched_keywords.append(kw)
    
    # Override category if stress detected (higher priority than theme)
    for kw in keywords.get('stress', []):
        if kw.lower() in text:
            mention.alert_category = 'stress'
            matched_keywords.append(kw)
            break
    
    # Override category if catalyst detected (can override stress too)
    for kw in keywords.get('catalyst', []):
        if kw.lower() in text:
            mention.alert_category = 'catalyst'
            mention.catalyst_type = CATALYST_TYPE_MAP.get(kw, 'POSITIVE_PENDING')
            matched_keywords.append(kw)
            break
    
    mention.keywords_matched = list(set(matched_keywords))
    
    return mention


def classify_with_claude(
    headline: str,
    snippet: str,
    bubble_type: str,
    anthropic_client=None
) -> Dict:
    """
    Use Claude to classify sentiment and catalyst type.
    
    Args:
        headline: Article headline
        snippet: Article snippet
        bubble_type: Bubble type for context
        anthropic_client: Optional pre-initialized Anthropic client
    
    Returns:
        Dict with sentiment_score, sentiment_label, catalyst_type
    """
    if anthropic_client is None:
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set, skipping Claude classification")
                return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'catalyst_type': None}
            anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("anthropic package not installed")
            return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'catalyst_type': None}
    
    prompt = f"""Classify this news mention for {bubble_type}:

Headline: {headline}
Snippet: {snippet or 'N/A'}

Return JSON only (no markdown):
{{
  "sentiment_score": <-1 to +1>,
  "sentiment_label": <"bullish" | "bearish" | "neutral">,
  "catalyst_type": <"POSITIVE_PENDING" | "NEGATIVE" | null>
}}

POSITIVE_PENDING = upcoming event likely to be positive (earnings beat expected, approval pending)
NEGATIVE = bad news or negative catalyst"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        return json.loads(result_text)
    except Exception as e:
        logger.error(f"Claude classification error: {e}")
        return {'sentiment_score': 0, 'sentiment_label': 'neutral', 'catalyst_type': None}


# =============================================================================
# STORAGE FUNCTIONS
# =============================================================================

class NSTStorage:
    """Storage operations for NST tables."""
    
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        if self.client is None:
            try:
                from supabase import create_client
                url = os.getenv('SUPABASE_URL')
                key = os.getenv('SUPABASE_KEY')
                if url and key:
                    self.client = create_client(url, key)
            except Exception as e:
                logger.error(f"Failed to create Supabase client: {e}")
    
    def is_connected(self) -> bool:
        return self.client is not None
    
    def save_mention(self, mention: NSTMention) -> Optional[Dict]:
        """Save a single NST mention to the database."""
        if not self.is_connected():
            logger.warning("Supabase not connected")
            return None
        
        record = {
            'source': mention.source,
            'headline': mention.headline,
            'snippet': mention.snippet,
            'url': mention.url,
            'published_at': mention.published_at,
            'bubble_type': mention.bubble_type,
            'attractor_mentioned': mention.attractor_mentioned,
            'attractor_ticker': mention.attractor_ticker,
            'alert_category': mention.alert_category,
            'keywords_matched': mention.keywords_matched,
            'sentiment_score': mention.sentiment_score,
            'sentiment_label': mention.sentiment_label,
            'catalyst_type': mention.catalyst_type,
        }
        
        try:
            response = self.client.table('nst_mentions').insert(record).execute()
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save NST mention: {e}")
        
        return None
    
    def save_mentions_batch(self, mentions: List[NSTMention]) -> int:
        """Save multiple NST mentions."""
        if not self.is_connected():
            return 0
        
        saved = 0
        for mention in mentions:
            if self.save_mention(mention):
                saved += 1
        
        return saved
    
    def get_mentions_for_date(
        self,
        bubble_type: str,
        date: str
    ) -> List[Dict]:
        """Get all mentions for a specific date and bubble type."""
        if not self.is_connected():
            return []
        
        try:
            start = f"{date}T00:00:00Z"
            end = f"{date}T23:59:59Z"
            
            response = self.client.table('nst_mentions').select('*').eq(
                'bubble_type', bubble_type
            ).gte('published_at', start).lt('published_at', end).execute()
            
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to get mentions: {e}")
            return []
    
    def save_daily_summary(self, summary: NSTDailySummary) -> Optional[Dict]:
        """Save or update daily summary."""
        if not self.is_connected():
            return None
        
        record = asdict(summary)
        
        try:
            response = self.client.table('nst_daily_summary').upsert(
                record,
                on_conflict='date,bubble_type'
            ).execute()
            
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save daily summary: {e}")
        
        return None
    
    def get_daily_summaries(
        self,
        bubble_type: str,
        days: int = 7
    ) -> List[Dict]:
        """Get daily summaries for the past N days."""
        if not self.is_connected():
            return []
        
        try:
            cutoff = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            response = self.client.table('nst_daily_summary').select('*').eq(
                'bubble_type', bubble_type
            ).gte('date', cutoff).order('date', desc=True).execute()
            
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to get daily summaries: {e}")
            return []
    
    def get_today_summary(self, bubble_type: str) -> Optional[Dict]:
        """Get today's summary for a bubble type."""
        if not self.is_connected():
            return None
        
        try:
            today = datetime.utcnow().strftime('%Y-%m-%d')
            
            response = self.client.table('nst_daily_summary').select('*').eq(
                'bubble_type', bubble_type
            ).eq('date', today).execute()
            
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to get today's summary: {e}")
        
        return None
    
    def get_recent_catalyst(self, bubble_type: str) -> Optional[str]:
        """Get the most recent catalyst type for a bubble."""
        if not self.is_connected():
            return None
        
        try:
            response = self.client.table('nst_mentions').select('catalyst_type').eq(
                'bubble_type', bubble_type
            ).eq('alert_category', 'catalyst').order(
                'published_at', desc=True
            ).limit(1).execute()
            
            if response.data and response.data[0].get('catalyst_type'):
                return response.data[0]['catalyst_type']
        except Exception as e:
            logger.error(f"Failed to get recent catalyst: {e}")
        
        return None


# =============================================================================
# DAILY AGGREGATION
# =============================================================================

def calculate_daily_summary(
    storage: NSTStorage,
    date: str,
    bubble_type: str
) -> NSTDailySummary:
    """
    Calculate and save daily summary for a bubble type.
    
    Args:
        storage: NSTStorage instance
        date: Date string (YYYY-MM-DD)
        bubble_type: Bubble type to summarize
    
    Returns:
        NSTDailySummary with aggregated stats
    """
    mentions = storage.get_mentions_for_date(bubble_type, date)
    
    total = len(mentions)
    attractor_count = sum(1 for m in mentions if m.get('attractor_mentioned'))
    stress_count = sum(1 for m in mentions if m.get('alert_category') == 'stress')
    catalyst_count = sum(1 for m in mentions if m.get('alert_category') == 'catalyst')
    
    # Calculate narrative density (attractor mentions / total)
    narrative_density = attractor_count / total if total > 0 else 0
    
    # Determine density level
    if total <= DENSITY_THRESHOLDS['LOW']:
        density_level = 'LOW'
    elif total <= DENSITY_THRESHOLDS['MODERATE']:
        density_level = 'MODERATE'
    elif total <= DENSITY_THRESHOLDS['HIGH']:
        density_level = 'HIGH'
    else:
        density_level = 'EXTREME'
    
    summary = NSTDailySummary(
        date=date,
        bubble_type=bubble_type,
        total_mentions=total,
        attractor_mentions=attractor_count,
        stress_mentions=stress_count,
        catalyst_mentions=catalyst_count,
        narrative_density=round(narrative_density, 3),
        density_level=density_level
    )
    
    storage.save_daily_summary(summary)
    
    return summary


def run_daily_aggregation(storage: NSTStorage = None, date: str = None):
    """
    Run daily aggregation for all bubble types.
    
    Args:
        storage: Optional NSTStorage instance
        date: Optional date string (defaults to today)
    """
    if storage is None:
        storage = NSTStorage()
    
    if date is None:
        date = datetime.utcnow().strftime('%Y-%m-%d')
    
    logger.info(f"Running NST daily aggregation for {date}")
    
    for bubble_type in ATTRACTOR_TICKERS.keys():
        summary = calculate_daily_summary(storage, date, bubble_type)
        logger.info(f"  {bubble_type}: {summary.total_mentions} mentions, "
                   f"{summary.attractor_mentions} attractor, density={summary.narrative_density:.3f}")


# =============================================================================
# API ENDPOINTS
# =============================================================================

def get_narrative_density(
    bubble_type: str,
    days: int = 7,
    storage: NSTStorage = None
) -> NarrativeDensityResponse:
    """
    Get NarrativeDensity for Attractor Mass calculator.
    
    Args:
        bubble_type: Which bubble type
        days: Number of days to aggregate
        storage: Optional NSTStorage instance
    
    Returns:
        NarrativeDensityResponse with scaled value (0-100)
    """
    if storage is None:
        storage = NSTStorage()
    
    summaries = storage.get_daily_summaries(bubble_type, days)
    
    if not summaries:
        return NarrativeDensityResponse(
            status='NO_DATA',
            narrative_density=0,
            scaled=50.0,  # Default neutral value
            total_mentions=0,
            attractor_mentions=0
        )
    
    total_mentions = sum(s['total_mentions'] for s in summaries)
    attractor_mentions = sum(s['attractor_mentions'] for s in summaries)
    
    density = attractor_mentions / total_mentions if total_mentions > 0 else 0
    
    return NarrativeDensityResponse(
        status='CALCULATED',
        narrative_density=round(density, 3),
        scaled=round(density * 100, 1),
        total_mentions=total_mentions,
        attractor_mentions=attractor_mentions
    )


def get_nst_status(
    bubble_type: str,
    storage: NSTStorage = None
) -> NSTStatusResponse:
    """
    Get NST status for exit rule logic.
    
    Args:
        bubble_type: Which bubble type
        storage: Optional NSTStorage instance
    
    Returns:
        NSTStatusResponse with density level and catalyst info
    """
    if storage is None:
        storage = NSTStorage()
    
    summary = storage.get_today_summary(bubble_type)
    
    if not summary:
        return NSTStatusResponse(
            density_level='LOW',
            total_mentions=0,
            catalyst_pending=False,
            catalyst_type=None
        )
    
    catalyst_type = None
    if summary.get('catalyst_mentions', 0) > 0:
        catalyst_type = storage.get_recent_catalyst(bubble_type)
    
    return NSTStatusResponse(
        density_level=summary.get('density_level', 'LOW'),
        total_mentions=summary.get('total_mentions', 0),
        catalyst_pending=summary.get('catalyst_mentions', 0) > 0,
        catalyst_type=catalyst_type
    )


# =============================================================================
# INTEGRATION WITH GOOGLE ALERTS PIPELINE
# =============================================================================

def process_alert_for_nst(
    alert: Dict,
    bubble_type: str,
    use_claude: bool = True,
    storage: NSTStorage = None,
    anthropic_client=None
) -> Optional[NSTMention]:
    """
    Process a Google Alert for NST tracking.
    
    This should be called during the Google Alerts ingest pipeline.
    
    Args:
        alert: Dict with title, content/snippet, url, published_at, source_name
        bubble_type: Which bubble type this alert belongs to
        use_claude: Whether to use Claude for sentiment classification
        storage: Optional NSTStorage instance
        anthropic_client: Optional pre-initialized Anthropic client
    
    Returns:
        NSTMention if processed successfully
    """
    headline = alert.get('title', '')
    snippet = alert.get('content', alert.get('snippet', ''))
    source = alert.get('source_name', 'google_alerts')
    url = alert.get('url', '')
    published_at = alert.get('published_at', '')
    
    # Classify the mention
    mention = classify_mention(
        headline=headline,
        snippet=snippet,
        source=source,
        url=url,
        published_at=published_at,
        bubble_type=bubble_type
    )
    
    # Optionally add Claude classification
    if use_claude and (mention.alert_category in ['attractor', 'catalyst', 'stress']):
        claude_result = classify_with_claude(
            headline=headline,
            snippet=snippet,
            bubble_type=bubble_type,
            anthropic_client=anthropic_client
        )
        mention.sentiment_score = claude_result.get('sentiment_score')
        mention.sentiment_label = claude_result.get('sentiment_label')
        mention.catalyst_type = claude_result.get('catalyst_type')
    
    # Save to database
    if storage is None:
        storage = NSTStorage()
    
    if storage.is_connected():
        storage.save_mention(mention)
    
    return mention


def infer_bubble_type_from_alert(alert_name: str, headline: str, snippet: str) -> Optional[str]:
    """
    Infer which bubble type an alert belongs to based on content.
    
    Args:
        alert_name: Name of the Google Alert
        headline: Article headline
        snippet: Article snippet
    
    Returns:
        Bubble type string or None if can't determine
    """
    text = (alert_name + ' ' + headline + ' ' + (snippet or '')).lower()
    
    # Score each bubble type by keyword matches
    scores = {}
    for bubble_type, keywords in NST_KEYWORDS.items():
        score = 0
        for category_keywords in keywords.values():
            for kw in category_keywords:
                if kw.lower() in text:
                    score += 1
        scores[bubble_type] = score
    
    if max(scores.values()) == 0:
        return None
    
    return max(scores, key=scores.get)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run NST operations from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NST Integration for FlowOS")
    parser.add_argument('command', choices=['aggregate', 'status', 'density'],
                       help='Command to run')
    parser.add_argument('--bubble', default='AI/Compute',
                       help='Bubble type (default: AI/Compute)')
    parser.add_argument('--days', type=int, default=7,
                       help='Days to aggregate (default: 7)')
    parser.add_argument('--date', help='Date for aggregation (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    storage = NSTStorage()
    
    if args.command == 'aggregate':
        run_daily_aggregation(storage, args.date)
    
    elif args.command == 'status':
        result = get_nst_status(args.bubble, storage)
        print(f"\nNST Status for {args.bubble}:")
        print(f"  Density Level: {result.density_level}")
        print(f"  Total Mentions: {result.total_mentions}")
        print(f"  Catalyst Pending: {result.catalyst_pending}")
        print(f"  Catalyst Type: {result.catalyst_type}")
    
    elif args.command == 'density':
        result = get_narrative_density(args.bubble, args.days, storage)
        print(f"\nNarrativeDensity for {args.bubble} ({args.days} days):")
        print(f"  Status: {result.status}")
        print(f"  Density: {result.narrative_density:.3f}")
        print(f"  Scaled (0-100): {result.scaled}")
        print(f"  Total Mentions: {result.total_mentions}")
        print(f"  Attractor Mentions: {result.attractor_mentions}")


if __name__ == "__main__":
    main()
