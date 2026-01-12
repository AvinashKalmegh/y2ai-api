"""
NST DIAL - News Sentiment Tracking
==================================
Integrates four news-based indicators into Enhanced AMRI overlay.

Components:
- NCI (Narrative Coherence Index): How clustered/similar articles are
- NPD (Narrative Polarity Drift): Direction of sentiment shift
- EVI (Event Volatility Index): Intensity of news flow
- Burst (Keyword Detection): Sudden keyword frequency spikes

Combined NST Stress Score:
- Burst: 35% weight
- EVI: 35% weight
- NCI: 15% weight
- NPD: 15% weight

Integration:
- Enhanced AMRI = 70% AMRI + 30% NST Stress
- Used as veto layer in Regime Arbiter

Regimes:
- CALM: NST < 25 (healthy news environment)
- MODERATE: NST 25-50 (some stress signals)
- ELEVATED: NST 50-75 (multiple stress signals)
- CRITICAL: NST > 75 (extreme news stress)
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================

# =============================================================================
# SAFE DATETIME PARSER
# =============================================================================

def safe_parse_datetime(dt_string) -> datetime:
    """
    Safely parse datetime strings with variable microsecond precision.
    Handles: '2026-01-10T13:47:55.62942', '2026-01-10T13:47:55Z', None, etc.
    """
    if dt_string is None or dt_string == "":
        return datetime.utcnow()
    
    # Convert to string if needed
    dt_string = str(dt_string)
    
    # Remove Z suffix
    dt_string = dt_string.replace("Z", "")
    
    # Handle microseconds - pad or truncate to 6 digits
    if "." in dt_string:
        main_part, micro_part = dt_string.rsplit(".", 1)
        # Pad to 6 digits or truncate
        micro_part = micro_part[:6].ljust(6, "0")
        dt_string = f"{main_part}.{micro_part}"
    
    try:
        return datetime.fromisoformat(dt_string)
    except ValueError:
        # Fallback: try without microseconds
        try:
            return datetime.fromisoformat(dt_string.split(".")[0])
        except ValueError:
            return datetime.utcnow()

# CONFIGURATION
# =============================================================================

NST_CONFIG = {
    # Component weights for composite score
    "weights": {
        "burst": 0.35,
        "evi": 0.35,
        "nci": 0.15,
        "npd": 0.15,
    },
    
    # NCI thresholds
    "nci": {
        "low": 30,       # 0-30: scattered topics (healthy)
        "moderate": 60,  # 30-60: some clustering
        "high": 80,      # 60-80: narratives converging
        # > 80: extreme coherence (echo chamber)
    },
    
    # NPD thresholds
    "npd": {
        "bullish_threshold": 0.15,
        "bearish_threshold": -0.15,
        "short_window": 3,
        "long_window": 7,
    },
    
    # EVI thresholds
    "evi": {
        "quiet": 20,     # < 20: unusually low activity
        "normal": 50,    # 20-50: typical environment
        "elevated": 80,  # 50-80: above-normal flow
        # > 80: crisis-level coverage
    },
    
    # Burst thresholds
    "burst": {
        "baseline_days": 14,
        "burst_threshold": 2.0,     # 2x baseline = burst
        "high_burst_count": 3,      # 3+ keywords = HIGH_BURST
        "moderate_burst_count": 1,  # 1-2 keywords = MODERATE
    },
    
    # Composite regime thresholds
    "regime_thresholds": {
        "calm": 25,
        "moderate": 50,
        "elevated": 75,
        # > 75 = Critical
    },
    
    # Regime score mappings
    "score_mappings": {
        "burst": {"NORMAL": 0, "MODERATE": 50, "HIGH_BURST": 100},
        "nci": {"LOW": 0, "MODERATE": 50, "HIGH": 100},
        "npd": {"STABLE": 0, "BULLISH_DRIFT": -25, "BEARISH_DRIFT": 50},
        "evi": {"QUIET": 0, "NORMAL": 25, "ELEVATED": 75, "CRISIS": 100},
    },
}

# Stop words for NCI
STOP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
    'could', 'would', 'about', 'which', 'when', 'make', 'like', 'into',
    'just', 'over', 'such', 'than', 'them', 'then', 'these', 'will',
    'with', 'this', 'that', 'from', 'they', 'what', 'says', 'said',
    'how', 'why', 'new', 'news'
}

# Tracked keywords for Burst detection
TRACKED_KEYWORDS = [
    'bubble', 'crash', 'correction', 'selloff', 'capitulation',
    'data center', 'power grid', 'energy', 'infrastructure',
    'nvidia', 'openai', 'microsoft', 'google', 'meta',
    'regulation', 'antitrust', 'china', 'tariff', 'export',
    'layoff', 'hiring freeze', 'cost cutting',
    'ipo', 'funding', 'valuation', 'investment',
    'breakthrough', 'agi', 'superintelligence',
]

# Urgency keywords for EVI
URGENCY_KEYWORDS = [
    'breaking', 'urgent', 'flash', 'alert', 'crisis',
    'crash', 'plunge', 'surge', 'spike', 'collapse',
    'emergency', 'halt', 'suspend', 'investigate',
    'shocking', 'unprecedented', 'historic'
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NCIData:
    """Narrative Coherence Index."""
    score: float
    regime: str
    top_category: Optional[str]
    top_keyword: Optional[str]
    article_count: int


@dataclass
class NPDData:
    """Narrative Polarity Drift."""
    drift: float
    regime: str
    short_avg: float
    long_avg: float


@dataclass
class EVIData:
    """Event Volatility Index."""
    score: float
    regime: str
    volume_component: float
    velocity_component: float
    urgency_component: float


@dataclass
class BurstData:
    """Burst Keyword Detection."""
    regime: str
    bursting_count: int
    top_bursting: List[Dict]


@dataclass
class NSTDialData:
    """Complete NST dial data."""
    date: str
    # Component data
    nci: NCIData
    npd: NPDData
    evi: EVIData
    burst: BurstData
    # Composite
    nst_score: float
    regime: str
    interpretation: str
    # Component scores (0-100)
    nci_score: float
    npd_score: float
    evi_score: float
    burst_score: float


# =============================================================================
# NST CALCULATOR
# =============================================================================

class NSTDialCalculator:
    """
    Calculate NST (News Sentiment Tracking) signals.
    
    Integrates NCI, NPD, EVI, and Burst into a single dial.
    
    Usage:
        calc = NSTDialCalculator()
        data = calc.calculate()
        calc.save_to_supabase(data)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.config = NST_CONFIG
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_articles(self, days: int = 14) -> List[Dict]:
        """Fetch processed articles from Supabase."""
        if not self.supabase:
            return []
        
        try:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            response = self.supabase.table("processed_articles") \
                .select("processed_at, title, sentiment, y2ai_category, one_line_summary") \
                .gte("processed_at", cutoff) \
                .order("processed_at", desc=True) \
                .limit(5000) \
                .execute()
            
            return response.data if response.data else []
        except Exception as e:
            logger.warning(f"Failed to fetch articles: {e}")
            return []
    
    # =========================================================================
    # NCI CALCULATION
    # =========================================================================
    
    def calculate_nci(self, articles: List[Dict]) -> NCIData:
        """
        Calculate Narrative Coherence Index.
        
        Measures how clustered/similar the day's articles are.
        High coherence = everyone talking about same risks = fragility signal.
        """
        today = datetime.utcnow().date()
        today_articles = [
            a for a in articles 
            if safe_parse_datetime(a["processed_at"]).date() == today
        ]
        
        if not today_articles:
            return NCIData(score=0, regime="LOW", top_category=None, top_keyword=None, article_count=0)
        
        # Category concentration
        categories = [a.get("y2ai_category") for a in today_articles if a.get("y2ai_category")]
        category_counts = Counter(categories)
        
        # Keyword extraction from titles
        all_words = []
        for article in today_articles:
            title = article.get("title", "")
            words = re.findall(r'\b[a-z]{4,}\b', title.lower())
            all_words.extend([w for w in words if w not in STOP_WORDS])
        
        word_counts = Counter(all_words)
        
        # Calculate coherence score
        if len(today_articles) < 3:
            score = 0.0
        else:
            # Category concentration (Herfindahl-like)
            if category_counts:
                total = sum(category_counts.values())
                cat_concentration = sum((c/total)**2 for c in category_counts.values())
            else:
                cat_concentration = 0
            
            # Keyword concentration
            if word_counts:
                top_10 = word_counts.most_common(10)
                keyword_concentration = sum(c for _, c in top_10) / len(all_words) if all_words else 0
            else:
                keyword_concentration = 0
            
            # Combine: higher concentration = higher coherence
            score = (cat_concentration * 50 + keyword_concentration * 50) * 100
            score = min(100, max(0, score))
        
        # Determine regime
        thresholds = self.config["nci"]
        if score < thresholds["low"]:
            regime = "LOW"
        elif score < thresholds["moderate"]:
            regime = "MODERATE"
        elif score < thresholds["high"]:
            regime = "HIGH"
        else:
            regime = "EXTREME"
        
        top_category = category_counts.most_common(1)[0][0] if category_counts else None
        top_keyword = word_counts.most_common(1)[0][0] if word_counts else None
        
        return NCIData(
            score=round(score, 1),
            regime=regime,
            top_category=top_category,
            top_keyword=top_keyword,
            article_count=len(today_articles)
        )
    
    # =========================================================================
    # NPD CALCULATION
    # =========================================================================
    
    def calculate_npd(self, articles: List[Dict]) -> NPDData:
        """
        Calculate Narrative Polarity Drift.
        
        Measures direction and speed of sentiment shift.
        Not "is sentiment positive" but "is it becoming more positive or negative?"
        """
        config = self.config["npd"]
        
        # Group by date and calculate daily sentiment
        daily_sentiment = {}
        for article in articles:
            date = safe_parse_datetime(article["processed_at"]).date()
            sentiment = article.get("sentiment", "neutral")
            
            # Convert to score
            score_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
            score = score_map.get(sentiment.lower() if sentiment else "", 0.0)
            
            if date not in daily_sentiment:
                daily_sentiment[date] = []
            daily_sentiment[date].append(score)
        
        # Calculate daily averages
        daily_avg = {d: sum(s)/len(s) for d, s in daily_sentiment.items() if s}
        
        if len(daily_avg) < config["short_window"]:
            return NPDData(drift=0.0, regime="STABLE", short_avg=0.0, long_avg=0.0)
        
        # Sort by date
        sorted_dates = sorted(daily_avg.keys(), reverse=True)
        
        # Short-term average (recent days)
        short_values = [daily_avg[d] for d in sorted_dates[:config["short_window"]]]
        short_avg = sum(short_values) / len(short_values)
        
        # Long-term average
        long_values = [daily_avg[d] for d in sorted_dates[:config["long_window"]]]
        long_avg = sum(long_values) / len(long_values)
        
        # Drift = difference
        drift = short_avg - long_avg
        
        # Determine regime
        if drift > config["bullish_threshold"]:
            regime = "BULLISH_DRIFT"
        elif drift < config["bearish_threshold"]:
            regime = "BEARISH_DRIFT"
        else:
            regime = "STABLE"
        
        return NPDData(
            drift=round(drift, 3),
            regime=regime,
            short_avg=round(short_avg, 3),
            long_avg=round(long_avg, 3)
        )
    
    # =========================================================================
    # EVI CALCULATION
    # =========================================================================
    
    def calculate_evi(self, articles: List[Dict]) -> EVIData:
        """
        Calculate Event Volatility Index.
        
        Measures intensity and urgency of news flow.
        High EVI = lots of breaking news, crisis-level coverage.
        """
        today = datetime.utcnow().date()
        
        # Separate today's articles and baseline
        today_articles = []
        baseline_articles = []
        
        for article in articles:
            date = safe_parse_datetime(article["processed_at"]).date()
            if date == today:
                today_articles.append(article)
            else:
                baseline_articles.append(article)
        
        # Calculate baseline daily average
        baseline_days = 14
        baseline_daily = len(baseline_articles) / baseline_days if baseline_days > 0 else 1
        
        # Volume component (vs baseline)
        volume_ratio = len(today_articles) / baseline_daily if baseline_daily > 0 else 1
        volume_component = min(100, max(0, (volume_ratio - 0.5) * 50))
        
        # Velocity component (articles per hour in last 24h)
        if today_articles:
            times = [safe_parse_datetime(a["processed_at"]) for a in today_articles]
            time_span = (max(times) - min(times)).total_seconds() / 3600 if len(times) > 1 else 24
            velocity = len(today_articles) / max(time_span, 1)
            velocity_component = min(100, velocity * 10)
        else:
            velocity_component = 0
        
        # Urgency component (urgent keywords in titles)
        urgency_count = 0
        for article in today_articles:
            title = article.get("title", "").lower()
            for keyword in URGENCY_KEYWORDS:
                if keyword in title:
                    urgency_count += 1
                    break
        
        urgency_ratio = urgency_count / len(today_articles) if today_articles else 0
        urgency_component = min(100, urgency_ratio * 200)
        
        # Composite EVI
        score = (
            volume_component * 0.40 +
            velocity_component * 0.30 +
            urgency_component * 0.30
        )
        
        # Determine regime
        thresholds = self.config["evi"]
        if score < thresholds["quiet"]:
            regime = "QUIET"
        elif score < thresholds["normal"]:
            regime = "NORMAL"
        elif score < thresholds["elevated"]:
            regime = "ELEVATED"
        else:
            regime = "CRISIS"
        
        return EVIData(
            score=round(score, 1),
            regime=regime,
            volume_component=round(volume_component, 1),
            velocity_component=round(velocity_component, 1),
            urgency_component=round(urgency_component, 1)
        )
    
    # =========================================================================
    # BURST CALCULATION
    # =========================================================================
    
    def calculate_burst(self, articles: List[Dict]) -> BurstData:
        """
        Calculate Burst Keyword Detection.
        
        Detects sudden spikes in keyword frequency.
        """
        config = self.config["burst"]
        today = datetime.utcnow().date()
        
        # Separate today's and baseline articles
        today_articles = []
        baseline_articles = []
        
        for article in articles:
            date = safe_parse_datetime(article["processed_at"]).date()
            if date == today:
                today_articles.append(article)
            else:
                baseline_articles.append(article)
        
        # Count keyword occurrences
        def count_keywords(article_list):
            counts = Counter()
            for article in article_list:
                text = (article.get("title", "") + " " + article.get("one_line_summary", "")).lower()
                for keyword in TRACKED_KEYWORDS:
                    if keyword in text:
                        counts[keyword] += 1
            return counts
        
        today_counts = count_keywords(today_articles)
        baseline_counts = count_keywords(baseline_articles)
        
        # Calculate baseline daily rate
        baseline_days = config["baseline_days"]
        baseline_daily = {k: v / baseline_days for k, v in baseline_counts.items()}
        
        # Find bursting keywords
        bursting = []
        for keyword, today_count in today_counts.items():
            baseline_rate = baseline_daily.get(keyword, 0)
            if baseline_rate > 0:
                ratio = today_count / baseline_rate
                if ratio >= config["burst_threshold"]:
                    bursting.append({
                        "keyword": keyword,
                        "today_count": today_count,
                        "baseline_rate": round(baseline_rate, 2),
                        "ratio": round(ratio, 2)
                    })
        
        # Sort by ratio
        bursting.sort(key=lambda x: -x["ratio"])
        
        # Determine regime
        burst_count = len(bursting)
        if burst_count >= config["high_burst_count"]:
            regime = "HIGH_BURST"
        elif burst_count >= config["moderate_burst_count"]:
            regime = "MODERATE"
        else:
            regime = "NORMAL"
        
        return BurstData(
            regime=regime,
            bursting_count=burst_count,
            top_bursting=bursting[:5]
        )
    
    # =========================================================================
    # COMPOSITE CALCULATION
    # =========================================================================
    
    def calculate_composite(
        self,
        nci: NCIData,
        npd: NPDData,
        evi: EVIData,
        burst: BurstData
    ) -> Tuple[float, str, str]:
        """
        Calculate composite NST score.
        
        Returns: (score, regime, interpretation)
        """
        weights = self.config["weights"]
        mappings = self.config["score_mappings"]
        
        # Convert regimes to scores
        nci_score = mappings["nci"].get(nci.regime, 50)
        npd_score = mappings["npd"].get(npd.regime, 0)
        evi_score = mappings["evi"].get(evi.regime, 25)
        burst_score = mappings["burst"].get(burst.regime, 0)
        
        # Calculate composite
        composite = (
            burst_score * weights["burst"] +
            evi_score * weights["evi"] +
            nci_score * weights["nci"] +
            npd_score * weights["npd"]
        )
        
        # Clamp to 0-100
        composite = min(100, max(0, composite))
        
        # Determine regime
        thresholds = self.config["regime_thresholds"]
        if composite < thresholds["calm"]:
            regime = "CALM"
            interpretation = "Healthy news environment. Normal operations."
        elif composite < thresholds["moderate"]:
            regime = "MODERATE"
            interpretation = "Some stress signals emerging. Monitor closely."
        elif composite < thresholds["elevated"]:
            regime = "ELEVATED"
            interpretation = "Multiple stress signals. Consider reducing exposure."
        else:
            regime = "CRITICAL"
            interpretation = "Extreme news stress. Defensive positioning recommended."
        
        return composite, regime, interpretation
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> NSTDialData:
        """
        Calculate all NST components and composite.
        
        Returns:
            NSTDialData with all metrics
        """
        logger.info("Calculating NST dial...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch articles
        articles = self.fetch_articles(days=14)
        
        if not articles:
            logger.warning("No articles found")
            # Return default/empty data
            nci = NCIData(score=0, regime="LOW", top_category=None, top_keyword=None, article_count=0)
            npd = NPDData(drift=0, regime="STABLE", short_avg=0, long_avg=0)
            evi = EVIData(score=0, regime="QUIET", volume_component=0, velocity_component=0, urgency_component=0)
            burst = BurstData(regime="NORMAL", bursting_count=0, top_bursting=[])
            
            return NSTDialData(
                date=date_str,
                nci=nci, npd=npd, evi=evi, burst=burst,
                nst_score=0, regime="CALM", interpretation="No data available",
                nci_score=0, npd_score=0, evi_score=0, burst_score=0
            )
        
        # Calculate components
        nci = self.calculate_nci(articles)
        npd = self.calculate_npd(articles)
        evi = self.calculate_evi(articles)
        burst = self.calculate_burst(articles)
        
        # Calculate composite
        nst_score, regime, interpretation = self.calculate_composite(nci, npd, evi, burst)
        
        # Get component scores
        mappings = self.config["score_mappings"]
        nci_score = mappings["nci"].get(nci.regime, 50)
        npd_score = mappings["npd"].get(npd.regime, 0)
        evi_score = mappings["evi"].get(evi.regime, 25)
        burst_score = mappings["burst"].get(burst.regime, 0)
        
        logger.info(f"NST Score: {nst_score:.1f} ({regime})")
        logger.info(f"  NCI: {nci.score:.1f} ({nci.regime})")
        logger.info(f"  NPD: {npd.drift:+.3f} ({npd.regime})")
        logger.info(f"  EVI: {evi.score:.1f} ({evi.regime})")
        logger.info(f"  Burst: {burst.bursting_count} keywords ({burst.regime})")
        
        return NSTDialData(
            date=date_str,
            nci=nci,
            npd=npd,
            evi=evi,
            burst=burst,
            nst_score=round(nst_score, 1),
            regime=regime,
            interpretation=interpretation,
            nci_score=nci_score,
            npd_score=npd_score,
            evi_score=evi_score,
            burst_score=burst_score,
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: NSTDialData) -> Optional[Dict]:
        """Save NST data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = {
            "date": data.date,
            "nst_score": data.nst_score,
            "regime": data.regime,
            "interpretation": data.interpretation,
            # Component scores
            "nci_score": data.nci_score,
            "npd_score": data.npd_score,
            "evi_score": data.evi_score,
            "burst_score": data.burst_score,
            # Component details
            "nci_regime": data.nci.regime,
            "nci_raw": data.nci.score,
            "npd_regime": data.npd.regime,
            "npd_drift": data.npd.drift,
            "evi_regime": data.evi.regime,
            "evi_raw": data.evi.score,
            "burst_regime": data.burst.regime,
            "burst_count": data.burst.bursting_count,
        }
        
        try:
            response = self.supabase.table("nst_dial_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved NST data for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save NST data: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run NST dial calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate NST Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = NSTDialCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"NST DIAL: {data.nst_score:.1f} ({data.regime})")
    print(f"{'='*60}")
    print(f"Interpretation: {data.interpretation}")
    print(f"\nComponents:")
    print(f"  NCI (15%):   {data.nci.score:5.1f} ({data.nci.regime}) - {data.nci.article_count} articles")
    print(f"  NPD (15%):   {data.npd.drift:+.3f} ({data.npd.regime})")
    print(f"  EVI (35%):   {data.evi.score:5.1f} ({data.evi.regime})")
    print(f"  Burst (35%): {data.burst.bursting_count} keywords ({data.burst.regime})")
    
    if data.burst.top_bursting:
        print(f"\nBursting Keywords:")
        for kw in data.burst.top_bursting[:3]:
            print(f"  - {kw['keyword']}: {kw['ratio']:.1f}x baseline")
    
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()