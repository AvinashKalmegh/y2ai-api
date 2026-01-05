"""
SENTIMENT DIAL
==============
Aggregates news sentiment from ARGUS-1 processed articles.
Tracks sentiment momentum and divergences.

Metrics:
- Sentiment Score: -100 to +100 (bearish to bullish)
- Sentiment Trend: 5-day moving average direction
- Sentiment Volatility: Standard deviation of daily scores
- Bull/Bear Ratio: Count of positive vs negative articles

Regime Thresholds:
- Euphoric: > +60 (extreme bullishness - contrarian warning)
- Bullish: +20 to +60 (healthy optimism)
- Neutral: -20 to +20 (balanced sentiment)
- Bearish: -60 to -20 (healthy pessimism)
- Panic: < -60 (extreme bearishness - contrarian opportunity)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SENTIMENT_CONFIG = {
    "lookback_days": 7,           # Days for sentiment aggregation
    "trend_days": 5,              # Days for trend calculation
    
    # Regime thresholds (sentiment score -100 to +100)
    "regime_thresholds": {
        "euphoric": 60,     # > 60
        "bullish": 20,      # 20 to 60
        "neutral_low": -20, # -20 to 20 = Neutral
        "bearish": -60,     # -60 to -20
        # < -60 = Panic
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SentimentData:
    """Daily sentiment data."""
    date: str
    # Core metrics
    sentiment_score: float        # -100 to +100
    sentiment_5d_avg: float       # 5-day moving average
    sentiment_trend: str          # Rising, Falling, Stable
    sentiment_volatility: float   # Std dev of daily scores
    # Article counts
    total_articles: int
    bullish_articles: int
    bearish_articles: int
    neutral_articles: int
    bull_bear_ratio: float
    # By pillar
    pillar_sentiment: Dict[str, float]
    # Regime
    regime: str
    interpretation: str


# =============================================================================
# SENTIMENT CALCULATOR
# =============================================================================

class SentimentDialCalculator:
    """
    Calculate sentiment metrics from news data.
    
    Usage:
        calc = SentimentDialCalculator()
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
        
        self.config = SENTIMENT_CONFIG
    
    def fetch_articles(self, days: int = None) -> pd.DataFrame:
        """Fetch processed articles from Supabase."""
        if not self.supabase:
            return pd.DataFrame()
        
        if days is None:
            days = self.config["lookback_days"]
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            response = self.supabase.table("processed_articles") \
                .select("date, sentiment_score, category, pillar, impact_score") \
                .gte("date", start_date) \
                .order("date", desc=True) \
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
        except Exception as e:
            logger.warning(f"Failed to fetch articles: {e}")
        
        return pd.DataFrame()
    
    def fetch_daily_sentiment_history(self, days: int = 10) -> pd.DataFrame:
        """Fetch daily sentiment history."""
        if not self.supabase:
            return pd.DataFrame()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            response = self.supabase.table("sentiment_dial_daily") \
                .select("date, sentiment_score") \
                .gte("date", start_date) \
                .order("date", desc=True) \
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
        except Exception as e:
            logger.warning(f"Failed to fetch sentiment history: {e}")
        
        return pd.DataFrame()
    
    def calculate_sentiment_score(self, articles: pd.DataFrame) -> float:
        """
        Calculate aggregate sentiment score.
        Weighted by impact score.
        """
        if articles.empty or "sentiment_score" not in articles.columns:
            return 0
        
        # Filter valid sentiment scores
        valid = articles[articles["sentiment_score"].notna()]
        
        if valid.empty:
            return 0
        
        # Weighted by impact score if available
        if "impact_score" in valid.columns and valid["impact_score"].notna().any():
            weights = valid["impact_score"].fillna(1)
            weighted_sum = (valid["sentiment_score"] * weights).sum()
            score = weighted_sum / weights.sum()
        else:
            score = valid["sentiment_score"].mean()
        
        # Scale to -100 to +100
        return max(-100, min(100, score * 100))
    
    def calculate_pillar_sentiment(self, articles: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment by pillar."""
        if articles.empty or "pillar" not in articles.columns:
            return {}
        
        pillar_sentiment = {}
        
        for pillar in articles["pillar"].dropna().unique():
            pillar_articles = articles[articles["pillar"] == pillar]
            score = self.calculate_sentiment_score(pillar_articles)
            pillar_sentiment[pillar] = round(score, 1)
        
        return pillar_sentiment
    
    def get_regime(self, score: float) -> str:
        """Determine regime from sentiment score."""
        thresholds = self.config["regime_thresholds"]
        
        if score > thresholds["euphoric"]:
            return "Euphoric"
        elif score > thresholds["bullish"]:
            return "Bullish"
        elif score > thresholds["neutral_low"]:
            return "Neutral"
        elif score > thresholds["bearish"]:
            return "Bearish"
        else:
            return "Panic"
    
    def get_trend(self, current: float, avg_5d: float) -> str:
        """Determine sentiment trend."""
        diff = current - avg_5d
        
        if diff > 5:
            return "Rising"
        elif diff < -5:
            return "Falling"
        else:
            return "Stable"
    
    def get_interpretation(self, regime: str, score: float, trend: str) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            "Euphoric": f"Extreme bullishness ({score:+.0f}) - contrarian warning, sentiment may be overextended",
            "Bullish": f"Healthy optimism ({score:+.0f}) - positive news flow supporting market",
            "Neutral": f"Balanced sentiment ({score:+.0f}) - mixed signals, watch for direction",
            "Bearish": f"Elevated pessimism ({score:+.0f}) - negative news dominating",
            "Panic": f"Extreme bearishness ({score:+.0f}) - contrarian opportunity if fundamentals intact"
        }
        base = interpretations.get(regime, f"Sentiment score: {score:+.0f}")
        return f"{base}. Trend: {trend}"
    
    def calculate(self) -> SentimentData:
        """Main calculation: compute sentiment metrics."""
        logger.info("Calculating sentiment metrics...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch articles
        articles = self.fetch_articles()
        
        if articles.empty:
            logger.warning("No article data available")
            return SentimentData(
                date=date_str,
                sentiment_score=0,
                sentiment_5d_avg=0,
                sentiment_trend="Unknown",
                sentiment_volatility=0,
                total_articles=0,
                bullish_articles=0,
                bearish_articles=0,
                neutral_articles=0,
                bull_bear_ratio=1.0,
                pillar_sentiment={},
                regime="Unknown",
                interpretation="No sentiment data available"
            )
        
        # Calculate sentiment score
        sentiment_score = self.calculate_sentiment_score(articles)
        
        # Get historical scores for trend
        history = self.fetch_daily_sentiment_history()
        if not history.empty and len(history) >= 5:
            sentiment_5d_avg = history["sentiment_score"].head(5).mean()
            sentiment_volatility = history["sentiment_score"].std()
        else:
            sentiment_5d_avg = sentiment_score
            sentiment_volatility = 0
        
        # Trend
        trend = self.get_trend(sentiment_score, sentiment_5d_avg)
        
        # Article counts
        total = len(articles)
        if "sentiment_score" in articles.columns:
            bullish = len(articles[articles["sentiment_score"] > 0.1])
            bearish = len(articles[articles["sentiment_score"] < -0.1])
            neutral = total - bullish - bearish
        else:
            bullish = bearish = neutral = 0
        
        bull_bear_ratio = bullish / bearish if bearish > 0 else (bullish if bullish > 0 else 1.0)
        
        # Pillar sentiment
        pillar_sentiment = self.calculate_pillar_sentiment(articles)
        
        # Regime
        regime = self.get_regime(sentiment_score)
        interpretation = self.get_interpretation(regime, sentiment_score, trend)
        
        result = SentimentData(
            date=date_str,
            sentiment_score=round(sentiment_score, 1),
            sentiment_5d_avg=round(sentiment_5d_avg, 1),
            sentiment_trend=trend,
            sentiment_volatility=round(sentiment_volatility, 2),
            total_articles=total,
            bullish_articles=bullish,
            bearish_articles=bearish,
            neutral_articles=neutral,
            bull_bear_ratio=round(bull_bear_ratio, 2),
            pillar_sentiment=pillar_sentiment,
            regime=regime,
            interpretation=interpretation
        )
        
        logger.info(f"Sentiment: {sentiment_score:+.1f}, Regime: {regime}")
        
        return result
    
    def save_to_supabase(self, data: SentimentData) -> bool:
        """Save sentiment data to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("sentiment_dial_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentiment Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = SentimentDialCalculator()
    
    print(f"\n{'='*60}")
    print("SENTIMENT DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"SENTIMENT SCORE: {data.sentiment_score:+.1f}")
    print(f"REGIME: {data.regime}")
    print(f"TREND: {data.sentiment_trend}")
    print(f"{'='*40}")
    
    print(f"\n5D Average: {data.sentiment_5d_avg:+.1f}")
    print(f"Volatility: {data.sentiment_volatility:.2f}")
    
    print(f"\n{'='*40}")
    print("ARTICLE BREAKDOWN")
    print(f"{'='*40}")
    print(f"  Total: {data.total_articles}")
    print(f"  Bullish: {data.bullish_articles}")
    print(f"  Bearish: {data.bearish_articles}")
    print(f"  Neutral: {data.neutral_articles}")
    print(f"  Bull/Bear Ratio: {data.bull_bear_ratio:.2f}")
    
    if data.pillar_sentiment:
        print(f"\n{'='*40}")
        print("PILLAR SENTIMENT")
        print(f"{'='*40}")
        for pillar, score in data.pillar_sentiment.items():
            print(f"  {pillar}: {score:+.1f}")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
