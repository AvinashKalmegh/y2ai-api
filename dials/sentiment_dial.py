"""
SENTIMENT DIAL - Market Data Composite
======================================
Combines VIX, VIX Term Structure, Credit Spreads, and Correlations
into a single 0-100 sentiment score.

Components (weighted):
- VIX Level (30%): Current fear gauge
- VIX Term Structure (20%): Contango/backwardation signal
- Credit Spread Momentum (25%): Risk appetite from credit markets
- Correlation Regime (25%): Diversification vs herding

Score Interpretation (0-100):
- 0-25: Complacent (excessive optimism - contrarian caution)
- 25-45: Calm (healthy risk appetite)
- 45-60: Neutral (balanced sentiment)
- 60-75: Anxious (elevated caution - watch for opportunities)
- 75-100: Fearful (high fear - contrarian opportunity)

Note: Lower scores = more complacent/bullish, Higher scores = more fearful/bearish
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
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
    # Component weights
    "weights": {
        "vix": 0.30,
        "term": 0.20,
        "credit": 0.25,
        "corr": 0.25,
    },
    
    # Regime thresholds (composite score)
    "regime_thresholds": {
        "complacent": 25,    # < 25
        "calm": 45,          # 25-45
        "neutral": 60,       # 45-60
        "anxious": 75,       # 60-75
        # > 75 = Fearful
    },
    
    # VIX scoring breakpoints
    "vix_breakpoints": [12, 15, 20, 25, 35],
    
    # Term structure bounds (% deviation from MA)
    "term_bounds": [-10, 10],
    
    # Credit momentum thresholds
    "credit_momentum_thresholds": [-0.3, -0.1, 0.1, 0.3],
    
    # HY spread adjustment thresholds
    "hy_spread_thresholds": {"high": 5.0, "elevated": 4.0, "low": 3.0},
    
    # Correlation scoring breakpoints (as decimal)
    "corr_breakpoints": [0.30, 0.40, 0.50, 0.60, 0.70],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SentimentData:
    """Sentiment dial data."""
    date: str
    # Composite
    sentiment_score: float      # 0-100
    regime: str                 # Complacent/Calm/Neutral/Anxious/Fearful
    interpretation: str
    # Component scores (0-100)
    vix_score: float
    term_score: float
    credit_score: float
    corr_score: float
    # Raw inputs
    vix: float
    vix_ma20: float
    term_spread: float          # % deviation from MA
    hy_spread: float
    credit_momentum: float      # 20D change
    correlation_20d: float      # As decimal (0-1)


# =============================================================================
# SENTIMENT CALCULATOR
# =============================================================================

class SentimentDialCalculator:
    """
    Calculate market sentiment from VIX, credit, and correlation data.
    
    Matches Google Sheets Sentiment_Dial logic exactly.
    
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
    
    def _get_latest(self, table: str) -> Optional[Dict]:
        """Get latest row from table."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table(table) \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.warning(f"Failed to fetch {table}: {e}")
        
        return None
    
    # =========================================================================
    # COMPONENT 1: VIX LEVEL (30% weight)
    # =========================================================================
    
    def calculate_vix_score(self, vix: float) -> float:
        """
        Score VIX level on 0-100 scale.
        
        Lower VIX = more complacent (lower score)
        Higher VIX = more fearful (higher score)
        
        Breakpoints from Google Sheets:
        - < 12: 0
        - 12-15: 0-25 (linear)
        - 15-20: 25-50 (linear)
        - 20-25: 50-75 (linear)
        - 25-35: 75-100 (linear)
        - > 35: 100
        """
        if vix < 12:
            return 0.0
        elif vix < 15:
            return 25 * (vix - 12) / 3
        elif vix < 20:
            return 25 + 25 * (vix - 15) / 5
        elif vix < 25:
            return 50 + 25 * (vix - 20) / 5
        elif vix < 35:
            return 75 + 25 * (vix - 25) / 10
        else:
            return 100.0
    
    # =========================================================================
    # COMPONENT 2: VIX TERM STRUCTURE (20% weight)
    # =========================================================================
    
    def calculate_term_score(self, vix: float, vix_ma20: float) -> tuple:
        """
        Score VIX term structure on 0-100 scale.
        
        Contango (VIX < MA) = complacent = lower score
        Backwardation (VIX > MA) = fear = higher score
        
        termSpread = (VIX - MA20) / MA20 * 100
        
        - <= -10%: 0 (very complacent)
        - >= +10%: 100 (very fearful)
        - Linear scale between
        
        Returns: (score, term_spread_pct)
        """
        if vix_ma20 <= 0:
            return 50.0, 0.0
        
        term_spread = (vix - vix_ma20) / vix_ma20 * 100
        
        if term_spread <= -10:
            score = 0.0
        elif term_spread >= 10:
            score = 100.0
        else:
            score = (term_spread + 10) * 5  # Linear from -10 to +10
        
        return score, term_spread
    
    # =========================================================================
    # COMPONENT 3: CREDIT SPREAD MOMENTUM (25% weight)
    # =========================================================================
    
    def calculate_credit_score(self, credit_momentum: float, hy_spread: float) -> float:
        """
        Score credit spread momentum on 0-100 scale.
        
        Tightening (negative momentum) = complacent = lower score
        Widening (positive momentum) = fear = higher score
        
        Momentum thresholds:
        - <= -0.3: 0 (strong tightening)
        - <= -0.1: 25
        - <= 0.1: 50 (stable)
        - <= 0.3: 75
        - > 0.3: 100 (strong widening)
        
        Adjust for absolute HY spread level:
        - > 5%: +20 (high spreads = more fear)
        - > 4%: +10
        - < 3%: -10 (low spreads = less fear)
        """
        # Base score from momentum
        if credit_momentum <= -0.3:
            score = 0.0
        elif credit_momentum <= -0.1:
            score = 25.0
        elif credit_momentum <= 0.1:
            score = 50.0
        elif credit_momentum <= 0.3:
            score = 75.0
        else:
            score = 100.0
        
        # Adjust for absolute level
        if hy_spread > 5.0:
            score = min(100, score + 20)
        elif hy_spread > 4.0:
            score = min(100, score + 10)
        elif hy_spread < 3.0:
            score = max(0, score - 10)
        
        return score
    
    # =========================================================================
    # COMPONENT 4: CORRELATION REGIME (25% weight)
    # =========================================================================
    
    def calculate_corr_score(self, correlation: float) -> float:
        """
        Score correlation on 0-100 scale.
        
        Lower correlation = healthy diversification = lower score
        Higher correlation = herding/stress = higher score
        
        Correlation breakpoints (as percentage):
        - < 30%: 0
        - 30-40%: 0-25 (linear)
        - 40-50%: 25-50 (linear)
        - 50-60%: 50-75 (linear)
        - 60-70%: 75-100 (linear)
        - > 70%: 100
        """
        # Convert to percentage if needed
        corr_pct = correlation * 100 if correlation <= 1 else correlation
        
        if corr_pct < 30:
            return 0.0
        elif corr_pct < 40:
            return 25 * (corr_pct - 30) / 10
        elif corr_pct < 50:
            return 25 + 25 * (corr_pct - 40) / 10
        elif corr_pct < 60:
            return 50 + 25 * (corr_pct - 50) / 10
        elif corr_pct < 70:
            return 75 + 25 * (corr_pct - 60) / 10
        else:
            return 100.0
    
    # =========================================================================
    # REGIME DETERMINATION
    # =========================================================================
    
    def get_regime(self, score: float) -> tuple:
        """
        Determine sentiment regime from composite score.
        
        Returns: (regime, interpretation)
        """
        thresholds = self.config["regime_thresholds"]
        
        if score < thresholds["complacent"]:
            return "Complacent", "Excessive optimism - contrarian caution"
        elif score < thresholds["calm"]:
            return "Calm", "Healthy risk appetite"
        elif score < thresholds["neutral"]:
            return "Neutral", "Balanced sentiment"
        elif score < thresholds["anxious"]:
            return "Anxious", "Elevated caution - watch for opportunities"
        else:
            return "Fearful", "High fear - contrarian opportunity"
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_vix_data(self) -> tuple:
        """Fetch VIX and VIX MA20."""
        data = self._get_latest("vix_dial_daily")
        
        if data:
            vix = data.get("vix", 15.0)
            vix_ma20 = data.get("vix_20d_ma", data.get("ma_20", vix))
            return vix, vix_ma20
        
        # Try vix_history table
        if self.supabase:
            try:
                response = self.supabase.table("vix_history") \
                    .select("close") \
                    .order("date", desc=True) \
                    .limit(21) \
                    .execute()
                
                if response.data and len(response.data) > 0:
                    vix = response.data[0]["close"]
                    if len(response.data) >= 20:
                        ma20 = sum(r["close"] for r in response.data[:20]) / 20
                    else:
                        ma20 = vix
                    return vix, ma20
            except Exception as e:
                logger.warning(f"VIX history fetch failed: {e}")
        
        return 15.0, 15.0  # Defaults
    
    def fetch_credit_data(self) -> tuple:
        """Fetch HY spread and credit momentum."""
        data = self._get_latest("credit_spread_daily")
        
        if data:
            hy_spread = data.get("hy_spread", 3.0)
            momentum = data.get("spread_momentum_20d", data.get("momentum_20d", 0.0))
            return hy_spread, momentum
        
        return 3.0, 0.0  # Defaults
    
    def fetch_correlation_data(self) -> float:
        """Fetch 20D average correlation."""
        # Try correlation_daily table (from correlation_dial.py)
        data = self._get_latest("correlation_daily")
        if data:
            # correlation_dial.py saves as corr_20d
            corr = data.get("corr_20d", data.get("avg_correlation_20d", data.get("correlation_20d", 0.40)))
            return corr
        
        # Try cluster_daily table (fallback)
        data = self._get_latest("cluster_daily")
        if data:
            return data.get("avg_correlation", 0.40)
        
        return 0.40  # Default
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> SentimentData:
        """
        Calculate sentiment composite score.
        
        Returns:
            SentimentData with all metrics
        """
        logger.info("Calculating sentiment dial...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        weights = self.config["weights"]
        
        # Fetch inputs
        vix, vix_ma20 = self.fetch_vix_data()
        hy_spread, credit_momentum = self.fetch_credit_data()
        correlation = self.fetch_correlation_data()
        
        # Calculate component scores
        vix_score = self.calculate_vix_score(vix)
        term_score, term_spread = self.calculate_term_score(vix, vix_ma20)
        credit_score = self.calculate_credit_score(credit_momentum, hy_spread)
        corr_score = self.calculate_corr_score(correlation)
        
        # Calculate composite
        composite = (
            vix_score * weights["vix"] +
            term_score * weights["term"] +
            credit_score * weights["credit"] +
            corr_score * weights["corr"]
        )
        
        # Get regime
        regime, interpretation = self.get_regime(composite)
        
        logger.info(f"Sentiment: {composite:.1f} ({regime})")
        logger.info(f"  VIX: {vix:.1f} -> score {vix_score:.1f}")
        logger.info(f"  Term: {term_spread:+.1f}% -> score {term_score:.1f}")
        logger.info(f"  Credit: momentum {credit_momentum:.2f}, HY {hy_spread:.2f}% -> score {credit_score:.1f}")
        logger.info(f"  Corr: {correlation*100:.1f}% -> score {corr_score:.1f}")
        
        return SentimentData(
            date=date_str,
            sentiment_score=round(composite, 1),
            regime=regime,
            interpretation=interpretation,
            vix_score=round(vix_score, 1),
            term_score=round(term_score, 1),
            credit_score=round(credit_score, 1),
            corr_score=round(corr_score, 1),
            vix=round(vix, 2),
            vix_ma20=round(vix_ma20, 2),
            term_spread=round(term_spread, 2),
            hy_spread=round(hy_spread, 2),
            credit_momentum=round(credit_momentum, 3),
            correlation_20d=round(correlation, 4),
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: SentimentData) -> Optional[Dict]:
        """Save sentiment data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = asdict(data)
        
        try:
            response = self.supabase.table("sentiment_dial_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved sentiment for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save sentiment: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run sentiment dial calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Sentiment Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = SentimentDialCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*50}")
    print(f"SENTIMENT DIAL: {data.sentiment_score:.1f} ({data.regime})")
    print(f"{'='*50}")
    print(f"Interpretation: {data.interpretation}")
    print(f"\nComponent Scores:")
    print(f"  VIX Level (30%):     {data.vix_score:5.1f}  (VIX: {data.vix:.1f})")
    print(f"  Term Structure (20%): {data.term_score:5.1f}  ({data.term_spread:+.1f}% vs MA20)")
    print(f"  Credit Spreads (25%): {data.credit_score:5.1f}  (HY: {data.hy_spread:.2f}%, Mom: {data.credit_momentum:+.2f})")
    print(f"  Correlations (25%):   {data.corr_score:5.1f}  ({data.correlation_20d*100:.1f}%)")
    print(f"{'='*50}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
