"""
MCI (Market Condition Index)
============================
Directional asymmetry indicator measuring market bias toward melt-up or collapse.

Scale: -100 to +100
  > +40: Reflexive Melt-Up (ride winners, protect downside)
  +10 to +40: Extension Zone (trend intact but weakening)
  -10 to +10: Knife Edge (dangerous - can flip either way)
  -10 to -40: Collapse Bias (prioritize defense)
  < -40: Break Path (structural collapse likely)

Components (4 factors, equal weight 25 each):
  1. Breadth Momentum - 5D change in overall breadth
  2. VIX Trend - 10D VIX direction (falling = bullish)
  3. Credit Trend - 10D spread direction (tightening = bullish)
  4. Pillar Momentum - Avg 5D pillar returns

Used by: BubbleOS Rotation Engine, Portfolio allocation decisions
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MCI_CONFIG = {
    # Lookback periods (trading days)
    "BREADTH_LOOKBACK": 5,
    "VIX_LOOKBACK": 10,
    "CREDIT_LOOKBACK": 10,
    "PILLAR_LOOKBACK": 5,
    
    # Component weights (must sum to 100)
    "WEIGHT_BREADTH": 25,
    "WEIGHT_VIX": 25,
    "WEIGHT_CREDIT": 25,
    "WEIGHT_PILLAR": 25,
    
    # Normalization thresholds (raw change that = max score)
    "BREADTH_THRESHOLD": 10,    # +/- 10% breadth change = max score
    "VIX_THRESHOLD": 3,         # +/- 3 VIX points = max score
    "CREDIT_THRESHOLD": 0.3,    # +/- 30 bps spread change = max score
    "PILLAR_THRESHOLD": 3,      # +/- 3% pillar return = max score
}

# Regime boundaries
MCI_REGIMES = {
    "Melt-Up": (40, 101),        # > +40
    "Extension": (10, 40),       # +10 to +40
    "Knife Edge": (-10, 10),     # -10 to +10
    "Collapse Bias": (-40, -10), # -10 to -40
    "Break Path": (-101, -40)    # < -40
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MCIComponent:
    """Single component of MCI."""
    name: str
    raw_value: float       # Original value (e.g., breadth change %)
    normalized: float      # Normalized to -1 to +1
    score: float           # Weighted score (-25 to +25)
    detail: str            # Human-readable detail


@dataclass
class MCIData:
    """Complete MCI calculation result."""
    date: str
    mci_score: float
    regime: str
    interpretation: str
    # Components
    breadth_component: float
    vix_component: float
    credit_component: float
    pillar_component: float
    # Raw values for debugging
    breadth_raw: float = 0.0
    vix_raw: float = 0.0
    credit_raw: float = 0.0
    pillar_raw: float = 0.0


# =============================================================================
# MCI CALCULATOR
# =============================================================================

class MCICalculator:
    """
    Calculate Market Condition Index.
    
    Usage:
        calc = MCICalculator()
        mci = calc.calculate()
        calc.save_to_supabase(mci)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize with optional Supabase credentials."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    # =========================================================================
    # COMPONENT CALCULATIONS
    # =========================================================================
    
    def calculate_breadth_momentum(self) -> MCIComponent:
        """
        Calculate Breadth Momentum component.
        Measures 5-day change in overall market breadth.
        Rising breadth = bullish (positive score)
        """
        config = MCI_CONFIG
        
        try:
            # Try to get from Supabase breadth_daily
            if self.supabase:
                response = self.supabase.table("breadth_daily") \
                    .select("date, breadth_20d") \
                    .order("date", desc=True) \
                    .limit(config["BREADTH_LOOKBACK"] + 1) \
                    .execute()
                
                if response.data and len(response.data) > config["BREADTH_LOOKBACK"]:
                    current = response.data[0]["breadth_20d"] * 100  # Convert to %
                    previous = response.data[config["BREADTH_LOOKBACK"]]["breadth_20d"] * 100
                    breadth_change = current - previous
                    
                    # Normalize to -1 to +1
                    normalized = max(-1, min(1, breadth_change / config["BREADTH_THRESHOLD"]))
                    score = normalized * config["WEIGHT_BREADTH"]
                    
                    return MCIComponent(
                        name="Breadth Momentum",
                        raw_value=breadth_change,
                        normalized=normalized,
                        score=round(score, 1),
                        detail=f"Current: {current:.1f}%, {config['BREADTH_LOOKBACK']}D ago: {previous:.1f}%, Change: {breadth_change:+.1f}%"
                    )
        except Exception as e:
            logger.warning(f"Breadth calculation failed: {e}")
        
        # Default if no data
        return MCIComponent(
            name="Breadth Momentum",
            raw_value=0,
            normalized=0,
            score=0,
            detail="No breadth data available"
        )
    
    def calculate_vix_trend(self) -> MCIComponent:
        """
        Calculate VIX Trend component.
        Measures 10-day VIX direction.
        Falling VIX = bullish (positive score)
        """
        config = MCI_CONFIG
        
        try:
            # Fetch VIX data
            vix_data = self._fetch_vix_history(config["VIX_LOOKBACK"] + 5)
            
            if vix_data is not None and len(vix_data) > config["VIX_LOOKBACK"]:
                current_vix = vix_data.iloc[-1]
                previous_vix = vix_data.iloc[-(config["VIX_LOOKBACK"] + 1)]
                vix_change = current_vix - previous_vix
                
                # Falling VIX is bullish, so invert the sign
                # Normalize to -1 to +1
                normalized = max(-1, min(1, -vix_change / config["VIX_THRESHOLD"]))
                score = normalized * config["WEIGHT_VIX"]
                
                return MCIComponent(
                    name="VIX Trend",
                    raw_value=vix_change,
                    normalized=normalized,
                    score=round(score, 1),
                    detail=f"Current: {current_vix:.1f}, {config['VIX_LOOKBACK']}D ago: {previous_vix:.1f}, Change: {vix_change:+.1f}"
                )
        except Exception as e:
            logger.warning(f"VIX calculation failed: {e}")
        
        return MCIComponent(
            name="VIX Trend",
            raw_value=0,
            normalized=0,
            score=0,
            detail="No VIX data available"
        )
    
    def calculate_credit_trend(self) -> MCIComponent:
        """
        Calculate Credit Trend component.
        Measures 10-day credit spread direction.
        Tightening spreads = bullish (positive score)
        """
        config = MCI_CONFIG
        
        try:
            # Fetch credit spread data (HY - Treasury)
            spread_data = self._fetch_credit_spread_history(config["CREDIT_LOOKBACK"] + 5)
            
            if spread_data is not None and len(spread_data) > config["CREDIT_LOOKBACK"]:
                current_spread = spread_data.iloc[-1]
                previous_spread = spread_data.iloc[-(config["CREDIT_LOOKBACK"] + 1)]
                spread_change = current_spread - previous_spread
                
                # Tightening (negative change) is bullish, so invert
                # Normalize to -1 to +1
                normalized = max(-1, min(1, -spread_change / config["CREDIT_THRESHOLD"]))
                score = normalized * config["WEIGHT_CREDIT"]
                
                return MCIComponent(
                    name="Credit Trend",
                    raw_value=spread_change,
                    normalized=normalized,
                    score=round(score, 1),
                    detail=f"Current: {current_spread*100:.0f}bps, {config['CREDIT_LOOKBACK']}D ago: {previous_spread*100:.0f}bps, Change: {spread_change*100:+.0f}bps"
                )
        except Exception as e:
            logger.warning(f"Credit calculation failed: {e}")
        
        return MCIComponent(
            name="Credit Trend",
            raw_value=0,
            normalized=0,
            score=0,
            detail="No credit spread data available"
        )
    
    def calculate_pillar_momentum(self) -> MCIComponent:
        """
        Calculate Pillar Momentum component.
        Average 5-day returns across all pillars.
        Positive returns = bullish
        """
        config = MCI_CONFIG
        
        try:
            # Try to get from Supabase pillar_index_daily
            if self.supabase:
                response = self.supabase.table("pillar_index_daily") \
                    .select("date, infra_5d, enterprise_5d, macro_5d, financial_5d, productivity_5d, demand_5d") \
                    .order("date", desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    row = response.data[0]
                    # Average of all pillar 5D momentum
                    momenta = [
                        row.get("infra_5d", 0) or 0,
                        row.get("enterprise_5d", 0) or 0,
                        row.get("macro_5d", 0) or 0,
                        row.get("financial_5d", 0) or 0,
                        row.get("productivity_5d", 0) or 0,
                        row.get("demand_5d", 0) or 0
                    ]
                    avg_momentum = sum(momenta) / len(momenta) * 100  # Convert to %
                    
                    # Normalize to -1 to +1
                    normalized = max(-1, min(1, avg_momentum / config["PILLAR_THRESHOLD"]))
                    score = normalized * config["WEIGHT_PILLAR"]
                    
                    return MCIComponent(
                        name="Pillar Momentum",
                        raw_value=avg_momentum,
                        normalized=normalized,
                        score=round(score, 1),
                        detail=f"Avg 5D return: {avg_momentum:+.2f}%"
                    )
        except Exception as e:
            logger.warning(f"Pillar momentum calculation failed: {e}")
        
        return MCIComponent(
            name="Pillar Momentum",
            raw_value=0,
            normalized=0,
            score=0,
            detail="No pillar data available"
        )
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def _fetch_vix_history(self, days: int) -> Optional[pd.Series]:
        """Fetch VIX history."""
        # Try Supabase first
        if self.supabase:
            try:
                start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
                response = self.supabase.table("vix_history") \
                    .select("date, close") \
                    .gte("date", start_date) \
                    .order("date") \
                    .execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    return df.set_index("date")["close"]
            except Exception as e:
                logger.warning(f"Supabase VIX fetch failed: {e}")
        
        # Fall back to yfinance
        if YFINANCE_AVAILABLE:
            try:
                vix = yf.Ticker("^VIX")
                hist = vix.history(period=f"{days*2}d")
                if len(hist) > 0:
                    return hist["Close"]
            except Exception as e:
                logger.warning(f"yfinance VIX fetch failed: {e}")
        
        return None
    
    def _fetch_credit_spread_history(self, days: int) -> Optional[pd.Series]:
        """
        Fetch credit spread history.
        Using HYG-TLT spread as proxy for credit spreads.
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # HYG = High Yield Corporate Bond ETF
            # TLT = 20+ Year Treasury ETF
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)
            
            hyg = yf.Ticker("HYG")
            tlt = yf.Ticker("TLT")
            
            hyg_hist = hyg.history(start=start_date, end=end_date)
            tlt_hist = tlt.history(start=start_date, end=end_date)
            
            if len(hyg_hist) > 0 and len(tlt_hist) > 0:
                # Align dates
                hyg_close = hyg_hist["Close"]
                tlt_close = tlt_hist["Close"]
                
                # Calculate yield differential proxy
                # HYG yield is inversely related to price
                # Higher spread = lower HYG price relative to TLT
                spread = (tlt_close / hyg_close - 1)  # Rough proxy
                
                return spread
        except Exception as e:
            logger.warning(f"Credit spread fetch failed: {e}")
        
        return None
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> MCIData:
        """
        Calculate MCI score and regime.
        
        Returns:
            MCIData with score, regime, and component breakdown
        """
        logger.info("Calculating MCI...")
        
        # Calculate all components
        breadth = self.calculate_breadth_momentum()
        vix = self.calculate_vix_trend()
        credit = self.calculate_credit_trend()
        pillar = self.calculate_pillar_momentum()
        
        # Sum component scores
        mci_score = breadth.score + vix.score + credit.score + pillar.score
        
        # Clamp to -100 to +100
        mci_score = max(-100, min(100, mci_score))
        
        # Determine regime
        regime = self._get_regime(mci_score)
        interpretation = self._get_interpretation(regime)
        
        logger.info(f"MCI: {mci_score:.1f} ({regime})")
        logger.info(f"  Breadth: {breadth.score:+.1f}")
        logger.info(f"  VIX: {vix.score:+.1f}")
        logger.info(f"  Credit: {credit.score:+.1f}")
        logger.info(f"  Pillar: {pillar.score:+.1f}")
        
        return MCIData(
            date=datetime.now().strftime("%Y-%m-%d"),
            mci_score=round(mci_score, 1),
            regime=regime,
            interpretation=interpretation,
            breadth_component=breadth.score,
            vix_component=vix.score,
            credit_component=credit.score,
            pillar_component=pillar.score,
            breadth_raw=breadth.raw_value,
            vix_raw=vix.raw_value,
            credit_raw=credit.raw_value,
            pillar_raw=pillar.raw_value
        )
    
    def _get_regime(self, score: float) -> str:
        """Determine MCI regime from score."""
        for regime, (low, high) in MCI_REGIMES.items():
            if low <= score < high:
                return regime
        return "Unknown"
    
    def _get_interpretation(self, regime: str) -> str:
        """Get action interpretation for regime."""
        interpretations = {
            "Melt-Up": "Ride winners, protect downside with trailing stops",
            "Extension": "Trend intact but weakening - trim beta, tighten stops",
            "Knife Edge": "Dangerous - can flip either way. Reduce exposure.",
            "Collapse Bias": "Prioritize defense - cut speculative positions",
            "Break Path": "Structural collapse likely - de-risk fully"
        }
        return interpretations.get(regime, "Unknown regime")
    
    # =========================================================================
    # ROTATION ENGINE INTEGRATION
    # =========================================================================
    
    def get_rotation_recommendation(self, break_probability: float = None) -> Dict:
        """
        Get rotation recommendation based on MCI and Break Probability.
        Implements BubbleOS Rotation Engine matrix.
        
        Args:
            break_probability: From AMRI (0-100). If None, fetches from Supabase.
            
        Returns:
            Dict with recommendation and action
        """
        mci = self.calculate()
        
        # Get break probability if not provided
        if break_probability is None:
            break_probability = self._get_break_probability()
        
        # Apply rotation matrix
        bp = break_probability
        score = mci.mci_score
        
        if bp < 30:
            if score > 30:
                recommendation = "Aggressive Beta"
                action = "Stay long leaders, full exposure"
            else:
                recommendation = "Normal Operations"
                action = "No change required"
        elif bp < 55:
            if score > 20:
                recommendation = "Respect Risk"
                action = "Stay long but trim exposures, add light hedges"
            elif score > -10:
                recommendation = "Reduce Beta"
                action = "Prepare rotation, reduce concentration"
            else:
                recommendation = "Defensive Tilt"
                action = "Cut speculative positions"
        elif bp < 75:
            if score > 0:
                recommendation = "Hedge Melt-Up"
                action = "Rotate to quality, hedge tail risk"
            else:
                recommendation = "Cut Risk"
                action = "Reduce exposure 20-35%, defensive posture"
        else:  # bp >= 75
            if score > 0:
                recommendation = "Break Imminent"
                action = "Do NOT add risk, prepare for regime flip"
            else:
                recommendation = "Phase 5 Protocol"
                action = "De-risk fully, defensive rotations, raise cash"
        
        return {
            "break_probability": bp,
            "mci_score": score,
            "mci_regime": mci.regime,
            "recommendation": recommendation,
            "action": action
        }
    
    def _get_break_probability(self) -> float:
        """Get break probability from AMRI data."""
        if not self.supabase:
            return 30  # Default moderate
        
        try:
            response = self.supabase.table("amri_daily") \
                .select("break_probability") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0].get("break_probability", 30)
        except Exception as e:
            logger.warning(f"Failed to get break probability: {e}")
        
        return 30
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, data: MCIData) -> bool:
        """Save MCI data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("mci_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved MCI data for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save MCI: {e}")
            return False
    
    def get_latest(self) -> Optional[MCIData]:
        """Get most recent MCI from Supabase."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table("mci_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return MCIData(**response.data[0])
        except Exception as e:
            logger.error(f"Failed to get latest MCI: {e}")
        return None


# =============================================================================
# SUPABASE TABLE
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mci_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    mci_score FLOAT,
    regime VARCHAR(20),
    interpretation TEXT,
    breadth_component FLOAT,
    vix_component FLOAT,
    credit_component FLOAT,
    pillar_component FLOAT,
    breadth_raw FLOAT,
    vix_raw FLOAT,
    credit_raw FLOAT,
    pillar_raw FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mci_daily_date ON mci_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run MCI calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate MCI")
    parser.add_argument("--rotation", action="store_true", help="Show rotation recommendation")
    parser.add_argument("--bp", type=float, default=None, help="Break probability (0-100)")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = MCICalculator()
    
    print(f"\n{'='*60}")
    print("MCI (Market Condition Index)")
    print(f"{'='*60}\n")
    
    mci = calc.calculate()
    
    print(f"Date: {mci.date}")
    print(f"\n{'='*40}")
    print(f"MCI SCORE: {mci.mci_score:+.1f}")
    print(f"REGIME: {mci.regime}")
    print(f"{'='*40}")
    print(f"\n{mci.interpretation}")
    
    print(f"\n{'='*40}")
    print("COMPONENTS")
    print(f"{'='*40}")
    print(f"  Breadth Momentum: {mci.breadth_component:+.1f} (raw: {mci.breadth_raw:+.1f}%)")
    print(f"  VIX Trend:        {mci.vix_component:+.1f} (raw: {mci.vix_raw:+.1f})")
    print(f"  Credit Trend:     {mci.credit_component:+.1f} (raw: {mci.credit_raw*100:+.0f}bps)")
    print(f"  Pillar Momentum:  {mci.pillar_component:+.1f} (raw: {mci.pillar_raw:+.2f}%)")
    
    print(f"\n{'='*40}")
    print("REGIME GUIDE")
    print(f"{'='*40}")
    print("  > +40  Melt-Up:       Ride winners, protect downside")
    print("  +10-40 Extension:     Trend intact, trim beta")
    print("  -10-10 Knife Edge:    Dangerous, reduce exposure")
    print("  -40--10 Collapse:     Prioritize defense")
    print("  < -40  Break Path:    De-risk fully")
    
    if args.rotation:
        print(f"\n{'='*40}")
        print("ROTATION RECOMMENDATION")
        print(f"{'='*40}")
        rec = calc.get_rotation_recommendation(break_probability=args.bp)
        print(f"  Break Probability: {rec['break_probability']:.0f}%")
        print(f"  MCI: {rec['mci_score']:+.1f} ({rec['mci_regime']})")
        print(f"\n  Recommendation: {rec['recommendation']}")
        print(f"  Action: {rec['action']}")
    
    if args.save:
        if calc.save_to_supabase(mci):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
