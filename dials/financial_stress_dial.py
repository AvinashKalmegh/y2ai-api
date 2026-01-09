"""
FINANCIAL STRESS DIAL
=====================
Tracks St. Louis Fed Financial Stress Index (STLFSI4) as a direct
measure of financial market stress and liquidity conditions.

Data Source:
- FRED API: STLFSI4 (St. Louis Fed Financial Stress Index)
- Updated weekly (every Thursday)

Interpretation (STLFSI4 is standardized, 0 = historical average):
- < -1.0: Very Low Stress (unusually calm markets)
- -1.0 to -0.5: Low Stress
- -0.5 to 0.5: Normal
- 0.5 to 1.0: Elevated Stress
- 1.0 to 2.0: High Stress
- > 2.0: Crisis-Level Stress

Historical Context:
- 2008 Crisis: STLFSI4 peaked at ~5.5
- 2020 COVID: STLFSI4 peaked at ~5.0
- Normal range: -1.0 to +1.0
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import requests
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

FINANCIAL_STRESS_CONFIG = {
    "fred_api_key": os.getenv("FRED_API_KEY"),
    "fred_base_url": "https://api.stlouisfed.org/fred/series/observations",
    "series_id": "STLFSI4",  # St. Louis Fed Financial Stress Index
    
    # Regime thresholds (STLFSI4 is standardized around 0)
    "regime_thresholds": {
        "very_low": -1.0,    # < -1.0: Unusually calm
        "low": -0.5,         # -1.0 to -0.5: Low stress
        "normal_low": 0.5,   # -0.5 to 0.5: Normal
        "elevated": 1.0,     # 0.5 to 1.0: Elevated
        "high": 2.0,         # 1.0 to 2.0: High
        # > 2.0: Crisis
    },
    
    # Score conversion (STLFSI4 to 0-100 scale)
    # Maps -2 to +4 range onto 0-100
    "score_range": {
        "min_stlfsi": -2.0,
        "max_stlfsi": 4.0,
    },
    
    # Multiplier impact on position sizing
    "multipliers": {
        "very_low": 1.0,     # Full risk
        "low": 1.0,
        "normal": 1.0,
        "elevated": 0.85,    # Reduce 15%
        "high": 0.70,        # Reduce 30%
        "crisis": 0.50,      # Reduce 50%
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FinancialStressData:
    """Financial stress data."""
    date: str
    # Raw STLFSI4 value
    stlfsi4: float
    # Converted to 0-100 scale
    stress_score: float
    # Regime
    regime: str
    # Position sizing multiplier
    multiplier: float
    # Trend
    stlfsi4_1w_ago: Optional[float]
    stlfsi4_change: Optional[float]
    trend: str
    # Interpretation
    interpretation: str


# =============================================================================
# FINANCIAL STRESS CALCULATOR
# =============================================================================

class FinancialStressCalculator:
    """
    Calculate financial stress from STLFSI4.
    
    Usage:
        calc = FinancialStressCalculator()
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
        
        self.config = FINANCIAL_STRESS_CONFIG
        self.fred_key = self.config["fred_api_key"]
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_stlfsi4(self, limit: int = 10) -> Optional[Dict]:
        """
        Fetch STLFSI4 from FRED API.
        
        Returns dict with current and historical values.
        """
        if not self.fred_key:
            logger.warning("FRED_API_KEY not set")
            return None
        
        params = {
            "series_id": self.config["series_id"],
            "api_key": self.fred_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        
        try:
            response = requests.get(
                self.config["fred_base_url"],
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                if observations:
                    # Parse observations
                    values = []
                    for obs in observations:
                        if obs["value"] != ".":
                            values.append({
                                "date": obs["date"],
                                "value": float(obs["value"])
                            })
                    
                    if values:
                        return {
                            "current": values[0],
                            "previous": values[1] if len(values) > 1 else None,
                            "history": values
                        }
            
            logger.warning(f"FRED API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch STLFSI4: {e}")
        
        return None
    
    # =========================================================================
    # CALCULATIONS
    # =========================================================================
    
    def stlfsi4_to_score(self, stlfsi4: float) -> float:
        """
        Convert STLFSI4 to 0-100 stress score.
        
        STLFSI4 typically ranges from -2 to +4.
        Map this to 0-100 scale.
        """
        score_range = self.config["score_range"]
        min_val = score_range["min_stlfsi"]
        max_val = score_range["max_stlfsi"]
        
        # Linear mapping
        score = (stlfsi4 - min_val) / (max_val - min_val) * 100
        
        # Clamp to 0-100
        return min(100, max(0, score))
    
    def get_regime(self, stlfsi4: float) -> tuple:
        """
        Determine stress regime from STLFSI4 value.
        
        Returns: (regime, multiplier, interpretation)
        """
        thresholds = self.config["regime_thresholds"]
        multipliers = self.config["multipliers"]
        
        if stlfsi4 < thresholds["very_low"]:
            regime = "Very Low"
            mult = multipliers["very_low"]
            interp = "Unusually calm markets. Possible complacency risk."
        elif stlfsi4 < thresholds["low"]:
            regime = "Low"
            mult = multipliers["low"]
            interp = "Low financial stress. Favorable conditions."
        elif stlfsi4 < thresholds["normal_low"]:
            regime = "Normal"
            mult = multipliers["normal"]
            interp = "Normal financial conditions."
        elif stlfsi4 < thresholds["elevated"]:
            regime = "Elevated"
            mult = multipliers["elevated"]
            interp = "Elevated stress. Monitor credit spreads and volatility."
        elif stlfsi4 < thresholds["high"]:
            regime = "High"
            mult = multipliers["high"]
            interp = "High financial stress. Reduce risk exposure."
        else:
            regime = "Crisis"
            mult = multipliers["crisis"]
            interp = "Crisis-level stress. Defensive positioning required."
        
        return regime, mult, interp
    
    def get_trend(self, current: float, previous: Optional[float]) -> tuple:
        """
        Determine stress trend.
        
        Returns: (change, trend_label)
        """
        if previous is None:
            return None, "Unknown"
        
        change = current - previous
        
        if change > 0.3:
            trend = "Rising Sharply"
        elif change > 0.1:
            trend = "Rising"
        elif change > -0.1:
            trend = "Stable"
        elif change > -0.3:
            trend = "Falling"
        else:
            trend = "Falling Sharply"
        
        return round(change, 3), trend
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> FinancialStressData:
        """
        Calculate financial stress metrics.
        
        Returns:
            FinancialStressData with all metrics
        """
        logger.info("Calculating financial stress...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch STLFSI4
        fred_data = self.fetch_stlfsi4(limit=10)
        
        if fred_data is None or not fred_data.get("current"):
            logger.warning("Could not fetch STLFSI4")
            return FinancialStressData(
                date=date_str,
                stlfsi4=0.0,
                stress_score=33.3,  # Default to "normal"
                regime="Unknown",
                multiplier=1.0,
                stlfsi4_1w_ago=None,
                stlfsi4_change=None,
                trend="Unknown",
                interpretation="Unable to fetch STLFSI4 data."
            )
        
        # Extract values
        current = fred_data["current"]["value"]
        previous = fred_data["previous"]["value"] if fred_data["previous"] else None
        
        # Calculate metrics
        stress_score = self.stlfsi4_to_score(current)
        regime, multiplier, interpretation = self.get_regime(current)
        change, trend = self.get_trend(current, previous)
        
        logger.info(f"STLFSI4: {current:.3f} ({regime})")
        logger.info(f"Stress Score: {stress_score:.1f}")
        logger.info(f"Trend: {trend} ({change:+.3f})" if change else f"Trend: {trend}")
        
        return FinancialStressData(
            date=date_str,
            stlfsi4=round(current, 3),
            stress_score=round(stress_score, 1),
            regime=regime,
            multiplier=multiplier,
            stlfsi4_1w_ago=round(previous, 3) if previous else None,
            stlfsi4_change=change,
            trend=trend,
            interpretation=interpretation,
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: FinancialStressData) -> Optional[Dict]:
        """Save financial stress data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = asdict(data)
        
        try:
            response = self.supabase.table("financial_stress_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved financial stress for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save financial stress: {e}")
        
        return None
    
    def save_history_to_supabase(self) -> int:
        """
        Fetch and save full STLFSI4 history to Supabase.
        
        Returns number of records saved.
        """
        if not self.supabase or not self.fred_key:
            return 0
        
        # Fetch full history
        params = {
            "series_id": self.config["series_id"],
            "api_key": self.fred_key,
            "file_type": "json",
            "observation_start": "2010-01-01",
        }
        
        try:
            response = requests.get(
                self.config["fred_base_url"],
                params=params,
                timeout=60
            )
            
            if response.status_code != 200:
                return 0
            
            data = response.json()
            observations = data.get("observations", [])
            
            count = 0
            for obs in observations:
                if obs["value"] == ".":
                    continue
                
                stlfsi4 = float(obs["value"])
                stress_score = self.stlfsi4_to_score(stlfsi4)
                regime, multiplier, _ = self.get_regime(stlfsi4)
                
                record = {
                    "date": obs["date"],
                    "stlfsi4": stlfsi4,
                    "stress_score": round(stress_score, 1),
                    "regime": regime,
                    "multiplier": multiplier,
                }
                
                try:
                    self.supabase.table("financial_stress_history") \
                        .upsert(record, on_conflict="date") \
                        .execute()
                    count += 1
                except:
                    pass
            
            logger.info(f"Saved {count} STLFSI4 history records")
            return count
            
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return 0


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run financial stress calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Financial Stress")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--history", action="store_true", help="Save full history")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = FinancialStressCalculator()
    
    if args.history:
        count = calc.save_history_to_supabase()
        print(f"Saved {count} history records")
        return
    
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"FINANCIAL STRESS: {data.stress_score:.1f} ({data.regime})")
    print(f"{'='*60}")
    print(f"STLFSI4: {data.stlfsi4:.3f}")
    print(f"Multiplier: {data.multiplier:.2f}")
    print(f"Trend: {data.trend}", end="")
    if data.stlfsi4_change is not None:
        print(f" ({data.stlfsi4_change:+.3f} vs last week)")
    else:
        print()
    print(f"\nInterpretation: {data.interpretation}")
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
