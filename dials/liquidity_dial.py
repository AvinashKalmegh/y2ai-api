"""
LIQUIDITY DIAL
==============
Measures market's ability to absorb shocks using Vol-of-Vol.
High vol-of-vol = VIX moving erratically = market makers retreating = liquidity evaporating.

Metric:
- Vol-of-Vol: 20-day standard deviation of VIX daily changes

Regime Thresholds:
- Healthy: < 1.0 (stable vol, liquidity intact)
- Caution: 1.0 - 2.0 (some instability)
- Fragile: 2.0 - 3.0 (liquidity stress)
- Crisis: > 3.0 (market makers retreating)

Why Vol-of-Vol matters:
- High vol-of-vol = VIX moving erratically day-to-day
- Market makers can't price risk when vol is unstable
- They widen spreads and reduce size = liquidity evaporates
- This is the transmission mechanism for shocks → cascades
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

LIQUIDITY_CONFIG = {
    "vol_of_vol_lookback": 20,    # Days for vol-of-vol calculation
    
    # Regime thresholds (vol-of-vol)
    "regime_thresholds": {
        "healthy": 1.0,    # < 1.0
        "caution": 2.0,    # 1.0 - 2.0
        "fragile": 3.0,    # 2.0 - 3.0
        # > 3.0 = Crisis
    },
    
    # Additional metrics lookbacks
    "volume_lookback": 20,        # For volume comparison
    "spread_proxy_lookback": 10,  # For spread proxy
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LiquidityData:
    """Liquidity assessment data."""
    date: str
    # Primary metric
    vol_of_vol: float
    regime: str
    # Supporting metrics
    vix_current: float
    vix_20d_avg: float
    vix_daily_change: float
    # Volume metrics (if available)
    volume_ratio: float           # Current vs 20D avg
    volume_regime: str
    # Interpretation
    interpretation: str


# =============================================================================
# LIQUIDITY CALCULATOR
# =============================================================================

class LiquidityDialCalculator:
    """
    Calculate liquidity metrics based on Vol-of-Vol.
    
    Usage:
        calc = LiquidityDialCalculator()
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
        
        self.config = LIQUIDITY_CONFIG
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_vix_history(self, days: int = 30) -> pd.Series:
        """Fetch VIX history."""
        # Try Supabase
        if self.supabase:
            try:
                start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
                
                response = self.supabase.table("vix_history") \
                    .select("date, close") \
                    .gte("date", start_date) \
                    .order("date", desc=True) \
                    .execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date", ascending=False)
                    return df.set_index("date")["close"]
            except Exception as e:
                logger.warning(f"Supabase VIX fetch failed: {e}")
        
        # Fall back to yfinance
        if YFINANCE_AVAILABLE:
            try:
                vix = yf.Ticker("^VIX")
                hist = vix.history(period=f"{days*2}d")
                if len(hist) > 0:
                    return hist["Close"].sort_index(ascending=False)
            except Exception as e:
                logger.warning(f"yfinance VIX fetch failed: {e}")
        
        return pd.Series()
    
    def fetch_spy_volume(self, days: int = 30) -> pd.Series:
        """Fetch SPY volume as market volume proxy."""
        if YFINANCE_AVAILABLE:
            try:
                spy = yf.Ticker("SPY")
                hist = spy.history(period=f"{days*2}d")
                if len(hist) > 0:
                    return hist["Volume"].sort_index(ascending=False)
            except:
                pass
        
        return pd.Series()
    
    # =========================================================================
    # VOL-OF-VOL CALCULATION
    # =========================================================================
    
    def calculate_vol_of_vol(self, vix_series: pd.Series) -> Optional[float]:
        """
        Calculate Vol-of-Vol (standard deviation of VIX daily changes).
        
        Args:
            vix_series: VIX closing values (sorted desc, newest first)
            
        Returns:
            Vol-of-Vol value or None if insufficient data
        """
        lookback = self.config["vol_of_vol_lookback"]
        
        if len(vix_series) < lookback + 1:
            logger.warning(f"Need {lookback + 1} days of VIX data, have {len(vix_series)}")
            return None
        
        # Calculate daily changes for last N days
        # Series is sorted desc, so index 0 is most recent
        daily_changes = []
        for i in range(lookback):
            change = vix_series.iloc[i] - vix_series.iloc[i + 1]
            daily_changes.append(change)
        
        # Calculate standard deviation
        vol_of_vol = np.std(daily_changes)
        
        return vol_of_vol
    
    def get_regime(self, vol_of_vol: float) -> str:
        """Determine regime from vol-of-vol."""
        thresholds = self.config["regime_thresholds"]
        
        if vol_of_vol < thresholds["healthy"]:
            return "Healthy"
        elif vol_of_vol < thresholds["caution"]:
            return "Caution"
        elif vol_of_vol < thresholds["fragile"]:
            return "Fragile"
        else:
            return "Crisis"
    
    def get_interpretation(self, regime: str, vol_of_vol: float) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            "Healthy": "Stable volatility - liquidity intact, market makers comfortable",
            "Caution": "Some vol instability - monitor liquidity conditions",
            "Fragile": "High vol instability - liquidity stress, spreads widening",
            "Crisis": "Extreme vol instability - market makers retreating, liquidity crisis"
        }
        base = interpretations.get(regime, "Unknown regime")
        return f"{base} (Vol-of-Vol: {vol_of_vol:.2f})"
    
    # =========================================================================
    # VOLUME ANALYSIS
    # =========================================================================
    
    def calculate_volume_metrics(self, volume_series: pd.Series) -> Dict:
        """Calculate volume-based liquidity metrics."""
        lookback = self.config["volume_lookback"]
        
        if volume_series.empty or len(volume_series) < lookback:
            return {
                "volume_ratio": 1.0,
                "volume_regime": "Unknown"
            }
        
        current_volume = volume_series.iloc[0]
        avg_volume = volume_series.iloc[:lookback].mean()
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume regime
        if volume_ratio < 0.5:
            volume_regime = "Very Low"
        elif volume_ratio < 0.8:
            volume_regime = "Low"
        elif volume_ratio < 1.2:
            volume_regime = "Normal"
        elif volume_ratio < 1.5:
            volume_regime = "High"
        else:
            volume_regime = "Very High"
        
        return {
            "volume_ratio": volume_ratio,
            "volume_regime": volume_regime
        }
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> LiquidityData:
        """
        Main calculation: compute liquidity metrics.
        
        Returns:
            LiquidityData with all metrics
        """
        logger.info("Calculating liquidity metrics...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch VIX
        vix_series = self.fetch_vix_history()
        
        if vix_series.empty:
            logger.warning("No VIX data available")
            return LiquidityData(
                date=date_str,
                vol_of_vol=0,
                regime="Unknown",
                vix_current=0,
                vix_20d_avg=0,
                vix_daily_change=0,
                volume_ratio=1.0,
                volume_regime="Unknown",
                interpretation="No VIX data available"
            )
        
        # Calculate Vol-of-Vol
        vol_of_vol = self.calculate_vol_of_vol(vix_series)
        
        if vol_of_vol is None:
            vol_of_vol = 0
            regime = "Unknown"
        else:
            regime = self.get_regime(vol_of_vol)
        
        # VIX metrics
        vix_current = vix_series.iloc[0]
        vix_20d_avg = vix_series.iloc[:20].mean() if len(vix_series) >= 20 else vix_series.mean()
        vix_daily_change = vix_series.iloc[0] - vix_series.iloc[1] if len(vix_series) >= 2 else 0
        
        # Volume metrics
        volume_series = self.fetch_spy_volume()
        volume_metrics = self.calculate_volume_metrics(volume_series)
        
        interpretation = self.get_interpretation(regime, vol_of_vol) if vol_of_vol else "Insufficient data"
        
        result = LiquidityData(
            date=date_str,
            vol_of_vol=round(vol_of_vol, 3) if vol_of_vol else 0,
            regime=regime,
            vix_current=round(vix_current, 2),
            vix_20d_avg=round(vix_20d_avg, 2),
            vix_daily_change=round(vix_daily_change, 2),
            volume_ratio=round(volume_metrics["volume_ratio"], 2),
            volume_regime=volume_metrics["volume_regime"],
            interpretation=interpretation
        )
        
        logger.info(f"Vol-of-Vol: {result.vol_of_vol}, Regime: {result.regime}")
        
        return result
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, data: LiquidityData) -> bool:
        """Save liquidity data to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("liquidity_dial_daily") \
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
    
    parser = argparse.ArgumentParser(description="Liquidity Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = LiquidityDialCalculator()
    
    print(f"\n{'='*60}")
    print("LIQUIDITY DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"VOL-OF-VOL: {data.vol_of_vol:.3f}")
    print(f"REGIME: {data.regime}")
    print(f"{'='*40}")
    
    print(f"\nVIX Current: {data.vix_current:.2f}")
    print(f"VIX 20D Avg: {data.vix_20d_avg:.2f}")
    print(f"VIX Daily Change: {data.vix_daily_change:+.2f}")
    
    print(f"\nVolume Ratio: {data.volume_ratio:.2f}x ({data.volume_regime})")
    
    print(f"\n{data.interpretation}")
    
    print(f"\n{'='*40}")
    print("REGIME THRESHOLDS")
    print(f"{'='*40}")
    print("  < 1.0 = Healthy (stable vol, liquidity intact)")
    print("  1.0 - 2.0 = Caution (some instability)")
    print("  2.0 - 3.0 = Fragile (liquidity stress)")
    print("  > 3.0 = Crisis (market makers retreating)")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
