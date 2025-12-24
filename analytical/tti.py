"""
TTI (Time-to-Instability) Calculator

Projects how many days until AMRI reaches critical thresholds
based on current trajectory (linear regression of recent values).

Formula: days_to_threshold = (threshold - current) / rate_of_change
"""

import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import TTI_THRESHOLDS, TTI_LOOKBACK_DAYS, AMRI_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TTIResult:
    """Time-to-Instability calculation result"""
    days_to_tension: Optional[int]     # Days until AMRI hits 60
    days_to_fragile: Optional[int]     # Days until AMRI hits 75
    days_to_break: Optional[int]       # Days until AMRI hits 90
    rate_of_change: float              # Daily AMRI change rate
    trajectory: str                    # ACCELERATING, STABLE, DECELERATING
    binding_threshold: str             # Which threshold is closest
    display: str                       # Human-readable display
    confidence: str                    # HIGH, MEDIUM, LOW
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def format_display(self) -> str:
        """Format for dashboard display"""
        if self.days_to_break and self.days_to_break <= 7:
            return f"⚠️ {self.days_to_break}D to BREAK"
        elif self.days_to_fragile and self.days_to_fragile <= 14:
            return f"🔸 {self.days_to_fragile}D to FRAGILE"
        elif self.days_to_tension and self.days_to_tension <= 30:
            return f"🔹 {self.days_to_tension}D to TENSION"
        elif self.rate_of_change > 0:
            return f"↗️ +{self.rate_of_change:.1f}/day"
        elif self.rate_of_change < 0:
            return f"↘️ {self.rate_of_change:.1f}/day"
        else:
            return "→ Stable"


class TTICalculator:
    """
    Calculate Time-to-Instability projections.
    
    Uses 10-day linear regression of AMRI values to project
    when thresholds will be crossed.
    """
    
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        if not self.client:
            self._init_client()
    
    def _init_client(self):
        """Initialize Supabase client"""
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if url and key:
                self.client = create_client(url, key)
        except Exception as e:
            logger.warning(f"Could not initialize Supabase: {e}")
            self.client = None
    
    def _fetch_amri_history(self, days: int = 10) -> List[Tuple[str, float]]:
        """
        Fetch recent AMRI values from argus_master_signals table.
        Falls back to calculating from components if table doesn't exist.
        """
        if not self.client:
            return []
        
        try:
            # Try the master signals table first
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            result = self.client.table("argus_master_signals")\
                .select("date, amri_composite")\
                .gte("date", start_date)\
                .lte("date", end_date)\
                .order("date", desc=False)\
                .execute()
            
            if result.data:
                return [(r["date"], r["amri_composite"]) for r in result.data]
        except Exception as e:
            logger.debug(f"argus_master_signals not available: {e}")
        
        # Fallback: use bubble_index bifurcation as proxy
        try:
            result = self.client.table("bubble_index_daily")\
                .select("date, bubble_index")\
                .order("date", desc=True)\
                .limit(days)\
                .execute()
            
            if result.data:
                # Bubble index 0-100 can proxy AMRI
                return [(r["date"], r["bubble_index"]) for r in reversed(result.data)]
        except Exception as e:
            logger.error(f"Error fetching bubble history: {e}")
        
        return []
    
    def calculate_rate_of_change(self, history: List[Tuple[str, float]]) -> Tuple[float, float]:
        """
        Calculate rate of change using linear regression.
        
        Returns: (slope per day, R² confidence)
        """
        if len(history) < 3:
            return 0.0, 0.0
        
        # Convert to numpy arrays
        x = np.arange(len(history))
        y = np.array([h[1] for h in history])
        
        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        # Slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # R² for confidence
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        y_pred = slope * x + (sum_y - slope * sum_x) / n
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return float(slope), float(r_squared)
    
    def project_days_to_threshold(self, current: float, threshold: float, rate: float) -> Optional[int]:
        """
        Project days until threshold is reached.
        
        Returns None if:
        - Already past threshold
        - Moving away from threshold
        - Rate is too slow (> 365 days)
        """
        if current >= threshold:
            return 0  # Already there
        
        if rate <= 0:
            return None  # Not approaching
        
        days = (threshold - current) / rate
        
        if days > 365:
            return None  # Too far out to be meaningful
        
        return max(1, int(days))
    
    def determine_trajectory(self, rate: float) -> str:
        """Classify trajectory based on rate of change"""
        if rate > 1.5:
            return "ACCELERATING"
        elif rate > 0.5:
            return "RISING"
        elif rate > -0.5:
            return "STABLE"
        elif rate > -1.5:
            return "FALLING"
        else:
            return "DECELERATING"
    
    def calculate(self, current_amri: float = None) -> TTIResult:
        """
        Calculate TTI projections.
        
        Args:
            current_amri: Current AMRI value. If None, fetches from history.
        """
        history = self._fetch_amri_history(TTI_LOOKBACK_DAYS)
        
        if not history:
            return TTIResult(
                days_to_tension=None,
                days_to_fragile=None,
                days_to_break=None,
                rate_of_change=0.0,
                trajectory="UNKNOWN",
                binding_threshold="NONE",
                display="No data",
                confidence="LOW",
            )
        
        # Use provided AMRI or latest from history
        if current_amri is None:
            current_amri = history[-1][1]
        
        # Calculate rate
        rate, r_squared = self.calculate_rate_of_change(history)
        
        # Project to thresholds
        days_tension = self.project_days_to_threshold(current_amri, 60, rate)
        days_fragile = self.project_days_to_threshold(current_amri, 75, rate)
        days_break = self.project_days_to_threshold(current_amri, 90, rate)
        
        # Determine binding (closest) threshold
        if days_break and days_break <= 7:
            binding = "BREAK"
        elif days_fragile and days_fragile <= 14:
            binding = "FRAGILE"
        elif days_tension and days_tension <= 30:
            binding = "TENSION"
        else:
            binding = "NONE"
        
        # Trajectory
        trajectory = self.determine_trajectory(rate)
        
        # Confidence from R²
        if r_squared > 0.7:
            confidence = "HIGH"
        elif r_squared > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        result = TTIResult(
            days_to_tension=days_tension,
            days_to_fragile=days_fragile,
            days_to_break=days_break,
            rate_of_change=round(rate, 2),
            trajectory=trajectory,
            binding_threshold=binding,
            display="",
            confidence=confidence,
        )
        
        result.display = result.format_display()
        
        return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    calculator = TTICalculator()
    result = calculator.calculate()
    
    print(f"\n{'='*60}")
    print("TTI CALCULATION")
    print(f"{'='*60}")
    print(f"Display: {result.display}")
    print(f"Rate of Change: {result.rate_of_change}/day")
    print(f"Trajectory: {result.trajectory}")
    print(f"Binding: {result.binding_threshold}")
    print(f"Confidence: {result.confidence}")
    print(f"\nProjections:")
    print(f"  Days to TENSION (60): {result.days_to_tension}")
    print(f"  Days to FRAGILE (75): {result.days_to_fragile}")
    print(f"  Days to BREAK (90): {result.days_to_break}")
