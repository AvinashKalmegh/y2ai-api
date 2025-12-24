"""
SAC (Shock Absorption Capacity) Calculator

Measures how much buffer remains before danger thresholds.
Higher SAC = more capacity to absorb shocks without breaking.

Formula: SAC = AMRI_buffer×0.30 + Bubble_buffer×0.20 + Contagion_buffer×0.25 + 
               Correlations_buffer×0.15 + Breadth_buffer×0.10
"""

import os
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from .config import SAC_WEIGHTS, AMRI_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SACComponents:
    """Individual SAC buffer components (0-100 scale, higher = more buffer)"""
    amri_buffer: float = 100.0      # Distance from AMRI danger zone
    bubble_buffer: float = 100.0    # Distance from bubble extremes
    contagion_buffer: float = 100.0 # Distance from contagion warning
    correlation_buffer: float = 100.0  # Correlation headroom
    breadth_buffer: float = 100.0   # Market breadth reserve


@dataclass
class SACResult:
    """SAC calculation result"""
    composite: float          # 0-100 (higher = more capacity)
    components: SACComponents
    weakest_link: str        # Which buffer is lowest
    risk_level: str          # AMPLE, ADEQUATE, THIN, CRITICAL
    display: str             # Human-readable
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['components'] = asdict(self.components)
        return result


class SACCalculator:
    """
    Calculate Shock Absorption Capacity.
    
    SAC measures remaining buffer - how much room exists before
    hitting danger thresholds.
    """
    
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        if not self.client:
            self._init_client()
    
    def _init_client(self):
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if url and key:
                self.client = create_client(url, key)
        except Exception as e:
            logger.warning(f"Could not initialize Supabase: {e}")
            self.client = None
    
    def _fetch_latest_data(self) -> Dict:
        """Fetch latest data from all relevant tables"""
        data = {"bubble": None, "hypergraph": None, "tracker": None}
        
        if not self.client:
            return data
        
        try:
            r = self.client.table("bubble_index_daily")\
                .select("*").order("date", desc=True).limit(1).execute()
            data["bubble"] = r.data[0] if r.data else None
            
            r = self.client.table("hypergraph_signals")\
                .select("*").order("date", desc=True).limit(1).execute()
            data["hypergraph"] = r.data[0] if r.data else None
            
            r = self.client.table("stock_tracker_daily")\
                .select("*").order("date", desc=True).limit(1).execute()
            data["tracker"] = r.data[0] if r.data else None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
        
        return data
    
    def calculate_amri_buffer(self, amri: float) -> float:
        """
        Calculate buffer from AMRI danger zone (75+).
        At AMRI 0 -> 100% buffer
        At AMRI 75 -> 0% buffer
        """
        danger_threshold = 75
        if amri >= danger_threshold:
            return 0.0
        return ((danger_threshold - amri) / danger_threshold) * 100
    
    def calculate_bubble_buffer(self, bubble: Optional[Dict]) -> float:
        """
        Calculate buffer from bubble regime.
        Uses bifurcation_score and regime, not raw bubble_index.
        
        INFRASTRUCTURE/ADOPTION = high buffer (safe)
        TRANSITION = medium buffer
        BUBBLE_WARNING = low buffer (danger)
        """
        if not bubble:
            return 50.0  # Unknown = assume moderate
        
        regime = bubble.get("regime", "TRANSITION")
        bifurcation = bubble.get("bifurcation_score", 0)
        
        # Map regime to buffer
        if regime == "INFRASTRUCTURE":
            return 90.0 + min(bifurcation * 10, 10)  # 90-100%
        elif regime == "ADOPTION":
            return 70.0 + bifurcation * 20  # 70-90%
        elif regime == "TRANSITION":
            return 40.0 + bifurcation * 30  # 40-70%
        else:  # BUBBLE_WARNING
            return max(0, 20 + bifurcation * 20)  # 0-40%
    
    def calculate_contagion_buffer(self, hypergraph: Optional[Dict]) -> float:
        """
        Calculate buffer from contagion warning (70+).
        """
        if not hypergraph:
            return 50.0
        
        contagion = hypergraph.get("contagion_score", 50)
        danger_threshold = 70
        
        if contagion >= danger_threshold:
            return 0.0
        return ((danger_threshold - contagion) / danger_threshold) * 100
    
    def calculate_correlation_buffer(self, hypergraph: Optional[Dict]) -> float:
        """
        Calculate correlation headroom.
        High cross-pillar ratio = low buffer.
        """
        if not hypergraph:
            return 50.0
        
        cross_pillar = hypergraph.get("cross_pillar_ratio", 0.5)
        danger_threshold = 0.85
        
        if cross_pillar >= danger_threshold:
            return 0.0
        return ((danger_threshold - cross_pillar) / danger_threshold) * 100
    
    def calculate_breadth_buffer(self, tracker: Optional[Dict]) -> float:
        """
        Calculate market breadth reserve.
        Based on pillar participation.
        """
        if not tracker:
            return 50.0
        
        pillars = tracker.get("pillars", [])
        if not pillars:
            return 50.0
        
        # Count pillars with positive performance
        positive_pillars = sum(1 for p in pillars 
                               if isinstance(p, dict) and p.get("change_5day", 0) > 0)
        total_pillars = len(pillars)
        
        if total_pillars == 0:
            return 50.0
        
        # More positive pillars = more buffer
        return (positive_pillars / total_pillars) * 100
    
    def determine_risk_level(self, sac: float) -> str:
        """Classify SAC into risk levels"""
        if sac >= 70:
            return "AMPLE"
        elif sac >= 50:
            return "ADEQUATE"
        elif sac >= 25:
            return "THIN"
        else:
            return "CRITICAL"
    
    def find_weakest_link(self, components: SACComponents) -> str:
        """Find which buffer is lowest"""
        buffers = {
            "AMRI": components.amri_buffer,
            "Bubble": components.bubble_buffer,
            "Contagion": components.contagion_buffer,
            "Correlation": components.correlation_buffer,
            "Breadth": components.breadth_buffer,
        }
        return min(buffers, key=buffers.get)
    
    def calculate(self, current_amri: float = 50.0) -> SACResult:
        """Calculate SAC from current data"""
        data = self._fetch_latest_data()
        
        components = SACComponents(
            amri_buffer=self.calculate_amri_buffer(current_amri),
            bubble_buffer=self.calculate_bubble_buffer(data["bubble"]),
            contagion_buffer=self.calculate_contagion_buffer(data["hypergraph"]),
            correlation_buffer=self.calculate_correlation_buffer(data["hypergraph"]),
            breadth_buffer=self.calculate_breadth_buffer(data["tracker"]),
        )
        
        # Weighted composite
        sac = (
            components.amri_buffer * SAC_WEIGHTS["AMRI_buffer"] +
            components.bubble_buffer * SAC_WEIGHTS["Bubble_buffer"] +
            components.contagion_buffer * SAC_WEIGHTS["Contagion_buffer"] +
            components.correlation_buffer * SAC_WEIGHTS["Correlations_buffer"] +
            components.breadth_buffer * SAC_WEIGHTS["Breadth_buffer"]
        )
        
        weakest = self.find_weakest_link(components)
        risk_level = self.determine_risk_level(sac)
        
        # Display format
        if risk_level == "CRITICAL":
            display = f"🔴 {sac:.0f}% ({weakest})"
        elif risk_level == "THIN":
            display = f"🟠 {sac:.0f}% ({weakest})"
        elif risk_level == "ADEQUATE":
            display = f"🟡 {sac:.0f}%"
        else:
            display = f"🟢 {sac:.0f}%"
        
        return SACResult(
            composite=round(sac, 1),
            components=components,
            weakest_link=weakest,
            risk_level=risk_level,
            display=display,
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    calculator = SACCalculator()
    result = calculator.calculate(current_amri=55)
    
    print(f"\n{'='*60}")
    print("SAC CALCULATION")
    print(f"{'='*60}")
    print(f"SAC Composite: {result.composite}%")
    print(f"Risk Level: {result.risk_level}")
    print(f"Weakest Link: {result.weakest_link}")
    print(f"Display: {result.display}")
    print(f"\nComponents:")
    print(f"  AMRI Buffer: {result.components.amri_buffer:.1f}%")
    print(f"  Bubble Buffer: {result.components.bubble_buffer:.1f}%")
    print(f"  Contagion Buffer: {result.components.contagion_buffer:.1f}%")
    print(f"  Correlation Buffer: {result.components.correlation_buffer:.1f}%")
    print(f"  Breadth Buffer: {result.components.breadth_buffer:.1f}%")
