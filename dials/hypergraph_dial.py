"""
HYPERGRAPH DIAL
===============
Wrapper for hypergraph contagion analysis.

Fetches and displays hypergraph metrics from Supabase.
The actual calculation is done by hypergraph/ module separately.

Metrics:
- Contagion Score (0-100): Cross-pillar risk level
- Stability Score (0-1): Structure stability
- Cross-Pillar Ratio (0-1): Diversification health
- Hyperedge Count: Number of correlated groups
- Bridge Stocks: Stocks connecting multiple pillars

Regimes:
- STABLE: Contagion < 40, Stability > 0.3
- ACCELERATING: Contagion 40-60
- FRAGMENTING: Stability < 0.15
- CONTAGION: Contagion > 70

VETO Conditions:
- Contagion > 75
- Regime = CONTAGION with Stability < 0.15
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

HYPERGRAPH_CONFIG = {
    "regime_thresholds": {
        "contagion_low": 40,
        "contagion_high": 70,
        "stability_critical": 0.15,
        "stability_low": 0.30,
        "cross_pillar_warning": 0.60,
        "cross_pillar_danger": 0.80,
    },
    "veto_conditions": {
        "contagion_threshold": 75,
        "stability_threshold": 0.15,
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HypergraphData:
    """Hypergraph metrics data."""
    date: str
    # Core metrics
    contagion_score: float
    stability_score: float
    cross_pillar_ratio: float
    # Structure metrics
    hyperedge_count: int
    avg_hyperedge_size: float
    max_hyperedge_size: int
    growth_rate_1d: float
    # Analysis
    regime: str
    veto: bool
    bridge_stocks: List[str]
    largest_hyperedge: List[str]
    # Interpretation
    contagion_interp: str
    stability_interp: str
    cross_pillar_interp: str


# =============================================================================
# HYPERGRAPH DIAL CALCULATOR
# =============================================================================

class HypergraphDialCalculator:
    """
    Fetch and analyze hypergraph metrics.
    
    Note: Actual hypergraph calculation is done by hypergraph/ module.
    This dial reads results from Supabase and provides analysis.
    
    Usage:
        calc = HypergraphDialCalculator()
        data = calc.calculate()
        # Data is already in Supabase from hypergraph pipeline
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.config = HYPERGRAPH_CONFIG
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_latest_metrics(self) -> Optional[Dict]:
        """Fetch latest hypergraph metrics from Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        try:
            response = self.supabase.table("hypergraph_signals") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0]
            
        except Exception as e:
            logger.error(f"Failed to fetch hypergraph metrics: {e}")
        
        return None
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def check_veto(self, contagion: float, stability: float, regime: str) -> bool:
        """Check if hypergraph signals VETO (high risk)."""
        veto_config = self.config["veto_conditions"]
        
        if contagion > veto_config["contagion_threshold"]:
            return True
        
        if regime == "CONTAGION" and stability < veto_config["stability_threshold"]:
            return True
        
        return False
    
    def get_contagion_interpretation(self, contagion: float) -> str:
        """Get interpretation for contagion score."""
        if contagion > 70:
            return "Cross-pillar risk HIGH - Diversification failing"
        elif contagion > 50:
            return "Cross-pillar risk elevated - Monitor closely"
        elif contagion > 30:
            return "Cross-pillar risk moderate"
        else:
            return "Cross-pillar risk normal - Healthy diversification"
    
    def get_stability_interpretation(self, stability: float) -> str:
        """Get interpretation for stability score."""
        if stability < 0.15:
            return "Structure collapsing - Maximum caution"
        elif stability < 0.30:
            return "Structure unstable - Reduce exposure"
        elif stability < 0.50:
            return "Structure weakening"
        else:
            return "Structure stable"
    
    def get_cross_pillar_interpretation(self, cross_pillar: float) -> str:
        """Get interpretation for cross-pillar ratio."""
        if cross_pillar > 0.80:
            return "Diversification failing - All pillars moving together"
        elif cross_pillar > 0.60:
            return "Diversification weakening"
        else:
            return "Diversification healthy"
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> HypergraphData:
        """
        Fetch and analyze hypergraph metrics.
        
        Returns:
            HypergraphData with analysis
        """
        logger.info("Fetching hypergraph metrics...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch from Supabase
        metrics = self.fetch_latest_metrics()
        
        if not metrics:
            logger.warning("No hypergraph metrics found")
            return HypergraphData(
                date=date_str,
                contagion_score=50.0,
                stability_score=0.50,
                cross_pillar_ratio=0.50,
                hyperedge_count=0,
                avg_hyperedge_size=0.0,
                max_hyperedge_size=0,
                growth_rate_1d=0.0,
                regime="UNKNOWN",
                veto=False,
                bridge_stocks=[],
                largest_hyperedge=[],
                contagion_interp="No data available",
                stability_interp="No data available",
                cross_pillar_interp="No data available",
            )
        
        # Extract values
        contagion = float(metrics.get("contagion_score", 50))
        stability = float(metrics.get("stability_score", 0.5))
        cross_pillar = float(metrics.get("cross_pillar_ratio", 0.5))
        regime = metrics.get("regime", "UNKNOWN")
        
        # Check veto
        veto = self.check_veto(contagion, stability, regime)
        
        # Get interpretations
        contagion_interp = self.get_contagion_interpretation(contagion)
        stability_interp = self.get_stability_interpretation(stability)
        cross_pillar_interp = self.get_cross_pillar_interpretation(cross_pillar)
        
        logger.info(f"Hypergraph: Contagion={contagion:.1f}, Regime={regime}, Veto={veto}")
        
        return HypergraphData(
            date=metrics.get("date", date_str),
            contagion_score=contagion,
            stability_score=stability,
            cross_pillar_ratio=cross_pillar,
            hyperedge_count=int(metrics.get("hyperedge_count", 0)),
            avg_hyperedge_size=float(metrics.get("avg_hyperedge_size", 0)),
            max_hyperedge_size=int(metrics.get("max_hyperedge_size", 0)),
            growth_rate_1d=float(metrics.get("growth_rate_1d", 0)),
            regime=regime,
            veto=veto,
            bridge_stocks=metrics.get("bridge_stocks", []) or [],
            largest_hyperedge=metrics.get("largest_hyperedge_tickers", []) or [],
            contagion_interp=contagion_interp,
            stability_interp=stability_interp,
            cross_pillar_interp=cross_pillar_interp,
        )
    
    # =========================================================================
    # HELPER FOR OTHER MODULES
    # =========================================================================
    
    def get_regime(self) -> Dict:
        """Get hypergraph regime for use by other modules."""
        data = self.calculate()
        return {
            "regime": data.regime,
            "contagion": data.contagion_score,
            "stability": data.stability_score,
            "cross_pillar": data.cross_pillar_ratio,
            "veto": data.veto
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run hypergraph dial."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hypergraph Dial")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = HypergraphDialCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"HYPERGRAPH SIGNALS - Contagion Layer")
    print(f"{'='*60}")
    print(f"Date: {data.date}")
    print(f"Regime: {data.regime}")
    print(f"VETO: {'YES' if data.veto else 'NO'}")
    print(f"\nCore Metrics:")
    print(f"  Contagion Score: {data.contagion_score:.1f}")
    print(f"    {data.contagion_interp}")
    print(f"  Stability: {data.stability_score*100:.1f}%")
    print(f"    {data.stability_interp}")
    print(f"  Cross-Pillar: {data.cross_pillar_ratio*100:.1f}%")
    print(f"    {data.cross_pillar_interp}")
    print(f"\nStructure:")
    print(f"  Hyperedge Count: {data.hyperedge_count}")
    print(f"  Avg Size: {data.avg_hyperedge_size:.2f}")
    print(f"  Max Size: {data.max_hyperedge_size}")
    print(f"  Growth Rate (1D): {data.growth_rate_1d*100:.1f}%")
    
    if data.bridge_stocks:
        print(f"\nBridge Stocks: {', '.join(data.bridge_stocks[:5])}")
    
    if data.largest_hyperedge:
        print(f"Largest Hyperedge: {', '.join(data.largest_hyperedge[:5])}")
    
    print(f"{'='*60}\n")
    
    return data


if __name__ == "__main__":
    main()
