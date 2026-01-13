"""
AMRI (ARGUS Master Regime Index) Calculator - CORRECTED VERSION
================================================================

FORMULA FROM GOOGLE SHEETS (AMRI_MASTER):

Core AMRI = CRS×0.25 + CCS×0.25 + SRS×0.25 + SDS×0.25

Components:
  - CRS: Correlation Regime Score (avg 20D correlation → 0-100)
  - CCS: Cluster Count Score (cluster count → 0-100)
  - SRS: Spread Regime Score (HY spread level → 0-100)
  - SDS: Sector Divergence Score (pillar divergence → 0-100)

Enhanced AMRI = 0.80 × Core + 0.20 × Bubble_Overlay

CORRECTIONS APPLIED:
1. Removed VIX as separate component (it's embedded in SRS)
2. All 4 components have equal 25% weights
3. Scoring functions match Google Sheets logic
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - CORRECTED
# =============================================================================

AMRI_WEIGHTS = {
    "CRS": 0.25,  # Correlation Regime Score
    "CCS": 0.25,  # Cluster Count Score  
    "SRS": 0.25,  # Spread Regime Score
    "SDS": 0.25,  # Structural Divergence Score
}

ENHANCED_WEIGHTS = {
    "CORE": 0.80,
    "OVERLAY": 0.20,
}

AMRI_THRESHOLDS = {
    "Stable": (0, 25),
    "Normal": (25, 40),
    "Elevated": (40, 55),
    "Tension": (55, 70),
    "Fragile": (70, 85),
    "Break": (85, 100),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AMRIComponents:
    """Individual AMRI component scores (0-100 scale)"""
    crs: float = 0.0   # Correlation Regime Score
    ccs: float = 0.0   # Cluster Count Score
    srs: float = 0.0   # Spread Regime Score
    sds: float = 0.0   # Structural Divergence Score
    
    # Status labels
    crs_status: str = "Unknown"
    ccs_status: str = "Unknown"
    srs_status: str = "Unknown"
    sds_status: str = "Unknown"


@dataclass
class AMRIResult:
    """Complete AMRI calculation result"""
    date: str
    core_amri: float
    enhanced_amri: float
    regime: str
    interpretation: str
    components: AMRIComponents
    bubble_overlay: float
    break_probability: float
    dominant_driver: str
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        return result


# =============================================================================
# AMRI CALCULATOR - CORRECTED
# =============================================================================

class AMRICalculator:
    """
    Calculate AMRI using the corrected 4-component formula.
    
    Usage:
        calc = AMRICalculator()
        result = calc.calculate()
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.client = create_client(self.supabase_url, self.supabase_key)
    
    def _get_latest(self, table: str) -> Optional[Dict]:
        """Get latest row from a table."""
        if not self.client:
            return None
        try:
            result = self.client.table(table) \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.warning(f"Failed to get {table}: {e}")
            return None
    
    # =========================================================================
    # COMPONENT CALCULATIONS - CORRECTED
    # =========================================================================
    
    def calculate_crs(self, correlation_data: Optional[Dict] = None) -> Tuple[float, str]:
        """
        Calculate Correlation Regime Score (CRS).
        
        Google Sheets logic:
        - Uses 20D average correlation
        - 0.30 correlation → 0 (healthy)
        - 0.60 correlation → 100 (critical)
        
        Returns: (score 0-100, status)
        """
        if correlation_data is None:
            correlation_data = self._get_latest("correlation_daily")
        
        if not correlation_data:
            correlation_data = self._get_latest("cluster_dial_daily")
        
        avg_corr = 0.42  # Default from Google Sheets
        if correlation_data:
            avg_corr = correlation_data.get("avg_correlation_20d") or \
                       correlation_data.get("avg_correlation") or 0.42
        
        # Scale: 0.30 → 0, 0.60 → 100
        if avg_corr <= 0.30:
            score = 0
        elif avg_corr >= 0.60:
            score = 100
        else:
            score = (avg_corr - 0.30) / (0.60 - 0.30) * 100
        
        # Determine status
        if score < 30:
            status = "Healthy"
        elif score < 50:
            status = "Caution"
        elif score < 70:
            status = "Stressed"
        else:
            status = "Critical"
        
        return round(score, 1), status
    
    def calculate_ccs(self, cluster_data: Optional[Dict] = None) -> Tuple[float, str]:
        """
        Calculate Cluster Count Score (CCS).
        
        Google Sheets logic:
        - Uses number of correlation clusters
        - 15+ clusters → 0 (healthy diversity)
        - 3 clusters → 100 (critical concentration)
        
        Formula from sheets: CCS = (15 - clusters) / 12 * 100, clamped 0-100
        
        Returns: (score 0-100, status)
        """
        if cluster_data is None:
            cluster_data = self._get_latest("cluster_dial_daily")
        
        clusters = 8  # Default from Google Sheets
        if cluster_data:
            clusters = cluster_data.get("cluster_count") or 8
        
        # Scale: 15 clusters → 0, 3 clusters → 100
        if clusters >= 15:
            score = 0
        elif clusters <= 3:
            score = 100
        else:
            score = (15 - clusters) / 12 * 100
        
        # Determine status
        if clusters >= 10:
            status = "Healthy"
        elif clusters >= 7:
            status = "Caution"
        elif clusters >= 5:
            status = "Stressed"
        else:
            status = "Critical"
        
        return round(score, 1), status
    
    def calculate_srs(self, spread_data: Optional[Dict] = None) -> Tuple[float, str]:
        """
        Calculate Spread Regime Score (SRS).
        
        Google Sheets logic:
        - Uses HY spread level
        - HY < 3% → ~0 (benign)
        - HY > 5% → ~100 (stressed)
        
        Returns: (score 0-100, status)
        """
        if spread_data is None:
            spread_data = self._get_latest("credit_spread_daily")
        
        if not spread_data:
            spread_data = self._get_latest("vix_daily")
        
        hy_spread = 2.81  # Default from Google Sheets (as percentage)
        if spread_data:
            hy_spread = spread_data.get("hy_spread") or 2.81
        
        # Scale: 3% → 0, 5% → 100
        if hy_spread <= 3.0:
            score = max(0, (hy_spread - 2.0) / 1.0 * 20)  # 2-3% gives 0-20
        elif hy_spread >= 5.0:
            score = 100
        else:
            score = 20 + (hy_spread - 3.0) / 2.0 * 80  # 3-5% gives 20-100
        
        # Determine status
        if hy_spread < 3.0:
            status = "Benign"
        elif hy_spread < 4.0:
            status = "Normal"
        elif hy_spread < 5.0:
            status = "Elevated"
        else:
            status = "Stressed"
        
        return round(score, 1), status
    
    def calculate_sds(self, breadth_data: Optional[Dict] = None) -> Tuple[float, str]:
        """
        Calculate Structural Divergence Score (SDS).
        
        Google Sheets logic (Fragility Model Condition 2):
        - Infra breadth > 55% AND Enterprise breadth < 35% → ACTIVE (100)
        - Otherwise scale based on divergence
        
        Current values: Infra 87.5%, Ent 7.7% → SDS = 100 (Critical)
        
        Returns: (score 0-100, status)
        """
        if breadth_data is None:
            breadth_data = self._get_latest("breadth_daily")
        
        # Defaults from Google Sheets
        infra_breadth = 0.875  # 87.5%
        ent_breadth = 0.077    # 7.7%
        
        if breadth_data:
            infra_breadth = breadth_data.get("infra_breadth") or 0.875
            ent_breadth = breadth_data.get("enterprise_breadth") or 0.077
        
        # Fragility Model Condition 2 check
        if infra_breadth > 0.55 and ent_breadth < 0.35:
            score = 100
            status = "Critical (ACTIVE)"
        else:
            # Calculate divergence
            divergence = abs(infra_breadth - ent_breadth)
            
            # Scale: 0.30 divergence → 0, 0.80 divergence → 100
            if divergence <= 0.30:
                score = 0
            elif divergence >= 0.80:
                score = 100
            else:
                score = (divergence - 0.30) / 0.50 * 100
            
            if divergence > 0.60:
                status = "Extreme"
            elif divergence > 0.40:
                status = "Elevated"
            else:
                status = "Normal"
        
        return round(score, 1), status
    
    def calculate_bubble_overlay(self) -> float:
        """
        Get bubble overlay from LPPLS, PSY, LZC indicators.
        Default: 5.23 from Google Sheets (Clear status)
        """
        # Try to get from bubble overlay dial
        diag_data = self._get_latest("bubble_overlay_daily")
        
        if diag_data:
            lppls = diag_data.get("lppls_score", 15.7) or 15.7
            psy = diag_data.get("psy_score", 0) or 0
            lzc = diag_data.get("lzc_score", 0) or 0
            return (lppls + psy + lzc) / 3
        
        return 5.23  # Default from Google Sheets
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> AMRIResult:
        """
        Calculate AMRI using corrected 4-component formula.
        
        Core AMRI = CRS×0.25 + CCS×0.25 + SRS×0.25 + SDS×0.25
        Enhanced = 0.80 × Core + 0.20 × Bubble_Overlay
        """
        logger.info("Calculating AMRI (corrected formula)...")
        
        # Calculate all components
        crs_score, crs_status = self.calculate_crs()
        ccs_score, ccs_status = self.calculate_ccs()
        srs_score, srs_status = self.calculate_srs()
        sds_score, sds_status = self.calculate_sds()
        
        components = AMRIComponents(
            crs=crs_score,
            ccs=ccs_score,
            srs=srs_score,
            sds=sds_score,
            crs_status=crs_status,
            ccs_status=ccs_status,
            srs_status=srs_status,
            sds_status=sds_status,
        )
        
        # Core AMRI = weighted sum
        core_amri = (
            crs_score * AMRI_WEIGHTS["CRS"] +
            ccs_score * AMRI_WEIGHTS["CCS"] +
            srs_score * AMRI_WEIGHTS["SRS"] +
            sds_score * AMRI_WEIGHTS["SDS"]
        )
        
        # Bubble overlay
        bubble_overlay = self.calculate_bubble_overlay()
        
        # Enhanced AMRI
        enhanced_amri = (
            ENHANCED_WEIGHTS["CORE"] * core_amri +
            ENHANCED_WEIGHTS["OVERLAY"] * bubble_overlay
        )
        
        # Determine regime
        regime = self._get_regime(core_amri)
        interpretation = self._get_interpretation(regime)
        
        # Determine dominant driver (highest weighted contribution)
        contributions = {
            "CRS": crs_score * AMRI_WEIGHTS["CRS"],
            "CCS": ccs_score * AMRI_WEIGHTS["CCS"],
            "SRS": srs_score * AMRI_WEIGHTS["SRS"],
            "SDS": sds_score * AMRI_WEIGHTS["SDS"],
        }
        dominant_driver = max(contributions, key=contributions.get)
        
        # Break probability
        break_prob = self._calculate_break_probability(core_amri)
        
        logger.info(f"Core AMRI: {core_amri:.1f} ({regime})")
        logger.info(f"Enhanced AMRI: {enhanced_amri:.1f}")
        logger.info(f"  CRS: {crs_score:.1f} × 0.25 = {crs_score * 0.25:.2f} ({crs_status})")
        logger.info(f"  CCS: {ccs_score:.1f} × 0.25 = {ccs_score * 0.25:.2f} ({ccs_status})")
        logger.info(f"  SRS: {srs_score:.1f} × 0.25 = {srs_score * 0.25:.2f} ({srs_status})")
        logger.info(f"  SDS: {sds_score:.1f} × 0.25 = {sds_score * 0.25:.2f} ({sds_status})")
        logger.info(f"Dominant: {dominant_driver}")
        
        return AMRIResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            core_amri=round(core_amri, 1),
            enhanced_amri=round(enhanced_amri, 1),
            regime=regime,
            interpretation=interpretation,
            components=components,
            bubble_overlay=round(bubble_overlay, 1),
            break_probability=round(break_prob, 0),
            dominant_driver=dominant_driver,
        )
    
    def _get_regime(self, score: float) -> str:
        """Determine AMRI regime from score."""
        for regime, (low, high) in AMRI_THRESHOLDS.items():
            if low <= score < high:
                return regime
        return "Break" if score >= 85 else "Normal"
    
    def _get_interpretation(self, regime: str) -> str:
        """Get interpretation for regime."""
        interpretations = {
            "Stable": "Normal conditions - maintain positions",
            "Normal": "Low stress - standard operations",
            "Elevated": "Elevated stress - monitor closely",
            "Tension": "Tension - reduce exposure, 43% of divergence events occur here",
            "Fragile": "Fragile - prioritize capital preservation",
            "Break": "Critical - de-risk fully"
        }
        return interpretations.get(regime, "Unknown")
    
    def _calculate_break_probability(self, amri: float) -> float:
        """Calculate break probability from AMRI."""
        if amri < 40:
            return 5 + (amri / 40) * 10
        elif amri < 55:
            return 15 + ((amri - 40) / 15) * 15
        elif amri < 70:
            return 30 + ((amri - 55) / 15) * 25
        elif amri < 85:
            return 55 + ((amri - 70) / 15) * 25
        else:
            return min(95, 80 + ((amri - 85) / 15) * 15)
    
    def save_to_supabase(self, result) -> None:
        """Save AMRI to Supabase (schema-compatible)."""
        import os
        import json
        from datetime import datetime
        from supabase import create_client
        
        client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
        
        row = {
            'date': result.date,
            'amri_score': result.core_amri,
            'regime': result.regime,
            'volatility_component': result.components.sds,
            'correlation_component': result.components.crs,
            'breadth_component': result.components.srs,
            'momentum_component': result.components.ccs,
            'components': json.dumps({
                'crs': result.components.crs,
                'ccs': result.components.ccs,
                'srs': result.components.srs,
                'sds': result.components.sds,
            }),
            'interpretation': result.interpretation,
            'created_at': datetime.now().isoformat(),
        }
        
        client.table('amri_daily').upsert(row, on_conflict='date').execute()
        logger.info(f"Saved AMRI {result.core_amri} to Supabase")
# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate AMRI (Corrected)")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = AMRICalculator()
    result = calc.calculate()
    
    print(f"\n{'='*60}")
    print("AMRI CALCULATION (CORRECTED 4-COMPONENT FORMULA)")
    print(f"{'='*60}")
    print(f"\nDate: {result.date}")
    print(f"\n{'─'*40}")
    print(f"CORE AMRI: {result.core_amri:.1f}")
    print(f"ENHANCED AMRI: {result.enhanced_amri:.1f}")
    print(f"REGIME: {result.regime}")
    print(f"BREAK PROBABILITY: {result.break_probability:.0f}%")
    print(f"{'─'*40}")
    print(f"\n{result.interpretation}")
    
    print(f"\n{'─'*40}")
    print("COMPONENTS (25% each)")
    print(f"{'─'*40}")
    print(f"  CRS (Correlation):  {result.components.crs:5.1f} × 0.25 = {result.components.crs * 0.25:5.2f}  ({result.components.crs_status})")
    print(f"  CCS (Clusters):     {result.components.ccs:5.1f} × 0.25 = {result.components.ccs * 0.25:5.2f}  ({result.components.ccs_status})")
    print(f"  SRS (Spreads):      {result.components.srs:5.1f} × 0.25 = {result.components.srs * 0.25:5.2f}  ({result.components.srs_status})")
    print(f"  SDS (Divergence):   {result.components.sds:5.1f} × 0.25 = {result.components.sds * 0.25:5.2f}  ({result.components.sds_status})")
    print(f"\n  Bubble Overlay: {result.bubble_overlay:.1f}")
    print(f"  Dominant Driver: {result.dominant_driver}")
    
    if args.save:
        if calc.save_to_supabase(result):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()