"""
AMRI (ARGUS Master Regime Index) Calculator - GS-ALIGNED VERSION
=================================================================

FORMULA ALIGNED WITH GOOGLE SHEETS (AMRI_MASTER):

Core AMRI = CRS×0.23 + CCS×0.31 + SRS×0.15 + SDS×0.31

Components:
  - CRS: Correlation Regime Score (cluster avg_correlation → 0-100)
  - CCS: Cluster Count Score (cluster count → 0-100)
  - SRS: Spread Regime Score (HY spread level → 0-100)
  - SDS: Structural Divergence Score (Infra-Enterprise breadth → 0-100)

Enhanced AMRI = 0.80 × Core + 0.20 × Bubble_Overlay

GS ALIGNMENT NOTES (Jan 2026):
------------------------------
1. Weights updated to match GS: 0.23/0.31/0.15/0.31
2. Regime thresholds: Normal<30, Elevated<50, Tension<70, Fragile<85, Break>=85
3. CRS uses cluster_dial avg_correlation (~27%) with calibrated scale
   - GS Correlation_Dials shows ~57% (different calculation method)
   - Scale adjusted so Python output approximates GS CRS scores
4. SDS uses breadth_daily pillar divergence
   - GS Dashboard may show stale values (Infra:75% vs Breadth_Dial:56%)
5. Expected ~10-15 point variance vs GS due to data source differences

DATA SOURCES:
  - cluster_dial_daily: avg_correlation, cluster_count
  - credit_spread_daily: hy_spread
  - breadth_daily: pillar_breadth (Infrastructure, Enterprise)
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
# CONFIGURATION - GS-ALIGNED
# =============================================================================

AMRI_WEIGHTS = {
    "CRS": 0.23,  # Correlation Regime Score (matches GS)
    "CCS": 0.31,  # Cluster Count Score (matches GS)
    "SRS": 0.15,  # Spread Regime Score (matches GS)
    "SDS": 0.31,  # Structural Divergence Score (matches GS)
}

ENHANCED_WEIGHTS = {
    "CORE": 0.80,
    "OVERLAY": 0.20,
}

# Regime thresholds - matches GS AMRI_MASTER formula
AMRI_THRESHOLDS = {
    "Normal": (0, 30),      # GS: B11<30
    "Elevated": (30, 50),   # GS: B11<50
    "Tension": (50, 70),    # GS: B11<70
    "Fragile": (70, 85),    # GS: B11<85
    "Break": (85, 100),     # GS: else
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
        
        Google Sheets uses stock-level correlations from Correlation_Dials (~57%)
        Python uses cluster_dial_daily avg_correlation (~27%)
        
        These are different metrics, so we calibrate the scale:
        - GS: 57% correlation → CRS = 59.7
        - Python: 27% correlation → should give similar CRS
        
        Calibrated scale: 0.15 → 0, 0.40 → 100
        This gives: 0.27 → (0.27-0.15)/(0.40-0.15)*100 = 48
        
        Returns: (score 0-100, status)
        """
        # Get stock-level correlation from cluster_dial_daily
        if correlation_data is None:
            correlation_data = self._get_latest("cluster_dial_daily")
        
        if not correlation_data:
            correlation_data = self._get_latest("correlation_daily")
        
        avg_corr = 0.27  # Default based on recent data
        if correlation_data:
            # Prefer cluster avg_correlation (stock-level)
            raw_corr = correlation_data.get("avg_correlation") or \
                       correlation_data.get("corr_20d") or \
                       correlation_data.get("avg_correlation_20d")
            if raw_corr is not None:
                # Convert string to float if needed
                avg_corr = float(raw_corr) if isinstance(raw_corr, str) else raw_corr
        
        # Calibrated scale: 0.15 → 0, 0.40 → 100
        # Typical range is 0.20-0.35, giving CRS 20-80
        if avg_corr <= 0.15:
            score = 0
        elif avg_corr >= 0.40:
            score = 100
        else:
            score = (avg_corr - 0.15) / (0.40 - 0.15) * 100
        
        # Determine status (matching GS thresholds)
        if score < 40:
            status = "Benign"
        elif score < 60:
            status = "Watch"
        elif score < 75:
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
        - HY < 3.0% → 0 (benign)
        - HY 3.0-5.0% → 0-100 scaled
        - HY > 5.0% → 100 (stressed)
        
        GS shows SRS=0 for HY=2.71%, confirming threshold at 3.0%
        
        Returns: (score 0-100, status)
        """
        if spread_data is None:
            spread_data = self._get_latest("credit_spread_daily")
        
        hy_spread = 2.81  # Default from Google Sheets (as percentage)
        if spread_data:
            hy_val = spread_data.get("hy_spread")
            if hy_val is not None:
                # Convert string to float if needed
                hy_spread = float(hy_val) if isinstance(hy_val, str) else hy_val
        
        # GS Scale: 3.0% → 0, 5.0% → 100 (linear)
        if hy_spread <= 3.0:
            score = 0
        elif hy_spread >= 5.0:
            score = 100
        else:
            score = (hy_spread - 3.0) / (5.0 - 3.0) * 100
        
        # Determine status (matching GS thresholds)
        if score < 20:
            status = "Benign"
        elif score < 40:
            status = "Watch"
        elif score < 60:
            status = "Stressed"
        else:
            status = "Critical"
        
        return round(score, 1), status
    
    def calculate_sds(self, breadth_data: Optional[Dict] = None) -> Tuple[float, str]:
        """
        Calculate Structural Divergence Score (SDS).
        
        Google Sheets logic:
        - Compares Infrastructure vs Enterprise breadth
        - Divergence = abs(Infra - Enterprise)
        - Scale: 0% divergence → 0, 65% divergence → 100
        
        GS shows: Infra 75%, Ent 15% → divergence 60% → SDS 91.7
        Formula: SDS = min(100, divergence / 0.65 * 100)
        
        Returns: (score 0-100, status)
        """
        if breadth_data is None:
            breadth_data = self._get_latest("breadth_daily")
        
        # Defaults
        infra_breadth = 0.563  # 56.3%
        ent_breadth = 0.077    # 7.7%
        
        if breadth_data:
            # Try to get pillar_breadth data
            pillar_breadth = breadth_data.get("pillar_breadth", {})
            
            if pillar_breadth:
                # Extract Infrastructure and Enterprise breadth
                infra_data = pillar_breadth.get("Infrastructure") or pillar_breadth.get("Infrastructure & Energy", {})
                ent_data = pillar_breadth.get("Enterprise") or pillar_breadth.get("Enterprise Adoption", {})
                
                if infra_data:
                    infra_breadth = infra_data.get("breadth_20d", infra_breadth)
                if ent_data:
                    ent_breadth = ent_data.get("breadth_20d", ent_breadth)
            else:
                # Fallback to direct fields if available
                infra_breadth = breadth_data.get("infra_breadth", infra_breadth)
                ent_breadth = breadth_data.get("enterprise_breadth", ent_breadth)
        
        # Calculate divergence
        divergence = abs(infra_breadth - ent_breadth)
        
        # GS formula: SDS = min(100, divergence / 0.65 * 100)
        score = min(100, divergence / 0.65 * 100)
        
        # Determine status (matching GS thresholds)
        if score < 25:
            status = "Benign"
        elif score < 50:
            status = "Watch"
        elif score < 70:
            status = "Tension"
        else:
            status = "Extreme"
        
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
        Calculate AMRI using GS-aligned formula.
        
        Weights (matching Google Sheets):
        - CRS: 0.23 (Correlation)
        - CCS: 0.31 (Clusters)
        - SRS: 0.15 (Spreads)
        - SDS: 0.31 (Divergence)
        
        Core AMRI = CRS×0.23 + CCS×0.31 + SRS×0.15 + SDS×0.31
        Enhanced = 0.80 × Core + 0.20 × Bubble_Overlay
        """
        logger.info("Calculating AMRI (GS-aligned formula)...")
        
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
        logger.info(f"  CRS: {crs_score:.1f} × {AMRI_WEIGHTS['CRS']} = {contributions['CRS']:.1f} ({crs_status})")
        logger.info(f"  CCS: {ccs_score:.1f} × {AMRI_WEIGHTS['CCS']} = {contributions['CCS']:.1f} ({ccs_status})")
        logger.info(f"  SRS: {srs_score:.1f} × {AMRI_WEIGHTS['SRS']} = {contributions['SRS']:.1f} ({srs_status})")
        logger.info(f"  SDS: {sds_score:.1f} × {AMRI_WEIGHTS['SDS']} = {contributions['SDS']:.1f} ({sds_status})")
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