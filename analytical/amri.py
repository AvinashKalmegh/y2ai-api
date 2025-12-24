"""
AMRI (ARGUS Master Regime Index) Calculator

Calculates the unified AMRI composite score from existing data sources:
- bubble_index_daily (VIX, CAPE, Credit Spreads, bifurcation_score)
- hypergraph_signals (contagion_score, stability, clusters)
- daily_signals (NLP signals, VETO triggers)
- stock_tracker_daily (pillar performance, rotation)

Formula: AMRI = CRS×0.25 + CCS×0.25 + SRS×0.20 + VIX×0.15 + SDS×0.15
"""

import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import numpy as np

from .config import (
    AMRI_WEIGHTS, AMRI_WEIGHTS_ALT, 
    AMRI_S_WEIGHTS, AMRI_B_WEIGHTS, AMRI_C_WEIGHTS,
    AMRI_THRESHOLDS, VIX_THRESHOLDS,
    Regime, Authority, Confidence
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AMRIComponents:
    """Individual AMRI component scores (0-100 scale)"""
    crs: float = 0.0   # Correlation Regime Score
    ccs: float = 0.0   # Cluster Concentration Score
    srs: float = 0.0   # Spread Regime Score
    vix: float = 0.0   # VIX contribution
    sds: float = 0.0   # Structural Divergence Score


@dataclass
class AMRIDecomposition:
    """AMRI broken into S/B/C components"""
    amri_s: float = 0.0  # Structural capacity to break
    amri_b: float = 0.0  # Behavioral pressure building
    amri_c: float = 0.0  # Catalyst risk


@dataclass
class AMRIResult:
    """Complete AMRI calculation result"""
    date: str
    composite: float  # 0-100
    components: AMRIComponents
    decomposition: AMRIDecomposition
    regime: Regime
    authority: Authority
    confidence: Confidence
    binding_constraint: str  # Which factor is driving the regime
    calculated_at: str
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['components'] = asdict(self.components)
        result['decomposition'] = asdict(self.decomposition)
        result['regime'] = self.regime.value
        result['authority'] = self.authority.value
        result['confidence'] = self.confidence.value
        return result


class AMRICalculator:
    """
    Calculate AMRI from existing Supabase tables.
    
    Pulls data from:
    - bubble_index_daily: VIX, credit spreads, bubble_index
    - hypergraph_signals: contagion, stability, cluster count
    - daily_signals: NLP signals
    - stock_tracker_daily: pillar divergence
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
    
    def _fetch_bubble_data(self, date: str) -> Optional[Dict]:
        """Fetch bubble index data for date"""
        if not self.client:
            return None
        try:
            result = self.client.table("bubble_index_daily")\
                .select("*")\
                .eq("date", date)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching bubble data: {e}")
            return None
    
    def _fetch_hypergraph_data(self, date: str) -> Optional[Dict]:
        """Fetch hypergraph signals for date"""
        if not self.client:
            return None
        try:
            result = self.client.table("hypergraph_signals")\
                .select("*")\
                .eq("date", date)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching hypergraph data: {e}")
            return None
    
    def _fetch_daily_signals(self, date: str) -> Optional[Dict]:
        """Fetch NLP daily signals for date"""
        if not self.client:
            return None
        try:
            result = self.client.table("daily_signals")\
                .select("*")\
                .eq("date", date)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching daily signals: {e}")
            return None
    
    def _fetch_stock_tracker(self, date: str) -> Optional[Dict]:
        """Fetch stock tracker data for date"""
        if not self.client:
            return None
        try:
            result = self.client.table("stock_tracker_daily")\
                .select("*")\
                .eq("date", date)\
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error fetching stock tracker: {e}")
            return None
    
    def _fetch_latest_of_each(self) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
        """Fetch most recent data from each table"""
        bubble = hypergraph = signals = tracker = None
        
        if not self.client:
            return bubble, hypergraph, signals, tracker
        
        try:
            # Bubble index
            r = self.client.table("bubble_index_daily")\
                .select("*").order("date", desc=True).limit(1).execute()
            bubble = r.data[0] if r.data else None
            
            # Hypergraph
            r = self.client.table("hypergraph_signals")\
                .select("*").order("date", desc=True).limit(1).execute()
            hypergraph = r.data[0] if r.data else None
            
            # Daily signals
            r = self.client.table("daily_signals")\
                .select("*").order("date", desc=True).limit(1).execute()
            signals = r.data[0] if r.data else None
            
            # Stock tracker
            r = self.client.table("stock_tracker_daily")\
                .select("*").order("date", desc=True).limit(1).execute()
            tracker = r.data[0] if r.data else None
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
        
        return bubble, hypergraph, signals, tracker
    
    def calculate_crs(self, hypergraph: Optional[Dict]) -> float:
        """
        Calculate Correlation Regime Score (CRS).
        Based on cross-pillar correlation intensity.
        """
        if not hypergraph:
            return 50.0  # Neutral default
        
        cross_pillar_ratio = hypergraph.get("cross_pillar_ratio", 0)
        avg_size = hypergraph.get("avg_hyperedge_size", 3)
        
        # Higher cross-pillar = higher CRS (more correlated = more risk)
        # Scale: 0-1 ratio -> 0-100 score
        crs = cross_pillar_ratio * 60 + min(avg_size / 10, 1) * 40
        return min(100, max(0, crs))
    
    def calculate_ccs(self, hypergraph: Optional[Dict]) -> float:
        """
        Calculate Cluster Concentration Score (CCS).
        Based on number of connected components (fewer = more concentrated = higher risk).
        """
        if not hypergraph:
            return 50.0
        
        hyperedge_count = hypergraph.get("hyperedge_count", 10)
        max_size = hypergraph.get("max_hyperedge_size", 5)
        
        # Fewer clusters = higher concentration = higher CCS
        # More than 15 clusters = healthy diversity
        # Less than 5 clusters = dangerous concentration
        if hyperedge_count >= 15:
            cluster_score = 20
        elif hyperedge_count >= 10:
            cluster_score = 40
        elif hyperedge_count >= 7:
            cluster_score = 60
        elif hyperedge_count >= 5:
            cluster_score = 80
        else:
            cluster_score = 95
        
        # Large max cluster size adds to score
        size_penalty = min(max_size / 30, 1) * 20
        
        ccs = cluster_score + size_penalty
        return min(100, max(0, ccs))
    
    def calculate_srs(self, bubble: Optional[Dict]) -> float:
        """
        Calculate Spread Regime Score (SRS).
        Based on credit spreads - wider = more stress = higher score.
        """
        if not bubble:
            return 50.0
        
        credit_zscore = bubble.get("credit_zscore", 0)
        
        # Z-score to 0-100 scale
        # z=0 -> 50, z=2 -> 80, z=-2 -> 20
        srs = 50 + credit_zscore * 15
        return min(100, max(0, srs))
    
    def calculate_vix_component(self, bubble: Optional[Dict]) -> float:
        """
        Calculate VIX contribution to AMRI.
        Higher VIX = higher risk = higher score.
        """
        if not bubble:
            return 50.0
        
        vix = bubble.get("vix", 15)
        vix_zscore = bubble.get("vix_zscore", 0)
        
        # VIX levels to score:
        # VIX 12-15 -> 20-30 (calm)
        # VIX 15-20 -> 30-50 (normal)
        # VIX 20-25 -> 50-65 (elevated)
        # VIX 25-35 -> 65-85 (high)
        # VIX 35+ -> 85-100 (extreme)
        
        if vix < 15:
            score = 20 + (vix / 15) * 10
        elif vix < 20:
            score = 30 + ((vix - 15) / 5) * 20
        elif vix < 25:
            score = 50 + ((vix - 20) / 5) * 15
        elif vix < 35:
            score = 65 + ((vix - 25) / 10) * 20
        else:
            score = 85 + min((vix - 35) / 20, 1) * 15
        
        return min(100, max(0, score))
    
    def calculate_sds(self, tracker: Optional[Dict]) -> float:
        """
        Calculate Structural Divergence Score (SDS).
        Measures spread between pillar performance (generals vs soldiers).
        """
        if not tracker:
            return 50.0
        
        pillars = tracker.get("pillars", [])
        if not pillars or len(pillars) < 2:
            return 50.0
        
        # Extract pillar performances
        perfs = [p.get("change_5day", 0) for p in pillars if isinstance(p, dict)]
        if len(perfs) < 2:
            return 50.0
        
        # Divergence = spread between best and worst pillar
        max_perf = max(perfs)
        min_perf = min(perfs)
        divergence = abs(max_perf - min_perf)
        
        # Scale: 0-10% spread = 0-50, 10-30% spread = 50-100
        if divergence < 0.10:
            sds = divergence / 0.10 * 50
        else:
            sds = 50 + min((divergence - 0.10) / 0.20, 1) * 50
        
        return min(100, max(0, sds))
    
    def calculate_amri_s(self, components: AMRIComponents) -> float:
        """Calculate AMRI-S (Structural capacity to break)"""
        return (
            components.crs * AMRI_S_WEIGHTS["CRS"] +
            components.ccs * AMRI_S_WEIGHTS["CCS"]
        )
    
    def calculate_amri_b(self, components: AMRIComponents, bubble: Optional[Dict]) -> float:
        """Calculate AMRI-B (Behavioral pressure building)"""
        bubble_index = bubble.get("bubble_index", 50) if bubble else 50
        return (
            components.sds * AMRI_B_WEIGHTS["SDS"] +
            bubble_index * AMRI_B_WEIGHTS["Bubble"]
        )
    
    def calculate_amri_c(self, components: AMRIComponents, hypergraph: Optional[Dict], signals: Optional[Dict]) -> float:
        """Calculate AMRI-C (Catalyst risk)"""
        contagion = hypergraph.get("contagion_score", 50) if hypergraph else 50
        
        # NST from daily signals
        nst = 50  # Default neutral
        if signals:
            veto_count = signals.get("veto_triggers", 0)
            thesis_balance = signals.get("thesis_balance", 0)
            # Higher veto = higher catalyst risk
            # Negative thesis balance = higher risk
            nst = 50 + veto_count * 5 - thesis_balance * 0.3
            nst = min(100, max(0, nst))
        
        return (
            components.srs * AMRI_C_WEIGHTS["SRS"] +
            nst * AMRI_C_WEIGHTS["NST"] +
            contagion * AMRI_C_WEIGHTS["Contagion"]
        )
    
    def determine_regime(self, amri: float, veto_active: bool = False) -> Tuple[Regime, str]:
        """Determine regime from AMRI score"""
        binding = "AMRI"
        
        # VETO override
        if veto_active:
            return Regime.FRAGILE, "VETO"
        
        for regime, (low, high) in AMRI_THRESHOLDS.items():
            if low <= amri < high:
                return Regime[regime], binding
        
        return Regime.BREAK if amri >= 90 else Regime.NORMAL, binding
    
    def determine_authority(self, regime: Regime, hypergraph: Optional[Dict], signals: Optional[Dict]) -> Authority:
        """Determine which authority level is driving the regime"""
        if regime == Regime.BREAK:
            return Authority.BREAK
        
        # Check for narrative/VETO signals
        if signals:
            veto_count = signals.get("veto_triggers", 0)
            if veto_count > 0:
                return Authority.NARRATIVE
        
        # Check for structural (hypergraph) signals
        if hypergraph:
            contagion = hypergraph.get("contagion_score", 0)
            if contagion > 70:
                return Authority.STRUCTURAL
        
        return Authority.MARKET
    
    def determine_confidence(self, bubble: Optional[Dict], hypergraph: Optional[Dict], signals: Optional[Dict]) -> Confidence:
        """Determine confidence level based on data freshness and consistency"""
        missing = 0
        if not bubble:
            missing += 1
        if not hypergraph:
            missing += 1
        if not signals:
            missing += 1
        
        if missing == 0:
            return Confidence.HIGH
        elif missing == 1:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW
    
    def calculate(self, date: str = None) -> AMRIResult:
        """
        Calculate complete AMRI for a given date.
        If no date provided, uses most recent data.
        """
        if date:
            bubble = self._fetch_bubble_data(date)
            hypergraph = self._fetch_hypergraph_data(date)
            signals = self._fetch_daily_signals(date)
            tracker = self._fetch_stock_tracker(date)
            calc_date = date
        else:
            bubble, hypergraph, signals, tracker = self._fetch_latest_of_each()
            calc_date = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate components
        components = AMRIComponents(
            crs=self.calculate_crs(hypergraph),
            ccs=self.calculate_ccs(hypergraph),
            srs=self.calculate_srs(bubble),
            vix=self.calculate_vix_component(bubble),
            sds=self.calculate_sds(tracker),
        )
        
        # Calculate composite AMRI using alternate weights
        amri = (
            components.crs * AMRI_WEIGHTS_ALT["CRS"] +
            components.ccs * AMRI_WEIGHTS_ALT["CCS"] +
            components.srs * AMRI_WEIGHTS_ALT["SRS"] +
            components.vix * AMRI_WEIGHTS_ALT["VIX"] +
            components.sds * AMRI_WEIGHTS_ALT["SDS"]
        )
        
        # Calculate decomposition
        decomposition = AMRIDecomposition(
            amri_s=self.calculate_amri_s(components),
            amri_b=self.calculate_amri_b(components, bubble),
            amri_c=self.calculate_amri_c(components, hypergraph, signals),
        )
        
        # Check for VETO
        veto_active = False
        if signals:
            veto_active = signals.get("veto_triggers", 0) > 0
        
        # Determine regime and authority
        regime, binding = self.determine_regime(amri, veto_active)
        authority = self.determine_authority(regime, hypergraph, signals)
        confidence = self.determine_confidence(bubble, hypergraph, signals)
        
        return AMRIResult(
            date=calc_date,
            composite=round(amri, 1),
            components=components,
            decomposition=decomposition,
            regime=regime,
            authority=authority,
            confidence=confidence,
            binding_constraint=binding,
            calculated_at=datetime.utcnow().isoformat(),
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    calculator = AMRICalculator()
    result = calculator.calculate()
    
    print(f"\n{'='*60}")
    print("AMRI CALCULATION")
    print(f"{'='*60}")
    print(f"Date: {result.date}")
    print(f"AMRI Composite: {result.composite}")
    print(f"Regime: {result.regime.value}")
    print(f"Authority: {result.authority.value}")
    print(f"Confidence: {result.confidence.value}")
    print(f"Binding: {result.binding_constraint}")
    print(f"\nComponents:")
    print(f"  CRS: {result.components.crs:.1f}")
    print(f"  CCS: {result.components.ccs:.1f}")
    print(f"  SRS: {result.components.srs:.1f}")
    print(f"  VIX: {result.components.vix:.1f}")
    print(f"  SDS: {result.components.sds:.1f}")
    print(f"\nDecomposition:")
    print(f"  AMRI-S: {result.decomposition.amri_s:.1f}")
    print(f"  AMRI-B: {result.decomposition.amri_b:.1f}")
    print(f"  AMRI-C: {result.decomposition.amri_c:.1f}")
