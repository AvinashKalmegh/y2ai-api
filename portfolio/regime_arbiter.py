"""
REGIME ARBITER
==============
Resolves conflicting signals into a single unified regime assessment.

Priority Hierarchy:
1. Fragility Model (structural break conditions)
2. AMRI + Bubble Overlay (structure + dynamics)
3. AMRI + Bubble Index (structural stress + valuation)
4. NST + Breadth (narrative + price)
5. Default regime assessment

Signal Sources:
- AMRI (4-score stress index)
- Bubble Index (valuation)
- Bubble Overlay (dynamics)
- Correlations (herding)
- Breadth (participation)
- NST Veto (narrative triggers)
- NCI (narrative coherence)
- NPD (narrative polarity drift)
- Burst (keyword bursts)
- EVI (event volatility)
- Hypergraph (contagion)

Output Regimes:
- NORMAL: No structural concern
- CAUTION: Elevated but manageable
- ELEVATED: Multiple warning signs
- FRAGILE: High risk, prioritize defense
- BREAK: Structural collapse likely
- VETO_ALERT: High-severity event detected
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

REGIME_THRESHOLDS = {
    # AMRI thresholds
    "amri_normal": 30,
    "amri_elevated": 50,
    "amri_fragile": 70,
    
    # Bubble Index thresholds
    "bubble_moderate": 50,
    "bubble_elevated": 70,
    
    # Correlation thresholds
    "corr_normal": 40,
    "corr_elevated": 60,
    
    # Breadth thresholds (%)
    "breadth_strong": 50,
    "breadth_collapsed": 25,
    
    # NST Veto thresholds
    "veto_moderate": 15,
    "veto_high": 30,
    
    # NCI thresholds
    "nci_low": 30,
    "nci_high": 60,
    
    # Fragility conditions for BREAK
    "fragility_break": 3,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignalReading:
    """Single signal reading."""
    name: str
    value: float
    regime: str
    classification: str  # BULLISH, NEUTRAL, BEARISH


@dataclass
class ArbiterResult:
    """Complete arbiter result."""
    date: str
    regime: str
    confidence: str  # HIGH, MEDIUM, LOW
    driver: str      # Primary driver of regime
    interpretation: str
    # Signal counts
    bullish_count: int
    bearish_count: int
    neutral_count: int
    # Signal lists
    bullish_signals: List[str]
    bearish_signals: List[str]
    neutral_signals: List[str]
    watch_list: List[str]
    # Raw signal values
    signals: Dict


# =============================================================================
# REGIME ARBITER
# =============================================================================

class RegimeArbiter:
    """
    Resolve conflicting signals into unified regime.
    
    Usage:
        arbiter = RegimeArbiter()
        result = arbiter.arbitrate()
        arbiter.save_to_supabase(result)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize arbiter."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    # =========================================================================
    # SIGNAL GATHERING
    # =========================================================================
    
    def gather_signals(self) -> Dict:
        """Gather all signals from various sources."""
        signals = {}
        
        # AMRI
        signals["amri"] = self._get_amri_signal()
        
        # Bubble Index
        signals["bubble_index"] = self._get_bubble_signal()
        
        # Correlations
        signals["correlations"] = self._get_correlation_signal()
        
        # Breadth
        signals["breadth"] = self._get_breadth_signal()
        
        # MCI
        signals["mci"] = self._get_mci_signal()
        
        # NCI (Narrative Coherence)
        signals["nci"] = self._get_nci_signal()
        
        # NPD (Narrative Polarity Drift)
        signals["npd"] = self._get_npd_signal()
        
        # EVI (Event Volatility)
        signals["evi"] = self._get_evi_signal()
        
        # Burst
        signals["burst"] = self._get_burst_signal()
        
        # Hypergraph
        signals["hypergraph"] = self._get_hypergraph_signal()
        
        # NST Veto
        signals["nst_veto"] = self._get_nst_veto_signal()
        
        return signals
    
    def _get_amri_signal(self) -> Dict:
        """Get AMRI signal."""
        default = {"score": 30, "regime": "Normal", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("amri_daily") \
                .select("amri_score, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "score": response.data[0].get("amri_score", 30),
                    "regime": response.data[0].get("regime", "Normal"),
                    "available": True
                }
        except Exception as e:
            logger.warning(f"Failed to get AMRI: {e}")
        
        return default
    
    def _get_bubble_signal(self) -> Dict:
        """Get Bubble Index signal."""
        default = {"score": 50, "regime": "Moderate", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("bubble_index_daily") \
                .select("bubble_index, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "score": response.data[0].get("bubble_index", 50),
                    "regime": response.data[0].get("regime", "Moderate"),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_correlation_signal(self) -> Dict:
        """Get correlation signal."""
        default = {"score": 40, "regime": "Normal", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("correlation_daily") \
                .select("avg_correlation, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                corr = response.data[0].get("avg_correlation", 0.4)
                return {
                    "score": corr * 100,  # Convert to %
                    "regime": response.data[0].get("regime", "Normal"),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_breadth_signal(self) -> Dict:
        """Get breadth signal."""
        default = {"infrastructure": 50, "enterprise": 50, "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("breadth_daily") \
                .select("breadth_20d, pillar_breadth") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                row = response.data[0]
                pillar_breadth = row.get("pillar_breadth", {})
                
                return {
                    "overall": row.get("breadth_20d", 0.5) * 100,
                    "infrastructure": pillar_breadth.get("Infrastructure", {}).get("breadth_20d", 0.5) * 100,
                    "enterprise": pillar_breadth.get("Enterprise", {}).get("breadth_20d", 0.5) * 100,
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_mci_signal(self) -> Dict:
        """Get MCI signal."""
        default = {"score": 0, "regime": "Knife Edge", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("mci_daily") \
                .select("mci_score, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "score": response.data[0].get("mci_score", 0),
                    "regime": response.data[0].get("regime", "Unknown"),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_nci_signal(self) -> Dict:
        """Get NCI (Narrative Coherence Index) signal."""
        default = {"score": 30, "regime": "Normal", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("nci_daily") \
                .select("nci_score, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "score": response.data[0].get("nci_score", 30),
                    "regime": response.data[0].get("regime", "Normal"),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_npd_signal(self) -> Dict:
        """Get NPD (Narrative Polarity Drift) signal."""
        default = {"score": 0, "regime": "STABLE", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("npd_daily") \
                .select("npd_score, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "score": response.data[0].get("npd_score", 0),
                    "regime": response.data[0].get("regime", "STABLE"),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_evi_signal(self) -> Dict:
        """Get EVI (Event Volatility Index) signal."""
        default = {"score": 0.3, "regime": "NORMAL", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("evi_daily") \
                .select("evi_score, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "score": response.data[0].get("evi_score", 0.3),
                    "regime": response.data[0].get("regime", "NORMAL"),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_burst_signal(self) -> Dict:
        """Get Burst (keyword detection) signal."""
        default = {"count": 0, "regime": "NORMAL", "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("burst_daily") \
                .select("burst_count, regime, bearish_bursts, bullish_bursts") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "count": response.data[0].get("burst_count", 0),
                    "regime": response.data[0].get("regime", "NORMAL"),
                    "bearish": response.data[0].get("bearish_bursts", 0),
                    "bullish": response.data[0].get("bullish_bursts", 0),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_hypergraph_signal(self) -> Dict:
        """Get Hypergraph (contagion) signal."""
        default = {"regime": "STABLE", "contagion": 0, "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("hypergraph_daily") \
                .select("regime, contagion_score, stability") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "regime": response.data[0].get("regime", "STABLE"),
                    "contagion": response.data[0].get("contagion_score", 0),
                    "stability": response.data[0].get("stability", 1),
                    "available": True
                }
        except:
            pass
        
        return default
    
    def _get_nst_veto_signal(self) -> Dict:
        """Get NST Veto signal."""
        default = {"count": 0, "available": False}
        
        if not self.supabase:
            return default
        
        try:
            response = self.supabase.table("nst_veto_daily") \
                .select("veto_count") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return {
                    "count": response.data[0].get("veto_count", 0),
                    "available": True
                }
        except:
            pass
        
        return default
    
    # =========================================================================
    # SIGNAL CLASSIFICATION
    # =========================================================================
    
    def classify_signals(self, signals: Dict) -> Tuple[List[str], List[str], List[str]]:
        """
        Classify each signal as bullish, bearish, or neutral.
        
        Returns:
            Tuple of (bullish_signals, bearish_signals, neutral_signals)
        """
        bullish = []
        bearish = []
        neutral = []
        
        s = signals
        t = REGIME_THRESHOLDS
        
        # === AMRI ===
        amri_score = s["amri"].get("score", 30)
        if amri_score >= t["amri_fragile"]:
            bearish.append(f"AMRI Fragile ({amri_score:.0f})")
        elif amri_score >= t["amri_elevated"]:
            bearish.append(f"AMRI Tension ({amri_score:.0f})")
        elif amri_score >= t["amri_normal"]:
            neutral.append(f"AMRI Elevated ({amri_score:.0f})")
        else:
            bullish.append(f"AMRI Normal ({amri_score:.0f})")
        
        # === Bubble Index ===
        bubble_score = s["bubble_index"].get("score", 50)
        if bubble_score >= t["bubble_elevated"]:
            bearish.append(f"Bubble High ({bubble_score:.0f})")
        elif bubble_score >= t["bubble_moderate"]:
            neutral.append(f"Bubble Elevated ({bubble_score:.0f})")
        else:
            bullish.append(f"Bubble Moderate ({bubble_score:.0f})")
        
        # === Correlations ===
        corr_score = s["correlations"].get("score", 40)
        if corr_score >= t["corr_elevated"]:
            bearish.append(f"Correlations Fragile ({corr_score:.0f}%)")
        elif corr_score >= t["corr_normal"]:
            neutral.append(f"Correlations Elevated ({corr_score:.0f}%)")
        else:
            bullish.append(f"Correlations Normal ({corr_score:.0f}%)")
        
        # === Breadth ===
        breadth_infra = s["breadth"].get("infrastructure", 50)
        breadth_ent = s["breadth"].get("enterprise", 50)
        
        if breadth_infra < t["breadth_collapsed"] and breadth_ent < t["breadth_collapsed"]:
            bearish.append(f"Breadth Collapsed ({breadth_infra:.0f}%/{breadth_ent:.0f}%)")
        elif breadth_infra > t["breadth_strong"] and breadth_ent > t["breadth_strong"]:
            bullish.append(f"Breadth Strong ({breadth_infra:.0f}%/{breadth_ent:.0f}%)")
        else:
            neutral.append(f"Breadth Mixed ({breadth_infra:.0f}%/{breadth_ent:.0f}%)")
        
        # === MCI ===
        mci_score = s["mci"].get("score", 0)
        if mci_score > 40:
            bullish.append(f"MCI Melt-Up ({mci_score:+.0f})")
        elif mci_score > 10:
            bullish.append(f"MCI Extension ({mci_score:+.0f})")
        elif mci_score > -10:
            neutral.append(f"MCI Knife Edge ({mci_score:+.0f})")
        elif mci_score > -40:
            bearish.append(f"MCI Collapse Bias ({mci_score:+.0f})")
        else:
            bearish.append(f"MCI Break Path ({mci_score:+.0f})")
        
        # === NCI ===
        if s["nci"].get("available"):
            nci_score = s["nci"].get("score", 30)
            if nci_score >= t["nci_high"]:
                bearish.append(f"NCI High ({nci_score:.0f})")
            elif nci_score >= t["nci_low"]:
                neutral.append(f"NCI Moderate ({nci_score:.0f})")
            else:
                bullish.append(f"NCI Low ({nci_score:.0f})")
        
        # === NPD ===
        if s["npd"].get("available"):
            npd_regime = s["npd"].get("regime", "STABLE")
            npd_score = s["npd"].get("score", 0)
            if npd_regime == "BULLISH_DRIFT":
                bullish.append(f"NPD Bullish ({npd_score:+.2f})")
            elif npd_regime == "BEARISH_DRIFT":
                bearish.append(f"NPD Bearish ({npd_score:+.2f})")
            else:
                neutral.append(f"NPD Stable ({npd_score:+.2f})")
        
        # === EVI ===
        if s["evi"].get("available"):
            evi_regime = s["evi"].get("regime", "NORMAL")
            evi_score = s["evi"].get("score", 0.3)
            if evi_regime == "VOLATILE":
                bearish.append(f"EVI Volatile ({evi_score:.2f})")
            elif evi_regime == "ELEVATED":
                neutral.append(f"EVI Elevated ({evi_score:.2f})")
            elif evi_regime == "STABLE":
                bullish.append(f"EVI Stable ({evi_score:.2f})")
        
        # === Burst ===
        if s["burst"].get("available"):
            burst_regime = s["burst"].get("regime", "NORMAL")
            burst_bearish = s["burst"].get("bearish", 0)
            burst_bullish = s["burst"].get("bullish", 0)
            
            if burst_regime == "ELEVATED":
                if burst_bearish > burst_bullish:
                    bearish.append(f"Bursts Bearish ({burst_bearish})")
                else:
                    bullish.append(f"Bursts Bullish ({burst_bullish})")
            elif burst_regime == "ACTIVE":
                neutral.append(f"Bursts Active ({s['burst'].get('count', 0)})")
        
        # === Hypergraph ===
        if s["hypergraph"].get("available"):
            hyper_regime = s["hypergraph"].get("regime", "STABLE")
            contagion = s["hypergraph"].get("contagion", 0)
            
            if hyper_regime == "CONTAGION":
                bearish.append(f"Hypergraph CONTAGION ({contagion:.0f})")
            elif hyper_regime == "FRAGMENTING":
                bearish.append(f"Hypergraph Fragmenting ({contagion:.0f})")
            elif hyper_regime == "ACCELERATING":
                neutral.append(f"Hypergraph Accelerating ({contagion:.0f})")
            elif hyper_regime == "STABLE":
                bullish.append(f"Hypergraph Stable ({contagion:.0f})")
        
        # === NST Veto ===
        veto_count = s["nst_veto"].get("count", 0)
        if veto_count >= t["veto_high"]:
            bearish.append(f"NST Veto High ({veto_count})")
        elif veto_count >= t["veto_moderate"]:
            neutral.append(f"NST Veto Moderate ({veto_count})")
        else:
            bullish.append(f"NST Veto Low ({veto_count})")
        
        return bullish, bearish, neutral
    
    # =========================================================================
    # FRAGILITY CHECK
    # =========================================================================
    
    def count_fragility_conditions(self, signals: Dict) -> int:
        """Count number of fragility conditions met."""
        count = 0
        t = REGIME_THRESHOLDS
        
        # AMRI fragile
        if signals["amri"].get("score", 0) >= t["amri_fragile"]:
            count += 1
        
        # Bubble high
        if signals["bubble_index"].get("score", 0) >= t["bubble_elevated"]:
            count += 1
        
        # Correlations fragile
        if signals["correlations"].get("score", 0) >= t["corr_elevated"]:
            count += 1
        
        # Breadth collapsed (both pillars)
        if (signals["breadth"].get("infrastructure", 50) < t["breadth_collapsed"] and
            signals["breadth"].get("enterprise", 50) < t["breadth_collapsed"]):
            count += 1
        
        # NST Veto high
        if signals["nst_veto"].get("count", 0) >= t["veto_high"]:
            count += 1
        
        # Hypergraph contagion
        if signals["hypergraph"].get("regime") == "CONTAGION":
            count += 1
        
        # MCI Break Path
        if signals["mci"].get("score", 0) < -40:
            count += 1
        
        return count
    
    # =========================================================================
    # MAIN ARBITRATION
    # =========================================================================
    
    def arbitrate(self) -> ArbiterResult:
        """
        Main arbitration: gather signals and determine unified regime.
        
        Returns:
            ArbiterResult with regime, confidence, and signal breakdown
        """
        logger.info("Running regime arbitration...")
        
        # Gather signals
        signals = self.gather_signals()
        
        # Classify signals
        bullish, bearish, neutral = self.classify_signals(signals)
        
        # Count fragility conditions
        fragility_count = self.count_fragility_conditions(signals)
        
        # Determine regime
        regime, confidence, driver = self._determine_regime(
            signals, bullish, bearish, neutral, fragility_count
        )
        
        # Build watch list
        watch_list = self._build_watch_list(signals, bearish)
        
        # Get interpretation
        interpretation = self._get_interpretation(regime)
        
        result = ArbiterResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            regime=regime,
            confidence=confidence,
            driver=driver,
            interpretation=interpretation,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            bullish_signals=bullish,
            bearish_signals=bearish,
            neutral_signals=neutral,
            watch_list=watch_list,
            signals=signals
        )
        
        logger.info(f"Regime: {regime} ({confidence})")
        logger.info(f"Bullish: {len(bullish)}, Bearish: {len(bearish)}, Neutral: {len(neutral)}")
        
        return result
    
    def _determine_regime(
        self,
        signals: Dict,
        bullish: List[str],
        bearish: List[str],
        neutral: List[str],
        fragility_count: int
    ) -> Tuple[str, str, str]:
        """
        Determine final regime, confidence, and driver.
        
        Returns:
            Tuple of (regime, confidence, driver)
        """
        t = REGIME_THRESHOLDS
        
        # === Priority 1: Fragility Model ===
        if fragility_count >= t["fragility_break"]:
            return "BREAK", "HIGH", f"Fragility conditions: {fragility_count}/7"
        
        # === Priority 2: NST Veto Override ===
        veto_count = signals["nst_veto"].get("count", 0)
        if veto_count >= t["veto_high"]:
            return "VETO_ALERT", "HIGH", f"NST Veto triggers: {veto_count}"
        
        # === Priority 3: Hypergraph Contagion ===
        if signals["hypergraph"].get("regime") == "CONTAGION":
            contagion = signals["hypergraph"].get("contagion", 0)
            if contagion > 75:
                return "BREAK", "HIGH", f"Hypergraph contagion: {contagion}"
            else:
                return "FRAGILE", "HIGH", f"Hypergraph contagion: {contagion}"
        
        # === Priority 4: Signal Counts ===
        n_bearish = len(bearish)
        n_bullish = len(bullish)
        total = n_bearish + n_bullish + len(neutral)
        
        if n_bearish >= 5:
            return "FRAGILE", "HIGH", f"Multiple bearish signals: {n_bearish}"
        elif n_bearish >= 3:
            return "ELEVATED", "MEDIUM", f"Elevated bearish signals: {n_bearish}"
        elif n_bearish >= n_bullish and n_bearish >= 2:
            return "CAUTION", "MEDIUM", "Bearish signals outweigh bullish"
        elif n_bullish > n_bearish + 2:
            return "NORMAL", "HIGH", f"Strong bullish signals: {n_bullish}"
        else:
            return "NORMAL", "MEDIUM", "Mixed signals favor normal"
    
    def _build_watch_list(self, signals: Dict, bearish: List[str]) -> List[str]:
        """Build watch list of items to monitor."""
        watch = []
        
        # Add any signal near threshold
        amri = signals["amri"].get("score", 0)
        if 25 <= amri < 30:
            watch.append(f"AMRI approaching elevated ({amri:.0f})")
        
        bubble = signals["bubble_index"].get("score", 0)
        if 45 <= bubble < 50:
            watch.append(f"Bubble Index approaching elevated ({bubble:.0f})")
        
        # Add bearish signals to watch
        for sig in bearish[:3]:  # Top 3 bearish
            watch.append(f"Monitor: {sig}")
        
        return watch
    
    def _get_interpretation(self, regime: str) -> str:
        """Get human-readable interpretation for regime."""
        interpretations = {
            "NORMAL": "No structural concern. Normal operations.",
            "CAUTION": "Elevated but manageable. Monitor closely.",
            "ELEVATED": "Multiple warning signs. Reduce risk exposure.",
            "FRAGILE": "High risk environment. Prioritize defense, hedge tail risk.",
            "BREAK": "Structural collapse likely. De-risk fully, raise cash.",
            "VETO_ALERT": "High-severity event detected. Do NOT add risk."
        }
        return interpretations.get(regime, "Unknown regime")
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, result: ArbiterResult) -> bool:
        """Save arbiter result to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return False
        
        row = {
            "date": result.date,
            "regime": result.regime,
            "confidence": result.confidence,
            "driver": result.driver,
            "interpretation": result.interpretation,
            "bullish_count": result.bullish_count,
            "bearish_count": result.bearish_count,
            "neutral_count": result.neutral_count,
            "bullish_signals": result.bullish_signals,
            "bearish_signals": result.bearish_signals,
            "neutral_signals": result.neutral_signals,
            "watch_list": result.watch_list
        }
        
        try:
            self.supabase.table("regime_arbiter_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved arbiter result for {result.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def get_latest(self) -> Optional[ArbiterResult]:
        """Get most recent arbiter result."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table("regime_arbiter_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                row = response.data[0]
                return ArbiterResult(
                    date=row["date"],
                    regime=row["regime"],
                    confidence=row["confidence"],
                    driver=row["driver"],
                    interpretation=row["interpretation"],
                    bullish_count=row["bullish_count"],
                    bearish_count=row["bearish_count"],
                    neutral_count=row["neutral_count"],
                    bullish_signals=row["bullish_signals"],
                    bearish_signals=row["bearish_signals"],
                    neutral_signals=row["neutral_signals"],
                    watch_list=row["watch_list"],
                    signals={}
                )
        except Exception as e:
            logger.error(f"Failed to get latest: {e}")
        
        return None


# =============================================================================
# SUPABASE TABLE
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS regime_arbiter_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    regime VARCHAR(30),
    confidence VARCHAR(20),
    driver TEXT,
    interpretation TEXT,
    bullish_count INT,
    bearish_count INT,
    neutral_count INT,
    bullish_signals JSONB,
    bearish_signals JSONB,
    neutral_signals JSONB,
    watch_list JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_arbiter_date ON regime_arbiter_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run regime arbitration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regime Arbiter")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--alert", action="store_true", help="Check and send alerts")
    args = parser.parse_args()
    
    arbiter = RegimeArbiter()
    
    print(f"\n{'='*60}")
    print("REGIME ARBITER")
    print(f"{'='*60}\n")
    
    result = arbiter.arbitrate()
    
    print(f"Date: {result.date}")
    print(f"\n{'='*40}")
    print(f"REGIME: {result.regime}")
    print(f"Confidence: {result.confidence}")
    print(f"Driver: {result.driver}")
    print(f"{'='*40}")
    print(f"\n{result.interpretation}")
    
    print(f"\n{'='*40}")
    print("SIGNAL BREAKDOWN")
    print(f"{'='*40}")
    print(f"\n🟢 BULLISH ({result.bullish_count}):")
    for sig in result.bullish_signals:
        print(f"   {sig}")
    
    print(f"\n🔴 BEARISH ({result.bearish_count}):")
    for sig in result.bearish_signals:
        print(f"   {sig}")
    
    print(f"\n🟡 NEUTRAL ({result.neutral_count}):")
    for sig in result.neutral_signals:
        print(f"   {sig}")
    
    if result.watch_list:
        print(f"\n{'='*40}")
        print("WATCH LIST")
        print(f"{'='*40}")
        for item in result.watch_list:
            print(f"   ⚠️ {item}")
    
    if args.save:
        if arbiter.save_to_supabase(result):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
