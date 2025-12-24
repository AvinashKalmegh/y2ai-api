"""
ARGUS-1 Master Calculator

Orchestrates all analytical modules to produce unified regime assessment.
This is the main entry point for the analytical layer.

Usage:
    from analytical import ARGUS1Calculator
    
    calc = ARGUS1Calculator()
    result = calc.run()
    
    print(result.regime)           # NORMAL, ELEVATED, TENSION, FRAGILE, BREAK
    print(result.amri.composite)   # 0-100
    print(result.tti.display)      # "3D to FRAGILE"
    print(result.format_summary()) # Full text summary
"""

import os
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

from .config import Regime, Authority, Confidence, DASHBOARD_CELLS
from .amri import AMRICalculator, AMRIResult
from .tti import TTICalculator, TTIResult
from .sac import SACCalculator, SACResult
from .fingerprints import FingerprintMatcher, FingerprintMatch
from .rotation import RotationTracker, RotationResult, RecoveryDetector, RecoveryResult
from .events import EventsTracker, EventsResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NSTResult:
    """News Sentiment Tracking result (from daily_signals)"""
    veto_active: bool = False
    veto_count: int = 0
    nci: float = 50.0       # Narrative Coherence Index
    npd: float = 0.0        # Narrative Polarity Drift
    burst_count: int = 0
    evi: float = 50.0       # Event Volatility Index
    thesis_balance: float = 0.0
    display: str = "Neutral"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContagionResult:
    """Hypergraph contagion result"""
    score: float = 50.0
    regime: str = "STABLE"
    stability: float = 0.5
    hyperedge_count: int = 10
    cross_pillar_ratio: float = 0.3
    data_age_hours: float = 0.0
    display: str = "Normal"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ARGUS1Result:
    """Complete ARGUS-1 calculation result"""
    # Timestamp
    date: str
    calculated_at: str
    
    # Core outputs
    regime: Regime
    authority: Authority
    confidence: Confidence
    
    # Component results
    amri: AMRIResult
    tti: TTIResult
    sac: SACResult
    nst: NSTResult
    contagion: ContagionResult
    fingerprint: FingerprintMatch
    rotation: RotationResult
    recovery: RecoveryResult
    events: EventsResult
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "date": self.date,
            "calculated_at": self.calculated_at,
            "regime": self.regime.value,
            "authority": self.authority.value,
            "confidence": self.confidence.value,
            "amri": self.amri.to_dict(),
            "tti": self.tti.to_dict(),
            "sac": self.sac.to_dict(),
            "nst": self.nst.to_dict(),
            "contagion": self.contagion.to_dict(),
            "fingerprint": self.fingerprint.to_dict(),
            "rotation": self.rotation.to_dict(),
            "recovery": self.recovery.to_dict(),
            "events": self.events.to_dict(),
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def format_summary(self) -> str:
        """Format human-readable summary"""
        lines = [
            "=" * 60,
            "ARGUS-1 REGIME ASSESSMENT",
            "=" * 60,
            f"Date: {self.date}",
            f"Regime: {self.regime.value}",
            f"Authority: {self.authority.value}",
            f"Confidence: {self.confidence.value}",
            "",
            "AMRI (Master Index)",
            f"  Composite: {self.amri.composite}",
            f"  Binding: {self.amri.binding_constraint}",
            f"  AMRI-S: {self.amri.decomposition.amri_s:.1f}",
            f"  AMRI-B: {self.amri.decomposition.amri_b:.1f}",
            f"  AMRI-C: {self.amri.decomposition.amri_c:.1f}",
            "",
            "TTI (Time-to-Instability)",
            f"  {self.tti.display}",
            f"  Rate: {self.tti.rate_of_change}/day",
            f"  Trajectory: {self.tti.trajectory}",
            "",
            "SAC (Shock Absorption)",
            f"  {self.sac.display}",
            f"  Weakest: {self.sac.weakest_link}",
            "",
            "NST (News Sentiment)",
            f"  VETO Active: {self.nst.veto_active}",
            f"  Thesis Balance: {self.nst.thesis_balance}",
            "",
            "Contagion",
            f"  Score: {self.contagion.score}",
            f"  Regime: {self.contagion.regime}",
            "",
            "Fingerprint",
            f"  {self.fingerprint.display}",
            "",
            "Rotation",
            f"  {self.rotation.display}",
            "",
            "Events",
            f"  {self.events.display}",
            "",
            "Recovery",
            f"  {self.recovery.display}",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def format_dashboard_cells(self) -> Dict[str, str]:
        """Format for Google Sheets dashboard compatibility"""
        return {
            DASHBOARD_CELLS["AMRI"]: str(self.amri.composite),
            DASHBOARD_CELLS["REGIME"]: self.regime.value,
            DASHBOARD_CELLS["AUTHORITY"]: self.authority.value,
            DASHBOARD_CELLS["CONFIDENCE"]: self.confidence.value,
            DASHBOARD_CELLS["AMRI_S"]: f"{self.amri.decomposition.amri_s:.1f}",
            DASHBOARD_CELLS["AMRI_B"]: f"{self.amri.decomposition.amri_b:.1f}",
            DASHBOARD_CELLS["AMRI_C"]: f"{self.amri.decomposition.amri_c:.1f}",
            DASHBOARD_CELLS["TTI"]: self.tti.display,
            DASHBOARD_CELLS["TTI_BINDING"]: self.tti.binding_threshold,
            DASHBOARD_CELLS["SAC"]: self.sac.display,
            DASHBOARD_CELLS["SAC_WEAKEST"]: self.sac.weakest_link,
            DASHBOARD_CELLS["CONTAGION"]: str(self.contagion.score),
            DASHBOARD_CELLS["CONTAGION_REGIME"]: self.contagion.regime,
            DASHBOARD_CELLS["STABILITY"]: f"{self.contagion.stability:.2f}",
            DASHBOARD_CELLS["HYPEREDGE_COUNT"]: str(self.contagion.hyperedge_count),
            DASHBOARD_CELLS["DATA_AGE"]: f"{self.contagion.data_age_hours:.1f}h",
            DASHBOARD_CELLS["NST"]: self.nst.display,
            DASHBOARD_CELLS["VETO_ACTIVE"]: "YES" if self.nst.veto_active else "NO",
            DASHBOARD_CELLS["NCI"]: f"{self.nst.nci:.1f}",
            DASHBOARD_CELLS["NPD"]: f"{self.nst.npd:.1f}",
            DASHBOARD_CELLS["BURST"]: str(self.nst.burst_count),
            DASHBOARD_CELLS["FINGERPRINT_EPISODE"]: self.fingerprint.episode,
            DASHBOARD_CELLS["FINGERPRINT_MATCH"]: f"{self.fingerprint.match_score:.0f}%",
            DASHBOARD_CELLS["FINGERPRINT_QUALITY"]: self.fingerprint.quality,
            DASHBOARD_CELLS["FINGERPRINT_PATTERN"]: self.fingerprint.pattern_type,
            DASHBOARD_CELLS["ROTATION"]: self.rotation.display,
            DASHBOARD_CELLS["LEADER"]: self.rotation.leader,
            DASHBOARD_CELLS["LAGGARD"]: self.rotation.laggard,
            DASHBOARD_CELLS["EVENT_STATUS"]: self.events.event_status,
            DASHBOARD_CELLS["NEXT_EVENT"]: self.events.next_event.name if self.events.next_event else "None",
            DASHBOARD_CELLS["DAYS_TO_EVENT"]: str(self.events.days_to_event),
            DASHBOARD_CELLS["RECOVERY"]: self.recovery.display,
        }


class ARGUS1Calculator:
    """
    Master calculator that orchestrates all analytical modules.
    """
    
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        if not self.client:
            self._init_client()
        
        # Initialize sub-calculators
        self.amri_calc = AMRICalculator(self.client)
        self.tti_calc = TTICalculator(self.client)
        self.sac_calc = SACCalculator(self.client)
        self.fingerprint_calc = FingerprintMatcher()
        self.rotation_calc = RotationTracker(self.client)
        self.recovery_calc = RecoveryDetector(self.client)
        self.events_calc = EventsTracker()
    
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
    
    def _fetch_nst_data(self) -> NSTResult:
        """Fetch NST data from daily_signals"""
        if not self.client:
            return NSTResult()
        
        try:
            result = self.client.table("daily_signals")\
                .select("*")\
                .order("date", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                d = result.data[0]
                veto_count = d.get("veto_triggers", 0)
                thesis = d.get("thesis_balance", 0)
                
                # Determine display
                if veto_count > 0:
                    display = f"⚠️ VETO ({veto_count})"
                elif thesis > 20:
                    display = "🟢 Bullish"
                elif thesis < -20:
                    display = "🔴 Bearish"
                else:
                    display = "🟡 Neutral"
                
                # VETO threshold: only trigger when count exceeds 25
                veto_threshold = 25
                return NSTResult(
                    veto_active=veto_count >= veto_threshold,
                    veto_count=veto_count,
                    nci=d.get("nci_score", 50),
                    npd=d.get("npd_score", 0),
                    burst_count=d.get("burst_count", 0),
                    evi=d.get("evi_score", 50),
                    thesis_balance=thesis,
                    display=display,
                )
        except Exception as e:
            logger.error(f"Error fetching NST data: {e}")
        
        return NSTResult()
    
    def _fetch_contagion_data(self) -> ContagionResult:
        """Fetch contagion data from hypergraph_signals"""
        if not self.client:
            return ContagionResult()
        
        try:
            result = self.client.table("hypergraph_signals")\
                .select("*")\
                .order("date", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                d = result.data[0]
                score = d.get("contagion_score", 50)
                
                # Calculate data age
                calc_at = d.get("calculated_at")
                if calc_at:
                    try:
                        calc_time = datetime.fromisoformat(calc_at.replace("Z", "+00:00"))
                        age = (datetime.now(calc_time.tzinfo) - calc_time).total_seconds() / 3600
                    except:
                        age = 0
                else:
                    age = 0
                
                # Determine display
                if score >= 70:
                    display = f"🔴 Contagion ({score:.0f})"
                elif score >= 50:
                    display = f"🟠 Elevated ({score:.0f})"
                else:
                    display = f"🟢 Normal ({score:.0f})"
                
                return ContagionResult(
                    score=score,
                    regime=d.get("regime", "STABLE"),
                    stability=d.get("stability_score", 0.5),
                    hyperedge_count=d.get("hyperedge_count", 10),
                    cross_pillar_ratio=d.get("cross_pillar_ratio", 0.3),
                    data_age_hours=age,
                    display=display,
                )
        except Exception as e:
            logger.error(f"Error fetching contagion data: {e}")
        
        return ContagionResult()
    
    def run(self, date: str = None) -> ARGUS1Result:
        """
        Run complete ARGUS-1 calculation.
        
        Args:
            date: Optional specific date. Defaults to latest data.
        
        Returns:
            ARGUS1Result with all components
        """
        calc_date = date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Running ARGUS-1 calculation for {calc_date}")
        
        # Calculate AMRI first (other calcs depend on it)
        amri = self.amri_calc.calculate(date)
        
        # Fetch raw data for dependent calculations
        nst = self._fetch_nst_data()
        contagion = self._fetch_contagion_data()
        
        # Calculate TTI with current AMRI
        tti = self.tti_calc.calculate(amri.composite)
        
        # Calculate SAC
        sac = self.sac_calc.calculate(amri.composite)
        
        # Fingerprint matching
        bubble_data = self.amri_calc._fetch_bubble_data(calc_date) if date else None
        current_vix = 18.0
        if bubble_data:
            current_vix = bubble_data.get("vix", 18.0)
        elif self.amri_calc.client:
            try:
                r = self.amri_calc.client.table("bubble_index_daily")\
                    .select("vix").order("date", desc=True).limit(1).execute()
                if r.data:
                    current_vix = r.data[0].get("vix", 18.0)
            except:
                pass
        
        fingerprint = self.fingerprint_calc.calculate(
            amri.composite,
            contagion.score,
            current_vix
        )
        
        # Rotation analysis
        rotation = self.rotation_calc.calculate()
        
        # Recovery detection
        recovery = self.recovery_calc.calculate(
            current_amri=amri.composite,
            amri_rate=tti.rate_of_change,
            current_vix=current_vix,
            veto_active=nst.veto_active
        )
        
        # Events calendar
        events = self.events_calc.calculate()
        
        return ARGUS1Result(
            date=calc_date,
            calculated_at=datetime.utcnow().isoformat(),
            regime=amri.regime,
            authority=amri.authority,
            confidence=amri.confidence,
            amri=amri,
            tti=tti,
            sac=sac,
            nst=nst,
            contagion=contagion,
            fingerprint=fingerprint,
            rotation=rotation,
            recovery=recovery,
            events=events,
        )
    
    def store_result(self, result: ARGUS1Result) -> bool:
        """Store result to Supabase argus_master_signals table"""
        if not self.client:
            logger.warning("No Supabase client, cannot store")
            return False
        
        try:
            record = {
                "date": result.date,
                "calculated_at": result.calculated_at,
                "regime": result.regime.value,
                "authority": result.authority.value,
                "confidence": result.confidence.value,
                "amri_composite": result.amri.composite,
                "amri_s": result.amri.decomposition.amri_s,
                "amri_b": result.amri.decomposition.amri_b,
                "amri_c": result.amri.decomposition.amri_c,
                "tti_display": result.tti.display,
                "tti_rate": result.tti.rate_of_change,
                "sac_composite": result.sac.composite,
                "sac_weakest": result.sac.weakest_link,
                "veto_active": result.nst.veto_active,
                "contagion_score": result.contagion.score,
                "fingerprint_episode": result.fingerprint.episode,
                "fingerprint_match": result.fingerprint.match_score,
                "rotation_leader": result.rotation.leader,
                "rotation_laggard": result.rotation.laggard,
                "next_event": result.events.next_event.name if result.events.next_event else None,
                "days_to_event": result.events.days_to_event,
                "full_result": result.to_dict(),
            }
            
            self.client.table("argus_master_signals").upsert(
                record,
                on_conflict="date"
            ).execute()
            
            logger.info(f"Stored ARGUS-1 result for {result.date}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing result: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    calculator = ARGUS1Calculator()
    result = calculator.run()
    
    print(result.format_summary())
    
    # Optionally store
    # calculator.store_result(result)
