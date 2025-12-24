"""
Pillar Rotation Tracker

Tracks leadership rotation across the 6 AI infrastructure pillars.
Identifies which pillars are leading/lagging and rotation patterns.
"""

import os
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .config import PILLARS, TICKER_TO_PILLAR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PillarPerformance:
    """Performance data for a single pillar"""
    name: str
    change_1d: float
    change_5d: float
    change_ytd: float
    rank: int  # 1 = best, 6 = worst


@dataclass
class RotationResult:
    """Pillar rotation analysis result"""
    leader: str
    laggard: str
    leader_change: float
    laggard_change: float
    spread: float  # Leader - laggard
    pattern: str   # BROADENING, NARROWING, ROTATING, STABLE
    all_pillars: List[PillarPerformance]
    display: str
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['all_pillars'] = [asdict(p) for p in self.all_pillars]
        return result


class RotationTracker:
    """Track pillar leadership rotation"""
    
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
    
    def _fetch_pillar_data(self) -> Optional[List[Dict]]:
        """Fetch pillar performance from stock_tracker_daily"""
        if not self.client:
            return None
        
        try:
            result = self.client.table("stock_tracker_daily")\
                .select("pillars, date")\
                .order("date", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data and result.data[0].get("pillars"):
                return result.data[0]["pillars"]
        except Exception as e:
            logger.error(f"Error fetching pillar data: {e}")
        
        return None
    
    def calculate(self) -> RotationResult:
        """Calculate rotation analysis"""
        pillar_data = self._fetch_pillar_data()
        
        if not pillar_data:
            return RotationResult(
                leader="UNKNOWN",
                laggard="UNKNOWN",
                leader_change=0,
                laggard_change=0,
                spread=0,
                pattern="UNKNOWN",
                all_pillars=[],
                display="No data",
            )
        
        # Parse pillar data
        performances = []
        for p in pillar_data:
            if isinstance(p, dict):
                performances.append({
                    "name": p.get("name", p.get("pillar", "Unknown")),
                    "change_1d": p.get("change_today", 0),
                    "change_5d": p.get("change_5day", 0),
                    "change_ytd": p.get("change_ytd", 0),
                })
        
        if not performances:
            return RotationResult(
                leader="UNKNOWN",
                laggard="UNKNOWN",
                leader_change=0,
                laggard_change=0,
                spread=0,
                pattern="UNKNOWN",
                all_pillars=[],
                display="No data",
            )
        
        # Sort by 5-day performance
        performances.sort(key=lambda x: x["change_5d"], reverse=True)
        
        # Create ranked list
        all_pillars = []
        for i, p in enumerate(performances):
            all_pillars.append(PillarPerformance(
                name=p["name"],
                change_1d=p["change_1d"],
                change_5d=p["change_5d"],
                change_ytd=p["change_ytd"],
                rank=i + 1,
            ))
        
        leader = all_pillars[0]
        laggard = all_pillars[-1]
        spread = leader.change_5d - laggard.change_5d
        
        # Determine pattern
        if spread > 0.15:
            pattern = "NARROWING"  # Leaders pulling away
        elif spread < 0.05:
            pattern = "BROADENING"  # Participation expanding
        else:
            pattern = "STABLE"
        
        display = f"📈 {leader.name} / 📉 {laggard.name}"
        
        return RotationResult(
            leader=leader.name,
            laggard=laggard.name,
            leader_change=round(leader.change_5d * 100, 1),
            laggard_change=round(laggard.change_5d * 100, 1),
            spread=round(spread * 100, 1),
            pattern=pattern,
            all_pillars=all_pillars,
            display=display,
        )


# =============================================================================
# RECOVERY SIGNAL DETECTOR
# =============================================================================

@dataclass
class RecoveryResult:
    """Recovery signal detection result"""
    active: bool
    strength: float  # 0-100
    days_since_trough: int
    conditions_met: List[str]
    display: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class RecoveryDetector:
    """
    Detect recovery signals - the opposite of TTI.
    Identifies when conditions are improving and safe to re-engage.
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
    
    def calculate(
        self,
        current_amri: float = 50.0,
        amri_rate: float = 0.0,
        current_vix: float = 18.0,
        veto_active: bool = False
    ) -> RecoveryResult:
        """
        Detect recovery signals.
        
        Recovery conditions:
        1. AMRI declining (rate < 0)
        2. AMRI below danger zone (<70)
        3. VIX normalizing (<25)
        4. No active VETO signals
        """
        conditions = []
        strength = 0
        
        # Check conditions
        if amri_rate < -0.5:
            conditions.append("AMRI_FALLING")
            strength += 25
        
        if current_amri < 70:
            conditions.append("AMRI_SAFE")
            strength += 25
        
        if current_vix < 25:
            conditions.append("VIX_NORMAL")
            strength += 25
        
        if not veto_active:
            conditions.append("NO_VETO")
            strength += 25
        
        active = len(conditions) >= 3
        
        if active:
            display = f"🟢 Recovery ({strength}%)"
        elif len(conditions) >= 2:
            display = f"🟡 Improving ({strength}%)"
        else:
            display = "⚪ No recovery signal"
        
        return RecoveryResult(
            active=active,
            strength=strength,
            days_since_trough=0,  # Would need history to calculate
            conditions_met=conditions,
            display=display,
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Rotation
    rotation = RotationTracker()
    rot_result = rotation.calculate()
    
    print(f"\n{'='*60}")
    print("ROTATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Leader: {rot_result.leader} ({rot_result.leader_change}%)")
    print(f"Laggard: {rot_result.laggard} ({rot_result.laggard_change}%)")
    print(f"Spread: {rot_result.spread}%")
    print(f"Pattern: {rot_result.pattern}")
    
    # Recovery
    recovery = RecoveryDetector()
    rec_result = recovery.calculate(current_amri=45, amri_rate=-1.2)
    
    print(f"\n{'='*60}")
    print("RECOVERY DETECTION")
    print(f"{'='*60}")
    print(f"Active: {rec_result.active}")
    print(f"Strength: {rec_result.strength}%")
    print(f"Conditions: {rec_result.conditions_met}")
    print(f"Display: {rec_result.display}")
