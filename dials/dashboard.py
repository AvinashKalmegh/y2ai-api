"""
DASHBOARD
=========
Main aggregated view of all Y2AI/ARGUS-1 indicators.
Pulls from all dials and presents unified market assessment.

Sections:
1. Regime Summary - Current regime from Arbiter
2. Key Metrics - AMRI, Bubble Index, MCI, VIX
3. Pillar Status - 6 pillar signals
4. Dial Summary - All dial readings
5. Alerts - Active warnings and watch items
6. Recommendations - Action guidance
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DashboardData:
    """Complete dashboard data."""
    date: str
    timestamp: str
    
    # Regime Summary
    regime: str
    regime_confidence: str
    regime_driver: str
    
    # Key Metrics
    amri_score: float
    amri_regime: str
    bubble_index: float
    bubble_regime: str
    mci_score: float
    mci_regime: str
    vix_level: float
    vix_regime: str
    
    # Signal Counts
    bullish_count: int
    bearish_count: int
    neutral_count: int
    
    # Pillar Status
    pillar_signals: Dict[str, str]
    leading_pillars: List[str]
    lagging_pillars: List[str]
    
    # Dial Summary
    dial_readings: Dict[str, Dict]
    
    # Alerts
    alerts: List[str]
    watch_items: List[str]
    
    # Portfolio
    shadow_nav: float
    shadow_return: float
    equity_exposure: float
    
    # Recommendations
    recommendation: str
    action: str


# =============================================================================
# DASHBOARD GENERATOR
# =============================================================================

class DashboardGenerator:
    """
    Generate unified dashboard from all data sources.
    
    Usage:
        dashboard = DashboardGenerator()
        data = dashboard.generate()
        dashboard.save_to_supabase(data)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize generator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    def _get_latest(self, table: str, columns: str = "*") -> Optional[Dict]:
        """Get latest row from a table."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table(table) \
                .select(columns) \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.warning(f"Failed to get {table}: {e}")
        
        return None
    
    def generate(self) -> DashboardData:
        """Generate complete dashboard."""
        logger.info("Generating dashboard...")
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Fetch all data sources
        arbiter = self._get_latest("regime_arbiter_daily") or {}
        amri = self._get_latest("amri_daily") or {}
        bubble = self._get_latest("bubble_index_daily") or {}
        mci = self._get_latest("mci_daily") or {}
        vix = self._get_latest("vix_dial_daily") or {}
        pillar = self._get_latest("pillar_index_daily") or {}
        portfolio = self._get_latest("portfolio_nav_daily") or {}
        breadth = self._get_latest("breadth_daily") or {}
        credit = self._get_latest("credit_spread_daily") or {}
        cluster = self._get_latest("cluster_dial_daily") or {}
        liquidity = self._get_latest("liquidity_dial_daily") or {}
        
        # Regime Summary
        regime = arbiter.get("regime", "Unknown")
        regime_confidence = arbiter.get("confidence", "Unknown")
        regime_driver = arbiter.get("driver", "")
        
        # Key Metrics
        amri_score = amri.get("amri_score", 0)
        amri_regime = amri.get("regime", "Unknown")
        bubble_index = bubble.get("bubble_index", 0)
        bubble_regime = bubble.get("regime", "Unknown")
        mci_score = mci.get("mci_score", 0)
        mci_regime = mci.get("regime", "Unknown")
        vix_level = vix.get("vix", 0)
        vix_regime = vix.get("combined_regime", "Unknown")
        
        # Signal counts
        bullish_count = arbiter.get("bullish_count", 0)
        bearish_count = arbiter.get("bearish_count", 0)
        neutral_count = arbiter.get("neutral_count", 0)
        
        # Pillar signals
        pillar_signals = self._calculate_pillar_signals(pillar)
        leading_pillars = [p for p, s in pillar_signals.items() if s == "LEADING"]
        lagging_pillars = [p for p, s in pillar_signals.items() if s in ["WEAKENING", "STRESSED"]]
        
        # Dial readings
        dial_readings = {
            "AMRI": {"score": amri_score, "regime": amri_regime},
            "Bubble": {"score": bubble_index, "regime": bubble_regime},
            "MCI": {"score": mci_score, "regime": mci_regime},
            "VIX": {"level": vix_level, "regime": vix_regime},
            "Breadth": {"score": breadth.get("breadth_20d", 0) * 100, "regime": breadth.get("regime", "Unknown")},
            "Credit": {"regime": credit.get("combined_regime", "Unknown")},
            "Cluster": {"count": cluster.get("cluster_count", 0), "regime": cluster.get("regime", "Unknown")},
            "Liquidity": {"vol_of_vol": liquidity.get("vol_of_vol", 0), "regime": liquidity.get("regime", "Unknown")},
        }
        
        # Alerts
        alerts = []
        watch_items = arbiter.get("watch_list", []) or []
        
        if regime in ["FRAGILE", "BREAK"]:
            alerts.append(f"⚠️ REGIME ALERT: {regime}")
        if amri_score > 70:
            alerts.append(f"🔴 AMRI HIGH: {amri_score:.0f}")
        if mci_score < -30:
            alerts.append(f"🔴 MCI BEARISH: {mci_score:+.0f}")
        if vix_level > 30:
            alerts.append(f"⚠️ VIX ELEVATED: {vix_level:.1f}")
        
        # Portfolio
        shadow_nav = portfolio.get("nav", 1000000)
        shadow_return = portfolio.get("cumulative_return", 0) * 100
        equity_exposure = portfolio.get("equity_exposure", 1) * 100
        
        # Recommendations
        recommendation, action = self._get_recommendation(regime, amri_score, mci_score)
        
        result = DashboardData(
            date=date_str,
            timestamp=timestamp,
            regime=regime,
            regime_confidence=regime_confidence,
            regime_driver=regime_driver,
            amri_score=amri_score,
            amri_regime=amri_regime,
            bubble_index=bubble_index,
            bubble_regime=bubble_regime,
            mci_score=mci_score,
            mci_regime=mci_regime,
            vix_level=vix_level,
            vix_regime=vix_regime,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            pillar_signals=pillar_signals,
            leading_pillars=leading_pillars,
            lagging_pillars=lagging_pillars,
            dial_readings=dial_readings,
            alerts=alerts,
            watch_items=watch_items,
            shadow_nav=shadow_nav,
            shadow_return=shadow_return,
            equity_exposure=equity_exposure,
            recommendation=recommendation,
            action=action
        )
        
        logger.info(f"Dashboard generated: Regime={regime}")
        
        return result
    
    def _calculate_pillar_signals(self, pillar_data: Dict) -> Dict[str, str]:
        """Calculate pillar signals from momentum data."""
        signals = {}
        
        pillar_map = {
            "Infrastructure": ("infra_5d", "infra_1m", "infra_3m"),
            "Enterprise": ("enterprise_5d", "enterprise_1m", "enterprise_3m"),
            "Macro": ("macro_5d", "macro_1m", "macro_3m"),
            "Financial": ("financial_5d", "financial_1m", "financial_3m"),
            "Productivity": ("productivity_5d", "productivity_1m", "productivity_3m"),
            "Demand": ("demand_5d", "demand_1m", "demand_3m"),
        }
        
        for pillar, (m5d, m1m, m3m) in pillar_map.items():
            val_5d = pillar_data.get(m5d, 0) or 0
            val_1m = pillar_data.get(m1m, 0) or 0
            val_3m = pillar_data.get(m3m, 0) or 0
            
            if val_5d > 0 and val_1m > 0 and val_3m > 0:
                signals[pillar] = "LEADING"
            elif val_5d < 0 and val_1m < 0 and val_3m < 0:
                signals[pillar] = "STRESSED"
            elif val_5d < 0 or val_1m < 0:
                signals[pillar] = "WEAKENING"
            else:
                signals[pillar] = "NEUTRAL"
        
        return signals
    
    def _get_recommendation(self, regime: str, amri: float, mci: float) -> tuple:
        """Get recommendation based on current readings."""
        if regime in ["BREAK", "VETO_ALERT"]:
            return "Phase 5 Protocol", "De-risk fully, raise cash to 50%+, defensive only"
        elif regime == "FRAGILE":
            return "Defensive Posture", "Cut speculative positions, reduce to 30% equity"
        elif regime == "ELEVATED":
            return "Reduce Risk", "Trim winners, tighten stops, reduce to 50-70% equity"
        elif regime == "CAUTION":
            return "Monitor Closely", "No new speculative positions, prepare rotation plans"
        else:
            if mci > 30:
                return "Ride Momentum", "Stay long leaders, protect gains with trailing stops"
            else:
                return "Normal Operations", "Maintain positions, follow pillar signals"
    
    def save_to_supabase(self, data: DashboardData) -> bool:
        """Save dashboard to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("dashboard_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def print_dashboard(self, data: DashboardData):
        """Print formatted dashboard to console."""
        print("\n" + "="*70)
        print("Y2AI ARGUS-1 DASHBOARD")
        print(f"Generated: {data.timestamp}")
        print("="*70)
        
        # Regime Box
        print(f"\n┌{'─'*40}┐")
        print(f"│ REGIME: {data.regime:^30} │")
        print(f"│ Confidence: {data.regime_confidence:^27} │")
        print(f"└{'─'*40}┘")
        
        if data.regime_driver:
            print(f"Driver: {data.regime_driver}")
        
        # Key Metrics
        print(f"\n{'─'*40}")
        print("KEY METRICS")
        print(f"{'─'*40}")
        print(f"  AMRI:   {data.amri_score:5.1f}  ({data.amri_regime})")
        print(f"  Bubble: {data.bubble_index:5.1f}  ({data.bubble_regime})")
        print(f"  MCI:    {data.mci_score:+5.1f}  ({data.mci_regime})")
        print(f"  VIX:    {data.vix_level:5.1f}  ({data.vix_regime})")
        
        # Signal Summary
        print(f"\n{'─'*40}")
        print("SIGNAL SUMMARY")
        print(f"{'─'*40}")
        print(f"  🟢 Bullish: {data.bullish_count}")
        print(f"  🔴 Bearish: {data.bearish_count}")
        print(f"  🟡 Neutral: {data.neutral_count}")
        
        # Pillar Status
        print(f"\n{'─'*40}")
        print("PILLAR STATUS")
        print(f"{'─'*40}")
        for pillar, signal in data.pillar_signals.items():
            icon = "🟢" if signal == "LEADING" else "🔴" if signal in ["WEAKENING", "STRESSED"] else "🟡"
            print(f"  {icon} {pillar}: {signal}")
        
        # Portfolio
        print(f"\n{'─'*40}")
        print("SHADOW PORTFOLIO")
        print(f"{'─'*40}")
        print(f"  NAV: ${data.shadow_nav:,.0f}")
        print(f"  Return: {data.shadow_return:+.2f}%")
        print(f"  Equity: {data.equity_exposure:.0f}%")
        
        # Alerts
        if data.alerts:
            print(f"\n{'─'*40}")
            print("⚠️ ALERTS")
            print(f"{'─'*40}")
            for alert in data.alerts:
                print(f"  {alert}")
        
        # Recommendation
        print(f"\n{'─'*40}")
        print("RECOMMENDATION")
        print(f"{'─'*40}")
        print(f"  {data.recommendation}")
        print(f"  → {data.action}")
        
        print("\n" + "="*70)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ARGUS-1 Dashboard")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    generator = DashboardGenerator()
    data = generator.generate()
    
    if args.json:
        import json
        print(json.dumps(asdict(data), indent=2, default=str))
    else:
        generator.print_dashboard(data)
    
    if args.save:
        if generator.save_to_supabase(data):
            print("\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
