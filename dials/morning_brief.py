"""
MORNING BRIEF
=============
Generates daily market brief for Y2AI/ARGUS-1 subscribers.
Pulls from all dials and creates narrative summary.

Sections:
1. Regime Status - Current market regime and changes
2. Key Signals - Top bullish/bearish signals
3. Pillar Analysis - Leading and lagging sectors
4. Risk Assessment - Key risks and watch items
5. Action Items - Specific recommendations
6. Market Context - VIX, credit, breadth
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MorningBrief:
    """Morning brief data."""
    date: str
    
    # Header
    headline: str
    regime_status: str
    
    # Key Metrics Summary
    amri: float
    bubble_index: float
    mci: float
    vix: float
    
    # Regime Analysis
    regime_section: str
    
    # Signal Analysis
    top_bullish: List[str]
    top_bearish: List[str]
    signals_section: str
    
    # Pillar Analysis
    leading_pillars: List[str]
    lagging_pillars: List[str]
    pillar_section: str
    
    # Risk Assessment
    risk_level: str
    key_risks: List[str]
    risk_section: str
    
    # Action Items
    actions: List[str]
    action_section: str
    
    # Full Brief
    full_brief: str


# =============================================================================
# MORNING BRIEF GENERATOR
# =============================================================================

class MorningBriefGenerator:
    """
    Generate morning brief from all data sources.
    
    Usage:
        gen = MorningBriefGenerator()
        brief = gen.generate()
        print(brief.full_brief)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize generator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    def _get_latest(self, table: str) -> Optional[Dict]:
        """Get latest row from table."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table(table) \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return response.data[0]
        except:
            pass
        return None
    
    def _get_previous(self, table: str) -> Optional[Dict]:
        """Get previous day's row."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table(table) \
                .select("*") \
                .order("date", desc=True) \
                .limit(2) \
                .execute()
            
            if response.data and len(response.data) > 1:
                return response.data[1]
        except:
            pass
        return None
    
    def generate(self) -> MorningBrief:
        """Generate morning brief."""
        logger.info("Generating morning brief...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        weekday = datetime.now().strftime("%A")
        
        # Fetch data
        arbiter = self._get_latest("regime_arbiter_daily") or {}
        arbiter_prev = self._get_previous("regime_arbiter_daily") or {}
        amri = self._get_latest("amri_daily") or {}
        bubble = self._get_latest("bubble_index_daily") or {}
        mci = self._get_latest("mci_daily") or {}
        vix = self._get_latest("vix_dial_daily") or {}
        pillar = self._get_latest("pillar_index_daily") or {}
        portfolio = self._get_latest("portfolio_nav_daily") or {}

        # New signal data
        skew_alerts = self._get_signal_alerts("options_skew", "signal", ["EXTREME", "ELEVATED"])
        etf_alerts = self._get_signal_alerts("etf_flows", "flow_signal", ["SIGNIFICANT", "NOTABLE"])
        borrow_alerts = self._get_borrow_alerts()
        divergence_alerts = self._get_divergence_alerts()
        
        # Key metrics
        amri_score = amri.get("amri_score", 0)
        bubble_index = bubble.get("bubble_index", 0)
        mci_score = mci.get("mci_score", 0)
        vix_level = vix.get("vix", 0) or vix.get("vix_level", 0)
        
        # Regime
        regime = arbiter.get("regime", "Unknown")
        prev_regime = arbiter_prev.get("regime", "Unknown")
        regime_changed = regime != prev_regime
        
        # Headline
        headline = self._generate_headline(regime, mci_score, regime_changed, prev_regime)
        regime_status = f"Current Regime: {regime}"
        if regime_changed:
            regime_status += f" (changed from {prev_regime})"
        
        # Signals
        bullish = arbiter.get("bullish_signals", []) or []
        bearish = arbiter.get("bearish_signals", []) or []
        top_bullish = bullish[:3]
        top_bearish = bearish[:3]
        
        # Pillars
        pillar_signals = self._calculate_pillar_signals(pillar)
        leading = [p for p, s in pillar_signals.items() if s == "LEADING"]
        lagging = [p for p, s in pillar_signals.items() if s in ["WEAKENING", "STRESSED"]]
        
        # Risk assessment
        risk_level, key_risks = self._assess_risks(regime, amri_score, mci_score, vix_level, bearish)
        
        # Actions
        actions = self._generate_actions(regime, amri_score, mci_score, leading, lagging)
        
        # Build sections
        regime_section = self._build_regime_section(regime, regime_changed, prev_regime, amri_score, bubble_index, mci_score, vix_level)
        signals_section = self._build_signals_section(top_bullish, top_bearish, len(bullish), len(bearish))
        new_signals_section = self._build_new_signals_section(skew_alerts, etf_alerts, borrow_alerts, divergence_alerts)
        pillar_section = self._build_pillar_section(pillar_signals, leading, lagging)
        risk_section = self._build_risk_section(risk_level, key_risks)
        action_section = self._build_action_section(actions)

        # Full brief
        full_brief = self._compile_brief(
            date_str, weekday, headline, regime_section, signals_section,
            new_signals_section, pillar_section, risk_section, action_section
        )
        
        return MorningBrief(
            date=date_str,
            headline=headline,
            regime_status=regime_status,
            amri=amri_score,
            bubble_index=bubble_index,
            mci=mci_score,
            vix=vix_level,
            regime_section=regime_section,
            top_bullish=top_bullish,
            top_bearish=top_bearish,
            signals_section=signals_section,
            leading_pillars=leading,
            lagging_pillars=lagging,
            pillar_section=pillar_section,
            risk_level=risk_level,
            key_risks=key_risks,
            risk_section=risk_section,
            actions=actions,
            action_section=action_section,
            full_brief=full_brief
        )
    
    def _get_signal_alerts(self, table: str, signal_col: str, alert_values: List[str]) -> List[Dict]:
        """Get tickers with elevated signals from a table (latest date)."""
        if not self.supabase:
            return []
        try:
            result = self.supabase.table(table) \
                .select(f"ticker,{signal_col},date") \
                .in_(signal_col, alert_values) \
                .order("date", desc=True) \
                .limit(20) \
                .execute()
            if result.data:
                latest_date = result.data[0].get("date")
                return [r for r in result.data if r.get("date") == latest_date]
        except Exception:
            pass
        return []

    def _get_borrow_alerts(self) -> List[Dict]:
        """Get tickers with borrow_zscore > 2.5."""
        if not self.supabase:
            return []
        try:
            result = self.supabase.table("borrow_rates") \
                .select("ticker,borrow_rate_annualized,borrow_zscore,classification,date") \
                .in_("classification", ["HTB", "EXTREME"]) \
                .order("date", desc=True) \
                .limit(20) \
                .execute()
            if result.data:
                latest_date = result.data[0].get("date")
                return [r for r in result.data if r.get("date") == latest_date]
        except Exception:
            pass
        return []

    def _get_divergence_alerts(self) -> List[Dict]:
        """Get recent credit divergence events."""
        if not self.supabase:
            return []
        try:
            result = self.supabase.table("credit_divergence") \
                .select("ticker,divergence_type,severity,dm_smoothed,date") \
                .order("date", desc=True) \
                .limit(20) \
                .execute()
            if result.data:
                latest_date = result.data[0].get("date")
                return [r for r in result.data if r.get("date") == latest_date]
        except Exception:
            pass
        return []

    def _build_new_signals_section(self, skew_alerts, etf_alerts, borrow_alerts, divergence_alerts) -> str:
        """Build new signals alert section."""
        lines = ["NEW SIGNAL ALERTS", "-" * 40]

        # Options Skew
        if skew_alerts:
            extreme = [a for a in skew_alerts if a.get("signal") == "EXTREME"]
            elevated = [a for a in skew_alerts if a.get("signal") == "ELEVATED"]
            label = "EXTREME" if extreme else "ELEVATED"
            top3 = (extreme or elevated)[:3]
            names = ", ".join(a["ticker"] for a in top3)
            lines.append(f"  Options Skew:     [{label}] — {names}")
        else:
            lines.append(f"  Options Skew:     [NORMAL]")

        # ETF Flows
        if etf_alerts:
            top3 = etf_alerts[:3]
            moves = ", ".join(f"{a['ticker']}({a.get('flow_signal','')})" for a in top3)
            lines.append(f"  ETF Flow:         {moves}")
        else:
            lines.append(f"  ETF Flow:         [NORMAL]")

        # Borrow Rates
        if borrow_alerts:
            names = ", ".join(a["ticker"] for a in borrow_alerts[:3])
            lines.append(f"  Borrow Rate:      [ALERT] — {names}")
        else:
            lines.append(f"  Borrow Rate:      [NORMAL]")

        # Credit Divergence
        if divergence_alerts:
            names = ", ".join(f"{a['ticker']}({a.get('divergence_type','')[:20]})" for a in divergence_alerts[:3])
            lines.append(f"  Credit Divergence:{names}")
        else:
            lines.append(f"  Credit Divergence:[NONE]")

        return "\n".join(lines)

    def _generate_headline(self, regime: str, mci: float, changed: bool, prev_regime: str) -> str:
        """Generate headline based on current conditions."""
        if changed and regime in ["FRAGILE", "BREAK"]:
            return f"⚠️ REGIME SHIFT: {prev_regime} → {regime}"
        elif regime == "BREAK":
            return "🔴 BREAK CONDITIONS - Maximum Defense"
        elif regime == "FRAGILE":
            return "⚠️ FRAGILE REGIME - Elevated Caution"
        elif regime == "ELEVATED":
            return "🟡 ELEVATED RISK - Monitor Closely"
        elif mci > 30:
            return "🟢 EXTENSION MODE - Ride Momentum"
        elif mci < -30:
            return "🔴 COLLAPSE BIAS - Defensive Posture"
        else:
            return "🟡 NORMAL OPERATIONS - Stay Vigilant"
    
    def _calculate_pillar_signals(self, pillar_data: Dict) -> Dict[str, str]:
        """Calculate pillar signals."""
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
    
    def _assess_risks(self, regime: str, amri: float, mci: float, vix: float, bearish: List) -> tuple:
        """Assess risk level and key risks."""
        risks = []
        
        if regime in ["BREAK", "FRAGILE"]:
            level = "HIGH"
        elif regime == "ELEVATED" or amri > 60:
            level = "ELEVATED"
        elif len(bearish) > 5:
            level = "MODERATE"
        else:
            level = "LOW"
        
        if amri > 70:
            risks.append(f"AMRI at stress levels ({amri:.0f})")
        if mci < -20:
            risks.append(f"MCI showing collapse bias ({mci:+.0f})")
        if vix > 25:
            risks.append(f"VIX elevated ({vix:.1f})")
        if len(bearish) > 5:
            risks.append(f"Multiple bearish signals ({len(bearish)})")
        
        if not risks:
            risks.append("No major risk flags currently")
        
        return level, risks
    
    def _generate_actions(self, regime: str, amri: float, mci: float, leading: List, lagging: List) -> List[str]:
        """Generate specific action items."""
        actions = []
        
        if regime in ["BREAK", "FRAGILE"]:
            actions.append("Reduce equity exposure to 30% or less")
            actions.append("Close speculative positions")
            actions.append("Add defensive hedges")
        elif regime == "ELEVATED":
            actions.append("Trim winners, tighten stops")
            actions.append("Avoid new speculative positions")
            actions.append("Review stop-loss levels")
        elif mci > 30:
            actions.append("Maintain long positions in leading pillars")
            actions.append("Use trailing stops to protect gains")
        else:
            actions.append("Follow pillar signals for positioning")
            actions.append("Monitor for regime changes")
        
        if leading:
            actions.append(f"Focus exposure on: {', '.join(leading)}")
        if lagging:
            actions.append(f"Reduce/avoid: {', '.join(lagging)}")
        
        return actions
    
    def _build_regime_section(self, regime: str, changed: bool, prev: str, amri: float, bubble: float, mci: float, vix: float) -> str:
        """Build regime analysis section."""
        lines = ["REGIME STATUS", "-" * 40]
        
        if changed:
            lines.append(f"⚠️ Regime changed: {prev} → {regime}")
        else:
            lines.append(f"Current regime: {regime}")
        
        lines.append("")
        lines.append("Key Metrics:")
        lines.append(f"  AMRI: {amri:.1f}")
        lines.append(f"  Bubble Index: {bubble:.1f}")
        lines.append(f"  MCI: {mci:+.1f}")
        lines.append(f"  VIX: {vix:.1f}")
        
        return "\n".join(lines)
    
    def _build_signals_section(self, bullish: List, bearish: List, total_bull: int, total_bear: int) -> str:
        """Build signals section."""
        lines = ["SIGNAL ANALYSIS", "-" * 40]
        lines.append(f"Total: {total_bull} bullish, {total_bear} bearish")
        
        if bullish:
            lines.append("\nTop Bullish:")
            for sig in bullish:
                lines.append(f"  🟢 {sig}")
        
        if bearish:
            lines.append("\nTop Bearish:")
            for sig in bearish:
                lines.append(f"  🔴 {sig}")
        
        return "\n".join(lines)
    
    def _build_pillar_section(self, signals: Dict, leading: List, lagging: List) -> str:
        """Build pillar analysis section."""
        lines = ["PILLAR ANALYSIS", "-" * 40]
        
        for pillar, signal in signals.items():
            icon = "🟢" if signal == "LEADING" else "🔴" if signal in ["WEAKENING", "STRESSED"] else "🟡"
            lines.append(f"  {icon} {pillar}: {signal}")
        
        if leading:
            lines.append(f"\nLeading: {', '.join(leading)}")
        if lagging:
            lines.append(f"Lagging: {', '.join(lagging)}")
        
        return "\n".join(lines)
    
    def _build_risk_section(self, level: str, risks: List) -> str:
        """Build risk assessment section."""
        lines = ["RISK ASSESSMENT", "-" * 40]
        lines.append(f"Risk Level: {level}")
        lines.append("")
        
        for risk in risks:
            lines.append(f"  • {risk}")
        
        return "\n".join(lines)
    
    def _build_action_section(self, actions: List) -> str:
        """Build action items section."""
        lines = ["ACTION ITEMS", "-" * 40]
        
        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. {action}")
        
        return "\n".join(lines)
    
    def _compile_brief(self, date: str, weekday: str, headline: str, regime: str, signals: str, new_signals: str, pillars: str, risk: str, actions: str) -> str:
        """Compile full brief."""
        return f"""
{'='*60}
Y2AI MORNING BRIEF
{weekday}, {date}
{'='*60}

{headline}

{regime}

{signals}

{new_signals}

{pillars}

{risk}

{actions}

{'='*60}
This brief is generated by ARGUS-1 Market Intelligence System.
Y2AI Research | https://y2ai.substack.com
{'='*60}
"""
    
    def save_to_supabase(self, brief: MorningBrief) -> bool:
        """Save brief to Supabase."""
        if not self.supabase:
            return False
        
        row = {
            "date": brief.date,
            "headline": brief.headline,
            "regime_status": brief.regime_status,
            "amri": brief.amri,
            "bubble_index": brief.bubble_index,
            "mci": brief.mci,
            "vix": brief.vix,
            "risk_level": brief.risk_level,
            "key_risks": brief.key_risks,
            "actions": brief.actions,
            "leading_pillars": brief.leading_pillars,
            "lagging_pillars": brief.lagging_pillars,
            "full_brief": brief.full_brief
        }
        
        try:
            self.supabase.table("morning_brief_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Morning Brief Generator")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--sections", action="store_true", help="Show individual sections")
    args = parser.parse_args()
    
    gen = MorningBriefGenerator()
    brief = gen.generate()
    
    if args.sections:
        print("\n" + brief.regime_section)
        print("\n" + brief.signals_section)
        print("\n" + brief.pillar_section)
        print("\n" + brief.risk_section)
        print("\n" + brief.action_section)
    else:
        print(brief.full_brief)
    
    if args.save:
        if gen.save_to_supabase(brief):
            print("\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
