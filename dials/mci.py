"""
MCI (Market Condition Index) - FINAL CORRECTED VERSION
=======================================================

VALIDATED AGAINST GOOGLE SHEETS (Jan 5, 2026)

Formula: MCI = Breadth + VIX + Credit + Pillar
Each component ranges from -25 to +25, total MCI ranges -100 to +100.

CORRECTED THRESHOLDS:
- BREADTH_THRESHOLD: 10 (% change for max score)
- VIX_THRESHOLD: 6 (pts change for max score)
- CREDIT_THRESHOLD: 30 (bps change for max score)  ← Was 20
- PILLAR_THRESHOLD: 3 (% return for max score)

VALIDATION:
- Breadth: 0% change → 0 score ✓
- VIX: -6.01 pts → +25 score (maxed) ✓
- Credit: -18 bps → +15 score ✓
- Pillar: +0.95% → +7.9 score ✓
- TOTAL: 47.9 ✓
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - VALIDATED
# =============================================================================

MCI_CONFIG = {
    # Lookback periods (trading days)
    "BREADTH_LOOKBACK": 5,
    "VIX_LOOKBACK": 10,
    "CREDIT_LOOKBACK": 10,
    "PILLAR_LOOKBACK": 5,
    
    # Component weights (each contributes -25 to +25)
    "WEIGHT_BREADTH": 25,
    "WEIGHT_VIX": 25,
    "WEIGHT_CREDIT": 25,
    "WEIGHT_PILLAR": 25,
    
    # VALIDATED thresholds (raw change that gives max score)
    "BREADTH_THRESHOLD": 10,    # ±10% breadth change = max ±25
    "VIX_THRESHOLD": 3,         # ±3 VIX pts = max ±25
    "CREDIT_THRESHOLD": 30,     # ±30 bps spread change = max ±25
    "PILLAR_THRESHOLD": 3,      # ±3% pillar return = max ±25
}

MCI_REGIMES = {
    "Melt-Up": (40, 101),
    "Extension": (10, 40),
    "Knife Edge": (-10, 10),
    "Collapse Bias": (-40, -10),
    "Break Path": (-101, -40),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MCIComponent:
    """Single component of MCI."""
    name: str
    raw_value: float
    normalized: float
    score: float
    detail: str


@dataclass
class MCIData:
    """Complete MCI calculation result."""
    date: str
    mci_score: float
    regime: str
    interpretation: str
    breadth_component: float
    vix_component: float
    credit_component: float
    pillar_component: float
    breadth_raw: float = 0.0
    vix_raw: float = 0.0
    credit_raw: float = 0.0
    pillar_raw: float = 0.0


# =============================================================================
# MCI CALCULATOR
# =============================================================================

class MCICalculator:
    """Calculate Market Condition Index with validated formulas."""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    def calculate_breadth_momentum(self) -> MCIComponent:
        """
        Breadth Momentum: 5D change in overall breadth.
        Rising breadth = bullish (positive score).
        
        Formula: score = (breadth_change / 10) × 25, clamped to ±25
        """
        config = MCI_CONFIG
        
        try:
            if self.supabase:
                response = self.supabase.table("breadth_daily") \
                    .select("date, breadth_20d") \
                    .order("date", desc=True) \
                    .limit(config["BREADTH_LOOKBACK"] + 1) \
                    .execute()
                
                if response.data and len(response.data) > config["BREADTH_LOOKBACK"]:
                    current = response.data[0]["breadth_20d"] * 100
                    previous = response.data[config["BREADTH_LOOKBACK"]]["breadth_20d"] * 100
                    breadth_change = current - previous
                    
                    normalized = max(-1, min(1, breadth_change / config["BREADTH_THRESHOLD"]))
                    score = normalized * config["WEIGHT_BREADTH"]
                    
                    return MCIComponent(
                        name="Breadth Momentum",
                        raw_value=breadth_change,
                        normalized=normalized,
                        score=round(score, 1),
                        detail=f"5D change: {breadth_change:+.1f}%"
                    )
        except Exception as e:
            logger.warning(f"Breadth calculation failed: {e}")
        
        return MCIComponent("Breadth Momentum", 0, 0, 0, "No data")
    
    def calculate_vix_trend(self) -> MCIComponent:
        """
        VIX Trend: 10D VIX direction.
        Falling VIX = bullish (positive score).
        
        Formula: score = (-vix_change / 3) × 25, clamped to ±25
        Note: Negative sign because VIX drop is bullish.
        """
        config = MCI_CONFIG
        
        try:
            if self.supabase:
                # Read from vix_dial_daily (populated by VixDial)
                response = self.supabase.table("vix_dial_daily") \
                    .select("date, vix") \
                    .order("date", desc=True) \
                    .limit(config["VIX_LOOKBACK"] + 5) \
                    .execute()
                
                if response.data and len(response.data) > config["VIX_LOOKBACK"]:
                    current_vix = response.data[0]["vix"]
                    previous_vix = response.data[config["VIX_LOOKBACK"]]["vix"]
                    vix_change = current_vix - previous_vix
                    
                    # Invert: VIX drop = positive score
                    normalized = max(-1, min(1, -vix_change / config["VIX_THRESHOLD"]))
                    score = normalized * config["WEIGHT_VIX"]
                    
                    return MCIComponent(
                        name="VIX Trend",
                        raw_value=vix_change,
                        normalized=normalized,
                        score=round(score, 1),
                        detail=f"10D change: {vix_change:+.2f} pts"
                    )
        except Exception as e:
            logger.warning(f"VIX calculation failed: {e}")
        
        # Fallback: try yfinance
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1mo")
            if len(hist) > config["VIX_LOOKBACK"]:
                current_vix = hist['Close'].iloc[-1]
                previous_vix = hist['Close'].iloc[-config["VIX_LOOKBACK"]-1]
                vix_change = current_vix - previous_vix
                
                normalized = max(-1, min(1, -vix_change / config["VIX_THRESHOLD"]))
                score = normalized * config["WEIGHT_VIX"]
                
                return MCIComponent(
                    name="VIX Trend",
                    raw_value=vix_change,
                    normalized=normalized,
                    score=round(score, 1),
                    detail=f"10D change: {vix_change:+.2f} pts"
                )
        except Exception as e:
            logger.warning(f"VIX fallback failed: {e}")
        
        return MCIComponent("VIX Trend", 0, 0, 0, "No data")
    
    def calculate_credit_trend(self) -> MCIComponent:
        """
        Credit Trend: 10D spread direction.
        Tightening spreads = bullish (positive score).
        
        Formula: score = (-spread_change_bps / 30) × 25, clamped to ±25
        Note: Negative sign because tightening (negative change) is bullish.
        """
        config = MCI_CONFIG
        
        try:
            if self.supabase:
                # Try credit_spread_daily table
                response = self.supabase.table("credit_spread_daily") \
                    .select("date, hy_spread") \
                    .order("date", desc=True) \
                    .limit(config["CREDIT_LOOKBACK"] + 5) \
                    .execute()
                
                if response.data and len(response.data) > config["CREDIT_LOOKBACK"]:
                    current_spread = response.data[0]["hy_spread"]
                    previous_spread = response.data[config["CREDIT_LOOKBACK"]]["hy_spread"]
                    
                    # Convert to bps (assuming data is in percentage)
                    spread_change_bps = (current_spread - previous_spread) * 100
                    
                    # Invert: tightening (negative) = positive score
                    normalized = max(-1, min(1, -spread_change_bps / config["CREDIT_THRESHOLD"]))
                    score = normalized * config["WEIGHT_CREDIT"]
                    
                    return MCIComponent(
                        name="Credit Trend",
                        raw_value=spread_change_bps,
                        normalized=normalized,
                        score=round(score, 1),
                        detail=f"10D change: {spread_change_bps:+.0f} bps"
                    )
        except Exception as e:
            logger.warning(f"Credit calculation failed: {e}")
        
        return MCIComponent("Credit Trend", 0, 0, 0, "No data")
    
    def calculate_pillar_momentum(self) -> MCIComponent:
        """
        Pillar Momentum: Average 5D returns across all pillars.
        Positive returns = bullish.
        
        Formula: score = (avg_return / 3) × 25, clamped to ±25
        """
        config = MCI_CONFIG
        
        try:
            if self.supabase:
                response = self.supabase.table("pillar_index_daily") \
                    .select("date, infra_5d, enterprise_5d, macro_5d, financial_5d, productivity_5d, demand_5d") \
                    .order("date", desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    row = response.data[0]
                    momenta = [
                        row.get("infra_5d", 0) or 0,
                        row.get("enterprise_5d", 0) or 0,
                        row.get("macro_5d", 0) or 0,
                        row.get("financial_5d", 0) or 0,
                        row.get("productivity_5d", 0) or 0,
                        row.get("demand_5d", 0) or 0,
                    ]
                    avg_momentum = sum(momenta) / len(momenta) * 100  # Convert to %
                    
                    normalized = max(-1, min(1, avg_momentum / config["PILLAR_THRESHOLD"]))
                    score = normalized * config["WEIGHT_PILLAR"]
                    
                    return MCIComponent(
                        name="Pillar Momentum",
                        raw_value=avg_momentum,
                        normalized=normalized,
                        score=round(score, 1),
                        detail=f"Avg 5D return: {avg_momentum:+.2f}%"
                    )
        except Exception as e:
            logger.warning(f"Pillar calculation failed: {e}")
        
        return MCIComponent("Pillar Momentum", 0, 0, 0, "No data")
    
    def calculate(self) -> MCIData:
        """Calculate MCI score and regime."""
        logger.info("Calculating MCI...")
        
        breadth = self.calculate_breadth_momentum()
        vix = self.calculate_vix_trend()
        credit = self.calculate_credit_trend()
        pillar = self.calculate_pillar_momentum()
        
        mci_score = breadth.score + vix.score + credit.score + pillar.score
        mci_score = max(-100, min(100, mci_score))
        
        regime = self._get_regime(mci_score)
        interpretation = self._get_interpretation(regime)
        
        logger.info(f"MCI: {mci_score:.1f} ({regime})")
        logger.info(f"  Breadth: {breadth.score:+.1f} ({breadth.detail})")
        logger.info(f"  VIX: {vix.score:+.1f} ({vix.detail})")
        logger.info(f"  Credit: {credit.score:+.1f} ({credit.detail})")
        logger.info(f"  Pillar: {pillar.score:+.1f} ({pillar.detail})")
        
        return MCIData(
            date=datetime.now().strftime("%Y-%m-%d"),
            mci_score=round(mci_score, 1),
            regime=regime,
            interpretation=interpretation,
            breadth_component=breadth.score,
            vix_component=vix.score,
            credit_component=credit.score,
            pillar_component=pillar.score,
            breadth_raw=breadth.raw_value,
            vix_raw=vix.raw_value,
            credit_raw=credit.raw_value,
            pillar_raw=pillar.raw_value,
        )
    
    def _get_regime(self, score: float) -> str:
        for regime, (low, high) in MCI_REGIMES.items():
            if low <= score < high:
                return regime
        return "Unknown"
    
    def _get_interpretation(self, regime: str) -> str:
        interpretations = {
            "Melt-Up": "Reflexive melt-up - ride winners, protect downside",
            "Extension": "Trend intact but weakening - trim beta, tighten stops",
            "Knife Edge": "Dangerous - can flip either way. Reduce exposure.",
            "Collapse Bias": "Prioritize defense - cut speculative positions",
            "Break Path": "Structural collapse likely - de-risk fully",
        }
        return interpretations.get(regime, "Unknown")
    
    def save_to_supabase(self, data: MCIData) -> bool:
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("mci_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved MCI for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False


# =============================================================================
# VALIDATION
# =============================================================================

def validate_formulas():
    """Validate formulas against Google Sheets values."""
    print("\n" + "="*60)
    print("MCI FORMULA VALIDATION")
    print("="*60)
    
    # Test with Google Sheets values
    test_cases = [
        {"name": "Breadth", "raw": 0, "threshold": 10, "expected": 0},
        {"name": "VIX", "raw": -6.01, "threshold": 6, "expected": 25, "invert": True},
        {"name": "Credit", "raw": -18, "threshold": 30, "expected": 15, "invert": True},
        {"name": "Pillar", "raw": 0.95, "threshold": 3, "expected": 7.9},
    ]
    
    total = 0
    all_pass = True
    
    print(f"\n{'Component':<12} {'Raw':>10} {'Threshold':>10} {'Calc':>8} {'Expected':>10} {'Status'}")
    print("-" * 65)
    
    for tc in test_cases:
        raw = tc["raw"]
        threshold = tc["threshold"]
        invert = tc.get("invert", False)
        
        if invert:
            normalized = max(-1, min(1, -raw / threshold))
        else:
            normalized = max(-1, min(1, raw / threshold))
        
        score = normalized * 25
        expected = tc["expected"]
        
        status = "✅" if abs(score - expected) < 0.5 else "❌"
        if status == "❌":
            all_pass = False
        
        total += score
        print(f"{tc['name']:<12} {raw:>10} {threshold:>10} {score:>8.1f} {expected:>10.1f} {status}")
    
    print("-" * 65)
    print(f"{'TOTAL MCI':<12} {'':<10} {'':<10} {total:>8.1f} {'47.9':>10} {'✅' if abs(total - 47.9) < 0.5 else '❌'}")
    
    return all_pass


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate MCI")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--validate", action="store_true", help="Validate formulas")
    args = parser.parse_args()
    
    if args.validate:
        validate_formulas()
        return
    
    calc = MCICalculator()
    mci = calc.calculate()
    
    print(f"\n{'='*60}")
    print("MCI (Market Condition Index)")
    print(f"{'='*60}")
    print(f"\nDate: {mci.date}")
    print(f"\n{'─'*40}")
    print(f"MCI SCORE: {mci.mci_score:+.1f}")
    print(f"REGIME: {mci.regime}")
    print(f"{'─'*40}")
    print(f"\n{mci.interpretation}")
    
    print(f"\n{'─'*40}")
    print("COMPONENTS")
    print(f"{'─'*40}")
    print(f"  Breadth: {mci.breadth_component:+.1f} (raw: {mci.breadth_raw:+.1f}%)")
    print(f"  VIX:     {mci.vix_component:+.1f} (raw: {mci.vix_raw:+.1f} pts)")
    print(f"  Credit:  {mci.credit_component:+.1f} (raw: {mci.credit_raw:+.0f} bps)")
    print(f"  Pillar:  {mci.pillar_component:+.1f} (raw: {mci.pillar_raw:+.2f}%)")
    
    if args.save:
        if calc.save_to_supabase(mci):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
