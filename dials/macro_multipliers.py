"""
MACRO MULTIPLIERS
=================
Applies macro regime multipliers to position sizing.
Final_Weight = Base_Weight × Corr_Mult × VIX_Mult × Credit_Mult × Breadth_Mult × Labor_Mult

With 30% floor to maintain minimum market exposure.

Multiplier Tables:
- Correlation: Healthy=1.00, Caution=0.90, Fragile=0.75, Stressed=0.50, Crisis=0.25
- VIX: Healthy=1.00, Caution=0.90, Fragile=0.75, Stressed=0.50, Crisis=0.25, Pre-Shock=0.80
- Credit: Healthy=1.00, Caution=0.85, Fragile=0.65, Stressed=0.45, Crisis=0.25
- Breadth: Healthy=1.00, Caution=0.90, Fragile=0.80, Stressed=0.60
- Labor: Healthy=1.00, Caution=0.90, Fragile=0.75, Crisis=0.50

Combined multiplier has 30% floor (MIN_MULTIPLIER = 0.30)
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MULTIPLIER TABLES
# =============================================================================

CORR_MULTIPLIERS = {
    "Healthy": 1.00,
    "Caution": 0.90,
    "Fragile": 0.75,
    "Stressed": 0.50,
    "Crisis": 0.25,
    "Collapse": 0.25,   # Legacy label
    "Unknown": 0.85
}

VIX_MULTIPLIERS = {
    "Healthy": 1.00,
    "Calm": 1.00,       # Legacy label
    "Caution": 0.90,
    "Normal": 0.90,     # Legacy label
    "Fragile": 0.75,
    "Watch": 0.75,      # Legacy label
    "Stressed": 0.50,
    "Elevated": 0.50,   # Legacy label
    "Crisis": 0.25,
    "Panic": 0.25,      # Legacy label
    "Pre-Shock": 0.80,  # Special: compression warning
    "Unknown": 0.85
}

CREDIT_MULTIPLIERS = {
    "Healthy": 1.00,
    "Normal": 1.00,     # Legacy label
    "Caution": 0.85,
    "Watch": 0.85,      # Legacy label
    "Fragile": 0.65,
    "Stress": 0.65,     # Legacy label
    "Stressed": 0.45,
    "Panic": 0.45,      # Legacy label
    "Crisis": 0.25,
    "Unknown": 0.85
}

BREADTH_MULTIPLIERS = {
    "Healthy": 1.00,
    "Caution": 0.90,
    "Fragile": 0.80,
    "Stressed": 0.60,
    "Unknown": 0.85
}

LABOR_MULTIPLIERS = {
    "Healthy": 1.00,
    "Caution": 0.90,
    "Fragile": 0.75,
    "Crisis": 0.50,
    "Unknown": 0.90
}

# Minimum combined multiplier (floor)
MIN_MULTIPLIER = 0.30

# Institutional exit signature modifier (DevBrief: Options Skew EXTREME + DM declining + Borrow elevated)
INST_EXIT_REDUCTION = 0.75  # reduce position by 25%


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MultiplierData:
    """Macro multiplier data."""
    date: str
    # Individual multipliers
    corr_regime: str
    corr_mult: float
    vix_regime: str
    vix_value: float
    vix_mult: float
    credit_regime: str
    credit_mult: float
    breadth_regime: str
    breadth_value: float
    breadth_mult: float
    labor_regime: str
    labor_mult: float
    # Combined
    raw_multiplier: float
    final_multiplier: float
    floor_applied: bool
    # Institutional exit signature
    inst_exit_tickers: list = None
    inst_exit_applied: bool = False
    # Interpretation
    interpretation: str = ""


# =============================================================================
# MULTIPLIER CALCULATOR
# =============================================================================

class MacroMultiplierCalculator:
    """
    Calculate combined macro multiplier for position sizing.
    
    Usage:
        calc = MacroMultiplierCalculator()
        data = calc.calculate()
        print(f"Final multiplier: {data.final_multiplier}")
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
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
        except Exception as e:
            logger.warning(f"Failed to fetch {table}: {e}")
        
        return None
    
    def get_correlation_regime(self) -> tuple:
        """Get correlation regime and multiplier."""
        data = self._get_latest("correlation_daily")
        
        if data:
            regime = data.get("regime", "Unknown")
        else:
            regime = "Unknown"
        
        mult = CORR_MULTIPLIERS.get(regime, 0.85)
        return regime, mult
    
    def get_vix_regime(self) -> tuple:
        """Get VIX regime, value, and multiplier."""
        data = self._get_latest("vix_dial_daily")
        
        if data:
            regime = data.get("combined_regime", data.get("regime", "Unknown"))
            vix_value = data.get("vix", 0)
        else:
            regime = "Unknown"
            vix_value = 0
        
        mult = VIX_MULTIPLIERS.get(regime, 0.85)
        return regime, vix_value, mult
    
    def get_credit_regime(self) -> tuple:
        """Get credit regime and multiplier."""
        data = self._get_latest("credit_spread_daily")
        
        if data:
            regime = data.get("combined_regime", "Unknown")
        else:
            regime = "Unknown"
        
        mult = CREDIT_MULTIPLIERS.get(regime, 0.85)
        return regime, mult
    
    def get_breadth_regime(self) -> tuple:
        """Get breadth regime, value, and multiplier."""
        data = self._get_latest("breadth_daily")
        
        if data:
            regime = data.get("regime", "Unknown")
            breadth_value = data.get("breadth_20d", 0)
            if isinstance(breadth_value, (int, float)):
                breadth_value = breadth_value * 100  # Convert to percentage
        else:
            regime = "Unknown"
            breadth_value = 0
        
        mult = BREADTH_MULTIPLIERS.get(regime, 0.85)
        return regime, breadth_value, mult
    
    def get_institutional_exit_tickers(self) -> list:
        """
        Detect institutional exit signature:
        Options Skew EXTREME + DM declining + Borrow rate elevated on same ticker.
        Returns list of tickers matching this pattern.
        """
        if not self.supabase:
            return []

        try:
            # Get tickers with EXTREME options skew (latest date)
            skew_result = self.supabase.table("options_skew") \
                .select("ticker,date") \
                .eq("signal", "EXTREME") \
                .order("date", desc=True) \
                .limit(50) \
                .execute()

            if not skew_result.data:
                return []

            latest_date = skew_result.data[0].get("date")
            extreme_tickers = set(
                r["ticker"] for r in skew_result.data if r.get("date") == latest_date
            )

            if not extreme_tickers:
                return []

            # Get DM scores for these tickers — declining = dm_smoothed < 50
            dm_result = self.supabase.table("dm_history") \
                .select("ticker,dm_smoothed") \
                .eq("date", latest_date) \
                .in_("ticker", list(extreme_tickers)) \
                .execute()

            declining_tickers = set(
                r["ticker"] for r in (dm_result.data or [])
                if r.get("dm_smoothed") is not None and r["dm_smoothed"] < 50
            )

            if not declining_tickers:
                return []

            # Get borrow rates for these tickers — elevated = HTB or EXTREME
            borrow_result = self.supabase.table("borrow_rates") \
                .select("ticker,classification") \
                .in_("ticker", list(declining_tickers)) \
                .in_("classification", ["ELEVATED", "HTB", "EXTREME"]) \
                .order("date", desc=True) \
                .limit(50) \
                .execute()

            if not borrow_result.data:
                return []

            borrow_latest = borrow_result.data[0].get("date") if borrow_result.data else None
            elevated_tickers = set(
                r["ticker"] for r in borrow_result.data if r.get("date") == borrow_latest
            )

            # Intersection: all three signals on same ticker
            exit_tickers = list(declining_tickers & elevated_tickers)

            if exit_tickers:
                logger.warning(
                    f"INSTITUTIONAL EXIT SIGNATURE detected on {len(exit_tickers)} tickers: "
                    f"{', '.join(exit_tickers)} — applying 25% size reduction"
                )

            return exit_tickers

        except Exception as e:
            logger.warning(f"Failed to check institutional exit: {e}")
            return []

    def get_labor_regime(self) -> tuple:
        """Get labor regime and multiplier."""
        data = self._get_latest("labor_dial_daily")
        
        if data:
            regime = data.get("combined_regime", "Unknown")
        else:
            regime = "Unknown"
        
        mult = LABOR_MULTIPLIERS.get(regime, 0.90)
        return regime, mult
    
    def calculate(self) -> MultiplierData:
        """Calculate combined macro multiplier."""
        logger.info("Calculating macro multipliers...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Get all regime data
        corr_regime, corr_mult = self.get_correlation_regime()
        vix_regime, vix_value, vix_mult = self.get_vix_regime()
        credit_regime, credit_mult = self.get_credit_regime()
        breadth_regime, breadth_value, breadth_mult = self.get_breadth_regime()
        labor_regime, labor_mult = self.get_labor_regime()
        
        # Calculate raw combined multiplier
        raw_multiplier = corr_mult * vix_mult * credit_mult * breadth_mult * labor_mult

        # Apply floor
        final_multiplier = max(MIN_MULTIPLIER, raw_multiplier)
        floor_applied = raw_multiplier < MIN_MULTIPLIER

        # Check for institutional exit signature
        inst_exit_tickers = self.get_institutional_exit_tickers()
        inst_exit_applied = len(inst_exit_tickers) > 0
        if inst_exit_applied:
            final_multiplier = round(final_multiplier * INST_EXIT_REDUCTION, 2)
            final_multiplier = max(MIN_MULTIPLIER, final_multiplier)

        # Interpretation
        interpretation = self._get_interpretation(final_multiplier, floor_applied)
        if inst_exit_applied:
            interpretation += f" | INST EXIT on {', '.join(inst_exit_tickers)} — 25% size reduction applied."

        result = MultiplierData(
            date=date_str,
            corr_regime=corr_regime,
            corr_mult=round(corr_mult, 2),
            vix_regime=vix_regime,
            vix_value=round(vix_value, 2),
            vix_mult=round(vix_mult, 2),
            credit_regime=credit_regime,
            credit_mult=round(credit_mult, 2),
            breadth_regime=breadth_regime,
            breadth_value=round(breadth_value, 1),
            breadth_mult=round(breadth_mult, 2),
            labor_regime=labor_regime,
            labor_mult=round(labor_mult, 2),
            raw_multiplier=round(raw_multiplier, 3),
            final_multiplier=round(final_multiplier, 2),
            floor_applied=floor_applied,
            inst_exit_tickers=inst_exit_tickers,
            inst_exit_applied=inst_exit_applied,
            interpretation=interpretation
        )

        logger.info(f"Final multiplier: {final_multiplier:.2f} (floor applied: {floor_applied}, inst exit: {inst_exit_applied})")
        
        return result
    
    def _get_interpretation(self, mult: float, floor_applied: bool) -> str:
        """Get human-readable interpretation."""
        if floor_applied:
            return f"Multiple stress signals - floor applied. Maintain {mult*100:.0f}% max equity."
        elif mult >= 0.90:
            return f"Healthy conditions - {mult*100:.0f}% equity exposure allowed"
        elif mult >= 0.70:
            return f"Some caution warranted - limit to {mult*100:.0f}% equity"
        elif mult >= 0.50:
            return f"Elevated risk - reduce to {mult*100:.0f}% equity"
        else:
            return f"High stress - defensive posture at {mult*100:.0f}% equity"
    
    def save_to_supabase(self, data: MultiplierData) -> bool:
        """Save multiplier data to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("macro_multipliers_daily") \
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
    
    parser = argparse.ArgumentParser(description="Macro Multipliers")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = MacroMultiplierCalculator()
    
    print(f"\n{'='*60}")
    print("MACRO MULTIPLIERS")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    
    print(f"\n{'='*40}")
    print("INDIVIDUAL MULTIPLIERS")
    print(f"{'='*40}")
    print(f"  Correlation: {data.corr_regime:12} → {data.corr_mult:.2f}")
    print(f"  VIX:         {data.vix_regime:12} → {data.vix_mult:.2f}  (VIX: {data.vix_value:.1f})")
    print(f"  Credit:      {data.credit_regime:12} → {data.credit_mult:.2f}")
    print(f"  Breadth:     {data.breadth_regime:12} → {data.breadth_mult:.2f}  ({data.breadth_value:.1f}%)")
    print(f"  Labor:       {data.labor_regime:12} → {data.labor_mult:.2f}")
    
    print(f"\n{'='*40}")
    print("COMBINED")
    print(f"{'='*40}")
    print(f"  Raw Multiplier:   {data.raw_multiplier:.3f}")
    print(f"  Final Multiplier: {data.final_multiplier:.2f}")
    print(f"  Floor Applied:    {data.floor_applied}")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
