"""
SIGNALS DIAL
============
Infrastructure signals aggregated from news analysis.
Pulls daily signal data from Y2AI API and calculates composite infrastructure signal.

Components (weighted):
- Capex Signal (40%): Capital expenditure announcements
- Energy Signal (30%): Power/energy infrastructure news
- Compute Signal (30%): Data center/compute infrastructure

Signal Regimes:
- STRONG_CYCLE: > +30 (Full infrastructure expansion)
- CYCLE_INTACT: +10 to +30 (Healthy cycle continuation)
- MIXED_SIGNALS: -10 to +10 (Conflicting indicators)
- WEAKENING: -30 to -10 (Cycle momentum fading)
- CYCLE_RISK: < -30 (Infrastructure thesis at risk)
- VETO_ALERT: High-severity event detected

Thesis Indicators:
- Thesis Balance: Net bullish vs bearish
- Cycle Support: Evidence for continued cycle
- Bubble Warnings: Speculative excess signals
- Constraint Evidence: Supply/resource constraints
- Demand Validation: Real demand confirmation
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()
import requests

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNALS_CONFIG = {
    "api_base_url": os.getenv("Y2AI_API_URL", "https://y2ai-api-production.up.railway.app"),
    
    # Signal regime thresholds
    "regime_thresholds": {
        "strong_cycle": 30,      # infra_signal > 30
        "cycle_intact": 10,      # infra_signal > 10
        "mixed": -10,            # infra_signal > -10
        "weakening": -30         # infra_signal > -30
        # Below -30 = CYCLE_RISK
    },
    
    # Weights for composite calculation
    "weights": {
        "capex": 0.40,
        "energy": 0.30,
        "compute": 0.30
    },
    
    # Veto thresholds
    "veto_threshold": 3,  # Number of high-severity events to trigger VETO
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignalsData:
    """Daily signals data."""
    date: str
    # Composite signal
    infrastructure_signal: float = 0.0
    signal_regime: str = "NO_DATA"
    # Component signals
    capex_signal: float = 0.0
    energy_signal: float = 0.0
    compute_signal: float = 0.0
    # Article metrics
    article_count: int = 0
    avg_impact_score: float = 0.0
    # Thesis indicators
    thesis_balance: float = 0.0
    cycle_support: int = 0
    bubble_warnings: int = 0
    constraint_evidence: int = 0
    demand_validation: int = 0
    # Alerts
    veto_triggers: int = 0
    depreciation_flags: int = 0
    # Interpretation
    interpretation: str = ""


# =============================================================================
# SIGNALS CALCULATOR
# =============================================================================

class SignalsDialCalculator:
    """
    Calculate infrastructure signals from news data.
    
    Usage:
        calc = SignalsDialCalculator()
        signals = calc.calculate()
        calc.save_to_supabase(signals)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.config = SIGNALS_CONFIG
        self.api_base = self.config["api_base_url"]
    
    # =========================================================================
    # API FUNCTIONS
    # =========================================================================
    
    def fetch_daily_signals(self, date: str = None) -> Optional[Dict]:
        """
        Fetch daily signals from Y2AI API.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            Dict with signal data or None if failed
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            url = f"{self.api_base}/daily-signals?date={date}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API returned {response.status_code} for {date}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to fetch signals: {e}")
            return None
    
    def fetch_from_supabase(self, date: str = None) -> Optional[Dict]:
        """
        Fetch signals from Supabase daily_signals table.
        Falls back to this if API is unavailable.
        """
        if not self.supabase:
            return None
        
        try:
            query = self.supabase.table("daily_signals").select("*")
            
            if date:
                query = query.eq("date", date)
            else:
                query = query.order("date", desc=True).limit(1)
            
            response = query.execute()
            
            if response.data:
                return response.data[0]
                
        except Exception as e:
            logger.warning(f"Supabase fetch failed: {e}")
        
        return None
    
    # =========================================================================
    # SIGNAL CALCULATIONS
    # =========================================================================
    
    def calculate_from_articles(self, date: str = None) -> Optional[SignalsData]:
        """
        Calculate signals directly from article data in Supabase.
        Used when API is unavailable.
        """
        if not self.supabase:
            return None
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Fetch articles for date
            response = self.supabase.table("articles") \
                .select("category, sentiment_score, impact_score, is_veto_trigger") \
                .eq("date", date) \
                .execute()
            
            if not response.data:
                return None
            
            articles = response.data
            
            # Calculate component signals
            capex_articles = [a for a in articles if a.get("category") == "capex"]
            energy_articles = [a for a in articles if a.get("category") == "energy"]
            compute_articles = [a for a in articles if a.get("category") == "compute"]
            
            def calc_signal(articles_list):
                if not articles_list:
                    return 0
                # Weighted average of sentiment * impact
                total = sum(
                    a.get("sentiment_score", 0) * a.get("impact_score", 1) 
                    for a in articles_list
                )
                return total / len(articles_list) * 100  # Scale to -100 to +100
            
            capex_signal = calc_signal(capex_articles)
            energy_signal = calc_signal(energy_articles)
            compute_signal = calc_signal(compute_articles)
            
            # Composite infrastructure signal
            weights = self.config["weights"]
            infra_signal = (
                capex_signal * weights["capex"] +
                energy_signal * weights["energy"] +
                compute_signal * weights["compute"]
            )
            
            # Count veto triggers
            veto_count = sum(1 for a in articles if a.get("is_veto_trigger"))
            
            # Determine regime
            regime = self._get_regime(infra_signal, veto_count)
            
            # Average impact
            avg_impact = sum(a.get("impact_score", 0) for a in articles) / len(articles) if articles else 0
            
            return SignalsData(
                date=date,
                infrastructure_signal=round(infra_signal, 1),
                signal_regime=regime,
                capex_signal=round(capex_signal, 1),
                energy_signal=round(energy_signal, 1),
                compute_signal=round(compute_signal, 1),
                article_count=len(articles),
                avg_impact_score=round(avg_impact, 2),
                veto_triggers=veto_count,
                interpretation=self._get_interpretation(regime)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate from articles: {e}")
            return None
    
    def _get_regime(self, infra_signal: float, veto_count: int) -> str:
        """Determine signal regime."""
        thresholds = self.config["regime_thresholds"]
        
        # Check for VETO first
        if veto_count >= self.config["veto_threshold"]:
            return "VETO_ALERT"
        
        if infra_signal > thresholds["strong_cycle"]:
            return "STRONG_CYCLE"
        elif infra_signal > thresholds["cycle_intact"]:
            return "CYCLE_INTACT"
        elif infra_signal > thresholds["mixed"]:
            return "MIXED_SIGNALS"
        elif infra_signal > thresholds["weakening"]:
            return "WEAKENING"
        else:
            return "CYCLE_RISK"
    
    def _get_interpretation(self, regime: str) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            "STRONG_CYCLE": "Full infrastructure expansion - strong capital deployment",
            "CYCLE_INTACT": "Healthy cycle continuation - maintain positions",
            "MIXED_SIGNALS": "Conflicting indicators - monitor closely",
            "WEAKENING": "Cycle momentum fading - reduce exposure",
            "CYCLE_RISK": "Infrastructure thesis at risk - defensive posture",
            "VETO_ALERT": "High-severity event detected - pause new positions",
            "NO_DATA": "Insufficient data for signal calculation"
        }
        return interpretations.get(regime, "Unknown regime")
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self, date: str = None) -> SignalsData:
        """
        Main calculation: fetch and process signals.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            SignalsData with all signal metrics
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Calculating signals for {date}")
        
        # Try API first
        api_data = self.fetch_daily_signals(date)
        
        if api_data:
            # Parse API response
            infra_signal = api_data.get("infrastructure_signal", 0)
            veto_count = api_data.get("veto_triggers", 0)
            
            return SignalsData(
                date=date,
                infrastructure_signal=api_data.get("infrastructure_signal", 0),
                signal_regime=api_data.get("signal_regime", self._get_regime(infra_signal, veto_count)),
                capex_signal=api_data.get("capex_signal", 0),
                energy_signal=api_data.get("energy_signal", 0),
                compute_signal=api_data.get("compute_signal", 0),
                article_count=api_data.get("article_count", 0),
                avg_impact_score=api_data.get("avg_impact_score", 0),
                thesis_balance=api_data.get("thesis_balance", 0),
                cycle_support=api_data.get("cycle_support", 0),
                bubble_warnings=api_data.get("bubble_warnings", 0),
                constraint_evidence=api_data.get("constraint_evidence", 0),
                demand_validation=api_data.get("demand_validation", 0),
                veto_triggers=api_data.get("veto_triggers", 0),
                depreciation_flags=api_data.get("depreciation_flags", 0),
                interpretation=self._get_interpretation(api_data.get("signal_regime", "NO_DATA"))
            )
        
        # Try Supabase daily_signals table
        sb_data = self.fetch_from_supabase(date)
        if sb_data:
            infra_signal = sb_data.get("infrastructure_signal", 0)
            veto_count = sb_data.get("veto_triggers", 0)
            
            return SignalsData(
                date=date,
                infrastructure_signal=sb_data.get("infrastructure_signal", 0),
                signal_regime=sb_data.get("signal_regime", self._get_regime(infra_signal, veto_count)),
                capex_signal=sb_data.get("capex_signal", 0),
                energy_signal=sb_data.get("energy_signal", 0),
                compute_signal=sb_data.get("compute_signal", 0),
                article_count=sb_data.get("article_count", 0),
                avg_impact_score=sb_data.get("avg_impact_score", 0),
                thesis_balance=sb_data.get("thesis_balance", 0),
                veto_triggers=sb_data.get("veto_triggers", 0),
                interpretation=self._get_interpretation(sb_data.get("signal_regime", "NO_DATA"))
            )
        
        # Try calculating from articles
        calc_data = self.calculate_from_articles(date)
        if calc_data:
            return calc_data
        
        # No data available
        logger.warning(f"No signal data available for {date}")
        return SignalsData(
            date=date,
            signal_regime="NO_DATA",
            interpretation="No signal data available"
        )
    
    # =========================================================================
    # AMRI ALIGNMENT
    # =========================================================================
    
    def check_amri_alignment(self) -> Dict:
        """
        Check if signals align with or contradict AMRI.
        High AMRI (stress) should align with negative signals.
        Low AMRI (healthy) should align with positive signals.
        """
        signals = self.calculate()
        
        # Get AMRI
        amri_score = 30  # Default
        if self.supabase:
            try:
                response = self.supabase.table("amri_daily") \
                    .select("amri_score") \
                    .order("date", desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    amri_score = response.data[0].get("amri_score", 30)
            except:
                pass
        
        amri_stressed = amri_score > 50
        signals_negative = signals.infrastructure_signal < 0
        
        aligned = (amri_stressed and signals_negative) or (not amri_stressed and not signals_negative)
        
        if aligned:
            message = "Signals and AMRI aligned - consistent regime reading"
        elif amri_stressed and not signals_negative:
            message = "DIVERGENCE: AMRI shows stress but news signals remain positive"
        else:
            message = "DIVERGENCE: News signals negative but AMRI shows healthy structure"
        
        return {
            "aligned": aligned,
            "amri_score": amri_score,
            "infrastructure_signal": signals.infrastructure_signal,
            "message": message
        }
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, data: SignalsData) -> bool:
        """Save signals data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("signals_dial_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved signals for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def get_history(self, days: int = 30) -> List[SignalsData]:
        """Get signals history."""
        if not self.supabase:
            return []
        
        try:
            response = self.supabase.table("signals_dial_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(days) \
                .execute()
            
            return [SignalsData(**row) for row in response.data]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


# =============================================================================
# SUPABASE TABLE
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signals_dial_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    infrastructure_signal FLOAT,
    signal_regime VARCHAR(20),
    capex_signal FLOAT,
    energy_signal FLOAT,
    compute_signal FLOAT,
    article_count INT,
    avg_impact_score FLOAT,
    thesis_balance FLOAT,
    cycle_support INT,
    bubble_warnings INT,
    constraint_evidence INT,
    demand_validation INT,
    veto_triggers INT,
    depreciation_flags INT,
    interpretation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_dial_date ON signals_dial_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run signals dial calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Signals Dial")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    parser.add_argument("--history", type=int, default=0, help="Show N days of history")
    parser.add_argument("--alignment", action="store_true", help="Check AMRI alignment")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = SignalsDialCalculator()
    
    print(f"\n{'='*60}")
    print("SIGNALS DIAL")
    print(f"{'='*60}\n")
    
    if args.history > 0:
        history = calc.get_history(args.history)
        print(f"Last {len(history)} days:\n")
        for sig in history[:10]:
            print(f"  {sig.date}: {sig.infrastructure_signal:+.1f} ({sig.signal_regime})")
    
    elif args.alignment:
        result = calc.check_amri_alignment()
        print(f"AMRI Score: {result['amri_score']:.0f}")
        print(f"Infrastructure Signal: {result['infrastructure_signal']:+.1f}")
        print(f"Aligned: {result['aligned']}")
        print(f"\n{result['message']}")
    
    else:
        signals = calc.calculate(args.date)
        
        print(f"Date: {signals.date}")
        print(f"\n{'='*40}")
        print(f"INFRASTRUCTURE SIGNAL: {signals.infrastructure_signal:+.1f}")
        print(f"REGIME: {signals.signal_regime}")
        print(f"{'='*40}")
        
        print(f"\n{signals.interpretation}")
        
        print(f"\n{'='*40}")
        print("COMPONENTS")
        print(f"{'='*40}")
        print(f"  Capex (40%):   {signals.capex_signal:+.1f}")
        print(f"  Energy (30%):  {signals.energy_signal:+.1f}")
        print(f"  Compute (30%): {signals.compute_signal:+.1f}")
        
        print(f"\n{'='*40}")
        print("THESIS INDICATORS")
        print(f"{'='*40}")
        print(f"  Thesis Balance:      {signals.thesis_balance:+.1f}")
        print(f"  Cycle Support:       {signals.cycle_support}")
        print(f"  Bubble Warnings:     {signals.bubble_warnings}")
        print(f"  Constraint Evidence: {signals.constraint_evidence}")
        print(f"  Demand Validation:   {signals.demand_validation}")
        
        print(f"\n{'='*40}")
        print("ALERTS")
        print(f"{'='*40}")
        print(f"  Veto Triggers:       {signals.veto_triggers}")
        print(f"  Depreciation Flags:  {signals.depreciation_flags}")
        print(f"  Articles Analyzed:   {signals.article_count}")
        
        if args.save:
            if calc.save_to_supabase(signals):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
