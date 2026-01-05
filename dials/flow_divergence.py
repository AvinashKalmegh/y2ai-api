"""
FLOW DIVERGENCE
===============
Compares ETF flows (passive/broad) vs Stock flows (active/specific).

Logic:
- ETF flows = passive money, retail, broad sector
- Stock flows = active money, institutional, specific names

Divergence Signals:
- ETF Inflow + Stock Outflow = "Distribution into Strength" (bearish)
  Smart money selling while passive buying
- ETF Outflow + Stock Inflow = "Accumulation into Weakness" (bullish)
  Smart money buying while passive selling
- Both aligned = Confirmation (trend likely continues)

Pillar Weights:
- Infrastructure: 30% (highest - core thesis)
- Enterprise: 25%
- Financial: 15%
- Macro: 10%
- Productivity: 10%
- Demand: 10%
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
# CONFIGURATION
# =============================================================================

DIVERGENCE_CONFIG = {
    # Pillar weights for scoring
    "weights": {
        "Infrastructure": 0.30,  # Highest - core thesis
        "Enterprise": 0.25,
        "Financial": 0.15,
        "Macro": 0.10,
        "Productivity": 0.10,
        "Demand": 0.10
    },
    
    # Regime score mappings
    "regime_scores": {
        "Strong Inflow": 2,
        "Inflow": 1,
        "Accumulate": 1,
        "Strong Accumulation": 2,
        "Accumulation": 1,
        "Neutral": 0,
        "Hold": 0,
        "Mixed": 0,
        "Outflow": -1,
        "Distribution": -1,
        "Reduce": -1,
        "Strong Outflow": -2,
        "Strong Distribution": -2
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PillarDivergence:
    """Divergence data for a single pillar."""
    pillar: str
    etf_flow: str
    stock_flow: str
    divergence: str
    divergence_type: str
    score: int
    signal: str


@dataclass
class FlowDivergenceData:
    """Complete flow divergence data."""
    date: str
    # Overall
    overall_signal: str
    divergence_score: float
    interpretation: str
    # Pillar details
    pillar_divergences: List[Dict]
    # Counts
    bullish_count: int
    bearish_count: int
    neutral_count: int


# =============================================================================
# FLOW DIVERGENCE CALCULATOR
# =============================================================================

class FlowDivergenceCalculator:
    """
    Calculate flow divergence between ETF and stock flows.
    
    Usage:
        calc = FlowDivergenceCalculator()
        data = calc.calculate()
        calc.save_to_supabase(data)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.config = DIVERGENCE_CONFIG
    
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
    
    def get_etf_flows(self) -> Dict[str, str]:
        """Get ETF flows by pillar."""
        data = self._get_latest("etf_dial_daily")
        
        if data and data.get("pillar_flows"):
            return data["pillar_flows"]
        
        return {}
    
    def get_stock_flows(self) -> Dict[str, str]:
        """Get stock flows by pillar."""
        data = self._get_latest("stock_flow_dial_daily")
        
        if data and data.get("pillar_flows"):
            # Convert list to dict
            pillar_flows = {}
            for pf in data["pillar_flows"]:
                pillar_flows[pf["pillar"]] = pf["regime"]
            return pillar_flows
        
        return {}
    
    def get_flow_score(self, regime: str) -> int:
        """Get numeric score for flow regime."""
        return self.config["regime_scores"].get(regime, 0)
    
    def determine_divergence(self, etf_flow: str, stock_flow: str) -> tuple:
        """
        Determine divergence type and signal.
        
        Returns:
            Tuple of (divergence_type, signal, score)
        """
        etf_score = self.get_flow_score(etf_flow)
        stock_score = self.get_flow_score(stock_flow)
        
        # ETF positive + Stock negative = Distribution into Strength (bearish)
        if etf_score > 0 and stock_score < 0:
            return "Distribution into Strength", "Bearish", -2
        
        # ETF negative + Stock positive = Accumulation into Weakness (bullish)
        if etf_score < 0 and stock_score > 0:
            return "Accumulation into Weakness", "Bullish", 2
        
        # Both positive = Confirmed Inflow (bullish)
        if etf_score > 0 and stock_score > 0:
            return "Confirmed Inflow", "Bullish", 1
        
        # Both negative = Confirmed Outflow (bearish)
        if etf_score < 0 and stock_score < 0:
            return "Confirmed Outflow", "Bearish", -1
        
        # Mixed or neutral
        return "No Divergence", "Neutral", 0
    
    def calculate_pillar_divergences(self, etf_flows: Dict, stock_flows: Dict) -> List[PillarDivergence]:
        """Calculate divergence for each pillar."""
        results = []
        
        for pillar, weight in self.config["weights"].items():
            etf_flow = etf_flows.get(pillar, "Unknown")
            stock_flow = stock_flows.get(pillar, "Unknown")
            
            if etf_flow == "Unknown" or stock_flow == "Unknown":
                divergence_type = "Unknown"
                signal = "Unknown"
                score = 0
            else:
                divergence_type, signal, score = self.determine_divergence(etf_flow, stock_flow)
            
            # Determine divergence text
            if "Distribution into Strength" in divergence_type:
                divergence = "ETF ↑ Stock ↓"
            elif "Accumulation into Weakness" in divergence_type:
                divergence = "ETF ↓ Stock ↑"
            elif "Confirmed Inflow" in divergence_type:
                divergence = "ETF ↑ Stock ↑"
            elif "Confirmed Outflow" in divergence_type:
                divergence = "ETF ↓ Stock ↓"
            else:
                divergence = "Neutral"
            
            results.append(PillarDivergence(
                pillar=pillar,
                etf_flow=etf_flow,
                stock_flow=stock_flow,
                divergence=divergence,
                divergence_type=divergence_type,
                score=score,
                signal=signal
            ))
        
        return results
    
    def calculate_weighted_score(self, pillar_divergences: List[PillarDivergence]) -> float:
        """Calculate weighted divergence score."""
        total_score = 0
        
        for pd in pillar_divergences:
            weight = self.config["weights"].get(pd.pillar, 0)
            total_score += pd.score * weight
        
        return total_score
    
    def calculate(self) -> FlowDivergenceData:
        """Main calculation: compute flow divergence."""
        logger.info("Calculating flow divergence...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Get flows
        etf_flows = self.get_etf_flows()
        stock_flows = self.get_stock_flows()
        
        if not etf_flows and not stock_flows:
            return FlowDivergenceData(
                date=date_str,
                overall_signal="Unknown",
                divergence_score=0,
                interpretation="No flow data available",
                pillar_divergences=[],
                bullish_count=0,
                bearish_count=0,
                neutral_count=0
            )
        
        # Calculate divergences
        pillar_divergences = self.calculate_pillar_divergences(etf_flows, stock_flows)
        
        # Calculate weighted score
        divergence_score = self.calculate_weighted_score(pillar_divergences)
        
        # Count signals
        bullish = sum(1 for pd in pillar_divergences if pd.signal == "Bullish")
        bearish = sum(1 for pd in pillar_divergences if pd.signal == "Bearish")
        neutral = sum(1 for pd in pillar_divergences if pd.signal in ["Neutral", "Unknown"])
        
        # Overall signal
        if divergence_score > 0.5:
            overall_signal = "Bullish"
        elif divergence_score < -0.5:
            overall_signal = "Bearish"
        else:
            overall_signal = "Neutral"
        
        # Interpretation
        interpretation = self._get_interpretation(overall_signal, divergence_score, bullish, bearish)
        
        result = FlowDivergenceData(
            date=date_str,
            overall_signal=overall_signal,
            divergence_score=round(divergence_score, 2),
            interpretation=interpretation,
            pillar_divergences=[asdict(pd) for pd in pillar_divergences],
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral
        )
        
        logger.info(f"Flow Divergence: {overall_signal} (score: {divergence_score:.2f})")
        
        return result
    
    def _get_interpretation(self, signal: str, score: float, bullish: int, bearish: int) -> str:
        """Get human-readable interpretation."""
        if signal == "Bullish":
            if score > 1:
                return f"Strong bullish divergence - smart money accumulating. {bullish} pillars bullish."
            else:
                return f"Mild bullish divergence - watch for confirmation. {bullish} pillars bullish."
        elif signal == "Bearish":
            if score < -1:
                return f"Strong bearish divergence - smart money distributing. {bearish} pillars bearish."
            else:
                return f"Mild bearish divergence - watch for confirmation. {bearish} pillars bearish."
        else:
            return f"No clear divergence signal. ETF and stock flows aligned or neutral."
    
    def save_to_supabase(self, data: FlowDivergenceData) -> bool:
        """Save divergence data to Supabase."""
        if not self.supabase:
            return False
        
        row = {
            "date": data.date,
            "overall_signal": data.overall_signal,
            "divergence_score": data.divergence_score,
            "interpretation": data.interpretation,
            "pillar_divergences": data.pillar_divergences,
            "bullish_count": data.bullish_count,
            "bearish_count": data.bearish_count,
            "neutral_count": data.neutral_count
        }
        
        try:
            self.supabase.table("flow_divergence_daily") \
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
    
    parser = argparse.ArgumentParser(description="Flow Divergence")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = FlowDivergenceCalculator()
    
    print(f"\n{'='*60}")
    print("FLOW DIVERGENCE DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"OVERALL SIGNAL: {data.overall_signal}")
    print(f"DIVERGENCE SCORE: {data.divergence_score:+.2f}")
    print(f"{'='*40}")
    
    print(f"\nSignal Counts:")
    print(f"  Bullish: {data.bullish_count}")
    print(f"  Bearish: {data.bearish_count}")
    print(f"  Neutral: {data.neutral_count}")
    
    print(f"\n{'='*40}")
    print("PILLAR DIVERGENCES")
    print(f"{'='*40}")
    for pd in data.pillar_divergences:
        print(f"  {pd['pillar']:15} {pd['divergence']:15} {pd['divergence_type']:25} ({pd['signal']})")
    
    print(f"\n{'='*40}")
    print("LEGEND")
    print(f"{'='*40}")
    print("  ETF ↑ Stock ↓ = Distribution into Strength (Bearish)")
    print("  ETF ↓ Stock ↑ = Accumulation into Weakness (Bullish)")
    print("  ETF ↑ Stock ↑ = Confirmed Inflow (Bullish)")
    print("  ETF ↓ Stock ↓ = Confirmed Outflow (Bearish)")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
