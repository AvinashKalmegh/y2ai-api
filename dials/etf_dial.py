"""
ETF DIAL
========
Tracks institutional money flows via ETF price/volume data.
Maps ETFs to pillars for sector flow analysis.

ETF Universe:
- SMH, SOXX: Semiconductor (Infrastructure & Energy)
- XLU: Utilities (Infrastructure & Energy)
- IGV, WCLD: Software/Cloud (Enterprise Adoption)
- XLF: Financials (Financial & Market)
- TLT, GLD: Bonds/Gold (Macro & Policy)
- BOTZ: Robotics/AI (Productivity & Labor)
- XLY: Consumer Discretionary (Demand Dynamics)
- SPY, QQQ: Benchmarks

Flow Signal Logic:
- Compare current volume to 20-day average
- > 20% above avg = Strong Inflow
- > 5% above avg = Inflow
- < -5% = Outflow
- < -20% = Strong Outflow
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

ETF_CONFIG = {
    "etfs": [
        {"ticker": "SMH",  "name": "VanEck Semiconductor",   "pillar": "Infrastructure", "weight": 0.5},
        {"ticker": "SOXX", "name": "iShares Semiconductor",  "pillar": "Infrastructure", "weight": 0.3},
        {"ticker": "XLU",  "name": "Utilities Select",       "pillar": "Infrastructure", "weight": 0.2},
        {"ticker": "IGV",  "name": "iShares Software",       "pillar": "Enterprise",     "weight": 0.6},
        {"ticker": "WCLD", "name": "WisdomTree Cloud",       "pillar": "Enterprise",     "weight": 0.4},
        {"ticker": "XLF",  "name": "Financial Select",       "pillar": "Financial",      "weight": 1.0},
        {"ticker": "TLT",  "name": "20+ Year Treasury",      "pillar": "Macro",          "weight": 0.5},
        {"ticker": "GLD",  "name": "Gold Trust",             "pillar": "Macro",          "weight": 0.5},
        {"ticker": "BOTZ", "name": "Global Robotics & AI",   "pillar": "Productivity",   "weight": 1.0},
        {"ticker": "XLY",  "name": "Consumer Discretionary", "pillar": "Demand",         "weight": 1.0},
        {"ticker": "SPY",  "name": "S&P 500",                "pillar": "Benchmark",      "weight": 1.0},
        {"ticker": "QQQ",  "name": "Nasdaq 100",             "pillar": "Benchmark",      "weight": 1.0},
    ],
    
    "flow_lookback": 20,        # Days for average volume
    "flow_threshold": 1.5,      # 1.5x avg = significant
    "momentum_days": 5,         # Days for flow momentum
    
    # Regime thresholds (volume vs average)
    "regime_thresholds": {
        "strong_inflow": 0.20,   # > 20% above avg
        "inflow": 0.05,          # > 5% above avg
        "outflow": -0.05,        # < -5%
        "strong_outflow": -0.20  # < -20%
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ETFFlowData:
    """ETF flow data for a single ETF."""
    ticker: str
    name: str
    pillar: str
    price: float
    price_change_1d: float
    price_change_5d: float
    volume: int
    avg_volume_20d: int
    volume_ratio: float
    flow_signal: str


@dataclass
class ETFDialData:
    """Aggregate ETF flow data."""
    date: str
    # Overall signals
    overall_flow: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    # Pillar flows
    pillar_flows: Dict[str, str]
    # Individual ETFs
    etf_flows: List[Dict]
    # Regime
    regime: str
    interpretation: str


# =============================================================================
# ETF CALCULATOR
# =============================================================================

class ETFDialCalculator:
    """
    Calculate ETF flow signals.
    
    Usage:
        calc = ETFDialCalculator()
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
        
        self.config = ETF_CONFIG
        self.etfs = self.config["etfs"]
    
    def fetch_etf_data(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch ETF price and volume data."""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(period=f"{days}d")
            
            if len(hist) > 0:
                hist = hist.sort_index(ascending=False)
                return hist
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
        
        return None
    
    def calculate_flow_signal(self, current_volume: int, avg_volume: int) -> str:
        """Determine flow signal from volume comparison."""
        if avg_volume == 0:
            return "Neutral"
        
        ratio = (current_volume - avg_volume) / avg_volume
        thresholds = self.config["regime_thresholds"]
        
        if ratio > thresholds["strong_inflow"]:
            return "Strong Inflow"
        elif ratio > thresholds["inflow"]:
            return "Inflow"
        elif ratio < thresholds["strong_outflow"]:
            return "Strong Outflow"
        elif ratio < thresholds["outflow"]:
            return "Outflow"
        else:
            return "Neutral"
    
    def calculate_etf_flows(self) -> List[ETFFlowData]:
        """Calculate flow data for all ETFs."""
        results = []
        lookback = self.config["flow_lookback"]
        
        for etf_info in self.etfs:
            ticker = etf_info["ticker"]
            hist = self.fetch_etf_data(ticker, lookback + 10)
            
            if hist is None or len(hist) < lookback:
                continue
            
            # Current values
            price = hist["Close"].iloc[0]
            volume = int(hist["Volume"].iloc[0])
            
            # Price changes
            price_1d = (hist["Close"].iloc[0] / hist["Close"].iloc[1] - 1) if len(hist) >= 2 else 0
            price_5d = (hist["Close"].iloc[0] / hist["Close"].iloc[5] - 1) if len(hist) >= 6 else 0
            
            # Average volume
            avg_volume = int(hist["Volume"].iloc[:lookback].mean())
            
            # Volume ratio and signal
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            flow_signal = self.calculate_flow_signal(volume, avg_volume)
            
            results.append(ETFFlowData(
                ticker=ticker,
                name=etf_info["name"],
                pillar=etf_info["pillar"],
                price=round(price, 2),
                price_change_1d=round(price_1d * 100, 2),
                price_change_5d=round(price_5d * 100, 2),
                volume=volume,
                avg_volume_20d=avg_volume,
                volume_ratio=round(volume_ratio, 2),
                flow_signal=flow_signal
            ))
        
        return results
    
    def aggregate_pillar_flows(self, etf_flows: List[ETFFlowData]) -> Dict[str, str]:
        """Aggregate flows by pillar."""
        pillar_signals = {}
        
        pillars = set(etf["pillar"] for etf in self.etfs if etf["pillar"] != "Benchmark")
        
        for pillar in pillars:
            pillar_etfs = [e for e in etf_flows if e.pillar == pillar]
            
            if not pillar_etfs:
                pillar_signals[pillar] = "Unknown"
                continue
            
            # Count inflows vs outflows
            inflows = sum(1 for e in pillar_etfs if "Inflow" in e.flow_signal)
            outflows = sum(1 for e in pillar_etfs if "Outflow" in e.flow_signal)
            
            if inflows > outflows:
                pillar_signals[pillar] = "Inflow"
            elif outflows > inflows:
                pillar_signals[pillar] = "Outflow"
            else:
                pillar_signals[pillar] = "Neutral"
        
        return pillar_signals
    
    def determine_overall_regime(self, etf_flows: List[ETFFlowData]) -> str:
        """Determine overall market regime from flows."""
        # Exclude benchmarks
        non_benchmark = [e for e in etf_flows if e.pillar != "Benchmark"]
        
        if not non_benchmark:
            return "Unknown"
        
        inflow_count = sum(1 for e in non_benchmark if "Inflow" in e.flow_signal)
        outflow_count = sum(1 for e in non_benchmark if "Outflow" in e.flow_signal)
        
        total = len(non_benchmark)
        inflow_pct = inflow_count / total
        outflow_pct = outflow_count / total
        
        if inflow_pct > 0.6:
            return "Risk-On"
        elif outflow_pct > 0.6:
            return "Risk-Off"
        elif inflow_pct > outflow_pct:
            return "Mild Risk-On"
        elif outflow_pct > inflow_pct:
            return "Mild Risk-Off"
        else:
            return "Neutral"
    
    def calculate(self) -> ETFDialData:
        """Main calculation: compute all ETF flow metrics."""
        logger.info("Calculating ETF flows...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate individual ETF flows
        etf_flows = self.calculate_etf_flows()
        
        if not etf_flows:
            return ETFDialData(
                date=date_str,
                overall_flow="Unknown",
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                pillar_flows={},
                etf_flows=[],
                regime="Unknown",
                interpretation="No ETF data available"
            )
        
        # Aggregate by pillar
        pillar_flows = self.aggregate_pillar_flows(etf_flows)
        
        # Overall regime
        regime = self.determine_overall_regime(etf_flows)
        
        # Count signals
        non_benchmark = [e for e in etf_flows if e.pillar != "Benchmark"]
        bullish = sum(1 for e in non_benchmark if "Inflow" in e.flow_signal)
        bearish = sum(1 for e in non_benchmark if "Outflow" in e.flow_signal)
        neutral = len(non_benchmark) - bullish - bearish
        
        # Overall flow direction
        if bullish > bearish:
            overall_flow = "Net Inflow"
        elif bearish > bullish:
            overall_flow = "Net Outflow"
        else:
            overall_flow = "Balanced"
        
        # Interpretation
        interpretation = f"{regime}: {bullish} ETFs showing inflows, {bearish} showing outflows"
        
        result = ETFDialData(
            date=date_str,
            overall_flow=overall_flow,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            pillar_flows=pillar_flows,
            etf_flows=[asdict(e) for e in etf_flows],
            regime=regime,
            interpretation=interpretation
        )
        
        logger.info(f"ETF Regime: {regime}, Overall: {overall_flow}")
        
        return result
    
    def save_to_supabase(self, data: ETFDialData) -> bool:
        """Save ETF data to Supabase."""
        if not self.supabase:
            return False
        
        row = {
            "date": data.date,
            "overall_flow": data.overall_flow,
            "bullish_count": data.bullish_count,
            "bearish_count": data.bearish_count,
            "neutral_count": data.neutral_count,
            "pillar_flows": data.pillar_flows,
            "etf_flows": data.etf_flows,
            "regime": data.regime,
            "interpretation": data.interpretation
        }
        
        try:
            self.supabase.table("etf_dial_daily") \
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
    
    parser = argparse.ArgumentParser(description="ETF Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = ETFDialCalculator()
    
    print(f"\n{'='*60}")
    print("ETF DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"REGIME: {data.regime}")
    print(f"OVERALL FLOW: {data.overall_flow}")
    print(f"{'='*40}")
    
    print(f"\nSignal Counts:")
    print(f"  Bullish (Inflows): {data.bullish_count}")
    print(f"  Bearish (Outflows): {data.bearish_count}")
    print(f"  Neutral: {data.neutral_count}")
    
    print(f"\n{'='*40}")
    print("PILLAR FLOWS")
    print(f"{'='*40}")
    for pillar, flow in data.pillar_flows.items():
        print(f"  {pillar}: {flow}")
    
    print(f"\n{'='*40}")
    print("ETF DETAILS")
    print(f"{'='*40}")
    for etf in data.etf_flows:
        print(f"  {etf['ticker']:5} {etf['flow_signal']:15} Vol: {etf['volume_ratio']:.1f}x  Price: {etf['price_change_1d']:+.1f}%")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
