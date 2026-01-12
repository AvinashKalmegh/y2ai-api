"""
STOCK FLOW DIAL
===============
Tracks accumulation/distribution in Universe stocks.
Uses volume analysis as flow proxy.

Flow Regime Logic:
- Strong Accumulation: Volume > 1.5x avg AND price up
- Accumulation: Volume > 1.2x avg AND price up
- Distribution: Volume > 1.2x avg AND price down
- Strong Distribution: Volume > 1.5x avg AND price down
- Neutral: Normal volume or mixed signals

Pillar Aggregation:
- Strong Inflow: 30%+ of pillar stocks accumulating
- Inflow: 15%+ of pillar stocks accumulating
- Outflow: 15%+ of pillar stocks distributing
- Strong Outflow: 30%+ of pillar stocks distributing
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

STOCK_FLOW_CONFIG = {
    "volume_lookback": 20,        # Days to calculate average volume
    
    # Flow regime thresholds (volume ratio)
    "regimes": {
        "strong_accumulation": 1.50,  # >1.5x volume + price up
        "accumulation": 1.20,         # >1.2x volume + price up
        "distribution": 1.20,         # >1.2x volume + price down
        "strong_distribution": 1.50   # >1.5x volume + price down
    },
    
    # Pillar aggregation thresholds
    "pillar_thresholds": {
        "strong_inflow": 0.30,   # 30%+ in accumulation
        "inflow": 0.15,          # 15%+ in accumulation
        "outflow": -0.15,        # 15%+ in distribution
        "strong_outflow": -0.30  # 30%+ in distribution
    }
}

# Pillar mapping
PILLAR_STOCKS = {
    "Infrastructure": ["TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT",
                      "CEG", "NRG", "EQIX", "DLR", "KLAC", "LRCX", "AMAT", "QCOM"],
    "Enterprise": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR",
                  "ADBE", "ORCL", "MDB", "DDOG", "ZS"],
    "Productivity": ["NET", "CRWD", "PANW"],
    "Demand": ["TSLA", "SHOP", "UBER"],
    "Macro": ["NXPI", "ON", "SMCI", "ARM"],
    "Financial": ["GS", "MS", "JKS", "FSLR"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StockFlow:
    """Flow data for a single stock."""
    ticker: str
    pillar: str
    price: float
    price_change_1d: float
    volume: int
    avg_volume: int
    volume_ratio: float
    dollar_volume_m: float
    flow_regime: str


@dataclass
class PillarFlow:
    """Aggregated flow data for a pillar."""
    pillar: str
    stock_count: int
    accumulating: int
    distributing: int
    neutral: int
    net_flow: float
    regime: str
    signal: str


@dataclass
class StockFlowDialData:
    """Complete stock flow dial data."""
    date: str
    # Pillar summaries
    pillar_flows: List[Dict]
    # Stock details
    stock_flows: List[Dict]
    # Overall
    overall_regime: str
    accumulating_count: int
    distributing_count: int
    interpretation: str


# =============================================================================
# STOCK FLOW CALCULATOR
# =============================================================================

class StockFlowCalculator:
    """
    Calculate stock flow signals.
    
    Usage:
        calc = StockFlowCalculator()
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
        
        self.config = STOCK_FLOW_CONFIG
        
        # Build ticker to pillar mapping
        self.ticker_to_pillar = {}
        for pillar, tickers in PILLAR_STOCKS.items():
            for ticker in tickers:
                self.ticker_to_pillar[ticker] = pillar
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers."""
        return list(self.ticker_to_pillar.keys())
    
    def fetch_stock_data(self, ticker: str, days: int = 25) -> Optional[Dict]:
        """Fetch stock price and volume data."""
        # Check if we have cached batch data
        if hasattr(self, '_batch_data') and ticker in self._batch_data:
            return self._batch_data[ticker]
        
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")
            
            if len(hist) < 2:
                return None
            
            hist = hist.sort_index(ascending=False)
            
            current_price = hist["Close"].iloc[0]
            prev_price = hist["Close"].iloc[1]
            current_volume = int(hist["Volume"].iloc[0])
            
            # Average volume (excluding today)
            avg_volume = int(hist["Volume"].iloc[1:self.config["volume_lookback"]+1].mean())
            
            return {
                "price": current_price,
                "price_change": (current_price - prev_price) / prev_price,
                "volume": current_volume,
                "avg_volume": avg_volume
            }
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return None
    
    def determine_flow_regime(self, price_change: float, volume_ratio: float) -> str:
        """Determine flow regime from price change and volume ratio."""
        thresholds = self.config["regimes"]
        
        if price_change > 0:  # Price up
            if volume_ratio >= thresholds["strong_accumulation"]:
                return "Strong Accumulation"
            elif volume_ratio >= thresholds["accumulation"]:
                return "Accumulation"
            else:
                return "Neutral"
        else:  # Price down
            if volume_ratio >= thresholds["strong_distribution"]:
                return "Strong Distribution"
            elif volume_ratio >= thresholds["distribution"]:
                return "Distribution"
            else:
                return "Neutral"
    
    def calculate_stock_flows(self) -> List[StockFlow]:
        """Calculate flow for all stocks."""
        results = []
        
        for ticker in self.get_all_tickers():
            data = self.fetch_stock_data(ticker)
            
            if data is None:
                continue
            
            volume_ratio = data["volume"] / data["avg_volume"] if data["avg_volume"] > 0 else 1.0
            dollar_volume = data["price"] * data["volume"] / 1_000_000  # In millions
            
            flow_regime = self.determine_flow_regime(data["price_change"], volume_ratio)
            
            results.append(StockFlow(
                ticker=ticker,
                pillar=self.ticker_to_pillar[ticker],
                price=round(data["price"], 2),
                price_change_1d=round(data["price_change"] * 100, 2),
                volume=data["volume"],
                avg_volume=data["avg_volume"],
                volume_ratio=round(volume_ratio, 2),
                dollar_volume_m=round(dollar_volume, 1),
                flow_regime=flow_regime
            ))
        
        return results
    
    def aggregate_pillar_flows(self, stock_flows: List[StockFlow]) -> List[PillarFlow]:
        """Aggregate stock flows by pillar."""
        results = []
        
        for pillar in PILLAR_STOCKS.keys():
            pillar_stocks = [s for s in stock_flows if s.pillar == pillar]
            
            if not pillar_stocks:
                continue
            
            total = len(pillar_stocks)
            accumulating = sum(1 for s in pillar_stocks if "Accumulation" in s.flow_regime)
            distributing = sum(1 for s in pillar_stocks if "Distribution" in s.flow_regime)
            neutral = total - accumulating - distributing
            
            # Net flow: positive = more accumulation, negative = more distribution
            net_flow = (accumulating - distributing) / total if total > 0 else 0
            
            # Determine pillar regime
            thresholds = self.config["pillar_thresholds"]
            if net_flow >= thresholds["strong_inflow"]:
                regime = "Strong Inflow"
                signal = "Bullish"
            elif net_flow >= thresholds["inflow"]:
                regime = "Inflow"
                signal = "Mild Bullish"
            elif net_flow <= thresholds["strong_outflow"]:
                regime = "Strong Outflow"
                signal = "Bearish"
            elif net_flow <= thresholds["outflow"]:
                regime = "Outflow"
                signal = "Mild Bearish"
            else:
                regime = "Neutral"
                signal = "Neutral"
            
            results.append(PillarFlow(
                pillar=pillar,
                stock_count=total,
                accumulating=accumulating,
                distributing=distributing,
                neutral=neutral,
                net_flow=round(net_flow, 2),
                regime=regime,
                signal=signal
            ))
        
        return results
    
    def calculate(self) -> StockFlowDialData:
        """Main calculation: compute all flow metrics."""
        logger.info("Calculating stock flows...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate individual stock flows
        stock_flows = self.calculate_stock_flows()
        
        if not stock_flows:
            return StockFlowDialData(
                date=date_str,
                pillar_flows=[],
                stock_flows=[],
                overall_regime="Unknown",
                accumulating_count=0,
                distributing_count=0,
                interpretation="No stock data available"
            )
        
        # Aggregate by pillar
        pillar_flows = self.aggregate_pillar_flows(stock_flows)
        
        # Overall counts
        accumulating = sum(1 for s in stock_flows if "Accumulation" in s.flow_regime)
        distributing = sum(1 for s in stock_flows if "Distribution" in s.flow_regime)
        
        # Overall regime
        total = len(stock_flows)
        net_ratio = (accumulating - distributing) / total if total > 0 else 0
        
        if net_ratio > 0.20:
            overall_regime = "Strong Accumulation"
        elif net_ratio > 0.05:
            overall_regime = "Accumulation"
        elif net_ratio < -0.20:
            overall_regime = "Strong Distribution"
        elif net_ratio < -0.05:
            overall_regime = "Distribution"
        else:
            overall_regime = "Neutral"
        
        interpretation = f"{overall_regime}: {accumulating} stocks accumulating, {distributing} distributing out of {total}"
        
        result = StockFlowDialData(
            date=date_str,
            pillar_flows=[asdict(p) for p in pillar_flows],
            stock_flows=[asdict(s) for s in stock_flows],
            overall_regime=overall_regime,
            accumulating_count=accumulating,
            distributing_count=distributing,
            interpretation=interpretation
        )
        
        logger.info(f"Stock Flow: {overall_regime}")
        
        return result
    
    def save_to_supabase(self, data: StockFlowDialData) -> bool:
        """Save stock flow data to Supabase."""
        if not self.supabase:
            return False
        
        row = {
            "date": data.date,
            "pillar_flows": data.pillar_flows,
            "stock_flows": data.stock_flows,
            "overall_regime": data.overall_regime,
            "accumulating_count": data.accumulating_count,
            "distributing_count": data.distributing_count,
            "interpretation": data.interpretation
        }
        
        try:
            self.supabase.table("stock_flow_dial_daily") \
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
    
    parser = argparse.ArgumentParser(description="Stock Flow Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--detail", action="store_true", help="Show stock detail")
    args = parser.parse_args()
    
    calc = StockFlowCalculator()
    
    print(f"\n{'='*60}")
    print("STOCK FLOW DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"OVERALL: {data.overall_regime}")
    print(f"{'='*40}")
    print(f"  Accumulating: {data.accumulating_count}")
    print(f"  Distributing: {data.distributing_count}")
    
    print(f"\n{'='*40}")
    print("PILLAR FLOWS")
    print(f"{'='*40}")
    for pf in data.pillar_flows:
        print(f"  {pf['pillar']:15} {pf['regime']:15} ({pf['accumulating']}↑ {pf['distributing']}↓)")
    
    if args.detail:
        print(f"\n{'='*40}")
        print("STOCK DETAIL")
        print(f"{'='*40}")
        for sf in sorted(data.stock_flows, key=lambda x: x['volume_ratio'], reverse=True)[:15]:
            print(f"  {sf['ticker']:5} {sf['flow_regime']:20} Vol: {sf['volume_ratio']:.1f}x  {sf['price_change_1d']:+.1f}%")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
