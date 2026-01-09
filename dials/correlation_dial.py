"""
CORRELATION DIAL - Pillar-Level Correlations
=============================================
Calculates rolling correlations between the 6 pillars to detect
correlation collapse (precursor to bifurcation events).

Logic:
- Calculate 15 pairwise correlations between 6 pillars
- Average across all pairs for single correlation metric
- Track both 20D (fast) and 60D (slow) windows
- High correlation = herding behavior = systemic risk

Regime Thresholds (based on average correlation):
- Healthy: < 30% (normal diversification)
- Caution: 30-50% (market still rotational but weakening)
- Fragile: 50-70% (elevated correlations, watch spreads + VIX)
- Stressed: 70-90% (high systemic beta, little diversification)
- Collapse: > 90% (liquidity fragmentation + bifurcation risk)

Historical Context:
- 2008 Subprime: Correlations → 0.95+
- 2020 COVID: Correlations → 0.92
- 2022 Tech Unwind: Correlations → 0.85
- Normal Markets: Correlations 0.10 - 0.40
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from itertools import combinations

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CORRELATION_CONFIG = {
    # Rolling windows
    "windows": {
        "fast": 20,   # 20-day correlation
        "slow": 60,   # 60-day correlation
    },
    
    # Regime thresholds (correlation as decimal)
    "regime_thresholds": {
        "healthy": 0.30,     # < 30%
        "caution": 0.50,     # 30-50%
        "fragile": 0.70,     # 50-70%
        "stressed": 0.90,    # 70-90%
        # > 90% = Collapse
    },
    
    # Pillar names (must match pillar_index table)
    "pillars": [
        "Infrastructure",
        "Enterprise",
        "Financial",
        "Macro",
        "Productivity",
        "Demand",
    ],
}

# Pillar to stocks mapping for calculating pillar returns
PILLAR_STOCKS = {
    "Infrastructure": ["TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT",
                      "CEG", "NRG", "EQIX", "DLR", "KLAC", "LRCX", "AMAT", "QCOM"],
    "Enterprise": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR",
                  "ADBE", "ORCL", "MDB", "DDOG", "ZS"],
    "Financial": ["GS", "MS", "JKS", "FSLR"],
    "Macro": ["NXPI", "ON", "SMCI", "ARM"],
    "Productivity": ["NET", "CRWD", "PANW"],
    "Demand": ["TSLA", "SHOP", "UBER"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CorrelationWindow:
    """Correlation data for a single window."""
    window_days: int
    mean_correlation: float
    regime: str
    notes: str
    pairwise_correlations: Dict[str, float]  # "Pillar1-Pillar2": correlation


@dataclass
class CorrelationDialData:
    """Complete correlation dial data."""
    date: str
    # 20-day window
    corr_20d: float
    regime_20d: str
    # 60-day window
    corr_60d: float
    regime_60d: str
    # Combined
    combined_signal: str
    action: str
    interpretation: str
    # Detail
    pairwise_20d: Dict[str, float]
    pairwise_60d: Dict[str, float]


# =============================================================================
# CORRELATION CALCULATOR
# =============================================================================

class CorrelationDialCalculator:
    """
    Calculate pillar-level correlations.
    
    Matches Google Sheets Correlation_Dials logic exactly:
    - 15 pairwise correlations between 6 pillars
    - 20D and 60D rolling windows
    - Regime based on average correlation level
    
    Usage:
        calc = CorrelationDialCalculator()
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
        
        self.config = CORRELATION_CONFIG
        self.pillars = self.config["pillars"]
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_pillar_returns(self, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Fetch pillar-level daily returns.
        
        First tries pillar_index_daily table, then calculates from stock prices.
        
        Returns:
            DataFrame with columns for each pillar, indexed by date
        """
        # Try fetching from pillar_index_daily table
        if self.supabase:
            try:
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                
                response = self.supabase.table("pillar_index_daily") \
                    .select("date, pillar, daily_return") \
                    .gte("date", start_date) \
                    .order("date", desc=False) \
                    .execute()
                
                if response.data and len(response.data) > 0:
                    df = pd.DataFrame(response.data)
                    df["date"] = pd.to_datetime(df["date"])
                    
                    # Pivot to get pillars as columns
                    pivot = df.pivot_table(
                        index="date",
                        columns="pillar",
                        values="daily_return"
                    )
                    
                    if len(pivot) > 20:
                        logger.info(f"Fetched {len(pivot)} days of pillar returns from pillar_index_daily")
                        return pivot
            except Exception as e:
                logger.warning(f"Could not fetch pillar_index_daily: {e}")
        
        # Fallback: calculate from stock prices
        return self._calculate_pillar_returns_from_stocks(days)
    
    def _calculate_pillar_returns_from_stocks(self, days: int = 90) -> Optional[pd.DataFrame]:
        """Calculate pillar returns from individual stock prices."""
        logger.info("Calculating pillar returns from stock prices...")
        
        # Try yfinance
        try:
            import yfinance as yf
            
            all_tickers = []
            for stocks in PILLAR_STOCKS.values():
                all_tickers.extend(stocks)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)
            
            # Fetch all stock prices
            data = yf.download(
                all_tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )
            
            if data.empty:
                return None
            
            # Get close prices
            if "Close" in data.columns:
                closes = data["Close"]
            else:
                closes = data
            
            # Calculate daily returns
            returns = closes.pct_change().dropna()
            
            # Calculate pillar returns (equal-weighted average of constituents)
            pillar_returns = pd.DataFrame(index=returns.index)
            
            for pillar, stocks in PILLAR_STOCKS.items():
                available = [s for s in stocks if s in returns.columns]
                if available:
                    pillar_returns[pillar] = returns[available].mean(axis=1)
            
            logger.info(f"Calculated {len(pillar_returns)} days of pillar returns from {len(all_tickers)} stocks")
            return pillar_returns
            
        except ImportError:
            logger.warning("yfinance not available")
        except Exception as e:
            logger.error(f"Error calculating pillar returns: {e}")
        
        return None
    
    # =========================================================================
    # CORRELATION CALCULATIONS
    # =========================================================================
    
    def calculate_pairwise_correlations(
        self, 
        returns: pd.DataFrame, 
        window: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate all 15 pairwise correlations between 6 pillars.
        
        Args:
            returns: DataFrame of pillar returns
            window: Number of days for rolling window
            
        Returns:
            Tuple of (average_correlation, {pair_name: correlation})
        """
        if len(returns) < window:
            logger.warning(f"Not enough data: {len(returns)} days < {window} window")
            return 0.0, {}
        
        # Get last 'window' days
        recent = returns.tail(window)
        
        # Get available pillars
        available_pillars = [p for p in self.pillars if p in recent.columns]
        
        if len(available_pillars) < 2:
            logger.warning(f"Not enough pillars: {len(available_pillars)}")
            return 0.0, {}
        
        # Calculate correlation matrix
        corr_matrix = recent[available_pillars].corr()
        
        # Extract all pairwise correlations
        pairwise = {}
        correlations = []
        
        for p1, p2 in combinations(available_pillars, 2):
            corr = corr_matrix.loc[p1, p2]
            if not pd.isna(corr):
                pair_name = f"{p1}-{p2}"
                pairwise[pair_name] = round(corr, 4)
                correlations.append(corr)
        
        # Calculate average
        avg_corr = np.mean(correlations) if correlations else 0.0
        
        return round(avg_corr, 4), pairwise
    
    def get_regime(self, correlation: float) -> Tuple[str, str]:
        """
        Determine regime from correlation level.
        
        Returns:
            Tuple of (regime, notes)
        """
        thresholds = self.config["regime_thresholds"]
        
        if correlation < thresholds["healthy"]:
            return "Healthy", "Normal diversification"
        elif correlation < thresholds["caution"]:
            return "Caution", "Market still rotational but weakening"
        elif correlation < thresholds["fragile"]:
            return "Fragile", "Elevated correlations. Watch spreads + VIX"
        elif correlation < thresholds["stressed"]:
            return "Stressed", "High systemic beta. Little diversification"
        else:
            return "Collapse", "Liquidity fragmentation + system bifurcation risk"
    
    def get_combined_signal(
        self, 
        regime_20d: str, 
        regime_60d: str
    ) -> Tuple[str, str]:
        """
        Determine combined signal from both windows.
        
        Returns:
            Tuple of (signal, action)
        """
        # Define severity order
        severity = {
            "Healthy": 0,
            "Caution": 1,
            "Fragile": 2,
            "Stressed": 3,
            "Collapse": 4,
        }
        
        sev_20 = severity.get(regime_20d, 0)
        sev_60 = severity.get(regime_60d, 0)
        
        # Both collapsed
        if regime_20d == "Collapse" and regime_60d == "Collapse":
            return "CRITICAL - Both windows collapsed", "Reduce equity exposure. Increase cash/quality."
        
        # One collapsed
        if regime_20d == "Collapse" or regime_60d == "Collapse":
            return "WARNING - One window collapsed", "Reduce equity exposure. Increase cash/quality."
        
        # Both stressed
        if regime_20d == "Stressed" and regime_60d == "Stressed":
            return "HIGH ALERT - Both windows stressed", "Tighten stops. Reduce position sizes."
        
        # One stressed
        if regime_20d == "Stressed" or regime_60d == "Stressed":
            return "ELEVATED - One window stressed", "Tighten stops. Reduce position sizes."
        
        # Both fragile
        if regime_20d == "Fragile" and regime_60d == "Fragile":
            return "CAUTION - Both windows fragile", "Monitor closely. Prepare contingency."
        
        # Default
        return "NORMAL", "Normal operations. Follow rotation signals."
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> CorrelationDialData:
        """
        Calculate correlation dial metrics.
        
        Returns:
            CorrelationDialData with all metrics
        """
        logger.info("Calculating correlation dial...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        windows = self.config["windows"]
        
        # Fetch pillar returns (need enough for 60D window)
        returns = self.fetch_pillar_returns(days=90)
        
        if returns is None or len(returns) < windows["fast"]:
            logger.error("Insufficient data for correlation calculation")
            return CorrelationDialData(
                date=date_str,
                corr_20d=0.0,
                regime_20d="Unknown",
                corr_60d=0.0,
                regime_60d="Unknown",
                combined_signal="No Data",
                action="Check data sources",
                interpretation="Insufficient data for correlation calculation",
                pairwise_20d={},
                pairwise_60d={},
            )
        
        # Calculate 20D correlations
        corr_20d, pairwise_20d = self.calculate_pairwise_correlations(returns, windows["fast"])
        regime_20d, notes_20d = self.get_regime(corr_20d)
        
        # Calculate 60D correlations
        corr_60d, pairwise_60d = self.calculate_pairwise_correlations(returns, windows["slow"])
        regime_60d, notes_60d = self.get_regime(corr_60d)
        
        # Get combined signal
        combined_signal, action = self.get_combined_signal(regime_20d, regime_60d)
        
        # Build interpretation
        interpretation = f"20D: {corr_20d*100:.1f}% ({regime_20d}), 60D: {corr_60d*100:.1f}% ({regime_60d})"
        
        logger.info(f"Correlation 20D: {corr_20d*100:.1f}% ({regime_20d})")
        logger.info(f"Correlation 60D: {corr_60d*100:.1f}% ({regime_60d})")
        logger.info(f"Combined: {combined_signal}")
        
        return CorrelationDialData(
            date=date_str,
            corr_20d=corr_20d,
            regime_20d=regime_20d,
            corr_60d=corr_60d,
            regime_60d=regime_60d,
            combined_signal=combined_signal,
            action=action,
            interpretation=interpretation,
            pairwise_20d=pairwise_20d,
            pairwise_60d=pairwise_60d,
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: CorrelationDialData) -> Optional[Dict]:
        """Save correlation data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = {
            "date": data.date,
            "corr_20d": data.corr_20d,
            "regime_20d": data.regime_20d,
            "corr_60d": data.corr_60d,
            "regime_60d": data.regime_60d,
            "combined_signal": data.combined_signal,
            "action": data.action,
            "interpretation": data.interpretation,
            "pairwise_20d": data.pairwise_20d,
            "pairwise_60d": data.pairwise_60d,
        }
        
        try:
            response = self.supabase.table("correlation_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved correlation data for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save correlation data: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run correlation dial calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Correlation Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = CorrelationDialCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"CORRELATION DIAL")
    print(f"{'='*60}")
    print(f"\n20-Day Window:")
    print(f"  Average Correlation: {data.corr_20d*100:.1f}%")
    print(f"  Regime: {data.regime_20d}")
    print(f"\n60-Day Window:")
    print(f"  Average Correlation: {data.corr_60d*100:.1f}%")
    print(f"  Regime: {data.regime_60d}")
    print(f"\nCombined Signal: {data.combined_signal}")
    print(f"Action: {data.action}")
    
    if data.pairwise_20d:
        print(f"\n20D Pairwise Correlations:")
        for pair, corr in sorted(data.pairwise_20d.items(), key=lambda x: -x[1])[:5]:
            print(f"  {pair}: {corr*100:.1f}%")
    
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
