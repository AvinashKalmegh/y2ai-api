"""
Pillar Breadth Divergence Backtest

Matches Google Sheets backtestPillarBreadthDivergences() methodology:
- Calculate pillar breadth (% of stocks above 20D MA)
- Find divergences: Infrastructure >60% AND Enterprise <40%
- Calculate forward breadth changes (5D, 10D, 20D)
- Generate summary statistics

Usage:
    python -m backtest.breadth_divergence --days 1000
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Universe mapping (matches Google Sheets Universe tab exactly)
# Infrastructure & Energy: 16 stocks
# Enterprise Adoption: 13 stocks
PILLARS_FOR_DIVERGENCE = {
    "Infrastructure & Energy": [
        "TSM", "ASML", "NVDA", "AMD", "MU", "AVGO", "VRT", "CEG", "NRG",
        "LRCX", "AMAT", "SMCI", "JKS", "RUN", "FSLR", "ENPH"
    ],
    "Enterprise Adoption": [
        "MSFT", "AMZN", "GOOGL", "SNOW", "PLTR", "ADBE", "ORCL", 
        "MDB", "DDOG", "ZS", "NET", "PANW", "CRWD"
    ]
}


@dataclass
class DivergenceSignal:
    """A single divergence instance"""
    date: str
    infra_breadth: float
    enterprise_breadth: float
    divergence_gap: float
    fwd_5d: Optional[float] = None
    fwd_10d: Optional[float] = None
    fwd_20d: Optional[float] = None


class BreadthDivergenceBacktest:
    """
    Backtest pillar breadth divergences against forward breadth changes.
    Matches Google Sheets backtestPillarBreadthDivergences() methodology.
    """
    
    def __init__(self, 
                 infra_threshold: float = 0.60,
                 enterprise_threshold: float = 0.40,
                 ma_period: int = 20):
        """
        Args:
            infra_threshold: Min Infrastructure breadth for divergence (default 60%)
            enterprise_threshold: Max Enterprise breadth for divergence (default 40%)
            ma_period: Moving average period for breadth calculation (default 20)
        """
        self.infra_threshold = infra_threshold
        self.enterprise_threshold = enterprise_threshold
        self.ma_period = ma_period
        
        self.prices_df: Optional[pd.DataFrame] = None
        self.breadth_df: Optional[pd.DataFrame] = None
        self.divergences: List[DivergenceSignal] = []
    
    def fetch_prices(self, start_date: str, end_date: str) -> bool:
        """Fetch historical prices for all universe stocks"""
        try:
            import yfinance as yf
            
            # Get all tickers from both pillars
            all_tickers = []
            for pillar, stocks in PILLARS_FOR_DIVERGENCE.items():
                all_tickers.extend(stocks)
            
            all_tickers = list(set(all_tickers))
            
            logger.info(f"Fetching prices for {len(all_tickers)} tickers...")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # Fetch all at once
            tickers_str = " ".join(all_tickers)
            data = yf.download(tickers_str, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.error("No price data returned")
                return False
            
            # Extract Close prices
            if isinstance(data.columns, pd.MultiIndex):
                self.prices_df = data['Close'].copy()
            else:
                self.prices_df = data.copy()
            
            # Fill missing values
            self.prices_df = self.prices_df.ffill().bfill()
            
            logger.info(f"Loaded {len(self.prices_df)} days of price data")
            logger.info(f"Date range: {self.prices_df.index[0].strftime('%Y-%m-%d')} to {self.prices_df.index[-1].strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_breadth(self) -> pd.DataFrame:
        """
        Calculate pillar breadth for each day.
        Breadth = % of stocks in pillar trading above their 20D MA
        """
        if self.prices_df is None:
            raise ValueError("No price data loaded")
        
        logger.info(f"Calculating {self.ma_period}D breadth...")
        
        # Calculate 20D MA for each stock
        ma_df = self.prices_df.rolling(window=self.ma_period).mean()
        
        # Calculate above/below MA for each stock
        above_ma = (self.prices_df > ma_df).astype(int)
        
        # Calculate breadth for each pillar
        breadth_data = {}
        
        for pillar, stocks in PILLARS_FOR_DIVERGENCE.items():
            # Filter to stocks we have data for
            available_stocks = [s for s in stocks if s in above_ma.columns]
            
            if available_stocks:
                # Breadth = sum of stocks above MA / total stocks
                pillar_above = above_ma[available_stocks].sum(axis=1)
                pillar_total = above_ma[available_stocks].notna().sum(axis=1)
                breadth_data[pillar] = pillar_above / pillar_total
                
                logger.info(f"  {pillar}: {len(available_stocks)} stocks")
            else:
                logger.warning(f"  {pillar}: No data available")
                breadth_data[pillar] = pd.Series(0, index=above_ma.index)
        
        self.breadth_df = pd.DataFrame(breadth_data)
        
        # Drop rows where MA isn't calculated yet
        self.breadth_df = self.breadth_df.iloc[self.ma_period:]
        
        logger.info(f"Calculated breadth for {len(self.breadth_df)} days")
        
        return self.breadth_df
    
    def find_divergences(self) -> List[DivergenceSignal]:
        """
        Find divergence instances where:
        - Infrastructure breadth >= 60%
        - Enterprise breadth <= 40%
        """
        if self.breadth_df is None:
            self.calculate_breadth()
        
        logger.info(f"Finding divergences (Infra >= {self.infra_threshold:.0%}, Enterprise <= {self.enterprise_threshold:.0%})...")
        
        self.divergences = []
        
        infra_col = "Infrastructure & Energy"
        ent_col = "Enterprise Adoption"
        
        # Need at least 20 days forward for outcome calculation
        for i in range(len(self.breadth_df) - 20):
            row = self.breadth_df.iloc[i]
            date = self.breadth_df.index[i]
            
            infra_breadth = row[infra_col]
            ent_breadth = row[ent_col]
            
            # Check divergence condition
            if infra_breadth >= self.infra_threshold and ent_breadth <= self.enterprise_threshold:
                # Calculate forward breadth changes (Infrastructure)
                # Note: In Google Sheets, forward = looking at LATER dates (more recent)
                # Since data is sorted ascending, forward = higher index
                
                fwd_5d = None
                fwd_10d = None
                fwd_20d = None
                
                if i + 5 < len(self.breadth_df):
                    fwd_5d = self.breadth_df.iloc[i + 5][infra_col] - infra_breadth
                
                if i + 10 < len(self.breadth_df):
                    fwd_10d = self.breadth_df.iloc[i + 10][infra_col] - infra_breadth
                
                if i + 20 < len(self.breadth_df):
                    fwd_20d = self.breadth_df.iloc[i + 20][infra_col] - infra_breadth
                
                signal = DivergenceSignal(
                    date=date.strftime("%Y-%m-%d"),
                    infra_breadth=infra_breadth,
                    enterprise_breadth=ent_breadth,
                    divergence_gap=infra_breadth - ent_breadth,
                    fwd_5d=fwd_5d,
                    fwd_10d=fwd_10d,
                    fwd_20d=fwd_20d
                )
                
                self.divergences.append(signal)
        
        logger.info(f"Found {len(self.divergences)} divergence instances")
        
        return self.divergences
    
    def get_summary(self) -> Dict:
        """Calculate summary statistics"""
        if not self.divergences:
            return {"total_divergences": 0}
        
        fwd_5d_values = [d.fwd_5d for d in self.divergences if d.fwd_5d is not None]
        fwd_10d_values = [d.fwd_10d for d in self.divergences if d.fwd_10d is not None]
        fwd_20d_values = [d.fwd_20d for d in self.divergences if d.fwd_20d is not None]
        
        summary = {
            "total_divergences": len(self.divergences),
            "avg_5d_delta": np.mean(fwd_5d_values) if fwd_5d_values else None,
            "avg_10d_delta": np.mean(fwd_10d_values) if fwd_10d_values else None,
            "avg_20d_delta": np.mean(fwd_20d_values) if fwd_20d_values else None,
            "pct_negative_5d": sum(1 for v in fwd_5d_values if v < 0) / len(fwd_5d_values) if fwd_5d_values else None,
            "pct_negative_10d": sum(1 for v in fwd_10d_values if v < 0) / len(fwd_10d_values) if fwd_10d_values else None,
            "pct_negative_20d": sum(1 for v in fwd_20d_values if v < 0) / len(fwd_20d_values) if fwd_20d_values else None,
        }
        
        return summary
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("PILLAR BREADTH DIVERGENCE BACKTEST")
        print("=" * 60)
        
        if self.breadth_df is not None:
            print(f"Period: {self.breadth_df.index[0].strftime('%Y-%m-%d')} to {self.breadth_df.index[-1].strftime('%Y-%m-%d')}")
            print(f"Total Trading Days: {len(self.breadth_df)}")
        
        print(f"\nDivergence Criteria:")
        print(f"  Infrastructure >= {self.infra_threshold:.0%}")
        print(f"  Enterprise <= {self.enterprise_threshold:.0%}")
        
        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)
        
        print(f"\nTotal Divergences Found: {summary['total_divergences']}")
        
        if summary['total_divergences'] > 0:
            print("\nForward Breadth Changes (Infrastructure):")
            
            if summary['avg_5d_delta'] is not None:
                print(f"  Avg 5D Δ:  {summary['avg_5d_delta']*100:+.2f}%  ({summary['pct_negative_5d']:.0%} negative)")
            
            if summary['avg_10d_delta'] is not None:
                print(f"  Avg 10D Δ: {summary['avg_10d_delta']*100:+.2f}%  ({summary['pct_negative_10d']:.0%} negative)")
            
            if summary['avg_20d_delta'] is not None:
                print(f"  Avg 20D Δ: {summary['avg_20d_delta']*100:+.2f}%  ({summary['pct_negative_20d']:.0%} negative)")
            
            print("\nInterpretation:")
            if summary['avg_20d_delta'] is not None and summary['avg_20d_delta'] < -0.10:
                print("  → Divergence predicts significant breadth contraction")
                print("  → When Infra leads and Enterprise lags, expect reversion")
            elif summary['avg_20d_delta'] is not None and summary['avg_20d_delta'] < 0:
                print("  → Divergence shows mild predictive signal")
            else:
                print("  → No clear predictive pattern")
        
        print("\n" + "=" * 60)


def run_backtest(start_date: str = None, end_date: str = None, days: int = 1000,
                 save_to_supabase: bool = True):
    """Run breadth divergence backtest"""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date is None:
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)
        start_date = start_dt.strftime("%Y-%m-%d")
    
    backtest = BreadthDivergenceBacktest(
        infra_threshold=0.60,
        enterprise_threshold=0.40,
        ma_period=20
    )
    
    # Fetch data
    if not backtest.fetch_prices(start_date, end_date):
        logger.error("Failed to fetch prices")
        return None
    
    # Calculate breadth
    backtest.calculate_breadth()
    
    # Find divergences
    backtest.find_divergences()
    
    # Print summary
    backtest.print_summary()
    
    # Save to Supabase
    if save_to_supabase:
        try:
            from .storage import get_backtest_storage
            
            storage = get_backtest_storage()
            if storage.is_connected():
                run_id = storage.create_run(
                    run_type="breadth_divergence",
                    config={
                        "infra_threshold": backtest.infra_threshold,
                        "enterprise_threshold": backtest.enterprise_threshold,
                        "ma_period": backtest.ma_period,
                        "days": days
                    }
                )
                
                if run_id:
                    saved_count = storage.save_divergences(run_id, backtest.divergences)
                    
                    summary = backtest.get_summary()
                    period_start = backtest.breadth_df.index[0].strftime("%Y-%m-%d")
                    period_end = backtest.breadth_df.index[-1].strftime("%Y-%m-%d")
                    storage.save_divergence_summary(run_id, summary, period_start, period_end)
                    
                    storage.complete_run(run_id)
                    print(f"\n✅ Saved {saved_count} divergences to Supabase (run_id={run_id})")
            else:
                logger.warning("Supabase not connected - results not saved to database")
        except Exception as e:
            logger.error(f"Failed to save to Supabase: {e}")
    
    return backtest


def main():
    parser = argparse.ArgumentParser(description="Run pillar breadth divergence backtest")
    parser.add_argument("--days", type=int, default=1000, help="Number of days to backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to Supabase")
    
    args = parser.parse_args()
    
    run_backtest(start_date=args.start, end_date=args.end, days=args.days,
                 save_to_supabase=not args.no_save)


if __name__ == "__main__":
    main()