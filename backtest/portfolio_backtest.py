"""
Portfolio Rotation Backtest Module

Matches Google Sheets backtest methodology:
- Load historical prices for Y2AI universe stocks
- Calculate daily pillar returns
- Test equal weight vs rotation strategy
- Compare against SPY benchmark
- Generate performance metrics (CAGR, Sharpe, Max DD)

Usage:
    python -m backtest.portfolio_backtest --days 365
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Y2AI Universe - 6 Pillars
UNIVERSE = {
    "Infrastructure & Energy": ["TSM", "ASML", "VRT", "EQIX", "DLR"],
    "Enterprise Adoption": ["MSFT", "CRM", "NOW", "ORCL"],
    "Macro Indicators": ["GOOGL", "META", "AMZN"],
    "Financial Flows": ["AVGO", "AMD", "INTC"],
    "Productivity": ["NVDA", "PLTR"],
    "Demand Depth": ["SNOW"],
}

# Simplified 3-pillar structure (matching your current setup)
PILLARS_3 = {
    "Supply Constraint": ["TSM", "ASML", "VRT"],
    "Capital Efficiency": ["GOOGL", "MSFT", "AMZN"],
    "Demand Depth": ["NVDA", "SNOW", "NOW"],
}


@dataclass
class BacktestConfig:
    """Configuration for portfolio backtest"""
    start_date: str
    end_date: str
    initial_capital: float = 100.0
    pillars: Dict[str, List[str]] = field(default_factory=lambda: PILLARS_3)
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    rotation_lookback: int = 20  # Days for momentum calculation
    overweight_factor: float = 2.0  # Weight multiplier for top pillars
    top_n_pillars: int = 2  # Number of pillars to overweight


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    total_days: int = 0


class PortfolioBacktest:
    """
    Portfolio rotation backtest matching Google Sheets methodology.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config
        self.prices_df: Optional[pd.DataFrame] = None
        self.pillar_returns_df: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
    
    def fetch_prices(self, start_date: str, end_date: str) -> bool:
        """Fetch historical prices for all universe stocks + SPY"""
        try:
            import yfinance as yf
            
            # Get all unique tickers
            all_tickers = set()
            for stocks in self.config.pillars.values():
                all_tickers.update(stocks)
            all_tickers.add("SPY")  # Add benchmark
            
            logger.info(f"Fetching prices for {len(all_tickers)} tickers...")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # Fetch all at once
            tickers_str = " ".join(all_tickers)
            data = yf.download(tickers_str, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.error("No price data returned")
                return False
            
            # Extract Close prices
            if 'Close' in data.columns.get_level_values(0):
                self.prices_df = data['Close'].copy()
            else:
                self.prices_df = data.copy()
            
            # Fill missing values forward
            self.prices_df = self.prices_df.ffill().bfill()
            
            logger.info(f"Loaded {len(self.prices_df)} days of price data")
            logger.info(f"Tickers: {list(self.prices_df.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return False
    
    def calculate_pillar_returns(self) -> pd.DataFrame:
        """Calculate daily returns for each pillar (equal-weighted within pillar)"""
        if self.prices_df is None:
            raise ValueError("No price data loaded")
        
        # Calculate daily returns for each stock
        stock_returns = self.prices_df.pct_change()
        
        # Calculate pillar returns (average of constituent stocks)
        pillar_returns = pd.DataFrame(index=stock_returns.index)
        
        for pillar_name, stocks in self.config.pillars.items():
            # Filter to stocks we have data for
            available_stocks = [s for s in stocks if s in stock_returns.columns]
            if available_stocks:
                pillar_returns[pillar_name] = stock_returns[available_stocks].mean(axis=1)
            else:
                logger.warning(f"No data for pillar: {pillar_name}")
                pillar_returns[pillar_name] = 0
        
        # Add SPY returns
        if "SPY" in stock_returns.columns:
            pillar_returns["SPY"] = stock_returns["SPY"]
        
        self.pillar_returns_df = pillar_returns.dropna()
        
        logger.info(f"Calculated returns for {len(self.config.pillars)} pillars")
        
        return self.pillar_returns_df
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run full backtest with equal weight and rotation strategies.
        
        Returns DataFrame with:
        - Date
        - Pillar returns
        - SPY return
        - Equal weight portfolio return & NAV
        - Rotation portfolio return & NAV
        """
        if self.pillar_returns_df is None:
            self.calculate_pillar_returns()
        
        returns = self.pillar_returns_df.copy()
        pillar_names = [c for c in returns.columns if c != "SPY"]
        n_pillars = len(pillar_names)
        
        logger.info(f"Running backtest with {n_pillars} pillars...")
        
        # Initialize results
        results = pd.DataFrame(index=returns.index)
        
        # Copy pillar returns
        for pillar in pillar_names:
            results[f"{pillar}_Ret"] = returns[pillar]
        
        results["SPY_Ret"] = returns["SPY"]
        
        # === EQUAL WEIGHT STRATEGY ===
        equal_weight = 1.0 / n_pillars
        results["EW_Return"] = sum(returns[p] * equal_weight for p in pillar_names)
        results["EW_NAV"] = (1 + results["EW_Return"]).cumprod() * self.config.initial_capital
        
        # === ROTATION STRATEGY ===
        # Overweight top N pillars by trailing momentum
        rotation_returns = []
        
        for i in range(len(returns)):
            if i < self.config.rotation_lookback:
                # Not enough history, use equal weight
                daily_ret = sum(returns[p].iloc[i] * equal_weight for p in pillar_names)
            else:
                # Calculate trailing momentum for each pillar
                lookback_start = i - self.config.rotation_lookback
                momentum = {}
                for pillar in pillar_names:
                    momentum[pillar] = returns[pillar].iloc[lookback_start:i].sum()
                
                # Rank pillars
                sorted_pillars = sorted(momentum.keys(), key=lambda x: momentum[x], reverse=True)
                top_pillars = sorted_pillars[:self.config.top_n_pillars]
                
                # Calculate weights
                # Top N get overweight_factor units, rest get 1 unit
                total_units = (self.config.top_n_pillars * self.config.overweight_factor + 
                              (n_pillars - self.config.top_n_pillars) * 1.0)
                
                weights = {}
                for pillar in pillar_names:
                    if pillar in top_pillars:
                        weights[pillar] = self.config.overweight_factor / total_units
                    else:
                        weights[pillar] = 1.0 / total_units
                
                # Calculate weighted return
                daily_ret = sum(returns[p].iloc[i] * weights[p] for p in pillar_names)
            
            rotation_returns.append(daily_ret)
        
        results["Rotation_Return"] = rotation_returns
        results["Rotation_NAV"] = (1 + results["Rotation_Return"]).cumprod() * self.config.initial_capital
        
        # === SPY BENCHMARK ===
        results["SPY_NAV"] = (1 + results["SPY_Ret"]).cumprod() * self.config.initial_capital
        
        # === EXCESS RETURNS ===
        results["EW_Excess"] = results["EW_Return"] - results["SPY_Ret"]
        results["Rotation_Excess"] = results["Rotation_Return"] - results["SPY_Ret"]
        results["EW_Cumul_Excess"] = results["EW_NAV"] - results["SPY_NAV"]
        results["Rotation_Cumul_Excess"] = results["Rotation_NAV"] - results["SPY_NAV"]
        
        self.results_df = results
        
        logger.info("Backtest complete")
        
        return results
    
    def calculate_metrics(self, return_col: str) -> PerformanceMetrics:
        """Calculate performance metrics for a return series"""
        if self.results_df is None:
            raise ValueError("No backtest results")
        
        returns = self.results_df[return_col].dropna()
        
        if len(returns) < 2:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        metrics.total_days = len(returns)
        
        # Total return
        nav_col = return_col.replace("_Return", "_NAV").replace("_Ret", "_NAV")
        if nav_col in self.results_df.columns:
            final_nav = self.results_df[nav_col].iloc[-1]
            metrics.total_return = (final_nav / self.config.initial_capital) - 1
        else:
            metrics.total_return = (1 + returns).prod() - 1
        
        # CAGR
        years = metrics.total_days / 252
        if years > 0:
            metrics.cagr = (1 + metrics.total_return) ** (1 / years) - 1
        
        # Volatility (annualized)
        metrics.volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        if metrics.volatility > 0:
            metrics.sharpe_ratio = metrics.cagr / metrics.volatility
        
        # Max Drawdown
        if nav_col in self.results_df.columns:
            nav = self.results_df[nav_col]
            rolling_max = nav.expanding().max()
            drawdown = (nav - rolling_max) / rolling_max
            metrics.max_drawdown = drawdown.min()
        
        # Win rate
        metrics.win_rate = (returns > 0).mean()
        
        # Best/worst day
        metrics.best_day = returns.max()
        metrics.worst_day = returns.min()
        
        return metrics
    
    def get_summary(self) -> str:
        """Generate text summary of backtest results"""
        if self.results_df is None:
            return "No backtest results"
        
        ew_metrics = self.calculate_metrics("EW_Return")
        rot_metrics = self.calculate_metrics("Rotation_Return")
        spy_metrics = self.calculate_metrics("SPY_Ret")
        
        lines = [
            "=" * 60,
            "PORTFOLIO ROTATION BACKTEST RESULTS",
            "=" * 60,
            f"Period: {self.results_df.index[0].strftime('%Y-%m-%d')} to {self.results_df.index[-1].strftime('%Y-%m-%d')}",
            f"Total Days: {ew_metrics.total_days}",
            f"Pillars: {len(self.config.pillars)}",
            "",
            "-" * 60,
            "EQUAL WEIGHT PORTFOLIO",
            "-" * 60,
            f"  Total Return: {ew_metrics.total_return:.2%}",
            f"  CAGR: {ew_metrics.cagr:.2%}",
            f"  Volatility: {ew_metrics.volatility:.2%}",
            f"  Sharpe Ratio: {ew_metrics.sharpe_ratio:.2f}",
            f"  Max Drawdown: {ew_metrics.max_drawdown:.2%}",
            f"  Win Rate: {ew_metrics.win_rate:.1%}",
            "",
            "-" * 60,
            "ROTATION PORTFOLIO (Momentum)",
            "-" * 60,
            f"  Total Return: {rot_metrics.total_return:.2%}",
            f"  CAGR: {rot_metrics.cagr:.2%}",
            f"  Volatility: {rot_metrics.volatility:.2%}",
            f"  Sharpe Ratio: {rot_metrics.sharpe_ratio:.2f}",
            f"  Max Drawdown: {rot_metrics.max_drawdown:.2%}",
            f"  Win Rate: {rot_metrics.win_rate:.1%}",
            "",
            "-" * 60,
            "SPY BENCHMARK",
            "-" * 60,
            f"  Total Return: {spy_metrics.total_return:.2%}",
            f"  CAGR: {spy_metrics.cagr:.2%}",
            f"  Volatility: {spy_metrics.volatility:.2%}",
            f"  Sharpe Ratio: {spy_metrics.sharpe_ratio:.2f}",
            f"  Max Drawdown: {spy_metrics.max_drawdown:.2%}",
            "",
            "-" * 60,
            "EXCESS RETURNS vs SPY",
            "-" * 60,
            f"  Equal Weight Alpha: {ew_metrics.total_return - spy_metrics.total_return:.2%}",
            f"  Rotation Alpha: {rot_metrics.total_return - spy_metrics.total_return:.2%}",
            f"  Rotation vs Equal Weight: {rot_metrics.total_return - ew_metrics.total_return:.2%}",
            "=" * 60,
        ]
        
        return "\n".join(lines)


def run_backtest(start_date: str = None, end_date: str = None, days: int = 365,
                 save_to_supabase: bool = True):
    """Run portfolio rotation backtest"""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date is None:
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)
        start_date = start_dt.strftime("%Y-%m-%d")
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        pillars=PILLARS_3,
        rotation_lookback=20,
        overweight_factor=2.0,
        top_n_pillars=1  # Overweight top 1 pillar
    )
    
    backtest = PortfolioBacktest(config)
    
    # Fetch data
    if not backtest.fetch_prices(start_date, end_date):
        logger.error("Failed to fetch prices")
        return None
    
    # Run backtest
    backtest.run_backtest()
    
    # Print summary
    print(backtest.get_summary())
    
    # Save to Supabase
    if save_to_supabase:
        try:
            from .storage import get_backtest_storage
            
            storage = get_backtest_storage()
            if storage.is_connected() and backtest.results_df is not None:
                run_id = storage.create_run(
                    run_type="portfolio_rotation",
                    config={
                        "pillars": list(config.pillars.keys()),
                        "rotation_lookback": config.rotation_lookback,
                        "overweight_factor": config.overweight_factor,
                        "top_n_pillars": config.top_n_pillars,
                        "days": days
                    }
                )
                
                if run_id:
                    # Prepare daily data
                    daily_data = []
                    for idx, row in backtest.results_df.iterrows():
                        daily_data.append({
                            "date": idx.strftime("%Y-%m-%d"),
                            "equal_weight_value": row.get("EW_NAV"),
                            "rotation_value": row.get("Rotation_NAV"),
                            "spy_value": row.get("SPY_NAV"),
                            "equal_weight_return": row.get("EW_Return"),
                            "rotation_return": row.get("Rotation_Return"),
                            "spy_return": row.get("SPY_Ret"),
                        })
                    
                    saved_count = storage.save_portfolio_daily(run_id, daily_data)
                    
                    # Save metrics
                    period_start = backtest.results_df.index[0].strftime("%Y-%m-%d")
                    period_end = backtest.results_df.index[-1].strftime("%Y-%m-%d")
                    
                    ew_metrics = backtest.calculate_metrics("EW_Return")
                    rot_metrics = backtest.calculate_metrics("Rotation_Return")
                    spy_metrics = backtest.calculate_metrics("SPY_Ret")
                    
                    storage.save_portfolio_metrics(run_id, "equal_weight", ew_metrics,
                                                   period_start, period_end,
                                                   ew_metrics.cagr - spy_metrics.cagr)
                    storage.save_portfolio_metrics(run_id, "rotation", rot_metrics,
                                                   period_start, period_end,
                                                   rot_metrics.cagr - spy_metrics.cagr)
                    storage.save_portfolio_metrics(run_id, "spy", spy_metrics,
                                                   period_start, period_end, 0.0)
                    
                    storage.complete_run(run_id)
                    print(f"\n✅ Saved {saved_count} daily records to Supabase (run_id={run_id})")
            elif not storage.is_connected():
                logger.warning("Supabase not connected - results not saved to database")
        except Exception as e:
            logger.error(f"Failed to save to Supabase: {e}")
    
    return backtest


def main():
    parser = argparse.ArgumentParser(description="Run portfolio rotation backtest")
    parser.add_argument("--days", type=int, default=365, help="Number of days to backtest")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to Supabase")
    
    args = parser.parse_args()
    
    run_backtest(start_date=args.start, end_date=args.end, days=args.days,
                 save_to_supabase=not args.no_save)


if __name__ == "__main__":
    main()