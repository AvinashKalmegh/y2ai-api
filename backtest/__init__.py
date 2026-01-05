"""
Y2AI Backtest Module

Matches Google Sheets backtest functionality:
- Breadth Divergence: backtestPillarBreadthDivergences()
- Portfolio Rotation: buildBacktestEngine()
- Historical Backfill: VIX/CAPE/Spreads data

Usage:
    python -m backtest.breadth_divergence --days 1000
    python -m backtest.portfolio_backtest --days 365
    python -m backtest.history --days 365
"""

from .breadth_divergence import BreadthDivergenceBacktest, run_backtest as run_divergence_backtest
from .portfolio_backtest import PortfolioBacktest, BacktestConfig, run_backtest as run_portfolio_backtest
from .history import HistoricalDataBackfill
from .storage import BacktestStorage, get_backtest_storage

__all__ = [
    "BreadthDivergenceBacktest",
    "PortfolioBacktest",
    "BacktestConfig",
    "HistoricalDataBackfill",
    "BacktestStorage",
    "get_backtest_storage",
    "run_divergence_backtest",
    "run_portfolio_backtest",
]
