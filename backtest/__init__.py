"""
ARGUS-1 Backtesting Module

Test ARGUS-1 regime signals against historical market data.

Usage:
    from backtest import BacktestRunner
    
    runner = BacktestRunner()
    results = runner.run(start_date="2024-01-01", end_date="2024-12-01")
    
    print(results.summary())
    results.export_report("backtest_results.html")
"""

from .data_loader import HistoricalDataLoader
from .outcomes import OutcomeCalculator
from .performance import PerformanceAnalyzer
from .runner import BacktestRunner
from .reports import ReportGenerator

__all__ = [
    "HistoricalDataLoader",
    "OutcomeCalculator", 
    "PerformanceAnalyzer",
    "BacktestRunner",
    "ReportGenerator",
]
