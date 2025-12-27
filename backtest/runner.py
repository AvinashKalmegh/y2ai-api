"""
ARGUS-1 Backtest Runner

Main orchestrator for backtesting ARGUS-1 regime signals against
historical market data.

Usage:
    from backtest import BacktestRunner
    
    runner = BacktestRunner()
    results = runner.run(start_date="2024-01-01", end_date="2024-12-01")
    
    print(results.summary())
    results.export_csv("backtest_signals.csv")
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .data_loader import HistoricalDataLoader, PointInTimeData
from .outcomes import OutcomeCalculator, MarketOutcome
from .performance import PerformanceAnalyzer, BacktestMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    start_date: str
    end_date: str
    benchmark: str = "SPY"
    
    # AMRI thresholds (tuned for better signal generation)
    # Lower thresholds = more sensitive to risk
    amri_normal: float = 30.0      # Was 40
    amri_elevated: float = 42.0    # Was 55
    amri_tension: float = 55.0     # Was 70
    amri_fragile: float = 70.0     # Was 85
    
    # VETO threshold
    veto_threshold: int = 25
    
    # Contagion threshold
    contagion_danger: float = 60.0  # Was 70
    
    # VIX thresholds for regime escalation
    vix_elevated: float = 18.0
    vix_tension: float = 22.0
    vix_fragile: float = 28.0


class BacktestResult:
    """Complete backtest result container"""
    
    def __init__(self, config: BacktestConfig, metrics: BacktestMetrics,
                 analyzer: PerformanceAnalyzer):
        self.config = config
        self.metrics = metrics
        self.analyzer = analyzer
    
    def summary(self) -> str:
        """Get text summary of results"""
        return self.metrics.summary()
    
    def export_csv(self, filepath: str):
        """Export signals to CSV"""
        df = self.analyzer.get_signals_df()
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Exported {len(df)} signals to {filepath}")
    
    def get_dataframe(self):
        """Get signals as DataFrame"""
        return self.analyzer.get_signals_df()


class BacktestRunner:
    """
    Run ARGUS-1 backtests against historical data.
    
    Simulates what ARGUS-1 would have signaled on each historical
    date and compares against actual market outcomes.
    """
    
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        self.data_loader: Optional[HistoricalDataLoader] = None
        self.outcome_calc: Optional[OutcomeCalculator] = None
        self.analyzer: Optional[PerformanceAnalyzer] = None
    
    def run(self, start_date: str, end_date: str,
            benchmark: str = "SPY",
            config: Optional[BacktestConfig] = None) -> BacktestResult:
        """
        Run full backtest.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            benchmark: Ticker for outcome calculation (default SPY)
            config: Optional custom configuration
        
        Returns:
            BacktestResult with metrics and signals
        """
        if config is None:
            config = BacktestConfig(start_date=start_date, end_date=end_date, benchmark=benchmark)
        
        logger.info("=" * 60)
        logger.info("ARGUS-1 BACKTEST")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Benchmark: {benchmark}")
        logger.info("=" * 60)
        
        # Initialize components
        self.data_loader = HistoricalDataLoader(self.client)
        self.outcome_calc = OutcomeCalculator(benchmark=benchmark)
        self.analyzer = PerformanceAnalyzer()
        
        # Load historical data
        logger.info("Loading historical data...")
        if not self.data_loader.load_all_data(start_date, end_date):
            raise RuntimeError("Failed to load historical data")
        
        # Load market prices for outcomes
        logger.info("Loading market prices...")
        if not self.outcome_calc.load_prices(start_date, end_date):
            raise RuntimeError("Failed to load market prices")
        
        # Get dates to process
        dates = self.data_loader.get_available_dates()
        dates = [d for d in dates if start_date <= d <= end_date]
        logger.info(f"Processing {len(dates)} dates...")
        
        # Process each date
        for i, date in enumerate(dates):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(dates)} dates...")
            
            # Get point-in-time data
            pit = self.data_loader.get_point_in_time(date)
            
            if not pit.is_valid():
                continue
            
            # Calculate regime (simplified ARGUS-1 logic)
            regime, amri = self._calculate_regime(pit, config)
            
            # Get market outcome
            outcome = self.outcome_calc.get_outcome(date)
            
            # Record signal
            self.analyzer.add_signal(
                date=date,
                regime=regime,
                amri=amri,
                veto_active=pit.veto_triggers >= config.veto_threshold,
                contagion=pit.contagion_score,
                sac=self._calculate_sac(pit),
                outcome=outcome
            )
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        metrics = self.analyzer.calculate_metrics()
        
        logger.info("Backtest complete!")
        
        return BacktestResult(config=config, metrics=metrics, analyzer=self.analyzer)
    
    def _calculate_regime(self, pit: PointInTimeData, 
                         config: BacktestConfig) -> Tuple[str, float]:
        """
        Calculate ARGUS-1 regime from point-in-time data.
        
        This is a simplified version of the full ARGUS-1 calculator
        for backtesting purposes.
        """
        # Calculate simplified AMRI
        amri = self._calculate_amri(pit)
        
        # Check for VETO override
        veto_active = pit.veto_triggers >= config.veto_threshold
        
        # Determine base regime from AMRI
        if amri < config.amri_normal:
            regime = "NORMAL"
        elif amri < config.amri_elevated:
            regime = "ELEVATED"
        elif amri < config.amri_tension:
            regime = "TENSION"
        elif amri < config.amri_fragile:
            regime = "FRAGILE"
        else:
            regime = "BREAK"
        
        # VIX can directly escalate regime (most reliable signal)
        if pit.vix >= config.vix_fragile:
            if regime in ("NORMAL", "ELEVATED", "TENSION"):
                regime = "FRAGILE"
        elif pit.vix >= config.vix_tension:
            if regime in ("NORMAL", "ELEVATED"):
                regime = "TENSION"
        elif pit.vix >= config.vix_elevated:
            if regime == "NORMAL":
                regime = "ELEVATED"
        
        # VETO can escalate regime
        if veto_active and regime in ("NORMAL", "ELEVATED"):
            regime = "TENSION"
        
        # High contagion can escalate
        if pit.contagion_score >= config.contagion_danger:
            if regime == "NORMAL":
                regime = "ELEVATED"
            elif regime == "ELEVATED":
                regime = "TENSION"
        
        return regime, amri
    
    def _calculate_amri(self, pit: PointInTimeData) -> float:
        """
        Calculate simplified AMRI score (0-100).
        
        VIX-weighted formula since VIX is the most reliable real-time signal.
        Higher score = higher risk.
        
        Components:
        - VIX contribution (40% weight) - primary driver
        - Bubble index (25% weight)
        - Contagion (20% weight)
        - Credit spreads (15% weight)
        """
        # VIX contribution (0-100 scale) - PRIMARY DRIVER
        # VIX < 12: very calm (0-20)
        # VIX 12-18: normal (20-40)
        # VIX 18-25: elevated (40-65)
        # VIX 25-35: high (65-85)
        # VIX > 35: extreme (85-100)
        vix = pit.vix
        if vix < 12:
            vix_score = vix * 1.67  # 0-20
        elif vix < 18:
            vix_score = 20 + (vix - 12) * 3.33  # 20-40
        elif vix < 25:
            vix_score = 40 + (vix - 18) * 3.57  # 40-65
        elif vix < 35:
            vix_score = 65 + (vix - 25) * 2.0   # 65-85
        else:
            vix_score = min(100, 85 + (vix - 35) * 1.5)
        
        # Bubble index contribution (already 0-100)
        bubble_score = pit.bubble_index
        
        # Contagion contribution (already 0-100)
        contagion_score = pit.contagion_score
        
        # Credit spread contribution
        # Normalize HY spread: 250-300 = normal, 400+ = elevated
        spread = pit.credit_spread_hy
        if spread < 300:
            spread_score = 20 + (spread / 300) * 25  # 20-45
        elif spread < 400:
            spread_score = 45 + ((spread - 300) / 100) * 25  # 45-70
        elif spread < 600:
            spread_score = 70 + ((spread - 400) / 200) * 20  # 70-90
        else:
            spread_score = min(100, 90 + ((spread - 600) / 400) * 10)
        
        # Weighted combination
        amri = (
            vix_score * 0.40 +       # VIX is primary
            bubble_score * 0.25 +    # Bubble index
            contagion_score * 0.20 + # Contagion
            spread_score * 0.15      # Credit spreads
        )
        
        return round(amri, 1)
    
    def _calculate_sac(self, pit: PointInTimeData) -> float:
        """Calculate simplified SAC (Shock Absorption Capacity)"""
        # Higher = more buffer
        
        # AMRI buffer (inverse of AMRI)
        amri = self._calculate_amri(pit)
        amri_buffer = max(0, 100 - amri)
        
        # Bubble buffer
        if pit.bubble_regime == "INFRASTRUCTURE":
            bubble_buffer = 90
        elif pit.bubble_regime == "ADOPTION":
            bubble_buffer = 70
        elif pit.bubble_regime == "TRANSITION":
            bubble_buffer = 50
        else:
            bubble_buffer = 20
        
        # Contagion buffer
        contagion_buffer = max(0, 100 - pit.contagion_score)
        
        # Combine
        sac = (amri_buffer * 0.4 + bubble_buffer * 0.3 + contagion_buffer * 0.3)
        
        return round(sac, 1)


def run_backtest_cli():
    """Command-line interface for running backtests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ARGUS-1 backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    parser.add_argument("--output", default="backtest_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    runner = BacktestRunner()
    results = runner.run(start_date=args.start, end_date=args.end, benchmark=args.benchmark)
    
    print(results.summary())
    results.export_csv(args.output)


if __name__ == "__main__":
    run_backtest_cli()