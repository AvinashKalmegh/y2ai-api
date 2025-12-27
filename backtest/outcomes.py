"""
Market Outcomes Calculator for ARGUS-1 Backtesting

Calculates actual market outcomes (returns, drawdowns) to compare
against ARGUS-1 regime signals.

Outcome Types:
- Forward returns (1D, 5D, 20D)
- Max drawdown in forward window
- Volatility spike detection
- Major correction detection (>10% decline)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketOutcome:
    """Market outcomes following a specific date"""
    date: str
    
    # Forward returns (%)
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_20d: float = 0.0
    return_60d: float = 0.0
    
    # Drawdown in forward window (%)
    max_dd_5d: float = 0.0
    max_dd_20d: float = 0.0
    max_dd_60d: float = 0.0
    
    # Volatility
    realized_vol_20d: float = 0.0
    vol_spike: bool = False
    
    # Event detection
    correction_5d: bool = False   # >5% decline
    correction_10d: bool = False  # >10% decline
    rally_5d: bool = False        # >5% gain
    rally_10d: bool = False       # >10% gain
    
    # For classification
    outcome_label: str = "NEUTRAL"  # GOOD, BAD, NEUTRAL


class OutcomeCalculator:
    """
    Calculate market outcomes for backtesting evaluation.
    
    Uses SPY as the primary benchmark for market outcomes,
    but can also track Y2AI-specific stock outcomes.
    """
    
    def __init__(self, benchmark: str = "SPY"):
        self.benchmark = benchmark
        self._price_df: Optional[pd.DataFrame] = None
    
    def load_prices(self, start_date: str, end_date: str, 
                   buffer_days: int = 90) -> bool:
        """
        Load benchmark prices for outcome calculation.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            buffer_days: Extra days after end_date for forward returns
        
        Returns:
            True if prices loaded successfully
        """
        try:
            import yfinance as yf
            
            # Extend end date by buffer to calculate forward returns
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=buffer_days)
            
            logger.info(f"Loading {self.benchmark} prices from {start_date} to {end_dt.strftime('%Y-%m-%d')}")
            
            ticker = yf.Ticker(self.benchmark)
            df = ticker.history(start=start_date, end=end_dt.strftime('%Y-%m-%d'))
            
            if df.empty:
                logger.error(f"No price data for {self.benchmark}")
                return False
            
            # Keep only Close price
            self._price_df = df[['Close']].copy()
            self._price_df.columns = ['close']
            self._price_df.index = pd.to_datetime(self._price_df.index).date
            self._price_df.index = self._price_df.index.astype(str)
            
            # Calculate returns
            self._price_df['return_1d'] = self._price_df['close'].pct_change(1) * 100
            self._price_df['return_5d'] = self._price_df['close'].pct_change(5) * 100
            self._price_df['return_20d'] = self._price_df['close'].pct_change(20) * 100
            
            # Calculate rolling volatility (20-day)
            self._price_df['vol_20d'] = self._price_df['return_1d'].rolling(20).std() * np.sqrt(252)
            
            logger.info(f"Loaded {len(self._price_df)} days of {self.benchmark} data")
            return True
            
        except Exception as e:
            logger.error(f"Error loading prices: {e}")
            return False
    
    def get_outcome(self, date: str) -> MarketOutcome:
        """
        Calculate market outcomes following a specific date.
        
        Args:
            date: Signal date (YYYY-MM-DD)
        
        Returns:
            MarketOutcome with forward returns and drawdowns
        """
        outcome = MarketOutcome(date=date)
        
        if self._price_df is None or len(self._price_df) == 0:
            return outcome
        
        if date not in self._price_df.index:
            # Find closest date
            all_dates = self._price_df.index.tolist()
            close_dates = [d for d in all_dates if d >= date]
            if not close_dates:
                return outcome
            date = min(close_dates)
        
        try:
            idx = self._price_df.index.get_loc(date)
            
            # Forward returns
            if idx + 1 < len(self._price_df):
                outcome.return_1d = self._calc_return(idx, idx + 1)
            if idx + 5 < len(self._price_df):
                outcome.return_5d = self._calc_return(idx, idx + 5)
            if idx + 20 < len(self._price_df):
                outcome.return_20d = self._calc_return(idx, idx + 20)
            if idx + 60 < len(self._price_df):
                outcome.return_60d = self._calc_return(idx, idx + 60)
            
            # Max drawdown in forward window
            outcome.max_dd_5d = self._calc_max_drawdown(idx, 5)
            outcome.max_dd_20d = self._calc_max_drawdown(idx, 20)
            outcome.max_dd_60d = self._calc_max_drawdown(idx, 60)
            
            # Volatility
            if idx + 20 < len(self._price_df):
                outcome.realized_vol_20d = self._calc_forward_vol(idx, 20)
                # Vol spike if realized vol > 25%
                outcome.vol_spike = outcome.realized_vol_20d > 25
            
            # Event detection
            outcome.correction_5d = outcome.max_dd_5d < -5
            outcome.correction_10d = outcome.max_dd_20d < -10
            outcome.rally_5d = outcome.return_5d > 5
            outcome.rally_10d = outcome.return_20d > 10
            
            # Overall outcome label
            outcome.outcome_label = self._classify_outcome(outcome)
            
        except Exception as e:
            logger.warning(f"Error calculating outcome for {date}: {e}")
        
        return outcome
    
    def _calc_return(self, start_idx: int, end_idx: int) -> float:
        """Calculate return between two indices"""
        start_price = self._price_df.iloc[start_idx]['close']
        end_price = self._price_df.iloc[end_idx]['close']
        return ((end_price / start_price) - 1) * 100
    
    def _calc_max_drawdown(self, start_idx: int, days: int) -> float:
        """Calculate max drawdown over forward window"""
        end_idx = min(start_idx + days, len(self._price_df))
        if start_idx >= end_idx:
            return 0.0
        
        prices = self._price_df.iloc[start_idx:end_idx]['close'].values
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            dd = ((price / peak) - 1) * 100
            if dd < max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calc_forward_vol(self, start_idx: int, days: int) -> float:
        """Calculate forward realized volatility"""
        end_idx = min(start_idx + days, len(self._price_df))
        if end_idx - start_idx < 5:
            return 0.0
        
        returns = self._price_df.iloc[start_idx:end_idx]['return_1d'].dropna()
        if len(returns) < 5:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    def _classify_outcome(self, outcome: MarketOutcome) -> str:
        """
        Classify outcome as GOOD, BAD, or NEUTRAL.
        
        BAD: Significant drawdown (>5%) or high volatility
        GOOD: Solid positive returns with limited drawdown
        NEUTRAL: Mixed or flat
        """
        # BAD: Major drawdown or vol spike
        if outcome.max_dd_20d < -10:
            return "VERY_BAD"
        if outcome.max_dd_20d < -5:
            return "BAD"
        if outcome.vol_spike and outcome.return_20d < 0:
            return "BAD"
        
        # GOOD: Positive returns, limited drawdown
        if outcome.return_20d > 5 and outcome.max_dd_20d > -5:
            return "VERY_GOOD"
        if outcome.return_20d > 2 and outcome.max_dd_20d > -3:
            return "GOOD"
        
        # NEUTRAL: Everything else
        return "NEUTRAL"
    
    def get_all_outcomes(self, dates: List[str]) -> Dict[str, MarketOutcome]:
        """
        Calculate outcomes for multiple dates.
        
        Args:
            dates: List of signal dates
        
        Returns:
            Dict mapping date -> MarketOutcome
        """
        outcomes = {}
        for date in dates:
            outcomes[date] = self.get_outcome(date)
        return outcomes
