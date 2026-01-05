"""
PILLAR INDEX CALCULATOR
=======================
Calculates daily returns and momentum for each of the 6 pillars.
Foundation module - MCI, BreadthDial, and PortfolioTracker depend on this.

Pillars:
- Infrastructure & Energy (16 stocks)
- Enterprise Adoption (13 stocks)
- Productivity & Labor (3 stocks)
- Demand Dynamics (3 stocks)
- Macro & Policy (4 stocks)
- Financial & Market (4 stocks)

Output:
- Daily returns per pillar
- Cumulative index (starting at 100)
- Momentum: 5D, 1M (21d), 3M (63d), 6M (126d), YTD
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np

# Try yfinance, but it may be blocked
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

# Pillar mapping (matches Google Sheets Universe)
PILLAR_MAP = {
    "Infrastructure & Energy": "Infrastructure",
    "Enterprise Adoption": "Enterprise", 
    "Macro & Policy": "Macro",
    "Financial & Market": "Financial",
    "Productivity & Labor": "Productivity",
    "Demand Dynamics": "Demand"
}

# Full pillar definitions with tickers and weights
PILLAR_STOCKS = {
    "Infrastructure & Energy": {
        "tickers": ["TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT", 
                   "CEG", "NRG", "EQIX", "DLR", "KLAC", "LRCX", "AMAT", "QCOM"],
        "weights": None  # Equal weight if None
    },
    "Enterprise Adoption": {
        "tickers": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR",
                   "ADBE", "ORCL", "MDB", "DDOG", "ZS"],
        "weights": None
    },
    "Productivity & Labor": {
        "tickers": ["NET", "CRWD", "PANW"],
        "weights": None
    },
    "Demand Dynamics": {
        "tickers": ["TSLA", "SHOP", "UBER"],
        "weights": None
    },
    "Macro & Policy": {
        "tickers": ["NXPI", "ON", "SMCI", "ARM"],
        "weights": None
    },
    "Financial & Market": {
        "tickers": ["GS", "MS", "JKS", "FSLR"],
        "weights": None
    }
}

# Momentum periods (trading days)
MOMENTUM_PERIODS = {
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126
}

# Short names for database columns
PILLAR_SHORT_NAMES = ["Infrastructure", "Enterprise", "Macro", "Financial", "Productivity", "Demand"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PillarDayData:
    """Single day's pillar data."""
    date: str
    # Daily returns
    infra_return: float = 0.0
    enterprise_return: float = 0.0
    macro_return: float = 0.0
    financial_return: float = 0.0
    productivity_return: float = 0.0
    demand_return: float = 0.0
    # Index values (cumulative, starting at 100)
    infra_index: float = 100.0
    enterprise_index: float = 100.0
    macro_index: float = 100.0
    financial_index: float = 100.0
    productivity_index: float = 100.0
    demand_index: float = 100.0
    # 5D momentum
    infra_5d: float = 0.0
    enterprise_5d: float = 0.0
    macro_5d: float = 0.0
    financial_5d: float = 0.0
    productivity_5d: float = 0.0
    demand_5d: float = 0.0
    # 1M momentum
    infra_1m: float = 0.0
    enterprise_1m: float = 0.0
    macro_1m: float = 0.0
    financial_1m: float = 0.0
    productivity_1m: float = 0.0
    demand_1m: float = 0.0
    # 3M momentum
    infra_3m: float = 0.0
    enterprise_3m: float = 0.0
    macro_3m: float = 0.0
    financial_3m: float = 0.0
    productivity_3m: float = 0.0
    demand_3m: float = 0.0
    # 6M momentum
    infra_6m: float = 0.0
    enterprise_6m: float = 0.0
    macro_6m: float = 0.0
    financial_6m: float = 0.0
    productivity_6m: float = 0.0
    demand_6m: float = 0.0
    # YTD momentum
    infra_ytd: float = 0.0
    enterprise_ytd: float = 0.0
    macro_ytd: float = 0.0
    financial_ytd: float = 0.0
    productivity_ytd: float = 0.0
    demand_ytd: float = 0.0


@dataclass
class PillarSignal:
    """Pillar signal for portfolio decisions."""
    pillar: str
    index_value: float
    momentum_5d: float
    momentum_1m: float
    momentum_3m: float
    signal: str  # LEADING, NEUTRAL, WEAKENING, STRESSED
    strength: float  # 0-100


# =============================================================================
# PILLAR INDEX CALCULATOR
# =============================================================================

class PillarIndexCalculator:
    """
    Calculates pillar indices and momentum from price data.
    
    Usage:
        calc = PillarIndexCalculator()
        results = calc.calculate(days=252)
        calc.save_to_supabase(results)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize with optional Supabase credentials."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers across all pillars."""
        all_tickers = []
        for pillar_data in PILLAR_STOCKS.values():
            all_tickers.extend(pillar_data["tickers"])
        return list(set(all_tickers))
    
    def fetch_price_history(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch price history for all tickers.
        
        Args:
            days: Number of trading days to fetch
            
        Returns:
            DataFrame with columns: Date, Ticker, Close
        """
        tickers = self.get_all_tickers()
        logger.info(f"Fetching price history for {len(tickers)} tickers, {days} days")
        
        # First try Supabase
        if self.supabase:
            df = self._fetch_from_supabase(tickers, days)
            if df is not None and len(df) > 0:
                logger.info(f"Loaded {len(df)} rows from Supabase")
                return df
        
        # Fall back to yfinance
        if YFINANCE_AVAILABLE:
            return self._fetch_from_yfinance(tickers, days)
        
        raise RuntimeError("No price data source available")
    
    def _fetch_from_supabase(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
        """Fetch from Supabase price_history table."""
        try:
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
            
            response = self.supabase.table("price_history") \
                .select("date, ticker, close") \
                .in_("ticker", tickers) \
                .gte("date", start_date) \
                .order("date", desc=True) \
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logger.warning(f"Supabase fetch failed: {e}")
        
        return None
    
    def _fetch_from_yfinance(self, tickers: List[str], days: int) -> pd.DataFrame:
        """Fetch from Yahoo Finance."""
        logger.info("Fetching from Yahoo Finance...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)  # Extra buffer for weekends
        
        all_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) > 0:
                    for date, row in hist.iterrows():
                        all_data.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "close": row["Close"]
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Fetched {len(df)} rows from yfinance")
        return df
    
    def calculate_pillar_returns(self, price_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate daily weighted returns for each pillar.
        
        Args:
            price_df: DataFrame with Date, Ticker, Close columns
            
        Returns:
            Dict of pillar name -> Series of daily returns indexed by date
        """
        # Pivot to get prices per ticker per day
        pivot = price_df.pivot_table(
            index="date", 
            columns="ticker", 
            values="close"
        ).sort_index()
        
        # Calculate daily returns
        returns = pivot.pct_change()
        
        pillar_returns = {}
        
        for pillar_full, pillar_data in PILLAR_STOCKS.items():
            pillar_short = PILLAR_MAP[pillar_full]
            tickers = pillar_data["tickers"]
            weights = pillar_data["weights"]
            
            # Get returns for tickers in this pillar
            available = [t for t in tickers if t in returns.columns]
            
            if not available:
                logger.warning(f"No tickers available for {pillar_short}")
                pillar_returns[pillar_short] = pd.Series(0, index=returns.index)
                continue
            
            pillar_rets = returns[available]
            
            if weights:
                # Weighted average
                w = np.array([weights.get(t, 1) for t in available])
                w = w / w.sum()
                pillar_returns[pillar_short] = (pillar_rets * w).sum(axis=1)
            else:
                # Equal weight
                pillar_returns[pillar_short] = pillar_rets.mean(axis=1)
        
        return pillar_returns
    
    def build_cumulative_index(self, pillar_returns: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Build cumulative index starting at 100.
        
        Args:
            pillar_returns: Dict of pillar -> daily return series
            
        Returns:
            Dict of pillar -> cumulative index series
        """
        pillar_index = {}
        
        for pillar, returns in pillar_returns.items():
            # Start at 100, compound returns
            index_values = (1 + returns.fillna(0)).cumprod() * 100
            pillar_index[pillar] = index_values
        
        return pillar_index
    
    def calculate_momentum(
        self, 
        pillar_index: Dict[str, pd.Series],
        periods: Dict[str, int] = None
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate momentum for various periods.
        
        Args:
            pillar_index: Dict of pillar -> index series
            periods: Dict of period name -> lookback days
            
        Returns:
            Dict of pillar -> Dict of period -> momentum series
        """
        if periods is None:
            periods = MOMENTUM_PERIODS
        
        momentum = {pillar: {} for pillar in pillar_index}
        
        for pillar, index in pillar_index.items():
            for period_name, lookback in periods.items():
                # Momentum = (current - lookback) / lookback
                shifted = index.shift(lookback)
                mom = (index - shifted) / shifted
                momentum[pillar][period_name] = mom
            
            # YTD momentum
            ytd_start = self._get_ytd_start_date(index.index)
            if ytd_start and ytd_start in index.index:
                ytd_start_val = index[ytd_start]
                momentum[pillar]["YTD"] = (index - ytd_start_val) / ytd_start_val
            else:
                momentum[pillar]["YTD"] = pd.Series(0, index=index.index)
        
        return momentum
    
    def _get_ytd_start_date(self, dates: pd.DatetimeIndex) -> Optional[str]:
        """Find first trading day of current year."""
        current_year = datetime.now().year
        year_dates = [d for d in dates if d.year == current_year]
        if year_dates:
            return min(year_dates)
        return None
    
    def calculate(self, days: int = 252) -> List[PillarDayData]:
        """
        Main calculation: fetch data and compute all pillar metrics.
        
        Args:
            days: Number of trading days to calculate
            
        Returns:
            List of PillarDayData objects, newest first
        """
        logger.info(f"Calculating pillar index for {days} days")
        
        # Fetch price history
        price_df = self.fetch_price_history(days)
        
        # Calculate pillar returns
        pillar_returns = self.calculate_pillar_returns(price_df)
        
        # Build cumulative index
        pillar_index = self.build_cumulative_index(pillar_returns)
        
        # Calculate momentum
        momentum = self.calculate_momentum(pillar_index)
        
        # Build results
        results = []
        
        # Get all dates (sorted descending - newest first)
        all_dates = sorted(pillar_returns["Infrastructure"].index, reverse=True)
        
        for date in all_dates[:days]:
            date_str = date.strftime("%Y-%m-%d")
            
            data = PillarDayData(date=date_str)
            
            # Daily returns
            data.infra_return = pillar_returns["Infrastructure"].get(date, 0) or 0
            data.enterprise_return = pillar_returns["Enterprise"].get(date, 0) or 0
            data.macro_return = pillar_returns["Macro"].get(date, 0) or 0
            data.financial_return = pillar_returns["Financial"].get(date, 0) or 0
            data.productivity_return = pillar_returns["Productivity"].get(date, 0) or 0
            data.demand_return = pillar_returns["Demand"].get(date, 0) or 0
            
            # Index values
            data.infra_index = pillar_index["Infrastructure"].get(date, 100) or 100
            data.enterprise_index = pillar_index["Enterprise"].get(date, 100) or 100
            data.macro_index = pillar_index["Macro"].get(date, 100) or 100
            data.financial_index = pillar_index["Financial"].get(date, 100) or 100
            data.productivity_index = pillar_index["Productivity"].get(date, 100) or 100
            data.demand_index = pillar_index["Demand"].get(date, 100) or 100
            
            # 5D momentum
            data.infra_5d = momentum["Infrastructure"]["5D"].get(date, 0) or 0
            data.enterprise_5d = momentum["Enterprise"]["5D"].get(date, 0) or 0
            data.macro_5d = momentum["Macro"]["5D"].get(date, 0) or 0
            data.financial_5d = momentum["Financial"]["5D"].get(date, 0) or 0
            data.productivity_5d = momentum["Productivity"]["5D"].get(date, 0) or 0
            data.demand_5d = momentum["Demand"]["5D"].get(date, 0) or 0
            
            # 1M momentum
            data.infra_1m = momentum["Infrastructure"]["1M"].get(date, 0) or 0
            data.enterprise_1m = momentum["Enterprise"]["1M"].get(date, 0) or 0
            data.macro_1m = momentum["Macro"]["1M"].get(date, 0) or 0
            data.financial_1m = momentum["Financial"]["1M"].get(date, 0) or 0
            data.productivity_1m = momentum["Productivity"]["1M"].get(date, 0) or 0
            data.demand_1m = momentum["Demand"]["1M"].get(date, 0) or 0
            
            # 3M momentum
            data.infra_3m = momentum["Infrastructure"]["3M"].get(date, 0) or 0
            data.enterprise_3m = momentum["Enterprise"]["3M"].get(date, 0) or 0
            data.macro_3m = momentum["Macro"]["3M"].get(date, 0) or 0
            data.financial_3m = momentum["Financial"]["3M"].get(date, 0) or 0
            data.productivity_3m = momentum["Productivity"]["3M"].get(date, 0) or 0
            data.demand_3m = momentum["Demand"]["3M"].get(date, 0) or 0
            
            # 6M momentum
            data.infra_6m = momentum["Infrastructure"]["6M"].get(date, 0) or 0
            data.enterprise_6m = momentum["Enterprise"]["6M"].get(date, 0) or 0
            data.macro_6m = momentum["Macro"]["6M"].get(date, 0) or 0
            data.financial_6m = momentum["Financial"]["6M"].get(date, 0) or 0
            data.productivity_6m = momentum["Productivity"]["6M"].get(date, 0) or 0
            data.demand_6m = momentum["Demand"]["6M"].get(date, 0) or 0
            
            # YTD momentum
            data.infra_ytd = momentum["Infrastructure"]["YTD"].get(date, 0) or 0
            data.enterprise_ytd = momentum["Enterprise"]["YTD"].get(date, 0) or 0
            data.macro_ytd = momentum["Macro"]["YTD"].get(date, 0) or 0
            data.financial_ytd = momentum["Financial"]["YTD"].get(date, 0) or 0
            data.productivity_ytd = momentum["Productivity"]["YTD"].get(date, 0) or 0
            data.demand_ytd = momentum["Demand"]["YTD"].get(date, 0) or 0
            
            results.append(data)
        
        logger.info(f"Calculated {len(results)} days of pillar data")
        return results
    
    def get_pillar_signals(self, data: List[PillarDayData] = None) -> List[PillarSignal]:
        """
        Generate pillar signals based on momentum.
        
        Signal logic:
        - LEADING: 5D > 0 AND 1M > 0 AND 3M > 0
        - NEUTRAL: Mixed signals
        - WEAKENING: 5D < 0 OR 1M < 0
        - STRESSED: 5D < 0 AND 1M < 0 AND 3M < 0
        
        Returns:
            List of PillarSignal objects for most recent date
        """
        if data is None:
            data = self.calculate(days=30)
        
        if not data:
            return []
        
        latest = data[0]  # Most recent
        
        signals = []
        
        pillar_mapping = [
            ("Infrastructure", latest.infra_index, latest.infra_5d, latest.infra_1m, latest.infra_3m),
            ("Enterprise", latest.enterprise_index, latest.enterprise_5d, latest.enterprise_1m, latest.enterprise_3m),
            ("Macro", latest.macro_index, latest.macro_5d, latest.macro_1m, latest.macro_3m),
            ("Financial", latest.financial_index, latest.financial_5d, latest.financial_1m, latest.financial_3m),
            ("Productivity", latest.productivity_index, latest.productivity_5d, latest.productivity_1m, latest.productivity_3m),
            ("Demand", latest.demand_index, latest.demand_5d, latest.demand_1m, latest.demand_3m),
        ]
        
        for pillar, index_val, m5d, m1m, m3m in pillar_mapping:
            # Determine signal
            if m5d > 0 and m1m > 0 and m3m > 0:
                signal = "LEADING"
                strength = min(100, (m5d + m1m + m3m) * 100 / 3 + 50)
            elif m5d < 0 and m1m < 0 and m3m < 0:
                signal = "STRESSED"
                strength = max(0, 50 + (m5d + m1m + m3m) * 100 / 3)
            elif m5d < 0 or m1m < 0:
                signal = "WEAKENING"
                strength = 25 + (m3m * 50 if m3m > 0 else 0)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            signals.append(PillarSignal(
                pillar=pillar,
                index_value=index_val,
                momentum_5d=m5d,
                momentum_1m=m1m,
                momentum_3m=m3m,
                signal=signal,
                strength=strength
            ))
        
        return signals
    
    def save_to_supabase(self, data: List[PillarDayData]) -> int:
        """
        Save pillar data to Supabase.
        
        Args:
            data: List of PillarDayData objects
            
        Returns:
            Number of rows upserted
        """
        if not self.supabase:
            logger.warning("Supabase not configured")
            return 0
        
        rows = []
        for d in data:
            row = asdict(d)
            # Convert NaN to None
            for k, v in row.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row[k] = None
            rows.append(row)
        
        try:
            # Upsert (insert or update on conflict)
            response = self.supabase.table("pillar_index_daily") \
                .upsert(rows, on_conflict="date") \
                .execute()
            
            logger.info(f"Upserted {len(rows)} rows to pillar_index_daily")
            return len(rows)
            
        except Exception as e:
            logger.error(f"Supabase save failed: {e}")
            return 0
    
    def get_latest(self) -> Optional[PillarDayData]:
        """Get most recent pillar data from Supabase."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table("pillar_index_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return PillarDayData(**response.data[0])
        except Exception as e:
            logger.error(f"Failed to get latest: {e}")
        
        return None


# =============================================================================
# SUPABASE TABLE CREATION
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pillar_index_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    -- Daily returns
    infra_return FLOAT,
    enterprise_return FLOAT,
    macro_return FLOAT,
    financial_return FLOAT,
    productivity_return FLOAT,
    demand_return FLOAT,
    -- Index values (cumulative, starting at 100)
    infra_index FLOAT,
    enterprise_index FLOAT,
    macro_index FLOAT,
    financial_index FLOAT,
    productivity_index FLOAT,
    demand_index FLOAT,
    -- 5D momentum
    infra_5d FLOAT,
    enterprise_5d FLOAT,
    macro_5d FLOAT,
    financial_5d FLOAT,
    productivity_5d FLOAT,
    demand_5d FLOAT,
    -- 1M momentum
    infra_1m FLOAT,
    enterprise_1m FLOAT,
    macro_1m FLOAT,
    financial_1m FLOAT,
    productivity_1m FLOAT,
    demand_1m FLOAT,
    -- 3M momentum
    infra_3m FLOAT,
    enterprise_3m FLOAT,
    macro_3m FLOAT,
    financial_3m FLOAT,
    productivity_3m FLOAT,
    demand_3m FLOAT,
    -- 6M momentum
    infra_6m FLOAT,
    enterprise_6m FLOAT,
    macro_6m FLOAT,
    financial_6m FLOAT,
    productivity_6m FLOAT,
    demand_6m FLOAT,
    -- YTD momentum
    infra_ytd FLOAT,
    enterprise_ytd FLOAT,
    macro_ytd FLOAT,
    financial_ytd FLOAT,
    productivity_ytd FLOAT,
    demand_ytd FLOAT,
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_pillar_index_date ON pillar_index_daily(date DESC);
"""


# =============================================================================
# CLI & TESTING
# =============================================================================

def main():
    """Run pillar index calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Pillar Index")
    parser.add_argument("--days", type=int, default=30, help="Days to calculate")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--signals", action="store_true", help="Show pillar signals")
    args = parser.parse_args()
    
    calc = PillarIndexCalculator()
    
    print(f"\n{'='*60}")
    print("PILLAR INDEX CALCULATOR")
    print(f"{'='*60}\n")
    
    # Calculate
    data = calc.calculate(days=args.days)
    
    if not data:
        print("No data calculated!")
        return
    
    # Show latest
    latest = data[0]
    print(f"Date: {latest.date}")
    print(f"\n{'='*40}")
    print("DAILY RETURNS")
    print(f"{'='*40}")
    print(f"  Infrastructure: {latest.infra_return*100:+.2f}%")
    print(f"  Enterprise:     {latest.enterprise_return*100:+.2f}%")
    print(f"  Macro:          {latest.macro_return*100:+.2f}%")
    print(f"  Financial:      {latest.financial_return*100:+.2f}%")
    print(f"  Productivity:   {latest.productivity_return*100:+.2f}%")
    print(f"  Demand:         {latest.demand_return*100:+.2f}%")
    
    print(f"\n{'='*40}")
    print("INDEX VALUES (Base 100)")
    print(f"{'='*40}")
    print(f"  Infrastructure: {latest.infra_index:.1f}")
    print(f"  Enterprise:     {latest.enterprise_index:.1f}")
    print(f"  Macro:          {latest.macro_index:.1f}")
    print(f"  Financial:      {latest.financial_index:.1f}")
    print(f"  Productivity:   {latest.productivity_index:.1f}")
    print(f"  Demand:         {latest.demand_index:.1f}")
    
    print(f"\n{'='*40}")
    print("MOMENTUM (5D / 1M / 3M)")
    print(f"{'='*40}")
    print(f"  Infrastructure: {latest.infra_5d*100:+.1f}% / {latest.infra_1m*100:+.1f}% / {latest.infra_3m*100:+.1f}%")
    print(f"  Enterprise:     {latest.enterprise_5d*100:+.1f}% / {latest.enterprise_1m*100:+.1f}% / {latest.enterprise_3m*100:+.1f}%")
    print(f"  Macro:          {latest.macro_5d*100:+.1f}% / {latest.macro_1m*100:+.1f}% / {latest.macro_3m*100:+.1f}%")
    print(f"  Financial:      {latest.financial_5d*100:+.1f}% / {latest.financial_1m*100:+.1f}% / {latest.financial_3m*100:+.1f}%")
    print(f"  Productivity:   {latest.productivity_5d*100:+.1f}% / {latest.productivity_1m*100:+.1f}% / {latest.productivity_3m*100:+.1f}%")
    print(f"  Demand:         {latest.demand_5d*100:+.1f}% / {latest.demand_1m*100:+.1f}% / {latest.demand_3m*100:+.1f}%")
    
    if args.signals:
        signals = calc.get_pillar_signals(data)
        print(f"\n{'='*40}")
        print("PILLAR SIGNALS")
        print(f"{'='*40}")
        for sig in signals:
            print(f"  {sig.pillar:15} {sig.signal:12} (strength: {sig.strength:.0f})")
    
    if args.save:
        saved = calc.save_to_supabase(data)
        print(f"\n✅ Saved {saved} rows to Supabase")


if __name__ == "__main__":
    main()
