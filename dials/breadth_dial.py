"""
BREADTH DIAL
============
Market participation indicator measuring how many stocks are participating in moves.

Metrics:
- Daily Breadth: % stocks up vs down
- 20D Breadth: % stocks above 20-day MA
- 50D Breadth: % stocks above 50-day MA
- Breadth Momentum: 5D change in 20D breadth
- Per-pillar breadth breakdown

Regime Thresholds:
- Healthy: >= 70% (Broad participation)
- Caution: 50-70% (Narrowing leadership)
- Fragile: 30-50% (Weak participation)
- Stressed: < 30% (Very narrow market)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

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

# Import pillar stocks from pillar_index
from .pillar_index import PILLAR_STOCKS, PILLAR_MAP

# Regime thresholds (based on 20D breadth)
BREADTH_REGIMES = {
    "Healthy": (0.70, 1.01),    # >= 70%
    "Caution": (0.50, 0.70),    # 50-70%
    "Fragile": (0.30, 0.50),    # 30-50%
    "Stressed": (0.00, 0.30)    # < 30%
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BreadthData:
    """Single day's breadth data."""
    date: str
    # Overall breadth metrics
    daily_breadth: float = 0.0      # % stocks up today
    breadth_20d: float = 0.0        # % stocks above 20D MA
    breadth_50d: float = 0.0        # % stocks above 50D MA
    breadth_momentum: float = 0.0   # 5D change in 20D breadth
    # Counts
    advancers: int = 0
    decliners: int = 0
    above_20d: int = 0
    above_50d: int = 0
    valid_tickers: int = 0
    # Regime
    regime: str = "Unknown"
    # Per-pillar breadth (JSON stored)
    pillar_breadth: Dict = None


@dataclass
class PillarBreadth:
    """Breadth data for a single pillar."""
    pillar: str
    total: int = 0
    advancers: int = 0
    decliners: int = 0
    above_20d: int = 0
    above_50d: int = 0
    breadth_20d: float = 0.0
    breadth_50d: float = 0.0
    daily_breadth: float = 0.0
    regime: str = "Unknown"


# =============================================================================
# BREADTH CALCULATOR
# =============================================================================

class BreadthCalculator:
    """
    Calculates market breadth metrics.
    
    Usage:
        calc = BreadthCalculator()
        data = calc.calculate()
        calc.save_to_supabase(data)
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
    
    def get_ticker_pillar_map(self) -> Dict[str, str]:
        """Map tickers to their pillar."""
        ticker_to_pillar = {}
        for pillar_full, pillar_data in PILLAR_STOCKS.items():
            pillar_short = PILLAR_MAP[pillar_full]
            for ticker in pillar_data["tickers"]:
                ticker_to_pillar[ticker] = pillar_short
        return ticker_to_pillar
    
    def fetch_price_history(self, days: int = 60) -> pd.DataFrame:
        """
        Fetch price history for all tickers.
        Need at least 50 days for 50D MA calculation.
        """
        tickers = self.get_all_tickers()
        logger.info(f"Fetching price history for {len(tickers)} tickers")
        
        # Try Supabase first
        if self.supabase:
            df = self._fetch_from_supabase(tickers, days)
            if df is not None and len(df) > 0:
                return df
        
        # Fall back to yfinance
        if YFINANCE_AVAILABLE:
            return self._fetch_from_yfinance(tickers, days)
        
        raise RuntimeError("No price data source available")
    
    def _fetch_from_supabase(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
        """Fetch from Supabase."""
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
        start_date = end_date - timedelta(days=days * 2)
        
        all_data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                for date, row in hist.iterrows():
                    all_data.append({
                        "date": date,
                        "ticker": ticker,
                        "close": row["Close"]
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        df = pd.DataFrame(all_data)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
        return df
    
    def calculate_breadth_metrics(self, price_df: pd.DataFrame) -> Dict:
        """
        Calculate all breadth metrics from price data.
        
        Returns dict with:
        - daily_breadth, breadth_20d, breadth_50d
        - advancers, decliners, above_20d, above_50d
        - per-ticker details
        """
        # Pivot to get prices per ticker per day
        pivot = price_df.pivot_table(
            index="date",
            columns="ticker", 
            values="close"
        ).sort_index()
        
        if len(pivot) < 2:
            return {}
        
        # Get latest and previous prices
        latest_prices = pivot.iloc[-1]
        prev_prices = pivot.iloc[-2]
        
        # Calculate daily returns
        daily_returns = (latest_prices - prev_prices) / prev_prices
        
        # Calculate 20D and 50D MAs
        ma_20 = pivot.tail(20).mean()
        ma_50 = pivot.tail(50).mean() if len(pivot) >= 50 else pivot.mean()
        
        # Count metrics
        advancers = (daily_returns > 0).sum()
        decliners = (daily_returns < 0).sum()
        above_20d = (latest_prices > ma_20).sum()
        above_50d = (latest_prices > ma_50).sum()
        valid_tickers = latest_prices.notna().sum()
        
        # Calculate percentages
        daily_breadth = advancers / valid_tickers if valid_tickers > 0 else 0
        breadth_20d = above_20d / valid_tickers if valid_tickers > 0 else 0
        breadth_50d = above_50d / valid_tickers if valid_tickers > 0 else 0
        
        # Calculate 5D breadth momentum
        if len(pivot) >= 25:
            # Breadth 5 days ago
            prices_5d_ago = pivot.iloc[-6]
            ma_20_5d_ago = pivot.iloc[-25:-5].mean()
            above_20d_5d_ago = (prices_5d_ago > ma_20_5d_ago).sum()
            breadth_20d_5d_ago = above_20d_5d_ago / valid_tickers if valid_tickers > 0 else 0
            breadth_momentum = breadth_20d - breadth_20d_5d_ago
        else:
            breadth_momentum = 0
        
        # Per-ticker details
        ticker_details = []
        for ticker in latest_prices.index:
            if pd.notna(latest_prices[ticker]):
                ticker_details.append({
                    "ticker": ticker,
                    "price": latest_prices[ticker],
                    "return": daily_returns.get(ticker, 0),
                    "ma_20": ma_20.get(ticker, 0),
                    "ma_50": ma_50.get(ticker, 0),
                    "above_20d": latest_prices[ticker] > ma_20.get(ticker, 0),
                    "above_50d": latest_prices[ticker] > ma_50.get(ticker, 0)
                })
        
        return {
            "daily_breadth": daily_breadth,
            "breadth_20d": breadth_20d,
            "breadth_50d": breadth_50d,
            "breadth_momentum": breadth_momentum,
            "advancers": int(advancers),
            "decliners": int(decliners),
            "above_20d": int(above_20d),
            "above_50d": int(above_50d),
            "valid_tickers": int(valid_tickers),
            "ticker_details": ticker_details
        }
    
    def calculate_pillar_breadth(self, price_df: pd.DataFrame) -> Dict[str, PillarBreadth]:
        """Calculate breadth metrics for each pillar."""
        ticker_to_pillar = self.get_ticker_pillar_map()
        
        # Pivot prices
        pivot = price_df.pivot_table(
            index="date",
            columns="ticker",
            values="close"
        ).sort_index()
        
        if len(pivot) < 2:
            return {}
        
        latest_prices = pivot.iloc[-1]
        prev_prices = pivot.iloc[-2]
        daily_returns = (latest_prices - prev_prices) / prev_prices
        
        ma_20 = pivot.tail(20).mean()
        ma_50 = pivot.tail(50).mean() if len(pivot) >= 50 else pivot.mean()
        
        # Initialize pillar breadth
        pillar_breadth = {}
        pillars = list(set(ticker_to_pillar.values()))
        
        for pillar in pillars:
            pillar_breadth[pillar] = {
                "total": 0,
                "advancers": 0,
                "decliners": 0,
                "above_20d": 0,
                "above_50d": 0
            }
        
        # Calculate per ticker
        for ticker in latest_prices.index:
            pillar = ticker_to_pillar.get(ticker)
            if not pillar or pd.isna(latest_prices[ticker]):
                continue
            
            pillar_breadth[pillar]["total"] += 1
            
            # Daily advance/decline
            if daily_returns.get(ticker, 0) > 0:
                pillar_breadth[pillar]["advancers"] += 1
            elif daily_returns.get(ticker, 0) < 0:
                pillar_breadth[pillar]["decliners"] += 1
            
            # Above MAs
            if latest_prices[ticker] > ma_20.get(ticker, 0):
                pillar_breadth[pillar]["above_20d"] += 1
            if latest_prices[ticker] > ma_50.get(ticker, 0):
                pillar_breadth[pillar]["above_50d"] += 1
        
        # Calculate percentages and regime
        results = {}
        for pillar, data in pillar_breadth.items():
            total = data["total"]
            if total == 0:
                continue
            
            breadth_20d = data["above_20d"] / total
            breadth_50d = data["above_50d"] / total
            daily_breadth = data["advancers"] / total
            
            # Determine regime
            regime = self._get_regime(breadth_20d)
            
            results[pillar] = PillarBreadth(
                pillar=pillar,
                total=total,
                advancers=data["advancers"],
                decliners=data["decliners"],
                above_20d=data["above_20d"],
                above_50d=data["above_50d"],
                breadth_20d=breadth_20d,
                breadth_50d=breadth_50d,
                daily_breadth=daily_breadth,
                regime=regime
            )
        
        return results
    
    def _get_regime(self, breadth_20d: float) -> str:
        """Determine regime based on 20D breadth."""
        for regime, (low, high) in BREADTH_REGIMES.items():
            if low <= breadth_20d < high:
                return regime
        return "Unknown"
    
    def calculate(self, days: int = 60) -> BreadthData:
        """
        Main calculation: compute all breadth metrics.
        
        Args:
            days: Days of price history to use
            
        Returns:
            BreadthData object with all metrics
        """
        logger.info("Calculating breadth metrics...")
        
        # Fetch prices
        price_df = self.fetch_price_history(days)
        
        if price_df is None or len(price_df) == 0:
            logger.error("No price data available")
            return BreadthData(date=datetime.now().strftime("%Y-%m-%d"))
        
        # Calculate overall breadth
        metrics = self.calculate_breadth_metrics(price_df)
        
        # Calculate per-pillar breadth
        pillar_metrics = self.calculate_pillar_breadth(price_df)
        
        # Get regime
        regime = self._get_regime(metrics.get("breadth_20d", 0))
        
        # Build result
        date_str = price_df["date"].max().strftime("%Y-%m-%d")
        
        # Convert pillar metrics to dict for JSON storage
        pillar_dict = {
            pillar: asdict(data) for pillar, data in pillar_metrics.items()
        }
        
        result = BreadthData(
            date=date_str,
            daily_breadth=metrics.get("daily_breadth", 0),
            breadth_20d=metrics.get("breadth_20d", 0),
            breadth_50d=metrics.get("breadth_50d", 0),
            breadth_momentum=metrics.get("breadth_momentum", 0),
            advancers=metrics.get("advancers", 0),
            decliners=metrics.get("decliners", 0),
            above_20d=metrics.get("above_20d", 0),
            above_50d=metrics.get("above_50d", 0),
            valid_tickers=metrics.get("valid_tickers", 0),
            regime=regime,
            pillar_breadth=pillar_dict
        )
        
        logger.info(f"Breadth calculated: 20D={result.breadth_20d*100:.1f}% ({result.regime})")
        return result
    
    def calculate_history(self, days: int = 30) -> List[BreadthData]:
        """
        Calculate breadth for multiple days (historical backfill).
        
        Args:
            days: Number of days to calculate
            
        Returns:
            List of BreadthData objects
        """
        logger.info(f"Calculating {days} days of breadth history...")
        
        # Fetch more price history
        price_df = self.fetch_price_history(days + 60)
        
        if price_df is None or len(price_df) == 0:
            return []
        
        # Pivot prices
        pivot = price_df.pivot_table(
            index="date",
            columns="ticker",
            values="close"
        ).sort_index()
        
        results = []
        dates = pivot.index[-days:]
        
        for i, date in enumerate(dates):
            # Use data up to this date
            idx = pivot.index.get_loc(date)
            if idx < 20:  # Need at least 20 days for MA
                continue
            
            subset_pivot = pivot.iloc[:idx+1]
            
            latest_prices = subset_pivot.iloc[-1]
            prev_prices = subset_pivot.iloc[-2]
            daily_returns = (latest_prices - prev_prices) / prev_prices
            
            ma_20 = subset_pivot.tail(20).mean()
            ma_50 = subset_pivot.tail(50).mean() if len(subset_pivot) >= 50 else subset_pivot.mean()
            
            valid_tickers = latest_prices.notna().sum()
            advancers = (daily_returns > 0).sum()
            decliners = (daily_returns < 0).sum()
            above_20d = (latest_prices > ma_20).sum()
            above_50d = (latest_prices > ma_50).sum()
            
            daily_breadth = advancers / valid_tickers if valid_tickers > 0 else 0
            breadth_20d = above_20d / valid_tickers if valid_tickers > 0 else 0
            breadth_50d = above_50d / valid_tickers if valid_tickers > 0 else 0
            
            # Momentum (5D change)
            if idx >= 25:
                prices_5d_ago = subset_pivot.iloc[-6]
                ma_20_5d_ago = subset_pivot.iloc[-25:-5].mean()
                above_20d_5d_ago = (prices_5d_ago > ma_20_5d_ago).sum()
                breadth_20d_5d_ago = above_20d_5d_ago / valid_tickers if valid_tickers > 0 else 0
                breadth_momentum = breadth_20d - breadth_20d_5d_ago
            else:
                breadth_momentum = 0
            
            regime = self._get_regime(breadth_20d)
            
            results.append(BreadthData(
                date=date.strftime("%Y-%m-%d"),
                daily_breadth=daily_breadth,
                breadth_20d=breadth_20d,
                breadth_50d=breadth_50d,
                breadth_momentum=breadth_momentum,
                advancers=int(advancers),
                decliners=int(decliners),
                above_20d=int(above_20d),
                above_50d=int(above_50d),
                valid_tickers=int(valid_tickers),
                regime=regime,
                pillar_breadth=None  # Skip pillar details for history
            ))
        
        logger.info(f"Calculated {len(results)} days of breadth history")
        return results
    
    def save_to_supabase(self, data: BreadthData) -> bool:
        """Save breadth data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return False
        
        row = {
            "date": data.date,
            "daily_breadth": data.daily_breadth,
            "breadth_20d": data.breadth_20d,
            "breadth_50d": data.breadth_50d,
            "breadth_momentum": data.breadth_momentum,
            "advancers": data.advancers,
            "decliners": data.decliners,
            "above_20d": data.above_20d,
            "above_50d": data.above_50d,
            "valid_tickers": data.valid_tickers,
            "regime": data.regime,
            "pillar_breadth": data.pillar_breadth
        }
        
        try:
            self.supabase.table("breadth_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved breadth data for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def get_latest(self) -> Optional[BreadthData]:
        """Get most recent breadth data from Supabase."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table("breadth_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                row = response.data[0]
                return BreadthData(
                    date=row["date"],
                    daily_breadth=row["daily_breadth"],
                    breadth_20d=row["breadth_20d"],
                    breadth_50d=row["breadth_50d"],
                    breadth_momentum=row["breadth_momentum"],
                    advancers=row["advancers"],
                    decliners=row["decliners"],
                    above_20d=row["above_20d"],
                    above_50d=row["above_50d"],
                    valid_tickers=row["valid_tickers"],
                    regime=row["regime"],
                    pillar_breadth=row.get("pillar_breadth")
                )
        except Exception as e:
            logger.error(f"Failed to get latest: {e}")
        return None


# =============================================================================
# SUPABASE TABLE
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS breadth_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    daily_breadth FLOAT,
    breadth_20d FLOAT,
    breadth_50d FLOAT,
    breadth_momentum FLOAT,
    advancers INT,
    decliners INT,
    above_20d INT,
    above_50d INT,
    valid_tickers INT,
    regime VARCHAR(20),
    pillar_breadth JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_breadth_daily_date ON breadth_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run breadth calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Market Breadth")
    parser.add_argument("--history", type=int, default=0, help="Days of history to calculate")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = BreadthCalculator()
    
    print(f"\n{'='*60}")
    print("BREADTH DIAL")
    print(f"{'='*60}\n")
    
    if args.history > 0:
        # Calculate history
        history = calc.calculate_history(days=args.history)
        
        print(f"Calculated {len(history)} days of history")
        
        if history:
            print(f"\nMost recent: {history[-1].date}")
            print(f"  20D Breadth: {history[-1].breadth_20d*100:.1f}%")
            print(f"  Regime: {history[-1].regime}")
    else:
        # Calculate current
        data = calc.calculate()
        
        print(f"Date: {data.date}")
        print(f"\n{'='*40}")
        print("OVERALL BREADTH")
        print(f"{'='*40}")
        print(f"  Daily (% Up):     {data.daily_breadth*100:.1f}%")
        print(f"  20D (Above MA):   {data.breadth_20d*100:.1f}%")
        print(f"  50D (Above MA):   {data.breadth_50d*100:.1f}%")
        print(f"  Momentum (5D):    {data.breadth_momentum*100:+.1f}%")
        print(f"\n  Advancers: {data.advancers}")
        print(f"  Decliners: {data.decliners}")
        print(f"  Valid Tickers: {data.valid_tickers}")
        print(f"\n  REGIME: {data.regime}")
        
        if data.pillar_breadth:
            print(f"\n{'='*40}")
            print("PILLAR BREADTH")
            print(f"{'='*40}")
            for pillar, pb in data.pillar_breadth.items():
                print(f"  {pillar:15} {pb['breadth_20d']*100:5.1f}% ({pb['regime']})")
        
        if args.save:
            if calc.save_to_supabase(data):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
