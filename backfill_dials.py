"""
Backfill Historical Dial Values
================================

Recalculates historical breadth and pillar values using TwelveData prices
stored in Supabase. This fixes MCI discrepancies caused by old yfinance data.

Usage:
    python backfill_dials.py              # Backfill last 15 days
    python backfill_dials.py --days 30    # Backfill last 30 days
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PILLAR_STOCKS = {
    "Infrastructure": ["TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT", "CEG", "NRG", "EQIX", "DLR"],
    "Capex": ["KLAC", "LRCX", "AMAT", "QCOM", "NXPI", "ON", "SMCI", "ARM"],
    "Enterprise": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR", "ADBE", "ORCL", "MDB", "DDOG"],
    "Cybersecurity": ["ZS", "NET", "CRWD", "PANW"],
    "Consumer": ["TSLA", "SHOP", "UBER"],
    "Financial": ["GS", "MS", "JKS", "FSLR"],
}


class DialBackfill:
    """Backfill historical dial values."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("SUPABASE_URL or SUPABASE_KEY not set")
            sys.exit(1)
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.price_df: Optional[pd.DataFrame] = None
    
    def load_price_history(self, days: int = 60) -> pd.DataFrame:
        """Load price history from Supabase."""
        logger.info(f"Loading price history from Supabase ({days} days)...")
        
        all_tickers = []
        for stocks in PILLAR_STOCKS.values():
            all_tickers.extend(stocks)
        all_tickers = list(set(all_tickers))
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Batch tickers to avoid 1000 row limit
        all_data = []
        for i in range(0, len(all_tickers), 15):
            batch = all_tickers[i:i+15]
            response = self.supabase.table("price_history") \
                .select("date, ticker, close") \
                .in_("ticker", batch) \
                .gte("date", start_date) \
                .order("date", desc=False) \
                .limit(5000) \
                .execute()
            
            if response.data:
                all_data.extend(response.data)
        
        if not all_data:
            logger.error("No price data found in Supabase!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        
        logger.info(f"Loaded {len(df)} price records ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
        self.price_df = df
        return df
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers from pillar stocks."""
        tickers = []
        for stocks in PILLAR_STOCKS.values():
            tickers.extend(stocks)
        return list(set(tickers))
    
    def calculate_breadth_for_date(self, target_date: datetime) -> Optional[Dict]:
        """Calculate breadth metrics for a specific date."""
        if self.price_df is None or self.price_df.empty:
            return None
        
        # Get data up to target date
        df = self.price_df[self.price_df["date"] <= target_date].copy()
        
        if df.empty:
            return None
        
        # Pivot to get tickers as columns
        pivot = df.pivot_table(index="date", columns="ticker", values="close")
        pivot = pivot.sort_index()
        
        # Need at least 50 days for 50D MA
        if len(pivot) < 20:
            logger.warning(f"Not enough data for {target_date.strftime('%Y-%m-%d')}")
            return None
        
        # Get latest row (target date or closest before)
        latest_date = pivot.index[pivot.index <= target_date].max()
        if pd.isna(latest_date):
            return None
        
        # Calculate 20D and 50D moving averages
        ma_20 = pivot.rolling(window=20).mean()
        ma_50 = pivot.rolling(window=50).mean()
        
        # Get values for target date
        latest_prices = pivot.loc[latest_date]
        latest_ma20 = ma_20.loc[latest_date]
        latest_ma50 = ma_50.loc[latest_date] if latest_date in ma_50.index else None
        
        # Calculate breadth (% above MA)
        valid_20 = latest_ma20.dropna()
        above_20 = (latest_prices[valid_20.index] > valid_20).sum()
        breadth_20d = (above_20 / len(valid_20)) * 100 if len(valid_20) > 0 else 0
        
        breadth_50d = 0
        if latest_ma50 is not None:
            valid_50 = latest_ma50.dropna()
            if len(valid_50) > 0:
                above_50 = (latest_prices[valid_50.index] > valid_50).sum()
                breadth_50d = (above_50 / len(valid_50)) * 100
        
        # Calculate per-pillar breadth
        pillar_breadth = {}
        for pillar, stocks in PILLAR_STOCKS.items():
            available = [s for s in stocks if s in valid_20.index]
            if available:
                above = (latest_prices[available] > latest_ma20[available]).sum()
                pillar_breadth[pillar] = round((above / len(available)) * 100, 1)
        
        return {
            "date": latest_date.strftime("%Y-%m-%d"),
            "breadth_20d": round(breadth_20d / 100, 4),  # Save as decimal
            "breadth_50d": round(breadth_50d / 100, 4),  # Save as decimal
            "stocks_above_20d": int(above_20),
            "stocks_above_50d": int((latest_prices[valid_50.index] > valid_50).sum()) if latest_ma50 is not None and len(valid_50) > 0 else 0,
            "total_stocks": len(valid_20),
            "pillar_breadth": pillar_breadth,
        }
    
    def calculate_pillar_for_date(self, target_date: datetime) -> Optional[Dict]:
        """Calculate pillar index for a specific date."""
        if self.price_df is None or self.price_df.empty:
            return None
        
        df = self.price_df[self.price_df["date"] <= target_date].copy()
        
        if df.empty:
            return None
        
        # Pivot
        pivot = df.pivot_table(index="date", columns="ticker", values="close")
        pivot = pivot.sort_index()
        
        if len(pivot) < 5:
            return None
        
        # Get latest date
        latest_date = pivot.index[pivot.index <= target_date].max()
        if pd.isna(latest_date):
            return None
        
        result = {"date": latest_date.strftime("%Y-%m-%d")}
        
        # Map pillar names to DB column names
        pillar_map = {
            "Infrastructure": "infra",
            "Capex": "productivity",  # Capex -> productivity in DB
            "Enterprise": "enterprise",
            "Cybersecurity": "macro",  # Cyber -> macro in DB
            "Consumer": "demand",
            "Financial": "financial",
        }
        
        for pillar, stocks in PILLAR_STOCKS.items():
            available = [s for s in stocks if s in pivot.columns]
            if not available:
                continue
            
            db_name = pillar_map.get(pillar, pillar.lower())
            
            # Calculate pillar index (normalized to 100)
            pillar_prices = pivot[available]
            if len(pillar_prices) > 0:
                # Use first available date as base
                base_prices = pillar_prices.iloc[0]
                normalized = (pillar_prices / base_prices) * 100
                pillar_index = normalized.mean(axis=1)
                
                latest_idx = pillar_index.loc[latest_date]
                if pd.notna(latest_idx):
                    result[f"{db_name}_index"] = round(float(latest_idx), 2)
                
                # Calculate returns
                pillar_returns = pillar_index.pct_change(fill_method=None)
                
                # 1D return
                if len(pillar_returns) >= 2 and pd.notna(pillar_returns.iloc[-1]):
                    result[f"{db_name}_return"] = round(float(pillar_returns.iloc[-1]) * 100, 2)
                
                # 5D return
                if len(pillar_index) >= 6:
                    ret_5d = (pillar_index.iloc[-1] / pillar_index.iloc[-6] - 1) * 100
                    if pd.notna(ret_5d):
                        result[f"{db_name}_5d"] = round(float(ret_5d), 2)
                
                # 1M return (21 days)
                if len(pillar_index) >= 22:
                    ret_1m = (pillar_index.iloc[-1] / pillar_index.iloc[-22] - 1) * 100
                    if pd.notna(ret_1m):
                        result[f"{db_name}_1m"] = round(float(ret_1m), 2)
        
        # Only return if we have more than just the date
        if len(result) <= 1:
            return None
        
        return result
    
    def backfill_breadth(self, days: int = 15):
        """Backfill breadth_daily for past N days."""
        logger.info(f"Backfilling breadth_daily for {days} trading days...")
        
        # Get actual trading dates from data
        available_dates = sorted(self.price_df["date"].unique(), reverse=True)
        dates_to_process = available_dates[:min(days, len(available_dates))]
        
        saved = 0
        for target_date in dates_to_process:
            result = self.calculate_breadth_for_date(target_date)
            if result is None:
                continue
            
            # Prepare record for Supabase (only columns that exist)
            record = {
                "date": result["date"],
                "breadth_20d": result["breadth_20d"],
                "breadth_50d": result["breadth_50d"],
            }
            
            try:
                self.supabase.table("breadth_daily") \
                    .upsert(record, on_conflict="date") \
                    .execute()
                saved += 1
                logger.info(f"  {result['date']}: breadth_20d={result['breadth_20d']*100:.1f}%")
            except Exception as e:
                logger.warning(f"  Failed to save {result['date']}: {e}")
        
        logger.info(f"Backfilled {saved} breadth records")
        return saved
    
    def backfill_pillar(self, days: int = 15):
        """Backfill pillar_index_daily for past N days."""
        logger.info(f"Backfilling pillar_index_daily for {days} trading days...")
        
        # Get actual trading dates from data
        available_dates = sorted(self.price_df["date"].unique(), reverse=True)
        dates_to_process = available_dates[:min(days, len(available_dates))]
        
        saved = 0
        for target_date in dates_to_process:
            result = self.calculate_pillar_for_date(target_date)
            if result is None:
                continue
            
            try:
                self.supabase.table("pillar_index_daily") \
                    .upsert(result, on_conflict="date") \
                    .execute()
                saved += 1
                logger.info(f"  {result['date']}: saved pillar index")
            except Exception as e:
                logger.warning(f"  Failed to save {result['date']}: {e}")
        
        logger.info(f"Backfilled {saved} pillar records")
        return saved
    
    def run(self, days: int = 15):
        """Run full backfill."""
        logger.info("=" * 60)
        logger.info("DIAL BACKFILL")
        logger.info(f"Started: {datetime.now()}")
        logger.info(f"Backfill period: {days} days")
        logger.info("=" * 60)
        
        # Load price history (need 60+ extra days for 50D MA calculations)
        self.load_price_history(days=max(days + 60, 75))
        
        if self.price_df is None or self.price_df.empty:
            logger.error("No price data - run price_sync.py first!")
            return
        
        # Get available trading dates from the data
        available_dates = sorted(self.price_df["date"].unique())
        logger.info(f"Available dates: {len(available_dates)} trading days")
        
        # Backfill
        self.backfill_breadth(days)
        self.backfill_pillar(days)
        
        logger.info("=" * 60)
        logger.info("BACKFILL COMPLETE")
        logger.info("Now run: python dials_runner.py --all")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backfill historical dial values")
    parser.add_argument("--days", type=int, default=15, help="Days to backfill")
    args = parser.parse_args()
    
    backfill = DialBackfill()
    backfill.run(args.days)


if __name__ == "__main__":
    main()