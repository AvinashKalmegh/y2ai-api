"""
Price Sync - TwelveData to Supabase
===================================

Syncs stock prices from TwelveData API to Supabase price_history table.
Run this ONCE per day before running dials.

Matches Google Sheets architecture:
- TwelveData → Price_History sheet → All dials read from sheet

Usage:
    python price_sync.py              # Sync today's prices (43 stocks)
    python price_sync.py --full       # Full history sync (252 days)
    python price_sync.py --days 30    # Sync last 30 days
    python price_sync.py --vix        # Also sync VIX data
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import requests
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

# TwelveData rate limits (free tier)
RATE_LIMIT_PER_MINUTE = 8  # Free tier: 8 API credits/minute
DELAY_BETWEEN_CALLS = 60 / RATE_LIMIT_PER_MINUTE + 0.5  # ~8 seconds between calls

# All 43 stocks from pillar universe
PILLAR_STOCKS = {
    "Infrastructure": ["TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT", "CEG", "NRG", "EQIX", "DLR"],
    "Capex": ["KLAC", "LRCX", "AMAT", "QCOM", "NXPI", "ON", "SMCI", "ARM"],
    "Enterprise": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR", "ADBE", "ORCL", "MDB", "DDOG"],
    "Cybersecurity": ["ZS", "NET", "CRWD", "PANW"],
    "Consumer": ["TSLA", "SHOP", "UBER"],
    "Financial": ["GS", "MS", "JKS", "FSLR"],
}

# Additional tickers for ETFs and benchmarks
ETF_TICKERS = ["QQQ", "SPY", "SMH", "SOXX", "IGV", "HACK", "LQD", "HYG", "TLT", "IWM"]

# =============================================================================
# TWELVEDATA CLIENT (with proper rate limiting)
# =============================================================================

class TwelveDataSync:
    """Sync prices from TwelveData to Supabase with rate limiting."""
    
    def __init__(self):
        self.api_key = os.getenv("TWELVE_API_KEY")
        self.base_url = "https://api.twelvedata.com"
        
        # Supabase
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        if not self.api_key:
            logger.error("TWELVE_API_KEY not set in .env")
            sys.exit(1)
        
        if not self.supabase:
            logger.error("SUPABASE_URL or SUPABASE_KEY not set in .env")
            sys.exit(1)
        
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < DELAY_BETWEEN_CALLS:
            wait_time = DELAY_BETWEEN_CALLS - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def fetch_ticker(self, ticker: str, days: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetch price history for a single ticker.
        
        Args:
            ticker: Stock symbol
            days: Number of days of history
            
        Returns:
            DataFrame with date, ticker, close, open, high, low, volume
        """
        self._rate_limit_wait()
        
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": days,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Check for errors
            if "code" in data and data["code"] != 200:
                logger.warning(f"{ticker}: API error - {data.get('message', 'Unknown')}")
                return None
            
            if "values" not in data:
                logger.warning(f"{ticker}: No data returned")
                return None
            
            df = pd.DataFrame(data["values"])
            df["ticker"] = ticker
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.rename(columns={"datetime": "date"})
            
            # Convert numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            logger.info(f"{ticker}: Fetched {len(df)} rows")
            return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            
        except Exception as e:
            logger.error(f"{ticker}: Request failed - {e}")
            return None
    
    def fetch_all_stocks(self, days: int = 5) -> pd.DataFrame:
        """
        Fetch all pillar stocks with rate limiting.
        
        Args:
            days: Number of days of history per ticker
            
        Returns:
            Combined DataFrame
        """
        all_tickers = []
        for pillar_tickers in PILLAR_STOCKS.values():
            all_tickers.extend(pillar_tickers)
        all_tickers = list(set(all_tickers))
        
        logger.info(f"Fetching {len(all_tickers)} stocks, {days} days each")
        logger.info(f"Rate limit: {RATE_LIMIT_PER_MINUTE}/min, ~{DELAY_BETWEEN_CALLS:.1f}s between calls")
        
        estimated_time = len(all_tickers) * DELAY_BETWEEN_CALLS / 60
        logger.info(f"Estimated time: {estimated_time:.1f} minutes")
        
        all_data = []
        for i, ticker in enumerate(all_tickers):
            logger.info(f"[{i+1}/{len(all_tickers)}] Fetching {ticker}...")
            df = self.fetch_ticker(ticker, days)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            logger.error("No data fetched!")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total: {len(combined)} rows for {len(all_data)} tickers")
        
        return combined
    
    def fetch_etfs(self, days: int = 5) -> pd.DataFrame:
        """Fetch ETF prices."""
        logger.info(f"Fetching {len(ETF_TICKERS)} ETFs...")
        
        all_data = []
        for ticker in ETF_TICKERS:
            df = self.fetch_ticker(ticker, days)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def fetch_vix(self, days: int = 30) -> pd.DataFrame:
        """Fetch VIX data."""
        logger.info(f"Fetching VIX ({days} days)...")
        df = self.fetch_ticker("VIX", days)
        
        if df is not None:
            df = df.rename(columns={"close": "vix"})
            return df[["date", "vix"]]
        
        return pd.DataFrame()
    
    def save_prices_to_supabase(self, df: pd.DataFrame) -> int:
        """
        Save price data to Supabase price_history table.
        Uses upsert to handle duplicates.
        
        Returns:
            Number of rows saved
        """
        if df.empty:
            logger.warning("No data to save")
            return 0
        
        # Prepare records
        records = []
        for _, row in df.iterrows():
            record = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "ticker": row["ticker"],
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
            }
            
            # Optional columns
            if "open" in row and pd.notna(row["open"]):
                record["open"] = float(row["open"])
            if "high" in row and pd.notna(row["high"]):
                record["high"] = float(row["high"])
            if "low" in row and pd.notna(row["low"]):
                record["low"] = float(row["low"])
            if "volume" in row and pd.notna(row["volume"]):
                record["volume"] = int(row["volume"])
            
            records.append(record)
        
        # Batch upsert (Supabase handles duplicates via date+ticker constraint)
        batch_size = 500
        saved = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                response = self.supabase.table("price_history") \
                    .upsert(batch, on_conflict="date,ticker") \
                    .execute()
                saved += len(batch)
                logger.info(f"Saved batch {i//batch_size + 1}: {len(batch)} rows")
            except Exception as e:
                logger.error(f"Failed to save batch: {e}")
        
        logger.info(f"Total saved: {saved} rows to price_history")
        return saved
    
    def save_vix_to_supabase(self, df: pd.DataFrame) -> int:
        """Save VIX data to vix_dial_daily table."""
        if df.empty:
            return 0
        
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "vix": float(row["vix"]) if pd.notna(row["vix"]) else None,
            })
        
        try:
            for record in records:
                self.supabase.table("vix_dial_daily") \
                    .upsert(record, on_conflict="date") \
                    .execute()
            logger.info(f"Saved {len(records)} VIX records")
            return len(records)
        except Exception as e:
            logger.error(f"Failed to save VIX: {e}")
            return 0
    
    def sync_daily(self, include_etfs: bool = True, include_vix: bool = True):
        """
        Daily sync - fetch today's prices and save to Supabase.
        Run this before dials_runner.py
        """
        logger.info("=" * 60)
        logger.info("DAILY PRICE SYNC")
        logger.info(f"Started: {datetime.now()}")
        logger.info("=" * 60)
        
        # Fetch stocks (5 days to handle weekends)
        stock_df = self.fetch_all_stocks(days=5)
        if not stock_df.empty:
            self.save_prices_to_supabase(stock_df)
        
        # Fetch ETFs
        if include_etfs:
            etf_df = self.fetch_etfs(days=5)
            if not etf_df.empty:
                self.save_prices_to_supabase(etf_df)
        
        # Fetch VIX
        if include_vix:
            vix_df = self.fetch_vix(days=15)
            if not vix_df.empty:
                self.save_vix_to_supabase(vix_df)
        
        logger.info("=" * 60)
        logger.info(f"SYNC COMPLETE - {self.request_count} API calls made")
        logger.info("=" * 60)
    
    def sync_full(self, days: int = 252):
        """
        Full history sync - fetch extended history.
        Run this once to backfill data.
        """
        logger.info("=" * 60)
        logger.info(f"FULL HISTORY SYNC ({days} days)")
        logger.info(f"Started: {datetime.now()}")
        logger.info("WARNING: This will take ~{:.0f} minutes".format(
            (43 + len(ETF_TICKERS)) * days / 252 * DELAY_BETWEEN_CALLS / 60
        ))
        logger.info("=" * 60)
        
        # Fetch stocks
        stock_df = self.fetch_all_stocks(days=days)
        if not stock_df.empty:
            self.save_prices_to_supabase(stock_df)
        
        # Fetch ETFs
        etf_df = self.fetch_etfs(days=days)
        if not etf_df.empty:
            self.save_prices_to_supabase(etf_df)
        
        # Fetch VIX
        vix_df = self.fetch_vix(days=days)
        if not vix_df.empty:
            self.save_vix_to_supabase(vix_df)
        
        logger.info("=" * 60)
        logger.info(f"FULL SYNC COMPLETE - {self.request_count} API calls")
        logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sync prices from TwelveData to Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python price_sync.py              # Daily sync (recommended)
  python price_sync.py --full       # Full 252-day backfill
  python price_sync.py --days 30    # Last 30 days
  python price_sync.py --no-etfs    # Skip ETFs
  python price_sync.py --no-vix     # Skip VIX

Run this BEFORE dials_runner.py to ensure fresh data.
        """
    )
    
    parser.add_argument("--full", action="store_true", 
                        help="Full history sync (252 days)")
    parser.add_argument("--days", type=int, default=5,
                        help="Days of history to fetch (default: 5)")
    parser.add_argument("--no-etfs", action="store_true",
                        help="Skip ETF sync")
    parser.add_argument("--no-vix", action="store_true",
                        help="Skip VIX sync")
    
    args = parser.parse_args()
    
    sync = TwelveDataSync()
    
    if args.full:
        sync.sync_full(days=252)
    else:
        # Override days if specified
        if args.days != 5:
            stock_df = sync.fetch_all_stocks(days=args.days)
            if not stock_df.empty:
                sync.save_prices_to_supabase(stock_df)
            
            if not args.no_etfs:
                etf_df = sync.fetch_etfs(days=args.days)
                if not etf_df.empty:
                    sync.save_prices_to_supabase(etf_df)
            
            if not args.no_vix:
                vix_df = sync.fetch_vix(days=args.days)
                if not vix_df.empty:
                    sync.save_vix_to_supabase(vix_df)
        else:
            sync.sync_daily(
                include_etfs=not args.no_etfs,
                include_vix=not args.no_vix
            )


if __name__ == "__main__":
    main()
