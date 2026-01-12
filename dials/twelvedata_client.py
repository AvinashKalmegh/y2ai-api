"""
TwelveData API Client
=====================

Shared client for fetching price data from TwelveData API.
Used by all dials to ensure consistency with Google Sheets data source.

API Documentation: https://twelvedata.com/docs
Free tier: 800 calls/day, 8 requests/minute
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

TWELVEDATA_CONFIG = {
    "api_key": os.getenv("TWELVE_API_KEY"),
    "base_url": "https://api.twelvedata.com",
    "rate_limit_delay": 0.15,  # 8 requests/min = ~7.5s between calls, but batch helps
    "batch_size": 8,  # Max symbols per batch request
    "retry_attempts": 3,
    "retry_delay": 2,
}


# =============================================================================
# TWELVEDATA CLIENT
# =============================================================================

class TwelveDataClient:
    """Client for fetching data from TwelveData API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or TWELVEDATA_CONFIG["api_key"]
        self.base_url = TWELVEDATA_CONFIG["base_url"]
        self.last_request_time = 0
        
        if not self.api_key:
            logger.warning("TWELVE_API_KEY not set in environment")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < TWELVEDATA_CONFIG["rate_limit_delay"]:
            time.sleep(TWELVEDATA_CONFIG["rate_limit_delay"] - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make a request to TwelveData API with retry logic."""
        if not self.api_key:
            logger.error("No API key configured")
            return None
        
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(TWELVEDATA_CONFIG["retry_attempts"]):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for API errors
                    if "code" in data and data["code"] != 200:
                        logger.warning(f"API error: {data.get('message', 'Unknown error')}")
                        return None
                    
                    return data
                    
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    logger.warning("Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")
                    
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < TWELVEDATA_CONFIG["retry_attempts"] - 1:
                    time.sleep(TWELVEDATA_CONFIG["retry_delay"])
        
        return None
    
    def fetch_time_series(
        self, 
        symbol: str, 
        interval: str = "1day",
        outputsize: int = 252,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch time series data for a single symbol.
        
        Args:
            symbol: Stock ticker (e.g., "NVDA")
            interval: Time interval ("1day", "1hour", etc.)
            outputsize: Number of data points
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        data = self._make_request("time_series", params)
        
        if not data or "values" not in data:
            logger.warning(f"No data for {symbol}")
            return None
        
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"datetime": "date"})
        
        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["ticker"] = symbol
        
        return df[["date", "ticker", "close", "open", "high", "low", "volume"]]
    
    def fetch_batch_time_series(
        self,
        symbols: List[str],
        interval: str = "1day",
        outputsize: int = 252,
        start_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch time series data for multiple symbols.
        
        Args:
            symbols: List of stock tickers
            interval: Time interval
            outputsize: Number of data points per symbol
            start_date: Start date (YYYY-MM-DD)
            
        Returns:
            Combined DataFrame with all symbols
        """
        all_data = []
        
        # Process in batches to avoid rate limits
        batch_size = TWELVEDATA_CONFIG["batch_size"]
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_str = ",".join(batch)
            
            logger.info(f"Fetching batch {i//batch_size + 1}: {batch_str}")
            
            params = {
                "symbol": batch_str,
                "interval": interval,
                "outputsize": outputsize,
            }
            
            if start_date:
                params["start_date"] = start_date
            
            data = self._make_request("time_series", params)
            
            if not data:
                continue
            
            # Handle single vs multiple symbols response format
            if len(batch) == 1:
                # Single symbol - data is directly in response
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["ticker"] = batch[0]
                    all_data.append(df)
            else:
                # Multiple symbols - data is nested by symbol
                for symbol in batch:
                    if symbol in data and "values" in data[symbol]:
                        df = pd.DataFrame(data[symbol]["values"])
                        df["ticker"] = symbol
                        all_data.append(df)
                    elif symbol in data and "status" in data[symbol]:
                        logger.warning(f"No data for {symbol}: {data[symbol].get('message', 'Unknown')}")
        
        if not all_data:
            logger.warning("No data fetched from TwelveData")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"])
        combined = combined.rename(columns={"datetime": "date"})
        
        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
        
        logger.info(f"Fetched {len(combined)} total rows for {len(symbols)} symbols")
        
        return combined[["date", "ticker", "close"]]
    
    def fetch_quote(self, symbol: str) -> Optional[dict]:
        """
        Fetch real-time quote for a symbol.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with price, change, volume, etc.
        """
        data = self._make_request("quote", {"symbol": symbol})
        return data
    
    def fetch_vix(self, outputsize: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch VIX index data.
        
        Note: TwelveData uses "VIX" for the CBOE Volatility Index.
        
        Returns:
            DataFrame with date and close (VIX value)
        """
        df = self.fetch_time_series("VIX", outputsize=outputsize)
        if df is not None:
            return df[["date", "close"]].rename(columns={"close": "vix"})
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fetch_prices(tickers: List[str], days: int = 252) -> pd.DataFrame:
    """
    Convenience function to fetch prices for multiple tickers.
    
    Args:
        tickers: List of stock tickers
        days: Number of trading days to fetch
        
    Returns:
        DataFrame with date, ticker, close columns
    """
    client = TwelveDataClient()
    
    if not client.api_key:
        logger.warning("TWELVE_API_KEY not set, falling back to yfinance")
        return _fallback_yfinance(tickers, days)
    
    df = client.fetch_batch_time_series(tickers, outputsize=days)
    
    if df.empty:
        logger.warning("TwelveData returned no data, falling back to yfinance")
        return _fallback_yfinance(tickers, days)
    
    return df


def fetch_vix_current() -> Optional[float]:
    """Fetch current VIX value."""
    client = TwelveDataClient()
    
    if not client.api_key:
        return None
    
    quote = client.fetch_quote("VIX")
    if quote and "close" in quote:
        return float(quote["close"])
    return None


def _fallback_yfinance(tickers: List[str], days: int) -> pd.DataFrame:
    """Fallback to yfinance if TwelveData fails."""
    try:
        import yfinance as yf
        
        end = datetime.now()
        start = end - timedelta(days=days * 2)  # Extra buffer for non-trading days
        
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            df = data[["Close"]].reset_index()
            df.columns = ["date", "close"]
            df["ticker"] = tickers[0]
        else:
            df = data["Close"].reset_index().melt(
                id_vars=["Date"],
                var_name="ticker",
                value_name="close"
            )
            df = df.rename(columns={"Date": "date"})
        
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "ticker", "close"]].dropna()
        
    except Exception as e:
        logger.error(f"yfinance fallback failed: {e}")
        return pd.DataFrame()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TwelveData client")
    parser.add_argument("--symbol", default="NVDA", help="Symbol to test")
    parser.add_argument("--days", type=int, default=10, help="Days of data")
    args = parser.parse_args()
    
    client = TwelveDataClient()
    
    print(f"\nTesting TwelveData API for {args.symbol}...")
    print(f"API Key configured: {'Yes' if client.api_key else 'No'}")
    
    if client.api_key:
        df = client.fetch_time_series(args.symbol, outputsize=args.days)
        if df is not None:
            print(f"\nFetched {len(df)} rows:")
            print(df.head(10))
        else:
            print("Failed to fetch data")
    else:
        print("Set TWELVE_API_KEY in .env file")
