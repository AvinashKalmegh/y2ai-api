"""
PRICE HISTORY BACKFILL
======================
Backfills Supabase price_history to match Google Sheets (2020-01-02).

TwelveData free tier: 800 calls/day, 8 calls/minute
53 tickers × 1 call each = 53 calls total (~7 minutes)
"""

import os
import time
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TWELVEDATA_API_KEY = os.getenv("TWELVE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Target start date (match Google Sheets)
START_DATE = "2020-01-02"
END_DATE = "2025-04-24"  # Day before current Supabase data starts

# All 53 tickers from your universe
TICKERS = [
    "ADBE", "AMAT", "AMD", "AMZN", "ARM", "ASML", "AVGO", "CEG", "CRM", "CRWD",
    "DDOG", "DLR", "EQIX", "FSLR", "GOOGL", "GS", "HACK", "HYG", "IGV", "INTC",
    "IWM", "JKS", "KLAC", "LQD", "LRCX", "MDB", "META", "MS", "MSFT", "MU",
    "NET", "NOW", "NRG", "NVDA", "NXPI", "ON", "ORCL", "PANW", "PLTR", "QCOM",
    "QQQ", "SHOP", "SMCI", "SMH", "SNOW", "SOXX", "SPY", "TLT", "TSLA", "TSM",
    "UBER", "VRT", "ZS"
]

# Rate limiting
CALLS_PER_MINUTE = 8
DELAY_BETWEEN_CALLS = 60 / CALLS_PER_MINUTE  # 7.5 seconds


def fetch_ticker_history(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch historical prices from TwelveData."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": 5000
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("status") == "error":
        raise Exception(f"API error for {ticker}: {data.get('message')}")
    
    if "values" not in data:
        raise Exception(f"No data returned for {ticker}")
    
    rows = []
    for v in data["values"]:
        rows.append({
            "date": v["datetime"],
            "ticker": ticker,
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
            "volume": int(float(v["volume"]))
        })
    
    return rows


def upload_to_supabase(rows: list) -> int:
    """Upload rows to Supabase price_history table."""
    from supabase import create_client
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Upsert in batches of 500
    batch_size = 500
    uploaded = 0
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            supabase.table("price_history").upsert(
                batch, 
                on_conflict="date,ticker"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"Upload error: {e}")
            # Try inserting one by one to find problematic rows
            for row in batch:
                try:
                    supabase.table("price_history").upsert(
                        [row],
                        on_conflict="date,ticker"
                    ).execute()
                    uploaded += 1
                except Exception as e2:
                    logger.warning(f"Skipping {row['ticker']} {row['date']}: {e2}")
    
    return uploaded


def run_backfill():
    """Main backfill function."""
    if not TWELVEDATA_API_KEY:
        raise Exception("TWELVEDATA_API_KEY not set in environment")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("SUPABASE_URL and SUPABASE_KEY must be set")
    
    logger.info("=" * 60)
    logger.info("PRICE HISTORY BACKFILL")
    logger.info("=" * 60)
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Tickers: {len(TICKERS)}")
    logger.info(f"Estimated time: ~{len(TICKERS) * DELAY_BETWEEN_CALLS / 60:.1f} minutes")
    logger.info("=" * 60)
    
    total_rows = 0
    failed_tickers = []
    
    for i, ticker in enumerate(TICKERS, 1):
        logger.info(f"[{i}/{len(TICKERS)}] Fetching {ticker}...")
        
        try:
            rows = fetch_ticker_history(ticker, START_DATE, END_DATE)
            logger.info(f"  Got {len(rows)} rows")
            
            uploaded = upload_to_supabase(rows)
            logger.info(f"  Uploaded {uploaded} rows to Supabase")
            
            total_rows += uploaded
            
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            failed_tickers.append(ticker)
        
        # Rate limiting (skip delay on last ticker)
        if i < len(TICKERS):
            time.sleep(DELAY_BETWEEN_CALLS)
    
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rows uploaded: {total_rows}")
    
    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers)}")
    else:
        logger.info("All tickers successful!")
    
    return total_rows, failed_tickers


if __name__ == "__main__":
    run_backfill()