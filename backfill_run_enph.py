"""
BACKFILL RUN AND ENPH
=====================
Adds missing tickers to Supabase price_history for pillar index alignment.

Run with: python backfill_run_enph.py

Requires environment variables:
- TWELVEDATA_API_KEY
- SUPABASE_URL  
- SUPABASE_KEY
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

# Date range (match Google Sheets base date through today)
START_DATE = "2020-01-02"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Tickers to backfill
TICKERS = ["RUN", "ENPH"]

# Rate limiting (TwelveData free tier: 8 calls/minute)
DELAY_BETWEEN_CALLS = 8  # seconds


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
    logger.info("BACKFILL: RUN and ENPH")
    logger.info("=" * 60)
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Tickers: {TICKERS}")
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
        
        if i < len(TICKERS):
            time.sleep(DELAY_BETWEEN_CALLS)
    
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rows uploaded: {total_rows}")
    
    if failed_tickers:
        logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")
    else:
        logger.info("All tickers successful!")
    
    # Next steps
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Recalculate pillar index (1260 days):")
    logger.info("   from dials.pillar_index import PillarIndexCalculator")
    logger.info("   calc = PillarIndexCalculator()")
    logger.info("   data = calc.calculate(days=1260)")
    logger.info("   calc.save_to_supabase(data)")
    logger.info("")
    logger.info("2. Recalculate dependent dials: MCI, Correlation, Cluster, Breadth")
    
    return total_rows, failed_tickers


if __name__ == "__main__":
    run_backfill()