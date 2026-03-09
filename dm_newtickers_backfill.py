"""
Targeted backfill for specific tickers.
Fetches full price history (2016-present) and calculates DM.
Does NOT touch any existing data for other tickers.

Usage: python backfill_tickers.py
"""
import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY') or os.getenv('TWELVE_API_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# CONFIGURATION
# ============================================================
TICKERS_TO_BACKFILL = [
    # Marketplace Watchlist + Intraday Scanner additions
    "FVRR", "GTLB", "HUBS", "TWLO", "UPWK",
    "RBLX", "BILL", "DOCN",
]

START_DATE = "2016-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

DELAY_BETWEEN_CALLS = 8  # seconds (Twelve Data free tier: 8/min)

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Semiconductors": "SMH",
    "Software": "IGV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Biotechnology": "XBI",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Energy": "XLE",
    "Communication Services": "XLC",
    "Nuclear": "URA",
    "Uranium": "URA",
    "Clean Energy": "TAN",
    "Cybersecurity": "HACK",
    "Aerospace & Defense": "ITA",
    "Defense": "ITA",
    "Transportation": "IYT",
}

# ============================================================
# PRICE FETCH
# ============================================================
def fetch_prices(ticker, start_date=START_DATE, end_date=END_DATE):
    """Fetch daily prices from Twelve Data."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "outputsize": 5000,
        "apikey": TWELVEDATA_API_KEY,
    }
    
    resp = requests.get(url, params=params)
    data = resp.json()
    
    if data.get("status") == "error":
        print(f"  API error for {ticker}: {data.get('message')}")
        return []
    
    values = data.get("values", [])
    if not values:
        print(f"  No data returned for {ticker}")
        return []
    
    rows = []
    for v in values:
        rows.append({
            "ticker": ticker,
            "date": v["datetime"],
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low": float(v["low"]),
            "close": float(v["close"]),
            "volume": int(v["volume"]),
        })
    
    return rows


def upload_prices(rows):
    """Upload price rows to Supabase."""
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            supabase.table("price_history").upsert(
                batch, on_conflict="ticker,date"
            ).execute()
        except Exception as e:
            print(f"  Price upload error: {e}")


# ============================================================
# DM CALCULATION  
# ============================================================
def load_prices_from_supabase(ticker):
    """Load all prices for a ticker from Supabase."""
    all_rows = []
    offset = 0
    page_size = 1000
    
    while True:
        result = supabase.table("price_history") \
            .select("date,close,volume") \
            .eq("ticker", ticker) \
            .order("date") \
            .range(offset, offset + page_size - 1) \
            .execute()
        
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_ticker_sector(ticker):
    """Get sector for a ticker from scanner_universe."""
    result = supabase.table("scanner_universe") \
        .select("sector") \
        .eq("ticker", ticker) \
        .execute()
    if result.data:
        return result.data[0]['sector']
    return "Information Technology"  # default


def calculate_dm(ticker_df, spy_df, etf_df, etf_symbol):
    """Calculate DM scores for a ticker."""
    results = []
    lookback = 20
    vol_baseline = 60
    
    for i in range(lookback, len(ticker_df)):
        date = ticker_df.iloc[i]['date']
        
        # Find matching SPY and ETF rows
        spy_row = spy_df[spy_df['date'] == date]
        etf_row = etf_df[etf_df['date'] == date]
        
        if spy_row.empty or etf_row.empty:
            continue
        
        # 20-day returns
        ticker_close = ticker_df.iloc[i]['close']
        ticker_close_20d = ticker_df.iloc[i - lookback]['close']
        return_20d = (ticker_close - ticker_close_20d) / ticker_close_20d
        
        spy_close = spy_row.iloc[0]['close']
        spy_match_20d = spy_df[spy_df['date'] == ticker_df.iloc[i - lookback]['date']]
        if spy_match_20d.empty:
            continue
        spy_return_20d = (spy_close - spy_match_20d.iloc[0]['close']) / spy_match_20d.iloc[0]['close']
        
        etf_close = etf_row.iloc[0]['close']
        etf_match_20d = etf_df[etf_df['date'] == ticker_df.iloc[i - lookback]['date']]
        if etf_match_20d.empty:
            continue
        etf_return_20d = (etf_close - etf_match_20d.iloc[0]['close']) / etf_match_20d.iloc[0]['close']
        
        # Relative strength
        rel_str_etf = (return_20d - etf_return_20d) * 100
        rel_str_spy = (return_20d - spy_return_20d) * 100
        
        # Volume Z-score
        vol_5d = ticker_df.iloc[max(0, i-4):i+1]['volume'].mean()
        vol_start = max(0, i - vol_baseline)
        vol_baseline_avg = ticker_df.iloc[vol_start:i+1]['volume'].mean()
        
        if vol_baseline_avg > 0:
            vol_std = ticker_df.iloc[vol_start:i+1]['volume'].std()
            volume_z = (vol_5d - vol_baseline_avg) / vol_std if vol_std > 0 else 0
        else:
            volume_z = 0
        
        # 20-day avg volume
        vol_20d_avg = ticker_df.iloc[max(0, i-19):i+1]['volume'].mean()
        vol_ratio = vol_5d / vol_baseline_avg if vol_baseline_avg > 0 else 1
        
        # DM Raw: RelStr_ETF * 0.50 + RelStr_SPY * 0.30 + Volume_Z * 0.20
        dm_raw = rel_str_etf * 0.50 + rel_str_spy * 0.30 + volume_z * 0.20
        
        results.append({
            'date': date.strftime('%Y-%m-%d'),
            'ticker': ticker_df.iloc[i].get('ticker', ''),
            'close': round(ticker_close, 2),
            'volume': int(ticker_df.iloc[i]['volume']),
            'return_20d': round(return_20d, 6),
            'etf_return_20d': round(etf_return_20d, 6),
            'spy_return_20d': round(spy_return_20d, 6),
            'vol_20d_avg': round(vol_20d_avg, 2),
            'vol_ratio': round(vol_ratio, 4),
            'rel_str_etf': round(rel_str_etf, 2),
            'rel_str_spy': round(rel_str_spy, 2),
            'volume_z': round(volume_z, 4),
            'dm_raw': round(dm_raw, 4),
            'etf': etf_symbol,
        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Smoothing: DM = clamp(EMA5(dm_raw), 10, 100)
    df['dm_smoothed'] = df['dm_raw'].ewm(span=5, adjust=False).mean()
    df['dm_smoothed'] = df['dm_smoothed'].clip(lower=10, upper=100).round(2)
    
    # ETF DM (using ETF's own rel strength vs SPY)
    df['etf_dm'] = 50  # placeholder
    
    # Lead = DM - ETF_DM
    df['lead'] = (df['dm_smoothed'] - df['etf_dm']).round(2)
    
    # Phase
    def get_phase(dm):
        if dm >= 70:
            return "STRONG"
        elif dm >= 50:
            return "BUILD"
        elif dm >= 30:
            return "WEAK"
        else:
            return "EXIT"
    
    df['phase'] = df['dm_smoothed'].apply(get_phase)
    
    return df


def upload_dm_history(df):
    """Upload DM results to Supabase dm_history."""
    records = df.to_dict('records')
    
    # Select only the columns that exist in dm_history
    fields = [
        'date', 'ticker', 'close', 'volume', 'return_20d',
        'etf_return_20d', 'spy_return_20d', 'vol_20d_avg',
        'vol_ratio', 'rel_str_etf', 'rel_str_spy', 'volume_z',
        'dm_raw', 'dm_smoothed', 'etf', 'etf_dm', 'lead', 'phase'
    ]
    
    clean_records = []
    for r in records:
        clean = {}
        for f in fields:
            val = r.get(f, '')
            # Convert numpy types to Python types
            if hasattr(val, 'item'):
                val = val.item()
            if pd.isna(val):
                val = None
            clean[f] = val
        clean_records.append(clean)
    
    batch_size = 500
    uploaded = 0
    for i in range(0, len(clean_records), batch_size):
        batch = clean_records[i:i + batch_size]
        try:
            supabase.table("dm_history").upsert(
                batch, on_conflict="ticker,date"
            ).execute()
            uploaded += len(batch)
        except Exception as e:
            print(f"  DM upload error at batch {i}: {e}")
            # Try one-by-one
            for row in batch:
                try:
                    supabase.table("dm_history").upsert(
                        [row], on_conflict="ticker,date"
                    ).execute()
                    uploaded += 1
                except Exception as e2:
                    print(f"  Skip {row['date']}: {e2}")
    
    return uploaded


def update_dm_latest(ticker, dm_df):
    """Update dm_latest with the most recent DM row."""
    if dm_df.empty:
        return
    
    last = dm_df.iloc[-1]
    
    record = {
        'ticker': ticker,
        'date': last['date'],
        'dm_smoothed': float(last['dm_smoothed']),
        'phase': last['phase'],
        'lead': float(last['lead']),
        'etf_dm': float(last.get('etf_dm', 50)),
        'dm_7d_ago': float(dm_df.iloc[-6]['dm_smoothed']) if len(dm_df) >= 6 else None,
        'dm_change': round(float(last['dm_smoothed']) - float(dm_df.iloc[-6]['dm_smoothed']), 2) if len(dm_df) >= 6 else None,
        'volume_z': float(last['volume_z']),
        'rel_str_etf': float(last['rel_str_etf']),
        'rel_str_spy': float(last['rel_str_spy']),
        'close': float(last['close']),
        'etf': last['etf'],
    }
    
    supabase.table("dm_latest").upsert(
        record, on_conflict="ticker"
    ).execute()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("TARGETED TICKER BACKFILL")
    print(f"Tickers: {', '.join(TICKERS_TO_BACKFILL)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("=" * 60)
    print()
    
    # Step 1: Fetch prices for new tickers
    print("PHASE 1: FETCHING PRICES")
    print("-" * 40)
    
    etfs_needed = set(["SPY"])
    for ticker in TICKERS_TO_BACKFILL:
        sector = get_ticker_sector(ticker)
        etf = SECTOR_ETF_MAP.get(sector, "SPY")
        etfs_needed.add(etf)
    
    all_to_fetch = TICKERS_TO_BACKFILL + list(etfs_needed)
    
    for i, ticker in enumerate(all_to_fetch):
        # Check if already has full data
        result = supabase.table("price_history") \
            .select("date", count="exact") \
            .eq("ticker", ticker) \
            .execute()
        existing = result.count or 0
        
        if existing >= 2400:  # ~10 years
            print(f"  {ticker}: {existing} rows already - SKIP")
            continue
        
        print(f"  {ticker}: Fetching ({existing} existing rows)...")
        rows = fetch_prices(ticker)
        
        if rows:
            upload_prices(rows)
            print(f"  {ticker}: OK {len(rows)} rows fetched and uploaded")
        else:
            print(f"  {ticker}: WARN No data returned")
        
        if i < len(all_to_fetch) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)
    
    print()
    
    # Step 2: Calculate DM
    print("PHASE 2: CALCULATING DM")
    print("-" * 40)
    
    # Load SPY once
    print("  Loading SPY prices...")
    spy_df = load_prices_from_supabase("SPY")
    print(f"  SPY: {len(spy_df)} rows")
    
    # Cache ETF dataframes
    etf_cache = {}
    
    for ticker in TICKERS_TO_BACKFILL:
        sector = get_ticker_sector(ticker)
        etf_symbol = SECTOR_ETF_MAP.get(sector, "SPY")
        
        print(f"  {ticker} (sector: {sector}, ETF: {etf_symbol})...")
        
        # Load ticker prices
        ticker_df = load_prices_from_supabase(ticker)
        if ticker_df.empty:
            print(f"  {ticker}: No price data. Skipping.")
            continue
        
        ticker_df['ticker'] = ticker
        print(f"    Prices: {len(ticker_df)} rows ({ticker_df['date'].min().strftime('%Y-%m-%d')} to {ticker_df['date'].max().strftime('%Y-%m-%d')})")
        
        # Load ETF
        if etf_symbol not in etf_cache:
            etf_cache[etf_symbol] = load_prices_from_supabase(etf_symbol)
            print(f"    Loaded {etf_symbol}: {len(etf_cache[etf_symbol])} rows")
        etf_df = etf_cache[etf_symbol]
        
        # Calculate DM
        dm_df = calculate_dm(ticker_df, spy_df, etf_df, etf_symbol)
        
        if dm_df.empty:
            print(f"    No DM results. Skipping.")
            continue
        
        print(f"    DM calculated: {len(dm_df)} rows")
        
        # Upload
        uploaded = upload_dm_history(dm_df)
        print(f"    Uploaded {uploaded} rows to dm_history")
        
        # Update dm_latest
        update_dm_latest(ticker, dm_df)
        print(f"    OK dm_latest updated (DM={dm_df.iloc[-1]['dm_smoothed']}, phase={dm_df.iloc[-1]['phase']})")
        print()
    
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print()
    print("Next: run 'python dm_push_to_sheets.py daily' to sync Google Sheets")


if __name__ == "__main__":
    main()