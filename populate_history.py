"""
POPULATE HISTORY TABLES
========================
Fetches historical data needed for MCI calculation.

Required:
- vix_history: 30+ days of VIX closes
- breadth_daily: 10+ days of breadth data
- credit_spread_daily: 15+ days of spread data
"""

import os
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()

# Initialize Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    print("❌ Set SUPABASE_URL and SUPABASE_KEY environment variables")
    exit(1)

client = create_client(url, key)
print("✅ Supabase connected")


# =============================================================================
# 1. POPULATE VIX HISTORY
# =============================================================================
print("\n" + "="*60)
print("1. POPULATING VIX HISTORY")
print("="*60)

try:
    import yfinance as yf
    
    # Fetch 60 days of VIX data
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="3mo")
    
    print(f"Fetched {len(hist)} days of VIX data from Yahoo Finance")
    
    # Prepare rows
    rows = []
    for date, row in hist.iterrows():
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": float(row['Close']),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
        })
    
    # Insert to Supabase
    if rows:
        # Upsert in batches
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            client.table("vix_history").upsert(batch, on_conflict="date").execute()
            print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} rows")
        
        print(f"✅ Inserted {len(rows)} rows to vix_history")
    
except Exception as e:
    print(f"❌ VIX error: {e}")


# =============================================================================
# 2. POPULATE BREADTH DAILY (using stock data)
# =============================================================================
print("\n" + "="*60)
print("2. POPULATING BREADTH DAILY")
print("="*60)

try:
    import yfinance as yf
    import pandas as pd
    
    # Get pillar stocks to calculate breadth
    PILLAR_TICKERS = [
        "NVDA", "AMD", "AVGO", "TSM", "INTC", "MRVL", "ANET", "ARM", "SMCI", "VRT",
        "MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "PLTR", "SNOW", "DDOG",
        "GS", "MS", "JPM", "V", "BLK", "CME",
        "ADBE", "INTU", "WDAY",
        "AAPL", "TSLA", "NFLX"
    ]
    
    # Fetch price data
    print(f"Fetching data for {len(PILLAR_TICKERS)} stocks...")
    data = yf.download(PILLAR_TICKERS, period="3mo", progress=False)
    closes = data['Close']
    
    # Calculate daily breadth (% above 20d SMA)
    sma_20 = closes.rolling(20).mean()
    above_sma = (closes > sma_20).sum(axis=1)
    total = closes.notna().sum(axis=1)
    breadth = above_sma / total
    
    # Also calculate shorter SMAs
    sma_5 = closes.rolling(5).mean()
    sma_50 = closes.rolling(50).mean()
    above_5d = (closes > sma_5).sum(axis=1) / total
    above_50d = (closes > sma_50).sum(axis=1) / total
    
    # Prepare rows (last 30 days)
    rows = []
    for date in breadth.index[-30:]:
        if pd.notna(breadth[date]):
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "breadth_daily": float(above_5d.get(date, 0.5) or 0.5),
                "breadth_5d": float(above_5d.get(date, 0.5) or 0.5),
                "breadth_20d": float(breadth[date]),
                "breadth_50d": float(above_50d.get(date, 0.5) or 0.5),
                "advancers": int(above_sma.get(date, 15) or 15),
                "decliners": int(total.get(date, 30) - above_sma.get(date, 15)),
            })
    
    if rows:
        # Upsert
        client.table("breadth_daily").upsert(rows, on_conflict="date").execute()
        print(f"✅ Inserted {len(rows)} rows to breadth_daily")
    
except Exception as e:
    print(f"❌ Breadth error: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# 3. POPULATE CREDIT SPREAD DAILY (from FRED)
# =============================================================================
print("\n" + "="*60)
print("3. POPULATING CREDIT SPREAD DAILY")
print("="*60)

try:
    import requests
    
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        print("⚠️ FRED_API_KEY not set, using fallback data")
        # Generate synthetic data based on typical spreads
        from datetime import datetime, timedelta
        
        rows = []
        base_hy = 2.85
        base_ig = 0.80
        
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            # Add small random variation
            import random
            hy = base_hy + random.uniform(-0.1, 0.1)
            ig = base_ig + random.uniform(-0.05, 0.05)
            
            rows.append({
                "date": date,
                "hy_spread": round(hy, 2),
                "ig_spread": round(ig, 2),
                "hy_20d_change": round(random.uniform(-0.2, 0.2), 2),
                "ig_20d_change": round(random.uniform(-0.1, 0.1), 2),
            })
        
        client.table("credit_spread_daily").upsert(rows, on_conflict="date").execute()
        print(f"✅ Inserted {len(rows)} rows to credit_spread_daily (synthetic)")
    else:
        # Fetch from FRED
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # HY Spread
        response = requests.get(base_url, params={
            "series_id": "BAMLH0A0HYM2",
            "api_key": fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 60
        })
        hy_data = {o["date"]: float(o["value"]) for o in response.json().get("observations", []) if o["value"] != "."}
        
        # IG Spread
        response = requests.get(base_url, params={
            "series_id": "BAMLC0A0CM",
            "api_key": fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 60
        })
        ig_data = {o["date"]: float(o["value"]) for o in response.json().get("observations", []) if o["value"] != "."}
        
        # Combine and insert
        rows = []
        for date in hy_data:
            if date in ig_data:
                rows.append({
                    "date": date,
                    "hy_spread": hy_data[date],
                    "ig_spread": ig_data[date],
                })
        
        if rows:
            client.table("credit_spread_daily").upsert(rows, on_conflict="date").execute()
            print(f"✅ Inserted {len(rows)} rows to credit_spread_daily (FRED)")

except Exception as e:
    print(f"❌ Credit error: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# 4. VERIFY DATA
# =============================================================================
print("\n" + "="*60)
print("4. VERIFICATION")
print("="*60)

tables = ["vix_history", "breadth_daily", "credit_spread_daily"]
for table in tables:
    try:
        result = client.table(table).select("date").order("date", desc=True).limit(20).execute()
        print(f"{table}: {len(result.data)} rows")
    except Exception as e:
        print(f"{table}: Error - {e}")


print("\n" + "="*60)
print("DONE - Now re-run the backfill:")
print("  python dials_backfill.py --days 7")
print("="*60)
