"""
Backfill SPOT price_history from IPO (April 2018) through Nov 4, 2025.
Existing data starts Nov 5, 2025. Uses Polygon API via pipeline functions.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from Dm_historical_pipeline import fetch_ticker_prices_polygon, get_supabase

supabase = get_supabase()

print("Fetching SPOT from Polygon: 2018-04-01 to 2025-11-04...")
rows = fetch_ticker_prices_polygon("SPOT", "2018-04-01", "2025-11-04")
print(f"Got {len(rows)} rows")

if rows:
    BATCH = 500
    uploaded = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i+BATCH]
        supabase.table("price_history").upsert(batch, on_conflict="ticker,date").execute()
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{len(rows)}")
    
    print(f"\nDone! {uploaded} rows uploaded to price_history.")
    print(f"Date range: {rows[0]['date']} to {rows[-1]['date']}")
    print(f"\nNext: run 'python Dm_historical_pipeline.py calculate --refresh-cache'")
    print(f"to compute DM for all SPOT rows.")