"""
Backfill price gaps for SPOT and STZ from Polygon API.

SPOT: Missing ~8 days around Feb 2-10, 2026
STZ:  285-day gap from Sep 2019 to Oct 2020

Usage: python dm_backfill_gaps.py
"""
import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")

if not POLYGON_API_KEY:
    print("ERROR: No POLYGON_API_KEY found in .env")
    exit(1)

GAPS = [
    {"ticker": "SPOT", "start": "2026-01-25", "end": "2026-02-14", "reason": "Pipeline gap, ~8 days missing"},
    {"ticker": "STZ",  "start": "2019-09-01", "end": "2020-11-01", "reason": "285-day historical gap"},
]

for gap in GAPS:
    ticker = gap["ticker"]
    start = gap["start"]
    end = gap["end"]
    print(f"\n{'='*50}")
    print(f"{ticker}: {gap['reason']}")
    print(f"Fetching {start} to {end} from Polygon...")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}

    try:
        resp = requests.get(url, params=params)
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            print(f"  No data returned from Polygon")
            continue

        rows = []
        for r in data.get("results", []):
            try:
                date = datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "open": float(r["o"]),
                    "high": float(r["h"]),
                    "low": float(r["l"]),
                    "close": float(r["c"]),
                    "volume": int(r["v"])
                })
            except (ValueError, KeyError):
                continue

        print(f"  Got {len(rows)} rows ({rows[0]['date']} to {rows[-1]['date']})")

        # Batch upsert
        uploaded = 0
        BATCH = 500
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            supabase.table("price_history").upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            uploaded += len(batch)
            time.sleep(0.2)

        print(f"  Uploaded {uploaded} rows to price_history")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*50}")
print("Done! Gaps filled.")
print("DM for these dates will be correct on next calculate run.")
print("If you already ran calculate --refresh-cache, run it again to pick up these new prices.")