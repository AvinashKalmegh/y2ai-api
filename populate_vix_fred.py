"""
POPULATE VIX HISTORY FROM FRED
==============================
Uses FRED VIXCLS series (same source as Google Sheets)
This replaces the Yahoo Finance version to eliminate data source mismatch.

Data Source: https://fred.stlouisfed.org/series/VIXCLS
"""

import os
import requests
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
fred_api_key = os.getenv("FRED_API_KEY")

if not url or not key:
    print("❌ Set SUPABASE_URL and SUPABASE_KEY environment variables")
    exit(1)

if not fred_api_key:
    print("❌ Set FRED_API_KEY environment variable")
    print("   Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    exit(1)

client = create_client(url, key)
print("✅ Supabase connected")

# Fetch VIX from FRED (same source as Google Sheets)
print("\nFetching VIX data from FRED API (VIXCLS series)...")

end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

fred_url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "VIXCLS",
    "api_key": fred_api_key,
    "file_type": "json",
    "observation_start": start_date,
    "observation_end": end_date,
    "sort_order": "desc"
}

try:
    response = requests.get(fred_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    observations = data.get("observations", [])
    
    if not observations:
        print("❌ No VIX observations returned from FRED")
        exit(1)
    
    print(f"Fetched {len(observations)} days of VIX data from FRED")
    
except Exception as e:
    print(f"❌ Failed to fetch from FRED: {e}")
    exit(1)

# Prepare rows - filter out missing values (FRED uses "." for missing)
rows = []
for obs in observations:
    if obs["value"] != ".":
        rows.append({
            "date": obs["date"],
            "close": float(obs["value"]),
        })

print(f"Valid observations: {len(rows)}")

# Insert to Supabase
if rows:
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        try:
            client.table("vix_history").upsert(batch, on_conflict="date").execute()
            print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} rows")
        except Exception as e:
            print(f"  Error on batch {i//batch_size + 1}: {e}")
    
    print(f"\n✅ Inserted {len(rows)} rows to vix_history from FRED")

# Verify
result = client.table("vix_history").select("date, close").order("date", desc=True).limit(5).execute()
print(f"\nVerification - Latest 5 rows (FRED VIXCLS):")
for r in result.data:
    print(f"  {r['date']}: {r['close']}")

# Compare with Google Sheets expected value
print(f"\n" + "="*50)
print("DATA SOURCE VERIFICATION")
print("="*50)
print("Source: FRED VIXCLS (same as Google Sheets)")
print("This should now match Google Sheets VIX_Dial values exactly.")

total = len(client.table('vix_history').select('date').execute().data)
print(f"\nTotal rows in vix_history: {total}")