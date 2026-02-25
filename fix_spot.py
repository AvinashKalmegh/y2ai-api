from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Step 1: Find SPOT's sector info from price_history neighbors
# SPOT is in XLC (Communication Services)
print("Adding SPOT to scanner_universe...")
sb.table("scanner_universe").upsert({
    "ticker": "SPOT",
    "sector": "Communication Services"
}, on_conflict="ticker").execute()
print("Done. SPOT added.")

# Verify
result = sb.table("scanner_universe").select("*").eq("ticker", "SPOT").execute()
print(f"Verified: {result.data}")

# Step 2: Check what dates SPOT is missing (Feb 14-17)
result = sb.table("price_history").select("date").eq("ticker", "SPOT").order("date", desc=True).limit(1).execute()
print(f"\nSPOT last price: {result.data[0]['date']}")

result = sb.table("dm_history").select("date").eq("ticker", "SPOT").order("date", desc=True).limit(1).execute()
print(f"SPOT last DM: {result.data[0]['date']}")

print("\nSPOT is now in the universe. Next daily pipeline run will:")
print("  1. Fetch SPOT prices for Feb 14-17")
print("  2. Calculate DM for those dates")
print("  3. Include SPOT in dm_latest")
print("\nRun: python3 Dm_historical_pipeline.py daily")