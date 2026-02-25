from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Check if SPOT is in scanner_universe
result = sb.table("scanner_universe").select("*").eq("ticker", "SPOT").execute()
print("SPOT in scanner_universe:", result.data if result.data else "NOT FOUND")

# Check SPOT price history
result = sb.table("price_history").select("date", count="exact").eq("ticker", "SPOT").execute()
print(f"SPOT price rows: {result.count}")
result = sb.table("price_history").select("date").eq("ticker", "SPOT").order("date", desc=True).limit(3).execute()
print(f"SPOT latest prices: {[r['date'] for r in result.data]}")

# Check SPOT dm_history
result = sb.table("dm_history").select("date").eq("ticker", "SPOT").order("date", desc=True).limit(3).execute()
print(f"SPOT latest DM dates: {[r['date'] for r in result.data]}")

# Check dead tickers
print()
for t in ["RUN", "DAY", "ARES"]:
    r = sb.table("scanner_universe").select("ticker").eq("ticker", t).execute()
    p = sb.table("price_history").select("date").eq("ticker", t).order("date", desc=True).limit(1).execute()
    last = p.data[0]["date"] if p.data else "none"
    print(f"{t}: in universe={bool(r.data)}, last price={last}")