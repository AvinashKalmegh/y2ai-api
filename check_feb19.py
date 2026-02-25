from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Check dm_history for Feb 19
r = sb.table("dm_history").select("ticker", count="exact").eq("date", "2026-02-19").execute()
print(f"dm_history Feb 19 rows: {r.count}")

# Check dm_latest date
r = sb.table("dm_latest").select("date,ticker", count="exact").execute()
print(f"dm_latest total rows: {r.count}")

# Check what date dm_latest has
r = sb.table("dm_latest").select("date").order("date", desc=True).limit(1).execute()
print(f"dm_latest latest date: {r.data[0]['date'] if r.data else 'empty'}")

# Check SPOT specifically
r = sb.table("dm_latest").select("date,dm_smoothed,phase").eq("ticker", "SPOT").execute()
if r.data:
    print(f"SPOT in dm_latest: date={r.data[0]['date']}, DM={r.data[0]['dm_smoothed']}, Phase={r.data[0]['phase']}")
else:
    print("SPOT not in dm_latest")