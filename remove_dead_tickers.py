from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

DEAD = ["RUN", "DAY", "ARES"]

for ticker in DEAD:
    print(f"\n--- {ticker} ---")

    # Check scanner_universe
    r = sb.table("scanner_universe").select("ticker").eq("ticker", ticker).execute()
    if r.data:
        sb.table("scanner_universe").delete().eq("ticker", ticker).execute()
        print(f"  Removed from scanner_universe")
    else:
        print(f"  Not in scanner_universe (already clean)")

    # Check dm_latest
    r = sb.table("dm_latest").select("ticker").eq("ticker", ticker).execute()
    if r.data:
        sb.table("dm_latest").delete().eq("ticker", ticker).execute()
        print(f"  Removed from dm_latest")
    else:
        print(f"  Not in dm_latest")

    # Count in dm_history and price_history (report only, don't delete history)
    r = sb.table("dm_history").select("date", count="exact").eq("ticker", ticker).execute()
    print(f"  dm_history rows: {r.count} (kept for historical record)")

    r = sb.table("price_history").select("date", count="exact").eq("ticker", ticker).execute()
    print(f"  price_history rows: {r.count} (kept for historical record)")

print("\nDone. Dead tickers removed from active tables.")
print("Historical data preserved in dm_history and price_history.")