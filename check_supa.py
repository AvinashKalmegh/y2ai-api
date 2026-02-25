from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
result = sb.table("dm_history").select("ticker", count="exact").eq("date", "2026-02-17").execute()
print(f"Feb 17 rows in dm_history: {result.count}")

required = ["APP","CCJ","DNN","TTD","MSTR","UEC","LEU","HAL","WDC","ENPH","UUUU","PDD","NCLH","FCX","RCL","LVS","PSKY","MRNA","WYNN","SMR"]
tickers = [r["ticker"] for r in result.data]
missing = [t for t in required if t not in tickers]
print(f"FlowOS missing: {missing if missing else 'None - all present'}")