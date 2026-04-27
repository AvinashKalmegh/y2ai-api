"""
Schema probe: sample 1 row from each table referenced by wave_r_validator
to learn the real column names. Throwaway script.
"""
import os
import json
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

CANDIDATES = [
    "hms_daily",
    "dm_history",
    "etf_flows",
    "etf_flows_history",
    "etf_netflows",
    "etf_reference",
    "etf_universe",
    "scanner_universe",
    "short_volume",
    "macro_history",
    "insider_transactions",
]

for table in CANDIDATES:
    try:
        r = sb.table(table).select("*").limit(1).execute()
        if r.data:
            cols = list(r.data[0].keys())
            print(f"\n[OK]  {table}")
            print(f"      columns ({len(cols)}): {cols}")
            print(f"      sample: {json.dumps(r.data[0], default=str)[:300]}")
        else:
            print(f"\n[OK]  {table}  (empty table)")
    except Exception as e:
        msg = str(e)
        if "PGRST205" in msg or "not find the table" in msg:
            print(f"\n[MISSING]  {table}")
        else:
            print(f"\n[ERROR]    {table}: {type(e).__name__}: {msg[:200]}")
