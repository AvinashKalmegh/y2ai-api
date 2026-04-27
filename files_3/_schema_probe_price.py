import os, json
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

for t in ["price_history", "price_daily", "prices"]:
    try:
        r = sb.table(t).select("*").limit(1).execute()
        if r.data:
            print(f"[OK] {t}: {list(r.data[0].keys())}")
            print(f"     sample: {json.dumps(r.data[0], default=str)[:250]}")
        else:
            print(f"[OK] {t} (empty)")
    except Exception as e:
        msg = str(e)
        marker = "MISSING" if "PGRST205" in msg or "not find the table" in msg else "ERROR"
        print(f"[{marker}] {t}: {msg[:120]}")
