import os
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

for tbl in ['dm_history', 'dm_daily', 'dm_signals']:
    try:
        r = sb.table(tbl).select('*').limit(1).execute()
        if r.data:
            print(f"{tbl}: columns = {list(r.data[0].keys())}")
        else:
            print(f"{tbl}: empty")
    except Exception as e:
        print(f"{tbl}: ERROR — {str(e)[:120]}")

# Sample QCOM row from whichever table exists
for tbl in ['dm_history', 'dm_daily']:
    try:
        r = sb.table(tbl).select('*').eq('ticker', 'QCOM').order('date', desc=True).limit(1).execute()
        if r.data:
            print(f"\n{tbl} — latest QCOM row:")
            for k, v in r.data[0].items():
                print(f"  {k}: {v}")
            break
    except Exception:
        continue
