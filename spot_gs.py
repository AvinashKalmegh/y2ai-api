from dotenv import load_dotenv
load_dotenv()
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Fetch SPOT Feb 17 from dm_history
fields = "date,ticker,close,volume,return_20d,etf_return_20d,spy_return_20d,vol_20d_avg,vol_ratio,rel_str_etf,rel_str_spy,volume_z,dm_raw,dm_smoothed,etf,etf_dm,lead,phase"
result = sb.table("dm_history").select(fields).eq("ticker", "SPOT").eq("date", "2026-02-17").execute()

if not result.data:
    print("SPOT Feb 17 not found in dm_history")
    exit(1)

r = result.data[0]
keys = fields.split(",")
row = [str(r.get(k, "")) for k in keys]
print(f"SPOT Feb 17: DM={r['dm_smoothed']}, Phase={r['phase']}")

# Connect to GS staging
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key("1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA").worksheet("DM_2024_2026")

sheet.append_rows([row], value_input_option="USER_ENTERED")
print("Appended SPOT Feb 17 to DM_2024_2026 staging.")