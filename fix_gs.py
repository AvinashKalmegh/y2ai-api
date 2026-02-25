from dotenv import load_dotenv
load_dotenv()
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Fetch all Feb 17 rows from Supabase
fields = "date,ticker,close,volume,return_20d,etf_return_20d,spy_return_20d,vol_20d_avg,vol_ratio,rel_str_etf,rel_str_spy,volume_z,dm_raw,dm_smoothed,etf,etf_dm,lead,phase"
all_rows = []
offset = 0
while True:
    result = sb.table("dm_history").select(fields).eq("date", "2026-02-17").order("ticker").range(offset, offset+999).execute()
    if not result.data:
        break
    all_rows.extend(result.data)
    if len(result.data) < 1000:
        break
    offset += 1000

print(f"Fetched {len(all_rows)} rows for Feb 17 from Supabase")

# Connect to GS staging
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key("1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA").worksheet("DM_2024_2026")

# Find and delete existing Feb 17 rows
all_data = sheet.get_all_values()
header = all_data[0]
keep = [header]
removed = 0
for row in all_data[1:]:
    if row[0] == "2026-02-17":
        removed += 1
    else:
        keep.append(row)
print(f"Removing {removed} old Feb 17 rows from GS")

# Build new Feb 17 rows
keys = fields.split(",")
new_rows = []
for r in all_rows:
    new_rows.append([str(r.get(k, "")) for k in keys])

# Rewrite sheet
final = keep + new_rows
sheet.clear()
sheet.update(range_name="A1", values=final, value_input_option="USER_ENTERED")
print(f"Done. Wrote {len(final)-1} total rows ({len(new_rows)} for Feb 17)")