# save as check_schema.py
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

r = client.table('bubble_index_daily').select('*').limit(1).execute()
if r.data:
    print("bubble_index_daily columns:", list(r.data[0].keys()))

r = client.table('amri_daily').select('*').limit(1).execute()
if r.data:
    print("amri_daily columns:", list(r.data[0].keys()))
else:
    print("amri_daily: table empty or doesn't exist")