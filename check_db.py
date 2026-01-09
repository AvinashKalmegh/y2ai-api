from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Check bubble_index_daily
r = client.table('bubble_index_daily').select('*').order('date', desc=True).limit(1).execute()
print('BUBBLE:', r.data[0] if r.data else 'No data')

# Check amri_daily  
r = client.table('amri_daily').select('*').order('date', desc=True).limit(1).execute()
print('AMRI:', r.data[0] if r.data else 'No data')