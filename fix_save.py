"""
Direct database update - bypasses schema mismatch
"""
from supabase import create_client
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
today = datetime.now().strftime('%Y-%m-%d')

# 1. Update bubble_index_daily with correct value
print("Updating bubble_index_daily...")
try:
    client.table('bubble_index_daily').upsert({
        'date': today,
        'bubble_index': 55.5,
        'regime': 'Caution',
    }, on_conflict='date').execute()
    print("✅ Bubble Index updated to 55.5")
except Exception as e:
    print(f"❌ Error: {e}")

# 2. Insert into amri_daily
print("\nUpdating amri_daily...")
try:
    client.table('amri_daily').upsert({
        'date': today,
        'amri_score': 53.6,
        'regime': 'Elevated',
    }, on_conflict='date').execute()
    print("✅ AMRI updated to 53.6")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. Verify
print("\nVerifying...")
r = client.table('bubble_index_daily').select('date, bubble_index, regime').order('date', desc=True).limit(1).execute()
print(f"Bubble: {r.data[0] if r.data else 'No data'}")

r = client.table('amri_daily').select('date, amri_score, regime').order('date', desc=True).limit(1).execute()
print(f"AMRI: {r.data[0] if r.data else 'No data'}")