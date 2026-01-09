"""
CHECK BUBBLE INDEX DATA
=======================
Diagnose why dashboard shows 100.0 instead of 55.9
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
client = create_client(url, key)

print("=" * 70)
print("BUBBLE INDEX DIAGNOSTIC")
print("=" * 70)

# Get all columns from bubble_index_daily
print("\n1. ALL COLUMNS IN bubble_index_daily (latest row):")
print("-" * 50)
r = client.table('bubble_index_daily').select('*').order('date', desc=True).limit(3).execute()
if r.data:
    for i, row in enumerate(r.data):
        print(f"\nRow {i+1} (date: {row.get('date')}):")
        for k, v in sorted(row.items()):
            if v is not None:
                print(f"  {k}: {v}")
else:
    print("No data found")

# Check what column dashboard.py uses
print("\n" + "=" * 70)
print("2. POTENTIAL COLUMN NAMES FOR BUBBLE INDEX:")
print("-" * 50)
print("""
The dashboard.py likely reads from one of these columns:
  - bubble_index
  - bubble_score  
  - score
  - value
  - index_value

If run_bubble_amri.py saved to 'bubble_index' but dashboard reads 
'bubble_score', they'll show different values.
""")

# Check recent dates
print("=" * 70)
print("3. RECENT BUBBLE INDEX RECORDS (by date):")
print("-" * 50)
r = client.table('bubble_index_daily').select('date, bubble_index, regime').order('date', desc=True).limit(10).execute()
if r.data:
    print(f"\n{'Date':<15} {'Bubble Index':<15} {'Regime':<15}")
    print("-" * 45)
    for row in r.data:
        bi = row.get('bubble_index', 'N/A')
        regime = row.get('regime', 'N/A')
        print(f"  {row['date']:<13} {bi:<15} {regime:<15}")

# Solution
print("\n" + "=" * 70)
print("4. SOLUTION:")
print("-" * 50)
print("""
If dashboard shows 100.0 but the record shows 55.9, check:

1. Does dashboard.py read 'bubble_index' or a different column?
   Look in dials/dashboard.py for the bubble query

2. Is there a date mismatch? Dashboard might fetch latest record
   which could be a different date than what run_bubble_amri.py saved

3. Is there cached data? Try running:
   python dials_backfill.py --today
   
   Then refresh the dashboard.
""")