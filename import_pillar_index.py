import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# UPDATE THIS PATH to where you saved the Excel file
EXCEL_PATH = r"C:\Users\avina\OneDrive\Desktop\y2ai\Vikram-Develop-This.xlsx"

# Read the Pillar_Index sheet
df = pd.read_excel(EXCEL_PATH, sheet_name='Pillar_Index')

# Map columns
column_mapping = {
    'Date': 'date',
    'Infrastructure': 'infra_return',
    'Enterprise': 'enterprise_return', 
    'Macro': 'macro_return',
    'Financial': 'financial_return',
    'Productivity': 'productivity_return',
    'Infra_Index': 'infra_index',
    'EntAdopt_Index': 'enterprise_index',
    'Macro_Index': 'macro_index',
    'FinMkt_Index': 'financial_index',
    'ProdLabor_Index': 'productivity_index',
    'Demand_Index': 'demand_index',
    'Infra_5D': 'infra_5d',
    'EntAdopt_5D': 'enterprise_5d',
    'Macro_5D': 'macro_5d',
    'FinMkt_5D': 'financial_5d',
    'ProdLabor_5D': 'productivity_5d',
    'Demand_5D': 'demand_5d',
    'Infra_1M': 'infra_1m',
    'EntAdopt_1M': 'enterprise_1m',
    'Macro_1M': 'macro_1m',
    'FinMkt_1M': 'financial_1m',
    'ProdLabor_1M': 'productivity_1m',
    'Demand_1M': 'demand_1m',
    'Infra_3M': 'infra_3m',
    'EntAdopt_3M': 'enterprise_3m',
    'Macro_3M': 'macro_3m',
    'FinMkt_3M': 'financial_3m',
    'ProdLabor_3M': 'productivity_3m',
    'Demand_3M': 'demand_3m',
    'Infra_6M': 'infra_6m',
    'EntAdopt_6M': 'enterprise_6m',
    'Macro_6M': 'macro_6m',
    'FinMkt_6M': 'financial_6m',
    'ProdLabor_6M': 'productivity_6m',
    'Demand_6M': 'demand_6m',
    'Infra_YTD': 'infra_ytd',
    'EntAdopt_YTD': 'enterprise_ytd',
    'Macro_YTD': 'macro_ytd',
    'FinMkt_YTD': 'financial_ytd',
    'ProdLabor_YTD': 'productivity_ytd',
    'Demand_YTD': 'demand_ytd',
}

# Select columns
cols_to_use = list(column_mapping.keys())
df_clean = df[cols_to_use].copy()
df_clean = df_clean.rename(columns=column_mapping)

# Add demand_return from column 6 (the misnamed "Updated" column)
df_clean['demand_return'] = df.iloc[:, 6]

# Convert date to string
df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')

# Replace NaN with None
df_clean = df_clean.where(pd.notnull(df_clean), None)

# Convert to records
records = df_clean.to_dict('records')

print(f'Total records to insert: {len(records)}')
print(f'Sample record (Jan 15):')
jan15 = [r for r in records if r['date'] == '2026-01-15'][0]
print(f"  infra_5d: {jan15['infra_5d']*100:.2f}%")
print(f"  enterprise_5d: {jan15['enterprise_5d']*100:.2f}%")
print(f"  productivity_5d: {jan15['productivity_5d']*100:.2f}%")
print(f"  demand_5d: {jan15['demand_5d']*100:.2f}%")

# Insert in batches
batch_size = 500
for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    result = supabase.table('pillar_index_daily').insert(batch).execute()
    print(f'Inserted batch {i//batch_size + 1}: {len(batch)} records')

print('Done! Pillar_Index synced from GS.')