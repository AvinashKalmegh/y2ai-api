"""
Sync Google Sheets Price_History to Supabase
This replaces TwelveData prices with GS prices for exact alignment.
"""

import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def sync_prices(excel_path: str, batch_size: int = 500):
    """
    Sync prices from Google Sheets Excel export to Supabase.
    
    Args:
        excel_path: Path to Vikram-Develop-This.xlsx
        batch_size: Number of rows per upsert batch
    """
    print("Loading Google Sheets Price_History...")
    xlsx = pd.ExcelFile(excel_path)
    gs_prices = pd.read_excel(xlsx, sheet_name='Price_History')
    
    print(f"Loaded {len(gs_prices)} rows from GS")
    print(f"Date range: {gs_prices['Date'].min()} to {gs_prices['Date'].max()}")
    print(f"Tickers: {gs_prices['Ticker'].nunique()}")
    
    # Format for Supabase
    gs_prices['date'] = pd.to_datetime(gs_prices['Date']).dt.strftime('%Y-%m-%d')
    gs_prices['ticker'] = gs_prices['Ticker']
    gs_prices['open'] = gs_prices['Open']
    gs_prices['high'] = gs_prices['High']
    gs_prices['low'] = gs_prices['Low']
    gs_prices['close'] = gs_prices['Close']
    gs_prices['volume'] = gs_prices['Volume']
    
    # Select columns for Supabase
    upload_df = gs_prices[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Remove any rows with NaN close prices
    upload_df = upload_df.dropna(subset=['close'])
    
    print(f"\nPrepared {len(upload_df)} rows for upload")
    
    # Connect to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Upload in batches
    total_uploaded = 0
    
    for i in range(0, len(upload_df), batch_size):
        batch = upload_df.iloc[i:i+batch_size]
        records = batch.to_dict('records')
        
        # Upsert (insert or update on conflict)
        response = supabase.table("price_history").upsert(
            records,
            on_conflict="date,ticker"
        ).execute()
        
        total_uploaded += len(records)
        print(f"Uploaded {total_uploaded}/{len(upload_df)} rows...")
    
    print(f"\n✅ Sync complete! {total_uploaded} rows upserted to Supabase")
    return total_uploaded

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    else:
        excel_path = "Vikram-Develop-This.xlsx"
    
    sync_prices(excel_path)