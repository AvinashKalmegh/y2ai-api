"""
One-off backfill for RUN (Sunrun).
- Inserts RUN into scanner_universe (sector=Semiconductors, ETF=SMH per peer convention)
- Fetches missing price rows (gap since last price_history date)
- Recomputes DM for full series and upserts dm_history
- Updates dm_latest
"""
import os
import sys
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from dm_newtickers_backfill import (
    supabase, fetch_prices, upload_prices,
    load_prices_from_supabase, calculate_dm,
    upload_dm_history, update_dm_latest,
    SECTOR_ETF_MAP,
)

TICKER = 'RUN'
SECTOR = 'Semiconductors'
ETF    = SECTOR_ETF_MAP.get(SECTOR, 'SPY')

print('='*60)
print(f'Backfilling {TICKER}')
print(f'  Sector: {SECTOR}')
print(f'  ETF:    {ETF}')
print('='*60)

# Step 1: scanner_universe insert (upsert)
print('\n[1/5] Ensuring scanner_universe has RUN...')
supabase.table('scanner_universe').upsert(
    {'ticker': TICKER, 'sector': SECTOR},
    on_conflict='ticker'
).execute()
r = supabase.table('scanner_universe').select('*').eq('ticker', TICKER).execute()
print(f'  scanner_universe: {r.data[0]}')

# Step 2 + 3: Fetch FULL price history from Twelve Data and upsert.
# Why full: existing RUN price_history rows have many NaN volumes, which break
# DM calc. Re-fetching the full series fills volumes consistently.
print(f'\n[2-3/5] Fetching FULL price history from Twelve Data (2016 to today)...')
today_str = date.today().strftime('%Y-%m-%d')
rows = fetch_prices(TICKER, start_date='2016-01-01', end_date=today_str)
if rows:
    upload_prices(rows)
    print(f'  Uploaded {len(rows)} rows to price_history (upsert on ticker,date)')
else:
    print(f'  ERROR: No rows returned from Twelve Data. Aborting.')
    sys.exit(1)

# Step 4: Load full price history, compute DM, upsert dm_history
print(f'\n[4/5] Computing DM over full RUN history...')
ticker_df = load_prices_from_supabase(TICKER)
spy_df = load_prices_from_supabase('SPY')
etf_df = load_prices_from_supabase(ETF)

if ticker_df.empty:
    print(f'  ERROR: No price_history rows for {TICKER}. Aborting.')
    sys.exit(1)
if spy_df.empty:
    print(f'  ERROR: No price_history rows for SPY. Aborting.')
    sys.exit(1)
if etf_df.empty:
    print(f'  ERROR: No price_history rows for {ETF}. Aborting.')
    sys.exit(1)

ticker_df['ticker'] = TICKER  # calculate_dm reads this column
dm_df = calculate_dm(ticker_df, spy_df, etf_df, ETF)
print(f'  Computed DM for {len(dm_df)} rows')

uploaded = upload_dm_history(dm_df)
print(f'  Upserted {uploaded} rows to dm_history')

# Step 5: dm_latest
print(f'\n[5/5] Updating dm_latest...')
update_dm_latest(TICKER, dm_df)
print(f'  dm_latest updated')

# Verify
print('\n' + '='*60)
print('VERIFICATION')
print('='*60)
r = supabase.table('dm_history').select('date,close,dm_smoothed,phase') \
    .eq('ticker', TICKER).order('date', desc=True).limit(5).execute()
print('dm_history (last 5 RUN rows):')
for row in r.data:
    print(f'  {row}')
r = supabase.table('dm_latest').select('*').eq('ticker', TICKER).execute()
print('\ndm_latest RUN row:')
if r.data:
    print(f'  {r.data[0]}')
else:
    print('  NOT FOUND')
