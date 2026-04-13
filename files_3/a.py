import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# -- CONFIG ------------------------------------------------------------------

LOOKBACK_DAYS = 30

TICKERS = {
    'T1_EQUITY':   ['TSLA', 'GOOGL', 'SATS'],
    'T2_HALO':     ['RKLB', 'ASTS', 'PL', 'TRMB', 'VSAT', 'KTOS', 'LHX'],
    'T3_THREAT':   ['T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'AMT', 'CCI', 'SBAC'],
    'T4_ARTEMIS':  ['BA', 'LMT', 'NOC', 'RTX', 'GD'],
    'T5_AI_SPACE': ['NVDA', 'MRVL', 'AMD'],
    'T6_BANKS':    ['MS', 'BAC', 'JPM', 'GS', 'C'],
}

ALL_TICKERS = [t for tier in TICKERS.values() for t in tier]

# -- SUPABASE ----------------------------------------------------------------

def _get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL/SUPABASE_KEY not set.")
    from supabase import create_client
    return create_client(url, key)

# -- FETCH DM HISTORY --------------------------------------------------------

def fetch_dm_history():
    client = _get_supabase()
    cutoff = (datetime.today() - timedelta(days=int(LOOKBACK_DAYS * 1.5))).strftime('%Y-%m-%d')

    rows = []
    offset, page = 0, 10000
    while True:
        r = (client.table('dm_history')
             .select('date,ticker,dm_smoothed,phase')
             .gte('date', cutoff)
             .in_('ticker', ALL_TICKERS)
             .order('date')
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page:
            break
        offset += page

    if not rows:
        raise RuntimeError("No DM history rows returned from Supabase.")

    df = pd.DataFrame(rows)
    df.rename(columns={'dm_smoothed': 'dm', 'ticker': 'ticker', 'date': 'date', 'phase': 'phase'}, inplace=True)
    df['date']   = pd.to_datetime(df['date'])
    df['dm']     = pd.to_numeric(df['dm'], errors='coerce').fillna(0)
    df['ticker'] = df['ticker'].str.strip().str.upper()
    df['phase']  = df['phase'].fillna('')

    return df

# -- ANALYSIS ----------------------------------------------------------------

def analyze(df):
    print('\n' + '='*70)
    print('  SPACEX IPO -- DM SIGNAL AUDIT')
    print(f'  {datetime.today().strftime("%B %d, %Y")} | Last {LOOKBACK_DAYS} sessions')
    print('='*70)

    tier_labels = {
        'T1_EQUITY':   'TIER 1 -- Direct SpaceX Equity Holders',
        'T2_HALO':     'TIER 2 -- Space Sector Re-Rating (Halo)',
        'T3_THREAT':   'TIER 3 -- Threatened Names (Spectrum + Starlink)',
        'T4_ARTEMIS':  'TIER 4 -- Artemis / NASA Ripple',
        'T5_AI_SPACE': 'TIER 5 -- AI Data Centers in Space',
        'T6_BANKS':    'TIER 6 -- Lead Underwriters',
    }

    for tier_key, tickers in TICKERS.items():
        print(f'\n  {tier_labels[tier_key]}')
        print(f'  {"-"*66}')
        print(f'  {"Ticker":<8} {"Latest DM":>10} {"Phase":<12} {"5d Chg":>8} {"10d Chg":>8} {"Trend"}')
        print(f'  {"-"*66}')

        for ticker in tickers:
            tdf = df[df['ticker'] == ticker].sort_values('date', ascending=False)
            if tdf.empty:
                print(f'  {ticker:<8} {"NO DATA":>10}')
                continue

            latest_dm    = tdf.iloc[0]['dm']
            latest_phase = tdf.iloc[0]['phase'] or '--'
            dm_5d_ago    = tdf.iloc[4]['dm'] if len(tdf) >= 5 else None
            dm_10d_ago   = tdf.iloc[9]['dm'] if len(tdf) >= 10 else None

            chg_5d  = latest_dm - dm_5d_ago  if dm_5d_ago  is not None else None
            chg_10d = latest_dm - dm_10d_ago if dm_10d_ago is not None else None

            trend = '* STRONG' if latest_dm >= 70 else '^ BUILD' if latest_dm >= 50 else '> WATCH' if latest_dm >= 30 else 'v WEAK'

            chg_5d_str  = f'{chg_5d:+.1f}'  if chg_5d  is not None else '--'
            chg_10d_str = f'{chg_10d:+.1f}' if chg_10d is not None else '--'

            print(f'  {ticker:<8} {latest_dm:>10.1f} {latest_phase:<12} {chg_5d_str:>8} {chg_10d_str:>8} {trend}')

    print('\n' + '='*70)

# -- MAIN --------------------------------------------------------------------

if __name__ == '__main__':
    print('Fetching DM history from Supabase...')
    df = fetch_dm_history()
    print(f'Loaded {len(df)} rows for {df["ticker"].nunique()} tickers.')
    analyze(df)
