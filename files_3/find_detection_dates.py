"""
find_detection_dates.py
--------------------------------------------------------------------------------
For each of the 9 PureSim positions opened April 1, 2026, finds the most recent
date where DM EMA5 crossed from below 65 to above 65 (current accumulation cycle
start). This is the signal detection date -- the gap between detection and entry
is the lead time used in investor presentations.

GSAT is priority -- Amazon acquisition news broke after hours April 1.
We entered GSAT at $66.42 that morning on signal alone. The detection date
tells us how far in advance the signal was reading the accumulation.

Output: detection_dates.csv + console table
--------------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
from datetime import date
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# -- CONFIG --------------------------------------------------------------------

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

TICKERS    = ['ARM', 'DOCN', 'GSAT', 'DELL', 'MRVL', 'RUN', 'AMD', 'ALB', 'HAL']
ENTRY_DATE = date(2026, 4, 1)
EMA_PERIOD = 5
DM_THRESHOLD = 65.0

# Notes for specific tickers
NOTES = {
    'GSAT': '* Amazon acquisition news after hours April 1 -- entered at $66.42, closed at $85'
}


# -- HELPERS -------------------------------------------------------------------

def compute_ema(values: pd.Series, period: int) -> pd.Series:
    """Exponential moving average -- same formula as GAS/FlowOS."""
    k = 2 / (period + 1)
    result = []
    for i, v in enumerate(values):
        if i == 0:
            result.append(v)
        else:
            result.append(v * k + result[-1] * (1 - k))
    return pd.Series(result, index=values.index)


def find_last_crossover(df: pd.DataFrame) -> tuple:
    """
    Find the most recent date where DM EMA5 crossed from BELOW 65 to ABOVE 65.
    Returns (detection_date, dm_at_detection) or (None, None) if not found.
    """
    df = df[df['date'] < str(ENTRY_DATE)].copy()
    if len(df) < EMA_PERIOD + 1:
        return None, None

    df = df.sort_values('date').reset_index(drop=True)
    ema = compute_ema(df['dm_raw'], EMA_PERIOD)

    detection_date = None
    detection_dm   = None

    for i in range(1, len(df)):
        # Crossover: previous day below threshold, current day above
        if ema.iloc[i] >= DM_THRESHOLD and ema.iloc[i-1] < DM_THRESHOLD:
            detection_date = df['date'].iloc[i]
            detection_dm   = ema.iloc[i]
            # Keep overwriting -- we want the LAST crossover before entry

    return detection_date, detection_dm


# -- MAIN ----------------------------------------------------------------------

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError('Set SUPABASE_URL and SUPABASE_KEY environment variables.')

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print('Loading DM history from Supabase...')
    print(f'Tickers: {", ".join(TICKERS)}')
    print()

    results = []

    for ticker in TICKERS:
        print(f'  Processing {ticker}...', end=' ')

        # Fetch all rows for this ticker before entry date
        response = (
            client.table('dm_history')
            .select('date, dm_smoothed')
            .eq('ticker', ticker)
            .lt('date', str(ENTRY_DATE))
            .order('date', desc=False)
            .execute()
        )

        data = response.data
        if not data:
            print('NO DATA')
            results.append({
                'ticker':         ticker,
                'detection_date': None,
                'dm_at_detection': None,
                'entry_date':     str(ENTRY_DATE),
                'lead_days':      None,
                'note':           'No data found'
            })
            continue

        df = pd.DataFrame(data)
        df['date']   = pd.to_datetime(df['date']).dt.date.astype(str)
        df['dm_raw'] = pd.to_numeric(df['dm_smoothed'], errors='coerce').fillna(0)

        detection_date, detection_dm = find_last_crossover(df)

        if detection_date is None:
            print('NO CROSSOVER FOUND')
            results.append({
                'ticker':          ticker,
                'detection_date':  None,
                'dm_at_detection': None,
                'entry_date':      str(ENTRY_DATE),
                'lead_days':       None,
                'note':            'No crossover above 65 found before entry'
            })
            continue

        lead_days = (ENTRY_DATE - date.fromisoformat(str(detection_date))).days
        note      = NOTES.get(ticker, '')
        print(f'detected {detection_date} ({lead_days} days lead)')

        results.append({
            'ticker':          ticker,
            'detection_date':  str(detection_date),
            'dm_at_detection': round(float(detection_dm), 1),
            'entry_date':      str(ENTRY_DATE),
            'lead_days':       lead_days,
            'note':            note
        })

    # -- OUTPUT ----------------------------------------------------------------

    df_out = pd.DataFrame(results)

    print()
    print('=' * 80)
    print('SIGNAL DETECTION DATES -- Current Accumulation Cycle')
    print('Detection = most recent DM EMA5 crossover from below 65 to above 65')
    print(f'Entry date for all: {ENTRY_DATE}')
    print('=' * 80)
    print(f'{"Ticker":<6} | {"Detection Date":<15} | {"DM":<6} | {"Lead Days":<10} | Note')
    print(f'{"------":<6}-+-{"---------------":<15}-+-{"------":<6}-+-{"----------":<10}-+-{"------------------------------------------"}')

    for r in results:
        dm_str   = f'{r["dm_at_detection"]}' if r['dm_at_detection'] else '--'
        lead_str = f'{r["lead_days"]} days'  if r['lead_days']       else '--'
        det_str  = r['detection_date']        if r['detection_date']  else '--'
        note     = r['note']
        print(f'{r["ticker"]:<6} | {det_str:<15} | {dm_str:<6} | {lead_str:<10} | {note}')

    print('=' * 80)
    print()

    # Save CSV
    out_path = 'detection_dates.csv'
    df_out.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
