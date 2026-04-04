"""
STRATEGY 5 HIT RATE ANALYSIS — Supabase Version
=================================================
Run this to compare against the GAS version (runPureSimHitRateAnalysis.gs)
which reads from Google Sheets DM history.

If both produce identical results, GS and Supabase data are confirmed consistent.
If different, investigate the discrepancy before building the V4_Universe selection.

Requirements:
    pip install supabase pandas numpy python-dotenv

Environment (.env):
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-service-role-key

Run:
    python strategy5_hitrate_supabase.py

Author: ARGUS Research — April 1, 2026
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase pandas numpy python-dotenv")
    exit(1)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Tickers to evaluate — must match GAS script exactly for comparison
TICKERS = [
    # PureSim 8 — primary question
    'AMD', 'ARM', 'DLR', 'EQIX', 'INTU', 'KLAC', 'MRVL', 'VRT',
    # Top Strategy 5 performers — comparison / sanity check
    'APP', 'META', 'PLTR', 'VST', 'TEAM', 'CRWD', 'NVDA', 'TSLA',
    # Tradeable 26 sample — comparison
    'DNN', 'CEG', 'MU', 'UUUU', 'LEU'
]

# Strategy 5 parameters — must match GAS script exactly
ENTRY_DM_MIN  = 65     # DM EMA5 must be >= this
EXIT_MA20     = 50     # 20d MA of DM drops below this = exit
HIT_RETURN    = 0.10   # 10% return = hit
FORWARD_DAYS  = 90     # trading days to measure return
MIN_DATA_PTS  = 25     # minimum rows required to evaluate

# Supabase table — adjust if column names differ
DM_TABLE      = 'dm_history'
DATE_COL      = 'date'
TICKER_COL    = 'ticker'
CLOSE_COL     = 'close'
DM_COL        = 'dm_smoothed'          # EMA5 smoothed DM score

# ── SUPABASE CONNECTION ───────────────────────────────────────────────────────

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


# ── DATA LOADER ───────────────────────────────────────────────────────────────

def load_dm_data(client, tickers):
    """Load all DM history for the specified tickers from Supabase."""
    print(f"Loading DM data for {len(tickers)} tickers from Supabase...")

    all_rows = []
    batch_size = 5  # load a few tickers at a time to avoid timeout

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        offset = 0
        page   = 10000

        while True:
            r = (client.table(DM_TABLE)
                 .select(f"{DATE_COL},{TICKER_COL},{CLOSE_COL},{DM_COL}")
                 .in_(TICKER_COL, batch)
                 .order(DATE_COL)
                 .range(offset, offset + page - 1)
                 .execute())

            all_rows.extend(r.data)
            if len(r.data) < page:
                break
            offset += page

        print(f"  Loaded batch {i//batch_size + 1}: {batch}")

    if not all_rows:
        print("ERROR: No data returned from Supabase")
        return {}

    df = pd.DataFrame(all_rows)
    df.rename(columns={
        DATE_COL:   'date',
        TICKER_COL: 'ticker',
        CLOSE_COL:  'close',
        DM_COL:     'dm',
    }, inplace=True)

    df['date']   = pd.to_datetime(df['date']).dt.date
    df['close']  = pd.to_numeric(df['close'], errors='coerce')
    df['dm']     = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date','ticker','close','dm'])
    df = df.sort_values(['ticker','date']).reset_index(drop=True)

    print(f"  Total rows loaded: {len(df):,}")
    print(f"  Unique tickers:    {df['ticker'].nunique()}")
    print(f"  Date range:        {df['date'].min()} to {df['date'].max()}")

    # Group by ticker
    by_ticker = {}
    for ticker, group in df.groupby('ticker'):
        by_ticker[ticker] = group.reset_index(drop=True)

    return by_ticker


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────

def compute_hit_rate(rows_df):
    """
    Apply Strategy 5 conditions to a single ticker's DM history.
    Returns: signals, hits, hit_rate, avg_return, best_return, worst_return
    """
    n = len(rows_df)
    if n < MIN_DATA_PTS:
        return 0, 0, 0.0, 0.0, 0.0, 0.0

    close  = rows_df['close'].values
    dm     = rows_df['dm'].values

    # Rolling indicators
    dm5     = pd.Series(dm).rolling(5,  min_periods=1).mean().values
    dm5prev = pd.Series(dm5).shift(1).fillna(0).values
    dm20    = pd.Series(dm).rolling(20, min_periods=1).mean().values

    signals    = 0
    hits       = 0
    total_ret  = 0.0
    best_ret   = -999.0
    worst_ret  =  999.0
    i          = 1

    while i < n:
        # Entry conditions
        if dm[i] < ENTRY_DM_MIN:
            i += 1
            continue
        if dm5[i] <= dm5prev[i]:
            i += 1
            continue
        if dm20[i] < EXIT_MA20:
            i += 1
            continue

        signals += 1
        entry_close = close[i]
        exit_return = 0.0

        # Find exit
        for j in range(i + 1, min(i + FORWARD_DAYS + 1, n)):
            exit_return = (close[j] - entry_close) / entry_close
            if dm20[j] < EXIT_MA20:
                break
            if j == min(i + FORWARD_DAYS, n - 1):
                break

        total_ret += exit_return
        if exit_return >= HIT_RETURN:
            hits += 1
        if exit_return > best_ret:
            best_ret  = exit_return
        if exit_return < worst_ret:
            worst_ret = exit_return

        # Skip ahead 20 days to avoid overlapping signals
        i += 20

    hit_rate   = (hits / signals * 100) if signals > 0 else 0.0
    avg_return = (total_ret / signals * 100) if signals > 0 else 0.0

    return signals, hits, round(hit_rate,1), round(avg_return,1), \
           round(best_ret*100,1) if best_ret > -999 else 0.0, \
           round(worst_ret*100,1) if worst_ret < 999 else 0.0


# ── VERDICT ───────────────────────────────────────────────────────────────────

def get_verdict(signals, hit_rate):
    if signals == 0:    return 'NO SIGNALS'
    if signals < 3:     return 'THIN (<3 signals)'
    if hit_rate >= 65:  return 'STRONG — KEEP'
    if hit_rate >= 50:  return 'QUALIFIED — KEEP'
    if hit_rate >= 40:  return 'BORDERLINE — REVIEW'
    return 'BELOW THRESHOLD — REMOVE'


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 62)
    print('STRATEGY 5 HIT RATE ANALYSIS — SUPABASE VERSION')
    print(f'Entry: DM >= {ENTRY_DM_MIN} + 5d MA rising')
    print(f'Exit:  20d MA < {EXIT_MA20}')
    print(f'Hit:   {int(HIT_RETURN*100)}% return within {FORWARD_DAYS} days')
    print('=' * 62)
    print()

    client   = get_client()
    by_ticker = load_dm_data(client, TICKERS)
    print()

    results = []
    for ticker in TICKERS:
        if ticker not in by_ticker:
            results.append({
                'ticker': ticker, 'data_pts': 0, 'signals': 0, 'hits': 0,
                'hit_rate': 0.0, 'avg_return': 0.0,
                'best_return': 0.0, 'worst_return': 0.0,
                'first_date': 'N/A', 'last_date': 'N/A',
                'verdict': 'NOT IN SUPABASE'
            })
            continue

        df         = by_ticker[ticker]
        signals, hits, hit_rate, avg_return, best_ret, worst_ret = compute_hit_rate(df)

        results.append({
            'ticker':       ticker,
            'data_pts':     len(df),
            'signals':      signals,
            'hits':         hits,
            'hit_rate':     hit_rate,
            'avg_return':   avg_return,
            'best_return':  best_ret,
            'worst_return': worst_ret,
            'first_date':   str(df['date'].min()),
            'last_date':    str(df['date'].max()),
            'verdict':      get_verdict(signals, hit_rate)
        })

    # ── Print results ──────────────────────────────────────────────────────
    puresim8   = ['AMD','ARM','DLR','EQIX','INTU','KLAC','MRVL','VRT']
    comparison = ['APP','META','PLTR','VST','TEAM','CRWD','NVDA','TSLA',
                  'DNN','CEG','MU','UUUU','LEU']

    print('=' * 62)
    print('RESULTS — PURESIM 8')
    print('=' * 62)
    print(f"{'Ticker':<6} | {'DtPts':>5} | {'Sigs':>4} | {'Hits':>4} | {'Hit%':>4} | {'AvgRet':>6} | {'Best':>5} | {'Worst':>5} | Verdict")
    print('-' * 95)
    for r in [x for x in results if x['ticker'] in puresim8]:
        print(f"{r['ticker']:<6} | {r['data_pts']:>5} | {r['signals']:>4} | "
              f"{r['hits']:>4} | {r['hit_rate']:>4.0f}% | {r['avg_return']:>5.1f}% | "
              f"{r['best_return']:>4.1f}% | {r['worst_return']:>4.1f}% | {r['verdict']}")

    print()
    print('=' * 62)
    print('RESULTS — COMPARISON TICKERS')
    print('=' * 62)
    print(f"{'Ticker':<6} | {'DtPts':>5} | {'Sigs':>4} | {'Hit%':>4} | {'AvgRet':>6} | Verdict")
    print('-' * 70)
    for r in [x for x in results if x['ticker'] in comparison]:
        print(f"{r['ticker']:<6} | {r['data_pts']:>5} | {r['signals']:>4} | "
              f"{r['hit_rate']:>4.0f}% | {r['avg_return']:>5.1f}% | {r['verdict']}")

    print()
    print('=' * 62)
    print('SUMMARY — PURESIM 8')
    print('=' * 62)
    p8_results = [r for r in results if r['ticker'] in puresim8]
    strong     = [r['ticker'] for r in p8_results if r['hit_rate'] >= 65 and r['signals'] >= 3]
    qualified  = [r['ticker'] for r in p8_results if 50 <= r['hit_rate'] < 65 and r['signals'] >= 3]
    borderline = [r['ticker'] for r in p8_results if 40 <= r['hit_rate'] < 50 and r['signals'] >= 3]
    below      = [r['ticker'] for r in p8_results if r['hit_rate'] < 40 and r['signals'] >= 3]
    thin       = [r['ticker'] for r in p8_results if r['signals'] < 3]

    print(f"STRONG    (65%+, 3+ signals): {', '.join(strong)    or 'none'}")
    print(f"QUALIFIED (50-65%):           {', '.join(qualified) or 'none'}")
    print(f"BORDERLINE(40-50%):           {', '.join(borderline)or 'none'}")
    print(f"BELOW     (<40%):             {', '.join(below)     or 'none'}")
    print(f"THIN      (<3 signals):       {', '.join(thin)      or 'none'}")
    print()
    print('Compare these results against runPureSimHitRateAnalysis.gs output.')
    print('If identical — GS and Supabase data confirmed consistent.')
    print('If different  — investigate data discrepancy before proceeding.')
    print('=' * 62)

    # ── Save results to CSV ──────────────────────────────────────────────────
    df = pd.DataFrame(results)
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy5_hitrate_supabase_results.csv')
    df.to_csv(outfile, index=False)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
