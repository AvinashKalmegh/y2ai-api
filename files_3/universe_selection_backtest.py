"""
UNIVERSE SELECTION BACKTEST
============================
Author:  ARGUS Research
Created: April 1, 2026
Version: 1.0

Purpose:
    Identifies which tickers from the full 530+ universe respond well
    to the DM signal. This is the UNIVERSE SELECTION methodology —
    separate from the Strategy 5 trading rules.

    The two-track design:
    - Track 1 (this script): DM crossover above 70 -> hit rate analysis
      -> produces the V4_Universe qualifying list
    - Track 2 (sim_core_v1.py): Strategy 5 three-condition entry
      -> applied to the V4_Universe for live trading

Methodology:
    Entry signal: DM (EMA5) crosses above 70
                  (previous day DM < 70, current day DM >= 70)
    Hit definition: Max return within FORWARD_DAYS calendar days >= HIT_RETURN
    Hit rate: hits / total crossings per ticker
    Baseline: 47% (random entry, validated January 2026)

Selection criteria (scenario-tested, confirm with Vikram before finalizing):
    Minimum signals: 10
    Minimum hit rate: 40%
    Avoid list: hit rate < 20% with >= 10 signals

Output:
    1. Console report — full universe ranked by hit rate
    2. CSV file — universe_selection_results.csv
    3. V4_Universe candidates — tickers that qualify
    4. Avoid list — tickers to never trade

Run:
    python universe_selection_backtest.py

Requirements:
    pip install supabase pandas numpy python-dotenv tqdm
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase pandas numpy python-dotenv tqdm")
    sys.exit(1)

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

CONFIG = {
    # Supabase connection
    'DM_TABLE':         'dm_history',
    'DM_DATE_COL':      'date',
    'DM_TICKER_COL':    'ticker',
    'DM_CLOSE_COL':     'close',
    'DM_SCORE_COL':     'dm_smoothed',          # EMA5 smoothed DM score

    # Backtest parameters
    'ENTRY_DM':         70,            # DM must cross ABOVE this threshold
    'HIT_RETURN':       0.10,          # 10%+ return = hit
    'FORWARD_DAYS':     90,            # calendar days to measure return

    # Selection thresholds
    'MIN_SIGNALS':      10,            # minimum crossings to be statistically meaningful
    'QUALIFY_HIT_RATE': 40,            # % hit rate to enter V4_Universe
    'AVOID_HIT_RATE':   20,            # % hit rate for avoid list

    # Date range
    'START_DATE':       '2016-01-01',
    'END_DATE':         '2026-03-31',
}

# -----------------------------------------------------------------------------
# SUPABASE CONNECTION
# -----------------------------------------------------------------------------

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


# -----------------------------------------------------------------------------
# DATA LOADER
# -----------------------------------------------------------------------------

def load_all_data(client, cfg):
    """
    Load full DM history from Supabase for all tickers.
    Loads year-by-year to avoid Supabase statement timeout.
    Returns DataFrame with columns: date, ticker, close, dm
    """
    table      = cfg['DM_TABLE']
    date_col   = cfg['DM_DATE_COL']
    ticker_col = cfg['DM_TICKER_COL']
    price_col  = cfg['DM_CLOSE_COL']
    dm_col     = cfg['DM_SCORE_COL']
    start      = cfg['START_DATE']
    end        = cfg['END_DATE']

    start_year = int(start[:4])
    end_year   = int(end[:4])

    print(f"Loading DM data from {start} to {end} (year by year)...")
    rows = []
    page = 10000

    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end   = f"{year}-12-31"
        offset  = 0

        while True:
            r = (client.table(table)
                 .select(f"{date_col},{ticker_col},{price_col},{dm_col}")
                 .gte(date_col, y_start)
                 .lte(date_col, y_end)
                 .order(date_col)
                 .range(offset, offset + page - 1)
                 .execute())
            rows.extend(r.data)
            if len(r.data) < page:
                break
            offset += page

        print(f"  {year}: {len(rows):,} total rows so far")

    print(f"  Total rows loaded: {len(rows):,}")

    df = pd.DataFrame(rows)
    df.rename(columns={
        date_col:   'date',
        ticker_col: 'ticker',
        price_col:  'close',
        dm_col:     'dm',
    }, inplace=True)

    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date', 'ticker', 'close', 'dm'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"  Tickers found: {df['ticker'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df


# -----------------------------------------------------------------------------
# CROSSOVER DETECTION + HIT RATE CALCULATION
# -----------------------------------------------------------------------------

def run_crossover_backtest(df, cfg):
    """
    For each ticker:
    1. Find every DM crossover above ENTRY_DM (prev < threshold, curr >= threshold)
    2. Measure max return within FORWARD_DAYS calendar days
    3. If max return >= HIT_RETURN, count as hit
    4. Calculate hit rate

    Returns DataFrame with per-ticker results.
    """
    entry_dm     = cfg['ENTRY_DM']
    hit_return   = cfg['HIT_RETURN']
    forward_days = cfg['FORWARD_DAYS']

    results = []
    tickers = sorted(df['ticker'].unique())
    total   = len(tickers)

    print(f"\nRunning crossover backtest on {total:,} tickers...")

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Processing {i:,}/{total:,} tickers...", end='\r')

        tdf = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        n   = len(tdf)

        if n < 20:
            continue

        # Find crossovers: prev DM < threshold, curr DM >= threshold
        signals     = 0
        hits        = 0
        total_ret   = 0.0
        best_ret    = -999.0
        worst_ret   = 999.0
        signal_dates = []

        for j in range(1, n):
            prev_dm = tdf.loc[j-1, 'dm']
            curr_dm = tdf.loc[j, 'dm']

            # Crossover condition
            if not (prev_dm < entry_dm and curr_dm >= entry_dm):
                continue

            entry_price = tdf.loc[j, 'close']
            entry_date  = tdf.loc[j, 'date']

            if entry_price <= 0:
                continue

            signals += 1
            signal_dates.append(str(entry_date))

            # Find max return within forward_days calendar days
            cutoff_date = entry_date + timedelta(days=forward_days)
            forward = tdf[(tdf['date'] > entry_date) & (tdf['date'] <= cutoff_date)]

            if forward.empty:
                # No forward data — skip this signal
                signals -= 1
                continue

            max_price = forward['close'].max()
            max_ret   = (max_price - entry_price) / entry_price

            total_ret += max_ret
            if max_ret >= hit_return:
                hits += 1
            if max_ret > best_ret:
                best_ret  = max_ret
            if max_ret < worst_ret:
                worst_ret = max_ret

        if signals == 0:
            continue

        hit_rate   = hits / signals * 100
        avg_return = total_ret / signals * 100

        results.append({
            'ticker':       ticker,
            'data_points':  n,
            'signals':      signals,
            'hits':         hits,
            'hit_rate':     round(hit_rate, 1),
            'avg_return':   round(avg_return, 1),
            'best_return':  round(best_ret * 100, 1),
            'worst_return': round(worst_ret * 100, 1),
            'first_date':   str(tdf['date'].iloc[0]),
            'last_date':    str(tdf['date'].iloc[-1]),
        })

    print(f"\n  Backtest complete. {len(results):,} tickers with signals.")
    return pd.DataFrame(results).sort_values('hit_rate', ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# SELECTION LOGIC
# -----------------------------------------------------------------------------

def classify_tickers(df, cfg):
    """
    Classify each ticker based on hit rate and signal count.
    """
    min_sig   = cfg['MIN_SIGNALS']
    qual_rate = cfg['QUALIFY_HIT_RATE']
    avoid_rate = cfg['AVOID_HIT_RATE']

    def classify(row):
        if row['signals'] < min_sig:
            return 'THIN — insufficient signals'
        if row['hit_rate'] >= qual_rate:
            return 'QUALIFY — V4_Universe candidate'
        if row['hit_rate'] < avoid_rate:
            return 'AVOID — below threshold'
        return 'MONITOR — borderline'

    df['verdict'] = df.apply(classify, axis=1)
    return df


# -----------------------------------------------------------------------------
# SCENARIO TESTING
# -----------------------------------------------------------------------------

def run_scenarios(df):
    """
    Test different threshold combinations to find the right universe size.
    """
    scenarios = [
        ('Min 5  signals + 40% HR',  5, 40),
        ('Min 5  signals + 50% HR',  5, 50),
        ('Min 5  signals + 60% HR',  5, 60),
        ('Min 10 signals + 40% HR', 10, 40),
        ('Min 10 signals + 50% HR', 10, 50),
        ('Min 10 signals + 60% HR', 10, 60),
        ('Min 15 signals + 40% HR', 15, 40),
        ('Min 15 signals + 50% HR', 15, 50),
        ('Min 20 signals + 35% HR', 20, 35),
        ('Min 20 signals + 40% HR', 20, 40),
        ('Min 20 signals + 45% HR', 20, 45),
        ('Min 20 signals + 50% HR', 20, 50),
        ('Min 25 signals + 40% HR', 25, 40),
        ('Min 30 signals + 40% HR', 30, 40),
    ]

    print("\n" + "="*75)
    print("SCENARIO TESTING — UNIVERSE SIZE AT DIFFERENT THRESHOLDS")
    print("="*75)
    print(f"{'Scenario':<35} | {'Size':6} | Top 10 Qualifiers")
    print(f"{'--------':<35} | {'------':6} | ------------------")

    for label, min_s, min_hr in scenarios:
        q = df[(df['signals'] >= min_s) & (df['hit_rate'] >= min_hr)]
        top10 = ', '.join(q.head(10)['ticker'].values)
        print(f"{label:<35} | {len(q):6} | {top10}")


# -----------------------------------------------------------------------------
# MAIN REPORT
# -----------------------------------------------------------------------------

def print_report(df, cfg):
    """
    Print the full results report.
    """
    min_sig   = cfg['MIN_SIGNALS']
    qual_rate = cfg['QUALIFY_HIT_RATE']
    avoid_rate = cfg['AVOID_HIT_RATE']

    qualified = df[(df['signals'] >= min_sig) & (df['hit_rate'] >= qual_rate)]
    avoid     = df[(df['signals'] >= min_sig) & (df['hit_rate'] < avoid_rate)]
    borderline = df[(df['signals'] >= min_sig) &
                    (df['hit_rate'] >= avoid_rate) &
                    (df['hit_rate'] < qual_rate)]

    print("\n" + "="*75)
    print("UNIVERSE SELECTION BACKTEST — FULL RESULTS")
    print(f"Entry: DM crosses above {cfg['ENTRY_DM']}")
    print(f"Hit: >{cfg['HIT_RETURN']*100:.0f}% max return within {cfg['FORWARD_DAYS']} calendar days")
    print(f"Baseline: 47% (random entry, validated Jan 2026)")
    print(f"Period: {cfg['START_DATE']} to {cfg['END_DATE']}")
    print("="*75)

    print(f"\n{'='*75}")
    print(f"V4_UNIVERSE CANDIDATES — {len(qualified)} tickers")
    print(f"(Min {min_sig} signals + {qual_rate}%+ hit rate)")
    print(f"{'='*75}")
    print(f"{'Ticker':8} | {'Signals':8} | {'Hits':5} | {'Hit%':6} | {'AvgRet':7} | {'Best':7} | {'Worst':7} | {'First Date':12}")
    print(f"{'--------':8} | {'--------':8} | {'-----':5} | {'------':6} | {'-------':7} | {'-------':7} | {'-------':7} | {'----------':12}")
    for _, r in qualified.iterrows():
        print(f"{r['ticker']:8} | {r['signals']:8} | {r['hits']:5} | {r['hit_rate']:5.1f}% | {r['avg_return']:6.1f}% | {r['best_return']:6.1f}% | {r['worst_return']:6.1f}% | {r['first_date'][:10]:12}")

    print(f"\n{'='*75}")
    print(f"BORDERLINE — {len(borderline)} tickers ({avoid_rate}-{qual_rate}% hit rate, {min_sig}+ signals)")
    print(f"{'='*75}")
    for _, r in borderline.head(20).iterrows():
        print(f"  {r['ticker']:8} | {r['signals']} signals | {r['hit_rate']:.1f}% HR | avg {r['avg_return']:.1f}%")

    print(f"\n{'='*75}")
    print(f"AVOID LIST — {len(avoid)} tickers (<{avoid_rate}% hit rate, {min_sig}+ signals)")
    print(f"{'='*75}")
    for _, r in avoid.head(20).iterrows():
        print(f"  {r['ticker']:8} | {r['signals']} signals | {r['hit_rate']:.1f}% HR | avg {r['avg_return']:.1f}%")

    print(f"\n{'='*75}")
    print(f"SUMMARY")
    print(f"{'='*75}")
    print(f"Total tickers with signals: {len(df):,}")
    print(f"V4_Universe candidates:     {len(qualified):,}")
    print(f"Borderline (monitor):       {len(borderline):,}")
    print(f"Avoid list:                 {len(avoid):,}")
    print(f"Thin (<{min_sig} signals):       {len(df[df['signals'] < min_sig]):,}")

    # Run scenario testing
    run_scenarios(df)

    return qualified, avoid


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("="*75)
    print("ARGUS UNIVERSE SELECTION BACKTEST")
    print("DM Crossover Above 70 — Full Universe — 2016-2026")
    print("="*75)

    client = get_client()

    # Load data
    df = load_all_data(client, CONFIG)

    # Run backtest
    results = run_crossover_backtest(df, CONFIG)

    # Classify tickers
    results = classify_tickers(results, CONFIG)

    # Print report
    qualified, avoid = print_report(results, CONFIG)

    # Save to CSV
    output_file = 'universe_selection_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")

    qualified_file = 'v4_universe_candidates.csv'
    qualified.to_csv(qualified_file, index=False)
    print(f"V4_Universe candidates saved to: {qualified_file}")

    print("\n="*75)
    print("DONE — share universe_selection_results.csv and")
    print("v4_universe_candidates.csv with Vikram for review.")
    print("="*75)


if __name__ == '__main__':
    main()
