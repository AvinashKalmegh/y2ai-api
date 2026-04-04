"""
WALK-FORWARD LOOKBACK WINDOW VALIDATION
=========================================
ARGUS Research -- PureSim Universe Selection
Author: Vikram Amsethi / Avinash Kalmegh
Created: April 2026

PURPOSE:
    Determines the optimal lookback window for PureSim Universe selection
    by testing whether the DM crossover hit rate methodology is predictive
    out-of-sample under three lookback approaches:

    1. FULL HISTORY      -- use all available data from ticker inception
    2. TRAILING 3 YEARS  -- use only the last 3 years of DM history
    3. TRAILING 5 YEARS  -- use only the last 5 years of DM history

SCIENTIFIC METHODOLOGY -- WALK-FORWARD VALIDATION:
    Split the full history into folds.
    For each fold:
        - Use the lookback window BEFORE the fold to select the universe
          (tickers qualifying at >= MIN_SIGNALS and >= PREF_HIT_RATE)
        - Measure actual hit rates DURING the fold (out of sample)
        - Compare predicted vs actual hit rates
    The lookback approach that best predicts out-of-sample performance wins.

FOLD DESIGN:
    Total history: ~2016-2026 (10 years)
    Fold size: 1 year
    Min training window: 2 years
    Folds tested: 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025

OUTPUT:
    1. Console summary -- which lookback approach predicts best
    2. CSV: walkforward_results.csv -- fold-by-fold detail
    3. CSV: lookback_comparison.csv -- final recommendation

DECISION THIS PRODUCES:
    The winning lookback approach becomes the documented methodology
    for PureSim Universe selection going forward.

Run:
    python walkforward_lookback_test.py

Requirements:
    pip install supabase pandas numpy python-dotenv tqdm
"""

import os
import sys
import csv
from datetime import date, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: pip install supabase")
    sys.exit(1)

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

CONFIG = {
    # Supabase
    'DM_TABLE':     'dm_history',
    'DATE_COL':     'date',
    'TICKER_COL':   'ticker',
    'CLOSE_COL':    'close',
    'DM_COL':       'dm_smoothed',

    # Signal
    'CROSSOVER_THRESHOLD': 70,
    'HIT_PCT':             0.10,
    'FORWARD_DAYS':        90,

    # Selection thresholds
    'MIN_SIGNALS':  10,
    'PREF_HIT_RATE': 40,

    # Walk-forward parameters
    'FOLD_YEARS':   [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'MIN_TRAIN_YRS': 2,     # minimum years of training data before first fold

    # Lookback windows to test
    'LOOKBACK_WINDOWS': {
        'full_history':    None,   # None = use all data before fold
        'trailing_3yr':    3,
        'trailing_5yr':    5,
    },

    # Output
    'OUTPUT_DETAIL':     'walkforward_results.csv',
    'OUTPUT_COMPARISON': 'lookback_comparison.csv',
}


# -----------------------------------------------------------------------------
# SUPABASE
# -----------------------------------------------------------------------------

def get_client() -> Client:
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def load_all_data(client: Client) -> pd.DataFrame:
    """Load complete DM history from Supabase -- all tickers, all dates. Year by year to avoid timeout."""
    table = CONFIG['DM_TABLE']
    print("Loading full DM history from Supabase (year by year)...")
    print("(This loads once and all walk-forward analysis runs in memory)")

    rows = []
    page_size = 10000
    for year in range(2016, 2027):
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        offset = 0
        while True:
            r = (client.table(table)
                 .select(f"{CONFIG['DATE_COL']},{CONFIG['TICKER_COL']},{CONFIG['CLOSE_COL']},{CONFIG['DM_COL']}")
                 .gte(CONFIG['DATE_COL'], y_start)
                 .lte(CONFIG['DATE_COL'], y_end)
                 .order(CONFIG['DATE_COL'])
                 .range(offset, offset + page_size - 1)
                 .execute())
            rows.extend(r.data)
            if len(r.data) < page_size:
                break
            offset += page_size
        print(f"  {year}: {len(rows):,} total rows")

    df = pd.DataFrame(rows)
    df.rename(columns={
        CONFIG['DATE_COL']:   'date',
        CONFIG['TICKER_COL']: 'ticker',
        CONFIG['CLOSE_COL']:  'close',
        CONFIG['DM_COL']:     'dm',
    }, inplace=True)

    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date', 'ticker', 'close', 'dm'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows | {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


# -----------------------------------------------------------------------------
# CROSSOVER + HIT RATE FUNCTIONS
# -----------------------------------------------------------------------------

def compute_hit_rate_for_period(ticker_df: pd.DataFrame) -> dict:
    """
    Compute DM crossover hit rate for a single ticker over a given period.
    Returns signals, hits, hit_rate, avg_max_return.
    """
    threshold = CONFIG['CROSSOVER_THRESHOLD']
    hit_pct   = CONFIG['HIT_PCT']
    fwd_days  = CONFIG['FORWARD_DAYS']

    rows = ticker_df.sort_values('date').reset_index(drop=True)
    if len(rows) < 5:
        return {'signals': 0, 'hits': 0, 'hit_rate': 0.0, 'avg_max_ret': 0.0}

    signals, hits = 0, 0
    returns = []

    for i in range(1, len(rows)):
        prev_dm = rows.loc[i-1, 'dm']
        curr_dm = rows.loc[i,   'dm']

        # Crossover
        if prev_dm < threshold and curr_dm >= threshold:
            entry_close = rows.loc[i, 'close']
            if entry_close <= 0:
                continue

            signals += 1
            end_idx = min(i + fwd_days, len(rows) - 1)
            fwd_closes = rows.loc[i+1 : end_idx, 'close'].values

            if len(fwd_closes) == 0:
                returns.append(0.0)
                continue

            max_return = (fwd_closes.max() - entry_close) / entry_close
            returns.append(max_return)
            if max_return >= hit_pct:
                hits += 1

    if signals == 0:
        return {'signals': 0, 'hits': 0, 'hit_rate': 0.0, 'avg_max_ret': 0.0}

    return {
        'signals':     signals,
        'hits':        hits,
        'hit_rate':    round(hits / signals * 100, 1),
        'avg_max_ret': round(np.mean(returns) * 100, 1),
    }


def select_universe(df: pd.DataFrame, train_start: date, train_end: date,
                    lookback_years: int = None) -> set:
    """
    Select qualifying tickers based on hit rates in the training period.
    If lookback_years is set, only use the trailing N years within training period.
    """
    min_sigs = CONFIG['MIN_SIGNALS']
    pref_hr  = CONFIG['PREF_HIT_RATE']

    # Apply lookback window
    if lookback_years:
        effective_start = date(train_end.year - lookback_years, train_end.month, train_end.day)
        effective_start = max(effective_start, train_start)
    else:
        effective_start = train_start

    train_df = df[(df['date'] >= effective_start) & (df['date'] <= train_end)]

    qualified = set()
    for ticker, tdf in train_df.groupby('ticker'):
        result = compute_hit_rate_for_period(tdf)
        if result['signals'] >= min_sigs and result['hit_rate'] >= pref_hr:
            qualified.add(ticker)

    return qualified


def measure_oos_performance(df: pd.DataFrame, universe: set,
                             fold_start: date, fold_end: date) -> dict:
    """
    Measure actual hit rates for the selected universe during the out-of-sample fold.
    Returns avg hit rate, universe size, tickers found.
    """
    fold_df = df[(df['date'] >= fold_start) & (df['date'] <= fold_end)]

    hit_rates = []
    tickers_found = []

    for ticker in universe:
        tdf = fold_df[fold_df['ticker'] == ticker]
        if len(tdf) < 5:
            continue
        result = compute_hit_rate_for_period(tdf)
        if result['signals'] > 0:
            hit_rates.append(result['hit_rate'])
            tickers_found.append(ticker)

    if not hit_rates:
        return {
            'oos_avg_hit_rate': 0.0,
            'oos_tickers_with_signals': 0,
            'universe_size': len(universe),
        }

    return {
        'oos_avg_hit_rate':         round(np.mean(hit_rates), 1),
        'oos_tickers_with_signals': len(tickers_found),
        'universe_size':            len(universe),
    }


# -----------------------------------------------------------------------------
# WALK-FORWARD ENGINE
# -----------------------------------------------------------------------------

def run_walkforward(df: pd.DataFrame) -> list:
    """
    Run walk-forward validation for all lookback windows across all folds.
    Returns list of result dicts.
    """
    fold_years      = CONFIG['FOLD_YEARS']
    lookback_windows = CONFIG['LOOKBACK_WINDOWS']
    results = []

    print(f"\nRunning walk-forward validation...")
    print(f"Folds: {fold_years}")
    print(f"Lookback windows: {list(lookback_windows.keys())}")
    print()

    for fold_year in fold_years:
        fold_start = date(fold_year, 1, 1)
        fold_end   = date(fold_year, 12, 31)
        train_end  = date(fold_year - 1, 12, 31)
        train_start = date(2016, 1, 1)  # full history start

        # Check we have enough training data
        min_train_start = date(fold_year - CONFIG['MIN_TRAIN_YRS'], 1, 1)
        if train_end < min_train_start:
            print(f"  Fold {fold_year}: Skipped -- insufficient training data")
            continue

        print(f"  Fold {fold_year}: train until {train_end} | test {fold_start} to {fold_end}")

        for window_name, lookback_years in lookback_windows.items():
            # Select universe using training data
            universe = select_universe(df, train_start, train_end, lookback_years)

            # Measure out-of-sample performance
            oos = measure_oos_performance(df, universe, fold_start, fold_end)

            # Also measure in-sample hit rate for comparison
            if lookback_years:
                is_start = date(fold_year - 1 - lookback_years, 1, 1)
                is_start = max(is_start, train_start)
            else:
                is_start = train_start

            train_df = df[(df['date'] >= is_start) & (df['date'] <= train_end)]
            in_sample_hrs = []
            for ticker in universe:
                tdf = train_df[train_df['ticker'] == ticker]
                r = compute_hit_rate_for_period(tdf)
                if r['signals'] > 0:
                    in_sample_hrs.append(r['hit_rate'])

            is_avg_hr = round(np.mean(in_sample_hrs), 1) if in_sample_hrs else 0.0
            prediction_error = round(abs(is_avg_hr - oos['oos_avg_hit_rate']), 1)

            results.append({
                'fold_year':        fold_year,
                'window':           window_name,
                'lookback_years':   lookback_years if lookback_years else 'full',
                'universe_size':    oos['universe_size'],
                'is_avg_hit_rate':  is_avg_hr,
                'oos_avg_hit_rate': oos['oos_avg_hit_rate'],
                'oos_tickers':      oos['oos_tickers_with_signals'],
                'prediction_error': prediction_error,
            })

            print(f"    {window_name:15} | universe: {oos['universe_size']:3} | "
                  f"in-sample HR: {is_avg_hr:5.1f}% | "
                  f"out-of-sample HR: {oos['oos_avg_hit_rate']:5.1f}% | "
                  f"error: {prediction_error:4.1f}pts")

    return results


# -----------------------------------------------------------------------------
# RESULTS SUMMARY
# -----------------------------------------------------------------------------

def summarize_results(results: list) -> pd.DataFrame:
    """
    Aggregate walk-forward results by lookback window.
    Lower prediction error = better predictive accuracy.
    """
    df = pd.DataFrame(results)

    summary = df.groupby('window').agg(
        folds_tested        = ('fold_year',        'count'),
        avg_universe_size   = ('universe_size',     'mean'),
        avg_is_hit_rate     = ('is_avg_hit_rate',   'mean'),
        avg_oos_hit_rate    = ('oos_avg_hit_rate',  'mean'),
        avg_prediction_error= ('prediction_error',  'mean'),
        min_prediction_error= ('prediction_error',  'min'),
        max_prediction_error= ('prediction_error',  'max'),
    ).round(1).reset_index()

    summary = summary.sort_values('avg_prediction_error')

    print()
    print('=' * 80)
    print('WALK-FORWARD VALIDATION RESULTS')
    print('Lower prediction error = better predictive accuracy = recommended approach')
    print('=' * 80)
    print(f'{"Window":<18} | {"Folds":5} | {"Universe":8} | {"IS HR%":6} | {"OOS HR%":7} | {"Avg Error":9} | {"Recommendation"}')
    print(f'{"------":<18} | {"-----":5} | {"--------":8} | {"------":6} | {"-------":7} | {"---------":9} | {"---------------"}')

    best_window = summary.iloc[0]['window']

    for _, r in summary.iterrows():
        rec = '* RECOMMENDED' if r['window'] == best_window else ''
        print(f'{r["window"]:<18} | {r["folds_tested"]:5.0f} | {r["avg_universe_size"]:8.1f} | '
              f'{r["avg_is_hit_rate"]:6.1f} | {r["avg_oos_hit_rate"]:7.1f} | '
              f'{r["avg_prediction_error"]:9.1f} | {rec}')

    print()
    print(f'RECOMMENDATION: Use {best_window.upper()} lookback for PureSim Universe selection')
    print(f'  Average prediction error: {summary.iloc[0]["avg_prediction_error"]:.1f} percentage points')
    print(f'  Average universe size:    {summary.iloc[0]["avg_universe_size"]:.0f} tickers')
    print('=' * 80)

    return summary


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    print('=' * 80)
    print('PURESIM UNIVERSE -- WALK-FORWARD LOOKBACK WINDOW VALIDATION')
    print('Testing: Full History vs Trailing 3yr vs Trailing 5yr')
    print('Metric: Which lookback best predicts out-of-sample hit rates?')
    print('=' * 80)
    print()

    client  = get_client()
    df      = load_all_data(client)
    results = run_walkforward(df)
    summary = summarize_results(results)

    # Save outputs
    detail_df = pd.DataFrame(results)
    detail_df.to_csv(CONFIG['OUTPUT_DETAIL'], index=False)
    summary.to_csv(CONFIG['OUTPUT_COMPARISON'], index=False)

    print(f'\nDetailed results saved to: {CONFIG["OUTPUT_DETAIL"]}')
    print(f'Summary comparison saved to: {CONFIG["OUTPUT_COMPARISON"]}')
    print()
    print('NEXT STEP:')
    print('  Use the recommended lookback window in dm_crossover_hitrate.py')
    print('  to produce the final PureSim_Universe candidate list.')


if __name__ == '__main__':
    main()
