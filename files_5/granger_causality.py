"""
ARGUS CAPITAL FLOW — GRANGER CAUSALITY ANALYSIS
=================================================
Purpose: Test whether the DM signal contains information about future
         price returns that is NOT already contained in past price returns.

This directly answers the momentum criticism:
"Your signal is just momentum in disguise."

If DM Granger-causes returns AFTER controlling for past returns,
the signal has independent predictive content beyond momentum.

The Test (in plain English):
  1. Can past PRICE RETURNS alone predict future price returns?  (baseline)
  2. Does adding past DM SCORES improve that prediction significantly?

If yes to (2): DM contains information beyond momentum. Signal is not
               just momentum in disguise.
If no  to (2): Momentum explains everything. DM adds nothing.

Statistical Method:
  - Vector Autoregression (VAR) framework
  - Granger causality F-test (Wald test)
  - Tested at multiple lag lengths (1, 5, 10, 20 trading days)
  - Run per-ticker AND pooled across full universe
  - p-value < 0.05 = DM Granger-causes returns at that lag

Developer Notes:
  - Requires daily DM score history and daily price history
  - Same data as lead_lag_analysis.py — can reuse the Supabase pulls
  - statsmodels library required: pip install statsmodels
  - Run time: 30-60 minutes for full universe
  - Output: granger_results.csv + granger_summary.txt

Output Files:
  granger_results.csv    — per-ticker Granger test results
  granger_summary.txt    — pooled summary + interpretation
"""

import os
import sys
import csv
import warnings
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase")
    sys.exit(1)

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    print("ERROR: pip install statsmodels")
    sys.exit(1)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    'DM_TABLE':      'dm_history',
    'DM_DATE_COL':   'date',
    'DM_TICKER_COL': 'ticker',
    'DM_PRICE_COL':  'close',
    'DM_SCORE_COL':  'dm_smoothed',

    # Granger test parameters
    'LAGS':          [20, 40, 60, 90],    # trading days — test at each lag
    'MIN_OBS':       120,               # minimum observations per ticker
    'START_DATE':    '2016-01-01',
    'END_DATE':      '2026-03-25',
    'P_VALUE_THRESHOLD': 0.05,          # significance level

    # Strategy 5 trades — to focus analysis on traded universe
    'TRADES_FILE':   'trades_strategy_5_momentum_confirm_cf_65_5d_ma_.csv',
}


# ── SUPABASE CONNECTION ───────────────────────────────────────────────────────

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


# ── DATA LOADER ───────────────────────────────────────────────────────────────

def load_ticker_history(client, ticker, start_date, end_date):
    """Load full daily history for one ticker."""
    table      = CONFIG['DM_TABLE']
    date_col   = CONFIG['DM_DATE_COL']
    ticker_col = CONFIG['DM_TICKER_COL']
    price_col  = CONFIG['DM_PRICE_COL']
    dm_col     = CONFIG['DM_SCORE_COL']

    rows, offset, page = [], 0, 5000
    while True:
        r = (client.table(table)
             .select(f"{date_col},{price_col},{dm_col}")
             .eq(ticker_col, ticker)
             .gte(date_col, str(start_date))
             .lte(date_col, str(end_date))
             .order(date_col)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page:
            break
        offset += page

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.rename(columns={
        date_col:   'date',
        price_col:  'close',
        dm_col:     'dm'
    }, inplace=True)
    df['date']  = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna().sort_values('date').reset_index(drop=True)
    return df


# ── STATIONARITY CHECK ────────────────────────────────────────────────────────

def make_stationary(series, name='series'):
    """
    Granger causality requires stationary series.
    Returns first-differenced series if original is non-stationary.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    p_value = result[1]
    if p_value > 0.05:
        # Non-stationary — difference it
        return series.diff().dropna(), True
    return series, False


# ── GRANGER TEST PER TICKER ───────────────────────────────────────────────────

def run_granger_for_ticker(df, ticker, lags):
    """
    Run Granger causality test: does DM Granger-cause price returns?

    Returns dict with results at each lag length.
    """
    if len(df) < CONFIG['MIN_OBS']:
        return None

    # Compute log returns
    df = df.copy()
    df['return'] = np.log(df['close']).diff()
    df = df.dropna()

    if len(df) < CONFIG['MIN_OBS']:
        return None

    # Make stationary
    returns_stat, returns_differenced = make_stationary(df['return'], 'returns')
    dm_stat, dm_differenced = make_stationary(df['dm'], 'dm')

    # Align lengths
    min_len = min(len(returns_stat), len(dm_stat))
    returns_stat = returns_stat.iloc[-min_len:].reset_index(drop=True)
    dm_stat      = dm_stat.iloc[-min_len:].reset_index(drop=True)

    if len(returns_stat) < CONFIG['MIN_OBS']:
        return None

    # Build test DataFrame: [returns, dm]
    # grangercausalitytests tests whether column 2 (dm) Granger-causes column 1 (returns)
    test_df = pd.DataFrame({
        'returns': returns_stat,
        'dm':      dm_stat
    }).dropna()

    if len(test_df) < CONFIG['MIN_OBS']:
        return None

    results = {
        'ticker':              ticker,
        'n_obs':               len(test_df),
        'returns_differenced': returns_differenced,
        'dm_differenced':      dm_differenced,
    }

    for lag in lags:
        if len(test_df) < lag * 3 + 10:
            results[f'p_value_lag{lag}']    = None
            results[f'f_stat_lag{lag}']     = None
            results[f'significant_lag{lag}'] = None
            continue
        try:
            gc = grangercausalitytests(test_df[['returns','dm']], maxlag=lag, verbose=False)
            # Extract F-test p-value at this lag
            p_val = gc[lag][0]['ssr_ftest'][1]
            f_stat = gc[lag][0]['ssr_ftest'][0]
            results[f'p_value_lag{lag}']    = round(p_val, 4)
            results[f'f_stat_lag{lag}']     = round(f_stat, 4)
            results[f'significant_lag{lag}'] = p_val < CONFIG['P_VALUE_THRESHOLD']
        except Exception as e:
            results[f'p_value_lag{lag}']    = None
            results[f'f_stat_lag{lag}']     = None
            results[f'significant_lag{lag}'] = None

    return results


# ── POOLED ANALYSIS ───────────────────────────────────────────────────────────

def run_pooled_granger(all_data, lags):
    """
    Pool all tickers into one long panel and run Granger test.
    This gives the aggregate signal across the full universe.
    """
    print("\nRunning pooled Granger causality across full universe...")

    frames = []
    for ticker, df in all_data.items():
        if len(df) < CONFIG['MIN_OBS']:
            continue
        d = df.copy()
        d['return'] = np.log(d['close']).diff()
        d['ticker'] = ticker
        frames.append(d[['date','return','dm','ticker']].dropna())

    if not frames:
        return None

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values('date').reset_index(drop=True)

    # Use aggregate: median DM and median return across all tickers per day
    daily = pooled.groupby('date').agg(
        median_return=('return','median'),
        median_dm=('dm','median'),
        n_tickers=('ticker','count')
    ).reset_index()

    daily = daily[daily['n_tickers'] >= 10].copy()

    if len(daily) < CONFIG['MIN_OBS']:
        return None

    ret_stat, _ = make_stationary(daily['median_return'])
    dm_stat, _  = make_stationary(daily['median_dm'])

    min_len = min(len(ret_stat), len(dm_stat))
    test_df = pd.DataFrame({
        'returns': ret_stat.iloc[-min_len:].values,
        'dm':      dm_stat.iloc[-min_len:].values
    }).dropna()

    pooled_results = {'n_obs': len(test_df)}

    for lag in lags:
        if len(test_df) < lag * 3 + 10:
            pooled_results[f'lag{lag}'] = None
            continue
        try:
            gc = grangercausalitytests(test_df[['returns','dm']], maxlag=lag, verbose=False)
            p_val  = gc[lag][0]['ssr_ftest'][1]
            f_stat = gc[lag][0]['ssr_ftest'][0]
            pooled_results[f'lag{lag}'] = {
                'p_value':    round(p_val, 4),
                'f_stat':     round(f_stat, 4),
                'significant': p_val < CONFIG['P_VALUE_THRESHOLD']
            }
        except Exception as e:
            pooled_results[f'lag{lag}'] = None

    return pooled_results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*65)
    print("  ARGUS — GRANGER CAUSALITY ANALYSIS")
    print("  Does DM predict returns beyond momentum?")
    print("="*65 + "\n")

    # Load traded universe from Strategy 5 trades
    trades_file = CONFIG['TRADES_FILE']
    if os.path.exists(trades_file):
        trades = pd.read_csv(trades_file, encoding='latin-1')
        universe = sorted(trades['ticker'].unique().tolist())
        print(f"Loaded {len(universe)} tickers from Strategy 5 trade file.")
    else:
        print(f"Trade file not found: {trades_file}")
        print("Will use full DM universe — this will take longer.")
        universe = None

    try:
        client = get_client()
        print("Connected to Supabase.\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # If no trade file, pull universe from DM table
    if universe is None:
        r = client.table(CONFIG['DM_TABLE']).select(
            CONFIG['DM_TICKER_COL']
        ).execute()
        universe = sorted(set(row[CONFIG['DM_TICKER_COL']] for row in r.data))
        print(f"Pulled {len(universe)} tickers from DM table.")

    lags       = CONFIG['LAGS']
    start_date = CONFIG['START_DATE']
    end_date   = CONFIG['END_DATE']

    results   = []
    all_data  = {}
    n         = len(universe)

    print(f"Running Granger tests for {n} tickers at lags {lags}...\n")

    for i, ticker in enumerate(universe):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{n} — {ticker}")

        df = load_ticker_history(client, ticker, start_date, end_date)
        if df.empty:
            continue

        all_data[ticker] = df
        result = run_granger_for_ticker(df, ticker, lags)
        if result:
            results.append(result)

    if not results:
        print("ERROR: No valid results produced.")
        sys.exit(1)

    # Write per-ticker results
    out_file = 'granger_results.csv'
    fieldnames = results[0].keys()
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-ticker results written to {out_file}")

    # Run pooled analysis
    pooled = run_pooled_granger(all_data, lags)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)

    lines = []
    def log(s=''):
        print(s)
        lines.append(s)

    log()
    log("="*65)
    log("  GRANGER CAUSALITY RESULTS")
    log("  Does DM Granger-cause price returns?")
    log("="*65)
    log(f"\n  Tickers tested:  {len(df_results)}")
    log(f"  Lags tested:     {lags} trading days")
    log(f"  Significance:    p < {CONFIG['P_VALUE_THRESHOLD']}")
    log()

    # Per-lag summary
    log("  ── PER-TICKER RESULTS ─────────────────────────────────────")
    log("  % of tickers where DM significantly Granger-causes returns:")
    log()
    for lag in lags:
        col = f'significant_lag{lag}'
        if col not in df_results.columns:
            continue
        valid    = df_results[df_results[col].notna()]
        sig      = df_results[df_results[col] == True]
        pct      = len(sig) / len(valid) * 100 if len(valid) > 0 else 0
        avg_p    = df_results[f'p_value_lag{lag}'].dropna().mean()
        log(f"  Lag {lag:>2}d:  {len(sig):>4}/{len(valid):<4} tickers significant "
            f"({pct:.0f}%)  |  avg p-value: {avg_p:.4f}")

    # Top significant tickers at lag 5
    log()
    log("  ── TOP 15 TICKERS — Strongest Granger Signal (Lag 5d) ────")
    if 'p_value_lag5' in df_results.columns:
        top = df_results.dropna(subset=['p_value_lag5']).nsmallest(15, 'p_value_lag5')
        for _, r in top.iterrows():
            sig_flags = []
            for lag in lags:
                col = f'significant_lag{lag}'
                if col in r and r[col]:
                    sig_flags.append(f"{lag}d")
            log(f"  {r['ticker']:<6}  p={r['p_value_lag5']:.4f}  "
                f"F={r.get('f_stat_lag5', 0):.2f}  "
                f"significant at: {', '.join(sig_flags) if sig_flags else 'none'}")

    # Pooled results
    if pooled:
        log()
        log("  ── POOLED UNIVERSE RESULTS ────────────────────────────────")
        log(f"  (Median DM and returns across full universe, {pooled['n_obs']} daily obs)")
        log()
        for lag in lags:
            key = f'lag{lag}'
            if key in pooled and pooled[key]:
                p   = pooled[key]['p_value']
                f   = pooled[key]['f_stat']
                sig = "✓ SIGNIFICANT" if pooled[key]['significant'] else "✗ not significant"
                log(f"  Lag {lag:>2}d:  p={p:.4f}  F={f:.2f}  {sig}")
            else:
                log(f"  Lag {lag:>2}d:  insufficient data")

    # Interpretation
    log()
    log("="*65)
    log("  INTERPRETATION")
    log("="*65)

    # Count significant at lag 5
    if 'significant_lag5' in df_results.columns:
        valid5 = df_results[df_results['significant_lag5'].notna()]
        sig5   = df_results[df_results['significant_lag5'] == True]
        pct5   = len(sig5) / len(valid5) * 100 if len(valid5) > 0 else 0

        log()
        if pct5 >= 40:
            log(f"  ✓✓ STRONG: {pct5:.0f}% of tickers show DM Granger-causes returns at 5d lag.")
            log("  The DM signal contains information about future returns that is NOT")
            log("  explained by past price returns alone.")
            log("  This is strong statistical evidence against the momentum criticism.")
        elif pct5 >= 20:
            log(f"  ✓  MODERATE: {pct5:.0f}% of tickers show Granger causality at 5d lag.")
            log("  DM adds predictive power beyond momentum for a meaningful subset.")
            log("  Partial counter to momentum criticism — strongest in specific sectors.")
        else:
            log(f"  ✗  WEAK: Only {pct5:.0f}% of tickers show Granger causality at 5d lag.")
            log("  Momentum may explain most of the DM signal's predictive power.")
            log("  Focus defense on temporal stability and lead-lag evidence instead.")

    log()
    log("  NOTE ON INTERPRETATION:")
    log("  Granger causality is NOT true causality — it is predictive precedence.")
    log("  'DM Granger-causes returns' means: past DM scores improve the prediction")
    log("  of future returns beyond what past returns alone predict.")
    log("  This is the correct statistical framing for 'signal leads price'.")
    log()
    log("  COMBINED WITH LEAD-LAG ANALYSIS:")
    log("  If Granger test is significant AND lead-lag shows positive lead days,")
    log("  this is the strongest possible evidence that DM detects accumulation")
    log("  BEFORE price momentum develops.")
    log()
    log("="*65)

    summary = '\n'.join(lines)
    with open('granger_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\nSummary written to granger_summary.txt")
    print("DONE.\n")


if __name__ == '__main__':
    run()
