"""
ARGUS — HMS PREDICTS DM? GRANGER CAUSALITY TEST
=================================================
Purpose: Test whether HMS (slow institutional money) Granger-causes
         DM (faster institutional money).

Hypothesis: Institutional capital deployment occurs in sequential waves.
  Wave 1: Slow money (pension funds, sovereign wealth, long-only mandates)
           detected by HMS — weeks to months lead time.
  Wave 2: Fast money (hedge funds, tactical allocators, momentum followers)
           detected by DM — days to weeks lead time.
  Wave 3: Price confirmation.

If HMS Granger-causes DM, the hypothesis is confirmed:
  - HMS is the EARLY WARNING layer
  - DM is the CONFIRMATION layer
  - Price is the OUTCOME layer

This would explain why using HMS as a simultaneous filter degraded
simulation performance — you were requiring both waves to peak together
rather than using HMS to detect Wave 1 and DM to time the Wave 2 entry.

Practical implication if confirmed:
  Entry signal = HMS already elevated AND DM just crossing 65 on rising MA
  This catches the trade at the moment fast money joins slow money —
  before price confirms.

Statistical method:
  - Granger causality F-test (VAR framework)
  - Tests: does past HMS predict future DM beyond past DM alone?
  - Also tests reverse: does past DM predict future HMS? (should be weaker)
  - Tested at lags: 5, 10, 20, 40 trading days
  - Per-ticker AND cross-sectional pooled analysis

Developer notes:
  - Requires daily HMS scores AND daily DM scores per ticker
  - HMS scores should come from the HMS_Daily spreadsheet or Supabase
  - DM scores come from the same dm_history table as other scripts
  - statsmodels required: pip install statsmodels
  - Run time: 30-60 minutes
  - Output: hms_dm_granger_results.csv + hms_dm_granger_summary.txt

Key output to look for:
  If HMS→DM Granger causality is significant at lag 10-20 days,
  and DM→HMS is NOT significant or weaker, this confirms HMS leads DM.
  That's the sequential wave hypothesis confirmed.
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
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
except ImportError:
    print("ERROR: pip install statsmodels")
    sys.exit(1)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # DM data
    'DM_TABLE':      'dm_history',
    'DM_DATE_COL':   'date',
    'DM_TICKER_COL': 'ticker',
    'DM_SCORE_COL':  'dm_smoothed',

    # HMS data — adjust table/column names to match your Supabase schema
    'HMS_TABLE':      'hms_daily',      # adjust to your HMS table name
    'HMS_DATE_COL':   'date',
    'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL':  'hms_score',      # adjust to your HMS score column name

    # Test parameters
    'LAGS':          [5, 10, 20, 40],   # trading days — test at each lag
    'MIN_OBS':       120,               # minimum overlapping observations
    'START_DATE':    '2016-01-01',
    'END_DATE':      '2026-03-25',
    'P_VALUE_THRESHOLD': 0.05,

    # Strategy 5 trades for universe
    'TRADES_FILE':   'trades_strategy_5_momentum_confirm_cf_65_5d_ma_.csv',
}


# ── SUPABASE CONNECTION ───────────────────────────────────────────────────────

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


# ── DATA LOADERS ──────────────────────────────────────────────────────────────

def load_dm_history(client, ticker, start_date, end_date):
    """Load daily DM scores for one ticker."""
    rows, offset, page = [], 0, 5000
    while True:
        r = (client.table(CONFIG['DM_TABLE'])
             .select(f"{CONFIG['DM_DATE_COL']},{CONFIG['DM_SCORE_COL']}")
             .eq(CONFIG['DM_TICKER_COL'], ticker)
             .gte(CONFIG['DM_DATE_COL'], str(start_date))
             .lte(CONFIG['DM_DATE_COL'], str(end_date))
             .order(CONFIG['DM_DATE_COL'])
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page:
            break
        offset += page

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['DM_DATE_COL']: 'date',
                       CONFIG['DM_SCORE_COL']: 'dm'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['dm']   = pd.to_numeric(df['dm'], errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


def load_hms_history(client, ticker, start_date, end_date):
    """Load daily HMS scores for one ticker."""
    rows, offset, page = [], 0, 5000
    while True:
        r = (client.table(CONFIG['HMS_TABLE'])
             .select(f"{CONFIG['HMS_DATE_COL']},{CONFIG['HMS_SCORE_COL']}")
             .eq(CONFIG['HMS_TICKER_COL'], ticker)
             .gte(CONFIG['HMS_DATE_COL'], str(start_date))
             .lte(CONFIG['HMS_DATE_COL'], str(end_date))
             .order(CONFIG['HMS_DATE_COL'])
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page:
            break
        offset += page

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['HMS_DATE_COL']: 'date',
                       CONFIG['HMS_SCORE_COL']: 'hms'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hms']  = pd.to_numeric(df['hms'], errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


# ── STATIONARITY ──────────────────────────────────────────────────────────────

def make_stationary(series):
    result = adfuller(series.dropna(), autolag='AIC')
    if result[1] > 0.05:
        return series.diff().dropna(), True
    return series, False


# ── GRANGER TEST ──────────────────────────────────────────────────────────────

def run_granger_test(x_series, y_series, label_x, label_y, lags):
    """
    Test whether x Granger-causes y.
    Returns dict of results at each lag.
    """
    # Align on common dates
    merged = pd.merge(
        x_series.rename(columns={x_series.columns[-1]: 'x'}),
        y_series.rename(columns={y_series.columns[-1]: 'y'}),
        on='date', how='inner'
    ).dropna()

    if len(merged) < CONFIG['MIN_OBS']:
        return None

    x_stat, _ = make_stationary(merged['x'])
    y_stat, _ = make_stationary(merged['y'])

    min_len = min(len(x_stat), len(y_stat))
    test_df = pd.DataFrame({
        'y': y_stat.iloc[-min_len:].values,
        'x': x_stat.iloc[-min_len:].values
    }).dropna()

    if len(test_df) < CONFIG['MIN_OBS']:
        return None

    results = {
        'direction': f'{label_x} → {label_y}',
        'n_obs':     len(test_df)
    }

    for lag in lags:
        if len(test_df) < lag * 3 + 10:
            results[f'p_lag{lag}'] = None
            results[f'f_lag{lag}'] = None
            results[f'sig_lag{lag}'] = None
            continue
        try:
            gc = grangercausalitytests(test_df[['y','x']], maxlag=lag, verbose=False)
            p  = gc[lag][0]['ssr_ftest'][1]
            f  = gc[lag][0]['ssr_ftest'][0]
            results[f'p_lag{lag}']   = round(p, 4)
            results[f'f_lag{lag}']   = round(f, 4)
            results[f'sig_lag{lag}'] = p < CONFIG['P_VALUE_THRESHOLD']
        except:
            results[f'p_lag{lag}']   = None
            results[f'f_lag{lag}']   = None
            results[f'sig_lag{lag}'] = None

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*65)
    print("  ARGUS — HMS PREDICTS DM? GRANGER CAUSALITY TEST")
    print("  Sequential Wave Hypothesis: Slow money leads fast money")
    print("="*65 + "\n")

    # Load universe
    if os.path.exists(CONFIG['TRADES_FILE']):
        trades  = pd.read_csv(CONFIG['TRADES_FILE'], encoding='latin-1')
        universe = sorted(trades['ticker'].unique().tolist())
        print(f"Loaded {len(universe)} tickers from Strategy 5 trade file.")
    else:
        print(f"Trade file not found. Will attempt full universe from DM table.")
        universe = None

    try:
        client = get_client()
        print("Connected to Supabase.\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if universe is None:
        r        = client.table(CONFIG['DM_TABLE']).select(CONFIG['DM_TICKER_COL']).execute()
        universe = sorted(set(row[CONFIG['DM_TICKER_COL']] for row in r.data))

    lags       = CONFIG['LAGS']
    start_date = CONFIG['START_DATE']
    end_date   = CONFIG['END_DATE']

    hms_to_dm_results = []  # HMS → DM (the hypothesis)
    dm_to_hms_results = []  # DM → HMS (the reverse — should be weaker)

    n = len(universe)
    print(f"Testing {n} tickers at lags {lags}...\n")
    print("Testing BOTH directions:")
    print("  HMS → DM  (does slow money predict fast money?)")
    print("  DM → HMS  (reverse — should be weaker if hypothesis is correct)\n")

    hms_missing = 0

    for i, ticker in enumerate(universe):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{n} — {ticker}")

        # Load both signals
        dm_df  = load_dm_history(client, ticker, start_date, end_date)
        hms_df = load_hms_history(client, ticker, start_date, end_date)

        if dm_df.empty:
            continue

        if hms_df.empty:
            hms_missing += 1
            continue

        # Prepare series with date column
        dm_series  = dm_df[['date','dm']].copy()
        hms_series = hms_df[['date','hms']].copy()

        # Test HMS → DM (the main hypothesis)
        r1 = run_granger_test(
            hms_series.rename(columns={'hms': 'hms'}),
            dm_series.rename(columns={'dm': 'dm'}),
            'HMS', 'DM', lags
        )
        if r1:
            r1['ticker'] = ticker
            hms_to_dm_results.append(r1)

        # Test DM → HMS (the reverse)
        r2 = run_granger_test(
            dm_series.rename(columns={'dm': 'dm'}),
            hms_series.rename(columns={'hms': 'hms'}),
            'DM', 'HMS', lags
        )
        if r2:
            r2['ticker'] = ticker
            dm_to_hms_results.append(r2)

    if hms_missing > 0:
        print(f"\nWARNING: {hms_missing} tickers had no HMS data.")
        print("Check that HMS_TABLE and HMS_SCORE_COL are correctly configured.")

    if not hms_to_dm_results:
        print("ERROR: No valid HMS→DM results. Check HMS data availability.")
        sys.exit(1)

    # Write CSVs
    df_hms_dm = pd.DataFrame(hms_to_dm_results)
    df_dm_hms = pd.DataFrame(dm_to_hms_results)

    df_hms_dm.to_csv('hms_dm_granger_results.csv', index=False)
    df_dm_hms.to_csv('dm_hms_granger_results.csv', index=False)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    lines = []
    def log(s=''):
        print(s)
        lines.append(s)

    log()
    log("="*65)
    log("  HMS → DM GRANGER CAUSALITY RESULTS")
    log("  Does slow money (HMS) predict fast money (DM)?")
    log("="*65)
    log(f"\n  Tickers tested: {len(df_hms_dm)}")
    log(f"  Lags tested:    {lags} trading days")
    log()

    log("  ── HMS → DM (Main Hypothesis) ─────────────────────────────")
    log("  % of tickers where HMS significantly Granger-causes DM:")
    log()
    for lag in lags:
        col = f'sig_lag{lag}'
        if col not in df_hms_dm.columns:
            continue
        valid = df_hms_dm[df_hms_dm[col].notna()]
        sig   = df_hms_dm[df_hms_dm[col] == True]
        pct   = len(sig)/len(valid)*100 if len(valid) > 0 else 0
        avg_p = df_hms_dm[f'p_lag{lag}'].dropna().mean()
        log(f"  Lag {lag:>2}d: {len(sig):>4}/{len(valid):<4} tickers ({pct:.0f}%) "
            f"significant  |  avg p-value: {avg_p:.4f}")

    log()
    log("  ── DM → HMS (Reverse — should be weaker) ──────────────────")
    log("  % of tickers where DM significantly Granger-causes HMS:")
    log()
    for lag in lags:
        col = f'sig_lag{lag}'
        if col not in df_dm_hms.columns:
            continue
        valid = df_dm_hms[df_dm_hms[col].notna()]
        sig   = df_dm_hms[df_dm_hms[col] == True]
        pct   = len(sig)/len(valid)*100 if len(valid) > 0 else 0
        avg_p = df_dm_hms[f'p_lag{lag}'].dropna().mean()
        log(f"  Lag {lag:>2}d: {len(sig):>4}/{len(valid):<4} tickers ({pct:.0f}%) "
            f"significant  |  avg p-value: {avg_p:.4f}")

    # Asymmetry analysis — the key finding
    log()
    log("="*65)
    log("  ASYMMETRY ANALYSIS — KEY FINDING")
    log("="*65)
    log()
    log("  The sequential wave hypothesis predicts:")
    log("  HMS→DM significance > DM→HMS significance")
    log("  Especially at longer lags (20-40 days)")
    log()

    for lag in lags:
        col = f'sig_lag{lag}'
        if col not in df_hms_dm.columns:
            continue
        v1 = df_hms_dm[df_hms_dm[col].notna()]
        v2 = df_dm_hms[df_dm_hms[col].notna()]
        s1 = df_hms_dm[df_hms_dm[col]==True]
        s2 = df_dm_hms[df_dm_hms[col]==True]
        p1 = len(s1)/len(v1)*100 if len(v1)>0 else 0
        p2 = len(s2)/len(v2)*100 if len(v2)>0 else 0
        arrow = "✓ HMS leads" if p1 > p2 else "✗ DM leads" if p2 > p1 else "= Equal"
        log(f"  Lag {lag:>2}d: HMS→DM {p1:.0f}%  vs  DM→HMS {p2:.0f}%  →  {arrow}")

    # Top tickers where HMS most strongly leads DM
    log()
    log("  ── TOP 10 TICKERS — Strongest HMS→DM Signal (Lag 20d) ────")
    if 'p_lag20' in df_hms_dm.columns:
        top = df_hms_dm.dropna(subset=['p_lag20']).nsmallest(10, 'p_lag20')
        for _, r in top.iterrows():
            sig_flags = []
            for lag in lags:
                if r.get(f'sig_lag{lag}'):
                    sig_flags.append(f"{lag}d")
            log(f"  {r['ticker']:<6}  HMS→DM p={r['p_lag20']:.4f}  "
                f"significant at: {', '.join(sig_flags) if sig_flags else 'none'}")

    # Interpretation
    log()
    log("="*65)
    log("  INTERPRETATION & IMPLICATIONS")
    log("="*65)

    if 'sig_lag20' in df_hms_dm.columns and 'sig_lag20' in df_dm_hms.columns:
        v1  = df_hms_dm[df_hms_dm['sig_lag20'].notna()]
        v2  = df_dm_hms[df_dm_hms['sig_lag20'].notna()]
        p1  = len(df_hms_dm[df_hms_dm['sig_lag20']==True])/len(v1)*100 if len(v1)>0 else 0
        p2  = len(df_dm_hms[df_dm_hms['sig_lag20']==True])/len(v2)*100 if len(v2)>0 else 0

        log()
        if p1 >= 30 and p1 > p2 * 1.3:
            log(f"  ✓✓ STRONG CONFIRMATION: Sequential wave hypothesis supported.")
            log(f"  HMS Granger-causes DM ({p1:.0f}%) significantly more than")
            log(f"  DM Granger-causes HMS ({p2:.0f}%) at 20-day lag.")
            log()
            log("  IMPLICATION: HMS is a leading indicator of DM.")
            log("  Slow institutional money (HMS) moves first.")
            log("  Fast institutional money (DM) follows weeks later.")
            log("  Price confirms after both waves have passed.")
            log()
            log("  REVISED ENTRY RULE (to test):")
            log("  Enter when: HMS elevated AND DM just crossing 65 on rising MA")
            log("  This catches the trade as fast money joins slow money.")
            log("  Exit: MA20 < 50 (unchanged)")
        elif p1 >= 20 and p1 > p2:
            log(f"  ✓  MODERATE: Some support for sequential wave hypothesis.")
            log(f"  HMS→DM ({p1:.0f}%) moderately stronger than DM→HMS ({p2:.0f}%)")
            log("  Hypothesis partially confirmed — worth further investigation.")
        else:
            log(f"  ✗  WEAK/NOT CONFIRMED at 20-day lag.")
            log(f"  HMS→DM ({p1:.0f}%) not clearly stronger than DM→HMS ({p2:.0f}%)")
            log("  HMS and DM may be measuring concurrent rather than sequential phenomena.")
            log("  Consider testing at longer lags (40-60 days).")

    log()
    log("  NOTE: Even if hypothesis is not confirmed at these lags,")
    log("  HMS may still add value as a universe pre-selector.")
    log("  The simulation tested HMS as a simultaneous filter — not")
    log("  as a sequential leading indicator. These are different uses.")
    log()
    log("="*65)

    summary = '\n'.join(lines)
    with open('hms_dm_granger_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\nResults written to:")
    print(f"  hms_dm_granger_results.csv  (HMS→DM per ticker)")
    print(f"  dm_hms_granger_results.csv  (DM→HMS per ticker)")
    print(f"  hms_dm_granger_summary.txt  (summary + interpretation)")
    print("\nDONE.\n")


if __name__ == '__main__':
    run()
