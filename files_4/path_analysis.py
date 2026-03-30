"""
ARGUS — PATH ANALYSIS: HMS → DM → RETURNS
==========================================
Structural Equation Model testing the sequential wave hypothesis.

THE MODEL:
                    β₁                    β₃
    HMS_{t-k}  ──────────►  DM_t  ──────────►  Return_{t+n}
                                    ↑
                            β₄ (direct path)
    HMS_{t-k}  ─────────────────────────────►  Return_{t+n}

Two structural equations:

  Equation 1 (DM equation):
    DM_t = α₁ + β₁·HMS_{t-k} + β₂·DM_{t-1} + ε₁

  Equation 2 (Returns equation):
    Return_{t+n} = α₂ + β₃·DM_t + β₄·HMS_{t-k} + β₅·Return_{t-1} + ε₂

KEY TESTS:
  β₁ significant  → HMS predicts DM (Wave 1 → Wave 2)
  β₃ significant  → DM predicts returns (Wave 2 → Wave 3)
  β₄ not significant → HMS has no DIRECT effect on returns beyond DM
                        (DM fully mediates HMS effect — complete mediation)
  β₄ significant  → HMS has independent effect on returns beyond DM
                        (partial mediation — HMS adds its own signal)

MEDIATION ANALYSIS (Sobel test):
  Indirect effect = β₁ × β₃
  Standard error of indirect effect via delta method
  Sobel z-statistic tests whether indirect path is significant

INTERPRETATION:
  Complete mediation (β₄ not sig, β₁×β₃ sig):
    → HMS predicts returns ONLY through DM
    → DM fully captures the HMS information
    → Use DM for timing, HMS for universe selection
    → Sequential wave hypothesis fully confirmed

  Partial mediation (β₄ sig, β₁×β₃ sig):
    → HMS has both direct and indirect effects
    → HMS and DM carry complementary information
    → Use both signals together — additive value
    → Sequential wave hypothesis partially confirmed

  No mediation (β₁ not sig):
    → HMS does not predict DM
    → Signals are independent or concurrent
    → Revisit hypothesis at different lag structures

Developer notes:
  - Requires daily HMS scores, DM scores, and price returns per ticker
  - k = lag for HMS (test at 10, 20, 40 days)
  - n = forward return horizon (test at 20, 40, 60 days)
  - Uses OLS regression (statsmodels) — no exotic dependencies
  - Sobel test computed manually from path coefficients
  - Run time: 20-40 minutes
  - Output: path_analysis_results.csv + path_analysis_summary.txt

Required: pip install statsmodels pandas numpy scipy
"""

import os
import sys
import csv
import warnings
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv
from scipy import stats

warnings.filterwarnings('ignore')

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase")
    sys.exit(1)

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    print("ERROR: pip install statsmodels")
    sys.exit(1)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # DM data
    'DM_TABLE':       'dm_history',
    'DM_DATE_COL':    'date',
    'DM_TICKER_COL':  'ticker',
    'DM_SCORE_COL':   'dm_smoothed',
    'DM_PRICE_COL':   'close',

    # HMS data — adjust to match your Supabase schema
    'HMS_TABLE':      'hms_daily',
    'HMS_DATE_COL':   'date',
    'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL':  'hms_score',

    # Path analysis parameters
    'HMS_LAGS':       [10, 20, 40],     # k: days HMS leads DM
    'RETURN_HORIZONS': [20, 40, 60],    # n: forward return days
    'MIN_OBS':        150,
    'START_DATE':     '2016-01-01',
    'END_DATE':       '2026-03-25',
    'P_VALUE_THRESHOLD': 0.05,

    # Universe
    'TRADES_FILE':    'trades_strategy_5_momentum_confirm_cf_65_5d_ma_.csv',
}


# ── SUPABASE ──────────────────────────────────────────────────────────────────

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY required in .env")
    return create_client(url, key)


# ── DATA LOADERS ──────────────────────────────────────────────────────────────

def load_ticker_data(client, ticker, start_date, end_date):
    """Load daily DM scores and prices for one ticker."""
    rows, offset, page = [], 0, 5000
    while True:
        r = (client.table(CONFIG['DM_TABLE'])
             .select(f"{CONFIG['DM_DATE_COL']},"
                     f"{CONFIG['DM_SCORE_COL']},"
                     f"{CONFIG['DM_PRICE_COL']}")
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
    df.rename(columns={
        CONFIG['DM_DATE_COL']:   'date',
        CONFIG['DM_SCORE_COL']:  'dm',
        CONFIG['DM_PRICE_COL']:  'close'
    }, inplace=True)
    df['date']  = pd.to_datetime(df['date']).dt.date
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


def load_hms_data(client, ticker, start_date, end_date):
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
    df.rename(columns={
        CONFIG['HMS_DATE_COL']:  'date',
        CONFIG['HMS_SCORE_COL']: 'hms'
    }, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hms']  = pd.to_numeric(df['hms'], errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


# ── STATIONARITY ──────────────────────────────────────────────────────────────

def make_stationary(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    if result[1] > 0.05:
        return series.diff().dropna(), True
    return series.dropna(), False


# ── SOBEL TEST ────────────────────────────────────────────────────────────────

def sobel_test(b1, se1, b3, se3):
    """
    Sobel test for significance of indirect path β₁ × β₃.
    Tests whether the mediated (indirect) effect is significant.
    Returns z-statistic and p-value.
    """
    indirect_effect = b1 * b3
    se_indirect     = np.sqrt(b3**2 * se1**2 + b1**2 * se3**2)
    if se_indirect == 0:
        return 0, 1.0
    z = indirect_effect / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# ── PATH ANALYSIS FOR ONE TICKER ──────────────────────────────────────────────

def run_path_analysis(dm_df, hms_df, ticker, hms_lag, return_horizon):
    """
    Run the full path analysis for one ticker at one lag/horizon combination.

    Equation 1: DM_t = α₁ + β₁·HMS_{t-k} + β₂·DM_{t-1} + ε₁
    Equation 2: Return_{t+n} = α₂ + β₃·DM_t + β₄·HMS_{t-k} + β₅·Return_{t-1} + ε₂
    """
    # Merge DM and HMS
    merged = pd.merge(
        dm_df[['date','dm','close']],
        hms_df[['date','hms']],
        on='date', how='inner'
    ).sort_values('date').reset_index(drop=True)

    if len(merged) < CONFIG['MIN_OBS'] + hms_lag + return_horizon:
        return None

    # Compute log returns
    merged['return'] = np.log(merged['close']).diff()

    # Create lagged variables
    merged['hms_lag']     = merged['hms'].shift(hms_lag)
    merged['dm_lag1']     = merged['dm'].shift(1)
    merged['return_lag1'] = merged['return'].shift(1)
    merged['return_fwd']  = merged['return'].shift(-return_horizon)

    merged = merged.dropna()

    if len(merged) < CONFIG['MIN_OBS']:
        return None

    # Make stationary where needed
    dm_s,  _  = make_stationary(merged['dm'])
    hms_s, _  = make_stationary(merged['hms_lag'])
    ret_s, _  = make_stationary(merged['return'])
    ret_f, _  = make_stationary(merged['return_fwd'])

    # Align all series to same length
    min_len = min(len(dm_s), len(hms_s), len(ret_s), len(ret_f),
                  len(merged['dm_lag1'].dropna()),
                  len(merged['return_lag1'].dropna()))

    if min_len < CONFIG['MIN_OBS']:
        return None

    # Build analysis DataFrame
    n = min_len
    df = pd.DataFrame({
        'dm':         dm_s.iloc[-n:].values,
        'hms_lag':    hms_s.iloc[-n:].values,
        'dm_lag1':    merged['dm_lag1'].dropna().iloc[-n:].values,
        'ret':        ret_s.iloc[-n:].values,
        'ret_lag1':   merged['return_lag1'].dropna().iloc[-n:].values,
        'ret_fwd':    ret_f.iloc[-n:].values,
    }).dropna()

    if len(df) < CONFIG['MIN_OBS']:
        return None

    result = {
        'ticker':         ticker,
        'hms_lag':        hms_lag,
        'return_horizon': return_horizon,
        'n_obs':          len(df),
    }

    # ── EQUATION 1: DM = α₁ + β₁·HMS_lag + β₂·DM_lag1 ──────────────────────
    try:
        X1 = sm.add_constant(df[['hms_lag', 'dm_lag1']])
        y1 = df['dm']
        eq1 = sm.OLS(y1, X1).fit()

        b1    = eq1.params.get('hms_lag', np.nan)
        se1   = eq1.bse.get('hms_lag', np.nan)
        p_b1  = eq1.pvalues.get('hms_lag', np.nan)
        r2_eq1 = eq1.rsquared

        result.update({
            'b1_hms_to_dm':    round(b1,  4) if not np.isnan(b1)   else None,
            'se1':             round(se1, 4) if not np.isnan(se1)  else None,
            'p_b1':            round(p_b1,4) if not np.isnan(p_b1) else None,
            'sig_b1':          p_b1 < CONFIG['P_VALUE_THRESHOLD'] if not np.isnan(p_b1) else None,
            'r2_eq1':          round(r2_eq1, 4),
        })
    except Exception as e:
        result.update({'b1_hms_to_dm': None, 'se1': None, 'p_b1': None,
                       'sig_b1': None, 'r2_eq1': None})
        b1, se1 = np.nan, np.nan

    # ── EQUATION 2: Return = α₂ + β₃·DM + β₄·HMS_lag + β₅·ret_lag1 ─────────
    try:
        X2 = sm.add_constant(df[['dm', 'hms_lag', 'ret_lag1']])
        y2 = df['ret_fwd']
        eq2 = sm.OLS(y2, X2).fit()

        b3    = eq2.params.get('dm',      np.nan)
        se3   = eq2.bse.get('dm',         np.nan)
        p_b3  = eq2.pvalues.get('dm',     np.nan)
        b4    = eq2.params.get('hms_lag', np.nan)
        se4   = eq2.bse.get('hms_lag',    np.nan)
        p_b4  = eq2.pvalues.get('hms_lag',np.nan)
        r2_eq2 = eq2.rsquared

        result.update({
            'b3_dm_to_ret':    round(b3,  4) if not np.isnan(b3)   else None,
            'se3':             round(se3, 4) if not np.isnan(se3)  else None,
            'p_b3':            round(p_b3,4) if not np.isnan(p_b3) else None,
            'sig_b3':          p_b3 < CONFIG['P_VALUE_THRESHOLD'] if not np.isnan(p_b3) else None,
            'b4_hms_direct':   round(b4,  4) if not np.isnan(b4)   else None,
            'p_b4':            round(p_b4,4) if not np.isnan(p_b4) else None,
            'sig_b4_direct':   p_b4 < CONFIG['P_VALUE_THRESHOLD'] if not np.isnan(p_b4) else None,
            'r2_eq2':          round(r2_eq2, 4),
        })
    except Exception as e:
        result.update({'b3_dm_to_ret': None, 'se3': None, 'p_b3': None,
                       'sig_b3': None, 'b4_hms_direct': None, 'p_b4': None,
                       'sig_b4_direct': None, 'r2_eq2': None})
        b3, se3 = np.nan, np.nan

    # ── SOBEL TEST: Indirect path β₁ × β₃ ───────────────────────────────────
    if not np.isnan(b1) and not np.isnan(b3) and \
       not np.isnan(se1) and not np.isnan(se3):
        z_sobel, p_sobel = sobel_test(b1, se1, b3, se3)
        indirect_effect  = b1 * b3
        result.update({
            'indirect_effect':  round(indirect_effect, 6),
            'sobel_z':          round(z_sobel, 4),
            'sobel_p':          round(p_sobel, 4),
            'sig_indirect':     p_sobel < CONFIG['P_VALUE_THRESHOLD'],
        })
    else:
        result.update({
            'indirect_effect': None, 'sobel_z': None,
            'sobel_p': None, 'sig_indirect': None
        })

    # ── MEDIATION TYPE ────────────────────────────────────────────────────────
    sig_b1  = result.get('sig_b1')
    sig_b3  = result.get('sig_b3')
    sig_b4  = result.get('sig_b4_direct')
    sig_ind = result.get('sig_indirect')

    if sig_b1 and sig_b3 and sig_ind:
        if not sig_b4:
            mediation = 'COMPLETE'    # HMS → DM → Returns, no direct HMS effect
        else:
            mediation = 'PARTIAL'     # Both direct and indirect paths significant
    elif sig_b1 and not sig_b3:
        mediation = 'NO_DM_EFFECT'   # HMS predicts DM but DM doesn't predict returns
    elif not sig_b1:
        mediation = 'NO_MEDIATION'   # HMS doesn't predict DM
    else:
        mediation = 'UNCLEAR'

    result['mediation_type'] = mediation

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*65)
    print("  ARGUS — PATH ANALYSIS")
    print("  HMS → DM → Returns: Structural Equation Model")
    print("  Sequential Wave Hypothesis Test")
    print("="*65 + "\n")

    # Load universe
    if os.path.exists(CONFIG['TRADES_FILE']):
        trades   = pd.read_csv(CONFIG['TRADES_FILE'], encoding='latin-1')
        universe = sorted(trades['ticker'].unique().tolist())
        print(f"Loaded {len(universe)} tickers from Strategy 5 trade file.")
    else:
        print("Trade file not found.")
        sys.exit(1)

    try:
        client = get_client()
        print("Connected to Supabase.\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    start_date = CONFIG['START_DATE']
    end_date   = CONFIG['END_DATE']
    hms_lags   = CONFIG['HMS_LAGS']
    horizons   = CONFIG['RETURN_HORIZONS']

    all_results = []
    n = len(universe)
    hms_missing = 0

    print(f"Running path analysis for {n} tickers...")
    print(f"HMS lags: {hms_lags} days | Return horizons: {horizons} days\n")

    for i, ticker in enumerate(universe):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{n} — {ticker}")

        dm_df  = load_ticker_data(client, ticker, start_date, end_date)
        hms_df = load_hms_data(client, ticker, start_date, end_date)

        if dm_df.empty:
            continue
        if hms_df.empty:
            hms_missing += 1
            continue

        # Test all lag/horizon combinations
        for hms_lag in hms_lags:
            for horizon in horizons:
                result = run_path_analysis(dm_df, hms_df, ticker, hms_lag, horizon)
                if result:
                    all_results.append(result)

    if hms_missing > 0:
        print(f"\nWARNING: {hms_missing} tickers had no HMS data.")

    if not all_results:
        print("ERROR: No valid results produced.")
        sys.exit(1)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv('path_analysis_results.csv', index=False)
    print(f"\nDetailed results written to path_analysis_results.csv")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    lines = []
    def log(s=''):
        print(s)
        lines.append(s)

    log()
    log("="*65)
    log("  PATH ANALYSIS SUMMARY")
    log("  HMS → DM → Returns: Sequential Wave Hypothesis")
    log("="*65)
    log(f"\n  Total path analyses run: {len(df_results)}")
    log(f"  Tickers analyzed:        {df_results['ticker'].nunique()}")
    log(f"  Lag/horizon combinations: {len(hms_lags)} × {len(horizons)}")

    # Results by lag/horizon combination
    for hms_lag in hms_lags:
        for horizon in horizons:
            sub = df_results[
                (df_results['hms_lag'] == hms_lag) &
                (df_results['return_horizon'] == horizon)
            ]
            if len(sub) == 0:
                continue

            log()
            log(f"  ── HMS lag {hms_lag}d → Return horizon {horizon}d ──────────────────")
            log(f"     Tickers: {len(sub)}")

            # Path 1: HMS → DM (β₁)
            sig_b1 = sub[sub['sig_b1']==True]
            log(f"     β₁ HMS→DM significant:      {len(sig_b1)}/{len(sub)} "
                f"({len(sig_b1)/len(sub)*100:.0f}%)")

            # Path 2: DM → Returns (β₃)
            sig_b3 = sub[sub['sig_b3']==True]
            log(f"     β₃ DM→Returns significant:  {len(sig_b3)}/{len(sub)} "
                f"({len(sig_b3)/len(sub)*100:.0f}%)")

            # Direct path: HMS → Returns (β₄)
            sig_b4 = sub[sub['sig_b4_direct']==True]
            log(f"     β₄ HMS direct significant:  {len(sig_b4)}/{len(sub)} "
                f"({len(sig_b4)/len(sub)*100:.0f}%)")

            # Indirect (Sobel)
            sig_ind = sub[sub['sig_indirect']==True]
            log(f"     Sobel indirect significant: {len(sig_ind)}/{len(sub)} "
                f"({len(sig_ind)/len(sub)*100:.0f}%)")

            # Mediation type breakdown
            med_counts = sub['mediation_type'].value_counts()
            log(f"     Mediation types:")
            for med_type, count in med_counts.items():
                pct = count/len(sub)*100
                marker = " ← KEY FINDING" if med_type == 'COMPLETE' and pct >= 20 else ""
                log(f"       {med_type:<20}: {count:>4} ({pct:.0f}%){marker}")

    # Top complete mediation tickers
    complete = df_results[df_results['mediation_type']=='COMPLETE']
    if len(complete) > 0:
        log()
        log("  ── COMPLETE MEDIATION TICKERS (strongest examples) ────────")
        log("  (HMS effect on returns flows entirely through DM)")
        top_complete = complete.nsmallest(15, 'sobel_p')[
            ['ticker','hms_lag','return_horizon','b1_hms_to_dm',
             'b3_dm_to_ret','indirect_effect','sobel_p','mediation_type']
        ]
        for _, r in top_complete.iterrows():
            log(f"  {r['ticker']:<6} lag{r['hms_lag']:>2}d/fwd{r['return_horizon']:>2}d | "
                f"β₁={r['b1_hms_to_dm']:>+.3f} "
                f"β₃={r['b3_dm_to_ret']:>+.3f} "
                f"indirect={r['indirect_effect']:>+.4f} "
                f"Sobel p={r['sobel_p']:.4f}")

    # Overall interpretation
    log()
    log("="*65)
    log("  INTERPRETATION")
    log("="*65)

    # Focus on 20-day lag, 40-day horizon as primary test
    primary = df_results[
        (df_results['hms_lag']==20) &
        (df_results['return_horizon']==40)
    ]

    if len(primary) > 0:
        complete_pct = (primary['mediation_type']=='COMPLETE').sum() / len(primary) * 100
        partial_pct  = (primary['mediation_type']=='PARTIAL').sum()  / len(primary) * 100
        b1_sig_pct   = primary['sig_b1'].sum() / len(primary) * 100
        b3_sig_pct   = primary['sig_b3'].sum() / len(primary) * 100
        b4_sig_pct   = primary['sig_b4_direct'].sum() / len(primary) * 100

        log()
        log(f"  Primary test: HMS lag 20d → DM → Returns fwd 40d")
        log(f"  ({len(primary)} tickers)")
        log()
        log(f"  β₁ (HMS→DM):        {b1_sig_pct:.0f}% of tickers significant")
        log(f"  β₃ (DM→Returns):    {b3_sig_pct:.0f}% of tickers significant")
        log(f"  β₄ (HMS direct):    {b4_sig_pct:.0f}% of tickers significant")
        log(f"  Complete mediation: {complete_pct:.0f}% of tickers")
        log(f"  Partial mediation:  {partial_pct:.0f}% of tickers")
        log()

        if complete_pct >= 25:
            log("  ✓✓ STRONG: Sequential wave hypothesis CONFIRMED")
            log("  HMS effect on returns flows through DM in 25%+ of tickers.")
            log("  DM fully mediates the HMS signal in these cases.")
            log()
            log("  IMPLICATION FOR SYSTEM ARCHITECTURE:")
            log("  HMS is Wave 1 (slow money detection)")
            log("  DM is Wave 2 (fast money confirmation)")
            log("  Price is Wave 3 (public confirmation)")
            log()
            log("  REVISED ENTRY RULE:")
            log("  Enter when: HMS elevated (Wave 1 detected) +")
            log("              DM >= 65 on rising 5d MA (Wave 2 confirmed) +")
            log("              Price > 20d price MA (momentum confirmed)")
            log("  This entry catches the trade at the Wave 1→Wave 2 transition.")
        elif complete_pct + partial_pct >= 25:
            log("  ✓  MODERATE: Sequential wave hypothesis PARTIALLY CONFIRMED")
            log("  HMS adds value but both direct and indirect paths significant.")
            log("  Use HMS + DM as complementary signals, not sequential gates.")
        else:
            log("  ✗  WEAK at primary test parameters.")
            log("  Check results at other lag/horizon combinations above.")
            log("  HMS and DM may operate at different time horizons than tested.")

    log()
    log("  PATH COEFFICIENT INTERPRETATION:")
    log("  β₁ > 0: HMS elevation predicts DM elevation k days later")
    log("  β₃ > 0: DM elevation predicts positive returns n days later")
    log("  β₄ ≈ 0: HMS has no effect on returns beyond what DM captures")
    log("  Indirect = β₁×β₃: The mediated effect (the chain)")
    log("  Sobel p < 0.05: The chain is statistically significant")
    log()
    log("="*65)

    summary = '\n'.join(lines)
    with open('path_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\nSummary written to path_analysis_summary.txt")
    print("\nOutput files:")
    print("  path_analysis_results.csv   — per ticker, per lag/horizon")
    print("  path_analysis_summary.txt   — interpreted summary")
    print("\nDONE.\n")


if __name__ == '__main__':
    run()
