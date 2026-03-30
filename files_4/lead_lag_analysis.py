"""
ARGUS CAPITAL FLOW — LEAD-LAG ANALYSIS
=======================================
Purpose: Determine whether the DM signal leads price movement or merely
         confirms it (momentum criticism test).

For each trade entry in Strategy 5:
  1. Find when DM score first crossed 65 (with 5d MA rising) — the "signal date"
  2. Find when price first made a meaningful move above entry price — the "price confirmation date"
  3. Compute lead time = price confirmation date - signal date

If DM consistently leads price by days or weeks, the signal is detecting
institutional accumulation BEFORE price confirms. This counters the
"it's just momentum" criticism.

If lead time is near zero or negative, the signal is reacting to price,
not anticipating it.

Developer notes:
- Adjust CONFIG field names to match your Supabase column names
- This uses the same dm_history table as the simulation scripts
- Requires daily price history — use the 'close' column
- Run time: ~10-30 minutes depending on universe size
- Output: lead_lag_results.csv + console summary

Output files:
  lead_lag_results.csv     — one row per trade with lead time
  lead_lag_summary.txt     — statistical summary for presentation
"""

import os
import sys
import csv
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase pandas numpy python-dotenv")
    sys.exit(1)

load_dotenv()

# ── CONFIG — adjust field names to match your Supabase ───────────────────────
CONFIG = {
    'DM_TABLE':      'dm_history',      # adjust to your table name
    'DM_DATE_COL':   'date',
    'DM_TICKER_COL': 'ticker',
    'DM_PRICE_COL':  'close',
    'DM_SCORE_COL':  'dm_smoothed',     # the DM score column

    # Lead-lag parameters
    'DM_ENTRY_THRESHOLD': 65,           # same as Strategy 5 entry
    'PRICE_CONFIRMATION_PCT': 5.0,      # price must rise >5% above entry price
    'LOOKBACK_DAYS': 252,                # how far back to look for DM crossing
    'LOOKAHEAD_DAYS': 120,              # how far forward to look for price confirmation

    # Strategy 5 trade file — path to the CSV from the simulation run
    'TRADES_FILE': 'trades_strategy_5_momentum_confirm_cf_65_5d_ma_.csv',
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
    """
    Load daily DM score and price for one ticker over a date range.
    Returns DataFrame: date, close, dm
    """
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
    df.rename(columns={date_col:'date', price_col:'close', dm_col:'dm'}, inplace=True)
    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna().sort_values('date').reset_index(drop=True)
    return df


# ── SIGNAL CROSSING FINDER ────────────────────────────────────────────────────

def find_first_signal_crossing(df, entry_date, lookback_days, threshold):
    """
    Working backwards from entry_date, find the first date when DM crossed
    above threshold with 5d MA rising.

    Returns the date of first crossing, or None if not found.
    """
    cutoff = entry_date - timedelta(days=lookback_days)
    window = df[(df['date'] >= cutoff) & (df['date'] <= entry_date)].copy()

    if len(window) < 6:
        return None

    # Compute 5d MA of DM
    window = window.sort_values('date').reset_index(drop=True)
    window['dm_ma5']      = window['dm'].rolling(5, min_periods=1).mean()
    window['dm_ma5_prev'] = window['dm_ma5'].shift(1)

    # Find rows where DM >= threshold AND 5d MA rising
    signal_rows = window[
        (window['dm'] >= threshold) &
        (window['dm_ma5'] > window['dm_ma5_prev'].fillna(0))
    ]

    if signal_rows.empty:
        return None

    # Return the FIRST crossing date
    return signal_rows.iloc[0]['date']


# ── PRICE CONFIRMATION FINDER ─────────────────────────────────────────────────

def find_price_confirmation(df, entry_date, entry_price, confirmation_pct, lookahead_days):
    """
    After entry_date, find the first date when price exceeded entry_price by
    more than confirmation_pct%.

    Returns the date of price confirmation, or None if not found within lookahead.
    """
    target_price = entry_price * (1 + confirmation_pct / 100)
    cutoff = entry_date + timedelta(days=lookahead_days)

    window = df[(df['date'] > entry_date) & (df['date'] <= cutoff)]

    confirmed = window[window['close'] >= target_price]
    if confirmed.empty:
        return None

    return confirmed.iloc[0]['date']


# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*65)
    print("  ARGUS LEAD-LAG ANALYSIS")
    print("  Does the DM signal lead price or follow it?")
    print("="*65 + "\n")

    # Load trades
    trades_file = CONFIG['TRADES_FILE']
    if not os.path.exists(trades_file):
        print(f"ERROR: Trades file not found: {trades_file}")
        print("Place the Strategy 5 trades CSV in the same directory as this script.")
        sys.exit(1)

    trades = pd.read_csv(trades_file, encoding='latin-1')
    trades['entry_date'] = pd.to_datetime(trades['entry_date']).dt.date
    trades['exit_date']  = pd.to_datetime(trades['exit_date']).dt.date
    trades['entry_price'] = pd.to_numeric(trades['entry_price'], errors='coerce')

    print(f"Loaded {len(trades):,} trades from {trades_file}")

    try:
        client = get_client()
        print("Connected to Supabase.\n")
    except Exception as e:
        print(f"ERROR connecting to Supabase: {e}")
        sys.exit(1)

    threshold       = CONFIG['DM_ENTRY_THRESHOLD']
    lookback        = CONFIG['LOOKBACK_DAYS']
    lookahead       = CONFIG['LOOKAHEAD_DAYS']
    confirm_pct     = CONFIG['PRICE_CONFIRMATION_PCT']

    results = []
    tickers_loaded  = {}  # cache ticker history
    n = len(trades)

    for i, row in trades.iterrows():
        ticker      = row['ticker']
        entry_date  = row['entry_date']
        entry_price = row['entry_price']
        return_pct  = row['return_pct']

        if i % 50 == 0:
            print(f"  Processing trade {i+1}/{n} — {ticker} {entry_date}")

        # Load ticker history if not cached
        if ticker not in tickers_loaded:
            load_start = entry_date - timedelta(days=lookback + 30)
            load_end   = entry_date + timedelta(days=lookahead + 30)
            df = load_ticker_history(client, ticker, load_start, load_end)
            tickers_loaded[ticker] = df
        else:
            df = tickers_loaded[ticker]

        if df.empty:
            results.append({
                'ticker': ticker, 'entry_date': entry_date,
                'entry_price': entry_price, 'return_pct': return_pct,
                'signal_date': None, 'confirm_date': None,
                'lead_days': None, 'confirm_days': None,
                'status': 'NO_DATA'
            })
            continue

        # Find first signal crossing (looking back from entry)
        signal_date = find_first_signal_crossing(df, entry_date, lookback, threshold)

        # Find price confirmation (looking forward from entry)
        confirm_date = find_price_confirmation(df, entry_date, entry_price, confirm_pct, lookahead)

        # Lead time = entry_date - signal_date
        # Positive = signal fired before we entered (we entered late)
        # Zero = signal fired on entry day
        lead_days = (entry_date - signal_date).days if signal_date else None

        # Confirmation days = how long after entry until price confirmed
        confirm_days = (confirm_date - entry_date).days if confirm_date else None

        results.append({
            'ticker':        ticker,
            'entry_date':    entry_date,
            'entry_price':   entry_price,
            'return_pct':    return_pct,
            'signal_date':   signal_date,
            'confirm_date':  confirm_date,
            'lead_days':     lead_days,
            'confirm_days':  confirm_days,
            'status':        'OK' if signal_date else 'NO_SIGNAL_FOUND'
        })

    # Write results CSV
    out_file = 'lead_lag_results.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {out_file}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    valid = df_results[df_results['lead_days'].notna()]
    confirmed = df_results[df_results['confirm_days'].notna()]

    summary_lines = []
    summary_lines.append("="*65)
    summary_lines.append("  LEAD-LAG ANALYSIS RESULTS")
    summary_lines.append("="*65)
    summary_lines.append(f"\n  Total trades analyzed:     {len(df_results):,}")
    summary_lines.append(f"  Signal date found:         {len(valid):,} ({len(valid)/len(df_results)*100:.0f}%)")
    summary_lines.append(f"  Price confirmed (>{confirm_pct}%):    {len(confirmed):,} ({len(confirmed)/len(df_results)*100:.0f}%)")

    if len(valid) > 0:
        summary_lines.append(f"\n  ── SIGNAL LEAD TIME ───────────────────────────────")
        summary_lines.append(f"  (Days from first DM crossing to trade entry)")
        summary_lines.append(f"  Mean lead days:   {valid['lead_days'].mean():.1f}")
        summary_lines.append(f"  Median lead days: {valid['lead_days'].median():.1f}")
        summary_lines.append(f"  Min lead days:    {valid['lead_days'].min():.0f}")
        summary_lines.append(f"  Max lead days:    {valid['lead_days'].max():.0f}")
        summary_lines.append(f"\n  Lead distribution:")
        summary_lines.append(f"    0 days (entered on signal day):   {(valid['lead_days']==0).sum()}")
        summary_lines.append(f"    1-5 days after signal:            {((valid['lead_days']>0)&(valid['lead_days']<=5)).sum()}")
        summary_lines.append(f"    6-14 days after signal:           {((valid['lead_days']>5)&(valid['lead_days']<=14)).sum()}")
        summary_lines.append(f"    15-30 days after signal:          {((valid['lead_days']>14)&(valid['lead_days']<=30)).sum()}")
        summary_lines.append(f"    >30 days after signal:            {(valid['lead_days']>30).sum()}")

    if len(confirmed) > 0:
        summary_lines.append(f"\n  ── PRICE CONFIRMATION TIME ────────────────────────")
        summary_lines.append(f"  (Days from entry until price rose >{confirm_pct}%)")
        summary_lines.append(f"  Mean confirm days:   {confirmed['confirm_days'].mean():.1f}")
        summary_lines.append(f"  Median confirm days: {confirmed['confirm_days'].median():.1f}")

        # Winners vs losers
        win_conf  = confirmed[confirmed['return_pct'] > 0]
        loss_conf = confirmed[confirmed['return_pct'] <= 0]
        summary_lines.append(f"\n  Confirmed on winning trades:  {len(win_conf):,}")
        summary_lines.append(f"  Confirmed on losing trades:   {len(loss_conf):,}")

    # Top 10 examples by longest lead time
    summary_lines.append(f"\n  ── TOP 10 TRADES WITH LONGEST SIGNAL LEAD ─────────")
    summary_lines.append(f"  (These are the strongest lead-lag examples)")
    if len(valid) > 0:
        top_leads = valid.nlargest(10, 'lead_days')[
            ['ticker','signal_date','entry_date','lead_days','return_pct']
        ]
        for _, r in top_leads.iterrows():
            summary_lines.append(
                f"  {r['ticker']:<6} Signal: {r['signal_date']}  "
                f"Entry: {r['entry_date']}  "
                f"Lead: {r['lead_days']:>3}d  "
                f"Return: {r['return_pct']:>+.1f}%"
            )

    # Early signal examples (entered on or 1 day after signal)
    early = valid[valid['lead_days'] <= 1]
    summary_lines.append(f"\n  ── EARLY SIGNAL ENTRIES (lead ≤ 1 day) ───────────")
    summary_lines.append(f"  {len(early)} trades entered within 1 day of first DM crossing")
    summary_lines.append(f"  These are the purest 'signal before price' examples")

    summary_lines.append(f"\n{'='*65}")
    summary_lines.append("  MOMENTUM CRITICISM ASSESSMENT")
    summary_lines.append("="*65)
    if len(valid) > 10:
        med_lead = valid['lead_days'].median()
        if med_lead <= 1:
            summary_lines.append(f"\n  ⚠️  MEDIAN LEAD = {med_lead:.0f} days")
            summary_lines.append("  The signal typically fires on or after the DM crossing.")
            summary_lines.append("  This is consistent with momentum criticism.")
            summary_lines.append("  Look at the distribution — are there meaningful early cases?")
        elif med_lead <= 7:
            summary_lines.append(f"\n  ✓  MEDIAN LEAD = {med_lead:.0f} days")
            summary_lines.append("  Signal fires ~1 week before entry on average.")
            summary_lines.append("  Partial counter to momentum criticism.")
        else:
            summary_lines.append(f"\n  ✓✓  MEDIAN LEAD = {med_lead:.0f} days")
            summary_lines.append("  Signal fires well before entry on average.")
            summary_lines.append("  Strong counter to momentum criticism.")

    summary = "\n".join(summary_lines)
    print(summary)

    with open('lead_lag_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\nSummary written to lead_lag_summary.txt")
    print("DONE.\n")


if __name__ == '__main__':
    run()
