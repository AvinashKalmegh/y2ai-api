"""
T1C SIMULATION CORE ENGINE
===========================
Shared by all 4 test scripts. Do not run directly.
Run t1c_test1_baseline.py, t1c_test2_dm_only.py,
    t1c_test3_higher_entry.py, t1c_test4_tighter_exit.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: pip install supabase pandas numpy python-dotenv")
    sys.exit(1)

load_dotenv()


def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be in .env")
    return create_client(url, key)


def load_dm(client, year):
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"
    print(f"  Loading DM {year}...")
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(CONFIG['DM_TABLE'])
             .select(f"{CONFIG['DM_DATE_COL']},{CONFIG['DM_TICKER_COL']},"
                     f"{CONFIG['DM_CLOSE_COL']},{CONFIG['DM_SCORE_COL']}")
             .gte(CONFIG['DM_DATE_COL'], load_from)
             .lte(CONFIG['DM_DATE_COL'], end)
             .order(CONFIG['DM_DATE_COL'], desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page: break
        offset += page
    if not rows:
        print(f"  WARNING: No DM data for {year}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['DM_DATE_COL']: 'date',
                       CONFIG['DM_TICKER_COL']: 'ticker',
                       CONFIG['DM_CLOSE_COL']: 'close',
                       CONFIG['DM_SCORE_COL']: 'dm'}, inplace=True)
    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date','ticker','close','dm'])
    df = df.drop_duplicates(subset=['date','ticker'], keep='last')
    df = df.sort_values(['ticker','date']).reset_index(drop=True)
    print(f"  DM: {len(df):,} rows | {df['ticker'].nunique()} tickers")
    return df


def load_hms(client, year):
    if CONFIG.get('HMS_MIN') is None:
        return pd.DataFrame()
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"
    print(f"  Loading HMS {year}...")
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(CONFIG['HMS_TABLE'])
             .select(f"{CONFIG['HMS_DATE_COL']},{CONFIG['HMS_TICKER_COL']},"
                     f"{CONFIG['HMS_SCORE_COL']}")
             .gte(CONFIG['HMS_DATE_COL'], load_from)
             .lte(CONFIG['HMS_DATE_COL'], end)
             .order(CONFIG['HMS_DATE_COL'], desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page: break
        offset += page
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['HMS_DATE_COL']: 'date',
                       CONFIG['HMS_TICKER_COL']: 'ticker',
                       CONFIG['HMS_SCORE_COL']: 'hms'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hms']  = pd.to_numeric(df['hms'], errors='coerce')
    df = df.dropna(subset=['date','ticker','hms'])
    df = df.drop_duplicates(subset=['date','ticker'], keep='last')
    df = df.sort_values(['ticker','date']).reset_index(drop=True)
    print(f"  HMS: {len(df):,} rows | {df['ticker'].nunique()} tickers")
    return df


def compute_signals(dm_df, hms_df):
    print("  Computing signals...")
    dm_df = dm_df.sort_values(['ticker','date']).copy()
    dm_df['ma5']      = dm_df.groupby('ticker')['dm'].transform(
                            lambda x: x.rolling(5,  min_periods=1).mean())
    dm_df['ma5_prev'] = dm_df.groupby('ticker')['ma5'].transform(
                            lambda x: x.shift(1))
    dm_df['ma20']     = dm_df.groupby('ticker')['dm'].transform(
                            lambda x: x.rolling(20, min_periods=1).mean())

    if not hms_df.empty:
        hms_dedup = hms_df.drop_duplicates(subset=['date','ticker'], keep='last')
        dm_df = pd.merge(dm_df, hms_dedup[['date','ticker','hms']],
                         on=['date','ticker'], how='left')
        dm_df['hms'] = dm_df.groupby('ticker')['hms'].transform(
                            lambda x: x.ffill())
    else:
        dm_df['hms'] = None

    hms_min = CONFIG.get('HMS_MIN')
    if hms_min is not None:
        hms_ok = dm_df['hms'].isna() | (dm_df['hms'] >= hms_min)
    else:
        hms_ok = True

    dm_df['entry_signal'] = (
        (dm_df['dm']  >= CONFIG['ENTRY_DM']) &
        (dm_df['ma5'] >  dm_df['ma5_prev'].fillna(0)) &
        hms_ok
    )
    dm_df['exit_signal'] = dm_df['ma20'] < CONFIG['EXIT_MA20']
    dm_df = dm_df.drop_duplicates(subset=['date','ticker'], keep='last')
    return dm_df


def run_year(year, signals_df, starting_nav, open_positions):
    y_start = date(year, 1, 1)
    y_end   = date(year, 12, 31)
    ydf     = signals_df[(signals_df['date'] >= y_start) &
                         (signals_df['date'] <= y_end)].copy()
    trading_days = sorted(ydf['date'].unique())
    if not trading_days:
        return None

    # Build fast lookup
    lookup = {}
    for _, row in ydf.iterrows():
        tk, dt = row['ticker'], row['date']
        if tk not in lookup: lookup[tk] = {}
        lookup[tk][dt] = row

    # Reconstruct cash from starting_nav minus invested
    invested = sum(p['shares'] * p['entry_price']
                   for p in open_positions.values())
    cash = starting_nav - invested

    closed_trades = []
    all_tickers   = list(lookup.keys())

    for day in trading_days:
        # EXITS
        to_close = [tk for tk, pos in open_positions.items()
                    if lookup.get(tk, {}).get(day, {}).get('exit_signal', False)]
        for tk in to_close:
            pos  = open_positions.pop(tk)
            d    = lookup[tk][day]
            ret  = (d['close'] - pos['entry_price']) / pos['entry_price'] * 100
            hold = (day - pos['entry_date']).days
            cash += pos['shares'] * d['close']
            closed_trades.append({
                'entry_date':  pos['entry_date'],
                'ticker':      tk,
                'entry_price': pos['entry_price'],
                'entry_dm':    pos['entry_dm'],
                'entry_ma20':  pos['entry_ma20'],
                'exit_date':   day,
                'exit_price':  d['close'],
                'exit_dm':     d['dm'],
                'exit_ma20':   d['ma20'],
                'return_pct':  ret,
                'hold_days':   hold
            })

        # ENTRIES
        if (len(open_positions) < CONFIG['MAX_POSITIONS'] and
                cash >= CONFIG['MIN_POSITION']):
            held = set(open_positions.keys())
            cands = [(tk, lookup[tk][day])
                     for tk in all_tickers
                     if tk not in held and
                        day in lookup.get(tk, {}) and
                        lookup[tk][day]['entry_signal'] and
                        not lookup[tk][day]['exit_signal']]
            cands.sort(key=lambda x: x[1]['dm'], reverse=True)

            for tk, d in cands:
                if len(open_positions) >= CONFIG['MAX_POSITIONS']: break
                if cash < CONFIG['MIN_POSITION']: break
                slots  = CONFIG['MAX_POSITIONS'] - len(open_positions)
                alloc  = min(cash / max(1, slots), cash)
                if alloc < CONFIG['MIN_POSITION']: break
                price  = d['close']
                if price <= 0: continue
                shares = int(alloc // price)
                if shares <= 0: continue
                cash -= shares * price
                open_positions[tk] = {
                    'shares':      shares,
                    'entry_date':  day,
                    'entry_price': price,
                    'entry_dm':    d['dm'],
                    'entry_ma20':  d['ma20'],
                }

    # Year-end NAV
    last_day  = trading_days[-1]
    final_nav = cash
    for tk, pos in open_positions.items():
        d = lookup.get(tk, {}).get(last_day)
        final_nav += pos['shares'] * (d['close'] if d is not None else pos['entry_price'])

    n         = len(closed_trades)
    returns   = [t['return_pct'] for t in closed_trades]
    hold_days = [t['hold_days']  for t in closed_trades]
    winners   = sum(1 for r in returns if r >= 10)
    zero_days = sum(1 for h in hold_days if h == 0)
    avg_ret   = sum(returns)   / n if n > 0 else 0
    avg_hold  = sum(hold_days) / n if n > 0 else 0
    hit_rate  = winners / n * 100 if n > 0 else 0
    year_ret  = (final_nav / starting_nav - 1) * 100

    return {
        'year': year, 'starting_nav': starting_nav,
        'final_nav': final_nav, 'year_return': year_ret,
        'closed_trades': closed_trades, 'open_positions': open_positions,
        'cash': cash, 'n_trades': n, 'hit_rate': hit_rate,
        'avg_return': avg_ret, 'avg_hold_days': avg_hold,
        'zero_day_holds': zero_days,
    }


def print_year(r):
    print(f"\n  {r['year']} | NAV: ${r['final_nav']:>12,.0f} | "
          f"Return: {r['year_return']:>+7.1f}% | "
          f"Trades: {r['n_trades']:>5} | "
          f"Hit: {r['hit_rate']:>5.1f}% | "
          f"AvgHold: {r['avg_hold_days']:>5.0f}d | "
          f"ZeroDays: {r['zero_day_holds']:>4}")
    if r['zero_day_holds'] > r['n_trades'] * 0.05 and r['n_trades'] > 0:
        print(f"  ⚠️  WARNING: {r['zero_day_holds']} zero-day holds "
              f"({r['zero_day_holds']/r['n_trades']*100:.1f}%). "
              f"STOP AND REPORT.")
        examples = [t for t in r['closed_trades'] if t['hold_days'] == 0][:3]
        for ex in examples:
            print(f"     {ex['ticker']} {ex['entry_date']} "
                  f"EntryDM:{ex['entry_dm']:.1f} ExitMA20:{ex['exit_ma20']:.1f} "
                  f"Ret:{ex['return_pct']:.1f}%")


def print_summary(year_results, test_name):
    print(f"\n{'='*65}")
    print(f"  {test_name}")
    print(f"  Entry DM: {CONFIG['ENTRY_DM']} | "
          f"Exit MA20: {CONFIG['EXIT_MA20']} | "
          f"HMS: {CONFIG.get('HMS_MIN','DISABLED')}")
    print(f"{'='*65}")

    initial = CONFIG['INITIAL_CAPITAL']
    final   = year_results[-1]['final_nav']
    years   = len(year_results)
    total   = (final / initial - 1) * 100
    cagr    = ((final / initial) ** (1/years) - 1) * 100

    all_trades = [t for yr in year_results for t in yr['closed_trades']]
    n          = len(all_trades)
    returns    = [t['return_pct'] for t in all_trades]
    holds      = [t['hold_days']  for t in all_trades]
    zero_days  = sum(1 for h in holds if h == 0)

    print(f"\n  {'Year':<6} {'NAV':>12} {'Ret%':>8} "
          f"{'Trades':>7} {'Hit%':>6} {'AvgHold':>8} {'ZeroD':>6}")
    print(f"  {'-'*57}")

    peak_nav = initial
    max_dd   = 0
    for yr in year_results:
        if yr['final_nav'] < peak_nav:
            dd = (yr['final_nav'] - peak_nav) / peak_nav * 100
            if dd < max_dd: max_dd = dd
        else:
            peak_nav = yr['final_nav']
        print(f"  {yr['year']:<6} ${yr['final_nav']:>11,.0f} "
              f"{yr['year_return']:>+7.1f}% "
              f"{yr['n_trades']:>7} "
              f"{yr['hit_rate']:>5.1f}% "
              f"{yr['avg_hold_days']:>7.0f}d "
              f"{yr['zero_day_holds']:>6}")

    print(f"\n  Total trades:    {n:,}")
    print(f"  Avg return:      {sum(returns)/n:.2f}%" if n > 0 else "")
    print(f"  Avg hold days:   {sum(holds)/n:.0f}" if n > 0 else "")
    print(f"  Zero-day holds:  {zero_days} ({zero_days/n*100:.1f}%)" if n > 0 else "")
    print(f"  Max drawdown:    {max_dd:.1f}%")
    print(f"\n  FINAL NAV:   ${final:,.0f}")
    print(f"  TOTAL RETURN: {total:+.1f}%")
    print(f"  CAGR:         {cagr:.1f}% per year")

    # Risk-adjusted metrics from yearly returns
    yearly_rets = [yr['year_return'] / 100 for yr in year_results]
    if len(yearly_rets) >= 2:
        rf = 0.04  # 4% annual risk-free rate
        excess = [r - rf for r in yearly_rets]
        avg_excess = np.mean(excess)
        std_ret = np.std(yearly_rets, ddof=1)
        sharpe = round(avg_excess / std_ret, 2) if std_ret > 0 else 0
        downside = [r for r in yearly_rets if r < rf]
        downside_std = np.std(downside, ddof=1) if len(downside) >= 2 else std_ret
        sortino = round(avg_excess / downside_std, 2) if downside_std > 0 else 0
        calmar = round((cagr / 100) / abs(max_dd / 100), 2) if max_dd < 0 else round(cagr / 100 / 0.01, 2)
        print(f"\n  ── RISK-ADJUSTED ──────────────────────────────────")
        print(f"  Sharpe Ratio:    {sharpe}  (>1.0 good, >2.0 excellent)")
        print(f"  Sortino Ratio:   {sortino}")
        print(f"  Calmar Ratio:    {calmar}  (CAGR / |max DD|)")

    print(f"\n  (Raw — apply 15-20% survivorship bias discount)")
    print(f"{'='*65}\n")


def run(test_name):
    print(f"\n{'='*65}")
    print(f"  {test_name}")
    print(f"  Entry DM >= {CONFIG['ENTRY_DM']} | "
          f"Exit MA20 < {CONFIG['EXIT_MA20']} | "
          f"HMS >= {CONFIG.get('HMS_MIN','DISABLED')}")
    print(f"{'='*65}")

    try:
        client = get_client()
        print("Connected to Supabase.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    nav            = CONFIG['INITIAL_CAPITAL']
    open_positions = {}
    year_results   = []

    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        dm_df  = load_dm(client, year)
        hms_df = load_hms(client, year)
        if dm_df.empty:
            print(f"  Skipping {year} — no DM data")
            continue
        signals_df = compute_signals(dm_df, hms_df)
        result     = run_year(year, signals_df, nav, open_positions)
        if result is None:
            print(f"  Skipping {year} — no trading days")
            continue
        print_year(result)
        year_results.append(result)
        nav            = result['final_nav']
        open_positions = result['open_positions']

        # Stop if zero-day warning threshold exceeded
        if (result['zero_day_holds'] > result['n_trades'] * 0.05 and
                result['n_trades'] > 0):
            print(f"\n  STOPPING at {year} due to zero-day hold warning.")
            print(f"  Send results so far back for review.")
            break

    if year_results:
        print_summary(year_results, test_name)

    print("DONE.\n")
