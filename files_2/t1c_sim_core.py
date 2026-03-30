"""
T1C SIMULATION CORE ENGINE v2
==============================
Shared by all test scripts. Do not run directly.
Supports:
  - Standard entry/exit (Tests 1-5)
  - HMS as exit signal (Test 6) via CONFIG['HMS_EXIT_MIN']
  - ETF flow position sizing (Test 7) via CONFIG['ETF_FLOW_SIZING']
"""

import os, sys, pandas as pd, numpy as np
from datetime import date
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
    if CONFIG.get('HMS_MIN') is None and CONFIG.get('HMS_EXIT_MIN') is None:
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


def load_etf_flows(client, year):
    if not CONFIG.get('ETF_FLOW_SIZING'):
        return pd.DataFrame()
    load_from = f"{year}-01-01"
    end       = f"{year}-12-31"
    print(f"  Loading ETF flows {year}...")
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(CONFIG['ETF_FLOW_TABLE'])
             .select(f"{CONFIG['ETF_FLOW_DATE_COL']},{CONFIG['ETF_FLOW_TICKER_COL']},"
                     f"{CONFIG['ETF_FLOW_DIR_COL']}")
             .gte(CONFIG['ETF_FLOW_DATE_COL'], load_from)
             .lte(CONFIG['ETF_FLOW_DATE_COL'], end)
             .order(CONFIG['ETF_FLOW_DATE_COL'], desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page: break
        offset += page
    if not rows:
        print(f"  WARNING: No ETF flow data for {year} — using neutral sizing")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['ETF_FLOW_DATE_COL']: 'date',
                       CONFIG['ETF_FLOW_TICKER_COL']: 'etf_ticker',
                       CONFIG['ETF_FLOW_DIR_COL']: 'flow_direction'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    print(f"  ETF flows: {len(df):,} rows")
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

    # Entry HMS gate (None = disabled)
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

    # Exit signal — MA20 < threshold
    dm_df['exit_signal'] = dm_df['ma20'] < CONFIG['EXIT_MA20']

    # HMS exit signal (Test 6) — exits if HMS drops below threshold
    hms_exit_min = CONFIG.get('HMS_EXIT_MIN')
    if hms_exit_min is not None and 'hms' in dm_df.columns:
        hms_exit = dm_df['hms'].notna() & (dm_df['hms'] < hms_exit_min)
        dm_df['exit_signal'] = dm_df['exit_signal'] | hms_exit

    dm_df = dm_df.drop_duplicates(subset=['date','ticker'], keep='last')
    return dm_df


def get_etf_alloc(etf_flows_df, day, ticker):
    """Get position allocation % based on ETF flow for ticker's sector."""
    if etf_flows_df.empty or not CONFIG.get('ETF_FLOW_SIZING'):
        return CONFIG.get('ETF_NEUTRAL_ALLOC_PCT', 0.05)

    sector_map = CONFIG.get('SECTOR_ETF_MAP', {})
    default_etf = sector_map.get('Default', 'SPY')

    # Simple approach: use SPY as proxy if no sector mapping
    # Developer can enhance with actual sector lookup from dm_daily
    etf = default_etf
    flow_row = etf_flows_df[
        (etf_flows_df['date'] == day) &
        (etf_flows_df['etf_ticker'] == etf)
    ]
    if flow_row.empty:
        return CONFIG.get('ETF_NEUTRAL_ALLOC_PCT', 0.05)

    direction = flow_row.iloc[0]['flow_direction']
    if direction == 'INFLOW':
        return CONFIG.get('ETF_INFLOW_ALLOC_PCT', 0.07)
    elif direction == 'OUTFLOW':
        return CONFIG.get('ETF_OUTFLOW_ALLOC_PCT', 0.03)
    else:
        return CONFIG.get('ETF_NEUTRAL_ALLOC_PCT', 0.05)


def run_year(year, signals_df, starting_nav, open_positions, etf_flows_df=None):
    if etf_flows_df is None:
        etf_flows_df = pd.DataFrame()

    y_start = date(year, 1, 1)
    y_end   = date(year, 12, 31)
    end_date_str = CONFIG.get('END_DATE')
    if end_date_str:
        from datetime import datetime as _dt
        y_end = min(y_end, _dt.strptime(end_date_str, '%Y-%m-%d').date())
    ydf     = signals_df[(signals_df['date'] >= y_start) &
                         (signals_df['date'] <= y_end)].copy()
    trading_days = sorted(ydf['date'].unique())
    if not trading_days:
        return None

    lookup = {}
    for _, row in ydf.iterrows():
        tk, dt = row['ticker'], row['date']
        if tk not in lookup: lookup[tk] = {}
        lookup[tk][dt] = row

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

                # Sizing logic — priority: DM sizing > ETF flow sizing > equal weight
                if CONFIG.get('DM_SIZING'):
                    dm_val = d['dm']
                    alloc_pct = 0.05  # default
                    for threshold, pct in CONFIG.get('DM_TIERS', []):
                        if dm_val >= threshold:
                            alloc_pct = pct
                            break
                    alloc = min(starting_nav * alloc_pct, cash)
                elif CONFIG.get('ETF_FLOW_SIZING'):
                    alloc_pct = get_etf_alloc(etf_flows_df, day, tk)
                    alloc = min(starting_nav * alloc_pct, cash)
                else:
                    slots = CONFIG['MAX_POSITIONS'] - len(open_positions)
                    alloc = min(cash / max(1, slots), cash)

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
          f"ZeroD: {r['zero_day_holds']:>4}")
    if r['zero_day_holds'] > r['n_trades'] * 0.05 and r['n_trades'] > 0:
        print(f"  ⚠️  WARNING: {r['zero_day_holds']} zero-day holds. STOP AND REPORT.")
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
          f"HMS entry: {CONFIG.get('HMS_MIN','DISABLED')} | "
          f"HMS exit: {CONFIG.get('HMS_EXIT_MIN','DISABLED')}")
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
    print(f"\n  FINAL NAV:    ${final:,.0f}")
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
        dm_df      = load_dm(client, year)
        hms_df     = load_hms(client, year)
        etf_df     = load_etf_flows(client, year)

        if dm_df.empty:
            print(f"  Skipping {year} — no DM data")
            continue

        signals_df = compute_signals(dm_df, hms_df)
        result     = run_year(year, signals_df, nav, open_positions, etf_df)

        if result is None:
            print(f"  Skipping {year} — no trading days")
            continue

        print_year(result)
        year_results.append(result)
        nav            = result['final_nav']
        open_positions = result['open_positions']

        if (result['zero_day_holds'] > result['n_trades'] * 0.05 and
                result['n_trades'] > 0):
            print(f"\n  STOPPING at {year} — zero-day hold warning. Send results.")
            break

    if year_results:
        print_summary(year_results, test_name)

    print("DONE.\n")


def load_macro(client, year):
    if not CONFIG.get('MACRO_SIZING'):
        return pd.DataFrame()
    load_from = f"{year}-01-01"
    end       = f"{year}-12-31"
    print(f"  Loading macro {year}...")
    rows, offset, page = [], 0, 5000
    while True:
        r = (client.table(CONFIG['MACRO_TABLE'])
             .select(f"{CONFIG['MACRO_DATE_COL']},{CONFIG['MACRO_VIX_COL']}")
             .gte(CONFIG['MACRO_DATE_COL'], load_from)
             .lte(CONFIG['MACRO_DATE_COL'], end)
             .order(CONFIG['MACRO_DATE_COL'], desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page: break
        offset += page
    if not rows:
        print(f"  WARNING: No macro data for {year}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['MACRO_DATE_COL']: 'date',
                       CONFIG['MACRO_VIX_COL']: 'vix'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['vix']  = pd.to_numeric(df['vix'], errors='coerce')
    df = df.dropna(subset=['date','vix'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Macro: {len(df):,} rows")
    return df


def load_insider(client, year):
    if not CONFIG.get('INSIDER_FILTER'):
        return pd.DataFrame()
    # Load 30 days extra for lookback
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"
    print(f"  Loading insider transactions {year}...")
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(CONFIG['INSIDER_TABLE'])
             .select(f"{CONFIG['INSIDER_DATE_COL']},{CONFIG['INSIDER_TICKER_COL']},"
                     f"{CONFIG['INSIDER_TYPE_COL']}")
             .gte(CONFIG['INSIDER_DATE_COL'], load_from)
             .lte(CONFIG['INSIDER_DATE_COL'], end)
             .order(CONFIG['INSIDER_DATE_COL'], desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page: break
        offset += page
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['INSIDER_DATE_COL']: 'date',
                       CONFIG['INSIDER_TICKER_COL']: 'ticker',
                       CONFIG['INSIDER_TYPE_COL']: 'txn_type'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    # Keep only sells
    sells = df[df['txn_type'] == CONFIG.get('INSIDER_SELL_VALUE','SELL')].copy()
    print(f"  Insider sells: {len(sells):,} rows")
    return sells


def load_short(client, year):
    if not CONFIG.get('SHORT_FILTER'):
        return pd.DataFrame()
    if year < CONFIG.get('SHORT_START_YEAR', 2018):
        return pd.DataFrame()
    load_from = f"{year}-01-01"
    end       = f"{year}-12-31"
    print(f"  Loading short interest {year}...")
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(CONFIG['SHORT_TABLE'])
             .select(f"{CONFIG['SHORT_DATE_COL']},{CONFIG['SHORT_TICKER_COL']},"
                     f"{CONFIG['SHORT_PCT_COL']}")
             .gte(CONFIG['SHORT_DATE_COL'], load_from)
             .lte(CONFIG['SHORT_DATE_COL'], end)
             .order(CONFIG['SHORT_DATE_COL'], desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page: break
        offset += page
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={CONFIG['SHORT_DATE_COL']: 'date',
                       CONFIG['SHORT_TICKER_COL']: 'ticker',
                       CONFIG['SHORT_PCT_COL']: 'short_pct'}, inplace=True)
    df['date']      = pd.to_datetime(df['date']).dt.date
    df['short_pct'] = pd.to_numeric(df['short_pct'], errors='coerce')
    df = df.dropna(subset=['date','ticker','short_pct'])
    print(f"  Short interest: {len(df):,} rows")
    return df


def get_macro_cap(macro_df, day):
    """Return position size cap based on VIX regime."""
    if macro_df.empty or not CONFIG.get('MACRO_SIZING'):
        return None  # No cap
    row = macro_df[macro_df['date'] <= day]
    if row.empty:
        return None
    vix = row.iloc[-1]['vix']
    if vix > CONFIG.get('MACRO_HIGH_STRESS_VIX', 25):
        return CONFIG.get('MACRO_HIGH_STRESS_CAP', 0.03)
    elif vix > CONFIG.get('MACRO_MED_STRESS_VIX', 20):
        return CONFIG.get('MACRO_MED_STRESS_CAP', 0.04)
    return None  # No cap in low stress


def has_insider_sell(insider_df, ticker, entry_date):
    """Returns True if insider selling exists in prior 30 days."""
    if insider_df.empty:
        return False
    lookback = CONFIG.get('INSIDER_LOOKBACK_DAYS', 30)
    from datetime import timedelta
    cutoff = entry_date - timedelta(days=lookback)
    sells = insider_df[
        (insider_df['ticker'] == ticker) &
        (insider_df['date'] >= cutoff) &
        (insider_df['date'] <= entry_date)
    ]
    return len(sells) > 0


def get_short_pct(short_df, ticker, day):
    """Return most recent short interest % for ticker."""
    if short_df.empty:
        return 0.0
    row = short_df[
        (short_df['ticker'] == ticker) &
        (short_df['date'] <= day)
    ]
    if row.empty:
        return 0.0
    return row.iloc[-1]['short_pct']


# Override run() to support Tests 9-11
_original_run = run

def run(test_name):
    print(f"\n{'='*65}")
    print(f"  {test_name}")
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
        dm_df      = load_dm(client, year)
        hms_df     = load_hms(client, year)
        etf_df     = load_etf_flows(client, year)
        macro_df   = load_macro(client, year)
        insider_df = load_insider(client, year)
        short_df   = load_short(client, year)

        if dm_df.empty:
            print(f"  Skipping {year} — no DM data")
            continue

        signals_df = compute_signals(dm_df, hms_df)
        result     = run_year_v2(year, signals_df, nav, open_positions,
                                 etf_df, macro_df, insider_df, short_df)

        if result is None:
            print(f"  Skipping {year} — no trading days")
            continue

        print_year(result)
        year_results.append(result)
        nav            = result['final_nav']
        open_positions = result['open_positions']

        if (result['zero_day_holds'] > result['n_trades'] * 0.05 and
                result['n_trades'] > 0):
            print(f"\n  STOPPING at {year} — zero-day hold warning. Send results.")
            break

    if year_results:
        print_summary(year_results, test_name)
    print("DONE.\n")


def run_year_v2(year, signals_df, starting_nav, open_positions,
                etf_df=None, macro_df=None, insider_df=None, short_df=None):
    if etf_df is None:   etf_df   = pd.DataFrame()
    if macro_df is None: macro_df = pd.DataFrame()
    if insider_df is None: insider_df = pd.DataFrame()
    if short_df is None: short_df = pd.DataFrame()

    y_start = date(year, 1, 1)
    y_end   = date(year, 12, 31)
    end_date_str = CONFIG.get('END_DATE')
    if end_date_str:
        from datetime import datetime as _dt
        y_end = min(y_end, _dt.strptime(end_date_str, '%Y-%m-%d').date())
    ydf     = signals_df[(signals_df['date'] >= y_start) &
                         (signals_df['date'] <= y_end)].copy()
    trading_days = sorted(ydf['date'].unique())
    if not trading_days:
        return None

    lookup = {}
    for _, row in ydf.iterrows():
        tk, dt = row['ticker'], row['date']
        if tk not in lookup: lookup[tk] = {}
        lookup[tk][dt] = row

    invested = sum(p['shares'] * p['entry_price']
                   for p in open_positions.values())
    cash = starting_nav - invested

    closed_trades = []
    all_tickers   = list(lookup.keys())

    # Pre-build short interest lookup for speed
    short_max = CONFIG.get('SHORT_MAX_PCT', 15.0)

    for day in trading_days:
        # Get macro cap for today
        macro_cap = get_macro_cap(macro_df, day)

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

                # Insider filter (Test 10)
                if CONFIG.get('INSIDER_FILTER') and has_insider_sell(insider_df, tk, day):
                    continue

                # Short interest filter (Test 11)
                if CONFIG.get('SHORT_FILTER') and year >= CONFIG.get('SHORT_START_YEAR', 2018):
                    if get_short_pct(short_df, tk, day) > short_max:
                        continue

                # Sizing — DM tiers first, then macro cap
                if CONFIG.get('DM_SIZING'):
                    dm_val = d['dm']
                    alloc_pct = 0.05
                    for threshold, pct in CONFIG.get('DM_TIERS', []):
                        if dm_val >= threshold:
                            alloc_pct = pct
                            break
                    # Apply macro cap override
                    if macro_cap is not None:
                        alloc_pct = min(alloc_pct, macro_cap)
                    alloc = min(starting_nav * alloc_pct, cash)
                elif CONFIG.get('ETF_FLOW_SIZING'):
                    alloc_pct = get_etf_alloc(etf_df, day, tk)
                    if macro_cap is not None:
                        alloc_pct = min(alloc_pct, macro_cap)
                    alloc = min(starting_nav * alloc_pct, cash)
                else:
                    slots = CONFIG['MAX_POSITIONS'] - len(open_positions)
                    alloc = min(cash / max(1, slots), cash)
                    if macro_cap is not None:
                        alloc = min(alloc, starting_nav * macro_cap)

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
