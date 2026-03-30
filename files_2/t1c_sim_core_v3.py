"""
T1C SIMULATION CORE ENGINE v3
==============================
Shared by all test scripts. Do not run directly.

v3 additions over v2:
  - Daily NAV tracking (cash + mark-to-market of all open positions)
  - Sharpe ratio calculation (annualized, risk-free rate 4%)
  - Sortino ratio calculation
  - run_simulation(config) API for sector and universe scripts
  - Calmar ratio (CAGR / abs(max drawdown))
  - All results now include: sharpe, sortino, calmar, daily_nav_log
"""

import os, sys, pandas as pd, numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: pip install supabase pandas numpy python-dotenv")
    sys.exit(1)

load_dotenv()

# Global CONFIG — set by each test script before calling run()
CONFIG = {}

RISK_FREE_ANNUAL = 0.04          # 4% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be in .env")
    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_table(client, table, select_cols, date_col, date_from, date_to):
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(table)
             .select(select_cols)
             .gte(date_col, date_from)
             .lte(date_col, date_to)
             .order(date_col, desc=False)
             .range(offset, offset + page - 1)
             .execute())
        rows.extend(r.data)
        if len(r.data) < page:
            break
        offset += page
    return rows


def load_dm(client, year):
    cfg = CONFIG
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"
    print(f"  Loading DM {year}...")
    rows = _load_table(client, cfg['DM_TABLE'],
                       f"{cfg['DM_DATE_COL']},{cfg['DM_TICKER_COL']},"
                       f"{cfg['DM_CLOSE_COL']},{cfg['DM_SCORE_COL']}",
                       cfg['DM_DATE_COL'], load_from, end)
    if not rows:
        print(f"  WARNING: No DM data for {year}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={cfg['DM_DATE_COL']: 'date',
                       cfg['DM_TICKER_COL']: 'ticker',
                       cfg['DM_CLOSE_COL']: 'close',
                       cfg['DM_SCORE_COL']: 'dm'}, inplace=True)
    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date','ticker','close','dm'])
    df = df.drop_duplicates(subset=['date','ticker'], keep='last')
    # Filter to universe if specified
    universe = CONFIG.get('UNIVERSE')
    if universe:
        df = df[df['ticker'].isin(universe)]
    df = df.sort_values(['ticker','date']).reset_index(drop=True)
    print(f"  DM: {len(df):,} rows | {df['ticker'].nunique()} tickers")
    return df


def load_hms(client, year):
    cfg = CONFIG
    if cfg.get('HMS_MIN') is None and cfg.get('HMS_EXIT_MIN') is None:
        return pd.DataFrame()
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"
    print(f"  Loading HMS {year}...")
    rows = _load_table(client, cfg['HMS_TABLE'],
                       f"{cfg['HMS_DATE_COL']},{cfg['HMS_TICKER_COL']},"
                       f"{cfg['HMS_SCORE_COL']}",
                       cfg['HMS_DATE_COL'], load_from, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={cfg['HMS_DATE_COL']: 'date',
                       cfg['HMS_TICKER_COL']: 'ticker',
                       cfg['HMS_SCORE_COL']: 'hms'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hms']  = pd.to_numeric(df['hms'], errors='coerce')
    df = df.dropna(subset=['date','ticker','hms'])
    df = df.drop_duplicates(subset=['date','ticker'], keep='last')
    df = df.sort_values(['ticker','date']).reset_index(drop=True)
    print(f"  HMS: {len(df):,} rows")
    return df


def load_macro(client, year):
    cfg = CONFIG
    if not cfg.get('MACRO_SIZING'):
        return pd.DataFrame()
    load_from = f"{year}-01-01"
    end       = f"{year}-12-31"
    print(f"  Loading macro {year}...")
    rows = _load_table(client, cfg['MACRO_TABLE'],
                       f"{cfg['MACRO_DATE_COL']},{cfg['MACRO_VIX_COL']}",
                       cfg['MACRO_DATE_COL'], load_from, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={cfg['MACRO_DATE_COL']: 'date',
                       cfg['MACRO_VIX_COL']: 'vix'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['vix']  = pd.to_numeric(df['vix'], errors='coerce')
    return df.dropna(subset=['date','vix']).sort_values('date').reset_index(drop=True)


def load_insider(client, year):
    cfg = CONFIG
    if not cfg.get('INSIDER_FILTER'):
        return pd.DataFrame()
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"
    print(f"  Loading insider {year}...")
    rows = _load_table(client, cfg['INSIDER_TABLE'],
                       f"{cfg['INSIDER_DATE_COL']},{cfg['INSIDER_TICKER_COL']},"
                       f"{cfg['INSIDER_TYPE_COL']}",
                       cfg['INSIDER_DATE_COL'], load_from, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={cfg['INSIDER_DATE_COL']: 'date',
                       cfg['INSIDER_TICKER_COL']: 'ticker',
                       cfg['INSIDER_TYPE_COL']: 'txn_type'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df[df['txn_type'] == cfg.get('INSIDER_SELL_VALUE', 'SELL')].copy()


def load_short(client, year):
    cfg = CONFIG
    if not cfg.get('SHORT_FILTER') or year < cfg.get('SHORT_START_YEAR', 2018):
        return pd.DataFrame()
    load_from = f"{year}-01-01"
    end       = f"{year}-12-31"
    print(f"  Loading short interest {year}...")
    rows = _load_table(client, cfg['SHORT_TABLE'],
                       f"{cfg['SHORT_DATE_COL']},{cfg['SHORT_TICKER_COL']},"
                       f"{cfg['SHORT_PCT_COL']}",
                       cfg['SHORT_DATE_COL'], load_from, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={cfg['SHORT_DATE_COL']: 'date',
                       cfg['SHORT_TICKER_COL']: 'ticker',
                       cfg['SHORT_PCT_COL']: 'short_pct'}, inplace=True)
    df['date']      = pd.to_datetime(df['date']).dt.date
    df['short_pct'] = pd.to_numeric(df['short_pct'], errors='coerce')
    return df.dropna(subset=['date','ticker','short_pct'])


def load_etf_flows(client, year):
    cfg = CONFIG
    if not cfg.get('ETF_FLOW_SIZING'):
        return pd.DataFrame()
    load_from = f"{year}-01-01"
    end       = f"{year}-12-31"
    rows = _load_table(client, cfg['ETF_FLOW_TABLE'],
                       f"{cfg['ETF_FLOW_DATE_COL']},{cfg['ETF_FLOW_TICKER_COL']},"
                       f"{cfg['ETF_FLOW_DIR_COL']}",
                       cfg['ETF_FLOW_DATE_COL'], load_from, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.rename(columns={cfg['ETF_FLOW_DATE_COL']: 'date',
                       cfg['ETF_FLOW_TICKER_COL']: 'etf_ticker',
                       cfg['ETF_FLOW_DIR_COL']: 'flow_direction'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(dm_df, hms_df):
    cfg = CONFIG
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

    hms_min = cfg.get('HMS_MIN')
    if hms_min is not None:
        hms_ok = dm_df['hms'].isna() | (dm_df['hms'] >= hms_min)
    else:
        hms_ok = True

    entry_dm = cfg.get('ENTRY_DM') or cfg.get('DM_ENTRY_MIN', 65)
    exit_ma20 = cfg.get('EXIT_MA20') or cfg.get('MA20_EXIT', 50)

    dm_df['entry_signal'] = (
        (dm_df['dm']  >= entry_dm) &
        (dm_df['ma5'] >  dm_df['ma5_prev'].fillna(0)) &
        hms_ok
    )
    dm_df['exit_signal'] = dm_df['ma20'] < exit_ma20

    hms_exit_min = cfg.get('HMS_EXIT_MIN')
    if hms_exit_min is not None and 'hms' in dm_df.columns:
        hms_exit = dm_df['hms'].notna() & (dm_df['hms'] < hms_exit_min)
        dm_df['exit_signal'] = dm_df['exit_signal'] | hms_exit

    dm_df = dm_df.drop_duplicates(subset=['date','ticker'], keep='last')
    return dm_df


# ─────────────────────────────────────────────────────────────────────────────
# SIZING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_dm_alloc_pct(dm_val):
    """DM-tiered allocation percentage."""
    cfg = CONFIG
    tiers = cfg.get('DM_TIERS')
    if tiers:
        for threshold, pct in tiers:
            if dm_val >= threshold:
                return pct
        return 0.035
    # New-style config (sector scripts)
    if dm_val >= 90: return cfg.get('TIER_90', 0.080)
    if dm_val >= 80: return cfg.get('TIER_80', 0.065)
    if dm_val >= 70: return cfg.get('TIER_70', 0.050)
    return cfg.get('TIER_65', 0.035)


def get_macro_cap(macro_df, day):
    cfg = CONFIG
    if macro_df.empty or not cfg.get('MACRO_SIZING'):
        return None
    row = macro_df[macro_df['date'] <= day]
    if row.empty:
        return None
    vix = row.iloc[-1]['vix']
    if vix > cfg.get('MACRO_HIGH_STRESS_VIX', 25):
        return cfg.get('MACRO_HIGH_STRESS_CAP', 0.03)
    elif vix > cfg.get('MACRO_MED_STRESS_VIX', 20):
        return cfg.get('MACRO_MED_STRESS_CAP', 0.04)
    return None


def has_insider_sell(insider_df, ticker, entry_date):
    if insider_df.empty:
        return False
    lookback = CONFIG.get('INSIDER_LOOKBACK_DAYS', 30)
    cutoff = entry_date - timedelta(days=lookback)
    sells = insider_df[
        (insider_df['ticker'] == ticker) &
        (insider_df['date'] >= cutoff) &
        (insider_df['date'] <= entry_date)
    ]
    return len(sells) > 0


def get_short_pct(short_df, ticker, day):
    if short_df.empty:
        return 0.0
    row = short_df[(short_df['ticker'] == ticker) & (short_df['date'] <= day)]
    if row.empty:
        return 0.0
    return row.iloc[-1]['short_pct']


# ─────────────────────────────────────────────────────────────────────────────
# [v3] SHARPE / SORTINO / CALMAR CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_risk_metrics(daily_nav_log, initial_nav):
    """
    Calculate Sharpe, Sortino, Calmar, and max drawdown
    from a daily NAV log (list of dicts with 'date' and 'nav').

    Returns dict with all metrics.
    """
    if len(daily_nav_log) < 2:
        return {
            'sharpe': None, 'sortino': None, 'calmar': None,
            'max_drawdown_pct': 0.0, 'daily_nav_series': []
        }

    navs = pd.Series(
        [e['nav'] for e in daily_nav_log],
        index=pd.to_datetime([e['date'] for e in daily_nav_log])
    )

    # Daily returns
    daily_returns = navs.pct_change().dropna()

    if len(daily_returns) < 2:
        return {
            'sharpe': None, 'sortino': None, 'calmar': None,
            'max_drawdown_pct': 0.0, 'daily_nav_series': daily_nav_log
        }

    # Risk-free daily rate
    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS_PER_YEAR

    # Excess returns
    excess = daily_returns - rf_daily

    # Sharpe
    if excess.std() > 0:
        sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    # Sortino (downside deviation only)
    downside = excess[excess < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sortino = 0.0

    # Max drawdown
    rolling_peak = navs.cummax()
    drawdowns    = (navs - rolling_peak) / rolling_peak * 100
    max_dd       = drawdowns.min()

    # CAGR from daily series
    days = (navs.index[-1] - navs.index[0]).days
    if days > 0:
        years = days / 365.25
        cagr  = ((navs.iloc[-1] / initial_nav) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # Calmar = CAGR / abs(max drawdown)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'sharpe':            round(sharpe, 2),
        'sortino':           round(sortino, 2),
        'calmar':            round(calmar, 2),
        'max_drawdown_pct':  round(max_dd, 1),
        'cagr_from_daily':   round(cagr, 1),
        'daily_nav_series':  daily_nav_log,
        'n_trading_days':    len(daily_returns),
    }


# ─────────────────────────────────────────────────────────────────────────────
# YEAR RUNNER — v3 with daily NAV tracking
# ─────────────────────────────────────────────────────────────────────────────

def run_year_v3(year, signals_df, starting_nav, open_positions,
                etf_df=None, macro_df=None, insider_df=None, short_df=None):
    """
    Run one year of simulation.
    Returns year result dict including daily_nav_log for Sharpe calculation.
    """
    cfg = CONFIG
    if etf_df is None:     etf_df     = pd.DataFrame()
    if macro_df is None:   macro_df   = pd.DataFrame()
    if insider_df is None: insider_df = pd.DataFrame()
    if short_df is None:   short_df   = pd.DataFrame()

    y_start = date(year, 1, 1)
    y_end   = date(year, 12, 31)
    ydf     = signals_df[(signals_df['date'] >= y_start) &
                         (signals_df['date'] <= y_end)].copy()
    trading_days = sorted(ydf['date'].unique())
    if not trading_days:
        return None

    # Build fast lookup: ticker -> date -> row
    lookup = {}
    for _, row in ydf.iterrows():
        tk, dt = row['ticker'], row['date']
        if tk not in lookup: lookup[tk] = {}
        lookup[tk][dt] = row

    invested = sum(p['shares'] * p['entry_price'] for p in open_positions.values())
    cash     = starting_nav - invested

    closed_trades = []
    all_tickers   = list(lookup.keys())
    daily_nav_log = []                          # [v3] daily NAV tracking
    short_max     = cfg.get('SHORT_MAX_PCT', 15.0)
    initial_capital = cfg.get('INITIAL_CAPITAL') or cfg.get('INITIAL_NAV', 10_000_000)

    for day in trading_days:
        macro_cap = get_macro_cap(macro_df, day)

        # ── EXITS ──────────────────────────────────────────────────────────
        to_close = [tk for tk, pos in open_positions.items()
                    if lookup.get(tk, {}).get(day, {}).get('exit_signal', False)]
        for tk in to_close:
            pos  = open_positions.pop(tk)
            d    = lookup[tk][day]
            ret  = (d['close'] - pos['entry_price']) / pos['entry_price'] * 100
            hold = (day - pos['entry_date']).days
            cash += pos['shares'] * d['close']
            closed_trades.append({
                'entry_date':  pos['entry_date'],  'ticker':      tk,
                'entry_price': pos['entry_price'], 'entry_dm':    pos['entry_dm'],
                'entry_ma20':  pos['entry_ma20'],  'exit_date':   day,
                'exit_price':  d['close'],          'exit_dm':     d['dm'],
                'exit_ma20':   d['ma20'],           'return_pct':  ret,
                'hold_days':   hold
            })

        # ── ENTRIES ────────────────────────────────────────────────────────
        max_pos = cfg.get('MAX_POSITIONS', 20)
        min_pos = cfg.get('MIN_POSITION', 500)

        if len(open_positions) < max_pos and cash >= min_pos:
            held  = set(open_positions.keys())
            cands = [(tk, lookup[tk][day])
                     for tk in all_tickers
                     if tk not in held and
                        day in lookup.get(tk, {}) and
                        lookup[tk][day]['entry_signal'] and
                        not lookup[tk][day]['exit_signal']]
            cands.sort(key=lambda x: x[1]['dm'], reverse=True)

            for tk, d in cands:
                if len(open_positions) >= max_pos: break
                if cash < min_pos: break

                # Filters
                if cfg.get('INSIDER_FILTER') and has_insider_sell(insider_df, tk, day):
                    continue
                if (cfg.get('SHORT_FILTER') and
                        year >= cfg.get('SHORT_START_YEAR', 2018) and
                        get_short_pct(short_df, tk, day) > short_max):
                    continue

                # Sizing
                dm_sizing = cfg.get('DM_SIZING') or cfg.get('SIZING_MODE') == 'dm_tiered'
                if dm_sizing:
                    alloc_pct = get_dm_alloc_pct(d['dm'])
                    if macro_cap is not None:
                        alloc_pct = min(alloc_pct, macro_cap)
                    alloc = min(initial_capital * alloc_pct, cash)
                elif cfg.get('ETF_FLOW_SIZING'):
                    alloc_pct = cfg.get('ETF_NEUTRAL_ALLOC_PCT', 0.05)
                    if macro_cap is not None:
                        alloc_pct = min(alloc_pct, macro_cap)
                    alloc = min(initial_capital * alloc_pct, cash)
                else:
                    slots = max_pos - len(open_positions)
                    alloc = min(cash / max(1, slots), cash)
                    if macro_cap is not None:
                        alloc = min(alloc, initial_capital * macro_cap)

                if alloc < min_pos: break
                price  = d['close']
                if price <= 0: continue
                shares = int(alloc // price)
                if shares <= 0: continue
                cash -= shares * price
                open_positions[tk] = {
                    'shares':      shares,  'entry_date':  day,
                    'entry_price': price,   'entry_dm':    d['dm'],
                    'entry_ma20':  d['ma20'],
                }

        # ── [v3] DAILY NAV LOG ─────────────────────────────────────────────
        # Mark-to-market: cash + value of all open positions at today's close
        mtm_nav = cash
        for tk, pos in open_positions.items():
            d = lookup.get(tk, {}).get(day)
            if d is not None:
                mtm_nav += pos['shares'] * d['close']
            else:
                # Use last known price if no data today
                mtm_nav += pos['shares'] * pos['entry_price']
        daily_nav_log.append({'date': day, 'nav': mtm_nav})

    # ── YEAR-END NAV ──────────────────────────────────────────────────────
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
        'year':           year,       'starting_nav':   starting_nav,
        'final_nav':      final_nav,  'year_return':    year_ret,
        'closed_trades':  closed_trades,
        'open_positions': open_positions,
        'cash':           cash,       'n_trades':       n,
        'hit_rate':       hit_rate,   'avg_return':     avg_ret,
        'avg_hold_days':  avg_hold,   'zero_day_holds': zero_days,
        'daily_nav_log':  daily_nav_log,          # [v3] new
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def print_summary(year_results, test_name, risk_metrics=None):
    cfg = CONFIG
    initial = cfg.get('INITIAL_CAPITAL') or cfg.get('INITIAL_NAV', 10_000_000)
    final   = year_results[-1]['final_nav']
    years   = len(year_results)
    total   = (final / initial - 1) * 100
    cagr    = ((final / initial) ** (1/years) - 1) * 100

    all_trades = [t for yr in year_results for t in yr['closed_trades']]
    n          = len(all_trades)
    returns    = [t['return_pct'] for t in all_trades]
    holds      = [t['hold_days']  for t in all_trades]
    zero_days  = sum(1 for h in holds if h == 0)

    print(f"\n{'='*65}")
    print(f"  {test_name}")
    print(f"{'='*65}")
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

    winners = sum(1 for r in returns if r >= 10)
    print(f"\n  Total trades:    {n:,}")
    print(f"  Winners (>10%):  {winners} ({winners/n*100:.1f}%)" if n > 0 else "")
    print(f"  Avg return:      {sum(returns)/n:.2f}%" if n > 0 else "")
    print(f"  Avg hold days:   {sum(holds)/n:.0f}" if n > 0 else "")
    print(f"  Zero-day holds:  {zero_days} ({zero_days/n*100:.1f}%)" if n > 0 else "")
    print(f"  Max drawdown:    {max_dd:.1f}%")

    # [v3] Risk metrics
    if risk_metrics:
        print(f"\n  ── RISK-ADJUSTED METRICS ──────────────────────────")
        print(f"  Sharpe Ratio:    {risk_metrics['sharpe']}"
              f"  (>1.0 good, >2.0 excellent)")
        print(f"  Sortino Ratio:   {risk_metrics['sortino']}")
        print(f"  Calmar Ratio:    {risk_metrics['calmar']}"
              f"  (CAGR / |max DD|)")
        print(f"  Max DD (daily):  {risk_metrics['max_drawdown_pct']:.1f}%"
              f"  (from daily NAV series)")
        print(f"  Trading days:    {risk_metrics['n_trading_days']:,}")

    print(f"\n  FINAL NAV:    ${final:,.0f}")
    print(f"  TOTAL RETURN: {total:+.1f}%")
    print(f"  CAGR:         {cagr:.1f}% per year")
    print(f"  (Raw — apply 15-20% survivorship bias discount)")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN FUNCTION — used by Tests 1-11
# ─────────────────────────────────────────────────────────────────────────────

def run(test_name):
    global CONFIG
    print(f"\n{'='*65}")
    print(f"  {test_name}")
    print(f"{'='*65}")

    try:
        client = get_client()
        print("Connected to Supabase.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    initial        = CONFIG.get('INITIAL_CAPITAL') or CONFIG.get('INITIAL_NAV', 10_000_000)
    nav            = initial
    open_positions = {}
    year_results   = []
    all_daily_nav  = []                          # [v3] accumulate across years

    start_year = CONFIG.get('START_YEAR') or int(CONFIG.get('START_DATE','2016-01-01')[:4])
    end_year   = CONFIG.get('END_YEAR')   or int(CONFIG.get('END_DATE','2026-03-15')[:4])

    for year in range(start_year, end_year + 1):
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
        result     = run_year_v3(year, signals_df, nav, open_positions,
                                 etf_df, macro_df, insider_df, short_df)

        if result is None:
            print(f"  Skipping {year} — no trading days")
            continue

        print_year(result)
        year_results.append(result)
        all_daily_nav.extend(result['daily_nav_log'])     # [v3]
        nav            = result['final_nav']
        open_positions = result['open_positions']

        if (result['zero_day_holds'] > result['n_trades'] * 0.05 and
                result['n_trades'] > 0):
            print(f"\n  STOPPING at {year} — zero-day hold warning.")
            break

    if year_results:
        # [v3] Calculate risk metrics from full daily NAV series
        risk_metrics = calculate_risk_metrics(all_daily_nav, initial)
        print_summary(year_results, test_name, risk_metrics)

    print("DONE.\n")


# ─────────────────────────────────────────────────────────────────────────────
# [v3] run_simulation(config) API — used by sector and universe scripts
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(config):
    """
    Run a full simulation with the given config dict.
    Returns a result dict with all metrics including Sharpe.

    This API is used by sector scripts and universe scripts.
    The global CONFIG is set from the passed config dict.
    """
    global CONFIG
    CONFIG = config

    try:
        client = get_client()
    except Exception as e:
        raise RuntimeError(f"Supabase connection failed: {e}")

    universe  = config.get('UNIVERSE', [])
    start_str = config.get('START_DATE', '2016-01-01')
    end_str   = config.get('END_DATE',   '2026-03-15')
    initial   = config.get('INITIAL_NAV', 10_000_000)

    start_year = int(start_str[:4])
    end_year   = int(end_str[:4])

    nav            = initial
    open_positions = {}
    year_results   = []
    all_daily_nav  = []
    skipped        = []

    for year in range(start_year, end_year + 1):
        dm_df = load_dm(client, year)
        if dm_df.empty:
            continue

        # Check which universe tickers actually have data
        if year == start_year and universe:
            found = set(dm_df['ticker'].unique())
            missing = [t for t in universe if t not in found]
            if missing:
                print(f"  Skipped (no DM data): {', '.join(missing)}")
                skipped.extend(missing)

        hms_df     = load_hms(client, year)
        macro_df   = load_macro(client, year)
        insider_df = load_insider(client, year)
        short_df   = load_short(client, year)

        signals_df = compute_signals(dm_df, hms_df)
        result     = run_year_v3(year, signals_df, nav, open_positions,
                                 pd.DataFrame(), macro_df, insider_df, short_df)

        if result is None:
            continue

        print_year(result)
        year_results.append(result)
        all_daily_nav.extend(result['daily_nav_log'])
        nav            = result['final_nav']
        open_positions = result['open_positions']

        if (result['zero_day_holds'] > result['n_trades'] * 0.05 and
                result['n_trades'] > 0):
            print(f"  STOPPING at {year} — zero-day hold warning.")
            break

    if not year_results:
        return {
            'error': 'No results — check universe and date range',
            'skipped_tickers': skipped
        }

    # Aggregate
    all_trades = [t for yr in year_results for t in yr['closed_trades']]
    n          = len(all_trades)
    returns    = [t['return_pct'] for t in all_trades]
    holds      = [t['hold_days']  for t in all_trades]
    years      = len(year_results)
    final_nav  = year_results[-1]['final_nav']
    total_ret  = (final_nav / initial - 1) * 100
    cagr       = ((final_nav / initial) ** (1/years) - 1) * 100 if years > 0 else 0

    winners   = sum(1 for r in returns if r >= 10)
    hit_rate  = winners / n * 100 if n > 0 else 0
    avg_ret   = sum(returns) / n  if n > 0 else 0
    avg_hold  = sum(holds) / n    if n > 0 else 0

    # Max drawdown from yearly NAV
    peak = initial
    max_dd = 0
    for yr in year_results:
        if yr['final_nav'] < peak:
            dd = (yr['final_nav'] - peak) / peak * 100
            if dd < max_dd: max_dd = dd
        else:
            peak = yr['final_nav']

    # [v3] Risk metrics from daily NAV
    risk = calculate_risk_metrics(all_daily_nav, initial)

    nav_log = [{'date': yr['year'], 'nav': yr['final_nav'],
                'return_pct': yr['year_return']}
               for yr in year_results]

    trade_log = []
    for t in all_trades:
        trade_log.append({
            'date':   t['exit_date'],  'ticker':    t['ticker'],
            'action': 'EXIT',          'shares':    0,
            'price':  t['exit_price'], 'dm':        t['exit_dm'],
            'pnl_pct': t['return_pct'], 'hold_days': t['hold_days'],
            'notes':  f"Entry {t['entry_date']} @ ${t['entry_price']:.2f}"
        })

    return {
        # Core metrics
        'cagr':             round(cagr, 1),
        'total_return':     round(total_ret, 1),
        'max_drawdown':     round(max_dd, 1),
        'final_nav':        round(final_nav, 0),
        'initial_nav':      initial,
        'hit_rate':         round(hit_rate, 1),
        'total_trades':     n,
        'winners':          winners,
        'losers':           n - winners,
        'avg_return':       round(avg_ret, 2),
        'avg_hold_days':    round(avg_hold, 0),

        # [v3] Risk-adjusted metrics
        'sharpe':           risk['sharpe'],
        'sortino':          risk['sortino'],
        'calmar':           risk['calmar'],
        'max_drawdown_daily': risk['max_drawdown_pct'],

        # Logs
        'nav_log':          nav_log,
        'trade_log':        trade_log,
        'daily_nav_log':    all_daily_nav,
        'skipped_tickers':  skipped,
        'year_results':     year_results,
    }
