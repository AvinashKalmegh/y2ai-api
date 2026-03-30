"""
ARGUS CAPITAL FLOW SIMULATION ENGINE — v1
==========================================
Shared core engine for all five strategy scripts.
Do not run directly — imported by each strategy script.

Developer notes:
- Adjust CONFIG field names to match your Supabase column names
- DM_SCORE_COL should point to dm_smoothed (or equivalent)
- PRICE_COL should point to the daily close price column
- All five strategy scripts use this same engine
- Risk-free rate: 4% annual (RISK_FREE_ANNUAL)
- Sharpe/Sortino/Calmar calculated from daily mark-to-market NAV series

Output files (written to same directory as strategy script):
- trades_<strategy_slug>.csv  : one row per closed trade — the audit trail
- nav_<strategy_slug>.csv     : one row per trading day — daily NAV log
"""

import os
import sys
import csv
import re
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

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
RISK_FREE_ANNUAL    = 0.04
TRADING_DAYS_YEAR   = 252
ZERO_DAY_THRESHOLD  = 0.05   # stop if >5% of trades are zero-day holds


# ── SUPABASE CONNECTION ───────────────────────────────────────────────────────

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


# ── DATA LOADER ───────────────────────────────────────────────────────────────

def load_year(client, cfg, year):
    """
    Load one year of DM data from Supabase.
    Loads Dec of prior year for MA warm-up.
    Returns DataFrame with columns: date, ticker, close, dm
    """
    table      = cfg['DM_TABLE']
    date_col   = cfg['DM_DATE_COL']
    ticker_col = cfg['DM_TICKER_COL']
    price_col  = cfg['DM_PRICE_COL']
    dm_col     = cfg['DM_SCORE_COL']

    load_from = f"{year - 1}-12-01"
    load_to   = f"{year}-12-31"

    print(f"  Loading {year}...")
    rows, offset, page = [], 0, 10000
    while True:
        r = (client.table(table)
             .select(f"{date_col},{ticker_col},{price_col},{dm_col}")
             .gte(date_col, load_from)
             .lte(date_col, load_to)
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
        ticker_col: 'ticker',
        price_col:  'close',
        dm_col:     'dm',
    }, inplace=True)

    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date', 'ticker', 'close', 'dm'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"  DM: {len(df):,} rows | {df['ticker'].nunique()} tickers")
    return df


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────

def compute_signals(df, cfg):
    """
    Compute entry and exit signals for each ticker on each date.

    Entry signals vary by strategy — see CONFIG:
      ENTRY_DM_MIN              : minimum CF score to enter
      ENTRY_MA5_RISING          : True = require 5d MA of DM to be rising
      ENTRY_PRICE_ABOVE_MA20    : True = require price > 20d price MA (Strategy 5)

    Exit signal: 20d MA of DM drops below EXIT_MA20_THRESHOLD
    """
    entry_dm   = cfg['ENTRY_DM_MIN']
    exit_ma20  = cfg['EXIT_MA20_THRESHOLD']
    req_ma5    = cfg.get('ENTRY_MA5_RISING', True)
    req_price  = cfg.get('ENTRY_PRICE_ABOVE_MA20', False)

    df = df.sort_values(['ticker', 'date']).copy()

    # DM moving averages
    df['dm_ma5']      = df.groupby('ticker')['dm'].transform(
        lambda x: x.rolling(5,  min_periods=1).mean())
    df['dm_ma5_prev'] = df.groupby('ticker')['dm_ma5'].transform(
        lambda x: x.shift(1))
    df['dm_ma20']     = df.groupby('ticker')['dm'].transform(
        lambda x: x.rolling(20, min_periods=1).mean())

    # Price MA20 (for Strategy 5)
    if req_price:
        df['price_ma20'] = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(20, min_periods=1).mean())
    else:
        df['price_ma20'] = df['close']  # dummy

    # Entry conditions
    cond_dm    = df['dm'] >= entry_dm
    cond_ma5   = (df['dm_ma5'] > df['dm_ma5_prev'].fillna(0)) if req_ma5 else True
    cond_price = (df['close'] > df['price_ma20']) if req_price else True

    df['entry_signal'] = cond_dm & cond_ma5 & cond_price

    # Exit condition
    df['exit_signal'] = df['dm_ma20'] < exit_ma20

    return df


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def get_allocation(cfg, dm_score, current_nav, cash, n_open, max_pos):
    """
    Return dollar allocation for a new position.

    SIZING_MODE options:
      'equal_weight' : cash / remaining slots (Strategies 1, 2, 3, 5)
      'dm_tiered'    : % of initial capital based on DM score (Strategy 4)
    """
    mode    = cfg.get('SIZING_MODE', 'equal_weight')
    initial = cfg['INITIAL_CAPITAL']
    min_pos = cfg.get('MIN_POSITION', 500)

    if mode == 'dm_tiered':
        if dm_score >= 90:
            pct = 0.080
        elif dm_score >= 80:
            pct = 0.065
        elif dm_score >= 70:
            pct = 0.050
        else:
            pct = 0.035
        alloc = min(initial * pct, cash)
    else:
        slots = max_pos - n_open
        alloc = min(cash / max(1, slots), cash)

    return alloc if alloc >= min_pos else 0


# ── YEAR RUNNER ───────────────────────────────────────────────────────────────

def run_year(year, signals_df, starting_nav, open_positions, cfg):
    """
    Run one calendar year of simulation.
    Returns year result dict including daily_nav_log.
    Closed trades now include exit_dm for the audit trail.
    """
    max_pos   = cfg.get('MAX_POSITIONS', 20)
    min_pos   = cfg.get('MIN_POSITION', 500)

    y_start = date(year, 1, 1)
    y_end   = date(year, 12, 31)
    ydf     = signals_df[
        (signals_df['date'] >= y_start) &
        (signals_df['date'] <= y_end)
    ].copy()

    trading_days = sorted(ydf['date'].unique())
    if not trading_days:
        return None

    # Fast lookup: ticker → date → row
    lookup = {}
    for _, row in ydf.iterrows():
        tk = row['ticker']
        dt = row['date']
        if tk not in lookup:
            lookup[tk] = {}
        lookup[tk][dt] = row

    invested = sum(p['shares'] * p['entry_price'] for p in open_positions.values())
    cash     = starting_nav - invested

    closed_trades = []
    daily_nav_log = []
    all_tickers   = list(lookup.keys())

    for day in trading_days:

        # ── EXITS ──────────────────────────────────────────────────────────
        to_close = [
            tk for tk, pos in open_positions.items()
            if lookup.get(tk, {}).get(day, {}).get('exit_signal', False)
        ]
        for tk in to_close:
            pos      = open_positions.pop(tk)
            d        = lookup[tk][day]
            ret      = (d['close'] - pos['entry_price']) / pos['entry_price'] * 100
            hold     = (day - pos['entry_date']).days
            proceeds = pos['shares'] * d['close']
            cash    += proceeds
            closed_trades.append({
                'ticker':       tk,
                'entry_date':   str(pos['entry_date']),
                'entry_price':  round(pos['entry_price'], 4),
                'entry_dm':     round(pos['entry_dm'], 2),
                'entry_shares': pos['shares'],
                'entry_cost':   round(pos['shares'] * pos['entry_price'], 2),
                'exit_date':    str(day),
                'exit_price':   round(d['close'], 4),
                'exit_dm':      round(d['dm'], 2),
                'exit_dm_ma20': round(d['dm_ma20'], 2),
                'proceeds':     round(proceeds, 2),
                'return_pct':   round(ret, 4),
                'hold_days':    hold,
                'pnl':          round(proceeds - pos['shares'] * pos['entry_price'], 2),
            })

        # ── ENTRIES ────────────────────────────────────────────────────────
        if len(open_positions) < max_pos and cash >= min_pos:
            held  = set(open_positions.keys())
            cands = [
                (tk, lookup[tk][day])
                for tk in all_tickers
                if tk not in held
                and day in lookup.get(tk, {})
                and lookup[tk][day]['entry_signal']
                and not lookup[tk][day]['exit_signal']
            ]
            cands.sort(key=lambda x: x[1]['dm'], reverse=True)

            for tk, d in cands:
                if len(open_positions) >= max_pos:
                    break
                if cash < min_pos:
                    break

                alloc = get_allocation(cfg, d['dm'], starting_nav, cash,
                                       len(open_positions), max_pos)
                if alloc < min_pos:
                    break

                price  = d['close']
                if price <= 0:
                    continue
                shares = int(alloc // price)
                if shares <= 0:
                    continue

                cash -= shares * price
                open_positions[tk] = {
                    'shares':      shares,
                    'entry_price': price,
                    'entry_date':  day,
                    'entry_dm':    d['dm'],
                }

        # ── DAILY MARK-TO-MARKET NAV ───────────────────────────────────────
        mtm = cash
        for tk, pos in open_positions.items():
            d    = lookup.get(tk, {}).get(day)
            mtm += pos['shares'] * (d['close'] if d is not None else pos['entry_price'])
        daily_nav_log.append({
            'date':          str(day),
            'nav':           round(mtm, 2),
            'cash':          round(cash, 2),
            'open_positions': len(open_positions),
        })

    # Year-end NAV
    last_day  = trading_days[-1]
    final_nav = cash
    for tk, pos in open_positions.items():
        d = lookup.get(tk, {}).get(last_day)
        final_nav += pos['shares'] * (d['close'] if d is not None else pos['entry_price'])

    n         = len(closed_trades)
    returns   = [t['return_pct'] for t in closed_trades]
    holds     = [t['hold_days']  for t in closed_trades]
    winners   = sum(1 for r in returns if r >= 10)
    zero_days = sum(1 for h in holds if h == 0)
    year_ret  = (final_nav / starting_nav - 1) * 100

    return {
        'year':           year,
        'starting_nav':   starting_nav,
        'final_nav':      final_nav,
        'year_return':    year_ret,
        'n_trades':       n,
        'hit_rate':       winners / n * 100 if n > 0 else 0,
        'avg_return':     sum(returns) / n  if n > 0 else 0,
        'avg_hold_days':  sum(holds) / n    if n > 0 else 0,
        'zero_day_holds': zero_days,
        'closed_trades':  closed_trades,
        'open_positions': open_positions,
        'cash':           cash,
        'daily_nav_log':  daily_nav_log,
    }


# ── RISK METRICS ──────────────────────────────────────────────────────────────

def calculate_risk_metrics(daily_nav_log, initial_nav):
    """
    Calculate Sharpe, Sortino, Calmar from daily NAV log.
    Risk-free rate: 4% annual.
    """
    if len(daily_nav_log) < 10:
        return {
            'sharpe': None, 'sortino': None,
            'calmar': None, 'max_drawdown_daily': None
        }

    navs = pd.Series(
        [e['nav'] for e in daily_nav_log],
        index=pd.to_datetime([e['date'] for e in daily_nav_log])
    )

    daily_returns = navs.pct_change().dropna()
    if len(daily_returns) < 5:
        return {
            'sharpe': None, 'sortino': None,
            'calmar': None, 'max_drawdown_daily': None
        }

    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS_YEAR
    excess   = daily_returns - rf_daily

    # Sharpe
    sharpe = (excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_YEAR)
              if excess.std() > 0 else 0.0)

    # Sortino
    downside = excess[excess < 0]
    sortino  = (excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_YEAR)
                if len(downside) > 1 and downside.std() > 0 else 0.0)

    # Max drawdown
    rolling_peak = navs.cummax()
    drawdowns    = (navs - rolling_peak) / rolling_peak * 100
    max_dd       = drawdowns.min()

    # CAGR from daily series
    days = (navs.index[-1] - navs.index[0]).days
    cagr = ((navs.iloc[-1] / initial_nav) ** (365.25 / days) - 1) * 100 if days > 0 else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        'sharpe':             round(sharpe, 2),
        'sortino':            round(sortino, 2),
        'calmar':             round(calmar, 2),
        'max_drawdown_daily': round(max_dd, 1),
        'n_trading_days':     len(daily_returns),
    }


# ── CSV OUTPUT ────────────────────────────────────────────────────────────────

def make_slug(strategy_name):
    """Convert strategy name to a safe filename slug."""
    slug = strategy_name.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug[:40]


def write_trade_log(strategy_name, all_trades, cfg):
    """
    Write all closed trades to CSV.
    One row per trade — the complete audit trail.
    Columns: strategy, ticker, entry_date, entry_price, entry_dm,
             entry_shares, entry_cost, exit_date, exit_price, exit_dm,
             exit_dm_ma20, proceeds, return_pct, hold_days, pnl
    """
    slug     = make_slug(strategy_name)
    filename = f"trades_{slug}.csv"

    fieldnames = [
        'strategy', 'ticker',
        'entry_date', 'entry_price', 'entry_dm', 'entry_shares', 'entry_cost',
        'exit_date',  'exit_price',  'exit_dm',  'exit_dm_ma20',
        'proceeds', 'return_pct', 'hold_days', 'pnl',
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in all_trades:
            row = {'strategy': strategy_name}
            row.update(t)
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    n_trades  = len(all_trades)
    n_winners = sum(1 for t in all_trades if t['return_pct'] >= 0)
    total_pnl = sum(t['pnl'] for t in all_trades)
    print(f"  [Trade log] {filename} — {n_trades:,} trades | "
          f"{n_winners} winners | Total P&L: ${total_pnl:,.0f}")
    return filename


def write_nav_log(strategy_name, all_daily_nav, cfg):
    """
    Write daily NAV history to CSV.
    One row per trading day — allows independent Sharpe verification.
    Columns: strategy, date, nav, cash, open_positions
    """
    slug     = make_slug(strategy_name)
    filename = f"nav_{slug}.csv"

    fieldnames = ['strategy', 'date', 'nav', 'cash', 'open_positions']

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_daily_nav:
            row = {'strategy': strategy_name}
            row.update(entry)
            writer.writerow(row)

    print(f"  [NAV log]   {filename} — {len(all_daily_nav):,} trading days")
    return filename


# ── PRINT HELPERS ─────────────────────────────────────────────────────────────

def print_year(r):
    print(f"  {r['year']} | NAV: ${r['final_nav']:>12,.0f} | "
          f"Return: {r['year_return']:>+7.1f}% | "
          f"Trades: {r['n_trades']:>5} | "
          f"Hit: {r['hit_rate']:>5.1f}% | "
          f"AvgHold: {r['avg_hold_days']:>5.0f}d | "
          f"ZeroD: {r['zero_day_holds']:>4}")

    if r['n_trades'] > 0 and r['zero_day_holds'] / r['n_trades'] > ZERO_DAY_THRESHOLD:
        print(f"\n  ⚠️  ZERO-DAY HOLD WARNING: {r['zero_day_holds']} of {r['n_trades']} trades")
        print("  STOPPING — zero-day hold threshold exceeded. Report this error.")
        return False
    return True


def print_summary(strategy_name, year_results, all_daily_nav, all_trades, cfg):
    initial = cfg['INITIAL_CAPITAL']
    final   = year_results[-1]['final_nav']
    years   = len(year_results)
    total   = (final / initial - 1) * 100
    cagr    = ((final / initial) ** (1 / years) - 1) * 100

    n         = len(all_trades)
    returns   = [t['return_pct'] for t in all_trades]
    holds     = [t['hold_days']  for t in all_trades]
    zero_days = sum(1 for h in holds if h == 0)

    peak   = initial
    max_dd = 0
    for yr in year_results:
        if yr['final_nav'] < peak:
            dd = (yr['final_nav'] - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
        else:
            peak = yr['final_nav']

    risk = calculate_risk_metrics(all_daily_nav, initial)

    w = '=' * 65
    print(f"\n{w}")
    print(f"  {strategy_name}")
    print(f"{w}")
    print(f"\n  {'Year':<6} {'NAV':>12} {'Ret%':>8} {'Trades':>7} "
          f"{'Hit%':>6} {'AvgHold':>8} {'ZeroD':>6}")
    print(f"  {'-' * 57}")
    for yr in year_results:
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
    print(f"\n  ── RISK-ADJUSTED ──────────────────────────────────")
    print(f"  Sharpe Ratio:    {risk['sharpe']}  (>1.0 good, >2.0 excellent)")
    print(f"  Sortino Ratio:   {risk['sortino']}")
    print(f"  Calmar Ratio:    {risk['calmar']}  (CAGR / |max DD|)")
    print(f"  Max DD (daily):  {risk['max_drawdown_daily']}%")
    print(f"  Trading days:    {risk.get('n_trading_days', 'N/A'):,}")
    print(f"\n  FINAL NAV:    ${final:,.0f}")
    print(f"  TOTAL RETURN: {total:+.1f}%")
    print(f"  CAGR:         {cagr:.1f}% per year")
    print(f"  (Raw — apply 15-20% survivorship bias discount)")
    print(f"{w}\n")


# ── MAIN RUN FUNCTION ─────────────────────────────────────────────────────────

def run(strategy_name, cfg):
    """
    Main entry point. Called by each strategy script.
    Produces console output + two CSV files (trade log, NAV log).
    """
    print(f"\n{'=' * 65}")
    print(f"  {strategy_name}")
    print(f"  Entry DM: {cfg['ENTRY_DM_MIN']} | "
          f"Exit MA20: {cfg['EXIT_MA20_THRESHOLD']} | "
          f"Sizing: {cfg.get('SIZING_MODE', 'equal_weight')} | "
          f"MA5 rising: {cfg.get('ENTRY_MA5_RISING', True)} | "
          f"Price>MA20: {cfg.get('ENTRY_PRICE_ABOVE_MA20', False)}")
    print(f"{'=' * 65}\n")

    try:
        client = get_client()
        print("Connected to Supabase.")
    except Exception as e:
        print(f"ERROR connecting to Supabase: {e}")
        sys.exit(1)

    initial        = cfg['INITIAL_CAPITAL']
    nav            = initial
    open_positions = {}
    year_results   = []
    all_daily_nav  = []
    all_trades     = []

    start_year = cfg.get('START_YEAR', 2016)
    end_year   = cfg.get('END_YEAR',   2026)

    for year in range(start_year, end_year + 1):
        raw_df = load_year(client, cfg, year)
        if raw_df.empty:
            print(f"  Skipping {year} — no data")
            continue

        signals_df = compute_signals(raw_df, cfg)
        result     = run_year(year, signals_df, nav, open_positions, cfg)

        if result is None:
            print(f"  Skipping {year} — no trading days in range")
            continue

        ok = print_year(result)
        year_results.append(result)
        all_daily_nav.extend(result['daily_nav_log'])
        all_trades.extend(result['closed_trades'])
        nav            = result['final_nav']
        open_positions = result['open_positions']

        if not ok:
            print(f"\nStopped at {year} due to zero-day hold error.")
            break

    if year_results:
        print_summary(strategy_name, year_results, all_daily_nav, all_trades, cfg)

        # ── WRITE AUDIT FILES ──────────────────────────────────────────────
        print("  ── OUTPUT FILES ────────────────────────────────────")
        write_trade_log(strategy_name, all_trades, cfg)
        write_nav_log(strategy_name, all_daily_nav, cfg)
        print(f"  Files written to: {os.getcwd()}")
        print(f"{'=' * 65}\n")

    print("DONE.\n")
