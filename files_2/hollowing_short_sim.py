"""
HOLLOWING SHORT SIMULATION — Python Version
============================================
Author:  Vikram Amsethi / ARGUS Research
Created: March 23, 2026
Version: 1.0

Purpose:
    Backtest the Hollowing Short thesis — short positions in AI-displaced
    sectors (consulting, wealth management, tools, medtech, insurers,
    construction) using DM signal inversion.

    The long sim enters when DM rises above 65 with momentum.
    This sim enters when DM falls below 35 with momentum falling.
    Exit when DM recovers above 50 (MA20) or time stop triggers.

Philosophy:
    The Hollowing thesis: AI displaces human labor in knowledge work.
    Capital that used to flow INTO these businesses starts flowing out.
    DM measures institutional capital flow — falling DM = capital leaving.
    We short the exit, not the narrative.

Entry:  DM <= 35 AND 5-day MA of DM is falling (ma5_now < ma5_prev)
Exit:   20-day MA of DM rises above 50 OR max hold days reached

Two universe tests run in parallel:
    NARROW  — 16 current Hollowing Short portfolio tickers
    BROADER — ~50 AI displacement tickers across all affected forces

Run sequence:
    python hollowing_short_sim.py --year all        # full decade
    python hollowing_short_sim.py --year 2020       # single year
    python hollowing_short_sim.py --year 2020-2025  # range
    python hollowing_short_sim.py --universe narrow # narrow only
    python hollowing_short_sim.py --universe broad  # broad only

Requirements:
    pip install supabase pandas numpy python-dotenv tqdm

Environment variables (.env):
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-service-role-key

Supabase tables expected:
    dm_daily:
        date        DATE
        ticker      TEXT
        close       FLOAT
        dm          FLOAT       (smoothed DM score, EMA5)
        dm_raw      FLOAT       (raw DM score)

    NOTE: Ask developer to confirm exact column names from Supabase schema.
    Adjust DM_COL constants in CONFIG below if different.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: supabase not installed. Run: pip install supabase")
    sys.exit(1)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Simulation parameters
    'INITIAL_CAPITAL':  1_000_000,
    'ENTRY_DM':         35,         # DM must be <= this to enter short
    'EXIT_MA20':        50,         # 20-day MA above this = exit short
    'MAX_HOLD_DAYS':    60,         # time stop — exit regardless after this many days
    'MAX_POSITIONS':    16,         # max simultaneous short positions
    'MIN_POSITION':     500,        # minimum $ per position

    # Supabase table names
    'DM_TABLE':         'dm_history',

    # Supabase column names — adjust if schema differs
    'DM_DATE_COL':      'date',
    'DM_TICKER_COL':    'ticker',
    'DM_CLOSE_COL':     'close',
    'DM_SCORE_COL':      'dm_smoothed',       # smoothed DM score (EMA5)

    # Year range
    'START_YEAR':       2016,
    'END_YEAR':         2026,
}

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSES
# ─────────────────────────────────────────────────────────────────────────────

# NARROW: current live Hollowing Short portfolio
# Entry prices are for reference only — sim uses actual historical prices
NARROW_UNIVERSE = [
    'BLDR',   # Construction — AI displaces project management
    'ACN',    # Consulting — AI displaces knowledge services
    'LPLA',   # Wealth management — AI displaces advisors
    'LDOS',   # Defense services — AI displaces analysts
    'ARE',    # Real estate — AI displaces office demand
    'SHW',    # Materials — construction/office cycle exposed
    'BAH',    # Consulting/defense services
    'WDAY',   # HR/finance tools — AI replaces workflow
    'RJF',    # Wealth management
    'ADBE',   # Creative tools — AI replaces designers
    'SCHW',   # Wealth management — AI displaces advisors
    'DXCM',   # Medtech — AI displaces monitoring labor
    'UNH',    # Insurers — AI restructures claims/admin
    'CRM',    # Sales tools — AI replaces SDR/CRM workflows
    'NOW',    # IT service management — AI replaces tickets
    'OMC',    # Advertising — AI displaces creative/media buying
]

# BROADER: expanded AI displacement universe across all affected forces
BROAD_UNIVERSE = [
    # Consulting / IT Services
    'ACN', 'CTSH', 'INFY', 'WIT', 'IBM', 'EPAM', 'GLOB', 'EXLS', 'IT',
    # Wealth Management / Financial Services
    'SCHW', 'RJF', 'LPLA', 'IVZ', 'BEN', 'AMP',
    # SaaS Tools (AI replaces workflow)
    'CRM', 'NOW', 'WDAY', 'ADBE', 'INTU', 'VEEV', 'HUBS',
    # Advertising / Media
    'OMC', 'IPG', 'TTD',
    # Staffing / HR
    'MAN', 'ADP', 'PAYX', 'RHI', 'ASGN',
    # Healthcare / Insurers
    'UNH', 'CVS', 'CI', 'HUM', 'MOH', 'ELV', 'DXCM',
    # Construction / Real Estate
    'BLDR', 'ARE', 'SHW',
    # Defense Services
    'BAH', 'LDOS', 'SAIC', 'CACI',
    # Medtech / Admin-heavy
    'MDT', 'BSX', 'ZBH', 'BAX',
]

# Remove duplicates while preserving order
BROAD_UNIVERSE = list(dict.fromkeys(BROAD_UNIVERSE))


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_supabase_client() -> Client:
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    return create_client(url, key)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dm_data(supabase: Client, year: int, universe: List[str]) -> pd.DataFrame:
    """
    Load DM data for a full year from Supabase, filtered to universe tickers.
    Loads 30 days before start for MA calculations.
    Returns DataFrame sorted by (ticker, date).
    """
    load_from = f"{year-1}-12-01"
    end       = f"{year}-12-31"

    print(f"  Loading DM {year} from Supabase ({load_from} → {end}) | {len(universe)} tickers...")

    all_rows  = []
    page_size = 10000
    offset    = 0

    while True:
        resp = (
            supabase.table(CONFIG['DM_TABLE'])
            .select(f"{CONFIG['DM_DATE_COL']},{CONFIG['DM_TICKER_COL']},"
                    f"{CONFIG['DM_CLOSE_COL']},{CONFIG['DM_SCORE_COL']}")
            .gte(CONFIG['DM_DATE_COL'], load_from)
            .lte(CONFIG['DM_DATE_COL'], end)
            .in_(CONFIG['DM_TICKER_COL'], universe)
            .order(CONFIG['DM_DATE_COL'], desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    if not all_rows:
        print(f"  WARNING: No DM data found for {year}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(columns={
        CONFIG['DM_DATE_COL']:   'date',
        CONFIG['DM_TICKER_COL']: 'ticker',
        CONFIG['DM_CLOSE_COL']:  'close',
        CONFIG['DM_SCORE_COL']:  'dm'
    }, inplace=True)

    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df          = df.dropna(subset=['date', 'ticker', 'close', 'dm'])
    df          = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"  DM loaded: {len(df):,} rows | {df['ticker'].nunique()} tickers")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(dm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute short entry/exit signals per ticker per day.

    Entry signal: DM <= ENTRY_DM AND 5d MA falling (ma5_now < ma5_prev)
    Exit signal:  20d MA of DM > EXIT_MA20

    Returns df with columns:
        date, ticker, close, dm, ma5, ma5_prev, ma20, entry_signal, exit_signal
    """
    if dm_df.empty:
        return pd.DataFrame()

    records = []

    for ticker, grp in dm_df.groupby('ticker'):
        grp = grp.sort_values('date').reset_index(drop=True)
        dm_series = grp['dm'].values

        ma5      = pd.Series(dm_series).rolling(5,  min_periods=3).mean().values
        ma5_prev = pd.Series(dm_series).rolling(5,  min_periods=3).mean().shift(1).values
        ma20     = pd.Series(dm_series).rolling(20, min_periods=10).mean().values

        for i, row in grp.iterrows():
            if np.isnan(ma5[i]) or np.isnan(ma20[i]):
                continue

            ma5_now  = ma5[i]
            ma5_p    = ma5_prev[i] if not np.isnan(ma5_prev[i]) else ma5_now
            ma20_now = ma20[i]
            dm_val   = row['dm']

            # Entry: DM at or below threshold AND 5d MA falling
            entry = (dm_val <= CONFIG['ENTRY_DM']) and (ma5_now < ma5_p)

            # Exit: 20d MA recovers above exit threshold
            exit_ = (ma20_now > CONFIG['EXIT_MA20'])

            records.append({
                'date':         row['date'],
                'ticker':       ticker,
                'close':        row['close'],
                'dm':           dm_val,
                'ma5':          ma5_now,
                'ma5_prev':     ma5_p,
                'ma20':         ma20_now,
                'entry_signal': entry,
                'exit_signal':  exit_,
            })

    if not records:
        return pd.DataFrame()

    signals_df = pd.DataFrame(records)
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    return signals_df


# ─────────────────────────────────────────────────────────────────────────────
# DM-TIERED SHORT SIZING
# Larger short when DM is lower — deeper conviction the thesis is working
# ─────────────────────────────────────────────────────────────────────────────

def get_short_alloc(dm: float, nav: float) -> float:
    """
    DM-tiered position sizing for shorts.
    Lower DM = more capital flow leaving = higher conviction short.
    """
    if dm <= 10:   return nav * 0.08   # deeply trapped
    if dm <= 20:   return nav * 0.065  # strong signal
    if dm <= 30:   return nav * 0.05   # standard
    return nav * 0.035                  # threshold entry


# ─────────────────────────────────────────────────────────────────────────────
# YEAR SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_year(
    year:           int,
    signals_df:     pd.DataFrame,
    starting_nav:   float,
    open_positions: dict,
    verbose:        bool = True
) -> dict:
    """
    Simulate one year of short trading.

    open_positions carries over from prior year:
        { ticker: { entry_price, entry_date, entry_dm, shares, alloc } }

    Short P&L: profit when price falls.
        pnl = (entry_price - current_price) * shares
    """
    if signals_df.empty:
        return {
            'year': year, 'starting_nav': starting_nav, 'final_nav': starting_nav,
            'year_return': 0, 'closed_trades': [], 'open_positions': open_positions,
            'cash': starting_nav, 'n_trades': 0, 'hit_rate': 0,
            'avg_return': 0, 'avg_hold_days': 0, 'zero_day_holds': 0,
        }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  YEAR {year}")
        print(f"  Starting NAV: ${starting_nav:,.0f} | Open carries: {len(open_positions)}")
        print(f"{'='*60}")

    year_start   = date(year, 1, 1)
    year_end     = date(year, 12, 31)
    year_signals = signals_df[signals_df['date'] >= year_start].copy()

    # Compute cash from carry-in positions
    # Short positions: cash = NAV - sum(entry_cost of open shorts)
    # entry_cost = entry_price * shares (what we received when we shorted)
    carried_proceeds = sum(p['entry_price'] * p['shares'] for p in open_positions.values())
    cash = starting_nav - carried_proceeds
    # Note: for shorts, cash starts negative for carried positions — that's correct
    # The P&L accrues as price falls

    closed_trades = []
    trading_days  = sorted(year_signals['date'].unique())

    for day in trading_days:
        day_data = year_signals[year_signals['date'] == day]
        day_map  = {row['ticker']: row for _, row in day_data.iterrows()}

        # ── EXITS ─────────────────────────────────────────────────────────────
        to_exit = []
        for ticker, pos in open_positions.items():
            if ticker not in day_map:
                continue
            row      = day_map[ticker]
            held     = (day - pos['entry_date']).days
            time_stop = held >= CONFIG['MAX_HOLD_DAYS']
            dm_exit   = row['exit_signal']

            if dm_exit or time_stop:
                exit_price = row['close']
                pnl_per_share = pos['entry_price'] - exit_price  # short P&L
                pnl_dollar    = pnl_per_share * pos['shares']
                pnl_pct       = pnl_per_share / pos['entry_price'] * 100

                # Return proceeds + P&L to cash
                cash += pos['entry_price'] * pos['shares'] + pnl_dollar

                reason = 'TIME_STOP' if time_stop else 'DM_RECOVERY'

                closed_trades.append({
                    'ticker':      ticker,
                    'entry_date':  pos['entry_date'],
                    'exit_date':   day,
                    'entry_price': pos['entry_price'],
                    'exit_price':  exit_price,
                    'entry_dm':    pos['entry_dm'],
                    'exit_ma20':   row['ma20'],
                    'shares':      pos['shares'],
                    'alloc':       pos['alloc'],
                    'return_pct':  pnl_pct,
                    'pnl_dollar':  pnl_dollar,
                    'hold_days':   held,
                    'exit_reason': reason,
                })
                to_exit.append(ticker)

                if verbose and pnl_pct != 0:
                    sign = '+' if pnl_pct >= 0 else ''
                    print(f"  EXIT  {ticker:6} | {reason} | "
                          f"{pos['entry_date']} → {day} ({held}d) | "
                          f"${pos['entry_price']:.2f} → ${exit_price:.2f} | "
                          f"{sign}{pnl_pct:.1f}%")

        for t in to_exit:
            del open_positions[t]

        # ── ENTRIES ───────────────────────────────────────────────────────────
        current_nav = cash + sum(
            (p['entry_price'] - day_map.get(tk, {}).get('close', p['entry_price'])) * p['shares']
            for tk, p in open_positions.items()
            if tk in day_map
        )
        current_nav = max(current_nav, cash)  # floor at cash if prices not available

        if len(open_positions) < CONFIG['MAX_POSITIONS']:
            for _, row in day_data[day_data['entry_signal']].iterrows():
                ticker = row['ticker']
                if ticker in open_positions:
                    continue
                if len(open_positions) >= CONFIG['MAX_POSITIONS']:
                    break

                alloc  = get_short_alloc(row['dm'], current_nav)
                alloc  = min(alloc, cash)  # can't short more than available cash
                if alloc < CONFIG['MIN_POSITION']:
                    continue

                shares = alloc / row['close']
                if shares <= 0:
                    continue

                # Short entry: we receive cash = shares * price
                cash -= alloc  # reserve as margin/collateral

                open_positions[ticker] = {
                    'entry_price': row['close'],
                    'entry_date':  day,
                    'entry_dm':    row['dm'],
                    'shares':      shares,
                    'alloc':       alloc,
                }

                if verbose:
                    tier = ('8%' if row['dm'] <= 10 else '6.5%' if row['dm'] <= 20
                            else '5%' if row['dm'] <= 30 else '3.5%')
                    print(f"  SHORT {ticker:6} [{tier}] | "
                          f"${row['close']:.2f} | DM: {row['dm']:.1f} | "
                          f"MA5: {row['ma5']:.1f}↓{row['ma5_prev']:.1f} | "
                          f"${alloc:,.0f}")

    # ── YEAR-END MARK TO MARKET ───────────────────────────────────────────────
    last_day   = trading_days[-1] if trading_days else year_end
    last_data  = year_signals[year_signals['date'] == last_day]
    last_map   = {row['ticker']: row for _, row in last_data.iterrows()}

    mark_pnl = 0
    for ticker, pos in open_positions.items():
        if ticker in last_map:
            current_px = last_map[ticker]['close']
            mark_pnl  += (pos['entry_price'] - current_px) * pos['shares']

    final_nav  = cash + sum(p['entry_price'] * p['shares'] for p in open_positions.values()) + mark_pnl
    year_ret   = (final_nav / starting_nav - 1) * 100

    # ── YEAR SUMMARY ──────────────────────────────────────────────────────────
    n        = len(closed_trades)
    returns  = [t['return_pct'] for t in closed_trades]
    winners  = [r for r in returns if r >= 10]
    losers   = [r for r in returns if r < 0]
    avg_ret  = sum(returns) / n if n > 0 else 0
    hit_rate = len(winners) / n * 100 if n > 0 else 0
    holds    = [t['hold_days'] for t in closed_trades]
    avg_hold = sum(holds) / n if n > 0 else 0
    zero_d   = sum(1 for h in holds if h == 0)

    time_stops = sum(1 for t in closed_trades if t['exit_reason'] == 'TIME_STOP')
    dm_exits   = sum(1 for t in closed_trades if t['exit_reason'] == 'DM_RECOVERY')

    if verbose:
        print(f"\n  ── {year} RESULTS ──────────────────────────────────")
        print(f"  Closed trades:    {n}")
        print(f"  Hit rate (>10%):  {hit_rate:.1f}%")
        print(f"  Avg return:       {avg_ret:.2f}%")
        print(f"  Avg hold days:    {avg_hold:.0f}")
        print(f"  Winners:          {len(winners)} | Losers: {len(losers)}")
        print(f"  DM exits:         {dm_exits} | Time stops: {time_stops}")
        print(f"  Zero-day holds:   {zero_d}  ← FLAG IF > 5% of trades")
        print(f"  Open positions:   {len(open_positions)}")
        print(f"  Cash:             ${cash:,.0f}")
        print(f"  Year-end NAV:     ${final_nav:,.0f} ({year_ret:+.2f}%)")

        if zero_d > n * 0.05 and n > 0:
            print(f"\n  ⚠️  WARNING: {zero_d} zero-day holds — investigate data/logic")

        if n > 0:
            print(f"\n  TOP 5 SHORTS:")
            top5 = sorted(closed_trades, key=lambda x: x['return_pct'], reverse=True)[:5]
            for t in top5:
                print(f"    {t['ticker']:6} | In: {t['entry_date']} @ ${t['entry_price']:.2f} | "
                      f"Out: {t['exit_date']} @ ${t['exit_price']:.2f} | "
                      f"{t['return_pct']:+.1f}% | {t['hold_days']}d | {t['exit_reason']}")

            print(f"\n  WORST 3 SHORTS:")
            worst3 = sorted(closed_trades, key=lambda x: x['return_pct'])[:3]
            for t in worst3:
                print(f"    {t['ticker']:6} | In: {t['entry_date']} @ ${t['entry_price']:.2f} | "
                      f"Out: {t['exit_date']} @ ${t['exit_price']:.2f} | "
                      f"{t['return_pct']:+.1f}% | {t['hold_days']}d | {t['exit_reason']}")

    return {
        'year':           year,
        'starting_nav':   starting_nav,
        'final_nav':      final_nav,
        'year_return':    year_ret,
        'closed_trades':  closed_trades,
        'open_positions': open_positions,
        'cash':           cash,
        'n_trades':       n,
        'hit_rate':       hit_rate,
        'avg_return':     avg_ret,
        'avg_hold_days':  avg_hold,
        'zero_day_holds': zero_d,
        'time_stops':     time_stops,
        'dm_exits':       dm_exits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DECADE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_decade_summary(year_results: list, label: str):
    print(f"\n{'='*60}")
    print(f"  HOLLOWING SHORT SIMULATION — {label}")
    print(f"  Entry: DM <= {CONFIG['ENTRY_DM']} + 5d MA falling")
    print(f"  Exit:  20d MA > {CONFIG['EXIT_MA20']} OR {CONFIG['MAX_HOLD_DAYS']}d time stop")
    print(f"{'='*60}")

    initial   = CONFIG['INITIAL_CAPITAL']
    final     = year_results[-1]['final_nav'] if year_results else initial
    total_ret = (final / initial - 1) * 100
    years     = len(year_results)
    cagr      = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    all_trades   = []
    for yr in year_results:
        all_trades.extend(yr['closed_trades'])

    total_trades = len(all_trades)
    all_returns  = [t['return_pct'] for t in all_trades]
    all_holds    = [t['hold_days']  for t in all_trades]
    avg_ret      = sum(all_returns) / total_trades if total_trades > 0 else 0
    hit_rate     = sum(1 for r in all_returns if r >= 10) / total_trades * 100 if total_trades > 0 else 0
    avg_hold     = sum(all_holds)   / total_trades if total_trades > 0 else 0
    time_stops   = sum(yr.get('time_stops', 0) for yr in year_results)
    dm_exits     = sum(yr.get('dm_exits',   0) for yr in year_results)

    print(f"\n  AGGREGATE STATISTICS")
    print(f"  Total trades:        {total_trades:,}")
    print(f"  Hit rate (>10%):     {hit_rate:.1f}%")
    print(f"  Avg return:          {avg_ret:.2f}%")
    print(f"  Avg hold days:       {avg_hold:.0f}")
    print(f"  DM-exit trades:      {dm_exits} ({dm_exits/max(total_trades,1)*100:.1f}%)")
    print(f"  Time-stop trades:    {time_stops} ({time_stops/max(total_trades,1)*100:.1f}%)")

    print(f"\n  CAPITAL PERFORMANCE")
    print(f"  Initial capital:     ${initial:,.0f}")
    print(f"  Final NAV:           ${final:,.0f}")
    print(f"  Total return:        {total_ret:+.1f}%")
    print(f"  CAGR ({years} years):    {cagr:.1f}%")
    print(f"  Note: Apply 15-20% survivorship bias discount")
    print(f"  Adjusted CAGR range: {cagr*0.80:.1f}% — {cagr*0.85:.1f}%")

    print(f"\n  YEARLY NAV PROGRESSION")
    print(f"  {'Year':<6} {'NAV':>12} {'Return':>8} {'Trades':>8} "
          f"{'Hit Rate':>10} {'Avg Hold':>10} {'TimeStops':>10}")
    print(f"  {'-'*66}")

    peak_nav = initial
    max_dd   = 0

    for yr in year_results:
        if yr['final_nav'] < peak_nav:
            dd = (yr['final_nav'] - peak_nav) / peak_nav * 100
            if dd < max_dd:
                max_dd = dd
        else:
            peak_nav = yr['final_nav']

        print(f"  {yr['year']:<6} "
              f"${yr['final_nav']:>11,.0f} "
              f"{yr['year_return']:>+7.1f}% "
              f"{yr['n_trades']:>8} "
              f"{yr['hit_rate']:>9.1f}% "
              f"{yr['avg_hold_days']:>9.0f}d "
              f"{yr.get('time_stops',0):>10}")

    print(f"\n  Max drawdown (peak to trough): {max_dd:.1f}%")

    # Per-ticker summary
    ticker_stats: Dict[str, dict] = {}
    for t in all_trades:
        tk = t['ticker']
        if tk not in ticker_stats:
            ticker_stats[tk] = {'trades': 0, 'winners': 0, 'returns': [], 'holds': []}
        ticker_stats[tk]['trades']  += 1
        ticker_stats[tk]['returns'].append(t['return_pct'])
        ticker_stats[tk]['holds'].append(t['hold_days'])
        if t['return_pct'] >= 10:
            ticker_stats[tk]['winners'] += 1

    print(f"\n  PER-TICKER BREAKDOWN (sorted by avg return)")
    print(f"  {'Ticker':<8} {'Trades':>7} {'Hit Rate':>10} {'Avg Return':>12} {'Avg Hold':>10}")
    print(f"  {'-'*50}")

    sorted_tickers = sorted(
        ticker_stats.items(),
        key=lambda x: sum(x[1]['returns']) / len(x[1]['returns']),
        reverse=True
    )
    for tk, s in sorted_tickers:
        avg_r = sum(s['returns']) / len(s['returns'])
        avg_h = sum(s['holds'])   / len(s['holds'])
        hr    = s['winners'] / s['trades'] * 100
        print(f"  {tk:<8} {s['trades']:>7} {hr:>9.1f}% {avg_r:>+11.1f}% {avg_h:>9.0f}d")

    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Hollowing Short Simulation')
    parser.add_argument('--year', default='all',
                        help='Year to run: 2016, 2020-2025, or all (default: all)')
    parser.add_argument('--universe', default='both',
                        choices=['narrow', 'broad', 'both'],
                        help='Universe to test: narrow (16), broad (~50), or both (default: both)')
    parser.add_argument('--interactive', action='store_true',
                        help='Pause between years for review')
    args = parser.parse_args()

    # Parse year range
    if args.year == 'all':
        years = list(range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1))
    elif '-' in args.year:
        start_y, end_y = args.year.split('-')
        years = list(range(int(start_y), int(end_y) + 1))
    else:
        years = [int(args.year)]

    universes = {}
    if args.universe in ('narrow', 'both'):
        universes['NARROW (16 tickers)'] = NARROW_UNIVERSE
    if args.universe in ('broad', 'both'):
        universes['BROAD (~50 tickers)'] = BROAD_UNIVERSE

    print(f"\nHOLLOWING SHORT SIMULATION")
    print(f"Years: {years[0]} — {years[-1]}")
    print(f"Entry: DM <= {CONFIG['ENTRY_DM']} + 5d MA falling")
    print(f"Exit:  20d MA > {CONFIG['EXIT_MA20']} OR {CONFIG['MAX_HOLD_DAYS']}d time stop")
    print(f"Max positions: {CONFIG['MAX_POSITIONS']}")
    print(f"Initial capital: ${CONFIG['INITIAL_CAPITAL']:,}")

    print("\nConnecting to Supabase...")
    try:
        supabase = get_supabase_client()
        print("Connected.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    all_universe_results = {}

    for label, universe in universes.items():
        print(f"\n\n{'#'*60}")
        print(f"  UNIVERSE: {label}")
        print(f"  Tickers: {', '.join(universe)}")
        print(f"{'#'*60}")

        nav            = CONFIG['INITIAL_CAPITAL']
        open_positions = {}
        year_results   = []

        for year in years:
            dm_df = load_dm_data(supabase, year, universe)

            if dm_df.empty:
                print(f"  Skipping {year} — no DM data")
                continue

            signals_df = compute_signals(dm_df)

            result = run_year(
                year           = year,
                signals_df     = signals_df,
                starting_nav   = nav,
                open_positions = open_positions,
                verbose        = True
            )

            year_results.append(result)
            nav            = result['final_nav']
            open_positions = result['open_positions']

            if args.interactive and year != years[-1]:
                resp = input(f"\nContinue to {year+1}? (y/n): ").strip().lower()
                if resp != 'y':
                    print("Simulation stopped by user.")
                    break

        if len(year_results) > 1:
            print_decade_summary(year_results, label)

        all_universe_results[label] = year_results

    # Side-by-side comparison if both universes ran
    if len(all_universe_results) == 2:
        labels = list(all_universe_results.keys())
        print(f"\n\n{'='*60}")
        print(f"  UNIVERSE COMPARISON")
        print(f"  {labels[0]:<30} vs  {labels[1]}")
        print(f"{'='*60}")

        for lbl, results in all_universe_results.items():
            if not results:
                continue
            initial  = CONFIG['INITIAL_CAPITAL']
            final    = results[-1]['final_nav']
            cagr     = ((final / initial) ** (1 / len(results)) - 1) * 100
            all_t    = [t for yr in results for t in yr['closed_trades']]
            hr       = sum(1 for t in all_t if t['return_pct'] >= 10) / max(len(all_t), 1) * 100
            avg_r    = sum(t['return_pct'] for t in all_t) / max(len(all_t), 1)
            print(f"\n  {lbl}")
            print(f"    CAGR:       {cagr:.1f}%")
            print(f"    Hit rate:   {hr:.1f}%")
            print(f"    Avg return: {avg_r:.2f}%")
            print(f"    Trades:     {len(all_t)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
