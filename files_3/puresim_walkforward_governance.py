"""
puresim_walkforward_governance.py
==================================
ARGUS Research — Shadow_Rotation Walkforward with Dynamic Universe Governance
True point-in-time validation of the two-layer methodology

PURPOSE:
    Validates the full live methodology as it actually operates:

    Layer 1 (monthly):  Universe governance using point-in-time hit rates
    Layer 2 (daily):    Shadow_Rotation on the curated universe

    No look-ahead. Every decision uses only data available at that historical
    moment. This is the rigorous validation.

METHODOLOGY:
    Composite (matches live Shadow_Rotation production code):
        composite = EMA5 * 0.5 + EMA_trend (capped ±20) + PnL% (capped -15/+30)

    Entry: DM >= 65 AND EMA5 rising AND Price > 20d Price MA AND DM_MA20 >= 50
    Exit:  DM_MA20 < 50 OR loss < -15%

    Governance (first-Monday monthly):
        Promote to ACTIVE: HR >= 65 AND signals >= 10 AND ETF correlation >= 0.35
        Demote to WATCH:   HR < 60 with signals >= 10
        Demote to THIN:    signals < 10
        Hysteresis 60-65:  hold current status

DATA WINDOW:
    DM history starts Jan 2016 (Supabase). Need 3-year hit-rate window.
    Simulation runs Jan 2019 -> May 2026 (7.5 years, 3 regime cycles).
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from dotenv import load_dotenv
load_dotenv()

# == CONFIGURATION ============================================================

SUPABASE_URL    = os.environ.get('SUPABASE_URL')
SUPABASE_KEY    = os.environ.get('SUPABASE_KEY')

DATA_LOAD_FROM  = '2016-01-01'
SIM_START_DATE  = '2019-01-01'
SIM_END_DATE    = '2026-05-15'

STARTING_NAV    = 1_000_000.0
RISK_FREE_DAILY = 0.043 / 252

DM_ENTRY        = 65
DM_EXIT_MA      = 50
LOSS_LIMIT      = -0.15

ROTATION_THRESHOLD = 20
USE_ROTATION       = True

PROMOTE_HR_MIN        = 65.0
PROMOTE_SIGNALS_MIN   = 10
PROMOTE_CORR_MIN      = 0.35
DEMOTE_WATCH_HR_MAX   = 60.0
DEMOTE_THIN_SIG_MAX   = 10

HIT_RATE_WINDOW_YEARS = 3
HIT_RETURN_THRESHOLD  = 0.10
HIT_WINDOW_DAYS       = 90
BUILD_DM_THRESHOLD    = 70
BUILD_MIN_DAYS        = 3

CORR_WINDOW_DAYS = 60

POSITION_CAPS = [5, 10, 20, 30]

OUTPUT_DIR = './results_walkforward_governance'

ETF_REFERENCE_CSV = 'ETF_Reference.csv'


# == SUPABASE PAGINATED LOADER (year-by-year with retry) ======================

def _paginated_yearly_load(client, table, columns, date_from, date_to,
                           filter_fn=None, label=None):
    """Load a date-range from Supabase year-by-year with retry on transient errors.
    Matches the pattern in dm_crossover_hitrate.py to avoid 57014 timeouts.
    filter_fn: optional callable that takes a query builder and returns it (for .in_ etc).
    """
    all_rows = []
    batch_size = 10000

    start_year = int(str(date_from)[:4])
    end_year   = int(str(date_to)[:4])

    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end   = f"{year}-12-31"
        offset  = 0

        while True:
            attempt, cur = 0, batch_size
            while True:
                try:
                    q = (client.table(table)
                         .select(columns)
                         .gte('date', y_start)
                         .lte('date', y_end)
                         .order('date').order('ticker')
                         .range(offset, offset + cur - 1))
                    if filter_fn is not None:
                        q = filter_fn(q)
                    response = q.execute()
                    break
                except Exception as e:
                    msg, etype = str(e), type(e).__name__
                    attempt += 1
                    transient = (
                        "57014" in msg or "502" in msg or "503" in msg or
                        "504" in msg or "timeout" in msg.lower() or
                        "Bad Gateway" in msg
                    )
                    if attempt >= 6 or not transient:
                        raise
                    cur = max(1000, cur // 2)
                    print(f"\n    [retry {attempt}] {table} year={year} "
                          f"page={cur} sleep {2**attempt}s ({etype})")
                    time.sleep(2 ** attempt)

            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)
            offset += cur
            if len(rows) < cur:
                break

        print(f"    {label or table} {year}: {len(all_rows):,} total rows")

    return all_rows


# == UNIVERSE LOADER ==========================================================

def load_candidate_pool():
    """Load the candidate ticker pool from puresim_universe.csv."""
    path = 'puresim_universe.csv'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    ticker_col = next((c for c in df.columns
                       if c.lower() in ('ticker', 'symbol', 'stock')), None)
    if not ticker_col:
        print(f"ERROR: No Ticker column in {path}. Columns: {list(df.columns)}")
        sys.exit(1)

    tickers = list(dict.fromkeys(
        df[ticker_col].dropna().str.strip().str.upper().tolist()))
    tickers = [t for t in tickers if t]
    print(f"  Candidate pool: {len(tickers)} tickers")
    return tickers


# == DATA LOADING =============================================================

def load_dm_history(candidate_pool):
    """Load DM history from Supabase dm_history (col: dm_smoothed)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    print(f"\n  Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"  Loading dm_history ({DATA_LOAD_FROM} to {SIM_END_DATE}) year-by-year...")

    all_rows = _paginated_yearly_load(
        client, 'dm_history',
        'date,ticker,close,dm_smoothed',
        DATA_LOAD_FROM, SIM_END_DATE,
        label='dm_history')

    print(f"  Total: {len(all_rows):,} rows")
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={'dm_smoothed': 'dm'})
    df['date']  = pd.to_datetime(df['date'])
    df['dm']    = pd.to_numeric(df['dm'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['date', 'dm', 'close'])
    df = df.rename(columns={'date': 'Date', 'ticker': 'Ticker',
                             'close': 'Price', 'dm': 'DM'})
    df = df[df['Ticker'].isin(candidate_pool)].copy()
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f"  After filter: {df['Ticker'].nunique()} tickers | "
          f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def load_etf_data(etf_tickers):
    """Load ETF price history from Supabase price_history table."""
    print(f"\n  Loading ETF prices for {len(etf_tickers)} ETFs from price_history...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    etf_list = list(etf_tickers)

    def add_etf_filter(q):
        return q.in_('ticker', etf_list)

    all_rows = _paginated_yearly_load(
        client, 'price_history',
        'date,ticker,close',
        DATA_LOAD_FROM, SIM_END_DATE,
        filter_fn=add_etf_filter,
        label='price_history(etfs)')

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  WARNING: No ETF data returned.")
        return df
    df['date']  = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['date', 'close'])
    df = df.rename(columns={'date': 'Date', 'ticker': 'Ticker',
                             'close': 'Price'})
    print(f"  ETF data: {len(df):,} rows, {df['Ticker'].nunique()} ETFs")
    return df


def add_indicators(df):
    """Add DM_MA20, EMA5, EMA5_prev, PriceMA20."""
    g = df.groupby('Ticker')
    df['DM_MA20']   = g['DM'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['EMA5']      = g['DM'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['EMA5_prev'] = g['EMA5'].shift(1)
    df['PriceMA20'] = g['Price'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    return df


def load_etf_mapping():
    """Load ticker -> ETF mapping from ETF_Reference.csv."""
    if not os.path.exists(ETF_REFERENCE_CSV):
        print(f"  WARNING: {ETF_REFERENCE_CSV} not found. ETF correlation gate disabled.")
        return {}

    df = pd.read_csv(ETF_REFERENCE_CSV)
    df.columns = [c.strip() for c in df.columns]
    if 'Ticker' not in df.columns or 'Current_ETF' not in df.columns:
        print(f"  WARNING: {ETF_REFERENCE_CSV} missing Ticker/Current_ETF cols. "
              "Correlation gate disabled.")
        return {}

    mapping = {}
    for _, r in df.iterrows():
        t = str(r['Ticker']).strip().upper()
        e = str(r.get('Current_ETF', '')).strip().upper()
        if t and e and e.lower() != 'nan':
            mapping[t] = e

    n_unique_etfs = len(set(mapping.values()))
    print(f"  Loaded ETF mapping: {len(mapping)} tickers -> {n_unique_etfs} unique ETFs")
    return mapping


# == POINT-IN-TIME HIT RATE ===================================================

def compute_hit_rate_as_of(df_ticker, as_of_date, window_years=3):
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(days=int(window_years * 365.25))

    window_data = df_ticker[
        (df_ticker['Date'] >= window_start) &
        (df_ticker['Date'] < as_of)
    ].copy().reset_index(drop=True)

    if len(window_data) < 30:
        return (0.0, 0, 0)

    window_data['build_flag'] = (window_data['DM'] >= BUILD_DM_THRESHOLD).astype(int)
    window_data['build_streak'] = window_data['build_flag'].groupby(
        (window_data['build_flag'] != window_data['build_flag'].shift()).cumsum()
    ).cumsum()

    window_data['signal_start'] = (
        (window_data['build_streak'] >= BUILD_MIN_DAYS) &
        (window_data['build_streak'].shift(1) < BUILD_MIN_DAYS)
    )

    signals = window_data[window_data['signal_start']].copy()
    if len(signals) == 0:
        return (0.0, 0, 0)

    hits = 0
    valid_signals = 0

    for _, sig in signals.iterrows():
        sig_date = sig['Date']
        sig_price = sig['Price']

        forward = window_data[
            (window_data['Date'] > sig_date) &
            (window_data['Date'] <= sig_date + pd.Timedelta(days=int(HIT_WINDOW_DAYS * 1.5)))
        ].head(HIT_WINDOW_DAYS)

        if len(forward) < HIT_WINDOW_DAYS * 0.7:
            continue

        valid_signals += 1
        max_price = forward['Price'].max()
        if (max_price - sig_price) / sig_price >= HIT_RETURN_THRESHOLD:
            hits += 1

    if valid_signals == 0:
        return (0.0, 0, 0)

    hit_rate = (hits / valid_signals) * 100
    return (hit_rate, valid_signals, hits)


def compute_all_hit_rates_as_of(df, as_of_date, candidate_pool):
    results = {}
    df_by_ticker = {t: g for t, g in df.groupby('Ticker')}
    for ticker in candidate_pool:
        if ticker not in df_by_ticker:
            results[ticker] = {'hit_rate': 0.0, 'signals': 0, 'hits': 0}
            continue
        hr, sigs, hits = compute_hit_rate_as_of(df_by_ticker[ticker], as_of_date)
        results[ticker] = {'hit_rate': hr, 'signals': sigs, 'hits': hits}
    return results


# == POINT-IN-TIME CORRELATION ================================================

def compute_etf_correlation_as_of(stock_df, etf_df, as_of_date,
                                    window_days=CORR_WINDOW_DAYS):
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(days=int(window_days * 1.5))

    stock_window = stock_df[
        (stock_df['Date'] >= window_start) &
        (stock_df['Date'] < as_of)
    ].copy()
    etf_window = etf_df[
        (etf_df['Date'] >= window_start) &
        (etf_df['Date'] < as_of)
    ].copy()

    if len(stock_window) < window_days * 0.5 or len(etf_window) < window_days * 0.5:
        return None

    stock_window['ret'] = stock_window['Price'].pct_change()
    etf_window['ret'] = etf_window['Price'].pct_change()

    merged = pd.merge(
        stock_window[['Date', 'ret']],
        etf_window[['Date', 'ret']],
        on='Date', suffixes=('_stock', '_etf')
    ).dropna()

    if len(merged) < window_days * 0.5:
        return None

    merged = merged.tail(window_days)
    if len(merged) < 20:
        return None

    corr = merged['ret_stock'].corr(merged['ret_etf'])
    return corr if not pd.isna(corr) else None


# == UNIVERSE STATE MANAGER ===================================================

class UniverseState:
    def __init__(self, candidate_pool):
        self.status = {t: 'THIN' for t in candidate_pool}
        self.transitions = []

    def get_active(self):
        return [t for t, s in self.status.items() if s == 'ACTIVE']

    def get_watch(self):
        return [t for t, s in self.status.items() if s == 'WATCH']

    def apply_governance(self, as_of_date, hit_rate_results, correlation_results, cap):
        promotions = 0
        demotions_watch = 0
        demotions_thin = 0
        hysteresis_holds = 0

        for ticker, current_status in list(self.status.items()):
            hr_data = hit_rate_results.get(ticker,
                {'hit_rate': 0.0, 'signals': 0, 'hits': 0})
            hr = hr_data['hit_rate']
            sigs = hr_data['signals']
            corr = correlation_results.get(ticker)

            new_status = current_status
            reason = ''

            if sigs < DEMOTE_THIN_SIG_MAX:
                new_status = 'THIN'
                reason = f'signals={sigs} < {DEMOTE_THIN_SIG_MAX}'
            elif current_status == 'ACTIVE':
                if hr < DEMOTE_WATCH_HR_MAX:
                    new_status = 'WATCH'
                    reason = f'HR={hr:.1f} < {DEMOTE_WATCH_HR_MAX} (demote)'
                elif hr >= PROMOTE_HR_MIN:
                    new_status = 'ACTIVE'
                    reason = f'HR={hr:.1f} >= {PROMOTE_HR_MIN} (hold ACTIVE)'
                else:
                    new_status = 'ACTIVE'
                    reason = f'HR={hr:.1f} in hysteresis 60-65 (hold ACTIVE)'
                    hysteresis_holds += 1
            elif current_status in ('WATCH', 'THIN'):
                if (hr >= PROMOTE_HR_MIN and
                    sigs >= PROMOTE_SIGNALS_MIN and
                    corr is not None and corr >= PROMOTE_CORR_MIN):
                    new_status = 'ACTIVE'
                    reason = f'HR={hr:.1f}, sigs={sigs}, corr={corr:.2f} (promote)'
                elif hr >= PROMOTE_HR_MIN:
                    if current_status == 'THIN':
                        new_status = 'WATCH'
                        reason = f'HR={hr:.1f} >= {PROMOTE_HR_MIN} but corr/sigs fail (THIN->WATCH)'
                    else:
                        new_status = 'WATCH'
                        reason = f'HR={hr:.1f} >= {PROMOTE_HR_MIN} but corr/sigs fail (hold WATCH)'
                elif current_status == 'WATCH' and hr < DEMOTE_WATCH_HR_MAX:
                    new_status = 'THIN'
                    reason = f'HR={hr:.1f} < {DEMOTE_WATCH_HR_MAX} (WATCH->THIN)'
                else:
                    reason = f'HR={hr:.1f} (hold {current_status})'
                    if current_status == 'WATCH' and hr >= DEMOTE_WATCH_HR_MAX and hr < PROMOTE_HR_MIN:
                        hysteresis_holds += 1

            if new_status != current_status:
                self.transitions.append({
                    'Date': as_of_date,
                    'Cap': cap,
                    'Ticker': ticker,
                    'From': current_status,
                    'To': new_status,
                    'HR': round(hr, 1),
                    'Signals': sigs,
                    'Correlation': round(corr, 3) if corr is not None else None,
                    'Reason': reason
                })
                self.status[ticker] = new_status

                if new_status == 'ACTIVE':
                    promotions += 1
                elif new_status == 'WATCH':
                    demotions_watch += 1
                elif new_status == 'THIN':
                    demotions_thin += 1

        return {
            'promotions': promotions,
            'demotions_watch': demotions_watch,
            'demotions_thin': demotions_thin,
            'hysteresis_holds': hysteresis_holds,
            'active_count': len(self.get_active()),
            'watch_count': len(self.get_watch())
        }


# == SHADOW_ROTATION CORE =====================================================

def composite_score(row, entry_px):
    """Quality score for held position. Uses EMA5 (smoothed) to match live code."""
    ema   = float(row['EMA5'])  if pd.notna(row.get('EMA5'))  else 0.0
    eprev = float(row['EMA5_prev']) if pd.notna(row.get('EMA5_prev')) else ema
    pnl   = ((float(row['Price']) - entry_px) / entry_px * 100) \
            if pd.notna(row.get('Price')) and entry_px > 0 else 0.0
    trend = min(max(ema - eprev, -20.0), 20.0)
    return ema * 0.5 + trend + min(max(pnl, -15.0), 30.0)


def qualifies_for_entry(row):
    if not (pd.notna(row.get('DM')) and row['DM'] >= DM_ENTRY):
        return False
    if not (pd.notna(row.get('EMA5')) and pd.notna(row.get('EMA5_prev'))
            and row['EMA5'] > row['EMA5_prev']):
        return False
    if not (pd.notna(row.get('PriceMA20')) and pd.notna(row.get('Price'))
            and row['Price'] > row['PriceMA20']):
        return False
    if not (pd.notna(row.get('DM_MA20')) and row['DM_MA20'] >= DM_EXIT_MA):
        return False
    return True


# == MAIN SIMULATION ==========================================================

def is_first_monday_of_month(date):
    if date.weekday() != 0:
        return False
    for d in range(1, date.day):
        check = date.replace(day=d)
        if check.weekday() == 0:
            return False
    return True


def run_walkforward(df, etf_df, etf_mapping, candidate_pool, cap):
    sim_name = f'Cap_{cap:02d}'
    pos_size = STARTING_NAV / cap
    open_pos = {}
    cash = STARTING_NAV
    trades = []
    daily_log = []
    governance_summary = []

    universe = UniverseState(candidate_pool)
    df_by_ticker = {t: g for t, g in df.groupby('Ticker')}
    etf_by_ticker = {t: g for t, g in etf_df.groupby('Ticker')} if len(etf_df) > 0 else {}

    sim_start = pd.Timestamp(SIM_START_DATE)
    sim_end = pd.Timestamp(SIM_END_DATE)
    trading_dates = sorted(df[df['Date'] >= sim_start]['Date'].unique())

    print(f"\n  Running walkforward for Cap={cap}...")
    print(f"  Trading dates: {len(trading_dates)}")

    print(f"  Initial governance pass at {sim_start.date()}...")
    hr_results = compute_all_hit_rates_as_of(df, sim_start, candidate_pool)
    corr_results = {}
    for ticker in candidate_pool:
        etf = etf_mapping.get(ticker)
        if etf and ticker in df_by_ticker and etf in etf_by_ticker:
            corr_results[ticker] = compute_etf_correlation_as_of(
                df_by_ticker[ticker], etf_by_ticker[etf], sim_start)
    init_summary = universe.apply_governance(sim_start, hr_results, corr_results, cap)
    init_summary['date'] = sim_start.date()
    governance_summary.append(init_summary)
    print(f"  Initial: {init_summary['active_count']} ACTIVE, "
          f"{init_summary['watch_count']} WATCH")

    last_governance_month = (sim_start.year, sim_start.month)

    for i, dt in enumerate(trading_dates):
        if i % 250 == 0 and i > 0:
            pct = 100 * i / len(trading_dates)
            print(f"    {pct:.0f}% — {dt.date()} — "
                  f"ACTIVE={len(universe.get_active())} — "
                  f"positions={len(open_pos)}")

        current_month = (dt.year, dt.month)
        if (current_month != last_governance_month and
            is_first_monday_of_month(dt)):
            hr_results = compute_all_hit_rates_as_of(df, dt, candidate_pool)
            corr_results = {}
            for ticker in candidate_pool:
                etf = etf_mapping.get(ticker)
                if etf and ticker in df_by_ticker and etf in etf_by_ticker:
                    corr_results[ticker] = compute_etf_correlation_as_of(
                        df_by_ticker[ticker], etf_by_ticker[etf], dt)
            summary = universe.apply_governance(dt, hr_results, corr_results, cap)
            summary['date'] = dt.date()
            governance_summary.append(summary)
            last_governance_month = current_month

        active_tickers = set(universe.get_active())
        day = df[df['Date'] == dt].set_index('Ticker')

        to_exit = []
        for ticker, ep in open_pos.items():
            if ticker not in day.index:
                continue
            row = day.loc[ticker]
            exit_flag = False
            reason = None
            if pd.notna(row.get('DM_MA20')) and row['DM_MA20'] < DM_EXIT_MA:
                exit_flag, reason = True, 'DM_MA20_EXIT'
            elif pd.notna(row.get('Price')) and ep > 0 \
                 and (row['Price'] - ep) / ep < LOSS_LIMIT:
                exit_flag, reason = True, 'LOSS_LIMIT'
            if exit_flag:
                xp = float(row['Price'])
                pnl = (xp - ep) / ep
                cash += pos_size * (1 + pnl)
                trades.append({'Sim': sim_name, 'Cap': cap,
                               'Ticker': ticker, 'Type': 'EXIT',
                               'Date': dt.date(), 'Exit_Price': round(xp, 2),
                               'Entry_Price': round(ep, 2),
                               'PnL_Pct': round(pnl * 100, 2),
                               'Reason': reason})
                to_exit.append(ticker)
        for t in to_exit:
            del open_pos[t]

        if USE_ROTATION and len(open_pos) >= cap:
            best_t, best_dm = None, -1
            for ticker in active_tickers:
                if ticker in open_pos:
                    continue
                if ticker not in day.index:
                    continue
                row = day.loc[ticker]
                if qualifies_for_entry(row):
                    if row['DM'] > best_dm:
                        best_dm = float(row['DM'])
                        best_t = ticker

            if best_t is not None:
                scores = {}
                for ticker, ep in open_pos.items():
                    if ticker in day.index:
                        scores[ticker] = composite_score(day.loc[ticker], ep)

                if scores:
                    weak_t = min(scores, key=scores.get)
                    weak_score = scores[weak_t]

                    if best_dm - weak_score >= ROTATION_THRESHOLD:
                        xp = float(day.loc[weak_t, 'Price'])
                        ep = open_pos[weak_t]
                        pnl = (xp - ep) / ep
                        cash += pos_size * (1 + pnl)
                        trades.append({'Sim': sim_name, 'Cap': cap,
                                       'Ticker': weak_t, 'Type': 'ROTATION_OUT',
                                       'Date': dt.date(), 'Exit_Price': round(xp, 2),
                                       'Entry_Price': round(ep, 2),
                                       'PnL_Pct': round(pnl * 100, 2),
                                       'Reason': f'Replaced by {best_t} '
                                                 f'(gap {best_dm - weak_score:.1f})'})
                        del open_pos[weak_t]

                        if cash >= pos_size * 0.9:
                            ep_new = float(day.loc[best_t, 'Price'])
                            if ep_new > 0:
                                open_pos[best_t] = ep_new
                                cash -= pos_size
                                trades.append({'Sim': sim_name, 'Cap': cap,
                                               'Ticker': best_t, 'Type': 'ROTATION_IN',
                                               'Date': dt.date(), 'Exit_Price': None,
                                               'Entry_Price': round(ep_new, 2),
                                               'PnL_Pct': None,
                                               'Reason': f'Replaced {weak_t}'})

        if len(open_pos) < cap and cash >= pos_size * 0.9:
            candidates = []
            for ticker in active_tickers:
                if ticker in open_pos:
                    continue
                if ticker not in day.index:
                    continue
                row = day.loc[ticker]
                if qualifies_for_entry(row):
                    candidates.append((ticker, float(row['DM'])))
            candidates.sort(key=lambda x: x[1], reverse=True)

            for ticker, _ in candidates:
                if len(open_pos) >= cap or cash < pos_size * 0.9:
                    break
                if ticker not in day.index:
                    continue
                ep = float(day.loc[ticker, 'Price'])
                if ep > 0:
                    open_pos[ticker] = ep
                    cash -= pos_size
                    trades.append({'Sim': sim_name, 'Cap': cap,
                                   'Ticker': ticker, 'Type': 'ENTRY',
                                   'Date': dt.date(), 'Exit_Price': None,
                                   'Entry_Price': round(ep, 2),
                                   'PnL_Pct': None,
                                   'Reason': 'Signal'})

        nav = cash
        for ticker, ep in open_pos.items():
            if ticker in day.index and pd.notna(day.loc[ticker, 'Price']):
                nav += pos_size * (float(day.loc[ticker, 'Price']) / ep)
            else:
                nav += pos_size

        daily_log.append({
            'Sim': sim_name, 'Cap': cap, 'Date': dt.date(),
            'NAV': round(nav, 2), 'Positions': len(open_pos),
            'ACTIVE_count': len(active_tickers), 'Cash': round(cash, 2)
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    daily_df = pd.DataFrame(daily_log)
    transitions_df = pd.DataFrame(universe.transitions) if universe.transitions else pd.DataFrame()
    governance_df = pd.DataFrame(governance_summary)

    return trades_df, daily_df, transitions_df, governance_df


# == METRICS ==================================================================

def calc_metrics(daily_df, cap, trades_df, transitions_df):
    if len(daily_df) < 2:
        return None

    daily_df = daily_df.sort_values('Date').reset_index(drop=True)
    nav = daily_df['NAV'].values

    total_return = (nav[-1] / nav[0] - 1) * 100
    years = len(nav) / 252
    cagr = (np.power(nav[-1] / nav[0], 1 / years) - 1) * 100 if years > 0 else 0

    daily_rets = pd.Series(nav).pct_change().dropna()
    excess = daily_rets - RISK_FREE_DAILY
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    peak = pd.Series(nav).cummax()
    drawdown = (pd.Series(nav) - peak) / peak * 100
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    rotations = 0
    win_rate = 0
    if len(trades_df) > 0:
        rot_outs = trades_df[trades_df['Type'] == 'ROTATION_OUT']
        rotations = len(rot_outs)
        exits = trades_df[trades_df['Type'].isin(['EXIT', 'ROTATION_OUT'])]
        if len(exits) > 0 and 'PnL_Pct' in exits.columns:
            pnls = exits['PnL_Pct'].dropna()
            if len(pnls) > 0:
                win_rate = (pnls > 0).sum() / len(pnls) * 100

    promotions = 0
    demotions = 0
    avg_active = 0
    if len(transitions_df) > 0:
        promotions = len(transitions_df[transitions_df['To'] == 'ACTIVE'])
        demotions = len(transitions_df[transitions_df['From'] == 'ACTIVE'])

    if 'ACTIVE_count' in daily_df.columns:
        avg_active = daily_df['ACTIVE_count'].mean()

    return {
        'Sim': f'Cap_{cap:02d}',
        'Cap': cap,
        'Sharpe': round(sharpe, 3),
        'CAGR': round(cagr, 2),
        'Calmar': round(calmar, 2),
        'Max_Drawdown': round(max_dd, 2),
        'Total_Return': round(total_return, 2),
        'Win_Rate': round(win_rate, 1),
        'Rotations': rotations,
        'Final_NAV': round(nav[-1], 0),
        'Trading_Days': len(nav),
        'Promotions': promotions,
        'Demotions': demotions,
        'Avg_Active_Universe': round(avg_active, 1)
    }


# == CHARTS ===================================================================

def build_charts(metrics_df, daily_all):
    print('\n  Building charts...')

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0, 1, len(POSITION_CAPS)))
    for i, cap in enumerate(POSITION_CAPS):
        sim_name = f'Cap_{cap:02d}'
        subset = daily_all[daily_all['Sim'] == sim_name].sort_values('Date')
        if len(subset) > 0:
            ax1.plot(pd.to_datetime(subset['Date']), subset['NAV'],
                     label=f'Cap {cap}', color=colors[i], linewidth=1.5)
    ax1.set_title('Walkforward — NAV Trajectory ($1M start, Jan 2019 -> May 2026)',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_yscale('log')

    ax2 = fig.add_subplot(gs[1, 0])
    sorted_m = metrics_df.sort_values('Cap')
    bars = ax2.bar(range(len(sorted_m)), sorted_m['Calmar'], color='steelblue', alpha=0.8)
    if not sorted_m.empty:
        best_idx = sorted_m['Calmar'].idxmax()
        bars[sorted_m.index.get_loc(best_idx)].set_color('darkorange')
    ax2.set_xticks(range(len(sorted_m)))
    ax2.set_xticklabels([f'Cap {c}' for c in sorted_m['Cap']], fontsize=9)
    ax2.set_title('Calmar Ratio by Cap (PRIMARY KPI)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Calmar')
    ax2.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, sorted_m['Calmar']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(sorted_m))
    w = 0.35
    ax3.bar(x - w/2, sorted_m['CAGR'], w, label='CAGR %', color='green', alpha=0.7)
    ax3.bar(x + w/2, sorted_m['Max_Drawdown'].abs(), w, label='|Max DD| %',
            color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Cap {c}' for c in sorted_m['Cap']], fontsize=9)
    ax3.set_title('CAGR vs Max Drawdown', fontsize=12, fontweight='bold')
    ax3.set_ylabel('%')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2, axis='y')

    ax4 = fig.add_subplot(gs[2, 0])
    bars4 = ax4.bar(range(len(sorted_m)), sorted_m['Sharpe'], color='purple', alpha=0.8)
    ax4.set_xticks(range(len(sorted_m)))
    ax4.set_xticklabels([f'Cap {c}' for c in sorted_m['Cap']], fontsize=9)
    ax4.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sharpe')
    ax4.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars4, sorted_m['Sharpe']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax5 = fig.add_subplot(gs[2, 1])
    bars5 = ax5.bar(range(len(sorted_m)), sorted_m['Rotations'], color='teal', alpha=0.8)
    ax5.set_xticks(range(len(sorted_m)))
    ax5.set_xticklabels([f'Cap {c}' for c in sorted_m['Cap']], fontsize=9)
    ax5.set_title('Total Rotations (7.5yr)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('# Rotations')
    ax5.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars5, sorted_m['Rotations']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{int(val)}', ha='center', va='bottom', fontsize=9)

    ax6 = fig.add_subplot(gs[3, 0])
    x = np.arange(len(sorted_m))
    ax6.bar(x - w/2, sorted_m['Promotions'], w, label='Promotions',
            color='green', alpha=0.7)
    ax6.bar(x + w/2, sorted_m['Demotions'], w, label='Demotions',
            color='red', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Cap {c}' for c in sorted_m['Cap']], fontsize=9)
    ax6.set_title('Governance Events', fontsize=12, fontweight='bold')
    ax6.set_ylabel('# Events')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.2, axis='y')

    ax7 = fig.add_subplot(gs[3, 1])
    bars7 = ax7.bar(range(len(sorted_m)), sorted_m['Avg_Active_Universe'],
                     color='orange', alpha=0.8)
    ax7.set_xticks(range(len(sorted_m)))
    ax7.set_xticklabels([f'Cap {c}' for c in sorted_m['Cap']], fontsize=9)
    ax7.set_title('Average ACTIVE Universe Size', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Tickers')
    ax7.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars7, sorted_m['Avg_Active_Universe']):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.savefig(f'{OUTPUT_DIR}/walkforward_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart: {OUTPUT_DIR}/walkforward_chart.png")


# == MAIN =====================================================================

def main():
    print('\n' + '='*75)
    print('  WALKFORWARD WITH DYNAMIC UNIVERSE GOVERNANCE')
    print(f'  Data: {DATA_LOAD_FROM} -> {SIM_END_DATE}')
    print(f'  Simulation: {SIM_START_DATE} -> {SIM_END_DATE} (7.5 years)')
    print(f'  Caps: {POSITION_CAPS}')
    print(f'  Composite uses EMA5 (matches live Shadow_Rotation)')
    print('='*75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\nLoading candidate pool...')
    candidate_pool = load_candidate_pool()

    print('\nLoading DM history from Supabase...')
    df = load_dm_history(candidate_pool)

    print('\nLoading ETF mapping...')
    etf_mapping = load_etf_mapping()
    etf_tickers = sorted(set(etf_mapping.values()))

    print('\nLoading ETF prices...')
    if etf_tickers:
        etf_df = load_etf_data(etf_tickers)
    else:
        print("  WARNING: No ETF mapping found. Correlation gate will be skipped.")
        etf_df = pd.DataFrame()

    print('\nComputing indicators...')
    df = add_indicators(df)

    all_metrics = []
    all_daily = []
    all_trades = []
    all_transitions = []
    all_governance = []

    for cap in POSITION_CAPS:
        trades_df, daily_df, transitions_df, governance_df = run_walkforward(
            df, etf_df, etf_mapping, candidate_pool, cap)

        if len(daily_df) == 0:
            print(f"  WARNING: No daily data for cap={cap}")
            continue

        m = calc_metrics(daily_df, cap, trades_df, transitions_df)
        if m is None:
            continue

        all_metrics.append(m)
        all_daily.append(daily_df)
        all_trades.append(trades_df)
        all_transitions.append(transitions_df)
        all_governance.append(governance_df)

        daily_df.to_csv(f'{OUTPUT_DIR}/daily_nav_cap_{cap:02d}.csv', index=False)
        if len(trades_df) > 0:
            trades_df.to_csv(f'{OUTPUT_DIR}/trades_cap_{cap:02d}.csv', index=False)
        if len(transitions_df) > 0:
            transitions_df.to_csv(f'{OUTPUT_DIR}/governance_events_cap_{cap:02d}.csv',
                                   index=False)
        if len(governance_df) > 0:
            governance_df.to_csv(f'{OUTPUT_DIR}/universe_state_cap_{cap:02d}.csv',
                                  index=False)

        print(f"  Cap {cap}: Calmar={m['Calmar']} | CAGR={m['CAGR']}% | "
              f"Sharpe={m['Sharpe']} | MDD={m['Max_Drawdown']}% | "
              f"Rotations={m['Rotations']} | Promos={m['Promotions']} | "
              f"Demos={m['Demotions']}")

    metrics_df = pd.DataFrame(all_metrics).sort_values('Cap')
    daily_all = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()

    print('\n' + '='*75)
    print('  SUMMARY')
    print('='*75)
    print(metrics_df[['Cap', 'Calmar', 'Sharpe', 'CAGR', 'Max_Drawdown',
                       'Total_Return', 'Rotations', 'Promotions', 'Demotions',
                       'Avg_Active_Universe']].to_string(index=False))

    if len(metrics_df) > 0:
        best = metrics_df.loc[metrics_df['Calmar'].idxmax()]
        print('\n' + '='*75)
        print(f'  BEST CAP BY CALMAR: Cap_{int(best["Cap"]):02d}')
        print('='*75)
        print(f'    Calmar:       {best["Calmar"]}')
        print(f'    CAGR:         {best["CAGR"]}%')
        print(f'    Max DD:       {best["Max_Drawdown"]}%')
        print(f'    Sharpe:       {best["Sharpe"]}')
        print(f'    Rotations:    {int(best["Rotations"])}')
        print(f'    Promotions:   {int(best["Promotions"])}')
        print(f'    Demotions:    {int(best["Demotions"])}')
        print(f'    Avg Active:   {best["Avg_Active_Universe"]} tickers')

    metrics_df.to_csv(f'{OUTPUT_DIR}/summary_all_caps.csv', index=False)
    print(f'\n  Summary: {OUTPUT_DIR}/summary_all_caps.csv')

    if len(daily_all) > 0 and len(metrics_df) > 0:
        build_charts(metrics_df, daily_all)

    print(f'\n  All results saved to: {OUTPUT_DIR}/')
    print('='*75 + '\n')


if __name__ == '__main__':
    main()
