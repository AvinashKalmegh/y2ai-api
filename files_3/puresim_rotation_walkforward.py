"""
puresim_rotation_walkforward.py
=================================
ARGUS Research -- PureSim Strategy 5
Full 10-Year Rotation Walkforward: Baseline vs Quality Rotation

PURPOSE:
    Tests whether quality rotation -- exiting the weakest current position
    to make room for a stronger qualified candidate -- adds value over the
    validated baseline rules across the full 10-year history.

    Runs three simulations from 2016-02-01 to 2026-03-31:

    SIM A -- BASELINE (current validated rules):
        Cap:         20 | Min history: 3yr | Rotation: OFF
        This is the anchor. Sharpe ~0.909 from prior validation.
        If Sim B does not beat this, rotation adds no value.

    SIM B -- ROTATION (same cap, rotation ON):
        Cap:         20 | Min history: 3yr | Rotation: ON
        Tests whether rotation adds value at constant cap.
        Direct answer to: does replacing weak positions improve returns?

    SIM C -- ROTATION + STRICT HISTORY:
        Cap:         20 | Min history: 5yr | Rotation: ON
        Tests rotation with quantum/thin-history names excluded.
        Answers: does stricter entry filter compound with rotation benefit?

    Each simulation runs across ROTATION_THRESHOLDS = [10, 15, 20, 25, 30]
    to find the optimal threshold before live implementation.

ROTATION RULE:
    When the book is full (positions == cap) AND qualified entry candidates
    exist, score all current positions by composite:
        composite = DM * 0.5 + EMA_trend (capped ±20) + PnL% (capped -15/+30)
    If the weakest current position composite < best candidate DM - threshold:
        exit weakest -> enter best candidate
    This uses the same validated DM signal on both sides. No new signal.

OUTPUT:
    results_rotation/
        summary_all_sims.csv           -- all sims x all thresholds, key metrics
        daily_nav_[sim]_t[threshold].csv -- daily NAV per sim per threshold
        rotation_events_[sim].csv      -- all rotation events fired
        rotation_walkforward_chart.png -- Sharpe comparison heatmap + NAV curves

BEFORE RUNNING:
    Place puresim_universe.csv in same folder as this script.
    Export: PureSim_Universe tab in NEW_FlowOS_preferred -> File -> Download -> CSV
    Status column values accepted: QUALIFY or ACTIVE

REQUIREMENTS:
    pip install supabase pandas numpy matplotlib seaborn python-dotenv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from itertools import product

warnings.filterwarnings('ignore')

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# -- CONFIGURATION --------------------------------------------------------------

SUPABASE_URL    = os.environ.get('SUPABASE_URL')
SUPABASE_KEY    = os.environ.get('SUPABASE_KEY')

# Full 10-year backtest window
START_DATE      = '2016-02-01'
END_DATE        = '2026-03-31'

STARTING_NAV    = 1_000_000.0
RISK_FREE_DAILY = 0.043 / 252

# Strategy parameters -- LOCKED, identical for all sims
DM_ENTRY        = 65
DM_EXIT_MA      = 50
LOSS_LIMIT      = -0.15

# Rotation thresholds to test
# Composite score gap required between best candidate and weakest position
ROTATION_THRESHOLDS = [10, 15, 20, 25, 30]

# Simulation definitions
SIMS = [
    {
        'name':         'Sim_A_Baseline',
        'label':        'Sim A -- Baseline (Cap 20, 3yr, no rotation)',
        'cap':          20,
        'min_history':  3,
        'rotation':     False,
        'color':        'steelblue',
    },
    {
        'name':         'Sim_B_Rotation',
        'label':        'Sim B -- Rotation ON (Cap 20, 3yr)',
        'cap':          20,
        'min_history':  3,
        'rotation':     True,
        'color':        'darkorange',
    },
    {
        'name':         'Sim_C_Rotation_5yr',
        'label':        'Sim C -- Rotation ON (Cap 20, 5yr)',
        'cap':          20,
        'min_history':  5,
        'rotation':     True,
        'color':        'green',
    },
]

OUTPUT_DIR = './results_rotation'


# -- UNIVERSE LOADER ------------------------------------------------------------

def load_universe():
    """Load governed PureSim universe from puresim_universe.csv."""
    path = 'puresim_universe.csv'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("  Export PureSim_Universe tab from NEW_FlowOS_preferred:")
        print("  File -> Download -> Comma Separated Values")
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    ticker_col = next((c for c in df.columns
                       if c.lower() in ('ticker','symbol','stock')), None)
    if not ticker_col:
        print(f"ERROR: No Ticker column in {path}. Columns: {list(df.columns)}")
        sys.exit(1)

    status_col = next((c for c in df.columns
                       if c.lower() in ('status','active','include')), None)
    if status_col:
        eligible = df[status_col].astype(str).str.upper().str.strip()
        df = df[eligible.isin(['QUALIFY', 'ACTIVE', 'TRUE', '1'])]

    tickers = list(dict.fromkeys(
        df[ticker_col].dropna().str.strip().str.upper().tolist()))
    tickers = [t for t in tickers if t]
    print(f"  Universe: {len(tickers)} tickers from {path}")
    return tickers


# -- DATA LOADING ---------------------------------------------------------------

def load_dm_history(universe):
    """Load full DM history from Supabase for the universe tickers."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    # Load extra history for indicators and MIN_HISTORY checks
    load_from_year = 2011
    end_year = int(END_DATE[:4])

    print(f"\n  Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"  Loading dm_history ({load_from_year} to {END_DATE}) year by year...")

    all_rows = []
    for year in range(load_from_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31' if year < end_year else END_DATE
        offset = 0
        batch_size = 10000
        while True:
            response = client.table('dm_history')\
                .select('date,ticker,close,dm_smoothed')\
                .gte('date', year_start)\
                .lte('date', year_end)\
                .order('date')\
                .range(offset, offset + batch_size - 1)\
                .execute()
            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch_size
            if len(rows) < batch_size:
                break
        print(f"    {year}: {len(all_rows):,} total rows")

    print(f"  Total: {len(all_rows):,} rows")
    df = pd.DataFrame(all_rows)
    df['date']  = pd.to_datetime(df['date'])
    df['dm_smoothed'] = pd.to_numeric(df['dm_smoothed'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['date', 'dm_smoothed', 'close'])
    df = df.rename(columns={'date': 'Date', 'ticker': 'Ticker',
                              'close': 'Price', 'dm_smoothed': 'DM'})
    df = df[df['Ticker'].isin(universe)].copy()
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f"  After filter: {df['Ticker'].nunique()} tickers | "
          f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def add_indicators(df):
    """Add DM_MA20, EMA5, EMA5_prev, PriceMA20."""
    g = df.groupby('Ticker')
    df['DM_MA20']   = g['DM'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['EMA5']      = g['DM'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['EMA5_prev'] = g['EMA5'].shift(1)
    df['PriceMA20'] = g['Price'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    return df


def get_history_years(df):
    """Return {ticker: years_of_history} using full loaded dataset."""
    result = {}
    for ticker, grp in df.groupby('Ticker'):
        days = (grp['Date'].max() - grp['Date'].min()).days
        result[ticker] = days / 365.25
    return result


# -- SIMULATION CORE ------------------------------------------------------------

def composite_score(row, entry_px):
    """Quality score for a currently held position."""
    dm    = float(row['DM'])    if pd.notna(row.get('DM'))    else 0.0
    ema   = float(row['EMA5'])  if pd.notna(row.get('EMA5'))  else 0.0
    eprev = float(row['EMA5_prev']) if pd.notna(row.get('EMA5_prev')) else ema
    pnl   = ((float(row['Price']) - entry_px) / entry_px * 100) \
            if pd.notna(row.get('Price')) and entry_px > 0 else 0.0
    trend = min(max(ema - eprev, -20.0), 20.0)
    return dm * 0.5 + trend + min(max(pnl, -15.0), 30.0)


def qualifies_for_entry(row, hist_yrs, ticker):
    """Check all entry conditions for a candidate ticker."""
    if hist_yrs.get(ticker, 0) < 1.0:            # must have some history
        return False
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


def run_sim(df, sim_name, cap, min_history_yrs, use_rotation,
             rotation_threshold, hist_yrs, trading_dates):
    """
    Run one simulation for one rotation threshold.
    Returns (trades_df, daily_df).
    """
    pos_size    = STARTING_NAV / cap
    open_pos    = {}    # {ticker: entry_price}
    cash        = STARTING_NAV
    trades      = []
    daily_log   = []
    rotation_ct = 0

    backtest_start = pd.Timestamp(START_DATE)

    for dt in trading_dates:
        day = df[df['Date'] == dt].set_index('Ticker')

        # -- EXITS ----------------------------------------------------------
        to_exit = []
        for ticker, ep in open_pos.items():
            if ticker not in day.index:
                continue
            row = day.loc[ticker]
            exit_flag = False
            reason    = None
            if pd.notna(row.get('DM_MA20')) and row['DM_MA20'] < DM_EXIT_MA:
                exit_flag, reason = True, 'DM_MA20_EXIT'
            elif pd.notna(row.get('Price')) and ep > 0 \
                 and (row['Price'] - ep) / ep < LOSS_LIMIT:
                exit_flag, reason = True, 'LOSS_LIMIT'
            if exit_flag:
                xp  = float(row['Price'])
                pnl = (xp - ep) / ep
                cash += pos_size * (1 + pnl)
                trades.append({'Sim': sim_name, 'Threshold': rotation_threshold,
                                'Ticker': ticker, 'Type': 'EXIT',
                                'Date': dt.date(), 'Exit_Price': round(xp, 2),
                                'Entry_Price': round(ep, 2),
                                'PnL_Pct': round(pnl * 100, 2),
                                'Reason': reason})
                to_exit.append(ticker)
        for t in to_exit:
            del open_pos[t]

        # -- QUALITY ROTATION -----------------------------------------------
        if use_rotation and len(open_pos) >= cap:
            # Find best qualified candidate not already held
            best_t, best_dm = None, -1
            for ticker, row in day.iterrows():
                if ticker in open_pos:
                    continue
                if hist_yrs.get(ticker, 0) < min_history_yrs:
                    continue
                if qualifies_for_entry(row, hist_yrs, ticker):
                    if row['DM'] > best_dm:
                        best_dm = float(row['DM'])
                        best_t  = ticker

            if best_t is not None:
                # Score all current positions
                scores = {}
                for ticker, ep in open_pos.items():
                    if ticker in day.index:
                        scores[ticker] = composite_score(day.loc[ticker], ep)

                if scores:
                    weak_t     = min(scores, key=scores.get)
                    weak_score = scores[weak_t]

                    if best_dm - weak_score >= rotation_threshold:
                        # Exit weakest
                        xp  = float(day.loc[weak_t, 'Price'])
                        ep  = open_pos[weak_t]
                        pnl = (xp - ep) / ep
                        cash += pos_size * (1 + pnl)
                        trades.append({
                            'Sim': sim_name, 'Threshold': rotation_threshold,
                            'Ticker': weak_t, 'Type': 'ROTATION_OUT',
                            'Date': dt.date(), 'Exit_Price': round(xp, 2),
                            'Entry_Price': round(ep, 2),
                            'PnL_Pct': round(pnl * 100, 2),
                            'Reason': f'Replaced by {best_t} '
                                       f'(gap {best_dm - weak_score:.1f})'
                        })
                        del open_pos[weak_t]

                        # Enter best candidate
                        if best_t in day.index and cash >= pos_size * 0.9:
                            ep_new = float(day.loc[best_t, 'Price'])
                            if ep_new > 0:
                                open_pos[best_t] = ep_new
                                cash -= pos_size
                                rotation_ct += 1
                                trades.append({
                                    'Sim': sim_name, 'Threshold': rotation_threshold,
                                    'Ticker': best_t, 'Type': 'ROTATION_IN',
                                    'Date': dt.date(), 'Exit_Price': None,
                                    'Entry_Price': round(ep_new, 2),
                                    'PnL_Pct': None,
                                    'Reason': f'Replaced {weak_t}'
                                })

        # -- STANDARD ENTRIES -----------------------------------------------
        if len(open_pos) < cap and cash >= pos_size * 0.9:
            candidates = []
            for ticker, row in day.iterrows():
                if ticker in open_pos:
                    continue
                if hist_yrs.get(ticker, 0) < min_history_yrs:
                    continue
                if qualifies_for_entry(row, hist_yrs, ticker):
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
                    trades.append({
                        'Sim': sim_name, 'Threshold': rotation_threshold,
                        'Ticker': ticker, 'Type': 'ENTRY',
                        'Date': dt.date(), 'Exit_Price': None,
                        'Entry_Price': round(ep, 2),
                        'PnL_Pct': None,
                        'Reason': 'Signal'
                    })

        # -- DAILY NAV (only for backtest window) ---------------------------
        if dt >= backtest_start:
            nav = cash
            for ticker, ep in open_pos.items():
                if ticker in day.index:
                    cp = float(day.loc[ticker, 'Price'])
                    nav += pos_size * (cp / ep) if ep > 0 else pos_size
                else:
                    nav += pos_size

            daily_log.append({
                'Date':         dt.date(),
                'Sim':          sim_name,
                'Threshold':    rotation_threshold,
                'NAV':          round(nav, 2),
                'NAV_Pct':      round((nav - STARTING_NAV) / STARTING_NAV * 100, 2),
                'Positions':    len(open_pos),
                'Cash':         round(cash, 2),
                'Rotation_Ct':  rotation_ct,
            })

    return pd.DataFrame(trades), pd.DataFrame(daily_log)


# -- METRICS --------------------------------------------------------------------

def calc_metrics(daily_df, sim_name, threshold, trades_df):
    r     = daily_df['NAV'].pct_change().fillna(0)
    ex    = r - RISK_FREE_DAILY
    sh    = (ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0
    tr    = (daily_df['NAV'].iloc[-1] - STARTING_NAV) / STARTING_NAV * 100
    cum   = (1 + r).cumprod()
    dd    = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    years = len(daily_df) / 252

    # Win rate from closed trades only
    closed = trades_df[(trades_df['Type'].isin(['EXIT','ROTATION_OUT']))
                        & trades_df['PnL_Pct'].notna()]
    win_rate = (closed['PnL_Pct'] > 0).mean() * 100 if len(closed) > 0 else 0
    rot_ct   = daily_df['Rotation_Ct'].max() if 'Rotation_Ct' in daily_df else 0

    return {
        'Sim':          sim_name,
        'Threshold':    threshold,
        'Final_NAV':    round(daily_df['NAV'].iloc[-1], 2),
        'Total_Return': round(tr, 2),
        'CAGR':         round(((daily_df['NAV'].iloc[-1] / STARTING_NAV)
                               ** (1 / years) - 1) * 100, 2) if years > 0 else 0,
        'Sharpe':       round(sh, 2),
        'Max_Drawdown': round(dd, 2),
        'Win_Rate':     round(win_rate, 2),
        'Total_Trades': len(closed),
        'Rotations':    int(rot_ct),
        'Avg_Positions': round(daily_df['Positions'].mean(), 1),
    }


# -- CHART ----------------------------------------------------------------------

def build_charts(all_metrics, all_daily):
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f'PureSim -- Quality Rotation Walkforward\n'
        f'{START_DATE} to {END_DATE}  |  Cap 20  |  $1M Starting NAV',
        fontsize=13)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    metrics_df = pd.DataFrame(all_metrics)

    # -- NAV curves for best threshold -------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    colors = {s['name']: s['color'] for s in SIMS}
    for sim_def in SIMS:
        name = sim_def['name']
        if sim_def['rotation']:
            # Use best threshold for rotation sims
            sim_m = metrics_df[metrics_df['Sim'] == name]
            if len(sim_m) > 0:
                best_t = sim_m.loc[sim_m['Sharpe'].idxmax(), 'Threshold']
            else:
                best_t = ROTATION_THRESHOLDS[0]
            sim_daily = all_daily[
                (all_daily['Sim'] == name) & (all_daily['Threshold'] == best_t)]
            label = f'{sim_def["label"]} (threshold={best_t})'
        else:
            sim_daily = all_daily[
                (all_daily['Sim'] == name) & (all_daily['Threshold'] == 0)]
            label = sim_def['label']

        if len(sim_daily) > 0:
            ax1.plot(pd.to_datetime(sim_daily['Date']), sim_daily['NAV'],
                     color=colors[name], lw=1.5, label=label)

    ax1.axhline(STARTING_NAV, color='gray', lw=0.5, ls=':')
    ax1.set_title('NAV Comparison -- Best Threshold Per Sim', fontsize=11)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # -- Sharpe heatmap by threshold ---------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    rot_sims   = [s['name'] for s in SIMS if s['rotation']]
    pivot_data = metrics_df[metrics_df['Sim'].isin(rot_sims)].pivot_table(
        index='Sim', columns='Threshold', values='Sharpe')
    if HAS_SEABORN and len(pivot_data) > 0:
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn',
                    ax=ax2, cbar=True)
    else:
        ax2.imshow(pivot_data.values if len(pivot_data) > 0
                   else np.zeros((2,5)), aspect='auto', cmap='RdYlGn')
        ax2.set_xticks(range(len(ROTATION_THRESHOLDS)))
        ax2.set_xticklabels(ROTATION_THRESHOLDS)
    ax2.set_title('Sharpe by Threshold', fontsize=11)

    # -- CAGR by threshold -------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    pivot_cagr = metrics_df[metrics_df['Sim'].isin(rot_sims)].pivot_table(
        index='Sim', columns='Threshold', values='CAGR')
    if HAS_SEABORN and len(pivot_cagr) > 0:
        sns.heatmap(pivot_cagr, annot=True, fmt='.1f', cmap='RdYlGn',
                    ax=ax3, cbar=True)
    ax3.set_title('CAGR % by Threshold', fontsize=11)

    # -- Rotation count by threshold ---------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    for sim_def in SIMS:
        if not sim_def['rotation']:
            continue
        name   = sim_def['name']
        sim_m  = metrics_df[metrics_df['Sim'] == name].sort_values('Threshold')
        ax4.plot(sim_m['Threshold'], sim_m['Rotations'],
                 marker='o', color=colors[name], label=name.replace('Sim_', ''))
    ax4.set_title('Rotation Events by Threshold', fontsize=11)
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('# Rotations (10yr)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # -- Key metrics bar chart -- best threshold ----------------------------
    metric_names = ['Sharpe', 'CAGR', 'Max_Drawdown', 'Win_Rate']
    titles       = ['Sharpe', 'CAGR %', 'Max DD %', 'Win Rate %']
    for idx, (m, title) in enumerate(zip(metric_names, titles)):
        ax = fig.add_subplot(gs[2, idx if idx < 3 else 2])
        labels, vals, bar_colors = [], [], []
        for sim_def in SIMS:
            name = sim_def['name']
            if sim_def['rotation']:
                sim_m = metrics_df[metrics_df['Sim'] == name]
                if len(sim_m) > 0:
                    best_t = sim_m.loc[sim_m['Sharpe'].idxmax(), 'Threshold']
                    row    = sim_m[sim_m['Threshold'] == best_t].iloc[0]
                else:
                    continue
            else:
                row = metrics_df[
                    (metrics_df['Sim'] == name) &
                    (metrics_df['Threshold'] == 0)].iloc[0]
            labels.append(name.replace('Sim_', '').replace('_', '\n'))
            vals.append(float(row[m]))
            bar_colors.append(colors[name])
        bars = ax.bar(range(len(labels)), vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(True, alpha=0.2, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(f'{OUTPUT_DIR}/rotation_walkforward_chart.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Chart: {OUTPUT_DIR}/rotation_walkforward_chart.png")


# -- MAIN -----------------------------------------------------------------------

def main():
    print('\n' + '='*65)
    print('  PURESIM -- QUALITY ROTATION WALKFORWARD (10-YEAR)')
    print(f'  {START_DATE} to {END_DATE}')
    print(f'  Sims: {len(SIMS)} | Thresholds: {ROTATION_THRESHOLDS}')
    print(f'  Total runs: {len(SIMS) + (len(SIMS)-1) * (len(ROTATION_THRESHOLDS)-1)}')
    print('='*65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print('\nLoading universe...')
    universe = load_universe()

    print('\nLoading DM history from Supabase...')
    df = load_dm_history(universe)
    df = add_indicators(df)

    hist_yrs      = get_history_years(df)
    trading_dates = sorted(df['Date'].unique())
    print(f"  Trading dates: {len(trading_dates)}")

    # Results storage
    all_metrics = []
    all_daily   = []
    all_trades  = []

    # Run simulations
    for sim_def in SIMS:
        name     = sim_def['name']
        cap      = sim_def['cap']
        min_hist = sim_def['min_history']
        rotation = sim_def['rotation']

        # Baseline runs once with threshold=0 (rotation=False ignores threshold)
        thresholds = ROTATION_THRESHOLDS if rotation else [0]

        for threshold in thresholds:
            print(f"\n  {name} | threshold={threshold}...")
            trades_df, daily_df = run_sim(
                df, name, cap, min_hist, rotation, threshold,
                hist_yrs, trading_dates)

            if len(daily_df) == 0:
                print(f"    WARNING: No daily data produced.")
                continue

            m = calc_metrics(daily_df, name, threshold, trades_df)
            all_metrics.append(m)
            all_daily.append(daily_df)
            all_trades.append(trades_df)

            rot_str = f" | Rotations: {m['Rotations']}" if rotation else ""
            print(f"    Sharpe: {m['Sharpe']} | CAGR: {m['CAGR']}% | "
                  f"MaxDD: {m['Max_Drawdown']}%{rot_str}")

    # Combine results
    metrics_df = pd.DataFrame(all_metrics)
    daily_all  = pd.concat(all_daily,  ignore_index=True) if all_daily  else pd.DataFrame()
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Print summary
    print('\n' + '='*65)
    print('  SUMMARY -- KEY METRICS')
    print('='*65)
    print(metrics_df[['Sim', 'Threshold', 'Sharpe', 'CAGR',
                        'Max_Drawdown', 'Win_Rate', 'Rotations',
                        'Total_Return']].to_string(index=False))

    # Baseline comparison
    baseline = metrics_df[metrics_df['Sim'] == 'Sim_A_Baseline']
    if len(baseline) > 0:
        base_sharpe = baseline.iloc[0]['Sharpe']
        base_cagr   = baseline.iloc[0]['CAGR']
        print(f'\n  Baseline Sharpe: {base_sharpe} | CAGR: {base_cagr}%')

        for sim_def in SIMS[1:]:  # skip baseline
            name   = sim_def['name']
            sim_m  = metrics_df[metrics_df['Sim'] == name]
            if len(sim_m) == 0:
                continue
            best   = sim_m.loc[sim_m['Sharpe'].idxmax()]
            delta  = best['Sharpe'] - base_sharpe
            verdict = '[OK] ROTATION ADDS VALUE' if delta > 0.05 \
                      else ('~ NEUTRAL' if delta > -0.05 else '[X] ROTATION HURTS')
            print(f'\n  {name}:')
            print(f'    Best Sharpe: {best["Sharpe"]} at threshold={best["Threshold"]}')
            print(f'    vs Baseline: {delta:+.3f} -- {verdict}')
            print(f'    Rotations at best threshold: {best["Rotations"]}')

    # Save outputs
    metrics_df.to_csv(f'{OUTPUT_DIR}/summary_all_sims.csv', index=False)

    if len(daily_all) > 0:
        for sim_def in SIMS:
            name = sim_def['name']
            thresholds = ROTATION_THRESHOLDS if sim_def['rotation'] else [0]
            for t in thresholds:
                subset = daily_all[(daily_all['Sim'] == name) &
                                    (daily_all['Threshold'] == t)]
                if len(subset) > 0:
                    subset.to_csv(
                        f'{OUTPUT_DIR}/daily_nav_{name}_t{t}.csv', index=False)

    if len(trades_all) > 0:
        rots = trades_all[trades_all['Type'].str.contains('ROTATION', na=False)]
        if len(rots) > 0:
            rots.to_csv(f'{OUTPUT_DIR}/rotation_events.csv', index=False)
            print(f'\n  Total rotation events across all runs: {len(rots)}')

    # Build charts
    if len(daily_all) > 0:
        build_charts(all_metrics, daily_all)

    print(f'\n  All results saved to: {OUTPUT_DIR}/')
    print('='*65)
    print('\n  INTERPRETATION GUIDE:')
    print('  Sim B Sharpe > Sim A Sharpe + 0.05 -> rotation adds value at cap 20')
    print('  Sim C Sharpe > Sim B Sharpe         -> 5yr history filter amplifies benefit')
    print('  Optimal threshold = highest Sharpe with fewest rotations')
    print('  If Sim A wins -> current rules are the answer. Stop here.')
    print('='*65)


if __name__ == '__main__':
    main()
