"""
puresim_cap_sweep_walkforward.py
=================================
ARGUS Research — Shadow_Rotation Cap Sweep
Full 10-Year Walkforward: Position Cap Optimization

PURPOSE:
    Tests what position cap (N) maximizes risk-adjusted return for the
    Shadow_Rotation methodology across the full 10-year Supabase history.

    Runs 8 simulations from 2016-02-01 to 2026-03-31:
        Cap = 1, 3, 5, 10, 15, 20, 30, 50

    Same entry/exit/rotation rules across all caps. Only N changes.

    PRIMARY KPI: Calmar Ratio (CAGR / |Max Drawdown|)
    SECONDARY:   Sharpe, Total Return, Rotation Count, Win Rate

METHODOLOGY (identical to live Shadow_Rotation, validated against Apps Script):
    Entry:    DM >= 65 AND EMA5 rising AND Price > 20d Price MA AND DM_MA20 >= 50
    Exit:     DM_MA20 < 50 OR loss < -15%
    Rotation: book full + (best_candidate_DM - weakest_composite) >= 20

    Composite score (held positions, asymmetric):
        composite = DM * 0.5 + EMA_trend (capped ±20) + PnL% (capped -15/+30)

    Candidate score (entries):
        score = DM   (current value only, must also pass entry conditions)

DATA:
    Reads dm_daily from Supabase. Reuses universe from puresim_universe.csv.
    Same patterns as puresim_rotation_walkforward.py.

OUTPUT:
    results_cap_sweep/
        summary_all_caps.csv           — one row per cap with all metrics
        daily_nav_cap_NN.csv           — daily NAV for each cap
        rotation_events_cap_NN.csv     — rotation events per cap
        cap_sweep_chart.png            — comparison chart

USAGE:
    1. Ensure puresim_universe.csv is in same folder
    2. Ensure .env has SUPABASE_URL and SUPABASE_KEY
    3. python puresim_cap_sweep_walkforward.py

REQUIREMENTS:
    pip install supabase pandas numpy matplotlib seaborn python-dotenv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

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

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

SUPABASE_URL    = os.environ.get('SUPABASE_URL')
SUPABASE_KEY    = os.environ.get('SUPABASE_KEY')

# Full 10-year backtest window
START_DATE      = '2016-02-01'
END_DATE        = '2026-03-31'

STARTING_NAV    = 1_000_000.0
RISK_FREE_DAILY = 0.043 / 252

# Strategy parameters — LOCKED, identical for all caps
DM_ENTRY        = 65
DM_EXIT_MA      = 50
LOSS_LIMIT      = -0.15
MIN_HISTORY_YRS = 3.0

# Rotation parameters — LOCKED at live Shadow_Rotation values
ROTATION_THRESHOLD = 20
USE_ROTATION       = True

# Position caps to sweep
POSITION_CAPS = [1, 3, 5, 10, 15, 20, 30, 50]

OUTPUT_DIR = './results_cap_sweep_ema5'


# ── UNIVERSE LOADER ────────────────────────────────────────────────────────────

def load_universe():
    """Load governed PureSim universe from puresim_universe.csv."""
    path = 'puresim_universe.csv'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("  Export PureSim_Universe tab from NEW_FlowOS_preferred:")
        print("  File → Download → Comma Separated Values")
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    ticker_col = next((c for c in df.columns
                       if c.lower() in ('ticker', 'symbol', 'stock')), None)
    if not ticker_col:
        print(f"ERROR: No Ticker column in {path}. Columns: {list(df.columns)}")
        sys.exit(1)

    status_col = next((c for c in df.columns
                       if c.lower() in ('status', 'active', 'include')), None)
    if status_col:
        eligible = df[status_col].astype(str).str.upper().str.strip()
        df = df[eligible.isin(['QUALIFY', 'ACTIVE', 'TRUE', '1'])]

    tickers = list(dict.fromkeys(
        df[ticker_col].dropna().str.strip().str.upper().tolist()))
    tickers = [t for t in tickers if t]
    print(f"  Universe: {len(tickers)} tickers from {path}")
    return tickers


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_dm_history(universe):
    """Load full DM history from Supabase for the universe tickers."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    # Load extra history for indicators and MIN_HISTORY checks
    load_from = '2011-01-01'

    print(f"\n  Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"  Loading dm_daily ({load_from} to {END_DATE})...")
    print(f"  This will take several minutes for full 10-year history...")

    all_rows = []
    batch_size = 10000

    start_year = int(load_from[:4])
    end_year   = int(END_DATE[:4])

    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end   = f"{year}-12-31"
        offset  = 0

        while True:
            attempt, cur = 0, batch_size
            while True:
                try:
                    response = client.table('dm_history')\
                        .select('date,ticker,close,dm_smoothed')\
                        .gte('date', y_start)\
                        .lte('date', y_end)\
                        .order('date').order('ticker')\
                        .range(offset, offset + cur - 1)\
                        .execute()
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
                    print(f"\n    [retry {attempt}] year={year} page={cur} sleep {2**attempt}s ({etype})")
                    time.sleep(2 ** attempt)

            rows = response.data
            if not rows:
                break
            all_rows.extend(rows)
            offset += cur
            print(f"    {len(all_rows):,} rows...", end='\r')
            if len(rows) < cur:
                break

        print(f"\n    {year}: {len(all_rows):,} total rows")

    print(f"\n  Total: {len(all_rows):,} rows")
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={'dm_smoothed': 'dm'})
    df['date']  = pd.to_datetime(df['date'])
    df['dm']    = pd.to_numeric(df['dm'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['date', 'dm', 'close'])
    df = df.rename(columns={'date': 'Date', 'ticker': 'Ticker',
                             'close': 'Price', 'dm': 'DM'})
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


# ── SIMULATION CORE ────────────────────────────────────────────────────────────

def composite_score(row, entry_px):
    """Quality score for a currently held position (asymmetric).
    Uses smoothed EMA5 of DM (matches live Shadow_Rotation production code)."""
    ema   = float(row['EMA5'])  if pd.notna(row.get('EMA5'))  else 0.0
    eprev = float(row['EMA5_prev']) if pd.notna(row.get('EMA5_prev')) else ema
    pnl   = ((float(row['Price']) - entry_px) / entry_px * 100) \
            if pd.notna(row.get('Price')) and entry_px > 0 else 0.0
    trend = min(max(ema - eprev, -20.0), 20.0)
    return ema * 0.5 + trend + min(max(pnl, -15.0), 30.0)


def qualifies_for_entry(row, hist_yrs, ticker, min_history_yrs):
    """Check all entry conditions for a candidate ticker."""
    if hist_yrs.get(ticker, 0) < min_history_yrs:
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


def run_sim(df, cap, hist_yrs, trading_dates):
    """
    Run one simulation for one position cap.
    Returns (trades_df, daily_df).
    """
    pos_size    = STARTING_NAV / cap
    open_pos    = {}    # {ticker: entry_price}
    cash        = STARTING_NAV
    trades      = []
    daily_log   = []
    rotation_ct = 0

    sim_name       = f'Cap_{cap:02d}'
    backtest_start = pd.Timestamp(START_DATE)

    for dt in trading_dates:
        day = df[df['Date'] == dt].set_index('Ticker')

        # ── EXITS ──────────────────────────────────────────────────────────
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
                trades.append({'Sim': sim_name, 'Cap': cap,
                                'Ticker': ticker, 'Type': 'EXIT',
                                'Date': dt.date(), 'Exit_Price': round(xp, 2),
                                'Entry_Price': round(ep, 2),
                                'PnL_Pct': round(pnl * 100, 2),
                                'Reason': reason})
                to_exit.append(ticker)
        for t in to_exit:
            del open_pos[t]

        # ── QUALITY ROTATION ───────────────────────────────────────────────
        if USE_ROTATION and len(open_pos) >= cap:
            # Find best qualified candidate not already held
            best_t, best_dm = None, -1
            for ticker, row in day.iterrows():
                if ticker in open_pos:
                    continue
                if qualifies_for_entry(row, hist_yrs, ticker, MIN_HISTORY_YRS):
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

                    if best_dm - weak_score >= ROTATION_THRESHOLD:
                        # Exit weakest
                        xp  = float(day.loc[weak_t, 'Price'])
                        ep  = open_pos[weak_t]
                        pnl = (xp - ep) / ep
                        cash += pos_size * (1 + pnl)
                        trades.append({
                            'Sim': sim_name, 'Cap': cap,
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
                                    'Sim': sim_name, 'Cap': cap,
                                    'Ticker': best_t, 'Type': 'ROTATION_IN',
                                    'Date': dt.date(), 'Exit_Price': None,
                                    'Entry_Price': round(ep_new, 2),
                                    'PnL_Pct': None,
                                    'Reason': f'Replaced {weak_t}'
                                })

        # ── STANDARD ENTRIES ───────────────────────────────────────────────
        if len(open_pos) < cap and cash >= pos_size * 0.9:
            candidates = []
            for ticker, row in day.iterrows():
                if ticker in open_pos:
                    continue
                if qualifies_for_entry(row, hist_yrs, ticker, MIN_HISTORY_YRS):
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
                        'Sim': sim_name, 'Cap': cap,
                        'Ticker': ticker, 'Type': 'ENTRY',
                        'Date': dt.date(), 'Exit_Price': None,
                        'Entry_Price': round(ep, 2),
                        'PnL_Pct': None,
                        'Reason': 'Signal'
                    })

        # ── DAILY NAV (only for backtest window) ───────────────────────────
        if dt >= backtest_start:
            nav = cash
            for ticker, ep in open_pos.items():
                if ticker in day.index and pd.notna(day.loc[ticker, 'Price']):
                    nav += pos_size * (float(day.loc[ticker, 'Price']) / ep)
                else:
                    nav += pos_size  # fallback for missing data day
            daily_log.append({
                'Sim': sim_name, 'Cap': cap, 'Date': dt.date(),
                'NAV': round(nav, 2), 'Positions': len(open_pos),
                'Cash': round(cash, 2)
            })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    daily_df  = pd.DataFrame(daily_log)
    return trades_df, daily_df


# ── METRICS ────────────────────────────────────────────────────────────────────

def calc_metrics(daily_df, cap, trades_df):
    """Compute all performance metrics for one simulation run."""
    if len(daily_df) < 2:
        return None

    daily_df = daily_df.sort_values('Date').reset_index(drop=True)
    nav = daily_df['NAV'].values

    # Returns
    total_return = (nav[-1] / nav[0] - 1) * 100
    years        = len(nav) / 252
    cagr         = (np.power(nav[-1] / nav[0], 1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe
    daily_rets = pd.Series(nav).pct_change().dropna()
    excess     = daily_rets - RISK_FREE_DAILY
    sharpe     = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # Max drawdown
    peak     = pd.Series(nav).cummax()
    drawdown = (pd.Series(nav) - peak) / peak * 100
    max_dd   = float(drawdown.min())

    # Calmar (primary KPI)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade-level stats
    rotations = 0
    win_rate  = 0
    avg_hold  = 0
    same_week_roundtrips = 0
    if len(trades_df) > 0:
        rot_outs = trades_df[trades_df['Type'] == 'ROTATION_OUT']
        rotations = len(rot_outs)

        exits = trades_df[trades_df['Type'].isin(['EXIT', 'ROTATION_OUT'])]
        if len(exits) > 0 and 'PnL_Pct' in exits.columns:
            pnls = exits['PnL_Pct'].dropna()
            if len(pnls) > 0:
                win_rate = (pnls > 0).sum() / len(pnls) * 100

    return {
        'Sim':           f'Cap_{cap:02d}',
        'Cap':           cap,
        'Sharpe':        round(sharpe, 3),
        'CAGR':          round(cagr, 2),
        'Calmar':        round(calmar, 2),
        'Max_Drawdown':  round(max_dd, 2),
        'Total_Return':  round(total_return, 2),
        'Win_Rate':      round(win_rate, 1),
        'Rotations':     rotations,
        'Final_NAV':     round(nav[-1], 0),
        'Trading_Days':  len(nav)
    }


# ── CHARTS ─────────────────────────────────────────────────────────────────────

def build_charts(metrics_df, daily_all):
    """Build comparison charts for the cap sweep."""
    print('\n  Building charts...')

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # ── NAV trajectories (top row, full width) ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0, 1, len(POSITION_CAPS)))

    for i, cap in enumerate(POSITION_CAPS):
        sim_name = f'Cap_{cap:02d}'
        subset = daily_all[daily_all['Sim'] == sim_name].sort_values('Date')
        if len(subset) > 0:
            ax1.plot(pd.to_datetime(subset['Date']), subset['NAV'],
                     label=f'Cap {cap}', color=colors[i], linewidth=1.5)

    ax1.set_title('Cap Sweep — NAV Trajectory ($1M start)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=9, loc='upper left', ncol=4)
    ax1.grid(True, alpha=0.2)
    ax1.set_yscale('log')

    # ── Calmar by cap (primary KPI) ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    metrics_sorted = metrics_df.sort_values('Cap')
    bars = ax2.bar(range(len(metrics_sorted)), metrics_sorted['Calmar'],
                    color='steelblue', alpha=0.8)
    best_idx = metrics_sorted['Calmar'].idxmax()
    bars[metrics_sorted.index.get_loc(best_idx)].set_color('darkorange')
    ax2.set_xticks(range(len(metrics_sorted)))
    ax2.set_xticklabels([f'Cap {c}' for c in metrics_sorted['Cap']], fontsize=9)
    ax2.set_title('Calmar Ratio by Cap (PRIMARY KPI)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Calmar Ratio')
    ax2.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, metrics_sorted['Calmar']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # ── CAGR and Drawdown comparison ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(metrics_sorted))
    width = 0.35
    ax3.bar(x - width/2, metrics_sorted['CAGR'], width, label='CAGR %',
            color='green', alpha=0.7)
    ax3.bar(x + width/2, metrics_sorted['Max_Drawdown'].abs(), width,
            label='|Max DD| %', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Cap {c}' for c in metrics_sorted['Cap']], fontsize=9)
    ax3.set_title('CAGR vs Max Drawdown by Cap', fontsize=12, fontweight='bold')
    ax3.set_ylabel('%')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2, axis='y')

    # ── Sharpe by cap ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    bars4 = ax4.bar(range(len(metrics_sorted)), metrics_sorted['Sharpe'],
                     color='purple', alpha=0.8)
    ax4.set_xticks(range(len(metrics_sorted)))
    ax4.set_xticklabels([f'Cap {c}' for c in metrics_sorted['Cap']], fontsize=9)
    ax4.set_title('Sharpe Ratio by Cap', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sharpe')
    ax4.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars4, metrics_sorted['Sharpe']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # ── Rotations by cap ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    bars5 = ax5.bar(range(len(metrics_sorted)), metrics_sorted['Rotations'],
                     color='teal', alpha=0.8)
    ax5.set_xticks(range(len(metrics_sorted)))
    ax5.set_xticklabels([f'Cap {c}' for c in metrics_sorted['Cap']], fontsize=9)
    ax5.set_title('Total Rotations by Cap (10yr)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('# Rotations')
    ax5.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars5, metrics_sorted['Rotations']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{int(val)}', ha='center', va='bottom', fontsize=9)

    plt.savefig(f'{OUTPUT_DIR}/cap_sweep_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart: {OUTPUT_DIR}/cap_sweep_chart.png")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*70)
    print('  SHADOW_ROTATION — CAP SWEEP WALKFORWARD (10-YEAR)')
    print(f'  {START_DATE} to {END_DATE}')
    print(f'  Caps to test: {POSITION_CAPS}')
    print(f'  Rotation threshold: {ROTATION_THRESHOLD}')
    print(f'  Primary KPI: Calmar Ratio')
    print('='*70)

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

    # Run simulations for each cap
    for cap in POSITION_CAPS:
        print(f"\n  Cap_{cap:02d} — running simulation...")
        trades_df, daily_df = run_sim(df, cap, hist_yrs, trading_dates)

        if len(daily_df) == 0:
            print(f"    WARNING: No daily data produced for cap={cap}.")
            continue

        m = calc_metrics(daily_df, cap, trades_df)
        if m is None:
            continue

        all_metrics.append(m)
        all_daily.append(daily_df)
        all_trades.append(trades_df)

        print(f"    Calmar: {m['Calmar']} | Sharpe: {m['Sharpe']} | "
              f"CAGR: {m['CAGR']}% | MaxDD: {m['Max_Drawdown']}% | "
              f"Rotations: {m['Rotations']}")

    # Combine results
    metrics_df = pd.DataFrame(all_metrics).sort_values('Cap')
    daily_all  = pd.concat(all_daily,  ignore_index=True) if all_daily  else pd.DataFrame()
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Print summary
    print('\n' + '='*70)
    print('  SUMMARY — ALL CAPS')
    print('='*70)
    print(metrics_df[['Cap', 'Calmar', 'Sharpe', 'CAGR', 'Max_Drawdown',
                       'Total_Return', 'Win_Rate', 'Rotations']].to_string(index=False))

    # Identify winner
    if len(metrics_df) > 0:
        best = metrics_df.loc[metrics_df['Calmar'].idxmax()]
        print('\n' + '='*70)
        print(f'  BEST CAP BY CALMAR: Cap_{int(best["Cap"]):02d}')
        print('='*70)
        print(f'    Calmar:       {best["Calmar"]}')
        print(f'    CAGR:         {best["CAGR"]}%')
        print(f'    Max Drawdown: {best["Max_Drawdown"]}%')
        print(f'    Sharpe:       {best["Sharpe"]}')
        print(f'    Total Return: {best["Total_Return"]}%')
        print(f'    Rotations:    {int(best["Rotations"])}')
        print(f'    Final NAV:    ${best["Final_NAV"]:,.0f}')

    # Save outputs
    metrics_df.to_csv(f'{OUTPUT_DIR}/summary_all_caps.csv', index=False)
    print(f'\n  Summary: {OUTPUT_DIR}/summary_all_caps.csv')

    if len(daily_all) > 0:
        for cap in POSITION_CAPS:
            sim_name = f'Cap_{cap:02d}'
            subset = daily_all[daily_all['Sim'] == sim_name]
            if len(subset) > 0:
                subset.to_csv(f'{OUTPUT_DIR}/daily_nav_cap_{cap:02d}.csv', index=False)

    if len(trades_all) > 0:
        for cap in POSITION_CAPS:
            sim_name = f'Cap_{cap:02d}'
            rots = trades_all[
                (trades_all['Sim'] == sim_name) &
                (trades_all['Type'].str.contains('ROTATION', na=False))
            ]
            if len(rots) > 0:
                rots.to_csv(f'{OUTPUT_DIR}/rotation_events_cap_{cap:02d}.csv', index=False)

    # Build charts
    if len(daily_all) > 0 and len(metrics_df) > 0:
        build_charts(metrics_df, daily_all)

    print(f'\n  All results saved to: {OUTPUT_DIR}/')
    print('='*70)
    print('\n  INTERPRETATION GUIDE:')
    print('  - Calmar > 2.0  = good risk-adjusted return')
    print('  - Calmar > 5.0  = exceptional')
    print('  - Calmar > 10.0 = rare; verify no bugs')
    print('  - Compare rotations per cap: lower N typically rotates more')
    print('  - Check that NAV trajectory survives 2018, 2020, 2022 drawdowns')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
