"""
puresim_inception_walkforward.py
==================================
ARGUS Research -- PureSim Strategy 5
Inception-to-Date Walkforward: Actual vs Modified Rules

BEFORE RUNNING -- ONE CSV NEEDED FROM GOOGLE SHEETS:

    puresim_positions.csv
        Spreadsheet:  NEW_FlowOS_preferred
        Tab:          Shadow_PureSim
        Export:       File -> Download -> CSV
        Required columns: Ticker, Entry_Date, Entry_Price
        (Exit_Date, Exit_Price if positions have been closed)
        Place in:     same folder as this script

    If column names differ from above, update POSITION_COL_MAP below.

WHAT THIS SCRIPT DOES:
    Runs two simulations from March 23, 2026 (PureSim inception) to today.

    SIM A -- ACTUAL rules:
        Cap 20 | Min history 3yr | No rotation
        Seeded with actual inception positions from puresim_positions.csv

    SIM B -- MODIFIED rules:
        Cap 25 | Min history 5yr | Quality rotation ON
        Same inception positions as Sim A

    Outputs day-by-day NAV comparison, position differences,
    rotation events, and a comparison chart.

REQUIREMENTS:
    pip install supabase pandas numpy matplotlib python-dotenv
"""

import os
import sys
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

from dotenv import load_dotenv
load_dotenv()

# -- COLUMN NAME MAP ------------------------------------------------------------
# If Shadow_PureSim tab uses different column names, update these.
POSITION_COL_MAP = {
    'ticker':      'Ticker',       # column with ticker symbol
    'entry_date':  'Entry_Date',   # column with entry date
    'entry_price': 'Entry_Price',  # column with entry price
    'exit_date':   'Exit_Date',    # column with exit date (blank if still open)
    'exit_price':  'Exit_Price',   # column with exit price (blank if still open)
}

# -- CONFIGURATION --------------------------------------------------------------
SUPABASE_URL    = os.environ.get('SUPABASE_URL')
SUPABASE_KEY    = os.environ.get('SUPABASE_KEY')

INCEPTION_DATE  = '2026-04-01'
END_DATE        = '2026-04-15'
STARTING_NAV    = 1_000_000.0

DM_ENTRY        = 65
DM_EXIT_MA      = 50
LOSS_LIMIT      = -0.15
RISK_FREE_DAILY = 0.043 / 252
ROTATION_THRESHOLD = 20

# Sim parameters
SIM_A_CAP             = 20
SIM_A_MIN_HISTORY_YRS = 3
SIM_A_ROTATION        = False

SIM_B_CAP             = 25
SIM_B_MIN_HISTORY_YRS = 5
SIM_B_ROTATION        = True

OUTPUT_DIR = './results_inception'


# -- CSV LOADERS ----------------------------------------------------------------

def load_universe():
    """
    Load governed PureSim universe from puresim_universe.csv.

    Export: PureSim_Universe tab in NEW_FlowOS_preferred -> File -> Download -> CSV
    Required columns: Ticker, Status
    Status values accepted: QUALIFY or ACTIVE (both treated as eligible)

    This is the governed list -- only these 96 tickers are eligible for
    entry in either simulation. Sim B's MIN_HISTORY_YEARS=5 filter
    further narrows this list during the simulation.
    """
    path = 'puresim_universe.csv'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("  Export PureSim_Universe tab from NEW_FlowOS_preferred:")
        print("  File -> Download -> Comma Separated Values (.csv)")
        print("  Place in same folder as this script.")
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Find ticker column
    ticker_col = None
    for c in df.columns:
        if c.lower() in ('ticker', 'symbol', 'stock'):
            ticker_col = c
            break
    if not ticker_col:
        print(f"ERROR: No Ticker column found in {path}.")
        print(f"  Columns found: {list(df.columns)}")
        sys.exit(1)

    # Find status column if present -- accept QUALIFY or ACTIVE
    status_col = None
    for c in df.columns:
        if c.lower() in ('status', 'active', 'include'):
            status_col = c
            break

    if status_col:
        eligible = df[status_col].astype(str).str.upper().str.strip()
        df = df[eligible.isin(['QUALIFY', 'ACTIVE', 'TRUE', '1'])]

    tickers = df[ticker_col].dropna().str.strip().str.upper().tolist()
    tickers = list(dict.fromkeys([t for t in tickers if t]))  # deduplicate
    print(f"  Universe loaded: {len(tickers)} tickers from {path}")
    return tickers


def load_inception_positions():
    """
    Load actual inception positions from puresim_positions.csv.
    Returns dict of {ticker: entry_price} for positions open at inception.
    """
    path = 'puresim_positions.csv'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("  Export Shadow_PureSim tab from NEW_FlowOS_preferred as CSV.")
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Map column names
    m = POSITION_COL_MAP
    ticker_col = m['ticker']
    entry_date_col = m['entry_date']
    entry_price_col = m['entry_price']
    exit_date_col = m['exit_date']

    # Validate required columns exist
    for col in [ticker_col, entry_date_col, entry_price_col]:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in {path}.")
            print(f"  Available columns: {list(df.columns)}")
            print(f"  Update POSITION_COL_MAP at the top of the script.")
            sys.exit(1)

    # Parse dates
    df[entry_date_col] = pd.to_datetime(df[entry_date_col], errors='coerce')
    if exit_date_col in df.columns:
        df[exit_date_col] = pd.to_datetime(df[exit_date_col], errors='coerce')

    # Filter: entered on or before inception date, not yet exited at inception
    inception_ts = pd.Timestamp(INCEPTION_DATE)
    mask = df[entry_date_col] <= inception_ts
    if exit_date_col in df.columns:
        # Keep positions where exit_date is blank or after inception
        mask &= (df[exit_date_col].isna() | (df[exit_date_col] > inception_ts))
    df_open = df[mask].copy()

    # Build {ticker: entry_price} dict
    positions = {}
    for _, row in df_open.iterrows():
        ticker = str(row[ticker_col]).strip().upper()
        try:
            price = float(row[entry_price_col])
            if price > 0:
                positions[ticker] = price
        except (ValueError, TypeError):
            print(f"  WARNING: Could not parse entry price for {ticker} -- skipping")

    print(f"  Inception positions loaded: {len(positions)} open at {INCEPTION_DATE}")
    for t, p in sorted(positions.items()):
        print(f"    {t}: ${p:.2f}")

    return positions


# -- SUPABASE DATA LOADING ------------------------------------------------------

def load_dm_history(universe):
    """Load DM history from Supabase for the universe tickers."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)

    load_from = '2021-01-01'  # 5yr lookback for history calculation
    print(f"\n  Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"  Loading dm_history ({load_from} to {END_DATE})...")

    all_rows = []
    batch_size = 10000
    offset = 0

    while True:
        response = client.table('dm_history')\
            .select('date,ticker,close,dm_smoothed')\
            .gte('date', load_from)\
            .lte('date', END_DATE)\
            .order('date')\
            .range(offset, offset + batch_size - 1)\
            .execute()

        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        offset += batch_size
        print(f"    {len(all_rows):,} rows...", end='\r')
        if len(rows) < batch_size:
            break

    print(f"\n  Total: {len(all_rows):,} rows")
    df = pd.DataFrame(all_rows)
    df['date']  = pd.to_datetime(df['date'])
    df['dm_smoothed'] = pd.to_numeric(df['dm_smoothed'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['date', 'dm_smoothed', 'close'])
    df = df.rename(columns={'date': 'Date', 'ticker': 'Ticker',
                              'close': 'Price', 'dm_smoothed': 'DM'})
    # Filter to universe
    df = df[df['Ticker'].isin(universe)].copy()
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f"  After universe filter: {df['Ticker'].nunique()} tickers")
    return df


def add_indicators(df):
    """Add DM_MA20, EMA5, EMA5_prev, PriceMA20."""
    df['DM_MA20']    = df.groupby('Ticker')['DM'].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    df['EMA5']       = df.groupby('Ticker')['DM'].transform(
        lambda x: x.ewm(span=5, adjust=False).mean())
    df['EMA5_prev']  = df.groupby('Ticker')['EMA5'].shift(1)
    df['PriceMA20']  = df.groupby('Ticker')['Price'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    return df


def get_history_years(df):
    result = {}
    for ticker, grp in df.groupby('Ticker'):
        days = (grp['Date'].max() - grp['Date'].min()).days
        result[ticker] = days / 365.25
    return result


# -- SIMULATION -----------------------------------------------------------------

def entry_candidates(day, open_pos, hist_yrs, min_yrs, cap):
    candidates = []
    for ticker, row in day.iterrows():
        if ticker in open_pos:
            continue
        if hist_yrs.get(ticker, 0) < min_yrs:
            continue
        if not (pd.notna(row['DM']) and row['DM'] >= DM_ENTRY):
            continue
        if not (pd.notna(row.get('EMA5')) and pd.notna(row.get('EMA5_prev'))
                and row['EMA5'] > row['EMA5_prev']):
            continue
        if not (pd.notna(row.get('PriceMA20')) and pd.notna(row['Price'])
                and row['Price'] > row['PriceMA20']):
            continue
        if not (pd.notna(row.get('DM_MA20')) and row['DM_MA20'] >= DM_EXIT_MA):
            continue
        candidates.append((ticker, row['DM']))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def composite_score(row, entry_px):
    dm    = row['DM'] if pd.notna(row.get('DM')) else 0
    trend = (row['EMA5'] - row['EMA5_prev']) \
            if pd.notna(row.get('EMA5')) and pd.notna(row.get('EMA5_prev')) else 0
    pnl   = ((row['Price'] - entry_px) / entry_px * 100) \
            if pd.notna(row.get('Price')) and entry_px > 0 else 0
    return dm * 0.5 + min(max(trend, -20), 20) + min(max(pnl, -15), 30)


def should_exit(row, entry_px):
    if pd.notna(row.get('DM_MA20')) and row['DM_MA20'] < DM_EXIT_MA:
        return True, 'DM_MA20_EXIT'
    if pd.notna(row.get('Price')) and entry_px > 0:
        if (row['Price'] - entry_px) / entry_px < LOSS_LIMIT:
            return True, 'LOSS_LIMIT'
    return False, None


def run_sim(df, name, cap, min_yrs, rotation, inception_pos):
    print(f"\n  Running {name}...")
    print(f"    Cap:{cap} | MinHistory:{min_yrs}yr | Rotation:{rotation}")

    hist_yrs   = get_history_years(df)
    pos_size   = STARTING_NAV / cap
    open_pos   = dict(inception_pos)
    cash       = STARTING_NAV - len(open_pos) * pos_size
    trades     = []
    daily_log  = []

    trading_dates = sorted(
        df[df['Date'] >= pd.Timestamp(INCEPTION_DATE)]['Date'].unique())

    for dt in trading_dates:
        day = df[df['Date'] == dt].set_index('Ticker')

        # Exits
        to_exit = []
        for t, ep in open_pos.items():
            if t not in day.index:
                continue
            flag, reason = should_exit(day.loc[t], ep)
            if flag:
                xp = day.loc[t, 'Price']
                pnl = (xp - ep) / ep
                cash += pos_size * (1 + pnl)
                trades.append({'Sim': name, 'Ticker': t,
                                'Type': 'EXIT', 'Date': dt.date(),
                                'Price': round(xp, 2),
                                'Entry_Price': ep,
                                'PnL_Pct': round(pnl * 100, 2),
                                'Reason': reason})
                to_exit.append(t)
        for t in to_exit:
            del open_pos[t]

        # Quality rotation (Sim B only)
        if rotation and len(open_pos) >= cap:
            cands = entry_candidates(day, open_pos, hist_yrs, min_yrs, cap)
            if cands:
                best_t, best_dm = cands[0]
                scores = {t: composite_score(day.loc[t], ep)
                          for t, ep in open_pos.items() if t in day.index}
                if scores:
                    weak_t = min(scores, key=scores.get)
                    if best_dm - scores[weak_t] >= ROTATION_THRESHOLD:
                        # Exit weak
                        xp = day.loc[weak_t, 'Price']
                        ep = open_pos[weak_t]
                        pnl = (xp - ep) / ep
                        cash += pos_size * (1 + pnl)
                        trades.append({'Sim': name, 'Ticker': weak_t,
                                       'Type': 'ROTATION_OUT',
                                       'Date': dt.date(),
                                       'Price': round(xp, 2),
                                       'Entry_Price': ep,
                                       'PnL_Pct': round(pnl * 100, 2),
                                       'Reason': f'Replaced by {best_t} '
                                                  f'(score gap {best_dm - scores[weak_t]:.0f})'})
                        del open_pos[weak_t]
                        # Enter best
                        if best_t in day.index and cash >= pos_size * 0.9:
                            ep_new = day.loc[best_t, 'Price']
                            if pd.notna(ep_new) and ep_new > 0:
                                open_pos[best_t] = ep_new
                                cash -= pos_size
                                trades.append({'Sim': name, 'Ticker': best_t,
                                               'Type': 'ROTATION_IN',
                                               'Date': dt.date(),
                                               'Price': round(ep_new, 2),
                                               'Entry_Price': None,
                                               'PnL_Pct': None,
                                               'Reason': f'Replaced {weak_t}'})

        # Entries
        cands = entry_candidates(day, open_pos, hist_yrs, min_yrs, cap)
        for t, _ in cands:
            if len(open_pos) >= cap or cash < pos_size * 0.9:
                break
            if t not in day.index:
                continue
            ep = day.loc[t, 'Price']
            if pd.notna(ep) and ep > 0:
                open_pos[t] = ep
                cash -= pos_size
                trades.append({'Sim': name, 'Ticker': t,
                                'Type': 'ENTRY', 'Date': dt.date(),
                                'Price': round(ep, 2),
                                'Entry_Price': None,
                                'PnL_Pct': None,
                                'Reason': 'Signal'})

        # Daily NAV
        nav = cash
        for t, ep in open_pos.items():
            if t in day.index:
                cp = day.loc[t, 'Price']
                nav += pos_size * (cp / ep) if pd.notna(cp) and ep > 0 \
                       else pos_size
            else:
                nav += pos_size

        daily_log.append({
            'Date':      dt.date(),
            'Sim':       name,
            'NAV':       round(nav, 2),
            'NAV_Pct':   round((nav - STARTING_NAV) / STARTING_NAV * 100, 2),
            'Positions': len(open_pos),
            'Cash':      round(cash, 2),
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_log)


# -- METRICS + CHART ------------------------------------------------------------

def metrics(daily_df, name):
    r = daily_df['NAV'].pct_change().fillna(0)
    ex = r - RISK_FREE_DAILY
    sh = (ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0
    cum = (1 + r).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return {
        'Sim':          name,
        'Final_NAV':    f'${daily_df["NAV"].iloc[-1]:,.2f}',
        'Total_Return': f'{(daily_df["NAV"].iloc[-1] - STARTING_NAV) / STARTING_NAV * 100:.2f}%',
        'Sharpe':       round(sh, 2),
        'Max_Drawdown': f'{dd:.2f}%',
        'Avg_Positions': round(daily_df['Positions'].mean(), 1),
    }


def build_chart(da, db, ma, mb):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'PureSim -- Inception Walkforward  |  {INCEPTION_DATE} to {END_DATE}\n'
        f'Sim A: Cap {SIM_A_CAP}, {SIM_A_MIN_HISTORY_YRS}yr, no rotation  '
        f'vs  Sim B: Cap {SIM_B_CAP}, {SIM_B_MIN_HISTORY_YRS}yr, rotation ON',
        fontsize=12)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pd.to_datetime(da['Date']), da['NAV'],
             color='steelblue', lw=2.5, label='Sim A -- Actual')
    ax1.plot(pd.to_datetime(db['Date']), db['NAV'],
             color='darkorange', lw=2.5, ls='--', label='Sim B -- Modified')
    ax1.axhline(STARTING_NAV, color='gray', lw=0.8, ls=':')
    ax1.set_title('NAV -- Actual vs Modified Rules', fontsize=11)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(pd.to_datetime(da['Date']), da['Positions'],
             color='steelblue', label='Sim A')
    ax2.plot(pd.to_datetime(db['Date']), db['Positions'],
             color='darkorange', ls='--', label='Sim B')
    ax2.axhline(SIM_A_CAP, color='steelblue', lw=0.5, ls=':')
    ax2.axhline(SIM_B_CAP, color='darkorange', lw=0.5, ls=':')
    ax2.set_title('Positions Held Per Day', fontsize=11)
    ax2.set_ylabel('# Positions')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    labels = ['Total Return', 'Sharpe', 'Max DD', 'Avg Pos']
    def parse(v):
        try: return float(str(v).replace('%','').replace('$','').replace(',',''))
        except: return 0
    va = [parse(ma['Total_Return']), ma['Sharpe'],
          parse(ma['Max_Drawdown']), ma['Avg_Positions']]
    vb = [parse(mb['Total_Return']), mb['Sharpe'],
          parse(mb['Max_Drawdown']), mb['Avg_Positions']]
    x = np.arange(len(labels))
    w = 0.35
    ba = ax3.bar(x - w/2, va, w, label='Sim A', color='steelblue', alpha=0.8)
    bb = ax3.bar(x + w/2, vb, w, label='Sim B', color='darkorange', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_title('Metrics Comparison', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.axhline(0, color='black', lw=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(list(ba) + list(bb), va + vb):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = f'{OUTPUT_DIR}/inception_walkforward_chart.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Chart: {out}")


# -- MAIN -----------------------------------------------------------------------

def main():
    print('\n' + '='*65)
    print('  PURESIM -- INCEPTION WALKFORWARD')
    print(f'  {INCEPTION_DATE} -> {END_DATE}')
    print(f'  Sim A: Cap {SIM_A_CAP} | {SIM_A_MIN_HISTORY_YRS}yr | No rotation')
    print(f'  Sim B: Cap {SIM_B_CAP} | {SIM_B_MIN_HISTORY_YRS}yr | Rotation ON')
    print('='*65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load universe and positions from CSV
    print('\nLoading local CSV files...')
    universe     = load_universe()
    inception_pos = load_inception_positions()

    # Load DM history
    print('\nLoading DM history from Supabase...')
    df = load_dm_history(universe)
    df = add_indicators(df)

    # Run simulations
    trades_a, daily_a = run_sim(df, 'Sim_A_Actual',
                                 SIM_A_CAP, SIM_A_MIN_HISTORY_YRS,
                                 SIM_A_ROTATION, inception_pos)
    trades_b, daily_b = run_sim(df, 'Sim_B_Modified',
                                 SIM_B_CAP, SIM_B_MIN_HISTORY_YRS,
                                 SIM_B_ROTATION, inception_pos)

    # Metrics
    ma = metrics(daily_a, 'Sim_A_Actual')
    mb = metrics(daily_b, 'Sim_B_Modified')

    print('\n' + '='*65)
    print('  RESULTS SUMMARY')
    print('='*65)
    comp = pd.DataFrame([ma, mb])
    print(comp.to_string(index=False))

    nav_a = float(str(ma['Final_NAV']).replace('$','').replace(',',''))
    nav_b = float(str(mb['Final_NAV']).replace('$','').replace(',',''))
    print(f'\n  NAV difference (B vs A): ${nav_b - nav_a:+,.2f}')

    # Trades unique to each sim
    entries_a = set(trades_a[trades_a['Type'] == 'ENTRY']['Ticker']) \
                if not trades_a.empty else set()
    entries_b = set(trades_b[trades_b['Type'].isin(
                    ['ENTRY','ROTATION_IN'])]['Ticker']) \
                if not trades_b.empty else set()

    only_b = entries_b - entries_a
    only_a = entries_a - entries_b

    print(f'\n  New entries in Sim B only (cap 25 + rotation):')
    print(f'    {sorted(only_b) if only_b else "none -- book full throughout"}')
    print(f'\n  Entries in Sim A excluded by 5yr rule in Sim B:')
    print(f'    {sorted(only_a) if only_a else "none"}')

    # Rotation events
    if not trades_b.empty:
        rots = trades_b[trades_b['Type'].str.contains('ROTATION', na=False)]
        if not rots.empty:
            print(f'\n  Rotation events ({len(rots)}):')
            print(rots[['Ticker','Type','Date','PnL_Pct','Reason']].to_string(index=False))
        else:
            print('\n  No rotation events fired during this window.')

    # Save outputs
    pd.concat([daily_a, daily_b]).to_csv(
        f'{OUTPUT_DIR}/daily_nav_comparison.csv', index=False)
    trades_a.to_csv(f'{OUTPUT_DIR}/positions_sim_a.csv', index=False)
    trades_b.to_csv(f'{OUTPUT_DIR}/positions_sim_b.csv', index=False)
    pd.DataFrame([ma, mb]).to_csv(
        f'{OUTPUT_DIR}/summary_comparison.csv', index=False)

    # Diff table
    all_t = sorted(only_a | only_b)
    if all_t:
        pd.DataFrame([{
            'Ticker': t,
            'In_SimA': t in entries_a,
            'In_SimB': t in entries_b,
            'Note': 'New in B' if t in only_b else 'Excluded in B (5yr rule)'
        } for t in all_t]).to_csv(
            f'{OUTPUT_DIR}/entries_exits_diff.csv', index=False)

    build_chart(daily_a, daily_b, ma, mb)

    print(f'\n  All results: {OUTPUT_DIR}/')
    print('='*65)


if __name__ == '__main__':
    main()
