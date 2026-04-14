"""
ticker_replacement_analysis.py
================================
ARGUS Research -- PureSim Strategy 5
Ticker Replacement Optimization

PURPOSE:
    Finds the optimal set of tickers for PureSim at both cap=20 and cap=25.
    Replaces the intuitive "watch list" approach with an empirical answer:
    which combination of QUALIFY tickers produces the best Sharpe with
    the lowest correlation penalty?

TWO INDEPENDENT RUNS:
    Run A -- Cap 20: Find the best 20 tickers from the full QUALIFY universe
    Run B -- Cap 25: Find the best 25 tickers from the full QUALIFY universe

METHODOLOGY:
    Step 1 -- Score every QUALIFY ticker individually:
        - Hit rate (from PureSim_Universe tab)
        - Backtest return (avg return per signal in DM_2024_2026)
        - DM trend (current EMA5 direction)
        - Composite score = weighted combination

    Step 2 -- Score every QUALIFY ticker for portfolio fit:
        - Incremental correlation: what does adding this ticker do to
          average pairwise correlation of the current set?
        - Diversification value = high hit rate + low incremental correlation

    Step 3 -- Greedy portfolio construction:
        Start with the highest-scoring ticker. Add tickers one by one,
        at each step choosing the candidate that maximizes:
            (hit_rate * HIT_WEIGHT) + (backtest_return * RETURN_WEIGHT)
            - (incremental_correlation * CORR_PENALTY)
        Stop at cap (20 or 25).

    Step 4 -- Compare to current active universe:
        Which current tickers are NOT in the optimal set?
        Which QUALIFY tickers are in the optimal set but not currently active?
        Output: ranked swap recommendations.

    Step 5 -- Backtest validation:
        Run the full backtest on the optimal portfolio vs current portfolio.
        Compare Sharpe, total return, max drawdown.

DATA SOURCE:
    Google Sheets -- Staging SS
    Sheet ID: 1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA
    Tabs: DM_2024_2026, PureSim_Universe

REQUIREMENTS:
    pip install pandas numpy gspread oauth2client scipy matplotlib seaborn

OUTPUT:
    results/replacement_cap20_optimal_set.csv
    results/replacement_cap20_swap_recommendations.csv
    results/replacement_cap20_backtest_comparison.csv
    results/replacement_cap25_optimal_set.csv
    results/replacement_cap25_swap_recommendations.csv
    results/replacement_cap25_backtest_comparison.csv
    results/replacement_comparison_chart.png
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

# -- CONFIGURATION --------------------------------------------------------------

CREDS_PATH     = 'credentials.json'
STAGING_SS_ID  = '1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA'
DM_HISTORY_TAB = 'DM_2024_2026'
UNIVERSE_TAB   = 'PureSim_Universe'

# Strategy parameters (unchanged)
DM_ENTRY       = 65
DM_EXIT_MA     = 50
LOSS_LIMIT     = -0.15
START_DATE     = '2023-01-02'
END_DATE       = '2026-03-15'
RISK_FREE_DAILY = 0.043 / 252

# Portfolio construction weights
HIT_WEIGHT     = 0.40    # Weight for hit rate in composite score
RETURN_WEIGHT  = 0.40    # Weight for avg backtest return
CORR_PENALTY   = 0.20    # Penalty for incremental correlation

# Current active PureSim tickers (as of April 14, 2026)
CURRENT_ACTIVE = [
    'MRVL', 'INTC', 'AMD', 'WDC', 'KLAC', 'DELL', 'GEV', 'GSAT',
    'NBIS', 'ARM', 'ALB', 'TER', 'NFLX', 'AMAT', 'HAL', 'DOW',
    'TWLO', 'DOCN', 'INSM', 'RUN'
]

# Caps to test independently
CAPS_TO_TEST = [20, 25]

OUTPUT_DIR = './results'


# -- DATA LOADING ---------------------------------------------------------------

def load_sheets():
    """Load DM_2024_2026 and PureSim_Universe from Google Sheets."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope  = ['https://spreadsheets.google.com/feeds',
                  'https://www.googleapis.com/auth/drive']
        creds  = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
        client = gspread.authorize(creds)
        ss     = client.open_by_key(STAGING_SS_ID)

        print('  Loading DM_2024_2026...')
        dm_ws   = ss.worksheet(DM_HISTORY_TAB)
        dm_df   = pd.DataFrame(dm_ws.get_all_records())
        print(f'    {len(dm_df):,} rows loaded')

        print('  Loading PureSim_Universe...')
        uni_ws  = ss.worksheet(UNIVERSE_TAB)
        uni_df  = pd.DataFrame(uni_ws.get_all_records())
        print(f'    {len(uni_df)} tickers loaded')

        return dm_df, uni_df

    except Exception as e:
        print(f'  ERROR connecting to Google Sheets: {e}')
        # Try CSV fallback
        dm_path  = 'DM_2024_2026.csv'
        uni_path = 'PureSim_Universe.csv'
        if os.path.exists(dm_path) and os.path.exists(uni_path):
            print('  Loading from CSV fallback...')
            dm_df  = pd.read_csv(dm_path)
            uni_df = pd.read_csv(uni_path)
            return dm_df, uni_df
        sys.exit('No data source. Export tabs to CSV first.')


def prepare_dm(dm_df):
    """Clean and prepare DM history."""
    dm_df.columns = [c.strip() for c in dm_df.columns]

    # Identify columns
    date_col   = next((c for c in dm_df.columns if 'date' in c.lower()), None)
    ticker_col = next((c for c in dm_df.columns if 'ticker' in c.lower()), None)
    dm_col     = next((c for c in dm_df.columns if c.upper() == 'DM'), None)
    ema5_col   = next((c for c in dm_df.columns
                       if 'ema5' in c.lower() or 'ma5' in c.lower()), None)
    price_col  = next((c for c in dm_df.columns
                       if c.lower() in ['price', 'close']), None)
    pma20_col  = next((c for c in dm_df.columns
                       if 'pricema20' in c.lower() or 'price_ma20' in c.lower()), None)

    rename = {date_col: 'Date', ticker_col: 'Ticker',
              dm_col: 'DM', price_col: 'Price'}
    if ema5_col:  rename[ema5_col]  = 'EMA5'
    if pma20_col: rename[pma20_col] = 'PriceMA20'
    dm_df = dm_df.rename(columns=rename)

    dm_df['Date']  = pd.to_datetime(dm_df['Date'],  errors='coerce')
    dm_df['DM']    = pd.to_numeric(dm_df['DM'],    errors='coerce')
    dm_df['Price'] = pd.to_numeric(dm_df['Price'], errors='coerce')
    if 'EMA5'      in dm_df.columns:
        dm_df['EMA5']      = pd.to_numeric(dm_df['EMA5'],      errors='coerce')
    if 'PriceMA20' in dm_df.columns:
        dm_df['PriceMA20'] = pd.to_numeric(dm_df['PriceMA20'], errors='coerce')

    dm_df = dm_df.dropna(subset=['Date'])
    dm_df = dm_df[(dm_df['Date'] >= START_DATE) & (dm_df['Date'] <= END_DATE)]
    dm_df = dm_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    dm_df['DM_MA20']    = dm_df.groupby('Ticker')['DM'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    if 'EMA5' in dm_df.columns:
        dm_df['EMA5_prev'] = dm_df.groupby('Ticker')['EMA5'].shift(1)
    dm_df['Price_prev'] = dm_df.groupby('Ticker')['Price'].shift(1)

    return dm_df


def get_qualify_universe(uni_df):
    """Extract QUALIFY tickers from PureSim_Universe."""
    uni_df.columns = [c.strip() for c in uni_df.columns]

    # Find ticker and status columns
    ticker_col = next((c for c in uni_df.columns if 'ticker' in c.lower()), None)
    status_col = next((c for c in uni_df.columns if 'status' in c.lower()), None)
    hitrate_col = next((c for c in uni_df.columns
                        if 'hit' in c.lower() and 'rate' in c.lower()), None)
    signals_col = next((c for c in uni_df.columns
                        if 'signal' in c.lower() or 'count' in c.lower()), None)

    if not ticker_col:
        print('  WARNING: Cannot find Ticker column in universe tab.')
        return pd.DataFrame()

    qualify = uni_df[
        uni_df[status_col].str.upper().isin(['QUALIFY', 'ACTIVE'])
    ].copy() if status_col else uni_df.copy()

    qualify = qualify.rename(columns={ticker_col: 'Ticker'})
    if hitrate_col: qualify = qualify.rename(columns={hitrate_col: 'Hit_Rate'})
    if signals_col: qualify = qualify.rename(columns={signals_col: 'Signal_Count'})

    if 'Hit_Rate' in qualify.columns:
        qualify['Hit_Rate'] = pd.to_numeric(qualify['Hit_Rate'], errors='coerce')

    print(f'  QUALIFY tickers: {len(qualify)}')
    return qualify[['Ticker'] + [c for c in ['Hit_Rate', 'Signal_Count']
                                  if c in qualify.columns]]


# -- TICKER SCORING -------------------------------------------------------------

def score_tickers_individually(dm_df, qualify_df):
    """
    Score every QUALIFY ticker on individual merit.
    Returns DataFrame with composite score per ticker.
    """
    print('\n  Scoring individual tickers...')

    ticker_scores = []
    qualify_tickers = qualify_df['Ticker'].tolist()
    available = [t for t in qualify_tickers if t in dm_df['Ticker'].unique()]

    print(f'  {len(available)} of {len(qualify_tickers)} QUALIFY tickers in DM history')

    for ticker in available:
        t_data = dm_df[dm_df['Ticker'] == ticker].copy()

        # -- Individual backtest --------------------------------------------
        trades   = []
        in_trade = False
        entry_px = None

        for _, row in t_data.iterrows():
            if not in_trade:
                # Check entry
                dm_ok    = pd.notna(row['DM']) and row['DM'] >= DM_ENTRY
                ema_ok   = True
                price_ok = True
                if 'EMA5' in row and 'EMA5_prev' in row:
                    if pd.notna(row['EMA5']) and pd.notna(row['EMA5_prev']):
                        ema_ok = row['EMA5'] > row['EMA5_prev']
                if 'PriceMA20' in row:
                    if pd.notna(row['PriceMA20']) and pd.notna(row['Price']):
                        price_ok = row['Price'] > row['PriceMA20']
                if dm_ok and ema_ok and price_ok and pd.notna(row['Price']):
                    in_trade = True
                    entry_px = row['Price']
            else:
                # Check exit
                exit_flag = False
                if pd.notna(row.get('DM_MA20')) and row['DM_MA20'] < DM_EXIT_MA:
                    exit_flag = True
                if pd.notna(row['Price']) and entry_px > 0:
                    if (row['Price'] - entry_px) / entry_px < LOSS_LIMIT:
                        exit_flag = True
                if exit_flag and pd.notna(row['Price']):
                    trades.append((row['Price'] - entry_px) / entry_px)
                    in_trade = False
                    entry_px = None

        avg_return  = np.mean(trades) if trades else 0
        signal_count = len(trades)
        win_rate    = np.mean([1 if t > 0 else 0 for t in trades]) if trades else 0

        # Get hit rate from universe tab
        uni_row  = qualify_df[qualify_df['Ticker'] == ticker]
        hit_rate = float(uni_row['Hit_Rate'].values[0]) / 100 \
                   if len(uni_row) > 0 and 'Hit_Rate' in uni_row.columns \
                   else win_rate

        # Normalize scores (0-1)
        ticker_scores.append({
            'Ticker':        ticker,
            'Hit_Rate':      hit_rate,
            'Avg_Return':    avg_return,
            'Signal_Count':  signal_count,
            'Win_Rate':      win_rate,
            'In_Current':    ticker in CURRENT_ACTIVE
        })

    scores_df = pd.DataFrame(ticker_scores)

    # Normalize for composite score
    if len(scores_df) > 0:
        scores_df['Hit_Rate_N']   = (scores_df['Hit_Rate'] - scores_df['Hit_Rate'].min()) / \
                                     (scores_df['Hit_Rate'].max() - scores_df['Hit_Rate'].min() + 1e-9)
        scores_df['Return_N']     = (scores_df['Avg_Return'] - scores_df['Avg_Return'].min()) / \
                                     (scores_df['Avg_Return'].max() - scores_df['Avg_Return'].min() + 1e-9)
        scores_df['Composite']    = (HIT_WEIGHT   * scores_df['Hit_Rate_N'] +
                                     RETURN_WEIGHT * scores_df['Return_N'])
        scores_df = scores_df.sort_values('Composite', ascending=False)

    return scores_df


# -- CORRELATION MATRIX --------------------------------------------------------

def build_return_matrix(dm_df, tickers):
    """Build daily return matrix for given tickers."""
    price_pivot = dm_df[dm_df['Ticker'].isin(tickers)].pivot_table(
        index='Date', columns='Ticker', values='Price'
    )
    return price_pivot.pct_change().dropna()


def incremental_correlation(new_ticker, current_set, return_matrix):
    """
    Calculate the average pairwise correlation added by including new_ticker
    in current_set. Lower is better (more diversification).
    """
    if not current_set or new_ticker not in return_matrix.columns:
        return 0.0

    available = [t for t in current_set if t in return_matrix.columns]
    if not available:
        return 0.0

    new_returns = return_matrix[new_ticker]
    corrs = [abs(new_returns.corr(return_matrix[t])) for t in available
             if pd.notna(new_returns.corr(return_matrix[t]))]
    return np.mean(corrs) if corrs else 0.0


# -- GREEDY PORTFOLIO CONSTRUCTION ---------------------------------------------

def build_optimal_portfolio(scores_df, dm_df, cap, label):
    """
    Greedy construction: add tickers one by one, maximizing composite score
    minus correlation penalty.
    """
    print(f'\n  Building optimal portfolio -- Cap {cap}...')

    available_tickers = scores_df['Ticker'].tolist()
    return_matrix     = build_return_matrix(dm_df, available_tickers)

    selected  = []
    remaining = available_tickers.copy()

    while len(selected) < cap and remaining:
        best_ticker = None
        best_score  = -999

        for ticker in remaining:
            if ticker not in scores_df['Ticker'].values:
                continue

            # Individual score
            row  = scores_df[scores_df['Ticker'] == ticker].iloc[0]
            ind_score = row['Composite']

            # Correlation penalty
            corr_pen = incremental_correlation(ticker, selected, return_matrix)

            # Combined score
            combined = ind_score - (CORR_PENALTY * corr_pen)

            if combined > best_score:
                best_score  = combined
                best_ticker = ticker

        if best_ticker:
            selected.append(best_ticker)
            remaining.remove(best_ticker)
            print(f'    [{len(selected):2d}] Added {best_ticker} '
                  f'(score: {best_score:.3f})')

    print(f'  [OK] Optimal portfolio built: {len(selected)} tickers')
    return selected


# -- BACKTEST ENGINE ------------------------------------------------------------

def run_backtest_for_portfolio(dm_df, tickers, cap):
    """Run PureSim backtest for a specific set of tickers."""
    df = dm_df[dm_df['Ticker'].isin(tickers)].copy()
    trading_dates   = sorted(df['Date'].unique())
    open_positions  = {}
    trades          = []
    daily_returns   = []

    for date in trading_dates:
        day_data = df[df['Date'] == date].set_index('Ticker')

        # Exits
        to_exit = []
        for ticker, entry_price in open_positions.items():
            if ticker not in day_data.index:
                continue
            row = day_data.loc[ticker]
            exit_flag = False
            if pd.notna(row.get('DM_MA20')) and row['DM_MA20'] < DM_EXIT_MA:
                exit_flag = True
            if pd.notna(row['Price']) and entry_price > 0:
                if (row['Price'] - entry_price) / entry_price < LOSS_LIMIT:
                    exit_flag = True
            if exit_flag:
                pnl = (row['Price'] - entry_price) / entry_price
                trades.append({'Ticker': ticker, 'PnL': pnl})
                to_exit.append(ticker)
        for t in to_exit:
            del open_positions[t]

        # Daily return
        if open_positions:
            pos_rets = []
            for ticker, entry_price in open_positions.items():
                if ticker in day_data.index:
                    curr = day_data.loc[ticker, 'Price']
                    prev = day_data.loc[ticker, 'Price_prev']
                    if pd.notna(curr) and pd.notna(prev) and prev > 0:
                        pos_rets.append((curr - prev) / prev)
            if pos_rets:
                daily_returns.append({'Date': date, 'Return': np.mean(pos_rets)})
        else:
            daily_returns.append({'Date': date, 'Return': 0.0})

        # Entries
        candidates = [(t, row['DM']) for t, row in day_data.iterrows()
                      if t not in open_positions
                      and pd.notna(row['DM']) and row['DM'] >= DM_ENTRY
                      and (not ('EMA5' in row and 'EMA5_prev' in row)
                           or (pd.notna(row.get('EMA5')) and pd.notna(row.get('EMA5_prev'))
                               and row['EMA5'] > row['EMA5_prev']))
                      and (not ('PriceMA20' in row)
                           or (pd.notna(row.get('PriceMA20')) and pd.notna(row['Price'])
                               and row['Price'] > row['PriceMA20']))]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, _ in candidates:
            if len(open_positions) < cap:
                px = day_data.loc[ticker, 'Price']
                if pd.notna(px) and px > 0:
                    open_positions[ticker] = px

    trades_df = pd.DataFrame(trades)   if trades        else pd.DataFrame()
    daily_df  = pd.DataFrame(daily_returns) if daily_returns else pd.DataFrame()
    return trades_df, daily_df


def calc_metrics(trades_df, daily_df, label):
    """Calculate performance metrics."""
    if trades_df.empty or daily_df.empty:
        return {'Label': label, 'Sharpe': 0, 'Total_Return': 0,
                'Max_Drawdown': 0, 'Win_Rate': 0, 'Trades': 0, 'Avg_Return': 0}

    returns   = daily_df['Return'].fillna(0)
    excess    = returns - RISK_FREE_DAILY
    sharpe    = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    cum_ret   = (1 + returns).prod() - 1
    cum_curve = (1 + returns).cumprod()
    rolling_max = cum_curve.cummax()
    max_dd    = ((cum_curve - rolling_max) / rolling_max).min()
    win_rate  = (trades_df['PnL'] > 0).mean() if len(trades_df) > 0 else 0

    return {
        'Label':        label,
        'Sharpe':       round(sharpe, 2),
        'Total_Return': round(cum_ret * 100, 1),
        'Max_Drawdown': round(max_dd * 100, 1),
        'Win_Rate':     round(win_rate * 100, 1),
        'Trades':       len(trades_df),
        'Avg_Return':   round(trades_df['PnL'].mean() * 100, 1) if len(trades_df) > 0 else 0
    }


# -- SWAP RECOMMENDATIONS -------------------------------------------------------

def generate_swap_recommendations(current, optimal, scores_df, cap):
    """
    Compare current active universe to optimal set.
    Output ranked swap recommendations.
    """
    to_remove = [t for t in current if t not in optimal]
    to_add    = [t for t in optimal  if t not in current]

    swaps = []
    for rem, add in zip(to_remove, to_add):
        rem_score = scores_df[scores_df['Ticker'] == rem]['Composite'].values
        add_score = scores_df[scores_df['Ticker'] == add]['Composite'].values
        swaps.append({
            'Remove':        rem,
            'Remove_Score':  round(rem_score[0], 3) if len(rem_score) > 0 else 0,
            'Add':           add,
            'Add_Score':     round(add_score[0], 3) if len(add_score) > 0 else 0,
            'Score_Delta':   round((add_score[0] if len(add_score) > 0 else 0) -
                                   (rem_score[0] if len(rem_score) > 0 else 0), 3)
        })

    swaps_df = pd.DataFrame(swaps).sort_values('Score_Delta', ascending=False)

    # Also list slots being added (cap 25 gets 5 new slots)
    new_slots = [t for t in optimal if t not in current and t not in
                 [s['Add'] for s in swaps]]

    return swaps_df, to_remove, to_add, new_slots


# -- MAIN -----------------------------------------------------------------------

def main():
    print('\n' + '='*65)
    print('  ARGUS RESEARCH -- TICKER REPLACEMENT OPTIMIZATION')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('='*65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print('\nLoading data...')
    dm_raw, uni_raw = load_sheets()
    dm_df    = prepare_dm(dm_raw)
    qualify_df = get_qualify_universe(uni_raw)

    if qualify_df.empty:
        sys.exit('No QUALIFY tickers found. Check PureSim_Universe tab.')

    # Score all tickers individually (done once, used for both caps)
    print('\nScoring individual tickers...')
    scores_df = score_tickers_individually(dm_df, qualify_df)
    scores_df.to_csv(f'{OUTPUT_DIR}/ticker_individual_scores.csv', index=False)
    print(f'  Top 15 tickers by composite score:')
    print(scores_df[['Ticker', 'Hit_Rate', 'Avg_Return',
                      'Signal_Count', 'Composite', 'In_Current']]
          .head(15).to_string(index=False))

    # Run independent analysis for each cap
    all_results = []

    for cap in CAPS_TO_TEST:
        print(f'\n{"="*65}')
        print(f'  CAP = {cap} -- INDEPENDENT ANALYSIS')
        print(f'{"="*65}')

        # Build optimal portfolio
        optimal = build_optimal_portfolio(scores_df, dm_df, cap, f'Cap{cap}')

        # Backtest: current portfolio at this cap
        print(f'\n  Backtesting CURRENT universe at cap={cap}...')
        current_for_cap = CURRENT_ACTIVE[:cap]  # Trim if cap < 20
        curr_trades, curr_daily = run_backtest_for_portfolio(
            dm_df, current_for_cap, cap
        )
        curr_metrics = calc_metrics(curr_trades, curr_daily,
                                     f'Current (cap={cap})')

        # Backtest: optimal portfolio at this cap
        print(f'  Backtesting OPTIMAL universe at cap={cap}...')
        opt_trades, opt_daily = run_backtest_for_portfolio(
            dm_df, optimal, cap
        )
        opt_metrics = calc_metrics(opt_trades, opt_daily,
                                    f'Optimal (cap={cap})')

        print(f'\n  COMPARISON -- Cap {cap}:')
        comp_df = pd.DataFrame([curr_metrics, opt_metrics])
        print(comp_df.to_string(index=False))

        improvement = opt_metrics['Sharpe'] - curr_metrics['Sharpe']
        print(f'\n  Sharpe improvement: {improvement:+.2f}')

        # Swap recommendations
        swaps_df, to_remove, to_add, new_slots = generate_swap_recommendations(
            CURRENT_ACTIVE, optimal, scores_df, cap
        )

        print(f'\n  SWAP RECOMMENDATIONS (Cap {cap}):')
        if len(swaps_df) > 0:
            print(swaps_df.to_string(index=False))
        if new_slots:
            print(f'\n  NEW SLOTS (no equivalent removal needed):')
            for t in new_slots:
                s = scores_df[scores_df['Ticker'] == t]['Composite'].values
                print(f'    ADD {t} (score: {s[0]:.3f})' if len(s) > 0 else f'    ADD {t}')

        # Save outputs
        pd.DataFrame({'Ticker': optimal,
                      'Rank': range(1, len(optimal)+1)}).to_csv(
            f'{OUTPUT_DIR}/replacement_cap{cap}_optimal_set.csv', index=False
        )
        swaps_df.to_csv(
            f'{OUTPUT_DIR}/replacement_cap{cap}_swap_recommendations.csv',
            index=False
        )
        comp_df.to_csv(
            f'{OUTPUT_DIR}/replacement_cap{cap}_backtest_comparison.csv',
            index=False
        )

        all_results.append({
            'cap':          cap,
            'current':      curr_metrics,
            'optimal':      opt_metrics,
            'optimal_set':  optimal,
            'swaps':        swaps_df,
            'to_remove':    to_remove,
            'to_add':       to_add,
            'new_slots':    new_slots
        })

    # -- Comparison chart across both caps -------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('PureSim -- Ticker Replacement Analysis\nCurrent vs Optimal Universe',
                 fontsize=14)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    metrics_list = ['Sharpe', 'Total_Return', 'Max_Drawdown', 'Win_Rate', 'Avg_Return']
    titles       = ['Sharpe Ratio', 'Total Return %', 'Max Drawdown %',
                    'Win Rate %', 'Avg Return per Trade %']
    colors       = ['steelblue', 'green', 'red', 'orange', 'purple']

    for idx, (metric, title, color) in enumerate(zip(metrics_list, titles, colors)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        x_labels = []
        y_values = []
        bar_colors = []
        for r in all_results:
            x_labels.append(f'Current\nCap{r["cap"]}')
            y_values.append(r['current'][metric])
            bar_colors.append('lightgray')
            x_labels.append(f'Optimal\nCap{r["cap"]}')
            y_values.append(r['optimal'][metric])
            bar_colors.append(color)

        bars = ax.bar(range(len(x_labels)), y_values, color=bar_colors)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Label bars
        for bar, val in zip(bars, y_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/replacement_comparison_chart.png', dpi=150)
    plt.close()

    # -- Final summary ----------------------------------------------------------
    print('\n' + '='*65)
    print('  FINAL SUMMARY')
    print('='*65)

    for r in all_results:
        cap  = r['cap']
        curr = r['current']
        opt  = r['optimal']
        print(f'\n  Cap {cap}:')
        print(f'    Current  -> Sharpe {curr["Sharpe"]} | '
              f'Return {curr["Total_Return"]}% | MaxDD {curr["Max_Drawdown"]}%')
        print(f'    Optimal  -> Sharpe {opt["Sharpe"]} | '
              f'Return {opt["Total_Return"]}% | MaxDD {opt["Max_Drawdown"]}%')
        print(f'    Improvement: Sharpe {opt["Sharpe"]-curr["Sharpe"]:+.2f}')
        print(f'    Swaps needed: {len(r["to_remove"])} replacements'
              + (f' + {len(r["new_slots"])} new slots' if r['new_slots'] else ''))

    print(f'\n  Results saved to: {OUTPUT_DIR}/')
    print('\n  Files:')
    for cap in CAPS_TO_TEST:
        print(f'    replacement_cap{cap}_optimal_set.csv')
        print(f'    replacement_cap{cap}_swap_recommendations.csv')
        print(f'    replacement_cap{cap}_backtest_comparison.csv')
    print('    replacement_comparison_chart.png')
    print('    ticker_individual_scores.csv')
    print('\n' + '='*65)


if __name__ == '__main__':
    main()
