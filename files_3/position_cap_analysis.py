"""
position_cap_analysis.py
========================
ARGUS Research -- PureSim Strategy 5
Position Cap Sensitivity Analysis

PURPOSE:
    Determines the optimal maximum position limit for PureSim Strategy 5
    by running a backtest simulation at multiple cap levels against the
    DM_2024_2026 history data.

    Replaces the arbitrary 20-position limit with an empirical answer.

FOUR TESTS:
    Test 1 -- Cap Sensitivity     : Sharpe/return/drawdown at caps 10/15/20/25/30
    Test 2 -- Marginal Contribution: Does position #16 add alpha or dilute?
    Test 3 -- Correlation Penalty : Are the current 20 names too correlated?
    Test 4 -- Missed Signal Cost  : What did skipped signals return on average?

DATA SOURCE:
    Google Sheets -- Staging SS
    Sheet ID: 1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA
    Tab: DM_2024_2026
    Columns: Date, Ticker, DM, EMA5, Price, PriceMA20, DM_Change, ...

    Also reads PureSim_Universe tab for QUALIFY tickers.

REQUIREMENTS:
    pip install pandas numpy gspread oauth2client scipy matplotlib seaborn

CREDENTIALS:
    Needs a Google Service Account JSON key with read access to the
    Staging spreadsheet. Set path in CREDS_PATH below, or use the
    existing credentials from the GAS project.

ENTRY RULES (PureSim Strategy 5):
    DM >= 65
    EMA5 (5d MA) rising vs prior day
    Price > PriceMA20 (20d price MA)

EXIT RULES:
    DM_MA20 < 50  (20-day moving average of DM falls below 50)
    OR loss > 15%

RESULTS:
    Saves CSV and PNG charts to ./results/ folder.
    Prints summary table to console.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from itertools import combinations

warnings.filterwarnings('ignore')

# -- CONFIGURATION --------------------------------------------------------------

# Google Sheets credentials -- update path to your service account JSON
CREDS_PATH      = 'credentials.json'

# Staging Spreadsheet ID
STAGING_SS_ID   = '1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA'

# Tab names
DM_HISTORY_TAB  = 'DM_2024_2026'
UNIVERSE_TAB    = 'PureSim_Universe'

# Strategy parameters
DM_ENTRY        = 65       # DM threshold for entry
DM_EXIT_MA      = 50       # DM 20-day MA threshold for exit
LOSS_LIMIT      = -0.15    # -15% loss limit
POSITION_CAPS   = [10, 15, 20, 25, 30]   # Test all five caps

# Backtest date range (use full available history)
START_DATE      = '2023-01-02'
END_DATE        = '2026-03-15'    # Use validated HMS backtest end date

# Risk-free rate for Sharpe (4.3% annual)
RISK_FREE_DAILY = 0.043 / 252

# Output folder
OUTPUT_DIR      = './results'


# -- DATA LOADING ---------------------------------------------------------------

def load_dm_history():
    """Load DM_2024_2026 from Google Sheets into a DataFrame."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope  = ['https://spreadsheets.google.com/feeds',
                  'https://www.googleapis.com/auth/drive']
        creds  = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
        client = gspread.authorize(creds)
        ss     = client.open_by_key(STAGING_SS_ID)
        ws     = ss.worksheet(DM_HISTORY_TAB)
        data   = ws.get_all_records()
        df     = pd.DataFrame(data)
        print(f'  DM_2024_2026 loaded: {len(df):,} rows')
        return df
    except Exception as e:
        print(f'  ERROR loading DM history: {e}')
        print('  Falling back to CSV if available...')
        if os.path.exists('DM_2024_2026.csv'):
            df = pd.read_csv('DM_2024_2026.csv')
            print(f'  Loaded from CSV: {len(df):,} rows')
            return df
        sys.exit('No data source available. Export DM_2024_2026 to CSV first.')


def load_universe():
    """Load QUALIFY tickers from PureSim_Universe tab."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope  = ['https://spreadsheets.google.com/feeds',
                  'https://www.googleapis.com/auth/drive']
        creds  = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
        client = gspread.authorize(creds)
        ss     = client.open_by_key(STAGING_SS_ID)
        ws     = ss.worksheet(UNIVERSE_TAB)
        data   = ws.get_all_records()
        df     = pd.DataFrame(data)
        qualify = df[df['Status'].str.upper().isin(['QUALIFY', 'ACTIVE'])]['Ticker'].tolist()
        print(f'  Universe loaded: {len(qualify)} QUALIFY tickers')
        return qualify
    except Exception as e:
        print(f'  WARNING: Could not load universe tab: {e}')
        print('  Will use all tickers in DM history.')
        return None


def prepare_data(df, universe=None):
    """Clean and prepare DM history for backtesting."""

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Find the date and ticker columns (handle variations)
    date_col   = next((c for c in df.columns if 'date' in c.lower()), None)
    ticker_col = next((c for c in df.columns if 'ticker' in c.lower()), None)
    dm_col     = next((c for c in df.columns if c.upper() == 'DM'), None)
    ema5_col   = next((c for c in df.columns if 'ema5' in c.lower() or
                       '5d' in c.lower() or 'ma5' in c.lower()), None)
    price_col  = next((c for c in df.columns if c.lower() == 'price' or
                       c.lower() == 'close'), None)
    pma20_col  = next((c for c in df.columns if 'pricema20' in c.lower() or
                       'price_ma20' in c.lower() or 'prma20' in c.lower()), None)

    if not all([date_col, ticker_col, dm_col, price_col]):
        print('Available columns:', df.columns.tolist())
        sys.exit('ERROR: Cannot identify required columns. Check column names.')

    # Rename to standard names
    rename = {date_col: 'Date', ticker_col: 'Ticker',
              dm_col: 'DM', price_col: 'Price'}
    if ema5_col:   rename[ema5_col]  = 'EMA5'
    if pma20_col:  rename[pma20_col] = 'PriceMA20'
    df = df.rename(columns=rename)

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['DM']    = pd.to_numeric(df['DM'],    errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    if 'EMA5'      in df.columns: df['EMA5']      = pd.to_numeric(df['EMA5'],      errors='coerce')
    if 'PriceMA20' in df.columns: df['PriceMA20'] = pd.to_numeric(df['PriceMA20'], errors='coerce')

    # Filter date range
    df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]

    # Filter to QUALIFY universe if provided
    if universe:
        df = df[df['Ticker'].isin(universe)]

    # Sort
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Calculate DM 20-day rolling average per ticker (exit threshold)
    df['DM_MA20'] = df.groupby('Ticker')['DM'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    # Calculate EMA5 prior day (for rising check) if EMA5 exists
    if 'EMA5' in df.columns:
        df['EMA5_prev'] = df.groupby('Ticker')['EMA5'].shift(1)

    # Calculate prior day price for return calculation
    df['Price_prev'] = df.groupby('Ticker')['Price'].shift(1)

    print(f'  Data prepared: {len(df):,} rows | '
          f'{df["Ticker"].nunique()} tickers | '
          f'{df["Date"].min().date()} to {df["Date"].max().date()}')

    return df


# -- ENTRY / EXIT SIGNAL GENERATORS --------------------------------------------

def check_entry(row):
    """
    Returns True if a ticker meets all three PureSim entry conditions.
    Condition 1: DM >= 65
    Condition 2: EMA5 rising (EMA5 > EMA5_prev)
    Condition 3: Price > PriceMA20
    """
    if pd.isna(row['DM']) or row['DM'] < DM_ENTRY:
        return False
    if 'EMA5' in row and 'EMA5_prev' in row:
        if pd.notna(row['EMA5']) and pd.notna(row['EMA5_prev']):
            if row['EMA5'] <= row['EMA5_prev']:
                return False
    if 'PriceMA20' in row:
        if pd.notna(row['PriceMA20']) and pd.notna(row['Price']):
            if row['Price'] <= row['PriceMA20']:
                return False
    return True


def check_exit(row, entry_price):
    """
    Returns True if exit conditions are met.
    Exit 1: DM_MA20 < 50
    Exit 2: Loss > 15%
    """
    if pd.notna(row.get('DM_MA20')) and row['DM_MA20'] < DM_EXIT_MA:
        return True
    if pd.notna(row['Price']) and entry_price > 0:
        pnl = (row['Price'] - entry_price) / entry_price
        if pnl < LOSS_LIMIT:
            return True
    return False


# -- BACKTEST ENGINE ------------------------------------------------------------

def run_backtest(df, max_positions):
    """
    Run PureSim Strategy 5 backtest with a given position cap.

    Returns:
        trades_df   : DataFrame of all completed trades
        daily_nav   : Series of daily portfolio returns
        skipped_df  : DataFrame of signals that fired but were skipped (book full)
    """

    trading_dates = sorted(df['Date'].unique())
    # Dict of open positions: {ticker: entry_price}
    open_positions = {}
    # List of completed trades
    trades = []
    # List of skipped entry signals
    skipped = []
    # Daily portfolio value (equal weight, track P&L)
    daily_returns = []

    for date in trading_dates:
        day_data = df[df['Date'] == date].set_index('Ticker')

        # -- 1. Check exits on open positions ------------------------------
        to_exit = []
        for ticker, entry_price in open_positions.items():
            if ticker not in day_data.index:
                continue
            row = day_data.loc[ticker]
            if check_exit(row, entry_price):
                exit_price = row['Price']
                pnl        = (exit_price - entry_price) / entry_price
                trades.append({
                    'Ticker':      ticker,
                    'Entry_Price': entry_price,
                    'Exit_Price':  exit_price,
                    'PnL':         pnl,
                    'Exit_Date':   date,
                    'Entry_Order': None   # filled below
                })
                to_exit.append(ticker)

        for t in to_exit:
            del open_positions[t]

        # -- 2. Calculate daily portfolio return ---------------------------
        if open_positions:
            pos_returns = []
            for ticker, entry_price in open_positions.items():
                if ticker in day_data.index and pd.notna(day_data.loc[ticker, 'Price']):
                    curr  = day_data.loc[ticker, 'Price']
                    prev  = day_data.loc[ticker, 'Price_prev'] \
                            if pd.notna(day_data.loc[ticker, 'Price_prev']) \
                            else entry_price
                    if prev > 0:
                        pos_returns.append((curr - prev) / prev)
            if pos_returns:
                daily_returns.append({'Date': date, 'Return': np.mean(pos_returns)})
        else:
            daily_returns.append({'Date': date, 'Return': 0.0})

        # -- 3. Check entries -----------------------------------------------
        candidates = []
        for ticker, row in day_data.iterrows():
            if ticker in open_positions:
                continue
            if check_entry(row):
                candidates.append((ticker, row['DM']))

        # Sort by DM descending -- take highest conviction first
        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, dm in candidates:
            if len(open_positions) < max_positions:
                entry_price = day_data.loc[ticker, 'Price']
                if pd.notna(entry_price) and entry_price > 0:
                    open_positions[ticker] = entry_price
            else:
                # Book full -- log as skipped signal
                row = day_data.loc[ticker]
                skipped.append({
                    'Date':        date,
                    'Ticker':      ticker,
                    'DM':          row['DM'],
                    'Entry_Price': row['Price']
                })

    # Close any remaining open positions at last price
    last_date = trading_dates[-1]
    last_data = df[df['Date'] == last_date].set_index('Ticker')
    for ticker, entry_price in open_positions.items():
        if ticker in last_data.index:
            exit_price = last_data.loc[ticker, 'Price']
            pnl        = (exit_price - entry_price) / entry_price
            trades.append({
                'Ticker':      ticker,
                'Entry_Price': entry_price,
                'Exit_Price':  exit_price,
                'PnL':         pnl,
                'Exit_Date':   last_date,
                'Entry_Order': None
            })

    trades_df  = pd.DataFrame(trades)   if trades  else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)  if skipped else pd.DataFrame()
    daily_df   = pd.DataFrame(daily_returns)

    return trades_df, daily_df, skipped_df


# -- PERFORMANCE METRICS --------------------------------------------------------

def calc_metrics(trades_df, daily_df, cap):
    """Calculate Sharpe, total return, max drawdown, win rate."""

    if trades_df.empty or daily_df.empty:
        return {'Cap': cap, 'Sharpe': 0, 'Total_Return': 0,
                'Max_Drawdown': 0, 'Win_Rate': 0, 'Trades': 0,
                'Avg_Return': 0, 'Avg_Hold_Days': 0}

    # Sharpe ratio
    returns   = daily_df['Return'].fillna(0)
    excess    = returns - RISK_FREE_DAILY
    sharpe    = (excess.mean() / excess.std() * np.sqrt(252)) \
                if excess.std() > 0 else 0

    # Total return (cumulative)
    cum_return = (1 + returns).prod() - 1

    # Max drawdown
    cum_curve  = (1 + returns).cumprod()
    rolling_max = cum_curve.cummax()
    drawdown    = (cum_curve - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Win rate
    win_rate = (trades_df['PnL'] > 0).mean() if len(trades_df) > 0 else 0

    # Average return per trade
    avg_return = trades_df['PnL'].mean() if len(trades_df) > 0 else 0

    return {
        'Cap':          cap,
        'Sharpe':       round(sharpe, 2),
        'Total_Return': round(cum_return * 100, 1),
        'Max_Drawdown': round(max_dd * 100, 1),
        'Win_Rate':     round(win_rate * 100, 1),
        'Trades':       len(trades_df),
        'Avg_Return':   round(avg_return * 100, 1),
    }


# -- TEST 1: CAP SENSITIVITY ----------------------------------------------------

def test1_cap_sensitivity(df):
    """Run backtest at each cap level. Compare Sharpe, return, drawdown."""

    print('\n' + '='*65)
    print('  TEST 1 -- POSITION CAP SENSITIVITY')
    print('='*65)

    results  = []
    all_trades = {}
    all_daily  = {}

    for cap in POSITION_CAPS:
        print(f'\n  Running cap={cap}...')
        trades_df, daily_df, skipped_df = run_backtest(df, cap)
        metrics = calc_metrics(trades_df, daily_df, cap)
        metrics['Skipped_Signals'] = len(skipped_df)
        results.append(metrics)
        all_trades[cap] = trades_df
        all_daily[cap]  = daily_df
        print(f'    Trades: {metrics["Trades"]} | '
              f'Sharpe: {metrics["Sharpe"]} | '
              f'Return: {metrics["Total_Return"]}% | '
              f'MaxDD: {metrics["Max_Drawdown"]}% | '
              f'WinRate: {metrics["Win_Rate"]}% | '
              f'Skipped: {metrics["Skipped_Signals"]}')

    results_df = pd.DataFrame(results)

    print('\n  SUMMARY:')
    print(results_df.to_string(index=False))

    # Find optimal cap by Sharpe
    best = results_df.loc[results_df['Sharpe'].idxmax()]
    print(f'\n  [OK] OPTIMAL CAP BY SHARPE: {int(best["Cap"])} positions')
    print(f'    Sharpe {best["Sharpe"]} | Return {best["Total_Return"]}% | '
          f'MaxDD {best["Max_Drawdown"]}%')

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(f'{OUTPUT_DIR}/test1_cap_sensitivity.csv', index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PureSim Strategy 5 -- Position Cap Sensitivity', fontsize=14)

    axes[0,0].bar(results_df['Cap'], results_df['Sharpe'], color='steelblue')
    axes[0,0].set_title('Sharpe Ratio by Cap')
    axes[0,0].set_xlabel('Position Cap')
    axes[0,0].axhline(y=results_df['Sharpe'].max(), color='red', linestyle='--', alpha=0.5)

    axes[0,1].bar(results_df['Cap'], results_df['Total_Return'], color='green')
    axes[0,1].set_title('Total Return % by Cap')
    axes[0,1].set_xlabel('Position Cap')

    axes[1,0].bar(results_df['Cap'], results_df['Max_Drawdown'].abs(), color='red')
    axes[1,0].set_title('Max Drawdown % by Cap (abs)')
    axes[1,0].set_xlabel('Position Cap')

    axes[1,1].bar(results_df['Cap'], results_df['Win_Rate'], color='orange')
    axes[1,1].set_title('Win Rate % by Cap')
    axes[1,1].set_xlabel('Position Cap')
    axes[1,1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/test1_cap_sensitivity.png', dpi=150)
    plt.close()
    print(f'\n  Saved: {OUTPUT_DIR}/test1_cap_sensitivity.csv')
    print(f'  Saved: {OUTPUT_DIR}/test1_cap_sensitivity.png')

    return results_df, all_trades, all_daily


# -- TEST 2: MARGINAL CONTRIBUTION ---------------------------------------------

def test2_marginal_contribution(df):
    """
    Does position #16 add alpha or dilute?
    Run backtest at cap=30, log each trade's entry order number,
    then compare average return by entry order.
    """

    print('\n' + '='*65)
    print('  TEST 2 -- MARGINAL CONTRIBUTION BY ENTRY ORDER')
    print('='*65)

    # Run at cap=30 to see as many positions as possible
    trading_dates = sorted(df['Date'].unique())
    open_positions = {}
    trades = []
    entry_counter = 0  # global counter across all dates

    for date in trading_dates:
        day_data = df[df['Date'] == date].set_index('Ticker')

        # Exits
        to_exit = []
        for ticker, (entry_price, entry_order) in open_positions.items():
            if ticker not in day_data.index:
                continue
            if check_exit(day_data.loc[ticker], entry_price):
                exit_price = day_data.loc[ticker, 'Price']
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    'Ticker':      ticker,
                    'PnL':         pnl,
                    'Entry_Order': entry_order,
                    'Exit_Date':   date
                })
                to_exit.append(ticker)

        for t in to_exit:
            del open_positions[t]

        # Entries
        candidates = [(t, r['DM']) for t, r in day_data.iterrows()
                      if t not in open_positions and check_entry(r)]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for ticker, dm in candidates:
            if len(open_positions) < 30:
                entry_price = day_data.loc[ticker, 'Price']
                if pd.notna(entry_price) and entry_price > 0:
                    entry_counter += 1
                    open_positions[ticker] = (entry_price, entry_counter)

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print('  No trades generated.')
        return

    # Bucket by entry order
    trades_df['Order_Bucket'] = pd.cut(
        trades_df['Entry_Order'],
        bins=[0, 5, 10, 15, 20, 25, 30, 999],
        labels=['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '30+']
    )

    bucket_stats = trades_df.groupby('Order_Bucket', observed=True)['PnL'].agg(
        ['mean', 'count', 'std']
    ).reset_index()
    bucket_stats.columns = ['Entry_Order_Bucket', 'Avg_Return', 'Count', 'Std']
    bucket_stats['Avg_Return_Pct'] = (bucket_stats['Avg_Return'] * 100).round(1)

    print('\n  Average return by entry order bucket:')
    print(bucket_stats[['Entry_Order_Bucket', 'Avg_Return_Pct', 'Count']].to_string(index=False))

    # Find where returns start declining
    returns_by_bucket = bucket_stats['Avg_Return_Pct'].values
    if len(returns_by_bucket) > 1:
        peak_bucket = bucket_stats.loc[bucket_stats['Avg_Return_Pct'].idxmax(),
                                        'Entry_Order_Bucket']
        print(f'\n  [OK] Peak performance at entry order: {peak_bucket}')
        print('    Positions beyond this bucket are likely diluting returns.')

    bucket_stats.to_csv(f'{OUTPUT_DIR}/test2_marginal_contribution.csv', index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bucket_stats['Entry_Order_Bucket'], bucket_stats['Avg_Return_Pct'],
            color='steelblue')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.title('PureSim -- Average Return by Position Entry Order\n'
              '(Does adding more positions add or dilute alpha?)')
    plt.xlabel('Entry Order Bucket (1st signal = 1)')
    plt.ylabel('Average Return %')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/test2_marginal_contribution.png', dpi=150)
    plt.close()
    print(f'\n  Saved: {OUTPUT_DIR}/test2_marginal_contribution.csv')
    print(f'  Saved: {OUTPUT_DIR}/test2_marginal_contribution.png')

    return bucket_stats


# -- TEST 3: CORRELATION PENALTY -----------------------------------------------

def test3_correlation_penalty(df, current_positions=None):
    """
    Are the current 20 PureSim names too correlated?
    High correlation = diversification benefit limited = smaller book better.

    current_positions: list of tickers in current book (optional).
    If not provided, uses the 20 most recent entry signals.
    """

    print('\n' + '='*65)
    print('  TEST 3 -- CORRELATION PENALTY')
    print('='*65)

    # Default: current PureSim holdings as of latest backtest
    if current_positions is None:
        current_positions = [
            'MRVL', 'INTC', 'AMD', 'WDC', 'KLAC', 'DELL', 'GEV', 'GSAT',
            'NBIS', 'ARM', 'ALB', 'TER', 'NFLX', 'AMAT', 'HAL', 'DOW',
            'TWLO', 'DOCN', 'INSM', 'RUN'
        ]
        print(f'  Using current 20 PureSim positions.')

    # Build price matrix
    price_pivot = df[df['Ticker'].isin(current_positions)].pivot_table(
        index='Date', columns='Ticker', values='Price'
    )
    # Daily returns
    returns_pivot = price_pivot.pct_change().dropna()

    if returns_pivot.empty or returns_pivot.shape[1] < 2:
        print('  Insufficient data for correlation analysis.')
        return

    corr_matrix = returns_pivot.corr()

    # Average pairwise correlation (excluding diagonal)
    mask   = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr_matrix.values[mask].mean()

    print(f'\n  Average pairwise correlation: {avg_corr:.3f}')

    if avg_corr > 0.7:
        print('  [!] HIGH CORRELATION (>0.7) -- names move together.')
        print('    Diversification benefit is limited.')
        print('    A smaller, higher-conviction book likely outperforms.')
        recommendation = 'REDUCE cap -- high correlation'
    elif avg_corr > 0.5:
        print('  MODERATE CORRELATION (0.5–0.7) -- some diversification benefit.')
        print('    Current cap may be near optimal.')
        recommendation = 'MAINTAIN cap -- moderate correlation'
    else:
        print('  LOW CORRELATION (<0.5) -- genuine diversification benefit.')
        print('    Larger book adds real risk reduction.')
        recommendation = 'CONSIDER HIGHER cap -- low correlation'

    print(f'\n  Recommendation: {recommendation}')

    # Top 10 most correlated pairs
    pairs = []
    tickers = corr_matrix.columns.tolist()
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            pairs.append({'Ticker1': t1, 'Ticker2': t2,
                          'Correlation': corr_matrix.loc[t1, t2]})
    pairs_df = pd.DataFrame(pairs).sort_values('Correlation', ascending=False)
    print('\n  Top 10 most correlated pairs:')
    print(pairs_df.head(10).to_string(index=False))

    # Save heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1,
                xticklabels=corr_matrix.columns,
                yticklabels=corr_matrix.columns)
    plt.title(f'PureSim Position Correlation Matrix\n'
              f'Avg pairwise: {avg_corr:.3f} -- {recommendation}')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/test3_correlation_matrix.png', dpi=150)
    plt.close()

    pairs_df.to_csv(f'{OUTPUT_DIR}/test3_correlation_pairs.csv', index=False)
    print(f'\n  Saved: {OUTPUT_DIR}/test3_correlation_matrix.png')
    print(f'  Saved: {OUTPUT_DIR}/test3_correlation_pairs.csv')

    return avg_corr, recommendation


# -- TEST 4: MISSED SIGNAL COST -------------------------------------------------

def test4_missed_signal_cost(df):
    """
    What did skipped signals return on average?
    Run backtest at cap=20, capture all skipped signals,
    then calculate their 30/60/90-day forward returns.

    If skipped signals > held signals in avg return, the cap is too tight.
    """

    print('\n' + '='*65)
    print('  TEST 4 -- MISSED SIGNAL COST')
    print('='*65)

    # Run at cap=20 to get realistic skipped signal pool
    _, _, skipped_df = run_backtest(df, max_positions=20)

    if skipped_df.empty:
        print('  No skipped signals found at cap=20. Book was never full.')
        return

    print(f'  Skipped signals captured: {len(skipped_df)}')

    # Calculate forward returns for skipped signals
    forward_returns = []

    for _, row in skipped_df.iterrows():
        ticker      = row['Ticker']
        entry_date  = row['Date']
        entry_price = row['Entry_Price']

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        ticker_data = df[df['Ticker'] == ticker].set_index('Date').sort_index()

        for horizon_days, label in [(30, '30d'), (60, '60d'), (90, '90d')]:
            target_date = entry_date + timedelta(days=horizon_days)
            # Find nearest available date
            future_data = ticker_data[ticker_data.index >= target_date]
            if future_data.empty:
                continue
            exit_price = future_data.iloc[0]['Price']
            if pd.notna(exit_price) and exit_price > 0:
                fwd_return = (exit_price - entry_price) / entry_price
                forward_returns.append({
                    'Ticker':    ticker,
                    'Date':      entry_date,
                    'DM':        row['DM'],
                    'Horizon':   label,
                    'Return':    fwd_return
                })

    if not forward_returns:
        print('  Could not calculate forward returns.')
        return

    fwd_df = pd.DataFrame(forward_returns)
    summary = fwd_df.groupby('Horizon')['Return'].agg(
        ['mean', 'median', 'count']
    ).reset_index()
    summary.columns = ['Horizon', 'Avg_Return', 'Median_Return', 'Count']
    summary['Avg_Return_Pct']    = (summary['Avg_Return']    * 100).round(1)
    summary['Median_Return_Pct'] = (summary['Median_Return'] * 100).round(1)

    print('\n  Forward returns of skipped signals:')
    print(summary[['Horizon', 'Avg_Return_Pct', 'Median_Return_Pct', 'Count']].to_string(index=False))

    # Compare to held position avg return at same horizons from Test 1 cap=20
    _, all_daily_20 = run_backtest(df, max_positions=20)[:2]
    # Use 30-day rolling average as proxy for held return
    avg_held_30d = all_daily_20['Return'].mean() * 30 * 252/252 * 100 if hasattr(all_daily_20, 'mean') else 0

    skipped_30d = summary[summary['Horizon'] == '30d']['Avg_Return_Pct'].values
    if len(skipped_30d) > 0:
        print(f'\n  Skipped signal avg 30d return: {skipped_30d[0]:.1f}%')
        if skipped_30d[0] > 5:
            print('  [!] Skipped signals are generating meaningful returns.')
            print('    The cap may be too tight. Consider raising it.')
        else:
            print('  [OK] Skipped signals not generating significant alpha.')
            print('    Current cap is likely appropriate.')

    fwd_df.to_csv(f'{OUTPUT_DIR}/test4_missed_signal_cost.csv', index=False)
    summary.to_csv(f'{OUTPUT_DIR}/test4_missed_signal_summary.csv', index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    x = summary['Horizon']
    plt.bar(x, summary['Avg_Return_Pct'], color='steelblue', label='Avg Return')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.title('PureSim -- Forward Returns of Skipped Signals\n'
              '(Cap=20, signals that fired but could not enter)')
    plt.xlabel('Holding Horizon')
    plt.ylabel('Average Forward Return %')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/test4_missed_signal_cost.png', dpi=150)
    plt.close()

    print(f'\n  Saved: {OUTPUT_DIR}/test4_missed_signal_cost.csv')
    print(f'  Saved: {OUTPUT_DIR}/test4_missed_signal_cost.png')

    return summary


# -- MAIN -----------------------------------------------------------------------

def main():
    print('\n' + '='*65)
    print('  ARGUS RESEARCH -- POSITION CAP SENSITIVITY ANALYSIS')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('='*65)

    # Load data
    print('\nLoading data...')
    raw_df   = load_dm_history()
    universe = load_universe()
    df       = prepare_data(raw_df, universe)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all four tests
    t1_results, all_trades, all_daily = test1_cap_sensitivity(df)
    test2_marginal_contribution(df)
    test3_correlation_penalty(df)
    test4_missed_signal_cost(df)

    # Final recommendation
    print('\n' + '='*65)
    print('  FINAL RECOMMENDATION')
    print('='*65)
    best_cap = t1_results.loc[t1_results['Sharpe'].idxmax(), 'Cap']
    best_sharpe = t1_results['Sharpe'].max()
    print(f'\n  Empirical optimal position cap: {int(best_cap)}')
    print(f'  Sharpe at optimal cap: {best_sharpe}')
    print(f'\n  Results saved to: {OUTPUT_DIR}/')
    print('\n  Charts:')
    print('    test1_cap_sensitivity.png    -- Sharpe/return/drawdown by cap')
    print('    test2_marginal_contribution.png -- Alpha by entry order')
    print('    test3_correlation_matrix.png -- Pairwise correlations')
    print('    test4_missed_signal_cost.png -- Skipped signal forward returns')
    print('\n' + '='*65)


if __name__ == '__main__':
    main()
