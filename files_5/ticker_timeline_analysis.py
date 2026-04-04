"""
ARGUS TICKER TIMELINE ANALYSIS
================================
Finds T-0 (HMS elevation), T-CF (CF signal crossing), T-Entry, T-Exit
for 7 target tickers across the full simulation history.

HOW TO RUN:
-----------
1. Export the following Google Sheets as CSV files:
   - HMS_Daily tab from SS: 1YhEhsssJ1kEKm3x-qHGDbysJXx1ej6I9dA-UIv8vgs4
     Save as: HMS_Daily.csv
   - DM_2016_2019 tab from SS: 1DgUqXnk9tXwoBdiR4yoFOClW3eIS7CVjqcW3RBBdFKk
     Save as: DM_2016_2019.csv
   - DM_2020_2023 tab from SS: 1IuiXxcF6z4mL3Mex7Mrh7DFsf_RpNfxzweI_XfXyjKM
     Save as: DM_2020_2023.csv
   - DM_2024_2026 tab from SS: 1GiLsxrgW-nssIhuUGYzn7SHYzcNjP4yg_7p1G2nPNiE
     Save as: DM_2024_2026.csv

2. Place all CSV files in the same folder as this script.

3. Run: python ticker_timeline_analysis.py

4. Output: ticker_timeline_results.csv and ticker_timeline_results.xlsx

REQUIREMENTS: pip install pandas openpyxl
"""

import pandas as pd
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================

# Target tickers with their best Strategy 5 trades
# [ticker, entry_date, entry_price, return_pct, exit_date, hold_days]
TARGET_TRADES = [
    ('DELL', '2024-03-07', 120.50,  14.72, '2024-06-26', 111),
    ('WDC',  '2019-07-16',  38.92,  15.47, '2019-10-16',  92),
    ('PCG',  '2022-09-07',  12.73,  17.75, '2022-11-21',  75),
    ('LRCX', '2019-09-26',  24.28,  10.38, '2019-12-09',  74),
    ('TRGP', '2016-09-23',  47.14,  -3.05, '2016-10-28',  35),
    ('VZ',   '2018-10-25',  56.43,   1.10, '2019-01-09',  76),
    ('ADSK', '2016-09-02',  68.01,   2.10, '2016-11-04',  63),
]

HMS_THRESHOLD   = 0.60  # HMS score considered "elevated" (0-1 scale)
CF_THRESHOLD    = 65    # CF/DM score for signal crossing
LOOKBACK_DAYS   = 400   # How far back to search before entry date

# ============================================================
# LOAD DATA
# ============================================================

def load_csv(filename, label):
    """Load a CSV file with error handling."""
    if not os.path.exists(filename):
        print(f"  ERROR: {filename} not found. Please export from Google Sheets.")
        return None
    print(f"  Loading {filename}...")
    df = pd.read_csv(filename, low_memory=False)
    print(f"  Loaded {len(df):,} rows from {label}")
    return df

def normalize_columns(df, label):
    """Normalize column names to uppercase for consistent lookup."""
    df.columns = [str(c).strip().upper() for c in df.columns]
    print(f"  {label} columns: {df.columns.tolist()[:8]}...")
    return df

def find_date_col(columns):
    for name in ['DATE', 'TRADE_DATE', 'DT', 'TRADING_DATE']:
        if name in columns: return name
    return None

def find_ticker_col(columns):
    for name in ['TICKER', 'SYMBOL', 'TKR']:
        if name in columns: return name
    return None

def find_hms_col(columns):
    for name in ['HMS_SCORE', 'HMS', 'SCORE', 'VALUE', 'HMS_DAILY']:
        if name in columns: return name
    return None

def find_dm_col(columns):
    for name in ['DM_SCORE', 'DM', 'CF_SCORE', 'CF', 'SCORE', 'VALUE']:
        if name in columns: return name
    return None

def find_ma5_col(columns):
    for name in ['DM_MA5', 'MA5', 'EMA5', 'MA_5D', '5D_MA', 'CF_MA5']:
        if name in columns: return name
    return None

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def find_hms_elevation(ticker, entry_date, hms_df):
    """Find the first date HMS crossed above threshold before entry."""
    date_col   = find_date_col(hms_df.columns)
    ticker_col = find_ticker_col(hms_df.columns)
    hms_col    = find_hms_col(hms_df.columns)

    if not all([date_col, ticker_col, hms_col]):
        print(f"    HMS column detection failed. Available: {hms_df.columns.tolist()}")
        return None, None

    # Filter to this ticker
    sub = hms_df[hms_df[ticker_col].str.strip().str.upper() == ticker.upper()].copy()
    if len(sub) == 0:
        print(f"    {ticker}: not found in HMS_Daily")
        return None, None

    # Parse dates and filter to lookback window
    sub[date_col] = pd.to_datetime(sub[date_col], errors='coerce')
    sub = sub.dropna(subset=[date_col])

    lookback = entry_date - pd.Timedelta(days=LOOKBACK_DAYS)
    sub = sub[(sub[date_col] >= lookback) & (sub[date_col] < entry_date)]
    sub = sub.sort_values(date_col)

    # Convert HMS score
    sub[hms_col] = pd.to_numeric(sub[hms_col], errors='coerce')
    elevated = sub[sub[hms_col] >= HMS_THRESHOLD]

    if len(elevated) == 0:
        print(f"    {ticker}: no HMS elevation found above {HMS_THRESHOLD} in lookback window")
        return None, None

    # First elevation date
    first = elevated.iloc[0]
    return first[date_col], round(first[hms_col], 4)


def find_cf_signal(ticker, entry_date, dm_dfs):
    """Find the first date CF crossed threshold with rising momentum before entry."""
    all_rows = []

    for dm_df, label in dm_dfs:
        if dm_df is None:
            continue

        date_col   = find_date_col(dm_df.columns)
        ticker_col = find_ticker_col(dm_df.columns)
        dm_col     = find_dm_col(dm_df.columns)
        ma5_col    = find_ma5_col(dm_df.columns)

        if not all([date_col, ticker_col, dm_col]):
            print(f"    DM column detection failed in {label}. Available: {dm_df.columns.tolist()[:8]}")
            continue

        sub = dm_df[dm_df[ticker_col].str.strip().str.upper() == ticker.upper()].copy()
        if len(sub) == 0:
            continue

        sub[date_col] = pd.to_datetime(sub[date_col], errors='coerce')
        sub = sub.dropna(subset=[date_col])

        lookback = entry_date - pd.Timedelta(days=LOOKBACK_DAYS)
        sub = sub[(sub[date_col] >= lookback) & (sub[date_col] < entry_date)]

        sub[dm_col] = pd.to_numeric(sub[dm_col], errors='coerce')
        if ma5_col:
            sub[ma5_col] = pd.to_numeric(sub[ma5_col], errors='coerce')

        for _, row in sub.iterrows():
            all_rows.append({
                'date': row[date_col],
                'dm':   row[dm_col],
                'ma5':  row[ma5_col] if ma5_col else None,
            })

    if not all_rows:
        print(f"    {ticker}: no DM data found in lookback window")
        return None, None

    df_all = pd.DataFrame(all_rows).sort_values('date').reset_index(drop=True)
    df_all['dm'] = pd.to_numeric(df_all['dm'], errors='coerce')

    # Find first crossing: DM >= threshold
    # If MA5 available, also check MA5 is rising (compare to 3 days prior)
    first_signal_date  = None
    first_signal_score = None

    for i, row in df_all.iterrows():
        dm_ok = (row['dm'] >= CF_THRESHOLD) if pd.notna(row['dm']) else False
        if not dm_ok:
            continue

        # Check MA5 rising if available
        if row['ma5'] is not None and pd.notna(row['ma5']) and i >= 3:
            prior_ma5 = df_all.loc[max(0, i-3), 'ma5']
            ma_rising = (row['ma5'] > prior_ma5) if pd.notna(prior_ma5) else True
        else:
            ma_rising = True

        if ma_rising:
            first_signal_date  = row['date']
            first_signal_score = round(row['dm'], 1)
            break

    return first_signal_date, first_signal_score


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== ARGUS TICKER TIMELINE ANALYSIS ===\n")

    # Load all data sources
    print("Loading data sources...")
    hms_daily  = load_csv('HMS_Daily.csv',      'HMS_Daily')
    hms_1619   = load_csv('HMS_2016_2019.csv',  'HMS_2016_2019')
    hms_2022   = load_csv('HMS_2020_2022.csv',  'HMS_2020_2022')
    dm_1619    = load_csv('DM_2016_2019.csv',   'DM_2016_2019')
    dm_2023    = load_csv('DM_2020_2023.csv',   'DM_2020_2023')
    dm_2426    = load_csv('DM_2024_2026.csv',   'DM_2024_2026')
    dm_dell    = load_csv('DELL_DM_backfill.csv','DELL_DM_backfill')

    # Combine all HMS sources
    hms_parts = [df for df in [hms_1619, hms_2022, hms_daily] if df is not None]
    if not hms_parts:
        print("\nERROR: No HMS data files found. Exiting.")
        sys.exit(1)
    for i, part in enumerate(hms_parts):
        hms_parts[i] = normalize_columns(part, f'HMS_part_{i}')
    hms_df = pd.concat(hms_parts, ignore_index=True)
    print(f"  Combined HMS: {len(hms_df):,} total rows")

    # Normalize columns
    hms_df  = normalize_columns(hms_df,  'HMS_Combined')
    if dm_1619 is not None: dm_1619 = normalize_columns(dm_1619, 'DM_2016_2019')
    if dm_2023 is not None: dm_2023 = normalize_columns(dm_2023, 'DM_2020_2023')
    if dm_2426 is not None: dm_2426 = normalize_columns(dm_2426, 'DM_2024_2026')
    if dm_dell is not None: dm_dell = normalize_columns(dm_dell, 'DELL_DM_backfill')

    dm_sources = [
        (dm_1619, 'DM_2016_2019'),
        (dm_2023, 'DM_2020_2023'),
        (dm_dell, 'DELL_DM_backfill'),
        (dm_2426, 'DM_2024_2026'),
    ]

    print("\nProcessing tickers...\n")
    results = []

    for trade in TARGET_TRADES:
        ticker, entry_str, entry_price, return_pct, exit_str, hold_days = trade
        entry_date = pd.to_datetime(entry_str)
        exit_date  = pd.to_datetime(exit_str)

        print(f"--- {ticker} ---")
        print(f"  Trade: Entry {entry_str} @ ${entry_price} | Exit {exit_str} | {return_pct:+.1f}% | {hold_days}d")

        # Find T-0: First HMS elevation
        t0_date, t0_score = find_hms_elevation(ticker, entry_date, hms_df)

        # Find T-CF: First CF signal
        t_cf_date, t_cf_score = find_cf_signal(ticker, entry_date, dm_sources)

        # Calculate gaps
        t0_to_entry  = (entry_date - t0_date).days  if t0_date  is not None else None
        tcf_to_entry = (entry_date - t_cf_date).days if t_cf_date is not None else None
        hms_led_cf   = (t_cf_date - t0_date).days   if (t0_date is not None and t_cf_date is not None) else None

        print(f"  T-0  (HMS elevation): {t0_date.date() if t0_date is not None else 'NOT FOUND'} | score={t0_score}")
        print(f"  T-CF (CF signal):     {t_cf_date.date() if t_cf_date is not None else 'NOT FOUND'} | score={t_cf_score}")
        print(f"  T-0 to Entry:         {t0_to_entry} days" if t0_to_entry else "  T-0 to Entry: Unknown")
        print(f"  T-CF to Entry:        {tcf_to_entry} days" if tcf_to_entry else "  T-CF to Entry: Unknown")
        print(f"  HMS led CF by:        {hms_led_cf} days" if hms_led_cf else "  HMS led CF by: Unknown")
        print()

        # Build tagline for slide
        tagline = (
            f"Entry {entry_str} at ${entry_price}. "
            f"Return over {hold_days} days: {return_pct:+.1f}%"
        )

        results.append({
            'Ticker':             ticker,
            'T0_Date':            t0_date.date()  if t0_date  is not None else 'NOT FOUND',
            'T0_HMS_Score':       t0_score        if t0_score is not None else '-',
            'T0_to_Entry_Days':   t0_to_entry     if t0_to_entry  is not None else '',
            'TCF_Date':           t_cf_date.date() if t_cf_date is not None else 'NOT FOUND',
            'TCF_CF_Score':       t_cf_score      if t_cf_score is not None else '-',
            'TCF_to_Entry_Days':  tcf_to_entry    if tcf_to_entry is not None else '',
            'Entry_Date':         entry_str,
            'Entry_Price':        entry_price,
            'Exit_Date':          exit_str,
            'Hold_Days':          hold_days,
            'Return_Pct':         return_pct,
            'HMS_Led_CF_By_Days': hms_led_cf      if hms_led_cf is not None else '',
            'Slide_Tagline':      tagline,
        })

    # Save results
    out_df = pd.DataFrame(results)

    # CSV
    out_df.to_csv('ticker_timeline_results.csv', index=False)
    print(f"Saved: ticker_timeline_results.csv")

    # Excel with formatting
    try:
        with pd.ExcelWriter('ticker_timeline_results.xlsx', engine='openpyxl') as writer:
            out_df.to_excel(writer, sheet_name='Ticker_Timeline', index=False)
            ws = writer.sheets['Ticker_Timeline']
            # Auto-width columns
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col) + 4
                ws.column_dimensions[col[0].column_letter].width = min(max_len, 40)
        print(f"Saved: ticker_timeline_results.xlsx")
    except Exception as e:
        print(f"Excel export skipped: {e} (CSV still saved)")

    print("\n=== COMPLETE ===")
    print(f"Processed {len(results)} tickers.")
    print("\nSUMMARY:")
    print(out_df[['Ticker','T0_Date','TCF_Date','Entry_Date','Return_Pct','T0_to_Entry_Days']].to_string(index=False))


if __name__ == '__main__':
    main()
