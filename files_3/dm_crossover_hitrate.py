"""
DM CROSSOVER HIT RATE — PureSim Universe Evaluation
=====================================================
Runs DM crossover hit rate analysis on the current PureSim Universe
with configurable lookback window.

Usage:
    python dm_crossover_hitrate.py                  # full history (default)
    python dm_crossover_hitrate.py --years 3        # trailing 3 years only
    python dm_crossover_hitrate.py --years 5        # trailing 5 years only

Compares results against the full-history baseline to show:
    - Which tickers STAY qualified
    - Which tickers DROP to WATCH or below
    - Which NEW tickers cross the threshold under the shorter window

Output:
    dm_crossover_hitrate_results.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from dotenv import load_dotenv

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase pandas numpy python-dotenv")
    sys.exit(1)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -- CONFIG -------------------------------------------------------------------

DM_TABLE    = 'dm_history'
DATE_COL    = 'date'
TICKER_COL  = 'ticker'
CLOSE_COL   = 'close'
DM_COL      = 'dm_smoothed'

ENTRY_DM      = 70       # DM must cross ABOVE this
HIT_RETURN    = 0.10     # 10%+ return = hit
FORWARD_DAYS  = 90       # calendar days to measure return
MIN_SIGNALS   = 10       # minimum crossings for qualification
QUALIFY_HR    = 40       # hit rate % to qualify
WATCH_HR      = 30       # hit rate % for WATCH status

# Current PureSim Universe reference file
PURESIM_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'v4_universe_candidates.csv')

# Google Sheets output
GS_CREDENTIALS    = 'credentials.json'
GS_SPREADSHEET    = 'copy-dm-history 2024-current'
GS_TAB            = 'PureSim_Universe'

GS_HEADERS = [
    'Ticker', 'Data_Points', 'Signals', 'Hits', 'Hit_Rate',
    'Avg_Return', 'Best_Return', 'Worst_Return', 'First_Date', 'Last_Date',
    'Status', 'Change', 'Run_Date', 'Lookback'
]


# -- SUPABASE -----------------------------------------------------------------

def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


# -- DATA LOADER --------------------------------------------------------------

def load_dm_data(client, start_date, end_date):
    """Load DM history year-by-year to avoid Supabase timeout."""
    start_year = start_date.year
    end_year   = end_date.year

    print(f"Loading DM data from {start_date} to {end_date} (year by year)...")
    rows = []
    page = 10000

    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end   = f"{year}-12-31"
        offset  = 0

        while True:
            r = (client.table(DM_TABLE)
                 .select(f"{DATE_COL},{TICKER_COL},{CLOSE_COL},{DM_COL}")
                 .gte(DATE_COL, y_start)
                 .lte(DATE_COL, y_end)
                 .order(DATE_COL)
                 .range(offset, offset + page - 1)
                 .execute())
            rows.extend(r.data)
            if len(r.data) < page:
                break
            offset += page

        print(f"  {year}: {len(rows):,} total rows")

    print(f"  Total: {len(rows):,} rows")

    df = pd.DataFrame(rows)
    df.rename(columns={
        DATE_COL:   'date',
        TICKER_COL: 'ticker',
        CLOSE_COL:  'close',
        DM_COL:     'dm',
    }, inplace=True)

    df['date']  = pd.to_datetime(df['date']).dt.date
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['dm']    = pd.to_numeric(df['dm'],    errors='coerce')
    df = df.dropna(subset=['date', 'ticker', 'close', 'dm'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"  Tickers: {df['ticker'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df


# -- CROSSOVER HIT RATE -------------------------------------------------------

def compute_crossover_hitrate(df):
    """Run DM crossover hit rate on all tickers in df."""
    results = []
    tickers = sorted(df['ticker'].unique())
    total   = len(tickers)

    print(f"\nRunning crossover analysis on {total:,} tickers...")

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')

        tdf = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        n   = len(tdf)

        if n < 20:
            continue

        signals    = 0
        hits       = 0
        total_ret  = 0.0
        best_ret   = -999.0
        worst_ret  = 999.0

        for j in range(1, n):
            prev_dm = tdf.loc[j-1, 'dm']
            curr_dm = tdf.loc[j, 'dm']

            if not (prev_dm < ENTRY_DM and curr_dm >= ENTRY_DM):
                continue

            entry_price = tdf.loc[j, 'close']
            entry_date  = tdf.loc[j, 'date']

            if entry_price <= 0:
                continue

            signals += 1

            cutoff_date = entry_date + timedelta(days=FORWARD_DAYS)
            forward = tdf[(tdf['date'] > entry_date) & (tdf['date'] <= cutoff_date)]

            if forward.empty:
                signals -= 1
                continue

            max_price = forward['close'].max()
            max_ret   = (max_price - entry_price) / entry_price

            total_ret += max_ret
            if max_ret >= HIT_RETURN:
                hits += 1
            if max_ret > best_ret:
                best_ret = max_ret
            if max_ret < worst_ret:
                worst_ret = max_ret

        if signals == 0:
            continue

        hit_rate   = hits / signals * 100
        avg_return = total_ret / signals * 100

        results.append({
            'ticker':      ticker,
            'data_points': n,
            'signals':     signals,
            'hits':        hits,
            'hit_rate':    round(hit_rate, 1),
            'avg_return':  round(avg_return, 1),
            'best_return': round(best_ret * 100, 1) if best_ret > -999 else 0.0,
            'worst_return':round(worst_ret * 100, 1) if worst_ret < 999 else 0.0,
            'first_date':  str(tdf['date'].iloc[0]),
            'last_date':   str(tdf['date'].iloc[-1]),
        })

    print(f"\n  Done. {len(results):,} tickers with signals.")
    return pd.DataFrame(results).sort_values('hit_rate', ascending=False).reset_index(drop=True)


# -- CLASSIFICATION ------------------------------------------------------------

def classify(row):
    if row['signals'] < MIN_SIGNALS:
        return 'THIN'
    if row['hit_rate'] >= QUALIFY_HR:
        return 'QUALIFY'
    if row['hit_rate'] >= WATCH_HR:
        return 'WATCH'
    return 'BELOW'


# -- LOAD CURRENT PURESIM UNIVERSE --------------------------------------------

def load_current_universe():
    """Load current PureSim Universe tickers from v4_universe_candidates.csv."""
    if not os.path.exists(PURESIM_FILE):
        print(f"  WARNING: {PURESIM_FILE} not found. Comparison disabled.")
        return set()
    df = pd.read_csv(PURESIM_FILE)
    tickers = set(df['ticker'].str.strip().str.upper())
    print(f"  Current PureSim Universe: {len(tickers)} tickers")
    return tickers


# -- GOOGLE SHEETS PUSH --------------------------------------------------------

def push_to_sheets(results_df, lookback_label):
    """Push qualified results to the PureSim_Universe tab in Google Sheets."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        logger.warning("gspread/oauth2client not installed. Skipping GS push.")
        return False

    if not os.path.exists(GS_CREDENTIALS):
        logger.warning(f"{GS_CREDENTIALS} not found. Skipping GS push.")
        return False

    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GS_CREDENTIALS, scope)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open(GS_SPREADSHEET)

        try:
            sheet = spreadsheet.worksheet(GS_TAB)
        except gspread.WorksheetNotFound:
            sheet = spreadsheet.add_worksheet(
                title=GS_TAB, rows=700, cols=len(GS_HEADERS)
            )

        run_date = datetime.now().strftime('%Y-%m-%d')

        rows = [GS_HEADERS]
        for _, r in results_df.iterrows():
            rows.append([
                r.get('ticker', ''),
                int(r.get('data_points', 0)),
                int(r.get('signals', 0)),
                int(r.get('hits', 0)),
                round(r.get('hit_rate', 0), 1),
                round(r.get('avg_return', 0), 1),
                round(r.get('best_return', 0), 1),
                round(r.get('worst_return', 0), 1),
                str(r.get('first_date', '')),
                str(r.get('last_date', '')),
                r.get('status', ''),
                r.get('change', ''),
                run_date,
                lookback_label,
            ])

        sheet.clear()
        sheet.update(range_name=f"A1:{chr(64+len(GS_HEADERS))}{len(rows)}", values=rows)
        logger.info(f"  Pushed {len(rows)-1} rows to {GS_SPREADSHEET} -> {GS_TAB}")
        return True

    except Exception as e:
        logger.error(f"  GS push failed: {e}")
        return False


# -- MAIN ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DM Crossover Hit Rate Analysis')
    parser.add_argument('--years', type=int, default=None,
                        help='Trailing lookback years (default: full history)')
    parser.add_argument('--push', action='store_true',
                        help='Push results to Google Sheets PureSim_Universe tab')
    args = parser.parse_args()

    lookback_label = f"trailing {args.years}yr" if args.years else "full history"

    print('=' * 75)
    print('DM CROSSOVER HIT RATE -- PURESIM UNIVERSE EVALUATION')
    print(f'Lookback: {lookback_label}')
    print(f'Entry: DM crosses above {ENTRY_DM}')
    print(f'Hit: {int(HIT_RETURN*100)}%+ max return within {FORWARD_DAYS} calendar days')
    print(f'Qualify: {QUALIFY_HR}%+ hit rate with {MIN_SIGNALS}+ signals')
    print('=' * 75)
    print()

    # Determine date range
    end_date = date.today()
    if args.years:
        start_date = date(end_date.year - args.years, end_date.month, end_date.day)
    else:
        start_date = date(2016, 1, 1)

    # Load data
    client = get_client()
    df = load_dm_data(client, start_date, end_date)

    # Load current universe for comparison
    current_universe = load_current_universe()

    # Run crossover analysis
    results = compute_crossover_hitrate(df)

    if results.empty:
        print("No results. Exiting.")
        return

    # Classify
    results['status'] = results.apply(classify, axis=1)

    # Compare against current universe
    if current_universe:
        results['in_current'] = results['ticker'].isin(current_universe)
        results['change'] = results.apply(lambda r: _get_change(r, current_universe), axis=1)
    else:
        results['in_current'] = False
        results['change'] = ''

    # -- PRINT RESULTS ---------------------------------------------------------

    qualified = results[results['status'] == 'QUALIFY']
    watch     = results[results['status'] == 'WATCH']
    below     = results[results['status'] == 'BELOW']
    thin      = results[results['status'] == 'THIN']

    print(f'\n{"=" * 75}')
    print(f'QUALIFIED TICKERS -- {len(qualified)} ({lookback_label})')
    print(f'{"=" * 75}')
    print(f"{'Ticker':<8}| {'Sigs':>5} | {'Hits':>4} | {'HR%':>5} | {'AvgRet':>7} | {'Status':<8} | Change")
    print('-' * 75)
    for _, r in qualified.iterrows():
        print(f"{r['ticker']:<8}| {r['signals']:>5} | {r['hits']:>4} | "
              f"{r['hit_rate']:>4.1f}% | {r['avg_return']:>6.1f}% | "
              f"{r['status']:<8} | {r.get('change','')}")

    # -- IMPACT ON CURRENT UNIVERSE --------------------------------------------

    if current_universe:
        print(f'\n{"=" * 75}')
        print(f'IMPACT ON CURRENT PURESIM UNIVERSE ({len(current_universe)} tickers)')
        print(f'{"=" * 75}')

        # Tickers that stay qualified
        stay = results[(results['in_current']) & (results['status'] == 'QUALIFY')]
        # Tickers that drop
        drop = results[(results['in_current']) & (results['status'] != 'QUALIFY')]
        # Current universe tickers not found in results (no signals)
        found_tickers = set(results['ticker'])
        missing = current_universe - found_tickers
        # New qualifiers not in current universe
        new = results[(~results['in_current']) & (results['status'] == 'QUALIFY')]

        print(f"\n  STAY QUALIFIED:    {len(stay)} tickers")
        if len(stay) > 0:
            for _, r in stay.head(20).iterrows():
                print(f"    {r['ticker']:<8} HR={r['hit_rate']:>5.1f}%  signals={r['signals']}")
            if len(stay) > 20:
                print(f"    ... and {len(stay)-20} more")

        print(f"\n  DROP (no longer qualify): {len(drop)} tickers")
        if len(drop) > 0:
            print(f"  {'Ticker':<8}| {'HR%':>5} | {'Sigs':>5} | {'Status':<8}")
            print(f"  {'-'*40}")
            for _, r in drop.iterrows():
                print(f"  {r['ticker']:<8}| {r['hit_rate']:>4.1f}% | {r['signals']:>5} | {r['status']}")

        if missing:
            print(f"\n  NO SIGNALS IN WINDOW: {len(missing)} tickers")
            print(f"    {', '.join(sorted(missing))}")

        print(f"\n  NEW QUALIFIERS (not in current universe): {len(new)} tickers")
        if len(new) > 0:
            for _, r in new.head(20).iterrows():
                print(f"    {r['ticker']:<8} HR={r['hit_rate']:>5.1f}%  signals={r['signals']}")
            if len(new) > 20:
                print(f"    ... and {len(new)-20} more")

        # Summary counts
        print(f'\n{"=" * 75}')
        print(f'SUMMARY')
        print(f'{"=" * 75}')
        print(f"  Current universe:     {len(current_universe)} tickers")
        print(f"  Stay qualified:       {len(stay)}")
        print(f"  Drop to WATCH/BELOW:  {len(drop)}")
        print(f"  No signals in window: {len(missing)}")
        print(f"  New qualifiers:       {len(new)}")
        print(f"  Net universe size:    {len(stay) + len(new)} tickers")

    print(f'\n{"=" * 75}')

    # -- SAVE ------------------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = f"_{args.years}yr" if args.years else "_full"
    outfile = os.path.join(script_dir, f'dm_crossover_hitrate_results{suffix}.csv')
    results.to_csv(outfile, index=False)
    print(f"\nSaved: {outfile}")

    # -- PUSH TO GOOGLE SHEETS -------------------------------------------------

    if args.push:
        print("\nPushing to Google Sheets...")
        push_to_sheets(results, lookback_label)
    else:
        print("\n(Use --push to write results to Google Sheets)")


def _get_change(row, current_universe):
    in_current = row['ticker'] in current_universe
    status     = row['status']

    if in_current and status == 'QUALIFY':
        return 'STAY'
    elif in_current and status == 'WATCH':
        return 'DROP->WATCH'
    elif in_current and status == 'BELOW':
        return 'DROP->BELOW'
    elif in_current and status == 'THIN':
        return 'DROP->THIN'
    elif not in_current and status == 'QUALIFY':
        return 'NEW'
    return ''


if __name__ == '__main__':
    main()
