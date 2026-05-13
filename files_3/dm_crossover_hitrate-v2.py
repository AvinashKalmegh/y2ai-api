"""
DM CROSSOVER HIT RATE — PureSim Universe Evaluation
=====================================================
Runs DM crossover hit rate analysis on the current PureSim Universe
with configurable lookback window.

Usage:
    python dm_crossover_hitrate.py                  # full history (default)
    python dm_crossover_hitrate.py --years 3        # trailing 3 years only
    python dm_crossover_hitrate.py --years 5        # trailing 5 years only

Compares results against last week's full status map (loaded from the
Sheets tab) and writes an explicit Change label for every ticker. The
Change column is never blank — every row carries one of:

    NEW              — ticker not present last Saturday
    STAY-QUALIFY     — unchanged QUALIFY (hit rate >= 40%, signals >= 10)
    STAY-WATCH       — unchanged WATCH (hit rate 30-40%)
    STAY-BELOW       — unchanged BELOW (hit rate < 30%)
    STAY-THIN        — unchanged THIN (signals < 10)
    OLD->NEW         — explicit transition, e.g. WATCH->QUALIFY, QUALIFY->WATCH

This replaces the previous _get_change() that only labeled QUALIFY-related
transitions and left every WATCH/BELOW/THIN-to-WATCH/BELOW/THIN move blank.
For a 597-ticker universe that left ~328 rows unlabeled, hiding genuine
transitions like BELOW->WATCH (climbing) or QUALIFY->WATCH (dropping) from
the Saturday governance review.

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

def load_dm_data(client, start_date, end_date, min_tickers=None):
    """Load DM history year-by-year with stable pagination and retries.

    Fixes BUG-003: prior version ordered only by date, which left ~600
    same-date rows un-tiebroken. Combined with OFFSET pagination this caused
    whole tickers to silently fall through page seams (Saturday Apr 25 run
    produced 387 rows instead of ~596).

    Stable ordering on (date, ticker) eliminates the seam shuffle. Retry
    loop catches transient 57014 / 5xx errors that previously corrupted a
    page silently.

    `min_tickers` is an optional sanity floor — if set and the load returns
    fewer unique tickers, raises so the workflow fails loudly instead of
    pushing a partial result.
    """
    import time

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
            attempt, cur = 0, page
            while True:
                try:
                    r = (client.table(DM_TABLE)
                         .select(f"{DATE_COL},{TICKER_COL},{CLOSE_COL},{DM_COL}")
                         .gte(DATE_COL, y_start)
                         .lte(DATE_COL, y_end)
                         .order(DATE_COL).order(TICKER_COL)  # stable ordering
                         .range(offset, offset + cur - 1)
                         .execute())
                    break
                except Exception as e:
                    msg, etype = str(e), type(e).__name__
                    attempt += 1
                    transient = (
                        "57014" in msg or "502" in msg or "503" in msg or
                        "504" in msg or "timeout" in msg.lower() or
                        "Bad Gateway" in msg or "ConnectionTerminated" in msg or
                        "RemoteProtocolError" in etype or "ReadError" in etype or
                        "ConnectError" in etype or "ConnectTimeout" in etype or
                        "ReadTimeout" in etype
                    )
                    if attempt >= 6 or not transient:
                        raise
                    cur = max(1000, cur // 2)
                    print(f"    [retry {attempt}] {DM_TABLE} {year} "
                          f"page={cur} sleep {2**attempt}s ({etype})")
                    time.sleep(2 ** attempt)

            rows.extend(r.data)
            if len(r.data) < cur:
                break
            offset += cur

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
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    n_tickers = df['ticker'].nunique()
    print(f"  Tickers: {n_tickers:,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    if min_tickers is not None and n_tickers < min_tickers:
        raise RuntimeError(
            f"Partial load detected: {n_tickers} tickers loaded, "
            f"expected at least {min_tickers}. Aborting to avoid producing "
            f"a partial result. Check Supabase pagination / connectivity."
        )

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
    """Load current PureSim Universe tickers from v4_universe_candidates.csv (fallback)."""
    if not os.path.exists(PURESIM_FILE):
        print(f"  WARNING: {PURESIM_FILE} not found. Comparison disabled.")
        return set()
    df = pd.read_csv(PURESIM_FILE)
    tickers = set(df['ticker'].str.strip().str.upper())
    print(f"  Current PureSim Universe (CSV): {len(tickers)} tickers")
    return tickers


def load_current_universe_from_sheets():
    """Read last week's QUALIFY tickers from the live PureSim_Universe tab.

    Kept as a thin wrapper around load_prior_statuses_from_sheets() for
    backwards compatibility with any caller that wants only the QUALIFY set.
    New code should prefer load_prior_statuses_from_sheets() so all
    transitions are visible in the Change column.
    """
    statuses = load_prior_statuses_from_sheets()
    return {t for t, s in statuses.items() if s == 'QUALIFY'}


def load_prior_statuses_from_sheets():
    """Read last week's COMPLETE status mapping from the live PureSim_Universe tab.

    Returns a dict {ticker: status} for every row in last Saturday's tab.
    The Sheets tab is the source of truth — it gets overwritten every
    Saturday by this same script — so loading it gives proper
    week-over-week comparison without a static CSV that drifts.

    Replaces the prior load_current_universe_from_sheets() that only
    returned QUALIFY tickers, leaving WATCH<->BELOW<->THIN transitions
    invisible in the Change column.
    """
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        print("  gspread not installed; prior statuses unavailable.")
        return {}

    if not os.path.exists(GS_CREDENTIALS):
        print(f"  {GS_CREDENTIALS} not found; prior statuses unavailable.")
        return {}

    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GS_CREDENTIALS, scope)
        gc = gspread.authorize(creds)
        sheet = gc.open(GS_SPREADSHEET).worksheet(GS_TAB)
        rows = sheet.get_all_records()
        statuses = {}
        for r in rows:
            ticker = str(r.get('Ticker', '')).strip().upper()
            status = str(r.get('Status', '')).strip().upper()
            if ticker and status:
                statuses[ticker] = status

        # Distribution log so the Saturday review can sanity-check
        # against last week's run summary.
        from collections import Counter
        dist = Counter(statuses.values())
        print(f"  Loaded prior statuses from Sheets tab '{GS_TAB}': "
              f"{len(statuses)} tickers")
        print(f"    Last week: " + " | ".join(
            f"{k}={v}" for k, v in sorted(dist.items())
        ))
        return statuses
    except Exception as e:
        print(f"  Sheets read failed ({type(e).__name__}: {e}); "
              f"prior statuses unavailable.")
        return {}


def load_prior_statuses_from_tsv(path):
    """Read prior status map from a TSV/CSV exported from the Sheet.

    Validation/replay path — lets us feed v2 a known historical snapshot
    instead of reading the live Sheet, without touching credentials.json
    or risking a push.
    """
    if not os.path.exists(path):
        print(f"  Prior TSV not found at {path}; prior statuses unavailable.")
        return {}
    sep = '\t' if path.lower().endswith('.tsv') else ','
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as e:
        print(f"  Failed to read {path} ({type(e).__name__}: {e}).")
        return {}

    if 'Ticker' not in df.columns or 'Status' not in df.columns:
        print(f"  {path} missing required columns 'Ticker'/'Status'; "
              f"got {list(df.columns)}.")
        return {}

    statuses = {}
    for _, r in df.iterrows():
        ticker = str(r.get('Ticker', '')).strip().upper()
        status = str(r.get('Status', '')).strip().upper()
        if ticker and status and status != 'NAN':
            statuses[ticker] = status

    from collections import Counter
    dist = Counter(statuses.values())
    print(f"  Loaded prior statuses from {path}: {len(statuses)} tickers")
    print(f"    Prior:    " + " | ".join(
        f"{k}={v}" for k, v in sorted(dist.items())
    ))
    return statuses


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
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for lookback window, YYYY-MM-DD '
                             '(default: today). Used to reconstruct prior '
                             'weeks for validation.')
    parser.add_argument('--prior-tsv', type=str, default=None,
                        help='Path to TSV/CSV with Ticker+Status columns to '
                             'use as the prior-week comparison (default: read '
                             'live Sheet). Disables --push.')
    args = parser.parse_args()

    if args.push and args.prior_tsv:
        print("ERROR: --push is not allowed with --prior-tsv. The TSV is a "
              "synthetic comparison; pushing under it would corrupt the live "
              "Sheet's week-over-week semantics.")
        sys.exit(1)

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
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        print(f"Using --end-date override: {end_date}")
    else:
        end_date = date.today()
    if args.years:
        start_date = date(end_date.year - args.years, end_date.month, end_date.day)
    else:
        start_date = date(2016, 1, 1)

    # Load data. In production runs (--push) require at least 550 tickers
    # to be loaded — guards against BUG-003-style partial loads where the
    # April 25 Saturday run produced 387 rows instead of ~596.
    client = get_client()
    min_tickers = 550 if args.push else None
    df = load_dm_data(client, start_date, end_date, min_tickers=min_tickers)

    # Load last week's full status map (not just QUALIFY) so the Change
    # column can show every transition, including WATCH<->BELOW<->THIN
    # moves that the prior QUALIFY-only loader hid as blanks.
    if args.prior_tsv:
        prior_statuses = load_prior_statuses_from_tsv(args.prior_tsv)
    else:
        prior_statuses = load_prior_statuses_from_sheets()

    # Guardrail: in production runs (--push), an empty prior-status map
    # means every row's Change column will be NEW — the BUG-001 silent
    # failure mode where a Sheets read error makes week-over-week
    # comparison impossible. Refuse to push under that condition.
    if args.push and not prior_statuses:
        print("ERROR: prior status map is empty — Change column would be "
              "NEW for every row. Refusing to push to Sheets. "
              "Check that the PureSim_Universe tab is reachable and the "
              "credentials.json has access to the spreadsheet.")
        sys.exit(1)

    # Run crossover analysis
    results = compute_crossover_hitrate(df)

    if results.empty:
        print("No results. Exiting.")
        return

    # Classify
    results['status'] = results.apply(classify, axis=1)

    # Compute Change for every row using full prior-status map
    results['change'] = results.apply(
        lambda r: _get_change(r, prior_statuses), axis=1
    )
    # Convenience flag for the impact section below
    prior_qualify = {t for t, s in prior_statuses.items() if s == 'QUALIFY'}
    results['in_current'] = results['ticker'].isin(prior_qualify)

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

    if prior_qualify:
        print(f'\n{"=" * 75}')
        print(f'IMPACT ON CURRENT PURESIM UNIVERSE ({len(prior_qualify)} prior-week QUALIFY)')
        print(f'{"=" * 75}')

        # Tickers that stay qualified
        stay = results[(results['in_current']) & (results['status'] == 'QUALIFY')]
        # Tickers that drop
        drop = results[(results['in_current']) & (results['status'] != 'QUALIFY')]
        # Current universe tickers not found in results (no signals)
        found_tickers = set(results['ticker'])
        missing = prior_qualify - found_tickers
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
        print(f"  Prior-week QUALIFY:   {len(prior_qualify)} tickers")
        print(f"  Stay qualified:       {len(stay)}")
        print(f"  Drop to WATCH/BELOW:  {len(drop)}")
        print(f"  No signals in window: {len(missing)}")
        print(f"  New qualifiers:       {len(new)}")
        print(f"  Net universe size:    {len(stay) + len(new)} tickers")

    # -- CHANGE COLUMN DISTRIBUTION (full transition matrix) ------------------

    print(f'\n{"=" * 75}')
    print(f'CHANGE COLUMN DISTRIBUTION')
    print(f'{"=" * 75}')
    from collections import Counter
    change_dist = Counter(results['change'])
    # Print STAY rows first, then transitions, then NEW
    stay_keys = sorted(k for k in change_dist if k.startswith('STAY-'))
    transition_keys = sorted(k for k in change_dist
                             if '->' in k and not k.startswith('STAY-'))
    other_keys = sorted(k for k in change_dist
                        if k not in stay_keys and k not in transition_keys)
    for k in stay_keys + transition_keys + other_keys:
        print(f"  {k:<24} {change_dist[k]:>4}")
    print(f"  {'TOTAL':<24} {sum(change_dist.values()):>4}")

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


def _get_change(row, prior_statuses):
    """Compute Change column for one row using last week's full status map.

    Every ticker gets a non-blank label. The four output forms are:

        NEW              — ticker not present last Saturday (or empty status)
        STAY-QUALIFY     — unchanged QUALIFY
        STAY-WATCH       — unchanged WATCH
        STAY-BELOW       — unchanged BELOW
        STAY-THIN        — unchanged THIN
        OLD->NEW         — explicit transition, e.g. 'WATCH->QUALIFY'

    `prior_statuses` is a dict {ticker_upper: status_upper}. If empty
    (e.g. the Sheets read failed), every row falls back to NEW so the
    review can still see this week's data, but the operator should treat
    the Change column as untrustworthy that week.
    """
    ticker     = row['ticker']
    new_status = row['status']
    old_status = prior_statuses.get(ticker, None)

    if not old_status:
        return 'NEW'
    if old_status == new_status:
        return f'STAY-{new_status}'
    return f'{old_status}->{new_status}'


if __name__ == '__main__':
    main()
