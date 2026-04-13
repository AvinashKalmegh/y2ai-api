"""
dm_crossover_hitrate.py
========================
ARGUS Research -- PureSim Universe Hit Rate Analysis
Run cadence: First Monday of each month

PURPOSE:
    Calculates per-ticker DM crossover hit rates across the full Supabase
    universe using a trailing 3-year lookback window.

    Writes results to STAGING for review.
    After review with Vikram, promote to production manually.

    A hit = max return >= 10% within 90 days of a DM crossover above 70.

STAGING SPREADSHEET:
    Sheet ID:  1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA
    Tab:       PureSim_Universe

PRODUCTION SPREADSHEET (promote after review):
    Sheet ID:  1YYwgMvY7I4_i0RD2OH0lM878OkK2WumdEz19UfFcUZ0
    Tab:       PureSim_Universe

TAB COLUMN ORDER (matches staging):
    Col 1:  Ticker
    Col 2:  Data_Points
    Col 3:  Signals
    Col 4:  Hits
    Col 5:  Hit_Rate
    Col 6:  Avg_Return
    Col 7:  Best_Return
    Col 8:  Worst_Return
    Col 9:  First_Date
    Col 10: Last_Date
    Col 11: Status
    Col 12: Change
    Col 13: Run_Date
    Col 14: Lookback

STATUS VALUES:
    QUALIFY -- hit rate >= 65% AND signals >= 10
    THIN    -- hit rate >= 65% BUT signals < 10 (statistically unreliable)
    WATCH   -- hit rate 40-65% (monitor, do not enter)
    BELOW   -- hit rate < 40% (signal does not work on this ticker)

WORKFLOW:
    1. Run this script  -> results land in STAGING
    2. Review staging with Vikram -- identify changes, flags, promotions
    3. python dm_crossover_hitrate.py --production  -> promote to production
    4. Document decisions in chronicle

USAGE:
    python dm_crossover_hitrate.py              # First Monday only, staging
    python dm_crossover_hitrate.py --force      # Any day, staging
    python dm_crossover_hitrate.py --production # Write to production instead
"""

import os
import logging
from datetime import date, timedelta

import gspread
from google.oauth2.service_account import Credentials
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s'
)
logger = logging.getLogger(__name__)


# -- Configuration --------------------------------------------------------------

SUPABASE_URL       = os.getenv('SUPABASE_URL')
SUPABASE_KEY       = os.getenv('SUPABASE_KEY')
SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials.json')

STAGING_SHEET_ID    = '1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA'
PRODUCTION_SHEET_ID = '1YYwgMvY7I4_i0RD2OH0lM878OkK2WumdEz19UfFcUZ0'
UNIVERSE_TAB_NAME   = 'PureSim_Universe'

HIT_RATE_QUALIFY     = 0.65   # >= 65% = QUALIFY or THIN
HIT_RATE_WATCH       = 0.40   # 40-65% = WATCH
MIN_SIGNALS          = 10     # below this = THIN even if hit rate qualifies
FORWARD_DAYS         = 90     # holding period for hit evaluation
HIT_RETURN_THRESHOLD = 0.10   # 10% max return within 90 days = hit
DM_CROSS_THRESHOLD   = 70     # DM must cross above this
LOOKBACK_YEARS       = 3      # trailing window in years

# Column indices (1-based for gspread)
COL_TICKER       = 1
COL_DATA_POINTS  = 2
COL_SIGNALS      = 3
COL_HITS         = 4
COL_HIT_RATE     = 5
COL_AVG_RETURN   = 6
COL_BEST_RETURN  = 7
COL_WORST_RETURN = 8
COL_FIRST_DATE   = 9
COL_LAST_DATE    = 10
COL_STATUS       = 11
COL_CHANGE       = 12
COL_RUN_DATE     = 13
COL_LOOKBACK     = 14


# -- Supabase helpers ----------------------------------------------------------

def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError('SUPABASE_URL and SUPABASE_KEY must be set in .env')
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_tickers(sb: Client, since: str) -> list[str]:
    """Fetch all distinct tickers from dm_history (with pagination) so dropped names are included."""
    logger.info('Fetching ticker list from Supabase dm_history...')
    tickers_set = set()
    offset, page = 0, 10000
    while True:
        result = (sb.table('dm_history')
                  .select('ticker')
                  .gte('date', since)
                  .range(offset, offset + page - 1)
                  .execute())
        for row in result.data:
            if row.get('ticker'):
                tickers_set.add(row['ticker'])
        if len(result.data) < page:
            break
        offset += page
    tickers = sorted(tickers_set)
    logger.info(f'  Found {len(tickers)} distinct tickers')
    return tickers


def fetch_dm_history(sb: Client, ticker: str, since: str) -> list[dict]:
    """Return DM rows for ticker from `since` onward, sorted ascending."""
    result = (
        sb.table('dm_history')
          .select('date, dm_smoothed')
          .eq('ticker', ticker)
          .gte('date', since)
          .order('date', desc=False)
          .execute()
    )
    return result.data


def fetch_price_history(sb: Client, ticker: str, since: str) -> dict[str, float]:
    """Return {date_str: close_price} for ticker from `since` onward."""
    result = (
        sb.table('price_history')
          .select('date, close')
          .eq('ticker', ticker)
          .gte('date', since)
          .execute()
    )
    return {row['date']: float(row['close']) for row in result.data if row.get('close')}


# -- Hit rate calculation -------------------------------------------------------

def classify_status(hit_rate: float, n_signals: int) -> str:
    """Assign status based on hit rate and signal count."""
    if hit_rate >= HIT_RATE_QUALIFY:
        return 'QUALIFY' if n_signals >= MIN_SIGNALS else 'THIN'
    elif hit_rate >= HIT_RATE_WATCH:
        return 'WATCH'
    else:
        return 'BELOW'


def calculate_hit_rate(dm_rows: list[dict], price_map: dict[str, float]) -> dict | None:
    """
    Detect DM crossings above DM_CROSS_THRESHOLD within the lookback window.
    For each crossing, measure the MAX return within FORWARD_DAYS.
    A hit = max return >= HIT_RETURN_THRESHOLD.

    Returns stats dict or None if insufficient data.
    """
    if len(dm_rows) < 2:
        return None

    data_points  = len(dm_rows)
    first_date   = dm_rows[0]['date']
    last_date    = dm_rows[-1]['date']
    prev_dm      = float(dm_rows[0].get('dm_smoothed') or 0)
    signals      = []
    sorted_dates = sorted(price_map.keys())

    for i in range(1, len(dm_rows)):
        curr_dm   = float(dm_rows[i].get('dm_smoothed') or 0)
        curr_date = dm_rows[i]['date']

        # Detect upward crossing only
        if prev_dm < DM_CROSS_THRESHOLD <= curr_dm:
            entry_price = price_map.get(curr_date)
            if not entry_price:
                prev_dm = curr_dm
                continue

            entry_dt = date.fromisoformat(curr_date)
            exit_dt  = entry_dt + timedelta(days=FORWARD_DAYS)

            # Find all prices within the 90-day window after entry
            window_prices = [
                price_map[d] for d in sorted_dates
                if curr_date < d <= exit_dt.isoformat()
            ]

            if window_prices and entry_price > 0:
                max_return = max((p - entry_price) / entry_price for p in window_prices)
                signals.append({
                    'date':       curr_date,
                    'max_return': max_return,
                    'hit':        max_return >= HIT_RETURN_THRESHOLD,
                    'evaluated':  True
                })
            elif exit_dt > date.today():
                # Less than 90 days old -- pending
                signals.append({
                    'date':       curr_date,
                    'max_return': None,
                    'hit':        None,
                    'evaluated':  False
                })

        prev_dm = curr_dm

    evaluated = [s for s in signals if s['evaluated'] and s['hit'] is not None]
    if not evaluated:
        return None

    hits         = sum(1 for s in evaluated if s['hit'])
    hit_rate     = hits / len(evaluated)
    returns      = [s['max_return'] for s in evaluated]
    avg_return   = sum(returns) / len(returns)
    best_return  = max(returns)
    worst_return = min(returns)
    n_signals    = len(evaluated)

    return {
        'ticker':       None,
        'data_points':  data_points,
        'signals':      n_signals,
        'hits':         hits,
        'hit_rate':     hit_rate,
        'avg_return':   avg_return,
        'best_return':  best_return,
        'worst_return': worst_return,
        'first_date':   first_date,
        'last_date':    last_date,
        'status':       classify_status(hit_rate, n_signals),
    }


# -- Google Sheets helpers -----------------------------------------------------

def get_sheet(sheet_id: str, tab_name: str):
    scope  = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds  = Credentials.from_service_account_file(SHEETS_CREDENTIALS, scopes=scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id).worksheet(tab_name)


def load_existing_statuses(ws) -> dict[str, str]:
    """Return {ticker: prior_status} for Change column comparison."""
    all_rows = ws.get_all_values()
    statuses = {}
    for row in all_rows[1:]:
        ticker = row[COL_TICKER  - 1].strip().upper() if len(row) >= COL_TICKER  else ''
        status = row[COL_STATUS  - 1].strip()         if len(row) >= COL_STATUS  else ''
        if ticker:
            statuses[ticker] = status
    return statuses


def write_all_results(ws, results: list[dict], prior_statuses: dict[str, str],
                      run_date: str, lookback_label: str) -> dict:
    """
    Overwrite the tab entirely with all results, sorted by hit_rate descending.
    Calculates Change column vs prior run.
    """
    headers = [
        'Ticker', 'Data_Points', 'Signals', 'Hits', 'Hit_Rate',
        'Avg_Return', 'Best_Return', 'Worst_Return',
        'First_Date', 'Last_Date', 'Status', 'Change', 'Run_Date', 'Lookback'
    ]

    results.sort(key=lambda r: r['hit_rate'], reverse=True)

    rows = [headers]
    counts = {'QUALIFY': 0, 'THIN': 0, 'WATCH': 0, 'BELOW': 0}

    for r in results:
        ticker     = r['ticker']
        new_status = r['status']
        old_status = prior_statuses.get(ticker, '')

        if not old_status:
            change = 'NEW'
        elif old_status != new_status:
            change = f'{old_status}->{new_status}'
        else:
            change = ''

        rows.append([
            ticker,
            r['data_points'],
            r['signals'],
            r['hits'],
            round(r['hit_rate']     * 100, 1),
            round(r['avg_return']   * 100, 1),
            round(r['best_return']  * 100, 1),
            round(r['worst_return'] * 100, 1),
            r['first_date'],
            r['last_date'],
            new_status,
            change,
            run_date,
            lookback_label
        ])
        counts[new_status] = counts.get(new_status, 0) + 1

    ws.clear()
    ws.update('A1', rows, value_input_option='USER_ENTERED')

    return counts


# -- Main ----------------------------------------------------------------------

def is_first_monday_of_month() -> bool:
    today = date.today()
    return today.weekday() == 0 and today.day <= 7


def run_full_analysis(force: bool = False, use_production: bool = False) -> None:

    if not force and not is_first_monday_of_month():
        logger.info('Today is not the first Monday of the month.')
        logger.info('Use --force to run anyway.')
        return

    since_date     = (date.today() - timedelta(days=LOOKBACK_YEARS * 365)).isoformat()
    lookback_label = f'trailing {LOOKBACK_YEARS}yr'
    run_date       = date.today().isoformat()
    sheet_id       = PRODUCTION_SHEET_ID if use_production else STAGING_SHEET_ID
    destination    = 'PRODUCTION' if use_production else 'STAGING'

    logger.info('=' * 65)
    logger.info('  DM CROSSOVER HIT RATE ANALYSIS')
    logger.info(f'  Run date:    {run_date}')
    logger.info(f'  Lookback:    {lookback_label} (from {since_date})')
    logger.info(f'  Destination: {destination}')
    logger.info(f'  Thresholds:  DM>{DM_CROSS_THRESHOLD} | '
                f'Max return>={HIT_RETURN_THRESHOLD*100:.0f}% within {FORWARD_DAYS}d | '
                f'QUALIFY>={HIT_RATE_QUALIFY*100:.0f}% + >={MIN_SIGNALS} signals')
    logger.info('=' * 65)

    sb = get_supabase()
    logger.info('[OK] Supabase connected')

    ws = get_sheet(sheet_id, UNIVERSE_TAB_NAME)
    logger.info(f'[OK] Sheets connected -- {UNIVERSE_TAB_NAME} ({destination})')

    prior_statuses = load_existing_statuses(ws)
    logger.info(f'  Prior run: {len(prior_statuses)} tickers on record')

    tickers = fetch_all_tickers(sb, since_date)
    total   = len(tickers)

    results   = []
    processed = 0
    skipped   = 0

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            logger.info(f'  Processing {i}/{total}...')

        dm_rows   = fetch_dm_history(sb, ticker, since_date)
        price_map = fetch_price_history(sb, ticker, since_date)

        stats = calculate_hit_rate(dm_rows, price_map)
        if stats is None:
            skipped += 1
            continue

        stats['ticker'] = ticker
        results.append(stats)
        processed += 1

    logger.info(f'  Processed: {processed} | Skipped (no crossings): {skipped}')
    logger.info('')
    logger.info(f'Writing {len(results)} results to {destination}...')

    counts = write_all_results(ws, results, prior_statuses, run_date, lookback_label)

    logger.info('')
    logger.info('=' * 65)
    logger.info('  ANALYSIS COMPLETE')
    logger.info('=' * 65)
    logger.info(f'  Tickers analysed:  {processed}')
    logger.info(f'  QUALIFY:           {counts.get("QUALIFY", 0)}')
    logger.info(f'  THIN:              {counts.get("THIN", 0)}')
    logger.info(f'  WATCH:             {counts.get("WATCH", 0)}')
    logger.info(f'  BELOW:             {counts.get("BELOW", 0)}')
    logger.info('')

    # Flag status changes
    changes = [r for r in results
               if r['ticker'] in prior_statuses
               and prior_statuses[r['ticker']] != r['status']]
    if changes:
        logger.warning(f'  [!]  {len(changes)} status change(s) this run:')
        for r in changes:
            logger.warning(f'     {r["ticker"]:8s}  '
                           f'{prior_statuses[r["ticker"]]} -> {r["status"]}  '
                           f'({r["hit_rate"]*100:.1f}%, {r["signals"]} signals)')

    new_tickers = [r for r in results
                   if r['ticker'] not in prior_statuses
                   and r['status'] in ('QUALIFY', 'THIN')]
    if new_tickers:
        logger.info(f'  [+] {len(new_tickers)} new ticker(s) at QUALIFY/THIN:')
        for r in new_tickers:
            logger.info(f'     {r["ticker"]:8s}  {r["status"]}  '
                        f'({r["hit_rate"]*100:.1f}%, {r["signals"]} signals)')

    logger.info('')
    if destination == 'STAGING':
        logger.info('  NEXT STEP: Review staging tab with Vikram.')
        logger.info('  Then run:  python dm_crossover_hitrate.py --production')
    logger.info(f'  Tab updated: {run_date}')
    logger.info('=' * 65)


# -- Entry point ---------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='DM crossover hit rate analysis -- runs first Monday of each month'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Run regardless of day/date'
    )
    parser.add_argument(
        '--production', action='store_true',
        help='Write to production sheet (use only after staging review)'
    )
    args = parser.parse_args()
    run_full_analysis(force=args.force, use_production=args.production)
