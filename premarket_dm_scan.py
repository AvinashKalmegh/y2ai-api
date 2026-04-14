"""
dm_premarket_scan.py

ARGUS Research -- Pre-Market Intelligence Layer
Version: 2.0 (April 2026)

PURPOSE:
    Pulls pre-market prices from Polygon for all ACTIVE/QUALIFY tickers
    in the PureSim universe plus the active Hollowing Short positions.
    Writes to DM_PreMarket tab in staging spreadsheet.

    This is Step 0 of the morning workflow. Run at 7:30 AM ET on trading days.
    Output is reviewed before running the full morning workflow script.

REPLACES:
    Original March 17, 2026 brief (V4 universe, 28 tickers hardcoded).
    Universe now reads dynamically from PureSim_Universe tab in production.

STAGING SPREADSHEET:
    Sheet ID:  1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA
    Tab:       DM_PreMarket

OUTPUT COLUMNS:
    A -- Scan_Time
    B -- Ticker
    C -- PreMarket_Price
    D -- Prev_Close
    E -- Gap_Pct
    F -- Direction  (UP / DOWN / FLAT -- threshold +/-0.5%)
    G -- DM_EOD
    H -- Source     (PURESIM / HOLLOWING_SHORT)
    I -- Earnings_Flag  (YES / NO -- earnings today or tomorrow)

UNIVERSE:
    Primary  -- ACTIVE or QUALIFY tickers from PureSim_Universe tab (production)
    Overlay  -- Active positions from Shadow_Hollowing tab (production)
    Combined -- deduplicated, labeled by source

EARNINGS FLAG:
    Reads from Earnings_Calendar tab in staging (if present).
    Tab columns: Ticker, Earnings_Date (YYYY-MM-DD)
    If no Earnings_Calendar tab exists, Earnings_Flag defaults to NO for all.

RUN CADENCE:
    7:30 AM ET, Monday-Friday (trading days only)
    Append rows -- do not overwrite. Preserves history.

USAGE:
    python premarket_dm_scan.py              # live run
    python premarket_dm_scan.py --dry-run    # print output, do not write
"""

import os
import logging
from datetime import date, datetime, timedelta

import requests
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

# -- Configuration ------------------------------------------------------------

POLYGON_API_KEY    = os.getenv('POLYGON_API_KEY')
SUPABASE_URL       = os.getenv('SUPABASE_URL')
SUPABASE_KEY       = os.getenv('SUPABASE_KEY')
SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials.json')

PRODUCTION_SHEET_ID = '1YYwgMvY7I4_i0RD2OH0lM878OkK2WumdEz19UfFcUZ0'
STAGING_SHEET_ID    = '1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA'

UNIVERSE_TAB        = 'PureSim_Universe'
HOLLOWING_TAB       = 'Shadow_Hollowing'
PREMARKET_TAB       = 'DM_PreMarket'
EARNINGS_CAL_TAB    = 'Earnings_Calendar'

GAP_THRESHOLD       = 0.5   # +/-0.5% = FLAT

# -- Google Sheets helpers ----------------------------------------------------

def get_sheet(sheet_id: str):
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds  = Credentials.from_service_account_file(SHEETS_CREDENTIALS, scopes=scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id)


def load_puresim_universe(ss) -> list[dict]:
    """
    Read ACTIVE and QUALIFY tickers from PureSim_Universe tab in production.
    Returns list of {ticker, status}.
    Data starts at row 7 (rows 1-6 are headers/metadata).
    """
    try:
        ws   = ss.worksheet(UNIVERSE_TAB)
        data = ws.get_all_values()
        results = []
        for row in data[6:]:   # skip first 6 rows
            ticker = str(row[0]).strip().upper() if len(row) > 0 else ''
            status = str(row[6]).strip().upper() if len(row) > 6 else ''
            if ticker and status in ('ACTIVE', 'QUALIFY'):
                results.append({'ticker': ticker, 'source': 'PURESIM'})
        logger.info(f'PureSim universe: {len(results)} ACTIVE/QUALIFY tickers loaded')
        return results
    except Exception as e:
        logger.error(f'Failed to load PureSim universe: {e}')
        return []


def load_hollowing_short_positions(ss) -> list[dict]:
    """
    Read active tickers from Shadow_Hollowing tab in production.
    Returns list of {ticker, source}.
    Looks for tickers in the holdings section -- rows with a ticker in col A
    and a non-zero share count in col B.
    """
    try:
        ws   = ss.worksheet(HOLLOWING_TAB)
        data = ws.get_all_values()
        results = []
        in_holdings = False
        for row in data:
            # Detect holdings section header
            if row and str(row[0]).strip().upper() == 'TICKER':
                in_holdings = True
                continue
            if not in_holdings:
                continue
            # Stop at trade log section
            if row and str(row[0]).strip().upper() in ('TRADE LOG', 'DATE'):
                break
            ticker = str(row[0]).strip().upper() if len(row) > 0 else ''
            shares = str(row[1]).strip()         if len(row) > 1 else ''
            if ticker and shares and shares not in ('0', '', 'SHARES'):
                results.append({'ticker': ticker, 'source': 'HOLLOWING_SHORT'})
        logger.info(f'Hollowing Short: {len(results)} active positions loaded')
        return results
    except Exception as e:
        logger.warning(f'Could not load Hollowing Short positions: {e}')
        return []


def load_prev_close_from_supabase(sb: Client, tickers: list[str]) -> dict[str, float]:
    """
    Pull yesterday's closing price from Supabase price_history for each ticker.
    Returns {ticker: close_price}.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    five_days_ago = (date.today() - timedelta(days=5)).isoformat()

    result = (
        sb.table('price_history')
          .select('ticker, date, close')
          .in_('ticker', tickers)
          .gte('date', five_days_ago)
          .lte('date', yesterday)
          .order('date', desc=True)
          .execute()
    )

    prev_close = {}
    for row in result.data:
        tk = row['ticker'].upper()
        if tk not in prev_close and row.get('close'):
            prev_close[tk] = float(row['close'])

    logger.info(f'Previous close prices loaded: {len(prev_close)} of {len(tickers)} tickers')
    return prev_close


def load_dm_eod_from_supabase(sb: Client, tickers: list[str]) -> dict[str, float]:
    """
    Pull yesterday's DM score from dm_history for each ticker.
    Returns {ticker: dm_score}.
    """
    five_days_ago = (date.today() - timedelta(days=5)).isoformat()
    yesterday     = (date.today() - timedelta(days=1)).isoformat()

    result = (
        sb.table('dm_history')
          .select('ticker, date, dm_smoothed')
          .in_('ticker', tickers)
          .gte('date', five_days_ago)
          .lte('date', yesterday)
          .order('date', desc=True)
          .execute()
    )

    dm_eod = {}
    for row in result.data:
        tk = row['ticker'].upper()
        if tk not in dm_eod and row.get('dm_smoothed') is not None:
            dm_eod[tk] = round(float(row['dm_smoothed']), 1)

    logger.info(f'DM EOD scores loaded: {len(dm_eod)} of {len(tickers)} tickers')
    return dm_eod


def load_earnings_flags(staging_ss, tickers: list[str]) -> dict[str, str]:
    """
    Read Earnings_Calendar tab in staging.
    Flag tickers with earnings today or tomorrow as YES.
    Tab columns: Ticker (col A), Earnings_Date YYYY-MM-DD (col B).
    Returns {ticker: 'YES' or 'NO'}.
    """
    flags = {t: 'NO' for t in tickers}
    today    = date.today()
    tomorrow = today + timedelta(days=1)

    try:
        ws   = staging_ss.worksheet(EARNINGS_CAL_TAB)
        data = ws.get_all_values()
        for row in data[1:]:   # skip header
            ticker        = str(row[0]).strip().upper() if len(row) > 0 else ''
            earnings_date = str(row[1]).strip()         if len(row) > 1 else ''
            if not ticker or not earnings_date:
                continue
            try:
                ed = date.fromisoformat(earnings_date)
                if ed in (today, tomorrow):
                    flags[ticker] = 'YES'
            except ValueError:
                continue
        flagged = sum(1 for v in flags.values() if v == 'YES')
        logger.info(f'Earnings flags: {flagged} ticker(s) reporting today or tomorrow')
    except Exception:
        logger.info('No Earnings_Calendar tab found -- earnings flags default to NO')

    return flags


# -- Polygon pre-market fetch -------------------------------------------------

def fetch_premarket_prices(tickers: list[str]) -> dict[str, dict]:
    """
    Pull pre-market snapshot prices from Polygon for all tickers.
    Uses the same snapshot endpoint as the existing intraday pipeline.
    Returns {ticker: {price, gap_pct}}.
    Price = prevDay close + todaysChange (correct for pre-market).
    Gap = todaysChangePerc (direct from Polygon).
    """
    if not POLYGON_API_KEY:
        raise EnvironmentError('POLYGON_API_KEY must be set in .env')

    joined = ','.join(tickers)
    url = (
        f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers'
        f'?tickers={joined}&apiKey={POLYGON_API_KEY}'
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f'Polygon fetch failed: {e}')
        return {}

    prices = {}
    for t in data.get('tickers', []):
        ticker = t.get('ticker', '').upper()
        prev_day = t.get('prevDay') or {}
        prev_close = prev_day.get('c')
        todays_change = t.get('todaysChange')
        todays_change_pct = t.get('todaysChangePerc')

        if prev_close and todays_change is not None:
            pm_price = float(prev_close) + float(todays_change)
            gap_pct = float(todays_change_pct) if todays_change_pct is not None else 0.0
            prices[ticker] = {'price': pm_price, 'gap_pct': gap_pct}

    logger.info(f'Polygon pre-market prices: {len(prices)} of {len(tickers)} tickers returned')
    return prices


# -- Main ---------------------------------------------------------------------

def run_premarket_scan(dry_run: bool = False) -> None:

    scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')
    logger.info('=' * 65)
    logger.info('  PRE-MARKET DM SCAN')
    logger.info(f'  {scan_time}')
    logger.info(f'  Dry run: {dry_run}')
    logger.info('=' * 65)

    # -- Connect ---------------------------------------------------------------
    sb         = create_client(SUPABASE_URL, SUPABASE_KEY)
    prod_ss    = get_sheet(PRODUCTION_SHEET_ID)
    staging_ss = get_sheet(STAGING_SHEET_ID)
    logger.info('[OK] Connected to Supabase and Google Sheets')

    # -- Load universe ---------------------------------------------------------
    puresim   = load_puresim_universe(prod_ss)
    hollowing = load_hollowing_short_positions(prod_ss)

    # Deduplicate -- PureSim takes priority, Hollowing Short fills in
    seen     = set()
    combined = []
    for entry in puresim + hollowing:
        if entry['ticker'] not in seen:
            seen.add(entry['ticker'])
            combined.append(entry)

    tickers = [e['ticker'] for e in combined]
    logger.info(f'Combined universe: {len(tickers)} unique tickers')

    # -- Load supporting data --------------------------------------------------
    prev_close     = load_prev_close_from_supabase(sb, tickers)
    dm_eod         = load_dm_eod_from_supabase(sb, tickers)
    earnings_flags = load_earnings_flags(staging_ss, tickers)

    # -- Fetch pre-market prices -----------------------------------------------
    premarket = fetch_premarket_prices(tickers)

    # -- Build output rows -----------------------------------------------------
    rows = []
    for entry in combined:
        ticker = entry['ticker']
        source = entry['source']

        snap      = premarket.get(ticker)
        pc        = prev_close.get(ticker)
        dm        = dm_eod.get(ticker, '')
        earn_flag = earnings_flags.get(ticker, 'NO')

        if snap:
            pm_price = round(snap['price'], 2)
            gap_pct  = round(snap['gap_pct'], 2)
            if gap_pct > GAP_THRESHOLD:
                direction = 'UP'
            elif gap_pct < -GAP_THRESHOLD:
                direction = 'DOWN'
            else:
                direction = 'FLAT'
        elif pc:
            # No pre-market data -- use prev close, gap = 0
            pm_price  = pc
            gap_pct   = 0.0
            direction = 'FLAT'
        else:
            pm_price  = ''
            gap_pct   = ''
            direction = 'FLAT'

        rows.append([
            scan_time,
            ticker,
            pm_price,
            pc or '',
            gap_pct,
            direction,
            dm,
            source,
            earn_flag
        ])

    # Sort by abs(Gap_Pct) descending -- biggest movers first
    rows.sort(
        key=lambda r: abs(r[4]) if isinstance(r[4], (int, float)) else 0,
        reverse=True
    )

    # -- Log summary -----------------------------------------------------------
    up_count   = sum(1 for r in rows if r[5] == 'UP')
    down_count = sum(1 for r in rows if r[5] == 'DOWN')
    flat_count = sum(1 for r in rows if r[5] == 'FLAT')
    earn_count = sum(1 for r in rows if r[8] == 'YES')

    logger.info('')
    logger.info(f'  UP:       {up_count}')
    logger.info(f'  DOWN:     {down_count}')
    logger.info(f'  FLAT:     {flat_count}')
    logger.info(f'  Earnings: {earn_count} ticker(s) reporting today/tomorrow')
    logger.info('')

    # Log significant movers
    significant = [r for r in rows if isinstance(r[4], float) and abs(r[4]) >= 2.0]
    if significant:
        logger.info('  Significant movers (+/-2%+):')
        for r in significant:
            logger.info(f'    {r[1]:8s}  {r[5]:4s}  {r[4]:+.2f}%  DM:{r[6]}  Earnings:{r[8]}')
    logger.info('')

    # -- Write to staging ------------------------------------------------------
    if dry_run:
        logger.info('  DRY RUN -- not writing to sheet')
        for r in rows[:20]:
            logger.info(f'  {r}')
    else:
        try:
            ws = staging_ss.worksheet(PREMARKET_TAB)
        except gspread.WorksheetNotFound:
            ws = staging_ss.add_worksheet(title=PREMARKET_TAB, rows=5000, cols=9)
            ws.append_row([
                'Scan_Time', 'Ticker', 'PreMarket_Price', 'Prev_Close',
                'Gap_Pct', 'Direction', 'DM_EOD', 'Source', 'Earnings_Flag'
            ])
            logger.info(f'  Created new tab: {PREMARKET_TAB}')

        ws.append_rows(rows, value_input_option='USER_ENTERED')
        logger.info(f'  [OK] {len(rows)} rows written to {PREMARKET_TAB}')

    # -- Run analysis ------------------------------------------------------------
    analyze_premarket(rows, scan_time, staging_ss, dry_run)

    logger.info('=' * 65)
    logger.info('  PRE-MARKET SCAN COMPLETE')
    logger.info('=' * 65)


# -- Analysis ------------------------------------------------------------------
# Columns: 0=scan_time, 1=ticker, 2=pm_price, 3=prev_close,
#           4=gap_pct, 5=direction, 6=dm_eod, 7=source, 8=earnings_flag

GAP_SIG = 2.0    # +/-2% = significant mover
DM_HIGH = 70     # DM above this = strong institutional signal
DM_LOW  = 30     # DM below this = weak / distribution


def _gap(r):
    return r[4] if isinstance(r[4], (int, float)) else 0.0

def _dm(r):
    return float(r[6]) if r[6] != '' else 0.0


ANALYSIS_TAB = 'DM_PreMarket_Analysis'


def analyze_premarket(rows, scan_time, staging_ss=None, dry_run=False):
    if not rows:
        return

    # -- Categorize -------------------------------------------------------
    up_count   = sum(1 for r in rows if r[5] == 'UP')
    down_count = sum(1 for r in rows if r[5] == 'DOWN')
    flat_count = sum(1 for r in rows if r[5] == 'FLAT')

    significant = sorted([r for r in rows if abs(_gap(r)) >= GAP_SIG],
                         key=lambda r: abs(_gap(r)), reverse=True)

    earnings = [r for r in rows if r[8] == 'YES']

    high_dm = sorted([r for r in rows if _dm(r) >= 85],
                     key=lambda r: _dm(r), reverse=True)[:5]
    low_dm  = sorted([r for r in rows if 0 < _dm(r) <= 15],
                     key=lambda r: _dm(r))[:5]

    # -- Label each significant mover ------------------------------------
    def _label(r):
        g, dm = _gap(r), _dm(r)
        if g <= -GAP_SIG and dm >= DM_HIGH: return 'WATCH EXIT'
        if g >= GAP_SIG  and dm <= DM_LOW:  return 'TRAP?'
        if g >= GAP_SIG  and dm >= DM_HIGH: return 'CONFIRMING'
        return ''

    # -- Print to terminal ------------------------------------------------
    print()
    print('=' * 60)
    print(f'  PRE-MARKET ANALYSIS -- {scan_time}')
    print(f'  Universe: {len(rows)} tickers')
    print('=' * 60)
    print()
    print(f'  MARKET TONE:  UP {up_count}  |  DOWN {down_count}  |  FLAT {flat_count}')
    print()

    if significant:
        print(f'  -- SIGNIFICANT MOVERS (+/-{GAP_SIG}%+) --------------------------')
        for r in significant:
            g, dm = _gap(r), _dm(r)
            earn = ' [EARNINGS]' if r[8] == 'YES' else ''
            lbl = _label(r)
            lbl_str = f' <- {lbl}' if lbl else ''
            print(f'  {r[1]:<8s}  {g:+7.2f}%  DM:{dm:6.1f}  {r[5]}{earn}{lbl_str}')
        print()
    else:
        print(f'  No significant movers (+/-{GAP_SIG}%+) this scan.')
        print()

    if earnings:
        print('  -- EARNINGS TODAY / TOMORROW ------------------------------')
        for r in earnings:
            print(f'  {r[1]:<8s}  {_gap(r):+7.2f}%  DM:{_dm(r):.1f}  {r[5]}')
        print()

    print('  -- NOTABLE DM READINGS --------------------------------------')
    if high_dm:
        print('  High DM (>=85):')
        for r in high_dm:
            print(f'    {r[1]:<8s}  DM:{_dm(r):5.1f}  Gap:{_gap(r):+.2f}%')
    if low_dm:
        print('  Low DM (<=15):')
        for r in low_dm:
            print(f'    {r[1]:<8s}  DM:{_dm(r):5.1f}  Gap:{_gap(r):+.2f}%')

    print()
    print('  Legend: CONFIRMING = gap + high DM aligned')
    print('          WATCH EXIT = big drop on high-DM position')
    print('          TRAP?      = big gap up on weak DM')
    print('=' * 60)
    print()

    # -- Write to Google Sheets -------------------------------------------
    if dry_run or staging_ss is None:
        return

    # Build sheet rows: Scan_Time, Ticker, Gap_Pct, DM_EOD, Direction, Signal, Earnings
    sheet_rows = []
    for r in significant:
        sheet_rows.append([
            scan_time,
            r[1],                                          # Ticker
            round(_gap(r), 2),                             # Gap_Pct
            _dm(r),                                        # DM_EOD
            r[5],                                          # Direction
            _label(r),                                     # Signal
            r[8],                                          # Earnings_Flag
        ])

    # Add high DM notable names (not already in significant)
    sig_tickers = {r[1] for r in significant}
    for r in high_dm:
        if r[1] not in sig_tickers:
            sheet_rows.append([
                scan_time, r[1], round(_gap(r), 2), _dm(r),
                r[5], 'HIGH DM', r[8],
            ])

    # Add low DM notable names
    for r in low_dm:
        if r[1] not in sig_tickers:
            sheet_rows.append([
                scan_time, r[1], round(_gap(r), 2), _dm(r),
                r[5], 'LOW DM', r[8],
            ])

    # Add summary row
    sheet_rows.append([
        scan_time, 'SUMMARY', '',
        f'UP:{up_count} DOWN:{down_count} FLAT:{flat_count}',
        f'{len(significant)} significant', f'{len(rows)} universe',
        f'{len(earnings)} earnings',
    ])

    if not sheet_rows:
        return

    try:
        ws = staging_ss.worksheet(ANALYSIS_TAB)
    except gspread.WorksheetNotFound:
        ws = staging_ss.add_worksheet(title=ANALYSIS_TAB, rows=5000, cols=7)
        ws.append_row([
            'Scan_Time', 'Ticker', 'Gap_Pct', 'DM_EOD',
            'Direction', 'Signal', 'Earnings_Flag'
        ])
        logger.info(f'  Created new tab: {ANALYSIS_TAB}')

    ws.append_rows(sheet_rows, value_input_option='USER_ENTERED')
    logger.info(f'  [OK] {len(sheet_rows)} analysis rows written to {ANALYSIS_TAB}')


# -- Entry point ---------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Pre-market DM scan -- run at 7:30 AM ET on trading days'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print output without writing to sheet'
    )
    args = parser.parse_args()
    run_premarket_scan(dry_run=args.dry_run)
