"""
Per-Ticker Options Put/Call Pipeline (yfinance)
=================================================
Fetches options chain data for 55 priority tickers via yfinance (free).
Aggregates put/call volume across near-term expirations (30 days).
Calculates Per_Call_Ratio, PC_20D_Avg, PC_Z_Score per ticker.
Stores in Supabase options_history table.

DATA SOURCE: Yahoo Finance via yfinance (free, no API key)
LIMITATION: Current-day snapshot only (no deep historical backfill).
            After 20 trading days of daily collection, rolling metrics become valid.
            yfinance can be throttled by Yahoo - use polite delays.

COMPANION: options_pipeline.py (CBOE market-wide P/C ratios, with history)

Usage:
  python options_perticker.py daily       # Fetch today's data for 55 tickers
  python options_perticker.py backfill    # Start collecting (same as daily)
  python options_perticker.py check NVDA  # Check a specific ticker
  python options_perticker.py push        # Push to Google Sheets
  python options_perticker.py test        # Test with 3 tickers

Timing: ~55 tickers at 2s delay = ~2 minutes. Run after 4:30 PM ET.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from supabase import create_client
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
except Exception:
    supabase = None
    logger.warning("Supabase not configured. Data won't be uploaded.")

# ============================================================
# CONFIGURATION
# ============================================================

# Delay between tickers to avoid Yahoo throttling
DELAY_BETWEEN_TICKERS = 2.0  # seconds

# How far out to aggregate expirations (calendar days)
NEAR_TERM_DAYS = 45

# Google Sheets config
SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "Options_PerTicker"

# Priority tickers (55 from dev spec: short portfolio + bypass layers + key names)
PRIORITY_TICKERS = [
    # Short portfolio / Services bypass
    "ACN", "CRM", "ADBE", "SCHW", "LPLA", "RJF", "INFY", "NOW", "ORCL", "SAP",
    "ADP", "PAYX", "IT", "AON", "AJG", "MMC", "SPGI", "MCO", "WIT", "CTSH",
    # Big tech / AI infra
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "TSM", "INTC",
    # Enterprise / Security
    "SNOW", "PLTR", "DDOG", "NET", "CRWD", "ZS", "PANW", "FTNT",
    # Financial
    "GS", "MS", "JPM", "V", "MA", "BLK", "ICE",
    # Infrastructure
    "EQIX", "DLR", "AMT", "CEG", "VST", "ANET", "DELL", "SMCI",
    # Other
    "HDB", "IBM",
]

# Deduplicate while preserving order
PRIORITY_TICKERS = list(dict.fromkeys(PRIORITY_TICKERS))


# ============================================================
# YFINANCE FETCH
# ============================================================
def fetch_ticker_options(ticker_symbol, near_term_days=NEAR_TERM_DAYS):
    """
    Fetch options chain for a ticker, aggregate put/call volume
    across all expirations within near_term_days.
    
    Returns dict with put_volume, call_volume, put_oi, call_oi or None on error.
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        
        if not expirations:
            return None
        
        # Filter to near-term expirations
        cutoff = (datetime.now() + timedelta(days=near_term_days)).strftime('%Y-%m-%d')
        near_exps = [e for e in expirations if e <= cutoff]
        
        if not near_exps:
            # Fall back to first available expiration
            near_exps = [expirations[0]]
        
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        
        for exp in near_exps:
            try:
                chain = ticker.option_chain(exp)
                
                # Volume
                total_call_vol += chain.calls['volume'].fillna(0).sum()
                total_put_vol += chain.puts['volume'].fillna(0).sum()
                
                # Open Interest
                total_call_oi += chain.calls['openInterest'].fillna(0).sum()
                total_put_oi += chain.puts['openInterest'].fillna(0).sum()
                
            except Exception as e:
                logger.debug(f"    {ticker_symbol} exp {exp}: {e}")
                continue
        
        return {
            'put_volume': int(total_put_vol),
            'call_volume': int(total_call_vol),
            'put_oi': int(total_put_oi),
            'call_oi': int(total_call_oi),
            'expirations_used': len(near_exps),
        }
    
    except Exception as e:
        logger.warning(f"  {ticker_symbol}: Error - {e}")
        return None


# ============================================================
# CALCULATIONS
# ============================================================
def calculate_pc_metrics(df):
    """Calculate Put_Call_Ratio, PC_20D_Avg, PC_Z_Score per ticker."""
    result_dfs = []
    
    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker].sort_values('date').copy()
        
        # Put/Call Volume Ratio
        tdf['put_call_ratio'] = np.where(
            tdf['call_volume'] > 0,
            (tdf['put_volume'] / tdf['call_volume']).round(4),
            0
        )
        
        # Put/Call OI Ratio (bonus metric)
        tdf['put_call_oi_ratio'] = np.where(
            tdf['call_oi'] > 0,
            (tdf['put_oi'] / tdf['call_oi']).round(4),
            0
        )
        
        # 20-day rolling average and Z-score
        tdf['pc_20d_avg'] = tdf['put_call_ratio'].rolling(
            window=20, min_periods=10
        ).mean().round(4)
        
        pc_std = tdf['put_call_ratio'].rolling(window=20, min_periods=10).std()
        tdf['pc_z_score'] = np.where(
            pc_std > 0,
            ((tdf['put_call_ratio'] - tdf['pc_20d_avg']) / pc_std).round(4),
            0
        )
        
        result_dfs.append(tdf)
    
    if not result_dfs:
        return pd.DataFrame()
    return pd.concat(result_dfs, ignore_index=True)


# ============================================================
# SUPABASE
# ============================================================
def upload_to_supabase(df):
    """Upload per-ticker options data to options_history table."""
    if supabase is None:
        logger.warning("Supabase not configured. Skipping upload.")
        return 0
    
    records = []
    for _, row in df.iterrows():
        record = {
            'date': str(row.get('date', '')),
            'ticker': str(row.get('ticker', '')),
            'put_volume': int(row['put_volume']) if pd.notna(row.get('put_volume')) else None,
            'call_volume': int(row['call_volume']) if pd.notna(row.get('call_volume')) else None,
            'put_call_ratio': float(row['put_call_ratio']) if pd.notna(row.get('put_call_ratio')) else None,
            'pc_20d_avg': float(row['pc_20d_avg']) if pd.notna(row.get('pc_20d_avg')) else None,
            'pc_z_score': float(row['pc_z_score']) if pd.notna(row.get('pc_z_score')) else None,
        }
        records.append(record)
    
    if not records:
        return 0
    
    BATCH_SIZE = 200
    uploaded = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("options_history") \
                .upsert(batch, on_conflict="date,ticker") \
                .execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error: {e}")
    
    return uploaded


def load_from_supabase(ticker=None):
    """Load options_history from Supabase, optionally filtered by ticker."""
    if supabase is None:
        return pd.DataFrame()
    
    all_rows = []
    offset = 0
    page_size = 1000
    
    while True:
        query = supabase.table("options_history").select("*").order("date", desc=True)
        if ticker:
            query = query.eq("ticker", ticker.upper())
        result = query.range(offset, offset + page_size - 1).execute()
        
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size
    
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ============================================================
# GOOGLE SHEETS
# ============================================================
def push_to_sheets():
    """Push options_history to Google Sheets."""
    import gspread
    from google.oauth2.service_account import Credentials

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    logger.info("Loading options_history from Supabase...")
    df = load_from_supabase()

    if df.empty:
        logger.info("No data to push.")
        return

    logger.info(f"  {len(df)} rows loaded")

    headers = [
        "Date", "Ticker", "Put_Volume", "Call_Volume",
        "Put_Call_Ratio", "PC_20D_Avg", "PC_Z_Score"
    ]

    sheet_rows = []
    for _, row in df.iterrows():
        sheet_rows.append([
            "'" + str(row.get('date', '')),
            row.get('ticker', ''),
            row.get('put_volume', ''),
            row.get('call_volume', ''),
            row.get('put_call_ratio', ''),
            row.get('pc_20d_avg', ''),
            row.get('pc_z_score', ''),
        ])

    for row in sheet_rows:
        for j in range(len(row)):
            if row[j] is None or (isinstance(row[j], float) and np.isnan(row[j])):
                row[j] = ''

    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    client = gspread.authorize(creds)

    try:
        spreadsheet = client.open(SHEETS_SPREADSHEET)
    except Exception:
        logger.error(f"Spreadsheet '{SHEETS_SPREADSHEET}' not found.")
        return

    try:
        old = spreadsheet.worksheet(SHEETS_TAB)
        spreadsheet.del_worksheet(old)
    except Exception:
        pass

    sheet = spreadsheet.add_worksheet(
        title=SHEETS_TAB,
        rows=len(sheet_rows) + 100,
        cols=len(headers)
    )

    sheet.update(range_name='A1', values=[headers], value_input_option='USER_ENTERED')
    sheet.format('1:1', {'textFormat': {'bold': True}})

    BATCH_SIZE = 5000
    total = 0
    for i in range(0, len(sheet_rows), BATCH_SIZE):
        batch = sheet_rows[i:i + BATCH_SIZE]
        sheet.update(range_name=f'A{i+2}', values=batch, value_input_option='USER_ENTERED')
        total += len(batch)
        logger.info(f"  Written {total}/{len(sheet_rows)} rows")
        time.sleep(1)

    logger.info(f"Push complete: {total} rows")


# ============================================================
# COMMANDS
# ============================================================
def run_daily(tickers=None):
    """
    Fetch today's options data for priority tickers via yfinance.
    ~55 tickers at 2s delay = ~2 minutes.
    """
    if tickers is None:
        tickers = PRIORITY_TICKERS
    
    logger.info("=" * 60)
    logger.info("PER-TICKER OPTIONS - DAILY (yfinance)")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info("=" * 60)
    
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    failed = []
    
    for i, ticker in enumerate(tickers):
        logger.info(f"  [{i+1}/{len(tickers)}] {ticker}...")
        
        result = fetch_ticker_options(ticker)
        
        if result:
            rows.append({
                'date': today,
                'ticker': ticker,
                'put_volume': result['put_volume'],
                'call_volume': result['call_volume'],
                'put_oi': result.get('put_oi', 0),
                'call_oi': result.get('call_oi', 0),
            })
            pc = result['put_volume'] / result['call_volume'] if result['call_volume'] > 0 else 0
            logger.info(f"    Put={result['put_volume']:>10,}  Call={result['call_volume']:>10,}  "
                        f"P/C={pc:.3f}  ({result['expirations_used']} exps)")
        else:
            failed.append(ticker)
            logger.warning(f"    No data")
        
        if i < len(tickers) - 1:
            time.sleep(DELAY_BETWEEN_TICKERS)
    
    if not rows:
        logger.info("No data fetched.")
        return
    
    df_today = pd.DataFrame(rows)
    
    # Load existing data for rolling calculations
    existing = load_from_supabase()
    if not existing.empty:
        keep_cols = ['date', 'ticker', 'put_volume', 'call_volume']
        existing = existing[[c for c in keep_cols if c in existing.columns]]
        combined = pd.concat([existing, df_today[keep_cols]], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
    else:
        combined = df_today
    
    # Calculate metrics across full history
    combined = calculate_pc_metrics(combined)
    
    # Extract today's rows with calculated metrics
    today_data = combined[combined['date'] == today].copy()
    
    # Upload
    uploaded = upload_to_supabase(today_data)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY COMPLETE")
    logger.info(f"  Fetched:  {len(rows)}/{len(tickers)} tickers")
    logger.info(f"  Uploaded: {uploaded} rows")
    if failed:
        logger.info(f"  Failed:   {', '.join(failed)}")
    
    # Show tickers with unusual P/C (Z > 1.5 or Z < -1.5)
    if 'pc_z_score' in today_data.columns:
        alerts = today_data[
            (today_data['pc_z_score'] > 1.5) | (today_data['pc_z_score'] < -1.5)
        ].sort_values('pc_z_score', ascending=False)
        
        if len(alerts) > 0:
            logger.info(f"\n  ALERTS (unusual P/C activity):")
            for _, row in alerts.iterrows():
                z = row.get('pc_z_score', 0)
                pc = row.get('put_call_ratio', 0)
                direction = "HIGH PUTS" if z > 0 else "HIGH CALLS"
                logger.info(f"    {row['ticker']:<6} P/C={pc:.3f}  Z={z:+.2f}  {direction}")


def run_backfill():
    """
    Start collecting per-ticker data.
    yfinance only provides current snapshots (no historical chains).
    After 20 trading days of daily runs, rolling metrics become valid.
    """
    logger.info("=" * 60)
    logger.info("PER-TICKER OPTIONS - BACKFILL START")
    logger.info("=" * 60)
    logger.info("")
    logger.info("NOTE: yfinance provides current-day snapshots only.")
    logger.info("Deep historical options data is not available for free.")
    logger.info("")
    logger.info("Strategy:")
    logger.info("  1. Run 'daily' each trading day after 4:30 PM ET")
    logger.info("  2. After 20 trading days, PC_20D_Avg and PC_Z_Score activate")
    logger.info("  3. Alerts will fire when per-ticker P/C spikes above Z=1.5")
    logger.info("")
    logger.info("Starting first collection now...")
    logger.info("")
    
    run_daily()


def run_test():
    """Test with 3 tickers to verify yfinance works."""
    logger.info("=" * 60)
    logger.info("PER-TICKER OPTIONS - TEST (3 tickers)")
    logger.info("=" * 60)
    
    test_tickers = ["AAPL", "NVDA", "CRM"]
    
    for ticker_sym in test_tickers:
        logger.info(f"\n  Testing {ticker_sym}...")
        result = fetch_ticker_options(ticker_sym)
        
        if result:
            pc = result['put_volume'] / result['call_volume'] if result['call_volume'] > 0 else 0
            pc_oi = result['put_oi'] / result['call_oi'] if result['call_oi'] > 0 else 0
            logger.info(f"    Expirations used: {result['expirations_used']}")
            logger.info(f"    Put volume:  {result['put_volume']:>12,}")
            logger.info(f"    Call volume: {result['call_volume']:>12,}")
            logger.info(f"    P/C (volume): {pc:.4f}")
            logger.info(f"    Put OI:      {result['put_oi']:>12,}")
            logger.info(f"    Call OI:     {result['call_oi']:>12,}")
            logger.info(f"    P/C (OI):     {pc_oi:.4f}")
        else:
            logger.error(f"    FAILED - check yfinance installation and network")
        
        time.sleep(DELAY_BETWEEN_TICKERS)
    
    logger.info(f"\nTEST COMPLETE")
    logger.info(f"\nIf all 3 tickers returned data, run:")
    logger.info(f"  python options_perticker.py daily     # Full 55-ticker collection")
    logger.info(f"  python options_perticker.py push      # Sync to Google Sheets")


def run_check(ticker):
    """Check options history for a specific ticker."""
    logger.info("=" * 60)
    logger.info(f"PER-TICKER OPTIONS - {ticker.upper()}")
    logger.info("=" * 60)
    
    # Fetch live data
    logger.info("\nLive data:")
    result = fetch_ticker_options(ticker.upper())
    if result:
        pc = result['put_volume'] / result['call_volume'] if result['call_volume'] > 0 else 0
        logger.info(f"  Put volume:   {result['put_volume']:>12,}")
        logger.info(f"  Call volume:  {result['call_volume']:>12,}")
        logger.info(f"  P/C Ratio:    {pc:.4f}")
        logger.info(f"  Expirations:  {result['expirations_used']}")
    else:
        logger.info("  No live data available")
    
    # Check Supabase history
    df = load_from_supabase(ticker.upper())
    if not df.empty:
        logger.info(f"\nHistory ({len(df)} days):")
        logger.info(f"  {'Date':<12} {'Put Vol':>10} {'Call Vol':>10} {'P/C':>8} {'20D Avg':>8} {'Z':>8}")
        logger.info(f"  {'-'*60}")
        for _, row in df.tail(20).iterrows():
            date = row.get('date', '?')
            pv = f"{row.get('put_volume', 0):>10,}" if row.get('put_volume') else "?"
            cv = f"{row.get('call_volume', 0):>10,}" if row.get('call_volume') else "?"
            pc_r = f"{row.get('put_call_ratio', 0):>8.4f}" if row.get('put_call_ratio') else "?"
            avg = f"{row.get('pc_20d_avg', 0):>8.4f}" if row.get('pc_20d_avg') else "N/A"
            z = f"{row.get('pc_z_score', 0):>8.2f}" if row.get('pc_z_score') else "N/A"
            logger.info(f"  {date:<12} {pv} {cv} {pc_r} {avg} {z}")
    else:
        logger.info("\nNo history in Supabase yet. Run 'daily' to start collecting.")


def run_push():
    """Push to Google Sheets."""
    logger.info("=" * 60)
    logger.info("PER-TICKER OPTIONS - PUSH TO SHEETS")
    logger.info("=" * 60)
    push_to_sheets()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Per-Ticker Options P/C Pipeline (yfinance)")
        print("=" * 50)
        print("Usage:")
        print("  python options_perticker.py daily          # Fetch 55 tickers (~2 min)")
        print("  python options_perticker.py backfill       # Start collecting (= daily)")
        print("  python options_perticker.py test           # Test with 3 tickers")
        print("  python options_perticker.py check NVDA     # Check specific ticker")
        print("  python options_perticker.py push           # Push to Google Sheets")
        print("")
        print("Data source: Yahoo Finance via yfinance (free)")
        print("Run daily after 4:30 PM ET. Rolling metrics valid after 20 days.")
        print(f"Priority tickers ({len(PRIORITY_TICKERS)}):")
        for i in range(0, len(PRIORITY_TICKERS), 10):
            print(f"  {', '.join(PRIORITY_TICKERS[i:i+10])}")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "daily":
        run_daily()
    elif cmd == "backfill":
        run_backfill()
    elif cmd == "test":
        run_test()
    elif cmd == "check" and len(sys.argv) >= 3:
        run_check(sys.argv[2])
    elif cmd == "push":
        run_push()
    else:
        print(f"Unknown command: {cmd}")
        print("Use: daily | backfill | test | check TICKER | push")
        sys.exit(1)