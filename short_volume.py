"""
Short Volume Ratio Signal (FINRA Reg SHO)
====================================================
Tracks daily short volume ratio for the scanner universe using free FINRA data.
A spike in short volume ratio signals aggressive institutional short positioning
— leads DM deterioration by 1-3 days.

DATA SOURCE:
  FINRA Consolidated NMS Short Volume:
    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
  Free, no auth, 11,000+ tickers, updated daily after market close.

  NASDAQ Reg SHO Threshold List (supplemental):
    https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{YYYYMMDD}.txt
  Stocks with persistent FTDs — strong hard-to-borrow proxy.

COMPUTATION:
  1. short_ratio = ShortVolume / TotalVolume (0-1 scale)
  2. short_zscore = (short_ratio - 30d rolling mean) / 30d rolling std
  3. Signal: NORMAL (z < 1.5) / ELEVATED (1.5-2.5) / EXTREME (> 2.5)
  4. Combine with DM direction: EXTREME + DM declining = institutional short building

OUTPUT:
  Supabase table: short_volume (date, ticker, short_volume, total_volume,
  short_ratio, short_zscore, signal, on_threshold_list)

Usage:
  python short_volume.py daily                   # Fetch latest trading day
  python short_volume.py daily --date 2026-03-13
  python short_volume.py backfill                # Backfill from 2024-01-01
  python short_volume.py backfill --start 2025-01-01
  python short_volume.py check NVDA              # Show history for a ticker
  python short_volume.py zscore                  # Recompute z-scores from history

Pull schedule: Daily at 7:00 AM ET (FINRA publishes previous day's data overnight).
Requires: pip install requests python-dotenv supabase pandas numpy
"""

import os
import sys
import time
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ============================================================
# CONFIGURATION
# ============================================================

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
THRESHOLD_URL = "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date}.txt"

SUPA_TABLE = "short_volume"
BATCH_SIZE = 500
ZSCORE_WINDOW = 30
ZSCORE_MIN_PERIODS = 10


# ============================================================
# UNIVERSE
# ============================================================
def load_universe():
    """Load all tickers from scanner_universe table."""
    sb = get_supabase()
    logger.info("Loading universe from scanner_universe...")

    all_rows = []
    offset = 0
    batch = 1000
    while True:
        result = sb.table("scanner_universe") \
            .select("ticker") \
            .range(offset, offset + batch - 1) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < batch:
            break
        offset += batch

    tickers = set(r['ticker'].strip() for r in all_rows if r.get('ticker', '').strip())
    logger.info(f"  Loaded {len(tickers)} tickers")
    return tickers


# ============================================================
# FINRA DATA FETCH
# ============================================================
def fetch_finra_short_volume(date_str):
    """
    Fetch FINRA consolidated short volume file for a date.
    Returns dict {ticker: {short_volume, total_volume, short_ratio}}
    """
    # FINRA uses YYYYMMDD format
    date_compact = date_str.replace("-", "")
    url = FINRA_URL.format(date=date_compact)

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            logger.warning(f"  FINRA data not available for {date_str}")
            return None
        if resp.status_code != 200:
            logger.warning(f"  FINRA HTTP {resp.status_code} for {date_str}")
            return None
    except Exception as e:
        logger.warning(f"  FINRA fetch error: {e}")
        return None

    data = {}
    lines = resp.text.strip().split("\n")

    for line in lines[1:]:  # skip header
        parts = line.split("|")
        if len(parts) < 5:
            continue

        ticker = parts[1].strip()
        try:
            short_vol = float(parts[2])
            total_vol = float(parts[4])
        except (ValueError, IndexError):
            continue

        if total_vol <= 0:
            continue

        # Aggregate across markets (same ticker can appear multiple times)
        if ticker in data:
            data[ticker]["short_volume"] += short_vol
            data[ticker]["total_volume"] += total_vol
        else:
            data[ticker] = {
                "short_volume": short_vol,
                "total_volume": total_vol,
            }

    # Compute ratios
    for ticker in data:
        sv = data[ticker]["short_volume"]
        tv = data[ticker]["total_volume"]
        data[ticker]["short_ratio"] = round(sv / tv, 4) if tv > 0 else 0

    logger.info(f"  FINRA {date_str}: {len(data)} tickers loaded")
    return data


def fetch_threshold_list(date_str):
    """
    Fetch NASDAQ Reg SHO Threshold list for a date.
    Returns set of tickers on the threshold list.
    """
    date_compact = date_str.replace("-", "")
    url = THRESHOLD_URL.format(date=date_compact)

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return set()
    except Exception:
        return set()

    tickers = set()
    lines = resp.text.strip().split("\n")
    for line in lines[1:]:  # skip header
        parts = line.split("|")
        if len(parts) >= 1:
            t = parts[0].strip()
            if t and t != "File Creation Time":
                tickers.add(t)

    if tickers:
        logger.info(f"  Threshold list {date_str}: {len(tickers)} securities")
    return tickers


# ============================================================
# DAILY RUN
# ============================================================
def run_daily(target_date=None):
    """Fetch short volume for a date, filter to universe, upload."""
    if target_date is None:
        # Default to last trading day
        today = datetime.now()
        if today.weekday() == 0:  # Monday -> Friday
            target_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        elif today.weekday() == 6:  # Sunday -> Friday
            target_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            target_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Short Volume — daily run for {target_date}")

    universe = load_universe()
    finra_data = fetch_finra_short_volume(target_date)
    if finra_data is None:
        return

    threshold_tickers = fetch_threshold_list(target_date)

    records = []
    for ticker in universe:
        if ticker not in finra_data:
            continue

        d = finra_data[ticker]
        records.append({
            "date": target_date,
            "ticker": ticker,
            "short_volume": int(d["short_volume"]),
            "total_volume": int(d["total_volume"]),
            "short_ratio": d["short_ratio"],
            "short_z_score": None,  # computed after from history
            "short_20d_avg": None,
            "on_threshold_list": ticker in threshold_tickers,
        })

    logger.info(f"  Matched {len(records)} tickers in universe")

    if not records:
        return

    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} rows to {SUPA_TABLE}")

    # Compute z-scores
    logger.info("  Computing z-scores from history...")
    tickers_list = [r['ticker'] for r in records]
    updated = compute_zscores(tickers_list)
    logger.info(f"  Updated z-scores for {updated} rows")


# ============================================================
# BACKFILL
# ============================================================
def run_backfill(start_date="2024-01-01"):
    """Backfill short volume from start_date to yesterday."""
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Short Volume — backfill from {start_date} to {end_date}")

    universe = load_universe()
    dates = pd.bdate_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    logger.info(f"  Trading days: {len(dates)}, Universe: {len(universe)} tickers")

    total_uploaded = 0

    for i, date_str in enumerate(dates):
        finra_data = fetch_finra_short_volume(date_str)
        if finra_data is None:
            continue

        threshold_tickers = fetch_threshold_list(date_str)

        records = []
        for ticker in universe:
            if ticker not in finra_data:
                continue

            d = finra_data[ticker]
            records.append({
                "date": date_str,
                "ticker": ticker,
                "short_volume": int(d["short_volume"]),
                "total_volume": int(d["total_volume"]),
                "short_ratio": d["short_ratio"],
                "short_z_score": None,
                "short_20d_avg": None,
                "on_threshold_list": ticker in threshold_tickers,
            })

        if records:
            uploaded = _upload_records(records)
            total_uploaded += uploaded

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(dates)} days ({total_uploaded} total rows)")

        time.sleep(0.25)  # courtesy delay

    logger.info(f"  Backfill complete: {total_uploaded} rows uploaded")

    # Compute z-scores for all tickers
    logger.info("  Computing z-scores for full history...")
    updated = compute_zscores(list(universe))
    logger.info(f"  Updated z-scores for {updated} rows")


# ============================================================
# Z-SCORE COMPUTATION
# ============================================================
def compute_zscores(tickers=None):
    """
    Load short volume history, compute 30-day rolling z-score,
    classify signal, update rows.
    """
    sb = get_supabase()
    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    all_rows = []
    ticker_batches = [tickers[i:i+50] for i in range(0, len(tickers), 50)] if tickers else [None]

    for batch in ticker_batches:
        offset = 0
        while True:
            query = sb.table(SUPA_TABLE).select("*").gte("date", cutoff)
            if batch:
                query = query.in_("ticker", batch)
            result = query.order("date").range(offset, offset + 999).execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            if len(result.data) < 1000:
                break
            offset += 1000

    if not all_rows:
        return 0

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])

    updates = []
    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker].copy()

        rolling_mean = tdf['short_ratio'].rolling(
            window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS
        ).mean()
        rolling_std = tdf['short_ratio'].rolling(
            window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS
        ).std()

        tdf['short_20d_avg'] = rolling_mean.round(4)

        tdf['short_z_score'] = np.where(
            rolling_std > 0,
            ((tdf['short_ratio'] - rolling_mean) / rolling_std).round(2),
            0
        )

        tdf['_signal'] = np.where(
            tdf['short_z_score'].abs() > 2.5, 'EXTREME',
            np.where(tdf['short_z_score'].abs() > 1.5, 'ELEVATED', 'NORMAL')
        )

        # Update the most recent row
        latest = tdf.iloc[-1]
        updates.append({
            "date": latest['date'].strftime("%Y-%m-%d"),
            "ticker": ticker,
            "short_volume": int(latest['short_volume']),
            "total_volume": int(latest['total_volume']),
            "short_ratio": float(latest['short_ratio']),
            "short_20d_avg": float(latest['short_20d_avg']) if pd.notna(latest['short_20d_avg']) else None,
            "short_z_score": float(latest['short_z_score']) if pd.notna(latest['short_z_score']) else None,
            "on_threshold_list": bool(latest.get('on_threshold_list', False)),
            "_signal": str(latest['_signal']),  # internal, not stored in DB
        })

    # Strip internal _signal before uploading (not a DB column)
    db_records = [{k: v for k, v in u.items() if k != '_signal'} for u in updates]
    if db_records:
        _upload_records(db_records)

    # Print alerts
    extreme = [u for u in updates if u['_signal'] == 'EXTREME']
    elevated = [u for u in updates if u['_signal'] == 'ELEVATED']
    threshold = [u for u in updates if u.get('on_threshold_list')]

    if extreme:
        logger.info(f"\n  EXTREME SHORT VOLUME ({len(extreme)} tickers):")
        for u in sorted(extreme, key=lambda x: abs(x['short_z_score'] or 0), reverse=True)[:10]:
            logger.info(f"    {u['ticker']:6s}  ratio={u['short_ratio']:.4f}  z={u['short_z_score']:+.2f}")
    if elevated:
        logger.info(f"\n  ELEVATED SHORT VOLUME ({len(elevated)} tickers):")
        for u in sorted(elevated, key=lambda x: abs(x['short_z_score'] or 0), reverse=True)[:10]:
            logger.info(f"    {u['ticker']:6s}  ratio={u['short_ratio']:.4f}  z={u['short_z_score']:+.2f}")
    if threshold:
        logger.info(f"\n  REG SHO THRESHOLD LIST ({len(threshold)} tickers):")
        for u in threshold:
            logger.info(f"    {u['ticker']:6s}  ratio={u['short_ratio']:.4f}")

    return len(updates)


# ============================================================
# SUPABASE
# ============================================================
def _upload_records(records):
    """Batch upsert to short_volume table."""
    sb = get_supabase()
    uploaded = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            sb.table(SUPA_TABLE).upsert(batch, on_conflict="date,ticker").execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


# ============================================================
# CHECK
# ============================================================
def check_ticker(ticker):
    """Display short volume history for a ticker."""
    sb = get_supabase()
    result = sb.table(SUPA_TABLE) \
        .select("*") \
        .eq("ticker", ticker.upper()) \
        .order("date", desc=True) \
        .range(0, 29) \
        .execute()

    if not result.data:
        logger.info(f"No history for {ticker}")
        return

    logger.info(f"\nShort Volume history for {ticker} (last {len(result.data)} days):")
    logger.info(f"  {'Date':12s} {'Short Vol':>12s} {'Total Vol':>12s} {'Ratio':>8s} {'Z-Score':>10s} {'Signal':>10s} {'Threshold':>10s}")
    logger.info(f"  {'-'*76}")

    for row in reversed(result.data):
        z = row.get('short_z_score')
        z_str = f"{z:+.2f}" if z is not None else "—"
        threshold = "YES" if row.get('on_threshold_list') else ""
        # Derive signal from z-score for display
        if z is not None:
            sig = "EXTREME" if abs(z) > 2.5 else ("ELEVATED" if abs(z) > 1.5 else "NORMAL")
        else:
            sig = "—"
        logger.info(
            f"  {row['date']:12s} "
            f"{row.get('short_volume', 0):>12,} "
            f"{row.get('total_volume', 0):>12,} "
            f"{row.get('short_ratio', 0):>8.4f} "
            f"{z_str:>10s} "
            f"{sig:>10s} "
            f"{threshold:>10s}"
        )


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Short Volume Ratio Signal (FINRA)")
    parser.add_argument("command", choices=["daily", "backfill", "check", "zscore"],
                        help="daily=fetch+upload, backfill=historical, check=history, zscore=recompute")
    parser.add_argument("ticker", nargs="?", help="Ticker for 'check' command")
    parser.add_argument("--date", type=str, help="Target date for daily (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date for backfill (default: 2024-01-01)")

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY not set in environment")
        sys.exit(1)

    if args.command == "check":
        if not args.ticker:
            logger.error("Usage: python short_volume.py check NVDA")
            sys.exit(1)
        check_ticker(args.ticker)

    elif args.command == "daily":
        run_daily(target_date=args.date)

    elif args.command == "backfill":
        run_backfill(start_date=args.start)

    elif args.command == "zscore":
        universe = load_universe()
        updated = compute_zscores(list(universe))
        logger.info(f"Z-score recompute done: {updated} tickers")


if __name__ == "__main__":
    main()
