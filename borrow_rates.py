"""
Hard-to-Borrow / Borrow Pressure Signal
====================================================
Composite borrow-difficulty score per ticker, built from free public data:

  1. FINRA Short Volume Ratio — high short_ratio = heavy borrowing demand
  2. NASDAQ Reg SHO Threshold List — persistent FTDs = definitionally HTB
  3. Short volume z-score spike — sudden increase in borrow pressure

These three signals are combined into a single borrow_pressure_score (0-100)
that proxies institutional short positioning cost.

COMPUTATION:
  1. Fetch FINRA short volume for latest trading day
  2. Fetch NASDAQ Reg SHO threshold list
  3. Pull short_volume history from Supabase for z-score context
  4. Composite score:
     - Base = short_ratio * 100 (0-100 scale, typically 20-60)
     - Threshold bonus: +25 if on Reg SHO threshold list
     - Z-score spike bonus: +15 if short_zscore > 2.0
  5. Classify: NORMAL (<40), ELEVATED (40-60), HTB (60-80), EXTREME (>80)
  6. borrow_zscore = 30-day rolling z-score of borrow_pressure_score

OUTPUT:
  Supabase table: borrow_rates (date, ticker, borrow_rate_annualized,
  available_shares, borrow_zscore, classification)

Usage:
  python borrow_rates.py daily                        # Full universe, upload
  python borrow_rates.py daily --tickers NVDA,TSLA    # Specific tickers
  python borrow_rates.py daily --dry-run              # Fetch + compute, no upload
  python borrow_rates.py daily --date 2026-03-14      # Specific date
  python borrow_rates.py zscore                       # Recalc z-scores from history
  python borrow_rates.py check NVDA                   # Show history for a ticker

Pull schedule: Daily at 7:00 AM ET — FINRA publishes previous day's data overnight.
Requires: pip install requests python-dotenv supabase pandas numpy
"""

import os
import sys
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

ZSCORE_WINDOW = 30
ZSCORE_MIN_PERIODS = 10
SUPA_TABLE = "borrow_rates"
BATCH_SIZE = 500

# Composite score thresholds
THRESHOLD_ELEVATED = 40.0
THRESHOLD_HTB = 60.0
THRESHOLD_EXTREME = 80.0

# Composite score weights
THRESHOLD_LIST_BONUS = 25   # on Reg SHO threshold list
ZSCORE_SPIKE_BONUS = 15     # short_zscore > 2.0


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

    tickers = [r['ticker'].strip() for r in all_rows if r.get('ticker', '').strip()]
    logger.info(f"  Loaded {len(tickers)} tickers")
    return tickers


# ============================================================
# CLASSIFICATION
# ============================================================
def _classify_score(score):
    """Classify borrow pressure score into signal categories."""
    if score is None:
        return "UNKNOWN"
    if score >= THRESHOLD_EXTREME:
        return "EXTREME"
    if score >= THRESHOLD_HTB:
        return "HTB"
    if score >= THRESHOLD_ELEVATED:
        return "ELEVATED"
    return "NORMAL"


# ============================================================
# FINRA SHORT VOLUME FETCH
# ============================================================
def _fetch_finra_short_volume(date_str):
    """
    Fetch FINRA consolidated short volume for a date.
    Returns dict {ticker: {short_volume, total_volume, short_ratio}}
    """
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

        if ticker in data:
            data[ticker]["short_volume"] += short_vol
            data[ticker]["total_volume"] += total_vol
        else:
            data[ticker] = {
                "short_volume": short_vol,
                "total_volume": total_vol,
            }

    for ticker in data:
        sv = data[ticker]["short_volume"]
        tv = data[ticker]["total_volume"]
        data[ticker]["short_ratio"] = round(sv / tv, 4) if tv > 0 else 0

    logger.info(f"  FINRA {date_str}: {len(data)} tickers loaded")
    return data


# ============================================================
# NASDAQ REG SHO THRESHOLD LIST
# ============================================================
def _fetch_threshold_list(date_str):
    """
    Fetch NASDAQ Reg SHO Threshold list for a date.
    Returns set of tickers on the threshold list.
    """
    date_compact = date_str.replace("-", "")
    url = THRESHOLD_URL.format(date=date_compact)

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            logger.info(f"  Threshold list not available for {date_str}")
            return set()
    except Exception:
        return set()

    tickers = set()
    lines = resp.text.strip().split("\n")
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) >= 1:
            t = parts[0].strip()
            if t and t != "File Creation Time":
                tickers.add(t)

    if tickers:
        logger.info(f"  Threshold list {date_str}: {len(tickers)} securities")
    return tickers


# ============================================================
# SHORT VOLUME Z-SCORES FROM SUPABASE
# ============================================================
def _get_short_volume_zscores(tickers):
    """
    Pull latest short_zscore from the short_volume table for each ticker.
    Returns dict {ticker: zscore}
    """
    sb = get_supabase()
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    zscores = {}
    ticker_batches = [tickers[i:i+50] for i in range(0, len(tickers), 50)]

    for batch in ticker_batches:
        try:
            result = sb.table("short_volume") \
                .select("ticker,short_zscore") \
                .in_("ticker", batch) \
                .gte("date", week_ago) \
                .order("date", desc=True) \
                .execute()

            if result.data:
                for row in result.data:
                    t = row['ticker']
                    if t not in zscores and row.get('short_zscore') is not None:
                        zscores[t] = float(row['short_zscore'])
        except Exception as e:
            logger.warning(f"  Error fetching short_volume z-scores: {e}")

    return zscores


# ============================================================
# COMPOSITE BORROW PRESSURE SCORE
# ============================================================
def compute_borrow_pressure(tickers, date_str=None):
    """
    Compute composite borrow pressure score for each ticker.
    Returns list of record dicts.
    """
    if date_str is None:
        # Find latest trading day (skip weekends)
        d = datetime.now()
        if d.hour < 7:  # before 7 AM, use previous day
            d -= timedelta(days=1)
        while d.weekday() >= 5:  # skip weekends
            d -= timedelta(days=1)
        date_str = d.strftime("%Y-%m-%d")

    logger.info(f"  Target date: {date_str}")

    # 1. FINRA short volume
    finra_data = _fetch_finra_short_volume(date_str)
    if finra_data is None:
        # Try previous trading day
        d = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        alt_date = d.strftime("%Y-%m-%d")
        logger.info(f"  Retrying with {alt_date}...")
        finra_data = _fetch_finra_short_volume(alt_date)
        if finra_data is not None:
            date_str = alt_date

    if finra_data is None:
        logger.error("  No FINRA data available")
        return []

    # 2. Threshold list
    threshold_tickers = _fetch_threshold_list(date_str)

    # 3. Short volume z-scores from existing pipeline
    sv_zscores = _get_short_volume_zscores(tickers)

    # 4. Build composite score
    records = []
    for ticker in tickers:
        t = ticker.upper()

        if t in finra_data:
            short_ratio = finra_data[t]["short_ratio"]
            short_vol = int(finra_data[t]["short_volume"])
            total_vol = int(finra_data[t]["total_volume"])
        else:
            short_ratio = 0.0
            short_vol = 0
            total_vol = 0

        # Base score: short_ratio scaled to 0-100
        base_score = short_ratio * 100

        # Bonus: on threshold list
        on_threshold = t in threshold_tickers
        threshold_bonus = THRESHOLD_LIST_BONUS if on_threshold else 0

        # Bonus: short volume z-score spike
        sv_z = sv_zscores.get(t, 0)
        zscore_bonus = ZSCORE_SPIKE_BONUS if sv_z > 2.0 else 0

        # Composite
        pressure_score = min(100, round(base_score + threshold_bonus + zscore_bonus, 2))

        records.append({
            "date": date_str,
            "ticker": t,
            "borrow_rate_annualized": pressure_score,  # using this column for composite score
            "available_shares": total_vol if total_vol > 0 else None,
            "borrow_zscore": None,  # computed from history after upload
            "classification": _classify_score(pressure_score),
        })

    return records


# ============================================================
# DAILY FETCH — ALL TICKERS
# ============================================================
def run_daily(tickers, dry_run=False, date_str=None):
    """Fetch borrow pressure for all tickers, upload to Supabase."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Borrow Rates — daily run for {today_str}")
    logger.info(f"  Tickers: {len(tickers)}")

    records = compute_borrow_pressure(tickers, date_str=date_str)

    if not records:
        logger.warning("  No records to upload.")
        return

    # Show notable
    notable = [r for r in records if r['classification'] != 'NORMAL']
    if notable:
        logger.info(f"\n  Notable borrow pressure ({len(notable)} tickers):")
        for r in sorted(notable, key=lambda x: x['borrow_rate_annualized'], reverse=True)[:20]:
            logger.info(
                f"    {r['ticker']:6s}  score={r['borrow_rate_annualized']:6.2f}  "
                f"class={r['classification']:8s}"
            )
    else:
        logger.info("  No tickers above ELEVATED threshold")

    if dry_run:
        logger.info("  DRY RUN — skipping upload")
        _print_table(records)
        return

    # Upload to Supabase
    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} rows to {SUPA_TABLE}")

    # Compute z-scores from history
    logger.info("  Computing z-scores from history...")
    updated = compute_zscores(tickers)
    logger.info(f"  Updated z-scores for {updated} rows")


# ============================================================
# Z-SCORE COMPUTATION (from Supabase history)
# ============================================================
def compute_zscores(tickers=None):
    """
    Load borrow pressure history from Supabase, compute 30-day rolling z-score,
    and update rows.
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
        logger.info("  No history found for z-score computation.")
        return 0

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])

    updates = []
    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker].copy()

        rolling_mean = tdf['borrow_rate_annualized'].rolling(
            window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS
        ).mean()
        rolling_std = tdf['borrow_rate_annualized'].rolling(
            window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS
        ).std()

        tdf['borrow_zscore'] = np.where(
            rolling_std > 0,
            ((tdf['borrow_rate_annualized'] - rolling_mean) / rolling_std).round(2),
            0
        )

        latest = tdf.iloc[-1]
        score = float(latest['borrow_rate_annualized']) if pd.notna(latest['borrow_rate_annualized']) else 0.0
        zscore = float(latest['borrow_zscore']) if pd.notna(latest['borrow_zscore']) else None

        updates.append({
            "date": latest['date'].strftime("%Y-%m-%d"),
            "ticker": ticker,
            "borrow_rate_annualized": score,
            "available_shares": latest.get('available_shares'),
            "borrow_zscore": zscore,
            "classification": _classify_score(score),
        })

    if updates:
        _upload_records(updates)

    # Print alerts
    extreme = [u for u in updates if u['classification'] == 'EXTREME']
    htb = [u for u in updates if u['classification'] == 'HTB']
    zscore_alerts = [u for u in updates if u.get('borrow_zscore') and abs(u['borrow_zscore']) > 2.5]

    if extreme:
        logger.info(f"\n  EXTREME BORROW PRESSURE ({len(extreme)} tickers):")
        for u in sorted(extreme, key=lambda x: x['borrow_rate_annualized'], reverse=True):
            logger.info(f"    {u['ticker']:6s}  score={u['borrow_rate_annualized']:6.2f}  z={u.get('borrow_zscore', 0):+.2f}")

    if htb:
        logger.info(f"\n  HARD TO BORROW ({len(htb)} tickers):")
        for u in sorted(htb, key=lambda x: x['borrow_rate_annualized'], reverse=True)[:10]:
            logger.info(f"    {u['ticker']:6s}  score={u['borrow_rate_annualized']:6.2f}  z={u.get('borrow_zscore', 0):+.2f}")

    if zscore_alerts:
        logger.info(f"\n  Z-SCORE ALERTS (|z| > 2.5, {len(zscore_alerts)} tickers):")
        for u in sorted(zscore_alerts, key=lambda x: abs(x['borrow_zscore']), reverse=True)[:10]:
            logger.info(f"    {u['ticker']:6s}  score={u['borrow_rate_annualized']:6.2f}  z={u['borrow_zscore']:+.2f}")

    return len(updates)


# ============================================================
# SUPABASE WRITE
# ============================================================
def _upload_records(records):
    """Batch upsert records to borrow_rates table."""
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
# CHECK — SHOW HISTORY FOR A TICKER
# ============================================================
def check_ticker(ticker):
    """Display borrow rate history for a specific ticker."""
    sb = get_supabase()
    result = sb.table(SUPA_TABLE) \
        .select("*") \
        .eq("ticker", ticker.upper()) \
        .order("date", desc=True) \
        .range(0, 29) \
        .execute()

    if not result.data:
        logger.info(f"No borrow rate history for {ticker}")
        return

    logger.info(f"\nBorrow Pressure history for {ticker} (last {len(result.data)} days):")
    logger.info(f"  {'Date':12s} {'Score':>10s} {'Z-Score':>10s} {'Class':>10s}")
    logger.info(f"  {'-'*44}")

    for row in reversed(result.data):
        z = row.get('borrow_zscore')
        z_str = f"{z:+.2f}" if z is not None else "   —"
        logger.info(
            f"  {row['date']:12s} "
            f"{row.get('borrow_rate_annualized', 0):10.2f} "
            f"{z_str:>10s} "
            f"{row.get('classification', '—'):>10s}"
        )


# ============================================================
# HELPERS
# ============================================================
def _print_table(records):
    """Print records as a formatted table."""
    logger.info(f"\n  {'Ticker':8s} {'Score':>10s} {'Class':>10s}")
    logger.info(f"  {'-'*30}")
    for r in sorted(records, key=lambda x: x['borrow_rate_annualized'], reverse=True)[:30]:
        logger.info(
            f"  {r['ticker']:8s} "
            f"{r['borrow_rate_annualized']:10.2f} "
            f"{r['classification']:>10s}"
        )
    if len(records) > 30:
        logger.info(f"  ... and {len(records) - 30} more")


# ============================================================
# SQL SCHEMA
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS borrow_rates (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    borrow_rate_annualized DECIMAL(8,4),
    available_shares BIGINT,
    borrow_zscore DECIMAL(6,2),
    classification VARCHAR(20),
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_borrow_rates_date ON borrow_rates(date DESC);
CREATE INDEX IF NOT EXISTS idx_borrow_rates_ticker ON borrow_rates(ticker);
CREATE INDEX IF NOT EXISTS idx_borrow_rates_class ON borrow_rates(classification);
"""


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Borrow Pressure Signal (FINRA + Reg SHO)")
    parser.add_argument("command", choices=["daily", "zscore", "check", "schema"],
                        help="daily=fetch+upload, zscore=recompute, check=show history, schema=print SQL")
    parser.add_argument("ticker", nargs="?", help="Ticker for 'check' command")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers (overrides universe)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + compute, no upload")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD (default: latest trading day)")

    args = parser.parse_args()

    if args.command == "schema":
        print(SCHEMA_SQL)
        return

    # Validate env
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY not set in environment")
        sys.exit(1)

    if args.command == "check":
        if not args.ticker:
            logger.error("Usage: python borrow_rates.py check NVDA")
            sys.exit(1)
        check_ticker(args.ticker)

    elif args.command == "daily":
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            tickers = load_universe()
        run_daily(tickers, dry_run=args.dry_run, date_str=args.date)

    elif args.command == "zscore":
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            tickers = load_universe()
        updated = compute_zscores(tickers)
        logger.info(f"Z-score recompute done: {updated} tickers updated")


if __name__ == "__main__":
    main()
