"""
ETF Creation/Redemption Signal (Polygon Reference API)
====================================================
Tracks daily shares outstanding changes for key ETFs to detect institutional
creation/redemption activity.  When authorized participants (APs) redeem large
ETF baskets, it signals institutional distribution BEFORE price moves.

ETF flows lead sector DM by 1-3 days — this is the earliest public signal
of institutional sector repositioning.

COMPUTATION:
  1. Fetch shares_outstanding from Polygon /v3/reference/tickers/{ticker}?date=
  2. Compare vs previous day stored in Supabase
  3. flow_pct = (yesterday - today) / yesterday * 100
     Positive flow_pct = redemption (outflow), negative = creation (inflow)
  4. Classify: |flow_pct| > 2% = SIGNIFICANT, > 0.5% = NOTABLE, else NORMAL

DATA SOURCE:
  Polygon /v3/reference/tickers/{ticker}?date={YYYY-MM-DD}
  Field: share_class_shares_outstanding
  Updates on T+1 basis (Friday data available Saturday)

ETF UNIVERSE (8 tickers, mapped to economic forces):
  QQQ  - Enterprise Demand
  XLK  - Technology
  SMH  - Semiconductors
  XLF  - Capital/Financials
  XLE  - Energy
  XLI  - Infrastructure
  ARKK - Innovation/Growth
  SPY  - Baseline

Usage:
  python etf_creations.py daily                  # Fetch today, compare vs yesterday
  python etf_creations.py daily --date 2026-03-14
  python etf_creations.py backfill               # Backfill from 2024-01-01
  python etf_creations.py backfill --start 2025-01-01
  python etf_creations.py check SPY              # Show history for a ticker
  python etf_creations.py dry-run                # Fetch + compute, no upload

Pull schedule: Daily at 8:00 AM ET — sponsors publish previous day's data overnight.
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
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

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

ETF_UNIVERSE = {
    "QQQ":  "Enterprise Demand",
    "XLK":  "Technology",
    "SMH":  "Semiconductors",
    "XLF":  "Capital/Financials",
    "XLE":  "Energy",
    "XLI":  "Infrastructure",
    "ARKK": "Innovation/Growth",
    "SPY":  "Baseline",
}

SUPA_TABLE = "etf_flows"
BATCH_SIZE = 500
DELAY_BETWEEN_CALLS = 0.25  # seconds — only 8 tickers, minimal load

# Thresholds for flow classification
THRESHOLD_SIGNIFICANT = 2.0   # |flow_pct| > 2%
THRESHOLD_NOTABLE = 0.5       # |flow_pct| > 0.5%


# ============================================================
# POLYGON — FETCH SHARES OUTSTANDING
# ============================================================
def fetch_shares_outstanding(ticker, date_str):
    """
    Fetch shares outstanding for a ticker on a specific date.
    Uses Polygon /v3/reference/tickers/{ticker}?date={date}

    Returns int shares_outstanding, or None on failure.
    """
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    params = {"date": date_str, "apiKey": POLYGON_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 429:
            logger.warning(f"  {ticker}: Rate limited, waiting 60s...")
            time.sleep(60)
            resp = requests.get(url, params=params, timeout=15)

        if resp.status_code != 200:
            logger.warning(f"  {ticker}: HTTP {resp.status_code}")
            return None

        data = resp.json()
        results = data.get("results", {})
        shares = results.get("share_class_shares_outstanding")

        if shares is None:
            # Try weighted_shares_outstanding as fallback
            shares = results.get("weighted_shares_outstanding")

        if shares and shares > 0:
            return int(shares)
        else:
            logger.warning(f"  {ticker} ({date_str}): No shares outstanding data")
            return None

    except requests.exceptions.Timeout:
        logger.warning(f"  {ticker}: Timeout")
        return None
    except Exception as e:
        logger.warning(f"  {ticker}: Error — {e}")
        return None


# ============================================================
# DAILY RUN
# ============================================================
def run_daily(target_date=None, dry_run=False):
    """
    Fetch shares outstanding for today (or target_date), compare vs previous
    stored value, compute flow_pct, classify, and upload.
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"ETF Creation/Redemption — daily run for {target_date}")

    # Load previous day's data from Supabase for comparison
    prev_data = _load_previous_day(target_date)

    records = []
    for ticker, sector in ETF_UNIVERSE.items():
        shares = fetch_shares_outstanding(ticker, target_date)
        if shares is None:
            continue

        prev_shares = prev_data.get(ticker)
        shares_change = None
        flow_pct = None
        flow_direction = "NEUTRAL"
        flow_signal = "NORMAL"

        if prev_shares and prev_shares > 0:
            shares_change = shares - prev_shares
            # Positive flow_pct = redemption (shares decreased = outflow)
            flow_pct = round((prev_shares - shares) / prev_shares * 100, 4)
            flow_pct = max(-99.9999, min(99.9999, flow_pct))

            if flow_pct > 0:
                flow_direction = "OUTFLOW"
            elif flow_pct < 0:
                flow_direction = "INFLOW"

            abs_pct = abs(flow_pct)
            if abs_pct >= THRESHOLD_SIGNIFICANT:
                flow_signal = "SIGNIFICANT"
            elif abs_pct >= THRESHOLD_NOTABLE:
                flow_signal = "NOTABLE"

        records.append({
            "date": target_date,
            "etf_ticker": ticker,
            "shares_outstanding": shares,
            "shares_change": shares_change,
            "flow_pct": flow_pct,
            "flow_direction": flow_direction,
            "flow_signal": flow_signal,
        })

        logger.info(
            f"  {ticker:5s}  shares={shares:>15,}  "
            f"change={shares_change if shares_change is not None else 'N/A':>12}  "
            f"flow={f'{flow_pct:+.4f}%' if flow_pct is not None else 'N/A':>10}  "
            f"{flow_direction:8s}  {flow_signal}"
        )

        time.sleep(DELAY_BETWEEN_CALLS)

    if not records:
        logger.warning("  No records to upload.")
        return

    # Print alerts
    notable = [r for r in records if r['flow_signal'] in ('NOTABLE', 'SIGNIFICANT')]
    if notable:
        logger.info(f"\n  ALERTS ({len(notable)} ETFs with notable flows):")
        for r in sorted(notable, key=lambda x: abs(x['flow_pct'] or 0), reverse=True):
            logger.info(
                f"    {r['etf_ticker']:5s}  {r['flow_direction']:8s}  "
                f"{r['flow_pct']:+.4f}%  ({r['flow_signal']})"
            )

    if dry_run:
        logger.info("  DRY RUN — skipping upload")
        return

    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} rows to {SUPA_TABLE}")


# ============================================================
# BACKFILL
# ============================================================
def run_backfill(start_date="2024-01-01"):
    """
    Backfill shares outstanding history from start_date to yesterday.
    Fetches each trading day for all 8 ETFs, computes flow metrics.
    """
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"ETF Creation/Redemption — backfill from {start_date} to {end_date}")

    # Generate all calendar dates
    dates = pd.bdate_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    logger.info(f"  Trading days to process: {len(dates)}")

    # First pass: fetch all shares outstanding
    all_data = {}  # {(date, ticker): shares}
    total_calls = len(dates) * len(ETF_UNIVERSE)
    calls_done = 0

    for date_str in dates:
        for ticker in ETF_UNIVERSE:
            shares = fetch_shares_outstanding(ticker, date_str)
            if shares:
                all_data[(date_str, ticker)] = shares
            calls_done += 1

            if calls_done % 100 == 0:
                logger.info(f"  Fetched {calls_done}/{total_calls} ({len(all_data)} valid)")

            time.sleep(DELAY_BETWEEN_CALLS)

    logger.info(f"  Fetch complete: {len(all_data)} data points")

    # Second pass: compute flow metrics
    records = []
    for ticker in ETF_UNIVERSE:
        ticker_dates = sorted([d for (d, t) in all_data if t == ticker])

        for i, date_str in enumerate(ticker_dates):
            shares = all_data[(date_str, ticker)]
            shares_change = None
            flow_pct = None
            flow_direction = "NEUTRAL"
            flow_signal = "NORMAL"

            if i > 0:
                prev_date = ticker_dates[i - 1]
                prev_shares = all_data.get((prev_date, ticker))
                if prev_shares and prev_shares > 0:
                    shares_change = shares - prev_shares
                    flow_pct = round((prev_shares - shares) / prev_shares * 100, 4)
                    # Clamp to DB precision DECIMAL(6,4) max ±99.9999
                    flow_pct = max(-99.9999, min(99.9999, flow_pct))

                    if flow_pct > 0:
                        flow_direction = "OUTFLOW"
                    elif flow_pct < 0:
                        flow_direction = "INFLOW"

                    abs_pct = abs(flow_pct)
                    if abs_pct >= THRESHOLD_SIGNIFICANT:
                        flow_signal = "SIGNIFICANT"
                    elif abs_pct >= THRESHOLD_NOTABLE:
                        flow_signal = "NOTABLE"

            records.append({
                "date": date_str,
                "etf_ticker": ticker,
                "shares_outstanding": shares,
                "shares_change": shares_change,
                "flow_pct": flow_pct,
                "flow_direction": flow_direction,
                "flow_signal": flow_signal,
            })

    logger.info(f"  Computed {len(records)} records")

    # Upload in batches
    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} rows to {SUPA_TABLE}")

    # Summary
    sig = [r for r in records if r['flow_signal'] == 'SIGNIFICANT']
    notable = [r for r in records if r['flow_signal'] == 'NOTABLE']
    logger.info(f"\n  Summary: {len(sig)} SIGNIFICANT events, {len(notable)} NOTABLE events")


# ============================================================
# SUPABASE
# ============================================================
def _upload_records(records):
    """Batch upsert to etf_flows table."""
    sb = get_supabase()
    uploaded = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            sb.table(SUPA_TABLE).upsert(batch, on_conflict="date,etf_ticker").execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


def _load_previous_day(target_date):
    """Load the most recent record per ETF before target_date."""
    sb = get_supabase()
    prev = {}

    for ticker in ETF_UNIVERSE:
        try:
            result = sb.table(SUPA_TABLE) \
                .select("etf_ticker,shares_outstanding") \
                .eq("etf_ticker", ticker) \
                .lt("date", target_date) \
                .order("date", desc=True) \
                .range(0, 0) \
                .execute()
            if result.data:
                prev[ticker] = result.data[0].get("shares_outstanding")
        except Exception:
            pass

    return prev


# ============================================================
# CHECK — SHOW HISTORY
# ============================================================
def check_ticker(ticker):
    """Display creation/redemption history for an ETF."""
    sb = get_supabase()
    result = sb.table(SUPA_TABLE) \
        .select("*") \
        .eq("etf_ticker", ticker.upper()) \
        .order("date", desc=True) \
        .range(0, 29) \
        .execute()

    if not result.data:
        logger.info(f"No history for {ticker}")
        return

    logger.info(f"\nETF Creation/Redemption history for {ticker} (last {len(result.data)} days):")
    logger.info(f"  {'Date':12s} {'Shares Out':>16s} {'Change':>12s} {'Flow %':>10s} {'Direction':>10s} {'Signal':>12s}")
    logger.info(f"  {'-'*74}")

    for row in reversed(result.data):
        shares = row.get('shares_outstanding', 0)
        change = row.get('shares_change')
        pct = row.get('flow_pct')
        logger.info(
            f"  {row['date']:12s} "
            f"{shares:>16,} "
            f"{change if change is not None else '—':>12} "
            f"{f'{pct:+.4f}%' if pct is not None else '—':>10} "
            f"{row.get('flow_direction', '—'):>10} "
            f"{row.get('flow_signal', '—'):>12}"
        )


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ETF Creation/Redemption Signal (Polygon)")
    parser.add_argument("command", choices=["daily", "backfill", "check", "dry-run"],
                        help="daily=fetch+upload, backfill=historical, check=show history, dry-run=no upload")
    parser.add_argument("ticker", nargs="?", help="Ticker for 'check' command")
    parser.add_argument("--date", type=str, help="Target date for daily (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date for backfill (default: 2024-01-01)")

    args = parser.parse_args()

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in environment")
        sys.exit(1)
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY not set in environment")
        sys.exit(1)

    if args.command == "check":
        if not args.ticker:
            logger.error("Usage: python etf_creations.py check SPY")
            sys.exit(1)
        check_ticker(args.ticker)

    elif args.command == "daily":
        run_daily(target_date=args.date)

    elif args.command == "dry-run":
        run_daily(dry_run=True)

    elif args.command == "backfill":
        run_backfill(start_date=args.start)


if __name__ == "__main__":
    main()
