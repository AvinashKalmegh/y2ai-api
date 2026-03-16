"""
Cross-Asset Divergence: Equity DM vs Credit Spreads
====================================================
Detects when the equity market (DM scores) disagrees with the credit market
(HY/IG spreads).  Bond traders are typically more informed on credit
deterioration — this divergence caught Enron, Lehman, SVB, and every major
corporate credit event before equity signals fired.

SIGNAL TYPES:
  1. BULLISH_EQUITY_BEARISH_CREDIT
     DM > 65 (equity bullish) AND credit regime is Stressed/Crisis
     → Credit market sees trouble equity doesn't. High-conviction short signal.

  2. BEARISH_EQUITY_BULLISH_CREDIT
     DM < 35 (equity bearish) AND credit regime is Healthy
     → Equity fear is overdone, credit doesn't confirm. Contrarian long signal.

  3. MACRO_DIVERGENCE
     Aggregate DM across universe is rising but HY spreads widening (z > 1.5)
     → Systemic warning, reduce overall exposure.

DATA SOURCES (all from existing Supabase tables — no new API calls):
  - dm_history: per-ticker DM scores (dm_smoothed)
  - credit_spread_daily: HY/IG spread regimes + changes
  - macro_history: credit_z_score for normalized signal

Usage:
  python credit_divergence.py daily              # Run divergence scan
  python credit_divergence.py daily --date 2026-03-13
  python credit_divergence.py backfill           # Backfill from 2024-01-01
  python credit_divergence.py backfill --start 2025-01-01
  python credit_divergence.py check              # Show recent divergence events

Pull schedule: Daily at 9:00 AM ET (after credit spreads and DM are updated).
Requires: pip install python-dotenv supabase pandas numpy
"""

import os
import sys
import logging
import argparse
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

SUPA_TABLE = "credit_divergence"
BATCH_SIZE = 500

# DM thresholds
DM_BULLISH = 65      # DM above this = equity bullish
DM_BEARISH = 35      # DM below this = equity bearish

# Credit regime thresholds (from credit_spread_dial.py)
BEARISH_CREDIT_REGIMES = {"Crisis", "Stressed", "Fragile"}
BULLISH_CREDIT_REGIMES = {"Healthy"}

# Macro divergence: credit_z_score threshold
CREDIT_Z_THRESHOLD = 1.5  # spreads widening significantly


# ============================================================
# DATA LOADERS
# ============================================================
def load_dm_for_date(date_str):
    """Load all ticker DM scores for a given date."""
    sb = get_supabase()
    all_rows = []
    offset = 0

    while True:
        result = sb.table("dm_history") \
            .select("ticker,dm_smoothed,close,phase") \
            .eq("date", date_str) \
            .range(offset, offset + 999) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000

    return all_rows


def load_credit_for_date(date_str):
    """Load credit spread regime for a given date."""
    sb = get_supabase()

    # Try credit_spread_daily first (has regime classification)
    result = sb.table("credit_spread_daily") \
        .select("*") \
        .eq("date", date_str) \
        .execute()

    if result.data:
        return result.data[0]

    # Fallback: try nearest prior date (credit data may lag by a day)
    result = sb.table("credit_spread_daily") \
        .select("*") \
        .lte("date", date_str) \
        .order("date", desc=True) \
        .range(0, 0) \
        .execute()

    return result.data[0] if result.data else None


def load_macro_for_date(date_str):
    """Load macro data (credit_z_score) for a date."""
    sb = get_supabase()

    result = sb.table("macro_history") \
        .select("date,hy_spread,ig_spread,credit_z_score") \
        .lte("date", date_str) \
        .order("date", desc=True) \
        .range(0, 0) \
        .execute()

    return result.data[0] if result.data else None


# ============================================================
# DIVERGENCE DETECTION
# ============================================================
def detect_divergences(date_str):
    """
    Cross-reference DM scores with credit spreads for a date.
    Returns list of divergence records.
    """
    dm_rows = load_dm_for_date(date_str)
    credit = load_credit_for_date(date_str)
    macro = load_macro_for_date(date_str)

    if not dm_rows:
        logger.warning(f"  No DM data for {date_str}")
        return []

    if not credit:
        logger.warning(f"  No credit spread data for {date_str}")
        return []

    hy_regime = credit.get("hy_regime", "Unknown")
    ig_regime = credit.get("ig_regime", "Unknown")
    combined_regime = credit.get("combined_regime", "Unknown")
    hy_spread = credit.get("hy_spread")
    hy_20d_change = credit.get("hy_20d_change")

    credit_z = macro.get("credit_z_score") if macro else None

    # Determine if credit is bearish or bullish
    credit_is_bearish = hy_regime in BEARISH_CREDIT_REGIMES
    credit_is_bullish = hy_regime in BULLISH_CREDIT_REGIMES

    # Macro-level: is there a systemic divergence?
    macro_credit_warning = credit_z is not None and credit_z > CREDIT_Z_THRESHOLD

    records = []
    ticker_divergences = []

    for row in dm_rows:
        ticker = row.get("ticker")
        dm = row.get("dm_smoothed")
        if dm is None or ticker is None:
            continue

        divergence_type = None
        severity = "NONE"

        # Type 1: Equity bullish but credit bearish
        if dm > DM_BULLISH and credit_is_bearish:
            divergence_type = "BULLISH_EQUITY_BEARISH_CREDIT"
            severity = "HIGH" if hy_regime == "Crisis" else "MODERATE"

        # Type 2: Equity bearish but credit bullish
        elif dm < DM_BEARISH and credit_is_bullish:
            divergence_type = "BEARISH_EQUITY_BULLISH_CREDIT"
            severity = "MODERATE"

        # Type 3: Macro divergence overlay — equity bullish + credit z-score spiking
        elif dm > DM_BULLISH and macro_credit_warning:
            divergence_type = "MACRO_DIVERGENCE"
            severity = "HIGH" if credit_z > 2.5 else "MODERATE"

        if divergence_type:
            records.append({
                "date": date_str,
                "ticker": ticker,
                "dm_smoothed": round(float(dm), 2),
                "hy_regime": hy_regime,
                "ig_regime": ig_regime,
                "hy_spread": float(hy_spread) if hy_spread else None,
                "hy_20d_change": float(hy_20d_change) if hy_20d_change else None,
                "credit_z_score": float(credit_z) if credit_z else None,
                "divergence_type": divergence_type,
                "severity": severity,
            })
            ticker_divergences.append((ticker, dm, divergence_type, severity))

    # Summary
    logger.info(f"  {date_str}: {len(dm_rows)} tickers scanned, {len(records)} divergences found")
    logger.info(f"  Credit regime: HY={hy_regime}, IG={ig_regime}, Combined={combined_regime}")
    if hy_spread:
        logger.info(f"  HY spread: {hy_spread:.2f}  20d change: {hy_20d_change or 0:+.2f}")
    if credit_z is not None:
        logger.info(f"  Credit Z-score: {credit_z:+.2f}")

    if ticker_divergences:
        # Group by type
        for dtype in ["BULLISH_EQUITY_BEARISH_CREDIT", "MACRO_DIVERGENCE", "BEARISH_EQUITY_BULLISH_CREDIT"]:
            matches = [(t, d, s) for t, d, dt, s in ticker_divergences if dt == dtype]
            if matches:
                logger.info(f"\n  {dtype} ({len(matches)} tickers):")
                for t, d, s in sorted(matches, key=lambda x: x[1], reverse=True)[:15]:
                    logger.info(f"    {t:6s}  DM={d:.1f}  severity={s}")

    return records


# ============================================================
# DAILY RUN
# ============================================================
def run_daily(target_date=None):
    """Run divergence detection for a single date."""
    if target_date is None:
        # Default to last trading day
        today = datetime.now()
        if today.weekday() == 0:
            target_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        elif today.weekday() == 6:
            target_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            target_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Credit Divergence — daily scan for {target_date}")

    records = detect_divergences(target_date)

    if not records:
        logger.info("  No divergences detected — equity and credit agree.")
        return

    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} divergence records to {SUPA_TABLE}")


# ============================================================
# BACKFILL
# ============================================================
def run_backfill(start_date="2024-01-01"):
    """Backfill divergence detection across historical dates."""
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Credit Divergence — backfill from {start_date} to {end_date}")

    dates = pd.bdate_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    logger.info(f"  Trading days: {len(dates)}")

    total_uploaded = 0
    total_divergences = 0

    for i, date_str in enumerate(dates):
        records = detect_divergences(date_str)
        if records:
            uploaded = _upload_records(records)
            total_uploaded += uploaded
            total_divergences += len(records)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i+1}/{len(dates)} days ({total_divergences} divergences found)")

    logger.info(f"\n  Backfill complete: {total_uploaded} rows uploaded, {total_divergences} total divergences")


# ============================================================
# CHECK — RECENT DIVERGENCES
# ============================================================
def check_recent():
    """Show recent divergence events."""
    sb = get_supabase()
    result = sb.table(SUPA_TABLE) \
        .select("*") \
        .order("date", desc=True) \
        .range(0, 49) \
        .execute()

    if not result.data:
        logger.info("No divergence events recorded.")
        return

    logger.info(f"\nRecent Credit Divergences (last {len(result.data)} events):")
    logger.info(f"  {'Date':12s} {'Ticker':8s} {'DM':>6s} {'HY Regime':>12s} {'Credit Z':>10s} {'Type':>35s} {'Severity':>10s}")
    logger.info(f"  {'-'*96}")

    for row in result.data:
        cz = row.get('credit_z_score')
        cz_str = f"{cz:+.2f}" if cz is not None else "—"
        logger.info(
            f"  {row['date']:12s} "
            f"{row.get('ticker', ''):8s} "
            f"{row.get('dm_smoothed', 0):>6.1f} "
            f"{row.get('hy_regime', '—'):>12s} "
            f"{cz_str:>10s} "
            f"{row.get('divergence_type', '—'):>35s} "
            f"{row.get('severity', '—'):>10s}"
        )


# ============================================================
# SUPABASE
# ============================================================
def _upload_records(records):
    """Batch upsert to credit_divergence table."""
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
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cross-Asset Divergence (Equity DM vs Credit)")
    parser.add_argument("command", choices=["daily", "backfill", "check"],
                        help="daily=scan today, backfill=historical, check=recent events")
    parser.add_argument("--date", type=str, help="Target date for daily (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date for backfill (default: 2024-01-01)")

    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY not set in environment")
        sys.exit(1)

    if args.command == "daily":
        run_daily(target_date=args.date)

    elif args.command == "backfill":
        run_backfill(start_date=args.start)

    elif args.command == "check":
        check_recent()


if __name__ == "__main__":
    main()
