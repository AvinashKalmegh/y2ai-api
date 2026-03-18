"""
Special Repo Rates / Repo Stress Signal (NY Fed SOFR Data)
====================================================
Monitors repo market stress using publicly available NY Fed secured rate data.
Individual security special rates require Bloomberg — this pipeline uses
aggregate SOFR/TGCR/BGCR data to detect repo market stress that precedes
short squeeze conditions and collateral hoarding.

SIGNAL COMPONENTS:
  1. SOFR Percentile Spread (p99 - p1): wide = more securities trading special
  2. TGCR vs SOFR Divergence: tri-party stress diverging from broader market
  3. Reverse Repo Volume: sharp drops = collateral hoarding
  4. SOFR Rate Z-score: sudden moves = funding stress

COMPOSITE: repo_stress_score (0-100)
  - Base = SOFR_spread_zscore * 25 (normalized)
  - + TGCR_SOFR_divergence_zscore * 25
  - + RRP_volume_drop * 25  (if volume drops > 1 std from mean)
  - + SOFR_rate_zscore * 25

DATA SOURCE:
  NY Fed Markets API (free, no auth):
  https://markets.newyorkfed.org/api/rates/secured/all/search.json

OUTPUT:
  Supabase table: repo_rates (date, sofr_rate, tgcr_rate, bgcr_rate,
  sofr_spread, rrp_volume, repo_stress_score, repo_stress_zscore, classification)

Usage:
  python repo_rates.py daily                    # Fetch latest, upload
  python repo_rates.py daily --dry-run          # Fetch + compute, no upload
  python repo_rates.py backfill                 # Backfill 2 years
  python repo_rates.py backfill --start 2025-01-01
  python repo_rates.py zscore                   # Recalc z-scores from history
  python repo_rates.py check                    # Show recent data

Pull schedule: Daily at 9:30 AM ET — NY Fed publishes previous day's rates ~8 AM ET.
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

NYFED_RATES_URL = "https://markets.newyorkfed.org/api/rates/secured/all/search.json"
NYFED_RRP_URL = "https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json"

ZSCORE_WINDOW = 30
ZSCORE_MIN_PERIODS = 10
SUPA_TABLE = "repo_rates"
BATCH_SIZE = 500

# Classification thresholds (repo_stress_score)
THRESHOLD_ELEVATED = 40.0
THRESHOLD_STRESSED = 60.0
THRESHOLD_CRITICAL = 80.0


# ============================================================
# NY FED DATA FETCH
# ============================================================
def fetch_nyfed_rates(start_date, end_date):
    """
    Fetch SOFR, TGCR, BGCR rates from NY Fed API.
    Returns dict keyed by date: {date: {sofr: {...}, tgcr: {...}, bgcr: {...}}}
    """
    params = {
        "startDate": start_date,
        "endDate": end_date,
    }

    try:
        resp = requests.get(NYFED_RATES_URL, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"  NY Fed rates API returned HTTP {resp.status_code}")
            return {}
    except Exception as e:
        logger.warning(f"  NY Fed rates fetch error: {e}")
        return {}

    data = resp.json()
    rates_by_date = {}

    for entry in data.get("refRates", []):
        date = entry.get("effectiveDate")
        rate_type = entry.get("type", "").upper()

        if date is None or rate_type == "SOFRAI":
            continue  # skip SOFR averages/index

        if date not in rates_by_date:
            rates_by_date[date] = {}

        rates_by_date[date][rate_type.lower()] = {
            "rate": entry.get("percentRate"),
            "p1": entry.get("percentPercentile1"),
            "p25": entry.get("percentPercentile25"),
            "p75": entry.get("percentPercentile75"),
            "p99": entry.get("percentPercentile99"),
            "volume": entry.get("volumeInBillions"),
        }

    logger.info(f"  NY Fed rates: {len(rates_by_date)} dates loaded")
    return rates_by_date


def fetch_nyfed_rrp(start_date, end_date):
    """
    Fetch reverse repo operation volumes from NY Fed API.
    Returns dict {date: volume_in_millions}
    """
    params = {
        "startDate": start_date,
        "endDate": end_date,
    }

    try:
        resp = requests.get(NYFED_RRP_URL, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"  NY Fed RRP API returned HTTP {resp.status_code}")
            return {}
    except Exception as e:
        logger.warning(f"  NY Fed RRP fetch error: {e}")
        return {}

    data = resp.json()
    rrp_by_date = {}

    for op in data.get("repo", {}).get("operations", []):
        date = op.get("operationDate")
        amount = op.get("totalAmtAccepted")
        if date and amount is not None:
            # Accumulate if multiple operations per day
            rrp_by_date[date] = rrp_by_date.get(date, 0) + amount

    logger.info(f"  NY Fed RRP: {len(rrp_by_date)} dates loaded")
    return rrp_by_date


# ============================================================
# CLASSIFICATION
# ============================================================
def _classify_stress(score):
    """Classify repo stress score."""
    if score is None:
        return "UNKNOWN"
    if score >= THRESHOLD_CRITICAL:
        return "CRITICAL"
    if score >= THRESHOLD_STRESSED:
        return "STRESSED"
    if score >= THRESHOLD_ELEVATED:
        return "ELEVATED"
    return "NORMAL"


# ============================================================
# BUILD RECORDS
# ============================================================
def build_daily_records(start_date, end_date):
    """
    Fetch NY Fed data and build repo stress records for a date range.
    Returns list of record dicts (without z-scores — those come from history).
    """
    rates_by_date = fetch_nyfed_rates(start_date, end_date)
    rrp_by_date = fetch_nyfed_rrp(start_date, end_date)

    if not rates_by_date:
        return []

    records = []
    for date in sorted(rates_by_date.keys()):
        day = rates_by_date[date]

        sofr = day.get("sofr", {})
        tgcr = day.get("tgcr", {})
        bgcr = day.get("bgcr", {})

        sofr_rate = sofr.get("rate")
        tgcr_rate = tgcr.get("rate")
        bgcr_rate = bgcr.get("rate")

        # SOFR percentile spread (p99 - p1): wide = collateral stress
        sofr_p99 = sofr.get("p99")
        sofr_p1 = sofr.get("p1")
        sofr_spread = round(sofr_p99 - sofr_p1, 4) if sofr_p99 is not None and sofr_p1 is not None else None

        # TGCR vs SOFR divergence
        tgcr_sofr_spread = round(tgcr_rate - sofr_rate, 4) if tgcr_rate is not None and sofr_rate is not None else None

        # Volumes
        sofr_volume = sofr.get("volume")
        rrp_volume = rrp_by_date.get(date)

        records.append({
            "date": date,
            "sofr_rate": sofr_rate,
            "tgcr_rate": tgcr_rate,
            "bgcr_rate": bgcr_rate,
            "sofr_spread": sofr_spread,
            "tgcr_sofr_spread": tgcr_sofr_spread,
            "sofr_volume": sofr_volume,
            "rrp_volume": rrp_volume,
            "repo_stress_score": None,   # computed from history
            "repo_stress_zscore": None,
            "classification": "NORMAL",
        })

    logger.info(f"  Built {len(records)} daily records")
    return records


# ============================================================
# STRESS SCORE COMPUTATION (from Supabase history)
# ============================================================
def compute_stress_scores(records_to_update=None):
    """
    Load repo history from Supabase, compute composite stress score
    and z-scores, then update rows.
    """
    sb = get_supabase()

    cutoff = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    all_rows = []
    offset = 0
    while True:
        result = sb.table(SUPA_TABLE).select("*") \
            .gte("date", cutoff) \
            .order("date") \
            .range(offset, offset + 999) \
            .execute()
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        offset += 1000

    if not all_rows:
        logger.info("  No history for stress score computation.")
        return 0

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Convert to numeric
    for col in ['sofr_rate', 'sofr_spread', 'tgcr_sofr_spread', 'sofr_volume', 'rrp_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Component z-scores (30-day rolling)
    def rolling_zscore(series):
        mean = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
        std = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std()
        return np.where(std > 0, (series - mean) / std, 0)

    # 1. SOFR spread z-score (higher = more stress)
    df['spread_z'] = rolling_zscore(df['sofr_spread'])

    # 2. TGCR-SOFR divergence z-score (negative = tri-party tighter = stress)
    df['tgcr_div_z'] = rolling_zscore(df['tgcr_sofr_spread'].abs())

    # 3. RRP volume z-score (large drops = collateral hoarding)
    # Invert: negative volume change = stress
    if df['rrp_volume'].notna().sum() >= ZSCORE_MIN_PERIODS:
        rrp_change = df['rrp_volume'].pct_change()
        df['rrp_z'] = -rolling_zscore(rrp_change)  # negative change = positive stress
    else:
        df['rrp_z'] = 0

    # 4. SOFR rate z-score (spikes = funding stress)
    df['sofr_z'] = rolling_zscore(df['sofr_rate'])

    # Composite stress score (0-100)
    # Each component contributes 0-25, clipped
    df['repo_stress_score'] = (
        np.clip(df['spread_z'] * 12.5 + 12.5, 0, 25) +
        np.clip(df['tgcr_div_z'] * 12.5 + 12.5, 0, 25) +
        np.clip(df['rrp_z'] * 12.5 + 12.5, 0, 25) +
        np.clip(df['sofr_z'] * 12.5 + 12.5, 0, 25)
    ).round(2)

    # Overall z-score of composite
    df['repo_stress_zscore'] = np.round(rolling_zscore(df['repo_stress_score']), 2)

    # Classification
    df['classification'] = df['repo_stress_score'].apply(_classify_stress)

    # Prepare updates
    updates = []
    for _, row in df.iterrows():
        updates.append({
            "date": row['date'].strftime("%Y-%m-%d"),
            "sofr_rate": float(row['sofr_rate']) if pd.notna(row['sofr_rate']) else None,
            "tgcr_rate": float(row['tgcr_rate']) if pd.notna(row['tgcr_rate']) else None,
            "bgcr_rate": float(row['bgcr_rate']) if pd.notna(row['bgcr_rate']) else None,
            "sofr_spread": float(row['sofr_spread']) if pd.notna(row['sofr_spread']) else None,
            "tgcr_sofr_spread": float(row['tgcr_sofr_spread']) if pd.notna(row['tgcr_sofr_spread']) else None,
            "sofr_volume": float(row['sofr_volume']) if pd.notna(row['sofr_volume']) else None,
            "rrp_volume": int(row['rrp_volume']) if pd.notna(row['rrp_volume']) else None,
            "repo_stress_score": float(row['repo_stress_score']) if pd.notna(row['repo_stress_score']) else None,
            "repo_stress_zscore": float(row['repo_stress_zscore']) if pd.notna(row['repo_stress_zscore']) else None,
            "classification": row['classification'],
        })

    if updates:
        _upload_records(updates)

    # Print latest status
    latest = df.iloc[-1]
    logger.info(f"\n  Latest repo stress ({latest['date'].strftime('%Y-%m-%d')}):")
    logger.info(f"    SOFR:   {latest['sofr_rate']}%  spread: {latest['sofr_spread']}")
    logger.info(f"    TGCR:   {latest['tgcr_rate']}%  BGCR: {latest['bgcr_rate']}%")
    logger.info(f"    Stress: {latest['repo_stress_score']:.1f}  z={latest['repo_stress_zscore']:+.2f}  [{latest['classification']}]")

    # Alerts
    stressed = df[df['classification'].isin(['STRESSED', 'CRITICAL'])]
    if len(stressed) > 0:
        recent_stressed = stressed.tail(5)
        logger.info(f"\n  STRESS ALERTS (last {len(recent_stressed)}):")
        for _, r in recent_stressed.iterrows():
            logger.info(f"    {r['date'].strftime('%Y-%m-%d')}  score={r['repo_stress_score']:.1f}  z={r['repo_stress_zscore']:+.2f}  [{r['classification']}]")

    return len(updates)


# ============================================================
# DAILY RUN
# ============================================================
def run_daily(dry_run=False):
    """Fetch latest repo data and upload."""
    # Fetch last 7 days to handle weekends/holidays
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info(f"Repo Rates — daily run")
    logger.info(f"  Fetching {start_date} to {end_date}")

    records = build_daily_records(start_date, end_date)

    if not records:
        logger.warning("  No records available.")
        return

    if dry_run:
        logger.info("  DRY RUN — showing data:")
        for r in records:
            logger.info(
                f"    {r['date']}  SOFR={r['sofr_rate']}  TGCR={r['tgcr_rate']}  "
                f"spread={r['sofr_spread']}  RRP={r['rrp_volume']}"
            )
        return

    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} rows to {SUPA_TABLE}")

    logger.info("  Computing stress scores from history...")
    updated = compute_stress_scores()
    logger.info(f"  Updated {updated} rows with stress scores")


# ============================================================
# BACKFILL
# ============================================================
def run_backfill(start_date="2024-01-01"):
    """Backfill repo data from NY Fed."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Repo Rates — backfill from {start_date} to {end_date}")

    # Fetch in 90-day chunks (NY Fed API limit)
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_records = 0

    while current < end:
        chunk_end = min(current + timedelta(days=90), end)
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        records = build_daily_records(s, e)
        if records:
            uploaded = _upload_records(records)
            total_records += uploaded
            logger.info(f"  {s} to {e}: {uploaded} rows")

        current = chunk_end + timedelta(days=1)

    logger.info(f"\n  Backfill complete: {total_records} total rows uploaded")

    # Now compute stress scores
    logger.info("  Computing stress scores from history...")
    updated = compute_stress_scores()
    logger.info(f"  Updated {updated} rows with stress scores")


# ============================================================
# CHECK — RECENT DATA
# ============================================================
def check_recent():
    """Show recent repo rate data."""
    sb = get_supabase()
    result = sb.table(SUPA_TABLE) \
        .select("*") \
        .order("date", desc=True) \
        .range(0, 29) \
        .execute()

    if not result.data:
        logger.info("No repo rate data recorded.")
        return

    logger.info(f"\nRepo Rates (last {len(result.data)} days):")
    logger.info(f"  {'Date':12s} {'SOFR':>6s} {'TGCR':>6s} {'Spread':>8s} {'RRP($M)':>10s} {'Stress':>8s} {'Z':>8s} {'Class':>10s}")
    logger.info(f"  {'-'*70}")

    for row in reversed(result.data):
        z = row.get('repo_stress_zscore')
        z_str = f"{z:+.2f}" if z is not None else "—"
        score = row.get('repo_stress_score')
        score_str = f"{score:.1f}" if score is not None else "—"
        rrp = row.get('rrp_volume')
        rrp_str = f"{rrp/1e6:,.0f}" if rrp is not None else "—"
        logger.info(
            f"  {row['date']:12s} "
            f"{row.get('sofr_rate', 0):6.2f} "
            f"{row.get('tgcr_rate', 0):6.2f} "
            f"{row.get('sofr_spread', 0):8.4f} "
            f"{rrp_str:>10s} "
            f"{score_str:>8s} "
            f"{z_str:>8s} "
            f"{row.get('classification', '—'):>10s}"
        )


# ============================================================
# SUPABASE
# ============================================================
def _upload_records(records):
    """Batch upsert to repo_rates table."""
    sb = get_supabase()
    uploaded = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            sb.table(SUPA_TABLE).upsert(batch, on_conflict="date").execute()
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"  Upload error at batch {i}: {e}")

    return uploaded


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS repo_rates (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    sofr_rate DECIMAL(6,4),
    tgcr_rate DECIMAL(6,4),
    bgcr_rate DECIMAL(6,4),
    sofr_spread DECIMAL(6,4),
    tgcr_sofr_spread DECIMAL(6,4),
    sofr_volume DECIMAL(10,2),
    rrp_volume BIGINT,
    repo_stress_score DECIMAL(6,2),
    repo_stress_zscore DECIMAL(6,2),
    classification VARCHAR(20),
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_repo_rates_date ON repo_rates(date DESC);
CREATE INDEX IF NOT EXISTS idx_repo_rates_class ON repo_rates(classification);
"""


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Repo Stress Signal (NY Fed SOFR/TGCR/BGCR)")
    parser.add_argument("command", choices=["daily", "backfill", "zscore", "check", "schema"],
                        help="daily=fetch+upload, backfill=historical, zscore=recompute, check=recent, schema=SQL")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date for backfill (default: 2024-01-01)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + compute, no upload")

    args = parser.parse_args()

    if args.command == "schema":
        print(SCHEMA_SQL)
        return

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY not set in environment")
        sys.exit(1)

    if args.command == "daily":
        run_daily(dry_run=args.dry_run)

    elif args.command == "backfill":
        run_backfill(start_date=args.start)

    elif args.command == "zscore":
        updated = compute_stress_scores()
        logger.info(f"Stress score recompute done: {updated} rows updated")

    elif args.command == "check":
        check_recent()


if __name__ == "__main__":
    main()
