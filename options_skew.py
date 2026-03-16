"""
Options Skew Signal (Polygon Options Starter)
====================================================
Measures the implied-volatility spread between OTM puts and OTM calls for each
ticker in the scanner universe.  A steep positive skew = institutions buying
downside protection before price moves — leads DM deterioration by 2-5 days.

COMPUTATION:
  1. Fetch Polygon options snapshot for each ticker
  2. Find contracts closest to 30-day expiry
  3. OTM Put  = strike ~5% below underlying price
  4. OTM Call = strike ~5% above underlying price
  5. skew_raw   = IV(OTM Put) - IV(OTM Call)
  6. skew_zscore = (skew_raw - 30d rolling mean) / 30d rolling std
  7. Signal: NORMAL (z < 1.5) / ELEVATED (1.5-2.5) / EXTREME (> 2.5)

ENDPOINT:
  /v3/snapshot/options/{underlyingAsset}
  Each contract includes: implied_volatility, details.strike_price,
  details.expiration_date, details.contract_type, underlying_asset.price

OUTPUT:
  Supabase table: options_skew (date, ticker, skew_raw, iv_otm_put,
  iv_otm_call, skew_zscore, signal)

Usage:
  python options_skew.py daily                  # Full universe, upload
  python options_skew.py daily --tickers NVDA,TSLA   # Specific tickers
  python options_skew.py daily --dry-run        # Fetch + compute, no upload
  python options_skew.py zscore                 # Recalc z-scores from history
  python options_skew.py check NVDA             # Show history for a ticker

Pull schedule: Daily at 4:30 PM ET after market close.
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

DELAY_BETWEEN_TICKERS = 0.5   # seconds — courtesy delay
TARGET_DTE = 30               # target days-to-expiry
OTM_PCT = 0.05                # 5% out-of-the-money
ZSCORE_WINDOW = 30            # rolling window for z-score
ZSCORE_MIN_PERIODS = 10       # min observations for z-score
SUPA_TABLE = "options_skew"
BATCH_SIZE = 500


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
# POLYGON OPTIONS SNAPSHOT — SKEW EXTRACTION
# ============================================================
def _get_underlying_price(ticker):
    """Get underlying stock price from Polygon stock snapshot."""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    try:
        resp = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=15)
        if resp.status_code == 200:
            t = resp.json().get("ticker", {})
            # Try last trade price, then day close, then prevDay close
            price = (t.get("lastTrade", {}).get("p")
                     or t.get("day", {}).get("c")
                     or t.get("prevDay", {}).get("c"))
            if price and price > 0:
                return float(price)
    except Exception:
        pass
    return None


def fetch_skew_for_ticker(ticker):
    """
    Fetch options snapshot for a ticker, find the best OTM put and OTM call
    near 30-day expiry, return their IVs and the skew.

    Returns dict with iv_otm_put, iv_otm_call, skew_raw, or None on failure.
    """
    base_url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {"limit": 250, "apiKey": POLYGON_API_KEY}

    today = datetime.now()
    target_expiry = today + timedelta(days=TARGET_DTE)

    underlying_price = None
    candidates_put = []
    candidates_call = []

    pages = 0
    while True:
        try:
            resp = requests.get(base_url, params=params, timeout=30)

            if resp.status_code == 403:
                logger.warning(f"  {ticker}: 403 Forbidden — need Options Starter tier")
                return None
            if resp.status_code == 429:
                logger.warning(f"  {ticker}: Rate limited, waiting 10s...")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                logger.warning(f"  {ticker}: HTTP {resp.status_code}")
                return None

            data = resp.json()
            if data.get("status") == "ERROR":
                logger.warning(f"  {ticker}: API error — {data.get('error', 'unknown')}")
                return None

            results = data.get("results", [])
            if not results:
                break

            for contract in results:
                # Get underlying price (same across all contracts for this ticker)
                if underlying_price is None:
                    ua = contract.get("underlying_asset", {})
                    underlying_price = ua.get("price")

                iv = contract.get("implied_volatility")
                if iv is None or iv <= 0:
                    continue

                details = contract.get("details", {})
                strike = details.get("strike_price")
                expiry_str = details.get("expiration_date")
                ctype = details.get("contract_type", "").lower()

                if not strike or not expiry_str or not ctype:
                    continue

                try:
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
                except ValueError:
                    continue

                dte = (expiry - today).days
                if dte < 7 or dte > 60:
                    continue  # skip weeklies and far-out

                entry = {
                    "strike": strike,
                    "iv": iv,
                    "dte": dte,
                    "dte_diff": abs(dte - TARGET_DTE),
                }

                if ctype == "put":
                    candidates_put.append(entry)
                elif ctype == "call":
                    candidates_call.append(entry)

            next_url = data.get("next_url")
            if next_url:
                base_url = next_url
                params = {"apiKey": POLYGON_API_KEY}
                pages += 1
                time.sleep(0.15)
            else:
                break

        except requests.exceptions.Timeout:
            logger.warning(f"  {ticker}: Timeout")
            return None
        except Exception as e:
            logger.warning(f"  {ticker}: Error — {e}")
            return None

    # Fallback: get underlying price from stock snapshot
    if underlying_price is None or underlying_price <= 0:
        underlying_price = _get_underlying_price(ticker)
    if underlying_price is None or underlying_price <= 0:
        logger.warning(f"  {ticker}: No underlying price")
        return None

    # Target strikes: 5% OTM
    put_target_strike = underlying_price * (1 - OTM_PCT)
    call_target_strike = underlying_price * (1 + OTM_PCT)

    # Score candidates: minimize distance to target strike + target DTE
    def score(c, target_strike):
        strike_dist = abs(c["strike"] - target_strike) / underlying_price
        dte_dist = c["dte_diff"] / TARGET_DTE
        return strike_dist + dte_dist

    if not candidates_put or not candidates_call:
        logger.warning(f"  {ticker}: Insufficient contracts (puts={len(candidates_put)}, calls={len(candidates_call)})")
        return None

    best_put = min(candidates_put, key=lambda c: score(c, put_target_strike))
    best_call = min(candidates_call, key=lambda c: score(c, call_target_strike))

    iv_put = round(best_put["iv"], 4)
    iv_call = round(best_call["iv"], 4)
    skew_raw = round(iv_put - iv_call, 4)

    return {
        "iv_otm_put": iv_put,
        "iv_otm_call": iv_call,
        "skew_raw": skew_raw,
        "underlying_price": round(underlying_price, 2),
        "put_strike": best_put["strike"],
        "put_dte": best_put["dte"],
        "call_strike": best_call["strike"],
        "call_dte": best_call["dte"],
    }


# ============================================================
# DAILY FETCH — ALL TICKERS
# ============================================================
def run_daily(tickers, dry_run=False):
    """Fetch skew for all tickers, upload to Supabase."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Options Skew — daily run for {today_str}")
    logger.info(f"  Tickers: {len(tickers)}")

    records = []
    success = 0
    fail = 0

    for i, ticker in enumerate(tickers):
        result = fetch_skew_for_ticker(ticker)
        if result is None:
            fail += 1
        else:
            records.append({
                "date": today_str,
                "ticker": ticker,
                "skew_raw": result["skew_raw"],
                "iv_otm_put": result["iv_otm_put"],
                "iv_otm_call": result["iv_otm_call"],
                "skew_zscore": None,   # computed after upload from history
                "signal": None,
            })
            success += 1

        if (i + 1) % 25 == 0:
            logger.info(f"  Progress: {i+1}/{len(tickers)} ({success} ok, {fail} fail)")

        time.sleep(DELAY_BETWEEN_TICKERS)

    logger.info(f"  Fetch complete: {success} ok, {fail} fail")

    if not records:
        logger.warning("  No records to upload.")
        return

    if dry_run:
        logger.info("  DRY RUN — skipping upload")
        _print_table(records)
        return

    # Upload raw skew to Supabase
    uploaded = _upload_records(records)
    logger.info(f"  Uploaded {uploaded} rows to {SUPA_TABLE}")

    # Now compute z-scores from history
    logger.info("  Computing z-scores from history...")
    updated = compute_zscores(tickers)
    logger.info(f"  Updated z-scores for {updated} rows")


# ============================================================
# Z-SCORE COMPUTATION (from Supabase history)
# ============================================================
def compute_zscores(tickers=None):
    """
    Load skew history from Supabase, compute 30-day rolling z-score,
    classify signal, and update rows.
    """
    sb = get_supabase()

    # Load history — last 60 days is enough for 30-day rolling
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

        rolling_mean = tdf['skew_raw'].rolling(
            window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS
        ).mean()
        rolling_std = tdf['skew_raw'].rolling(
            window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS
        ).std()

        tdf['skew_zscore'] = np.where(
            rolling_std > 0,
            ((tdf['skew_raw'] - rolling_mean) / rolling_std).round(2),
            0
        )

        # Classify signal
        tdf['signal'] = np.where(
            tdf['skew_zscore'].abs() > 2.5, 'EXTREME',
            np.where(tdf['skew_zscore'].abs() > 1.5, 'ELEVATED', 'NORMAL')
        )

        # Only update the most recent row per ticker (today's run)
        latest = tdf.iloc[-1]
        updates.append({
            "date": latest['date'].strftime("%Y-%m-%d"),
            "ticker": ticker,
            "skew_raw": float(latest['skew_raw']) if pd.notna(latest['skew_raw']) else None,
            "iv_otm_put": float(latest['iv_otm_put']) if pd.notna(latest.get('iv_otm_put')) else None,
            "iv_otm_call": float(latest['iv_otm_call']) if pd.notna(latest.get('iv_otm_call')) else None,
            "skew_zscore": float(latest['skew_zscore']) if pd.notna(latest['skew_zscore']) else None,
            "signal": str(latest['signal']),
        })

    # Batch upsert updated rows
    if updates:
        _upload_records(updates)

    # Print alerts
    extreme = [u for u in updates if u['signal'] == 'EXTREME']
    elevated = [u for u in updates if u['signal'] == 'ELEVATED']
    if extreme:
        logger.info(f"\n  EXTREME SKEW ({len(extreme)} tickers):")
        for u in sorted(extreme, key=lambda x: abs(x['skew_zscore'] or 0), reverse=True):
            logger.info(f"    {u['ticker']:6s}  z={u['skew_zscore']:+.2f}  raw={u['skew_raw']:.4f}  put_iv={u['iv_otm_put']:.4f}  call_iv={u['iv_otm_call']:.4f}")
    if elevated:
        logger.info(f"\n  ELEVATED SKEW ({len(elevated)} tickers):")
        for u in sorted(elevated, key=lambda x: abs(x['skew_zscore'] or 0), reverse=True)[:10]:
            logger.info(f"    {u['ticker']:6s}  z={u['skew_zscore']:+.2f}  raw={u['skew_raw']:.4f}")

    return len(updates)


# ============================================================
# SUPABASE WRITE
# ============================================================
def _upload_records(records):
    """Batch upsert records to options_skew table."""
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
    """Display skew history for a specific ticker."""
    sb = get_supabase()
    result = sb.table(SUPA_TABLE) \
        .select("*") \
        .eq("ticker", ticker.upper()) \
        .order("date", desc=True) \
        .range(0, 29) \
        .execute()

    if not result.data:
        logger.info(f"No skew history for {ticker}")
        return

    logger.info(f"\nOptions Skew history for {ticker} (last {len(result.data)} days):")
    logger.info(f"  {'Date':12s} {'Skew_Raw':>10s} {'IV_Put':>10s} {'IV_Call':>10s} {'Z-Score':>10s} {'Signal':>10s}")
    logger.info(f"  {'-'*64}")

    for row in reversed(result.data):
        z = row.get('skew_zscore')
        z_str = f"{z:+.2f}" if z is not None else "   —"
        sig = row.get('signal', '—')
        logger.info(
            f"  {row['date']:12s} "
            f"{row.get('skew_raw', 0):10.4f} "
            f"{row.get('iv_otm_put', 0):10.4f} "
            f"{row.get('iv_otm_call', 0):10.4f} "
            f"{z_str:>10s} "
            f"{sig:>10s}"
        )


# ============================================================
# HELPERS
# ============================================================
def _print_table(records):
    """Print records as a formatted table."""
    logger.info(f"\n  {'Ticker':8s} {'Skew_Raw':>10s} {'IV_Put':>10s} {'IV_Call':>10s}")
    logger.info(f"  {'-'*40}")
    for r in sorted(records, key=lambda x: x['skew_raw'], reverse=True)[:30]:
        logger.info(
            f"  {r['ticker']:8s} "
            f"{r['skew_raw']:10.4f} "
            f"{r['iv_otm_put']:10.4f} "
            f"{r['iv_otm_call']:10.4f}"
        )
    if len(records) > 30:
        logger.info(f"  ... and {len(records) - 30} more")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Options Skew Signal (Polygon)")
    parser.add_argument("command", choices=["daily", "zscore", "check"],
                        help="daily=fetch+upload, zscore=recompute, check=show history")
    parser.add_argument("ticker", nargs="?", help="Ticker for 'check' command")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers (overrides universe)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + compute, no upload")

    args = parser.parse_args()

    # Validate env
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set in environment")
        sys.exit(1)
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL / SUPABASE_KEY not set in environment")
        sys.exit(1)

    if args.command == "check":
        if not args.ticker:
            logger.error("Usage: python options_skew.py check NVDA")
            sys.exit(1)
        check_ticker(args.ticker)

    elif args.command == "daily":
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            tickers = load_universe()
        run_daily(tickers, dry_run=args.dry_run)

    elif args.command == "zscore":
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            tickers = load_universe()
        updated = compute_zscores(tickers)
        logger.info(f"Z-score recompute done: {updated} tickers updated")


if __name__ == "__main__":
    main()
