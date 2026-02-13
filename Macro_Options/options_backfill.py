"""
Options P/C Ratio Backfill (Polygon Starter) - ASYNC VERSION
================================================================
Reconstructs 2 years of daily per-ticker put/call ratios using
concurrent API calls (10 parallel) for ~10x speedup.

CRM benchmark: ~7 hrs sequential → ~45 min async
Full 55 tickers: ~200 hrs sequential → ~20 hrs async

Usage:
  python options_backfill.py estimate              # Estimate contract counts
  python options_backfill.py run CRM               # Backfill one ticker
  python options_backfill.py run ADBE NOW WDAY     # Backfill several
  python options_backfill.py run --all             # Backfill all 55
  python options_backfill.py status                # Show progress

Requires: pip install aiohttp requests python-dotenv supabase
"""

import os
import sys
import time
import json
import logging
import asyncio
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
except Exception:
    supabase = None

# ============================================================
# CONFIG
# ============================================================
CONCURRENCY = 10              # Parallel API calls
CHECKPOINT_EVERY = 500        # Save progress every N contracts
CHECKPOINT_DIR = Path("backfill_checkpoints")

# Date range: 2 years
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

PRIORITY_TICKERS = [
    "CRM", "ADBE", "NOW", "WDAY", "ACN", "OMC", "LPLA", "SCHW", "RJF",
    "UNH", "KTOS", "AVAV", "MRCY", "WDC",
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "TSM", "INTC",
    "SNOW", "PLTR", "DDOG", "NET", "CRWD", "ZS", "PANW", "FTNT", "ORCL", "SAP",
    "INFY", "ADP", "PAYX", "IT", "AON", "AJG", "MMC", "SPGI", "MCO",
    "WIT", "CTSH", "HDB", "IBM",
    "GS", "MS", "JPM", "V", "MA", "BLK", "ICE",
    "EQIX", "DLR", "AMT", "CEG", "VST", "ANET", "DELL", "SMCI",
]
PRIORITY_TICKERS = list(dict.fromkeys(PRIORITY_TICKERS))

# Google Sheets
SHEETS_SPREADSHEET = "Macro_and_Options_History"
SHEETS_TAB = "Options_PutCall"


# ============================================================
# STEP 1: LIST CONTRACTS (synchronous - paginated, fast)
# ============================================================
def list_all_contracts(ticker):
    """
    List all option contracts for a ticker: both expired AND active.
    Two passes: first expired contracts, then active ones.
    Filtered to contracts expiring within our backfill window.
    """
    contracts = []
    seen_tickers = set()

    # Pass 1: Expired contracts, Pass 2: Active contracts
    for include_expired in ["true", "false"]:
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "expired": include_expired,
            "expiration_date.gte": START_DATE,
            "limit": 1000,
            "apiKey": POLYGON_API_KEY,
            "order": "asc",
            "sort": "expiration_date",
        }
        label = "expired" if include_expired == "true" else "active"

        page = 0
        while True:
            try:
                resp = requests.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    logger.warning("  Rate limited, waiting 10s...")
                    time.sleep(10)
                    continue
                if resp.status_code != 200:
                    logger.warning(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json()
                results = data.get("results", [])

                if not results:
                    break

                for c in results:
                    opt_ticker = c.get("ticker")
                    if opt_ticker not in seen_tickers:
                        contracts.append({
                            "option_ticker": opt_ticker,
                            "contract_type": c.get("contract_type"),
                            "expiration_date": c.get("expiration_date"),
                        })
                        seen_tickers.add(opt_ticker)

                page += 1
                if page % 10 == 0:
                    logger.info(f"    Page {page} ({label}): {len(contracts)} contracts so far...")

                next_url = data.get("next_url")
                if next_url:
                    url = next_url
                    params = {"apiKey": POLYGON_API_KEY}
                    time.sleep(0.15)
                else:
                    break

            except Exception as e:
                logger.error(f"  Error listing {label} contracts: {e}")
                break

        logger.info(f"    {label.capitalize()} pass done: {len(contracts)} total")

    return contracts


# ============================================================
# STEP 2: FETCH DAILY VOLUMES (async, concurrent)
# ============================================================
async def fetch_contract_volume(session, semaphore, option_ticker, start_date, end_date):
    """
    Fetch daily volume bars for one option contract (async).
    Returns list of {date, volume} dicts.
    """
    url = (f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}"
           f"/range/1/day/{start_date}/{end_date}")
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    async with semaphore:
        for attempt in range(3):  # retry up to 3 times
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(5 + attempt * 5)
                        continue
                    if resp.status != 200:
                        return option_ticker, []

                    data = await resp.json()
                    results = data.get("results", [])

                    if not results:
                        return option_ticker, []

                    daily = []
                    for bar in results:
                        ts = bar.get("t", 0)
                        dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                        vol = bar.get("v", 0) or 0
                        if vol > 0:
                            daily.append({"date": dt, "volume": int(vol)})

                    return option_ticker, daily

            except asyncio.TimeoutError:
                await asyncio.sleep(2)
                continue
            except Exception as e:
                return option_ticker, []

        return option_ticker, []


async def fetch_all_volumes(contracts, start_date, end_date, processed_set):
    """
    Fetch daily volumes for all contracts concurrently.
    Yields (option_ticker, contract_type, daily_bars) as they complete.
    """
    remaining = [c for c in contracts if c["option_ticker"] not in processed_set]

    if not remaining:
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5, force_close=False)

    # Build lookup: option_ticker -> contract_type
    type_lookup = {c["option_ticker"]: c["contract_type"] for c in remaining}

    results = {}  # option_ticker -> daily_bars
    total = len(remaining)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches to allow checkpointing
        batch_size = CHECKPOINT_EVERY
        for batch_start in range(0, total, batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            tasks = [
                fetch_contract_volume(session, semaphore, c["option_ticker"], start_date, end_date)
                for c in batch
            ]

            completed = 0
            for coro in asyncio.as_completed(tasks):
                opt_ticker, daily = await coro
                results[opt_ticker] = daily
                completed += 1

                # Progress every 100
                global_done = batch_start + completed
                if global_done % 100 == 0:
                    logger.info(f"    [{global_done}/{total}] contracts fetched...")

            # Yield batch results for checkpointing
            yield batch_start + len(batch), results.copy(), type_lookup

            # Small pause between batches
            await asyncio.sleep(0.5)


# ============================================================
# BACKFILL ONE TICKER (async orchestrator)
# ============================================================
async def backfill_ticker_async(ticker):
    """Full async backfill for one ticker."""
    start_time = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILL: {ticker}")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Concurrency: {CONCURRENCY} parallel requests")
    logger.info(f"{'='*60}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{ticker}_checkpoint.json"
    volume_file = CHECKPOINT_DIR / f"{ticker}_volumes.json"

    # Load checkpoint if exists
    processed_contracts = set()
    daily_volumes = {}

    if checkpoint_file.exists() and volume_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_contracts = set(checkpoint.get("processed", []))
        with open(volume_file, 'r') as f:
            daily_volumes = json.load(f)
        logger.info(f"  Resuming: {len(processed_contracts)} contracts already done")

    # Step 1: List contracts
    logger.info(f"\n  Step 1: Listing contracts for {ticker}...")
    contracts = list_all_contracts(ticker)
    logger.info(f"    Found {len(contracts)} contracts")

    if not contracts:
        logger.error(f"  No contracts found for {ticker}")
        return 0

    remaining_count = len([c for c in contracts if c["option_ticker"] not in processed_contracts])
    if remaining_count > 0:
        est_minutes = (remaining_count / CONCURRENCY * 1.5) / 60  # ~1.5s per call, /concurrency
        logger.info(f"    Remaining: {remaining_count}")
        logger.info(f"    Estimated time: {est_minutes:.0f} minutes (async {CONCURRENCY}x)")

        # Step 2: Fetch volumes concurrently
        logger.info(f"\n  Step 2: Fetching daily volumes (async)...")

        async for progress, batch_results, type_lookup in fetch_all_volumes(
            contracts, START_DATE, END_DATE, processed_contracts
        ):
            # Aggregate batch results into daily_volumes
            for opt_ticker, daily_bars in batch_results.items():
                if opt_ticker in processed_contracts:
                    continue

                ctype = type_lookup.get(opt_ticker, "")

                for bar in daily_bars:
                    dt = bar["date"]
                    vol = bar["volume"]

                    if dt not in daily_volumes:
                        daily_volumes[dt] = {"put_volume": 0, "call_volume": 0}

                    if ctype == "put":
                        daily_volumes[dt]["put_volume"] += vol
                    elif ctype == "call":
                        daily_volumes[dt]["call_volume"] += vol

                processed_contracts.add(opt_ticker)

            # Checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump({"processed": list(processed_contracts)}, f)
            with open(volume_file, 'w') as f:
                json.dump(daily_volumes, f)
            logger.info(f"    Checkpoint: {len(processed_contracts)}/{len(contracts)} contracts")

    # Step 3: Calculate metrics
    logger.info(f"\n  Step 3: Calculating metrics...")

    if not daily_volumes:
        logger.error(f"  No volume data for {ticker}")
        return 0

    rows = []
    for dt, vols in sorted(daily_volumes.items()):
        rows.append({
            "date": dt,
            "ticker": ticker,
            "put_volume": vols["put_volume"],
            "call_volume": vols["call_volume"],
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    df["put_call_ratio"] = np.where(
        df["call_volume"] > 0,
        (df["put_volume"] / df["call_volume"]).round(4),
        0
    )

    df["pc_20d_avg"] = df["put_call_ratio"].rolling(
        window=20, min_periods=10
    ).mean().round(4)

    pc_std = df["put_call_ratio"].rolling(window=20, min_periods=10).std()
    df["pc_z_score"] = np.where(
        pc_std > 0,
        ((df["put_call_ratio"] - df["pc_20d_avg"]) / pc_std).round(4),
        0
    )

    df = df[(df["put_volume"] > 0) | (df["call_volume"] > 0)]

    elapsed = (time.time() - start_time) / 60
    logger.info(f"    Trading days: {len(df)}")
    logger.info(f"    Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"    Avg P/C: {df['put_call_ratio'].mean():.4f}")
    logger.info(f"    Elapsed: {elapsed:.1f} minutes")

    # Step 4: Upload
    upload_ok = False
    if supabase:
        logger.info(f"\n  Step 4: Uploading to Supabase...")
        records = []
        for _, row in df.iterrows():
            records.append({
                'date': str(row['date']),
                'ticker': str(row['ticker']),
                'put_volume': int(row['put_volume']),
                'call_volume': int(row['call_volume']),
                'put_call_ratio': float(row['put_call_ratio']) if pd.notna(row['put_call_ratio']) else None,
                'pc_20d_avg': float(row['pc_20d_avg']) if pd.notna(row['pc_20d_avg']) else None,
                'pc_z_score': float(row['pc_z_score']) if pd.notna(row['pc_z_score']) else None,
            })

        uploaded = 0
        BATCH_SIZE = 500
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            try:
                supabase.table("options_history") \
                    .upsert(batch, on_conflict="date,ticker") \
                    .execute()
                uploaded += len(batch)
            except Exception as e:
                logger.error(f"    Upload error: {e}")

        logger.info(f"    Uploaded {uploaded} rows")
        upload_ok = (uploaded == len(records))

        if not upload_ok:
            # Save CSV as backup
            csv_path = CHECKPOINT_DIR / f"{ticker}_backfill.csv"
            df.to_csv(csv_path, index=False)
            logger.warning(f"    Partial upload. Backup saved to {csv_path}")
            logger.warning(f"    Re-run to retry upload (checkpoints preserved)")
    else:
        csv_path = CHECKPOINT_DIR / f"{ticker}_backfill.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"    Saved to {csv_path}")
        upload_ok = True

    # Only clean up checkpoints after successful upload
    if upload_ok:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if volume_file.exists():
            volume_file.unlink()

    logger.info(f"\n  DONE: {ticker} — {len(df)} rows in {elapsed:.1f} minutes")
    return len(df)


# ============================================================
# ESTIMATE (synchronous, quick)
# ============================================================
def run_estimate():
    """Estimate contract counts per ticker."""
    logger.info("=" * 60)
    logger.info("BACKFILL ESTIMATE")
    logger.info(f"Contracts with expiration >= {START_DATE}")
    logger.info("=" * 60)

    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set")
        return

    total_contracts = 0
    ticker_counts = []

    for i, ticker in enumerate(PRIORITY_TICKERS):
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "expired": "true",
            "expiration_date.gte": START_DATE,
            "limit": 1000,
            "apiKey": POLYGON_API_KEY,
        }

        count = 0
        has_more = False

        for p in range(5):  # up to 5 pages = 5000 contracts
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    break
                data = resp.json()
                results = data.get("results", [])
                count += len(results)

                next_url = data.get("next_url")
                if next_url:
                    url = next_url
                    params = {"apiKey": POLYGON_API_KEY}
                    has_more = (p == 4)
                else:
                    break
            except:
                break
            time.sleep(0.15)

        suffix = "+" if has_more else ""
        est_async_min = (count / CONCURRENCY * 1.5) / 60
        logger.info(f"  [{i+1:2d}/{len(PRIORITY_TICKERS)}] {ticker:<6} "
                     f"{count:>6,}{suffix} contracts  "
                     f"~{est_async_min:.0f} min (async {CONCURRENCY}x)")

        ticker_counts.append({"ticker": ticker, "contracts": count})
        total_contracts += count
        time.sleep(0.15)

    logger.info(f"\n  Total contracts: {total_contracts:,}+")
    est_hours = (total_contracts / CONCURRENCY * 1.5) / 3600
    logger.info(f"  Estimated total: {est_hours:.1f} hours (async {CONCURRENCY}x)")
    logger.info(f"  Sequential would be: {est_hours * CONCURRENCY:.0f} hours")


# ============================================================
# STATUS
# ============================================================
def run_status():
    """Show backfill progress per ticker."""
    logger.info("=" * 60)
    logger.info("BACKFILL STATUS")
    logger.info("=" * 60)

    if supabase is None:
        logger.error("Supabase not configured.")
        return

    done_count = 0
    for ticker in PRIORITY_TICKERS:
        try:
            result = supabase.table("options_history") \
                .select("date") \
                .eq("ticker", ticker) \
                .order("date") \
                .limit(1) \
                .execute()

            if not result.data:
                continue

            first_date = result.data[0]["date"]

            result2 = supabase.table("options_history") \
                .select("date") \
                .eq("ticker", ticker) \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            last_date = result2.data[0]["date"]

            # Count rows
            result3 = supabase.table("options_history") \
                .select("date", count="exact") \
                .eq("ticker", ticker) \
                .execute()
            count = len(result3.data)

            logger.info(f"  {ticker:<6}  {count:>4} days  {first_date} to {last_date}")
            done_count += 1
        except Exception:
            pass

    logger.info(f"\n  Tickers with data: {done_count}/{len(PRIORITY_TICKERS)}")

    # Check in-progress
    if CHECKPOINT_DIR.exists():
        checkpoints = list(CHECKPOINT_DIR.glob("*_checkpoint.json"))
        if checkpoints:
            logger.info(f"\n  In-progress (resumable):")
            for cp in checkpoints:
                t = cp.stem.replace("_checkpoint", "")
                with open(cp) as f:
                    data = json.load(f)
                logger.info(f"    {t}: {len(data.get('processed', []))} contracts done")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Options P/C Ratio Backfill (Async)")
        print("=" * 50)
        print()
        print(f"Concurrency: {CONCURRENCY} parallel requests")
        print(f"Tickers: {len(PRIORITY_TICKERS)}")
        print(f"Period: {START_DATE} to {END_DATE}")
        print()
        print("Usage:")
        print("  python options_backfill.py estimate           # Check contract counts")
        print("  python options_backfill.py run CRM            # Backfill one ticker")
        print("  python options_backfill.py run ADBE NOW WDAY  # Backfill several")
        print("  python options_backfill.py run --all          # Backfill all 55")
        print("  python options_backfill.py status             # Show progress")
        print()
        print("Features:")
        print(f"  - {CONCURRENCY}x concurrent API calls (~10x faster)")
        print(f"  - Checkpoints every {CHECKPOINT_EVERY} contracts (resume if interrupted)")
        print(f"  - Uploads to Supabase as each ticker completes")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "estimate":
        run_estimate()

    elif cmd == "run":
        if len(sys.argv) < 3:
            print("Specify tickers or --all")
            sys.exit(1)

        if sys.argv[2] == "--all":
            tickers = PRIORITY_TICKERS
        else:
            tickers = [t.upper() for t in sys.argv[2:]]

        logger.info(f"Backfilling {len(tickers)} tickers (async {CONCURRENCY}x)")

        total_rows = 0
        for ticker in tickers:
            rows = asyncio.run(backfill_ticker_async(ticker))
            total_rows += rows

        logger.info(f"\n{'='*60}")
        logger.info(f"ALL DONE: {total_rows} rows across {len(tickers)} tickers")

    elif cmd == "status":
        run_status()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == '__main__':
    main()


    # python Macro_Options/options_backfill.py run INTC SNOW PLTR DDOG NET CRWD ZS PANW FTNT ORCL SAP INFY ADP PAYX IT AON AJG MMC SPGI MCO WIT CTSH HDB IBM GS MS JPM V MA BLK ICE EQIX DLR AMT CEG VST ANET DELL SMCI LPLA SCHW RJF UNH KTOS AVAV MRCY WDC