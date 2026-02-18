"""
universe_scanner_crossings.py
FlowOS Universe Scanner - Phase 3: Crossings + Outcomes

Detects DM crossings above 70 and calculates 90-day forward returns.

Usage:
  python universe_scanner_crossings.py detect     # Scan universe_dm_daily for crossings
  python universe_scanner_crossings.py evaluate   # Calculate outcomes for mature crossings
  python universe_scanner_crossings.py report      # Show crossing stats and candidates
  python universe_scanner_crossings.py all         # Run detect + evaluate + report

Pipeline: Weekly run (after DM calculation).
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Crossing parameters
CROSSING_THRESHOLD = 70        # DM must cross above this
HIT_RETURN_THRESHOLD = 0.10    # 10% return = hit
OUTCOME_WINDOW_DAYS = 90       # Calendar days for outcome evaluation
MIN_CROSSINGS_CANDIDATE = 10   # Minimum evaluated crossings for candidate status
MIN_CROSSINGS_DEGRADATION = 5  # Minimum for degradation flag
CANDIDATE_HIT_RATE = 0.50      # 50% hit rate for new candidate
DEGRADATION_HIT_RATE = 0.40    # Below 40% flags degradation

# Era definitions for consistency check
ERAS = {
    "Pre-COVID (2016-2019)": ("2016-01-01", "2019-12-31"),
    "COVID+Recovery (2020-2022)": ("2020-01-01", "2022-12-31"),
    "AI Cycle (2023-2026)": ("2023-01-01", "2026-12-31"),
}

PREFERRED_28 = [
    "CEG", "CRWD", "TSM", "APP", "VRT", "MU", "NVDA", "CCJ",
    "DNN", "PLTR", "TSLA", "TTD", "MSTR", "UEC", "LEU", "HAL",
    "WDC", "ENPH", "UUUU", "PDD", "NCLH", "FCX", "RCL", "LVS",
    "PSKY", "MRNA", "WYNN", "SMR"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# DETECT: Find all DM crossings above 70
# ============================================================

def detect_crossings():
    """
    Scan universe_dm_daily for upward crossings above 70.
    A crossing occurs when dm_smoothed on day T >= 70 AND day T-1 < 70.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: DETECTING DM CROSSINGS")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    
    # Get all tickers from universe_tickers (one row per ticker)
    logger.info("Loading tickers from universe_tickers...")
    tickers_result = supabase.table("universe_tickers").select("ticker").execute()
    unique_tickers = sorted(r["ticker"] for r in tickers_result.data)
    logger.info(f"  {len(unique_tickers)} tickers to scan")
    
    # Get existing crossings to avoid duplicates
    existing = supabase.table("universe_crossings").select("ticker,cross_date").execute()
    existing_set = set()
    if existing.data:
        existing_set = {(r["ticker"], r["cross_date"]) for r in existing.data}
    logger.info(f"  {len(existing_set)} existing crossings (will skip)")
    
    # Process each ticker
    new_crossings = []
    processed = 0
    
    for i, ticker in enumerate(unique_tickers):
        # Fetch DM history for this ticker, sorted by date
        dm_data = supabase.table("universe_dm_daily") \
            .select("date,dm_smoothed,close") \
            .eq("ticker", ticker) \
            .order("date") \
            .execute()
        
        if not dm_data.data or len(dm_data.data) < 2:
            continue
        
        rows = dm_data.data
        
        for j in range(1, len(rows)):
            prev_dm = float(rows[j-1]["dm_smoothed"]) if rows[j-1]["dm_smoothed"] else 0
            curr_dm = float(rows[j]["dm_smoothed"]) if rows[j]["dm_smoothed"] else 0
            
            # Upward crossing: previous < 70, current >= 70
            if prev_dm < CROSSING_THRESHOLD and curr_dm >= CROSSING_THRESHOLD:
                cross_date = rows[j]["date"]
                
                if (ticker, cross_date) not in existing_set:
                    new_crossings.append({
                        "ticker": ticker,
                        "cross_date": cross_date,
                        "entry_dm": round(curr_dm, 2),
                        "entry_price": float(rows[j]["close"]) if rows[j]["close"] else None,
                        "evaluated": False,
                    })
        
        processed += 1
        if (i + 1) % 500 == 0:
            logger.info(f"  Scanned {i + 1}/{len(unique_tickers)} tickers ({len(new_crossings)} new crossings)")
    
    logger.info(f"  Scan complete: {len(new_crossings)} new crossings found")
    
    # Write to Supabase
    if new_crossings:
        BATCH = 500
        uploaded = 0
        for i in range(0, len(new_crossings), BATCH):
            batch = new_crossings[i:i + BATCH]
            try:
                supabase.table("universe_crossings").upsert(
                    batch, on_conflict="ticker,cross_date"
                ).execute()
                uploaded += len(batch)
            except Exception as e:
                logger.error(f"  Upsert error at batch {i}: {e}")
        
        logger.info(f"  Written {uploaded} new crossings to universe_crossings")
    
    return len(new_crossings)


# ============================================================
# EVALUATE: Calculate 90-day outcomes for mature crossings
# ============================================================

def evaluate_outcomes():
    """
    Batch-evaluate all crossings that have 90+ days of forward price data.
    Groups by ticker, pulls price history once per ticker, evaluates all crossings.
    Calculates 30d, 60d, and 90d max returns.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: EVALUATING CROSSING OUTCOMES (BATCH)")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    
    # --- Step 1: Get ALL unevaluated crossings (paginate past 1000 limit) ---
    logger.info("  Loading unevaluated crossings...")
    all_unevaluated = []
    page_size = 1000
    offset = 0
    
    while True:
        batch = supabase.table("universe_crossings") \
            .select("id,ticker,cross_date,entry_price") \
            .eq("evaluated", False) \
            .order("ticker") \
            .range(offset, offset + page_size - 1) \
            .execute()
        
        if not batch.data:
            break
        all_unevaluated.extend(batch.data)
        if len(batch.data) < page_size:
            break
        offset += page_size
    
    if not all_unevaluated:
        logger.info("  No unevaluated crossings found.")
        return 0
    
    logger.info(f"  {len(all_unevaluated)} unevaluated crossings loaded")
    
    # --- Step 2: Group crossings by ticker ---
    from collections import defaultdict
    ticker_crossings = defaultdict(list)
    for c in all_unevaluated:
        ticker_crossings[c["ticker"]].append(c)
    
    logger.info(f"  {len(ticker_crossings)} unique tickers to process")
    
    # --- Step 3: Get the latest date in universe_dm_daily for data boundary check ---
    latest = supabase.table("universe_dm_daily") \
        .select("date").order("date", desc=True).limit(1).execute()
    latest_date_str = latest.data[0]["date"] if latest.data else None
    
    if not latest_date_str:
        logger.error("  No data in universe_dm_daily!")
        return 0
    
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    logger.info(f"  Latest DM data: {latest_date_str}")
    
    # --- Step 4: Process each ticker ---
    total_evaluated = 0
    total_hits = 0
    total_skipped = 0
    updates = []
    ticker_count = 0
    
    for ticker, crossings in ticker_crossings.items():
        ticker_count += 1
        
        # Pull ALL price data for this ticker in one query (paginated)
        price_rows = []
        p_offset = 0
        while True:
            p_batch = supabase.table("universe_dm_daily") \
                .select("date,close") \
                .eq("ticker", ticker) \
                .order("date") \
                .range(p_offset, p_offset + page_size - 1) \
                .execute()
            
            if not p_batch.data:
                break
            price_rows.extend(p_batch.data)
            if len(p_batch.data) < page_size:
                break
            p_offset += page_size
        
        if not price_rows:
            total_skipped += len(crossings)
            continue
        
        # Build date->price lookup
        price_map = {}
        sorted_dates = []
        for row in price_rows:
            if row["close"] is not None:
                price_map[row["date"]] = float(row["close"])
                sorted_dates.append(row["date"])
        
        if not sorted_dates:
            total_skipped += len(crossings)
            continue
        
        last_available_date = sorted_dates[-1]
        last_available_dt = datetime.strptime(last_available_date, "%Y-%m-%d")
        
        # Evaluate each crossing for this ticker
        for crossing in crossings:
            cross_date_str = crossing["cross_date"]
            entry_price = float(crossing["entry_price"]) if crossing["entry_price"] else None
            
            if not entry_price or entry_price <= 0:
                updates.append({
                    "id": crossing["id"],
                    "evaluated": True,
                    "max_return_30d": 0, "max_return_60d": 0, "max_return_90d": 0,
                    "max_price_90d": 0, "is_hit": False,
                })
                total_evaluated += 1
                continue
            
            cross_dt = datetime.strptime(cross_date_str, "%Y-%m-%d")
            end_90d = cross_dt + timedelta(days=90)
            
            # Check if we have 90 days of forward data available
            if last_available_dt < end_90d:
                total_skipped += 1
                continue  # Leave evaluated=False
            
            # Find prices in 30d, 60d, 90d windows
            end_30d_str = (cross_dt + timedelta(days=30)).strftime("%Y-%m-%d")
            end_60d_str = (cross_dt + timedelta(days=60)).strftime("%Y-%m-%d")
            end_90d_str = end_90d.strftime("%Y-%m-%d")
            
            max_30d = 0
            max_60d = 0
            max_90d = 0
            
            for d in sorted_dates:
                if d <= cross_date_str:
                    continue
                if d > end_90d_str:
                    break
                p = price_map[d]
                if d <= end_30d_str and p > max_30d:
                    max_30d = p
                if d <= end_60d_str and p > max_60d:
                    max_60d = p
                if p > max_90d:
                    max_90d = p
            
            ret_30d = (max_30d / entry_price) - 1 if max_30d > 0 else 0
            ret_60d = (max_60d / entry_price) - 1 if max_60d > 0 else 0
            ret_90d = (max_90d / entry_price) - 1 if max_90d > 0 else 0
            is_hit = ret_90d >= HIT_RETURN_THRESHOLD
            
            if is_hit:
                total_hits += 1
            
            updates.append({
                "id": crossing["id"],
                "evaluated": True,
                "max_return_30d": round(ret_30d, 4),
                "max_return_60d": round(ret_60d, 4),
                "max_return_90d": round(ret_90d, 4),
                "max_price_90d": round(max_90d, 2),
                "is_hit": is_hit,
            })
            total_evaluated += 1
        
        # Batch write every 200 updates
        if len(updates) >= 200:
            _flush_updates(supabase, updates)
            updates = []
        
        if ticker_count % 200 == 0:
            logger.info(f"  Processed {ticker_count}/{len(ticker_crossings)} tickers "
                       f"({total_evaluated} evaluated, {total_hits} hits, {total_skipped} skipped)")
    
    # Flush remaining updates
    if updates:
        _flush_updates(supabase, updates)
    
    logger.info("")
    logger.info(f"  DONE: {total_evaluated} evaluated, {total_hits} hits "
                f"({total_hits/max(total_evaluated,1)*100:.1f}%), {total_skipped} skipped (insufficient forward data)")
    return total_evaluated


def _flush_updates(supabase, updates):
    """Write evaluation results back to Supabase in batches."""
    for u in updates:
        crossing_id = u.pop("id")
        supabase.table("universe_crossings").update(u).eq("id", crossing_id).execute()
    updates.clear()


# ============================================================
# REPORT: Generate weekly report with era analysis + CSV
# ============================================================

def _load_all_evaluated(supabase):
    """Load all evaluated crossings, paginating past 1000 limit."""
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        batch = supabase.table("universe_crossings") \
            .select("ticker,cross_date,entry_price,max_return_30d,max_return_60d,max_return_90d,is_hit") \
            .eq("evaluated", True) \
            .range(offset, offset + page_size - 1) \
            .execute()
        if not batch.data:
            break
        all_rows.extend(batch.data)
        if len(batch.data) < page_size:
            break
        offset += page_size
    return all_rows


def _classify_era(cross_date_str):
    """Return which era a crossing belongs to."""
    for era_name, (start, end) in ERAS.items():
        if start <= cross_date_str <= end:
            return era_name
    return "Unknown"


def _build_ticker_stats(all_evaluated):
    """Build per-ticker statistics from evaluated crossings."""
    from collections import defaultdict
    
    stats = defaultdict(lambda: {
        "total": 0, "hits": 0,
        "returns_30d": [], "returns_60d": [], "returns_90d": [],
        "eras": defaultdict(lambda: {"total": 0, "hits": 0}),
    })
    
    for r in all_evaluated:
        ticker = r["ticker"]
        s = stats[ticker]
        s["total"] += 1
        if r["is_hit"]:
            s["hits"] += 1
        
        if r.get("max_return_30d") is not None:
            s["returns_30d"].append(float(r["max_return_30d"]))
        if r.get("max_return_60d") is not None:
            s["returns_60d"].append(float(r["max_return_60d"]))
        if r.get("max_return_90d") is not None:
            s["returns_90d"].append(float(r["max_return_90d"]))
        
        era = _classify_era(r["cross_date"])
        s["eras"][era]["total"] += 1
        if r["is_hit"]:
            s["eras"][era]["hits"] += 1
    
    return dict(stats)


def generate_report():
    """Generate enhanced weekly report: summary, candidates, degradation, era check, full CSV."""
    report_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info("=" * 60)
    logger.info("UNIVERSE SCANNER WEEKLY REPORT")
    logger.info(f"Date: {report_date}")
    logger.info("=" * 60)
    
    supabase = get_supabase()
    md = []
    
    # ============================================================
    # COLLECT ALL DATA
    # ============================================================
    
    # --- All evaluated crossings ---
    logger.info("  Loading evaluated crossings...")
    all_evaluated = _load_all_evaluated(supabase)
    ticker_stats = _build_ticker_stats(all_evaluated) if all_evaluated else {}
    
    total_eval_crossings = len(all_evaluated)
    total_hits = sum(1 for r in all_evaluated if r["is_hit"])
    overall_hit_rate = total_hits / max(total_eval_crossings, 1)
    
    logger.info(f"  {total_eval_crossings} evaluated crossings across {len(ticker_stats)} tickers")
    
    # --- Latest DM snapshot ---
    latest = supabase.table("universe_dm_daily").select("date").order("date", desc=True).limit(1).execute()
    latest_date = latest.data[0]["date"] if latest.data else report_date
    
    # Health counts
    total_count = supabase.table("universe_dm_daily") \
        .select("ticker", count="exact").eq("date", latest_date).execute().count or 0
    above_70_count = supabase.table("universe_dm_daily") \
        .select("ticker", count="exact").eq("date", latest_date).gte("dm_smoothed", 70).execute().count or 0
    above_50_count = supabase.table("universe_dm_daily") \
        .select("ticker", count="exact").eq("date", latest_date).gte("dm_smoothed", 50).execute().count or 0
    below_30_count = supabase.table("universe_dm_daily") \
        .select("ticker", count="exact").eq("date", latest_date).lt("dm_smoothed", 30).execute().count or 0
    
    total_crossings_count = supabase.table("universe_crossings") \
        .select("id", count="exact").execute().count or 0
    evaluated_count = supabase.table("universe_crossings") \
        .select("id", count="exact").eq("evaluated", True).execute().count or 0
    
    # Approaching threshold
    approaching = supabase.table("universe_dm_daily") \
        .select("ticker,dm_smoothed,close") \
        .eq("date", latest_date) \
        .gte("dm_smoothed", 60).lt("dm_smoothed", 70) \
        .order("dm_smoothed", desc=True) \
        .execute()
    approaching_count = len(approaching.data) if approaching.data else 0
    
    # Top DM non-preferred
    top_dm = supabase.table("universe_dm_daily") \
        .select("ticker,dm_smoothed,close") \
        .eq("date", latest_date) \
        .gte("dm_smoothed", 70) \
        .order("dm_smoothed", desc=True) \
        .limit(100) \
        .execute()
    
    # ============================================================
    # BUILD CANDIDATES, DEGRADATION, ERA ANALYSIS
    # ============================================================
    
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    def _era_hit_rate(era_dict, era_name):
        e = era_dict.get(era_name, {"total": 0, "hits": 0})
        return e["hits"] / e["total"] if e["total"] > 0 else None
    
    candidates = []
    degraded = []
    full_ranked = []
    inconsistent = []
    
    era_names = list(ERAS.keys())
    
    for ticker, s in ticker_stats.items():
        if s["total"] < 1:
            continue
        
        hit_rate = s["hits"] / s["total"]
        avg_30d = _avg(s["returns_30d"])
        avg_60d = _avg(s["returns_60d"])
        avg_90d = _avg(s["returns_90d"])
        
        era_rates = {}
        for era_name in era_names:
            era_rates[era_name] = _era_hit_rate(s["eras"], era_name)
        
        row = {
            "ticker": ticker, "crossings": s["total"], "hits": s["hits"],
            "hit_rate": hit_rate, "avg_30d": avg_30d, "avg_60d": avg_60d, "avg_90d": avg_90d,
            "era_rates": era_rates, "in_preferred": ticker in PREFERRED_28,
        }
        
        # Full ranked (10+ crossings)
        if s["total"] >= MIN_CROSSINGS_CANDIDATE:
            full_ranked.append(row)
        
        # Candidates: not preferred, 50%+ hit rate, 10+ crossings
        if ticker not in PREFERRED_28 and s["total"] >= MIN_CROSSINGS_CANDIDATE and hit_rate >= CANDIDATE_HIT_RATE:
            candidates.append(row)
        
        # Degradation: preferred, below 40%, 5+ crossings
        if ticker in PREFERRED_28 and s["total"] >= MIN_CROSSINGS_DEGRADATION and hit_rate < DEGRADATION_HIT_RATE:
            degraded.append(row)
        
        # Consistency: era hit rates vary by more than 20 points
        valid_era_rates = [r for r in era_rates.values() if r is not None]
        if len(valid_era_rates) >= 2:
            if max(valid_era_rates) - min(valid_era_rates) > 0.20:
                if s["total"] >= MIN_CROSSINGS_CANDIDATE:
                    inconsistent.append(row)
    
    candidates.sort(key=lambda x: -x["hit_rate"])
    degraded.sort(key=lambda x: x["hit_rate"])
    full_ranked.sort(key=lambda x: -x["hit_rate"])
    inconsistent.sort(key=lambda x: -x["hit_rate"])
    
    # ============================================================
    # BUILD MARKDOWN
    # ============================================================
    
    md.append("# FlowOS Universe Scanner Report")
    md.append(f"**Date:** {latest_date}  ")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    md.append("")
    
    # --- Summary Table ---
    md.append("## Summary")
    md.append("")
    md.append("| Metric | Count | % of Universe |")
    md.append("|--------|------:|:-------------:|")
    md.append(f"| Total tickers scanned | {total_count:,} | 100% |")
    md.append(f"| DM >= 70 (active signals) | {above_70_count:,} | {above_70_count/max(total_count,1)*100:.1f}% |")
    md.append(f"| DM 50-69 (warming) | {above_50_count - above_70_count:,} | {(above_50_count - above_70_count)/max(total_count,1)*100:.1f}% |")
    md.append(f"| DM 60-69 (approaching) | {approaching_count:,} | {approaching_count/max(total_count,1)*100:.1f}% |")
    md.append(f"| DM < 30 (cold) | {below_30_count:,} | {below_30_count/max(total_count,1)*100:.1f}% |")
    md.append("")
    md.append("| Crossings | Count |")
    md.append("|-----------|------:|")
    md.append(f"| Total recorded | {total_crossings_count:,} |")
    md.append(f"| Evaluated | {evaluated_count:,} |")
    md.append(f"| Pending (insufficient forward data) | {total_crossings_count - evaluated_count:,} |")
    if total_eval_crossings > 0:
        md.append(f"| Overall hit rate (10%+ in 90d) | {overall_hit_rate*100:.1f}% |")
    md.append("")
    
    # --- Section 1: New Candidates ---
    md.append("## 1. New Candidates")
    md.append(f"Tickers NOT in Preferred 28 with hit rate >= {CANDIDATE_HIT_RATE*100:.0f}% and {MIN_CROSSINGS_CANDIDATE}+ evaluated crossings.")
    md.append("")
    
    if candidates:
        md.append(f"**{len(candidates)} candidates found.**")
        md.append("")
        md.append("| Ticker | Crossings | Hits | Hit Rate | Avg 30d | Avg 60d | Avg 90d |")
        md.append("|--------|----------:|-----:|---------:|--------:|--------:|--------:|")
        for c in candidates[:30]:
            md.append(f"| {c['ticker']} | {c['crossings']} | {c['hits']} | "
                     f"{c['hit_rate']*100:.0f}% | {c['avg_30d']*100:.1f}% | "
                     f"{c['avg_60d']*100:.1f}% | {c['avg_90d']*100:.1f}% |")
        if len(candidates) > 30:
            md.append(f"| ... | | | | | | +{len(candidates)-30} more (see CSV) |")
    elif ticker_stats:
        md.append("No candidates meeting criteria yet.")
    else:
        md.append("No evaluated crossings yet. Run `evaluate` first.")
    md.append("")
    
    # --- Section 2: Degradation Flags ---
    md.append("## 2. Degradation Flags")
    md.append(f"Preferred 28 tickers with hit rate below {DEGRADATION_HIT_RATE*100:.0f}% and {MIN_CROSSINGS_DEGRADATION}+ crossings.")
    md.append("")
    
    if degraded:
        md.append(f"**{len(degraded)} flags.**")
        md.append("")
        md.append("| Ticker | Crossings | Hits | Hit Rate | Avg 90d | Flag |")
        md.append("|--------|----------:|-----:|---------:|--------:|:----:|")
        for d in degraded:
            md.append(f"| {d['ticker']} | {d['crossings']} | {d['hits']} | "
                     f"{d['hit_rate']*100:.0f}% | {d['avg_90d']*100:.1f}% | :warning: |")
    elif ticker_stats:
        md.append("No degradation flags. All Preferred 28 tickers above threshold.")
    else:
        md.append("No evaluated crossings yet.")
    md.append("")
    
    # --- Section 3: Era Consistency Check ---
    md.append("## 3. Era Consistency Check")
    md.append("Tickers where hit rate varies more than 20 points across eras (inconsistent signal).")
    md.append("")
    era_short = ["Pre-COVID", "COVID+Rec", "AI Cycle"]
    
    if inconsistent:
        md.append(f"**{len(inconsistent)} inconsistent tickers.**")
        md.append("")
        header = "| Ticker | Overall | " + " | ".join(era_short) + " | Spread | In Pref? |"
        sep = "|--------|--------:|" + "|".join(["--------:" for _ in era_short]) + "|-------:|:--------:|"
        md.append(header)
        md.append(sep)
        for row in inconsistent[:30]:
            era_vals = []
            valid_rates = []
            for en in era_names:
                r = row["era_rates"].get(en)
                if r is not None:
                    era_vals.append(f"{r*100:.0f}%")
                    valid_rates.append(r)
                else:
                    era_vals.append("--")
            spread = (max(valid_rates) - min(valid_rates)) * 100 if valid_rates else 0
            pref = "Yes" if row["in_preferred"] else ""
            md.append(f"| {row['ticker']} | {row['hit_rate']*100:.0f}% | "
                     + " | ".join(era_vals)
                     + f" | {spread:.0f}pp | {pref} |")
    elif ticker_stats:
        md.append("All tickers with sufficient data show consistent hit rates across eras.")
    else:
        md.append("No evaluated crossings yet.")
    md.append("")
    
    # --- Section 4: Full Ranked List ---
    md.append("## 4. Full Ranked List")
    md.append(f"All {len(full_ranked)} tickers with {MIN_CROSSINGS_CANDIDATE}+ evaluated crossings, sorted by hit rate.")
    md.append("")
    
    if full_ranked:
        header = "| Rank | Ticker | Crossings | Hits | Hit Rate | Avg 30d | Avg 60d | Avg 90d | " + " | ".join(era_short) + " | Pref? |"
        sep = "|-----:|--------|----------:|-----:|---------:|--------:|--------:|--------:|" + "|".join(["--------:" for _ in era_short]) + "|:-----:|"
        md.append(header)
        md.append(sep)
        for i, row in enumerate(full_ranked[:50], 1):
            era_vals = []
            for en in era_names:
                r = row["era_rates"].get(en)
                era_vals.append(f"{r*100:.0f}%" if r is not None else "--")
            pref = "Yes" if row["in_preferred"] else ""
            md.append(f"| {i} | {row['ticker']} | {row['crossings']} | {row['hits']} | "
                     f"{row['hit_rate']*100:.0f}% | {row['avg_30d']*100:.1f}% | "
                     f"{row['avg_60d']*100:.1f}% | {row['avg_90d']*100:.1f}% | "
                     + " | ".join(era_vals)
                     + f" | {pref} |")
        if len(full_ranked) > 50:
            md.append(f"| ... | +{len(full_ranked)-50} more (see CSV) | | | | | | | | | | |")
    md.append("")
    
    # --- Approaching + Top 20 Discovery ---
    md.append("## 5. Approaching Threshold (DM 60-69)")
    md.append(f"{approaching_count} tickers as of {latest_date}. Top 30:")
    md.append("")
    
    if approaching.data:
        md.append("| Ticker | DM | Price | Note |")
        md.append("|--------|---:|------:|------|")
        for r in approaching.data[:30]:
            note = "Preferred" if r["ticker"] in PREFERRED_28 else ""
            md.append(f"| {r['ticker']} | {float(r['dm_smoothed']):.1f} | ${float(r['close']):.2f} | {note} |")
        if approaching_count > 30:
            md.append(f"| ... | | | +{approaching_count - 30} more |")
    md.append("")
    
    md.append("## 6. Top 20 DM Tickers (Not in Preferred 28)")
    md.append("")
    if top_dm.data:
        md.append("| Rank | Ticker | DM | Price |")
        md.append("|-----:|--------|---:|------:|")
        rank = 0
        for r in top_dm.data:
            if r["ticker"] not in PREFERRED_28:
                rank += 1
                md.append(f"| {rank} | {r['ticker']} | {float(r['dm_smoothed']):.1f} | ${float(r['close']):.2f} |")
                if rank >= 20:
                    break
    md.append("")
    
    md.append("---")
    md.append(f"*FlowOS Universe Scanner v2.0 | Threshold: {CROSSING_THRESHOLD} | "
              f"Hit: {HIT_RETURN_THRESHOLD*100:.0f}%+ in {OUTCOME_WINDOW_DAYS}d | "
              f"Candidate: {CANDIDATE_HIT_RATE*100:.0f}%/{MIN_CROSSINGS_CANDIDATE}+ | "
              f"Degradation: <{DEGRADATION_HIT_RATE*100:.0f}%/{MIN_CROSSINGS_DEGRADATION}+*")
    
    # ============================================================
    # WRITE FILES
    # ============================================================
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # --- Markdown report ---
    md_filename = f"universe_report_{latest_date}.md"
    md_filepath = os.path.join(reports_dir, md_filename)
    with open(md_filepath, "w") as f:
        f.write("\n".join(md))
    logger.info(f"  Report saved: {md_filepath}")
    
    # --- CSV export (full ranked list) ---
    if full_ranked:
        import csv
        csv_filename = f"universe_candidates_ranked_{latest_date}.csv"
        csv_filepath = os.path.join(reports_dir, csv_filename)
        
        fieldnames = ["rank", "ticker", "crossings", "hits", "hit_rate",
                       "avg_return_30d", "avg_return_60d", "avg_return_90d",
                       "in_preferred"]
        for en in era_names:
            fieldnames.append(f"hr_{en[:8].lower().replace(' ', '_')}")
        
        with open(csv_filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(full_ranked, 1):
                csv_row = {
                    "rank": i,
                    "ticker": row["ticker"],
                    "crossings": row["crossings"],
                    "hits": row["hits"],
                    "hit_rate": round(row["hit_rate"], 4),
                    "avg_return_30d": round(row["avg_30d"], 4),
                    "avg_return_60d": round(row["avg_60d"], 4),
                    "avg_return_90d": round(row["avg_90d"], 4),
                    "in_preferred": row["in_preferred"],
                }
                for en in era_names:
                    key = f"hr_{en[:8].lower().replace(' ', '_')}"
                    r = row["era_rates"].get(en)
                    csv_row[key] = round(r, 4) if r is not None else ""
                writer.writerow(csv_row)
        
        logger.info(f"  CSV saved: {csv_filepath}")
    
    # ============================================================
    # CONSOLE SUMMARY
    # ============================================================
    
    logger.info("")
    logger.info(f"  SUMMARY: {total_count} tickers | {above_70_count} active (DM>=70) | "
                f"{evaluated_count}/{total_crossings_count} crossings evaluated | "
                f"overall hit rate: {overall_hit_rate*100:.1f}%")
    
    if candidates:
        logger.info(f"  CANDIDATES: {len(candidates)} new (top 5):")
        for c in candidates[:5]:
            logger.info(f"    {c['ticker']:6s}  {c['hits']}/{c['crossings']}  "
                       f"hit: {c['hit_rate']*100:.0f}%  avg90d: {c['avg_90d']*100:.1f}%")
    
    if degraded:
        logger.info(f"  DEGRADATION: {len(degraded)} flags:")
        for d in degraded:
            logger.info(f"    {d['ticker']:6s}  {d['hits']}/{d['crossings']}  "
                       f"hit: {d['hit_rate']*100:.0f}%  *** REVIEW ***")
    
    if inconsistent:
        logger.info(f"  INCONSISTENT: {len(inconsistent)} tickers with >20pp era spread")
    
    if not ticker_stats:
        logger.info("  No evaluated crossings yet. Run 'evaluate' first.")
    
    return md_filepath


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowOS Universe Scanner - Crossings")
    parser.add_argument("command", choices=["detect", "evaluate", "report", "all"],
                        help="detect: find crossings | evaluate: calculate outcomes | report: weekly report | all: run everything")
    
    args = parser.parse_args()
    
    if args.command == "detect":
        detect_crossings()
    elif args.command == "evaluate":
        evaluate_outcomes()
    elif args.command == "report":
        generate_report()
    elif args.command == "all":
        detect_crossings()
        evaluate_outcomes()
        generate_report()