"""
Backfill SPOT DM for Feb 17 only.
Uses the same calculation logic as Dm_historical_pipeline.py.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import pandas as pd
from supabase import create_client

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Constants (match pipeline exactly)
W_REL_STR_ETF = 0.50
W_REL_STR_SPY = 0.30
W_VOLUME_Z = 0.20
REL_STR_SCALE = 500
EMA_PERIOD = 5
RETURN_PERIOD = 19

TICKER = "SPOT"
ETF = "XLC"
TARGET_DATE = "2026-02-17"

def fetch_prices(ticker, limit=120):
    rows = []
    offset = 0
    while True:
        r = sb.table("price_history").select("date,close,volume") \
            .eq("ticker", ticker).order("date", desc=True) \
            .range(offset, offset+limit-1).execute()
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < limit:
            break
        offset += limit
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])
    return df.sort_values("date").set_index("date")

def calc_rel_str(ticker_ret, bench_ret):
    diff = ticker_ret - bench_ret
    return max(0, min(100, 50 + diff * REL_STR_SCALE))

def calc_vol_z(vol_5d, vol_baseline):
    if pd.isna(vol_baseline) or vol_baseline <= 0 or pd.isna(vol_5d):
        return 50.0
    ratio = vol_5d / vol_baseline
    return max(0, min(100, (ratio - 0.5) * 66.67))

print("Fetching prices...")
spot_df = fetch_prices(TICKER)
spy_df = fetch_prices("SPY")
etf_df = fetch_prices(ETF)
print(f"  SPOT: {len(spot_df)} rows, last: {spot_df.index[-1].strftime('%Y-%m-%d')}")
print(f"  SPY:  {len(spy_df)} rows, last: {spy_df.index[-1].strftime('%Y-%m-%d')}")
print(f"  {ETF}:  {len(etf_df)} rows, last: {etf_df.index[-1].strftime('%Y-%m-%d')}")

# Check SPOT has Feb 17 price
if pd.Timestamp(TARGET_DATE) not in spot_df.index:
    print(f"\nERROR: SPOT has no price for {TARGET_DATE}. Cannot calculate DM.")
    exit(1)

# Calculate returns
spot_df["return_20d"] = spot_df["close"].pct_change(RETURN_PERIOD)
spy_df["spy_return_20d"] = spy_df["close"].pct_change(RETURN_PERIOD)
etf_df["etf_return_20d"] = etf_df["close"].pct_change(RETURN_PERIOD)

# Volume metrics for SPOT
spot_df["vol_5d_avg"] = spot_df["volume"].rolling(5).mean()
spot_df["vol_20d_avg"] = spot_df["volume"].rolling(20).mean()
spot_df["vol_ratio"] = spot_df["volume"] / spot_df["vol_20d_avg"]

# Volume baseline (60-cal-day window minus recent 5)
dates = spot_df.index
cutoffs = dates - pd.Timedelta(days=60)
window_starts = dates.searchsorted(cutoffs, side="left")
indices = np.arange(len(dates))
total_in_window = indices + 1 - window_starts
baseline_counts = np.maximum(total_in_window - 5, 1)
vol_arr = spot_df["volume"].values
cs = np.concatenate(([0.0], np.cumsum(vol_arr)))
window_sums = cs[indices + 1] - cs[window_starts]
recent_sums = cs[indices + 1] - cs[np.maximum(indices - 4, 0)]
baseline_sums = window_sums - recent_sums
baseline_avg = baseline_sums / baseline_counts
baseline_avg[:5] = np.nan
baseline_avg[total_in_window < 10] = np.nan
spot_df["vol_baseline_avg"] = baseline_avg

# Merge benchmarks
merged = spot_df.join(spy_df[["spy_return_20d"]], how="left")
merged = merged.join(etf_df[["etf_return_20d"]], how="left")
merged["spy_return_20d"] = merged["spy_return_20d"].ffill()
merged["etf_return_20d"] = merged["etf_return_20d"].ffill()

# DM components
merged["rel_str_etf"] = merged.apply(
    lambda r: calc_rel_str(r["return_20d"], r["etf_return_20d"])
    if pd.notna(r["etf_return_20d"]) and pd.notna(r["return_20d"]) else 50.0, axis=1)
merged["rel_str_spy"] = merged.apply(
    lambda r: calc_rel_str(r["return_20d"], r["spy_return_20d"])
    if pd.notna(r["spy_return_20d"]) and pd.notna(r["return_20d"]) else 50.0, axis=1)
merged["volume_z"] = merged.apply(
    lambda r: calc_vol_z(r["vol_5d_avg"], r["vol_baseline_avg"]), axis=1)

# DM Raw + Smoothed (EMA across full series, then extract target date)
merged["dm_raw"] = (
    merged["rel_str_etf"] * W_REL_STR_ETF +
    merged["rel_str_spy"] * W_REL_STR_SPY +
    merged["volume_z"] * W_VOLUME_Z
).clip(0, 100)
merged["dm_smoothed"] = merged["dm_raw"].ewm(span=EMA_PERIOD, adjust=False).mean().clip(0, 100)

# ETF DM
etf_df["etf_vol_5d"] = etf_df["volume"].rolling(5).mean()
etf_dates = etf_df.index
etf_cutoffs = etf_dates - pd.Timedelta(days=60)
etf_ws = etf_dates.searchsorted(etf_cutoffs, side="left")
etf_idx = np.arange(len(etf_dates))
etf_tiw = etf_idx + 1 - etf_ws
etf_bc = np.maximum(etf_tiw - 5, 1)
etf_va = etf_df["volume"].values
etf_cs = np.concatenate(([0.0], np.cumsum(etf_va)))
etf_wsums = etf_cs[etf_idx + 1] - etf_cs[etf_ws]
etf_rsums = etf_cs[etf_idx + 1] - etf_cs[np.maximum(etf_idx - 4, 0)]
etf_bsums = etf_wsums - etf_rsums
etf_bavg = etf_bsums / etf_bc
etf_bavg[:5] = np.nan
etf_bavg[etf_tiw < 10] = np.nan
etf_df["etf_vol_baseline"] = etf_bavg

etf_df2 = etf_df.join(spy_df[["spy_return_20d"]], how="left")
etf_df2["spy_return_20d"] = etf_df2["spy_return_20d"].ffill()
etf_df2["etf_rel_str_spy"] = etf_df2.apply(
    lambda r: calc_rel_str(r["etf_return_20d"], r["spy_return_20d"])
    if pd.notna(r["etf_return_20d"]) and pd.notna(r["spy_return_20d"]) else 50.0, axis=1)
etf_df2["etf_vol_z"] = etf_df2.apply(
    lambda r: calc_vol_z(r["etf_vol_5d"], r["etf_vol_baseline"]), axis=1)
etf_df2["etf_dm_raw"] = (etf_df2["etf_rel_str_spy"] * 0.70 + etf_df2["etf_vol_z"] * 0.30).clip(0, 100)
etf_df2["etf_dm"] = etf_df2["etf_dm_raw"].ewm(span=EMA_PERIOD, adjust=False).mean().clip(0, 100)

merged = merged.join(etf_df2[["etf_dm"]], how="left")
merged["etf_dm"] = merged["etf_dm"].ffill().fillna(50.0)
merged["lead"] = merged["dm_smoothed"] - merged["etf_dm"]

# Phase
def classify_phase(dm, etf):
    if dm >= 60 and etf >= 60: return "STRONG"
    if dm >= 40 and etf < 40: return "EARLY"
    if dm >= 40 and 40 <= etf < 60: return "BUILD"
    if dm < 40 and etf >= 60: return "EXHAUST"
    if dm < 40: return "WEAK"
    return "TRANS"

merged["phase"] = merged.apply(lambda r: classify_phase(r["dm_smoothed"], r["etf_dm"]), axis=1)

# Extract Feb 17 row
target = pd.Timestamp(TARGET_DATE)
if target not in merged.index:
    print(f"ERROR: {TARGET_DATE} not in calculated series")
    exit(1)

r = merged.loc[target]
row = {
    "ticker": TICKER,
    "date": TARGET_DATE,
    "close": round(float(r["close"]), 2),
    "volume": int(r["volume"]),
    "return_20d": round(float(r["return_20d"]), 6) if pd.notna(r["return_20d"]) else None,
    "etf_return_20d": round(float(r["etf_return_20d"]), 6) if pd.notna(r["etf_return_20d"]) else None,
    "spy_return_20d": round(float(r["spy_return_20d"]), 6) if pd.notna(r["spy_return_20d"]) else None,
    "vol_20d_avg": round(float(r["vol_20d_avg"]), 2) if pd.notna(r["vol_20d_avg"]) else None,
    "vol_ratio": round(float(r["vol_ratio"]), 4) if pd.notna(r["vol_ratio"]) else None,
    "rel_str_etf": round(float(r["rel_str_etf"]), 2),
    "rel_str_spy": round(float(r["rel_str_spy"]), 2),
    "volume_z": round(float(r["volume_z"]), 2),
    "dm_raw": round(float(r["dm_raw"]), 2),
    "dm_smoothed": round(float(r["dm_smoothed"]), 2),
    "etf": ETF,
    "etf_dm": round(float(r["etf_dm"]), 2),
    "lead": round(float(r["lead"]), 2),
    "phase": r["phase"],
}

print(f"\nSPOT Feb 17 DM:")
print(f"  DM Raw:      {row['dm_raw']}")
print(f"  DM Smoothed: {row['dm_smoothed']}")
print(f"  ETF DM:      {row['etf_dm']}")
print(f"  Lead:        {row['lead']}")
print(f"  Phase:       {row['phase']}")
print(f"  Close:       {row['close']}")

# Upsert to dm_history
sb.table("dm_history").upsert([row], on_conflict="ticker,date").execute()
print(f"\nUpserted to dm_history.")

# Update dm_latest
from datetime import datetime
latest_row = dict(row)
# Get 7d ago DM
r7 = sb.table("dm_history").select("dm_smoothed").eq("ticker", TICKER) \
    .lt("date", TARGET_DATE).order("date", desc=True).limit(1).execute()
dm_7d_ago = r7.data[0]["dm_smoothed"] if r7.data else row["dm_smoothed"]
dm_change = round(row["dm_smoothed"] - dm_7d_ago, 2)

latest_row["dm_7d_ago"] = dm_7d_ago
latest_row["dm_change"] = dm_change
latest_row["updated_at"] = datetime.utcnow().isoformat()

sb.table("dm_latest").upsert([latest_row], on_conflict="ticker").execute()
print(f"Updated dm_latest. DM 7d ago: {dm_7d_ago}, Change: {dm_change}")
print("\nDone.")