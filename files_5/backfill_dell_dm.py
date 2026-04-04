"""
Backfill DELL DM scores for the gap period (2022-01-26 to 2024-03-07).
Uses the same DM calculation logic as backfill_spot.py / Dm_historical_pipeline.py.
Outputs: DELL_DM_backfill.csv
"""
import numpy as np
import pandas as pd
import yfinance as yf

# DM formula constants (match pipeline exactly)
W_REL_STR_ETF = 0.50
W_REL_STR_SPY = 0.30
W_VOLUME_Z = 0.20
REL_STR_SCALE = 500
EMA_PERIOD = 5
RETURN_PERIOD = 19

TICKER = "DELL"
ETF = "XLK"

# Need extra lead-in for 20d returns + volume baselines
START = "2021-06-01"
END = "2024-03-08"

def calc_rel_str(ticker_ret, bench_ret):
    diff = ticker_ret - bench_ret
    return max(0, min(100, 50 + diff * REL_STR_SCALE))

def calc_vol_z(vol_5d, vol_baseline):
    if pd.isna(vol_baseline) or vol_baseline <= 0 or pd.isna(vol_5d):
        return 50.0
    ratio = vol_5d / vol_baseline
    return max(0, min(100, (ratio - 0.5) * 66.67))

print("Downloading price data from yfinance...")
dell_raw = yf.download(TICKER, start=START, end=END, progress=False)
spy_raw = yf.download("SPY", start=START, end=END, progress=False)
etf_raw = yf.download(ETF, start=START, end=END, progress=False)

print(f"  {TICKER}: {len(dell_raw)} rows ({dell_raw.index[0].date()} to {dell_raw.index[-1].date()})")
print(f"  SPY:  {len(spy_raw)} rows")
print(f"  {ETF}:  {len(etf_raw)} rows")

# Build dataframes
dell_df = pd.DataFrame({
    "close": dell_raw["Close"].squeeze(),
    "volume": dell_raw["Volume"].squeeze(),
}).dropna()

spy_df = pd.DataFrame({
    "close": spy_raw["Close"].squeeze(),
}).dropna()

etf_df = pd.DataFrame({
    "close": etf_raw["Close"].squeeze(),
    "volume": etf_raw["Volume"].squeeze(),
}).dropna()

# Returns
dell_df["return_20d"] = dell_df["close"].pct_change(RETURN_PERIOD)
spy_df["spy_return_20d"] = spy_df["close"].pct_change(RETURN_PERIOD)
etf_df["etf_return_20d"] = etf_df["close"].pct_change(RETURN_PERIOD)

# DELL volume metrics
dell_df["vol_5d_avg"] = dell_df["volume"].rolling(5).mean()
dell_df["vol_20d_avg"] = dell_df["volume"].rolling(20).mean()
dell_df["vol_ratio"] = dell_df["volume"] / dell_df["vol_20d_avg"]

# Volume baseline (60-cal-day window minus recent 5)
dates = dell_df.index
cutoffs = dates - pd.Timedelta(days=60)
window_starts = dates.searchsorted(cutoffs, side="left")
indices = np.arange(len(dates))
total_in_window = indices + 1 - window_starts
baseline_counts = np.maximum(total_in_window - 5, 1)
vol_arr = dell_df["volume"].values
cs = np.concatenate(([0.0], np.cumsum(vol_arr)))
window_sums = cs[indices + 1] - cs[window_starts]
recent_sums = cs[indices + 1] - cs[np.maximum(indices - 4, 0)]
baseline_sums = window_sums - recent_sums
baseline_avg = baseline_sums / baseline_counts
baseline_avg[:5] = np.nan
baseline_avg[total_in_window < 10] = np.nan
dell_df["vol_baseline_avg"] = baseline_avg

# Merge benchmarks
merged = dell_df.join(spy_df[["spy_return_20d"]], how="left")
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

# DM Raw + Smoothed
merged["dm_raw"] = (
    merged["rel_str_etf"] * W_REL_STR_ETF +
    merged["rel_str_spy"] * W_REL_STR_SPY +
    merged["volume_z"] * W_VOLUME_Z
).clip(0, 100)
merged["dm"] = merged["dm_raw"].ewm(span=EMA_PERIOD, adjust=False).mean().clip(0, 100)

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
merged["lead"] = merged["dm"] - merged["etf_dm"]

# Phase
def classify_phase(dm, etf):
    if dm >= 60 and etf >= 60: return "STRONG"
    if dm >= 40 and etf < 40: return "EARLY"
    if dm >= 40 and 40 <= etf < 60: return "BUILD"
    if dm < 40 and etf >= 60: return "EXHAUST"
    if dm < 40: return "WEAK"
    return "TRANS"

merged["phase"] = merged.apply(lambda r: classify_phase(r["dm"], r["etf_dm"]), axis=1)

# Filter to gap period only and format for the timeline script
gap_start = pd.Timestamp("2022-01-26")
gap_end = pd.Timestamp("2024-03-07")
output = merged[(merged.index >= gap_start) & (merged.index <= gap_end)].copy()
output = output.reset_index()
output = output.rename(columns={"index": "Date"})
output["Ticker"] = TICKER
output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")

# Match DM_2020_2023 column format
out_cols = ["Date", "Ticker", "close", "volume", "return_20d", "etf_return_20d",
            "spy_return_20d", "vol_20d_avg", "vol_ratio", "rel_str_etf", "rel_str_spy",
            "volume_z", "dm_raw", "dm", "etf_dm", "lead", "phase"]
# Rename to match expected format
output = output.rename(columns={
    "close": "Close", "volume": "Volume", "return_20d": "Return_20d",
    "etf_return_20d": "ETF_Return_20d", "spy_return_20d": "SPY_Return_20d",
    "vol_20d_avg": "Vol_20d_Avg", "vol_ratio": "Vol_Ratio",
    "rel_str_etf": "RelStr_ETF", "rel_str_spy": "RelStr_SPY",
    "volume_z": "Volume_Z", "dm_raw": "DM_Raw", "dm": "DM",
    "etf_dm": "ETF_DM", "lead": "Lead", "phase": "Phase",
})

final_cols = ["Date", "Ticker", "Close", "Volume", "Return_20d", "ETF_Return_20d",
              "SPY_Return_20d", "Vol_20d_Avg", "Vol_Ratio", "RelStr_ETF", "RelStr_SPY",
              "Volume_Z", "DM_Raw", "DM", "ETF_DM", "Lead", "Phase"]
output = output[[c for c in final_cols if c in output.columns]]

output.to_csv("DELL_DM_backfill.csv", index=False)
print(f"\nSaved: DELL_DM_backfill.csv ({len(output)} rows)")
print(f"Date range: {output['Date'].iloc[0]} to {output['Date'].iloc[-1]}")

# Show key stats
dm_series = pd.to_numeric(output["DM"])
print(f"DM range: {dm_series.min():.1f} to {dm_series.max():.1f}")
above_65 = output[dm_series >= 65]
if len(above_65) > 0:
    print(f"First DM >= 65: {above_65.iloc[0]['Date']} (DM={above_65.iloc[0]['DM']:.1f})")
else:
    print("No DM >= 65 found in gap period")
