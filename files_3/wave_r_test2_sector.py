"""
WAVE R TEST 2 — SECTOR BREAKDOWN
Y2AI Research | April 2026

Does Wave R appear more strongly in certain sectors?
Tests the hypothesis that concentrated institutional positions
(utilities, semis, energy) produce stronger Wave R signals.

INPUTS:  wave_r_events.csv, wave_r_performance.csv, ETF_Reference.csv
         (export ETF_Reference from Supabase: Ticker, Sector_Name, Current_ETF)
OUTPUTS: wave_r_sector_breakdown.csv, wave_r_sector_summary.txt

RUN:     python wave_r_test2_sector.py
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── CONFIG ────────────────────────────────────────────────────────────────────
HMS_STABLE  = 0.05
HMS_DECLINE = 0.05
MIN_EVENTS  = 100   # minimum events per sector to report

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading files...")
events = pd.read_csv("wave_r_events.csv")
perf   = pd.read_csv("wave_r_performance.csv")

# Load ETF_Reference — export from Supabase as CSV
# Columns needed: Ticker, Sector_Name, Current_ETF
try:
    etf_ref = pd.read_csv("ETF_Reference.csv")
    etf_ref = etf_ref.rename(columns={"Ticker": "ticker"})
    has_etf_ref = True
    print(f"  ETF_Reference loaded: {len(etf_ref):,} tickers")
except FileNotFoundError:
    print("  ⚠ ETF_Reference.csv not found — will use Current_ETF from events if available")
    has_etf_ref = False

# ── RECLASSIFY WITH CORRECT HMS THRESHOLD ─────────────────────────────────────
def reclassify(hms_chg):
    if pd.isna(hms_chg):    return "UNKNOWN"
    if hms_chg > HMS_STABLE: return "WAVE_R_STRONG"
    if abs(hms_chg) <= HMS_STABLE: return "WAVE_R"
    if hms_chg <= -HMS_DECLINE: return "GENUINE_EXIT"
    return "AMBIGUOUS"

events["classification"] = events["hms_chg5"].apply(reclassify)

# ── MERGE SECTOR INFO ─────────────────────────────────────────────────────────
if has_etf_ref:
    events = events.merge(
        etf_ref[["ticker", "Sector_Name", "Current_ETF"]],
        on="ticker", how="left"
    )
    sector_col = "Sector_Name"
    etf_col    = "Current_ETF"
else:
    # Fall back to ticker-level if no ETF_Reference
    print("  Running ticker-level analysis only (no sector mapping)")
    sector_col = "ticker"
    etf_col    = "ticker"

# Merge classification back into performance
perf_fixed = perf.drop(columns=["classification"], errors="ignore").merge(
    events[["date", "ticker", "classification", sector_col, etf_col]],
    on=["date", "ticker"], how="left"
)

# ── SECTOR-LEVEL ANALYSIS ─────────────────────────────────────────────────────
print("\nRunning sector breakdown...\n")

sectors = perf_fixed[sector_col].dropna().unique()
results = []

for sector in sorted(sectors):
    sub = perf_fixed[perf_fixed[sector_col] == sector]
    wr  = sub[sub["classification"].isin(["WAVE_R", "WAVE_R_STRONG"])]
    ge  = sub[sub["classification"] == "GENUINE_EXIT"]

    if len(wr) < MIN_EVENTS or len(ge) < MIN_EVENTS:
        continue

    row = {"sector": sector, "wr_n": len(wr), "ge_n": len(ge)}

    for w in [30, 60, 90]:
        col = f"dm_recovery_{w}d"
        wr_v = wr[col].dropna()
        ge_v = ge[col].dropna()

        if len(wr_v) < 10 or len(ge_v) < 10:
            row[f"wr_{w}d"] = None
            row[f"ge_{w}d"] = None
            row[f"diff_{w}d"] = None
            row[f"p_{w}d"] = None
            row[f"sig_{w}d"] = None
            continue

        t, p = stats.ttest_ind(wr_v, ge_v)
        diff = wr_v.mean() - ge_v.mean()
        row[f"wr_{w}d"]   = round(wr_v.mean(), 2)
        row[f"ge_{w}d"]   = round(ge_v.mean(), 2)
        row[f"diff_{w}d"] = round(diff, 2)
        row[f"p_{w}d"]    = round(p, 4)
        row[f"sig_{w}d"]  = "✓" if p < 0.05 and diff > 0 else ("✗" if p < 0.05 else "~")

    results.append(row)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("diff_30d", ascending=False)

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("═" * 80)
print("WAVE R — SECTOR BREAKDOWN")
print(f"{'Sector':25} {'WR_n':>7} {'GE_n':>7} "
      f"{'30d diff':>9} {'sig':>4} "
      f"{'60d diff':>9} {'sig':>4} "
      f"{'90d diff':>9} {'sig':>4}")
print("─" * 80)

lines = []
for _, r in df_results.iterrows():
    line = (f"{str(r['sector']):25} {int(r['wr_n']):>7,} {int(r['ge_n']):>7,} "
            f"{r.get('diff_30d', 'N/A'):>9} {r.get('sig_30d', ''):>4} "
            f"{r.get('diff_60d', 'N/A'):>9} {r.get('sig_60d', ''):>4} "
            f"{r.get('diff_90d', 'N/A'):>9} {r.get('sig_90d', ''):>4}")
    print(line)
    lines.append(line)

print("═" * 80)

# Top 5 sectors where Wave R is strongest
top5 = df_results[df_results["sig_30d"] == "✓"].head(5)
print(f"\nTop sectors where Wave R is strongest (30d, significant):")
for _, r in top5.iterrows():
    print(f"  {r['sector']:25}: WR={r['wr_30d']:+.2f} vs GE={r['ge_30d']:+.2f} "
          f"diff={r['diff_30d']:+.2f} p={r['p_30d']:.4f}")

# Bottom sectors — Wave R weakest or reversed
weak = df_results[df_results["diff_30d"] < 0].head(3)
if len(weak) > 0:
    print(f"\nSectors where Wave R is WEAKEST or reversed (30d):")
    for _, r in weak.iterrows():
        print(f"  {r['sector']:25}: WR={r['wr_30d']:+.2f} vs GE={r['ge_30d']:+.2f} "
              f"diff={r['diff_30d']:+.2f}")

# Save
df_results.to_csv("wave_r_sector_breakdown.csv", index=False)

# Write summary
with open("wave_r_sector_summary.txt", "w") as f:
    f.write("WAVE R — SECTOR BREAKDOWN SUMMARY\n")
    f.write(f"Y2AI Research | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sectors analyzed: {len(df_results)}\n")
    f.write(f"Min events per sector: {MIN_EVENTS}\n\n")
    f.write("IMPLICATION FOR HOLLOWING SHORT SCANNER:\n")
    f.write("  Apply Wave R filter most aggressively in sectors where diff > 2.0\n")
    f.write("  and p < 0.05. These sectors have the most Wave R contamination.\n\n")
    f.write("SECTOR TABLE:\n")
    for line in lines:
        f.write(line + "\n")

print(f"\nSaved: wave_r_sector_breakdown.csv")
print(f"Saved: wave_r_sector_summary.txt")
