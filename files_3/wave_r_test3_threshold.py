"""
WAVE R TEST 3 — THRESHOLD SENSITIVITY
Y2AI Research | April 2026

Is 0.05 the right HMS threshold? Tests three values:
  0.03 — tighter (HMS must be very flat to qualify as Wave R)
  0.05 — current (confirmed fix from original bug)
  0.08 — looser  (more events classified as Wave R)

If effect size peaks at 0.03: signal is sharpest when HMS is truly flat.
If effect size peaks at 0.08: the 0.05 threshold is too tight.
If all three are similar: threshold choice is robust.

INPUTS:  wave_r_events.csv, wave_r_performance.csv
OUTPUTS: wave_r_threshold_sensitivity.csv, printed summary

RUN:     python wave_r_test3_threshold.py
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading files...")
events = pd.read_csv("wave_r_events.csv")
perf   = pd.read_csv("wave_r_performance.csv")
print(f"  Events: {len(events):,} | Performance rows: {len(perf):,}\n")

# ── TEST THREE THRESHOLDS ─────────────────────────────────────────────────────
THRESHOLDS = [0.03, 0.05, 0.08]
WINDOWS    = [30, 60, 90]
results    = []

for threshold in THRESHOLDS:
    print(f"Testing HMS threshold = {threshold}...")

    def reclassify(hms_chg, t=threshold):
        if pd.isna(hms_chg):      return "UNKNOWN"
        if hms_chg > t:            return "WAVE_R_STRONG"
        if abs(hms_chg) <= t:      return "WAVE_R"
        if hms_chg <= -t:          return "GENUINE_EXIT"
        return "AMBIGUOUS"

    ev = events.copy()
    ev["classification"] = ev["hms_chg5"].apply(reclassify)

    counts = ev["classification"].value_counts()
    wr_n  = counts.get("WAVE_R", 0) + counts.get("WAVE_R_STRONG", 0)
    ge_n  = counts.get("GENUINE_EXIT", 0)

    pf = perf.drop(columns=["classification"], errors="ignore").merge(
        ev[["date", "ticker", "classification"]],
        on=["date", "ticker"], how="left"
    )

    wave_r  = pf[pf["classification"].isin(["WAVE_R", "WAVE_R_STRONG"])]
    genuine = pf[pf["classification"] == "GENUINE_EXIT"]

    row = {
        "threshold": threshold,
        "wr_events": int(wr_n),
        "ge_events": int(ge_n),
        "wr_pct":    round(wr_n / len(ev) * 100, 1),
        "ge_pct":    round(ge_n / len(ev) * 100, 1),
    }

    print(f"  Wave R: {wr_n:,} ({row['wr_pct']}%) | Genuine: {ge_n:,} ({row['ge_pct']}%)")

    for w in WINDOWS:
        col  = f"dm_recovery_{w}d"
        wr_v = wave_r[col].dropna()
        ge_v = genuine[col].dropna()

        if len(wr_v) < 10 or len(ge_v) < 10:
            row[f"wr_mean_{w}d"]  = None
            row[f"ge_mean_{w}d"]  = None
            row[f"diff_{w}d"]     = None
            row[f"effect_{w}d"]   = None
            row[f"p_{w}d"]        = None
            row[f"sig_{w}d"]      = None
            continue

        t_stat, p = stats.ttest_ind(wr_v, ge_v)
        effect = (wr_v.mean() - ge_v.mean()) / (pd.concat([wr_v, ge_v]).std() + 1e-9)

        row[f"wr_mean_{w}d"] = round(wr_v.mean(), 2)
        row[f"ge_mean_{w}d"] = round(ge_v.mean(), 2)
        row[f"diff_{w}d"]    = round(wr_v.mean() - ge_v.mean(), 2)
        row[f"effect_{w}d"]  = round(effect, 4)
        row[f"p_{w}d"]       = round(p, 6)
        row[f"sig_{w}d"]     = p < 0.05

        print(f"    {w}d: WR={wr_v.mean():+.2f} GE={ge_v.mean():+.2f} "
              f"diff={wr_v.mean()-ge_v.mean():+.2f} effect={effect:+.4f} p={p:.6f}")

    results.append(row)
    print()

df = pd.DataFrame(results)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("═" * 70)
print("THRESHOLD SENSITIVITY SUMMARY")
print("═" * 70)
print(f"\n{'Threshold':>10} | {'WR events':>10} | {'GE events':>10} | "
      f"{'30d diff':>9} | {'60d diff':>9} | {'90d diff':>9} | {'Best effect':>11}")
print("-" * 90)

for _, r in df.iterrows():
    effects = [r.get(f"effect_{w}d") for w in WINDOWS if r.get(f"effect_{w}d") is not None]
    best_effect = max(effects) if effects else None
    print(f"  {r['threshold']:>8} | {r['wr_events']:>10,} | {r['ge_events']:>10,} | "
          f"{r.get('diff_30d', 'N/A'):>9} | "
          f"{r.get('diff_60d', 'N/A'):>9} | "
          f"{r.get('diff_90d', 'N/A'):>9} | "
          f"{best_effect:>11.4f}" if best_effect else "          N/A")

print()

# Interpretation
best_effect_by_threshold = {}
for _, r in df.iterrows():
    effects = [r.get(f"effect_{w}d") for w in WINDOWS if r.get(f"effect_{w}d") is not None]
    best_effect_by_threshold[r["threshold"]] = max(effects) if effects else 0

best_threshold = max(best_effect_by_threshold, key=best_effect_by_threshold.get)

print("INTERPRETATION:")
if best_threshold == 0.03:
    print("  Peak effect at threshold 0.03 — signal is sharpest when HMS is truly flat.")
    print("  RECOMMENDATION: Tighten HMS_STABLE to 0.03 in production filter.")
elif best_threshold == 0.05:
    print("  Peak effect at threshold 0.05 — current threshold is optimal.")
    print("  RECOMMENDATION: Keep HMS_STABLE at 0.05. Choice is validated.")
elif best_threshold == 0.08:
    print("  Peak effect at threshold 0.08 — current threshold may be too tight.")
    print("  RECOMMENDATION: Widen HMS_STABLE to 0.08 to capture more Wave R events.")

# Check if all similar (robust)
diffs = list(best_effect_by_threshold.values())
if max(diffs) - min(diffs) < 0.01:
    print("\n  NOTE: Effect sizes are similar across all thresholds — result is ROBUST.")
    print("  The Wave R finding does not depend sensitively on the HMS threshold choice.")

df.to_csv("wave_r_threshold_sensitivity.csv", index=False)
print(f"\nSaved: wave_r_threshold_sensitivity.csv")
