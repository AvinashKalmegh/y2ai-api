"""
Re-classify the existing 120,070 events with the corrected HMS threshold
(HMS is 0-1 scale, not 0-100) and rerun the Wave R vs Genuine Exit stats.

Uses the already-saved CSVs from the last full run — no Supabase calls.
"""
import pandas as pd
import numpy as np
from scipy import stats

HMS_STABLE  = 0.05
HMS_DECLINE = 0.05

perf   = pd.read_csv("wave_r_performance.csv")
events = pd.read_csv("wave_r_events.csv")

def reclassify(hms_chg):
    if pd.isna(hms_chg):
        return "UNKNOWN"
    if hms_chg > HMS_STABLE:
        return "WAVE_R_STRONG"
    if abs(hms_chg) <= HMS_STABLE:
        return "WAVE_R"
    if hms_chg <= -HMS_DECLINE:
        return "GENUINE_EXIT"
    return "AMBIGUOUS"

events["classification_fixed"] = events["hms_chg5"].apply(reclassify)

print("Reclassification breakdown:")
for cls, n in events["classification_fixed"].value_counts().items():
    print(f"  {cls:20}: {n:7,} ({n/len(events)*100:.1f}%)")
print()

perf_fixed = perf.drop(columns=["classification"]).merge(
    events[["date","ticker","classification_fixed"]].rename(
        columns={"classification_fixed":"classification"}),
    on=["date","ticker"], how="left",
)

wave_r  = perf_fixed[perf_fixed["classification"].isin(["WAVE_R","WAVE_R_STRONG"])]
genuine = perf_fixed[perf_fixed["classification"] == "GENUINE_EXIT"]

print(f"Wave R: {len(wave_r):,} | Genuine Exit: {len(genuine):,}\n")

for w in [30, 60, 90]:
    col = f"dm_recovery_{w}d"
    wr = wave_r[col].dropna()
    ge = genuine[col].dropna()
    if len(wr) < 10 or len(ge) < 10:
        print(f"{w}d: insufficient data (wr={len(wr)}, ge={len(ge)})")
        continue
    t, p = stats.ttest_ind(wr, ge)
    effect = (wr.mean() - ge.mean()) / (pd.concat([wr, ge]).std() + 1e-9)
    direction = "WR>GE" if wr.mean() > ge.mean() else "WR<GE"
    sig = "SIGNIFICANT" if p < 0.05 else "not sig"
    print(f"{w}d:  WR avg={wr.mean():+.2f} (n={len(wr):,})  "
          f"GE avg={ge.mean():+.2f} (n={len(ge):,})  "
          f"p={p:.4f}  {sig}  effect={effect:+.3f}  {direction}")

print("\nMulti-factor confirmed (wave_r_confirmed) within each class:")
for cls, sub in [("Wave R", wave_r), ("Genuine Exit", genuine)]:
    conf = sub[sub["wave_r_confirmed"] == True]
    single = sub[sub["wave_r_confirmed"] != True]
    print(f"\n  {cls}: confirmed n={len(conf):,} | single n={len(single):,}")
    for w in [30, 60, 90]:
        c = conf[f"dm_recovery_{w}d"].dropna()
        s = single[f"dm_recovery_{w}d"].dropna()
        if len(c) > 5 and len(s) > 5:
            print(f"    {w}d: confirmed={c.mean():+.2f}  single={s.mean():+.2f}  "
                  f"diff={c.mean()-s.mean():+.2f}")

perf_fixed.to_csv("wave_r_performance_fixed.csv", index=False)
print("\nSaved: wave_r_performance_fixed.csv")
