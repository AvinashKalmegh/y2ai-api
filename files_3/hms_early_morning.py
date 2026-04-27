"""
ARGUS — HMS EARLY WARNING BACKTEST — TESTS A / B / C (2022-2026)
Y2AI Research | April 2026

Runs three tests on the already-generated entry signals from
hms_time_consuming.py:

    Test A — Original: HMS >= 0.40 at entry vs HMS < 0.40
    Test B — Early vs Mature: HMS < 0.30 vs HMS >= 0.45
    Test C — Rising HMS in mid-accumulation band: HMS rising over 20d AND
             0.30-0.45 vs HMS >= 0.45

Reads hms_backtest_results.csv (already generated, 64k signals over 2022-2026)
so there's no Supabase re-pull. Runs in seconds.

OUTPUTS:
    hms_backtest_abc_results.csv
    hms_backtest_abc_summary.txt
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from pathlib import Path

FORWARD_WINDOWS = [30, 60, 90]
MIN_SIGNALS     = 20

print("═" * 70)
print("ARGUS — HMS EARLY WARNING — TESTS A / B / C (2022-2026)")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("═" * 70 + "\n")

src = Path("hms_backtest_results.csv")
if not src.exists():
    print(f"ERROR: {src} not found. Run hms_time_consuming.py first.")
    raise SystemExit(1)

print(f"Loading {src}...")
df = pd.read_csv(src)
df["date"] = pd.to_datetime(df["date"])
print(f"  {len(df):,} signals | {df['ticker'].nunique()} tickers | "
      f"{df['date'].min().date()} → {df['date'].max().date()}")

df_valid = df[df["hms_at_entry"].notna()].copy()
print(f"  With HMS at entry: {len(df_valid):,}\n")


def tt_row(label, group_a, group_b, name_a="A", name_b="B", sig_when="A>B"):
    """Print t-test between two groups across 30/60/90d, return dict of stats."""
    out = {}
    print(f"  {label}  ({name_a} n={len(group_a):,} | {name_b} n={len(group_b):,})")
    for w in FORWARD_WINDOWS:
        col = f"ret_{w}d"
        a = group_a[col].dropna()
        b = group_b[col].dropna()
        if len(a) < MIN_SIGNALS or len(b) < MIN_SIGNALS:
            print(f"    {w}d: insufficient data ({name_a}={len(a)}, {name_b}={len(b)})")
            continue
        t, p = stats.ttest_ind(a, b)
        diff = a.mean() - b.mean()
        sig  = "✓ SIG" if p < 0.05 else "~ ns"
        if sig_when == "A>B":
            direct = f"{name_a} > {name_b} ✓" if diff > 0 else f"{name_b} > {name_a} ✗"
        else:
            direct = f"{name_a} > {name_b}" if diff > 0 else f"{name_b} > {name_a}"
        print(f"    {w}d: {name_a}={a.mean():+.3f}% (n={len(a):,})  "
              f"{name_b}={b.mean():+.3f}% (n={len(b):,})  "
              f"diff={diff:+.3f}  p={p:.4f}  {sig}  {direct}")
        out[w] = {
            "a_mean": round(a.mean(), 3), "a_n": len(a),
            "b_mean": round(b.mean(), 3), "b_n": len(b),
            "diff":   round(diff, 3),
            "p":      round(p, 4),
            "sig":    bool(p < 0.05),
        }
    return out


# ── TEST A ────────────────────────────────────────────────────────────────────
print("═" * 70)
print("TEST A — HMS >= 0.40 vs HMS < 0.40 at entry")
print("═" * 70)

high = df_valid[df_valid["hms_at_entry"] >= 0.40]
low  = df_valid[df_valid["hms_at_entry"] <  0.40]
test_a = tt_row("Test A", high, low, name_a="High", name_b="Low")

# ── TEST B ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("TEST B — Early (HMS < 0.30) vs Mature (HMS >= 0.45)")
print("  (hypothesis: early entries = more runway = better 60-90d returns)")
print("═" * 70)

early  = df_valid[df_valid["hms_at_entry"] <  0.30]
mature = df_valid[df_valid["hms_at_entry"] >= 0.45]
mid    = df_valid[(df_valid["hms_at_entry"] >= 0.30) & (df_valid["hms_at_entry"] < 0.45)]
print(f"  Early (HMS<0.30):     {len(early):,}")
print(f"  Mid   (0.30-0.45):    {len(mid):,}")
print(f"  Mature (HMS>=0.45):   {len(mature):,}")
test_b = tt_row("Test B", early, mature, name_a="Early", name_b="Mature")

# Also show mid as reference
for w in FORWARD_WINDOWS:
    col = f"ret_{w}d"
    m = mid[col].dropna()
    if len(m) >= MIN_SIGNALS:
        print(f"    {w}d REFERENCE — Mid band: {m.mean():+.3f}% (n={len(m):,})")

# ── TEST C ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("TEST C — Rising HMS in mid-band (rising 20d AND 0.30-0.45) vs Mature")
print("═" * 70)

df_valid["hms_rising_20d"] = (
    df_valid["hms_at_entry"] > df_valid["hms_20d_prior"]
) & df_valid["hms_20d_prior"].notna()

mid_rising = df_valid[
    (df_valid["hms_rising_20d"]) &
    (df_valid["hms_at_entry"] >= 0.30) &
    (df_valid["hms_at_entry"] <  0.45)
]
print(f"  Mid-Rising (0.30-0.45, rising 20d): {len(mid_rising):,}")
print(f"  Mature (HMS>=0.45):                 {len(mature):,}")
test_c = tt_row("Test C", mid_rising, mature, name_a="Mid-Rising", name_b="Mature")

# ── BY YEAR (Test B, 30d) ────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("BY YEAR — Early vs Mature at 30d")
print("═" * 70)

df_valid["year"] = df_valid["date"].dt.year
for year in sorted(df_valid["year"].unique()):
    sub = df_valid[df_valid["year"] == year]
    e = sub[sub["hms_at_entry"] <  0.30]["ret_30d"].dropna()
    m = sub[sub["hms_at_entry"] >= 0.45]["ret_30d"].dropna()
    if len(e) < 5 or len(m) < 5:
        continue
    diff = e.mean() - m.mean()
    direction = "✓" if diff > 0 else "✗"
    print(f"  {int(year)}: Early={e.mean():+.2f}% (n={len(e):,})  "
          f"Mature={m.mean():+.2f}% (n={len(m):,})  "
          f"diff={diff:+.2f}  {direction}")

# ── LEAD TIME COMPOSITION ─────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("HMS LEAD TIME COMPOSITION")
print("═" * 70)
for look_back, col in [(5, "hms_5d_prior"), (10, "hms_10d_prior"), (20, "hms_20d_prior")]:
    sub = df_valid[df_valid[col].notna()]
    if len(sub) < 10:
        continue
    avg   = sub[col].mean()
    pct30 = (sub[col] <  0.30).mean() * 100
    pct40 = (sub[col] >= 0.40).mean() * 100
    print(f"  {look_back:2d}d before entry (n={len(sub):,}): "
          f"avg={avg:.3f}  <0.30: {pct30:.0f}%  >=0.40: {pct40:.0f}%")

# ── VERDICT ──────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("VERDICT")
print("═" * 70)

a30 = test_a.get(30, {}).get("diff")
a90 = test_a.get(90, {}).get("diff")
b30 = test_b.get(30, {}).get("diff")
b90 = test_b.get(90, {}).get("diff")
c30 = test_c.get(30, {}).get("diff")
c90 = test_c.get(90, {}).get("diff")

print(f"\n  Test A (High vs Low):           30d {a30:+.2f} / 90d {a90:+.2f}"
      if a30 is not None else "  Test A: insufficient data")
print(f"  Test B (Early vs Mature):       30d {b30:+.2f} / 90d {b90:+.2f}"
      if b30 is not None else "  Test B: insufficient data")
print(f"  Test C (Mid-Rising vs Mature):  30d {c30:+.2f} / 90d {c90:+.2f}"
      if c30 is not None else "  Test C: insufficient data")

print("""
READING:
  If Test A is negative at 90d AND Test B is positive at 90d:
    → HMS elevated at entry marks LATE accumulation; filter HURTS long holds.
    → Early (low HMS) entries have the most 60-90d runway.
    → Correct use of HMS is a PRE-ENTRY scanner (HMS rising + DM not yet at 65).
""")

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_rows = []
for name, res in [("A_high_vs_low", test_a),
                  ("B_early_vs_mature", test_b),
                  ("C_midrising_vs_mature", test_c)]:
    for w, r in res.items():
        out_rows.append({"test": name, "window_d": w, **r})
pd.DataFrame(out_rows).to_csv("hms_backtest_abc_results.csv", index=False)

with open("hms_backtest_abc_summary.txt", "w", encoding="utf-8") as f:
    f.write("ARGUS — HMS EARLY WARNING — TESTS A / B / C (2022-2026)\n")
    f.write(f"Y2AI Research | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Source: hms_backtest_results.csv ({len(df):,} signals)\n\n")
    for name, res in [("A  (HMS >= 0.40  vs  HMS < 0.40)", test_a),
                      ("B  (HMS < 0.30   vs  HMS >= 0.45)", test_b),
                      ("C  (Mid-Rising   vs  HMS >= 0.45)", test_c)]:
        f.write(f"Test {name}\n")
        for w, r in res.items():
            f.write(f"  {w}d: diff={r['diff']:+.3f} p={r['p']:.4f} "
                    f"({'sig' if r['sig'] else 'ns'})\n")
        f.write("\n")

print("\nSaved: hms_backtest_abc_results.csv")
print("Saved: hms_backtest_abc_summary.txt")
print(f"Done: {datetime.now().strftime('%H:%M:%S')}")
print("═" * 70)
