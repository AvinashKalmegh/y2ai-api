"""
ARGUS — HMS EARLY WARNING BACKTEST — PHASE 2 (2016-2021)
Y2AI Research | April 2026

Real Phase 2 run — pulls dm_history / hms_daily / price_history for 2016-2021,
computes PureSim entry signals, runs Tests A / B / C, and compares against
Phase 1 (2022-2026) results.

Pulls are chunked year-by-year to stay under PostgREST's statement timeout
on deep OFFSET queries.

OUTPUTS:
    hms_backtest_phase2_results.csv
    hms_backtest_phase2_abc_results.csv
    hms_backtest_phase2_summary.txt
    hms_combined_summary.txt
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATE_FROM        = "2016-01-01"
DATE_TO          = "2021-12-31"
DM_ENTRY         = 65.0
EMA_PERIOD       = 5
PRICE_MA_DAYS    = 20
FORWARD_WINDOWS  = [30, 60, 90]
MIN_SIGNALS      = 20
PAGE_SIZE        = 1000

# Phase 1 stats (2022-2026) hardcoded for combined verdict
PHASE1 = {
    "total_signals": 64377,
    "A_diff_30d": 0.213,  "A_p_30d": 0.3071,
    "A_diff_60d": -0.732, "A_p_60d": 0.0081,
    "A_diff_90d": -1.213, "A_p_90d": 0.0000,
    "B_diff_30d": -0.420, "B_p_30d": 0.2278,
    "B_diff_60d":  1.644, "B_p_60d": 0.0002,
    "B_diff_90d":  1.953, "B_p_90d": 0.0000,
    "C_diff_30d": -0.583, "C_p_30d": 0.0475,
    "C_diff_60d":  0.028, "C_p_60d": 0.9411,
    "C_diff_90d":  1.224, "C_p_90d": 0.0016,
}

print("═" * 70)
print("ARGUS — HMS EARLY WARNING — PHASE 2 (2016-2021)")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Data range: {DATE_FROM} → {DATE_TO}")
print("═" * 70 + "\n")


# ── CHUNKED PULL ──────────────────────────────────────────────────────────────
def pull_year(table, select, year, date_col="date"):
    """Pull one calendar year from a table with paging + retries."""
    date_from = f"{year}-01-01"
    date_to   = f"{year}-12-31"
    rows, offset = [], 0
    while True:
        attempt, cur = 0, PAGE_SIZE
        while True:
            try:
                r = (sb.table(table)
                     .select(select)
                     .gte(date_col, date_from)
                     .lte(date_col, date_to)
                     .order(date_col).order("ticker")
                     .range(offset, offset + cur - 1)
                     .execute())
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    raise
                cur = max(100, cur // 2)
                print(f"      [retry {attempt}] {table} {year} page={cur} "
                      f"sleep {2**attempt}s")
                time.sleep(2 ** attempt)
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < cur:
            break
        offset += cur
    return rows


def pull_table_chunked(table, select, date_col="date"):
    years = list(range(int(DATE_FROM[:4]), int(DATE_TO[:4]) + 1))
    print(f"  {table}: pulling {len(years)} year(s)...")
    all_rows = []
    for y in years:
        rows = pull_year(table, select, y, date_col)
        all_rows.extend(rows)
        print(f"    {y}: {len(rows):,} rows | cumulative: {len(all_rows):,}")
    return pd.DataFrame(all_rows)


# ── PULL DATA ─────────────────────────────────────────────────────────────────
print("Pulling DM scores (dm_history.dm_smoothed)...")
dm_raw = pull_table_chunked("dm_history", "date,ticker,dm_smoothed")
dm_raw["date"] = pd.to_datetime(dm_raw["date"])
dm_raw = dm_raw.rename(columns={"dm_smoothed": "dm_score"})
print(f"  TOTAL: {len(dm_raw):,} rows | {dm_raw['ticker'].nunique()} tickers\n")

print("Pulling HMS scores (hms_daily.hms_score)...")
hms_raw = pull_table_chunked("hms_daily", "date,ticker,hms_score")
hms_raw["date"] = pd.to_datetime(hms_raw["date"])
print(f"  TOTAL: {len(hms_raw):,} rows | {hms_raw['ticker'].nunique()} tickers\n")

print("Pulling price data (price_history.close)...")
price_raw = pull_table_chunked("price_history", "date,ticker,close")
price_raw["date"] = pd.to_datetime(price_raw["date"])
print(f"  TOTAL: {len(price_raw):,} rows | {price_raw['ticker'].nunique()} tickers\n")

tickers = sorted(set(dm_raw["ticker"]) & set(hms_raw["ticker"]) & set(price_raw["ticker"]))
print(f"Universe (DM ∩ HMS ∩ Price): {len(tickers)} tickers\n")

# ── PIVOT ─────────────────────────────────────────────────────────────────────
print("Building pivots...")
dm_pivot    = dm_raw.pivot_table(index="date", columns="ticker", values="dm_score")
hms_pivot   = hms_raw.pivot_table(index="date", columns="ticker", values="hms_score")
price_pivot = price_raw.pivot_table(index="date", columns="ticker", values="close")

common = dm_pivot.index.intersection(hms_pivot.index).intersection(price_pivot.index)
dm_pivot    = dm_pivot.loc[common].sort_index()
hms_pivot   = hms_pivot.loc[common].sort_index()
price_pivot = price_pivot.loc[common].sort_index()
dates       = list(dm_pivot.index)
dates_arr   = np.array(dates)
n_dates     = len(dates_arr)
print(f"  Common dates: {n_dates} | {dates[0].date()} → {dates[-1].date()}\n")

# ── SIGNALS ───────────────────────────────────────────────────────────────────
print("Computing EMA5 and Price MA20...")
dm_ema5    = dm_pivot.ewm(span=EMA_PERIOD, adjust=False).mean()
price_ma20 = price_pivot.rolling(PRICE_MA_DAYS).mean()

print("Scanning for entry signals...")
signals = []
max_fwd = max(FORWARD_WINDOWS)
for ticker in tickers:
    if ticker not in dm_pivot.columns:
        continue
    dm_s    = dm_pivot[ticker].values
    ema_s   = dm_ema5[ticker].values
    hms_s   = hms_pivot[ticker].values   if ticker in hms_pivot.columns   else np.full(n_dates, np.nan)
    price_s = price_pivot[ticker].values if ticker in price_pivot.columns else np.full(n_dates, np.nan)
    pma_s   = price_ma20[ticker].values  if ticker in price_pivot.columns else np.full(n_dates, np.nan)

    for i in range(PRICE_MA_DAYS + 1, n_dates - max_fwd):
        dm_now    = dm_s[i]
        ema_now   = ema_s[i]
        ema_prev  = ema_s[i-1]
        price_now = price_s[i]
        pma_now   = pma_s[i]

        if pd.isna(dm_now) or pd.isna(ema_now) or pd.isna(ema_prev) or pd.isna(price_now) or pd.isna(pma_now):
            continue
        if dm_now < DM_ENTRY:
            continue
        if ema_now <= ema_prev:
            continue
        if price_now <= pma_now:
            continue

        hms_now = hms_s[i]

        fwd = {}
        for w in FORWARD_WINDOWS:
            if i + w < n_dates:
                p_fwd = price_s[i + w]
                fwd[f"ret_{w}d"] = (p_fwd - price_now) / price_now * 100 \
                    if not pd.isna(p_fwd) and price_now > 0 else None
            else:
                fwd[f"ret_{w}d"] = None

        hms_5d_prior  = hms_s[i-5]  if i >= 5  else np.nan
        hms_10d_prior = hms_s[i-10] if i >= 10 else np.nan
        hms_20d_prior = hms_s[i-20] if i >= 20 else np.nan

        signals.append({
            "date":          dates_arr[i],
            "ticker":        ticker,
            "dm_at_entry":   round(float(dm_now), 2),
            "ema5_at_entry": round(float(ema_now), 2),
            "hms_at_entry":  round(float(hms_now), 4)  if not pd.isna(hms_now)  else None,
            "hms_5d_prior":  round(float(hms_5d_prior), 4)  if not pd.isna(hms_5d_prior)  else None,
            "hms_10d_prior": round(float(hms_10d_prior), 4) if not pd.isna(hms_10d_prior) else None,
            "hms_20d_prior": round(float(hms_20d_prior), 4) if not pd.isna(hms_20d_prior) else None,
            **fwd,
        })

df = pd.DataFrame(signals)
df.to_csv("hms_backtest_phase2_results.csv", index=False)
print(f"Total entry signals: {len(df):,}")
df_valid = df[df["hms_at_entry"].notna()].copy()
print(f"With HMS at entry:   {len(df_valid):,}\n")

if len(df_valid) == 0:
    print("No signals with HMS data — nothing to analyse. Exiting.")
    raise SystemExit(0)


# ── TT HELPER ────────────────────────────────────────────────────────────────
def run_ttest(label, group_a, group_b, name_a, name_b):
    out = {}
    print(f"\n  {label}  ({name_a} n={len(group_a):,} | {name_b} n={len(group_b):,})")
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
        direct = f"{name_a} > {name_b}" if diff > 0 else f"{name_b} > {name_a}"
        print(f"    {w}d: {name_a}={a.mean():+.3f}% (n={len(a):,})  "
              f"{name_b}={b.mean():+.3f}% (n={len(b):,})  "
              f"diff={diff:+.3f}  p={p:.4f}  {sig}  {direct}")
        out[w] = {
            "a_mean": round(a.mean(), 3), "a_n": len(a),
            "b_mean": round(b.mean(), 3), "b_n": len(b),
            "diff":   round(diff, 3), "p": round(p, 4),
            "sig":    bool(p < 0.05),
        }
    return out


# ── TEST A ────────────────────────────────────────────────────────────────────
print("═" * 70)
print("TEST A — HMS >= 0.40 vs HMS < 0.40 at entry")
print("═" * 70)
high = df_valid[df_valid["hms_at_entry"] >= 0.40]
low  = df_valid[df_valid["hms_at_entry"] <  0.40]
test_a = run_ttest("Test A", high, low, "High", "Low")

# ── TEST B ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("TEST B — Early (HMS < 0.30) vs Mature (HMS >= 0.45)")
print("═" * 70)
early  = df_valid[df_valid["hms_at_entry"] <  0.30]
mature = df_valid[df_valid["hms_at_entry"] >= 0.45]
mid    = df_valid[(df_valid["hms_at_entry"] >= 0.30) & (df_valid["hms_at_entry"] < 0.45)]
print(f"  Early (HMS<0.30):     {len(early):,}")
print(f"  Mid   (0.30-0.45):    {len(mid):,}")
print(f"  Mature (HMS>=0.45):   {len(mature):,}")
test_b = run_ttest("Test B", early, mature, "Early", "Mature")

# ── TEST C ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("TEST C — Mid-Rising (0.30-0.45 AND rising 20d) vs Mature")
print("═" * 70)
df_valid["hms_rising_20d"] = (
    df_valid["hms_at_entry"] > df_valid["hms_20d_prior"]
) & df_valid["hms_20d_prior"].notna()
mid_rising = df_valid[
    (df_valid["hms_rising_20d"]) &
    (df_valid["hms_at_entry"] >= 0.30) &
    (df_valid["hms_at_entry"] <  0.45)
]
print(f"  Mid-Rising: {len(mid_rising):,}")
print(f"  Mature:     {len(mature):,}")
test_c = run_ttest("Test C", mid_rising, mature, "Mid-Rising", "Mature")

# ── BY YEAR ──────────────────────────────────────────────────────────────────
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

# ── COMBINED VERDICT (P1 + P2) ───────────────────────────────────────────────
print(f"\n{'═'*70}")
print("COMBINED VERDICT — PHASE 1 (2022-2026) + PHASE 2 (2016-2021)")
print("═" * 70)

for test_name, p2, key in [("A (High vs Low)", test_a, "A"),
                           ("B (Early vs Mature)", test_b, "B"),
                           ("C (Mid-Rising vs Mature)", test_c, "C")]:
    print(f"\nTest {test_name}:")
    for w in FORWARD_WINDOWS:
        p1d = PHASE1.get(f"{key}_diff_{w}d")
        p1p = PHASE1.get(f"{key}_p_{w}d")
        p2d = p2.get(w, {}).get("diff")
        p2p = p2.get(w, {}).get("p")
        if p2d is None:
            print(f"  {w}d: P1 diff={p1d:+.3f} p={p1p:.4f}  |  P2 insufficient data")
            continue
        both_sig_same_dir = (p1p < 0.05 and p2p < 0.05 and
                             (p1d > 0) == (p2d > 0))
        mark = "✓✓ both sig same dir" if both_sig_same_dir else \
               ("~ disagree" if (p1d > 0) != (p2d > 0) else "~")
        print(f"  {w}d: P1 diff={p1d:+.3f} p={p1p:.4f}  |  "
              f"P2 diff={p2d:+.3f} p={p2p:.4f}  {mark}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_rows = []
for name, res in [("A_high_vs_low", test_a),
                  ("B_early_vs_mature", test_b),
                  ("C_midrising_vs_mature", test_c)]:
    for w, r in res.items():
        out_rows.append({"phase": "P2_2016_2021", "test": name, "window_d": w, **r})
pd.DataFrame(out_rows).to_csv("hms_backtest_phase2_abc_results.csv", index=False)

with open("hms_backtest_phase2_summary.txt", "w", encoding="utf-8") as f:
    f.write("ARGUS — HMS EARLY WARNING — PHASE 2 (2016-2021)\n")
    f.write(f"Y2AI Research | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Signals: {len(df):,} | With HMS: {len(df_valid):,}\n\n")
    for name, res in [("A (HMS>=0.40 vs <0.40)", test_a),
                      ("B (Early <0.30 vs Mature >=0.45)", test_b),
                      ("C (Mid-Rising vs Mature)", test_c)]:
        f.write(f"Test {name}\n")
        for w, r in res.items():
            f.write(f"  {w}d: diff={r['diff']:+.3f} p={r['p']:.4f} "
                    f"({'sig' if r['sig'] else 'ns'})\n")
        f.write("\n")

with open("hms_combined_summary.txt", "w", encoding="utf-8") as f:
    f.write("ARGUS — HMS EARLY WARNING COMBINED (2016-2026)\n")
    f.write(f"Y2AI Research | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Phase 1 (2022-2026): {PHASE1['total_signals']:,} signals\n")
    f.write(f"Phase 2 (2016-2021): {len(df):,} signals\n\n")
    for test_name, p2, key in [("A (High vs Low)", test_a, "A"),
                               ("B (Early vs Mature)", test_b, "B"),
                               ("C (Mid-Rising vs Mature)", test_c, "C")]:
        f.write(f"Test {test_name}\n")
        for w in FORWARD_WINDOWS:
            p1d = PHASE1.get(f"{key}_diff_{w}d")
            p1p = PHASE1.get(f"{key}_p_{w}d")
            p2d = p2.get(w, {}).get("diff")
            p2p = p2.get(w, {}).get("p")
            f.write(f"  {w}d: P1 diff={p1d:+.3f} p={p1p:.4f}  |  ")
            if p2d is None:
                f.write("P2 insufficient\n")
            else:
                f.write(f"P2 diff={p2d:+.3f} p={p2p:.4f}\n")
        f.write("\n")

print("\nSaved: hms_backtest_phase2_results.csv")
print("Saved: hms_backtest_phase2_abc_results.csv")
print("Saved: hms_backtest_phase2_summary.txt")
print("Saved: hms_combined_summary.txt")
print(f"Done: {datetime.now().strftime('%H:%M:%S')}")
print("═" * 70)
