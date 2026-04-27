"""
ARGUS — HMS EARLY WARNING BACKTEST
Y2AI Research | April 2026

Tests whether adding HMS >= threshold as an entry filter improves PureSim
forward returns vs. the baseline DM >= 65 + 5d EMA rising + price > 20d MA rule.

Portfolio A (control):  DM >= 65, 5d EMA rising, close > 20d price MA
Portfolio B (treatment): A + HMS >= threshold at entry

OUTPUTS:
    hms_backtest_results.csv     — every signal with HMS context + fwd returns
    hms_threshold_sweep.csv      — stats across 3 HMS thresholds
    hms_backtest_summary.txt     — narrative verdict
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
DATE_FROM        = "2022-01-01"
DATE_TO          = "2026-04-22"
DM_ENTRY         = 65.0
EMA_PERIOD       = 5
PRICE_MA_DAYS    = 20
HMS_THRESHOLDS   = [0.35, 0.40, 0.45]
FORWARD_WINDOWS  = [30, 60, 90]   # trading days
MIN_SIGNALS      = 20
PAGE_SIZE        = 1000

print("═" * 70)
print("ARGUS — HMS EARLY WARNING BACKTEST")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Data range: {DATE_FROM} → {DATE_TO}")
print("═" * 70 + "\n")


# ── CHUNKED SUPABASE PULL ─────────────────────────────────────────────────────
def pull_table(table, select, date_col="date",
               date_from=DATE_FROM, date_to=DATE_TO,
               chunk=PAGE_SIZE):
    """Paginated pull with retries; server-side date filter + order."""
    rows, offset = [], 0
    while True:
        attempt = 0
        cur = chunk
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
                print(f"    [retry {attempt}] {table} page={cur} sleep {2**attempt}s")
                time.sleep(2 ** attempt)
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < cur:
            break
        offset += cur
    return pd.DataFrame(rows)


# ── PULL DATA ─────────────────────────────────────────────────────────────────
print("Pulling DM scores (dm_history.dm_smoothed)...")
dm_raw = pull_table("dm_history", "date,ticker,dm_smoothed")
dm_raw["date"] = pd.to_datetime(dm_raw["date"])
dm_raw = dm_raw.rename(columns={"dm_smoothed": "dm_score"})
print(f"  {len(dm_raw):,} rows | {dm_raw['ticker'].nunique()} tickers")

print("Pulling HMS scores (hms_daily.hms_score)...")
hms_raw = pull_table("hms_daily", "date,ticker,hms_score")
hms_raw["date"] = pd.to_datetime(hms_raw["date"])
print(f"  {len(hms_raw):,} rows | {hms_raw['ticker'].nunique()} tickers")

print("Pulling price data (price_history.close)...")
price_raw = pull_table("price_history", "date,ticker,close")
price_raw["date"] = pd.to_datetime(price_raw["date"])
print(f"  {len(price_raw):,} rows | {price_raw['ticker'].nunique()} tickers")

tickers = sorted(
    set(dm_raw["ticker"]) & set(hms_raw["ticker"]) & set(price_raw["ticker"])
)
print(f"\nUniverse (DM ∩ HMS ∩ Price): {len(tickers)} tickers\n")

# ── BUILD PIVOT TABLES ────────────────────────────────────────────────────────
print("Building pivot tables...")
dm_pivot    = dm_raw.pivot_table(index="date", columns="ticker", values="dm_score")
hms_pivot   = hms_raw.pivot_table(index="date", columns="ticker", values="hms_score")
price_pivot = price_raw.pivot_table(index="date", columns="ticker", values="close")

common_dates = dm_pivot.index.intersection(hms_pivot.index).intersection(price_pivot.index)
dm_pivot    = dm_pivot.loc[common_dates].sort_index()
hms_pivot   = hms_pivot.loc[common_dates].sort_index()
price_pivot = price_pivot.loc[common_dates].sort_index()
dates       = list(dm_pivot.index)
print(f"Common dates: {len(dates)} | {dates[0].date()} → {dates[-1].date()}\n")

# ── COMPUTE EMA + PRICE MA ────────────────────────────────────────────────────
print("Computing 5d EMA of DM and 20d MA of price...")
dm_ema5    = dm_pivot.ewm(span=EMA_PERIOD, adjust=False).mean()
price_ma20 = price_pivot.rolling(PRICE_MA_DAYS).mean()

# ── IDENTIFY ENTRY SIGNALS ────────────────────────────────────────────────────
print("Identifying entry signals...")
signals   = []
dates_arr = np.array(dates)
n_dates   = len(dates_arr)
max_fwd   = max(FORWARD_WINDOWS)

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
        if ema_now <= ema_prev:      # 5d EMA must be rising
            continue
        if price_now <= pma_now:     # close must be above 20d MA
            continue

        hms_now = hms_s[i]

        fwd_returns = {}
        for w in FORWARD_WINDOWS:
            if i + w < n_dates:
                p_fwd = price_s[i + w]
                if not pd.isna(p_fwd) and price_now > 0:
                    fwd_returns[f"ret_{w}d"] = (p_fwd - price_now) / price_now * 100
                else:
                    fwd_returns[f"ret_{w}d"] = None
            else:
                fwd_returns[f"ret_{w}d"] = None

        hms_5d_prior  = hms_s[i-5]  if i >= 5  else np.nan
        hms_10d_prior = hms_s[i-10] if i >= 10 else np.nan
        hms_20d_prior = hms_s[i-20] if i >= 20 else np.nan

        signals.append({
            "date":          dates_arr[i],
            "ticker":        ticker,
            "dm_at_entry":   round(float(dm_now), 2),
            "ema5_at_entry": round(float(ema_now), 2),
            "hms_at_entry":  round(float(hms_now), 4) if not pd.isna(hms_now) else None,
            "hms_5d_prior":  round(float(hms_5d_prior), 4)  if not pd.isna(hms_5d_prior)  else None,
            "hms_10d_prior": round(float(hms_10d_prior), 4) if not pd.isna(hms_10d_prior) else None,
            "hms_20d_prior": round(float(hms_20d_prior), 4) if not pd.isna(hms_20d_prior) else None,
            **fwd_returns,
        })

df = pd.DataFrame(signals)
print(f"Total entry signals found: {len(df):,}\n")

if len(df) == 0:
    print("No signals — nothing to analyse. Exiting.")
    raise SystemExit(0)

# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
print("═" * 70)
print("HMS THRESHOLD SWEEP — Portfolio A (DM only) vs Portfolio B (DM + HMS)")
print("═" * 70)

sweep_results = []
for threshold in HMS_THRESHOLDS:
    df["hms_confirmed"]    = df["hms_at_entry"] >= threshold
    df["hms_unconfirmed"]  = (df["hms_at_entry"] < threshold) & df["hms_at_entry"].notna()

    confirmed   = df[df["hms_confirmed"]]
    unconfirmed = df[df["hms_unconfirmed"]]
    all_signals = df[df["hms_at_entry"].notna()]

    print(f"\nHMS Threshold: {threshold}")
    print(f"  All signals:     {len(all_signals):,}")
    print(f"  HMS confirmed:   {len(confirmed):,} "
          f"({len(confirmed)/max(len(all_signals),1)*100:.1f}%)")
    print(f"  HMS unconfirmed: {len(unconfirmed):,} "
          f"({len(unconfirmed)/max(len(all_signals),1)*100:.1f}%)")

    row = {"threshold": threshold, "confirmed_n": len(confirmed), "unconfirmed_n": len(unconfirmed)}

    for w in FORWARD_WINDOWS:
        col = f"ret_{w}d"
        c = confirmed[col].dropna()
        u = unconfirmed[col].dropna()
        a = all_signals[col].dropna()

        if len(c) < MIN_SIGNALS or len(u) < MIN_SIGNALS:
            print(f"    {w}d: insufficient data (confirmed={len(c)}, unconfirmed={len(u)})")
            for k in (f"conf_mean_{w}d", f"unconf_mean_{w}d", f"all_mean_{w}d",
                      f"diff_{w}d", f"p_{w}d", f"sig_{w}d"):
                row[k] = None
            continue

        t_stat, p = stats.ttest_ind(c, u)
        diff   = c.mean() - u.mean()
        effect = diff / (pd.concat([c, u]).std() + 1e-9)
        sig    = "✓ SIGNIFICANT" if p < 0.05 else "~ not sig"
        direction = "HMS > No-HMS" if diff > 0 else "No-HMS > HMS"
        print(f"    {w}d: HMS={c.mean():+.2f}% (n={len(c):,})  "
              f"No-HMS={u.mean():+.2f}% (n={len(u):,})  "
              f"diff={diff:+.2f}  p={p:.4f}  {sig}  {direction}")

        row[f"conf_mean_{w}d"]   = round(c.mean(), 3)
        row[f"unconf_mean_{w}d"] = round(u.mean(), 3)
        row[f"all_mean_{w}d"]    = round(a.mean(), 3)
        row[f"diff_{w}d"]        = round(diff, 3)
        row[f"p_{w}d"]           = round(p, 4)
        row[f"sig_{w}d"]         = bool(p < 0.05 and diff > 0)

    sweep_results.append(row)

# ── LEAD TIME ─────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("HMS LEAD TIME ANALYSIS — How elevated is HMS before DM entry?")
print("═" * 70)

df_valid = df[df["hms_at_entry"].notna()]
for look_back, col in [(5, "hms_5d_prior"), (10, "hms_10d_prior"), (20, "hms_20d_prior")]:
    sub = df_valid[df_valid[col].notna()]
    if len(sub) < 10:
        continue
    avg = sub[col].mean()
    pct = (sub[col] >= 0.40).mean() * 100
    print(f"  HMS {look_back:2d}d before entry: avg={avg:.3f}  "
          f"pct>=0.40: {pct:.1f}%  (n={len(sub):,})")

# ── BY YEAR ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("BY YEAR (threshold=0.40, 30d forward return)")
print("═" * 70)

df["year"] = pd.to_datetime(df["date"]).dt.year
df["hms_confirmed_040"] = df["hms_at_entry"] >= 0.40

for year in sorted(df["year"].dropna().unique()):
    sub = df[df["year"] == year]
    c = sub[sub["hms_confirmed_040"]]["ret_30d"].dropna()
    u = sub[~sub["hms_confirmed_040"] & sub["hms_at_entry"].notna()]["ret_30d"].dropna()
    if len(c) < 10 or len(u) < 10:
        continue
    diff = c.mean() - u.mean()
    direction = "✓" if diff > 0 else "✗"
    print(f"  {int(year)}: HMS={c.mean():+.2f}% (n={len(c):,})  "
          f"No-HMS={u.mean():+.2f}% (n={len(u):,})  diff={diff:+.2f}  {direction}")

# ── SECTOR BREAKDOWN (uses local ETF_Reference.csv) ───────────────────────────
print(f"\n{'═'*70}")
print("SECTOR BREAKDOWN (threshold=0.40, 30d forward return)")
print("═" * 70)

try:
    etf_df = pd.read_csv("ETF_Reference.csv").rename(columns={"Ticker": "ticker"})
    df_sector = df.merge(etf_df[["ticker", "Sector_Name"]], on="ticker", how="left")
    for sector in sorted(df_sector["Sector_Name"].dropna().unique()):
        sub = df_sector[df_sector["Sector_Name"] == sector]
        c = sub[sub["hms_confirmed_040"]]["ret_30d"].dropna()
        u = sub[~sub["hms_confirmed_040"] & sub["hms_at_entry"].notna()]["ret_30d"].dropna()
        if len(c) < MIN_SIGNALS or len(u) < MIN_SIGNALS:
            continue
        diff = c.mean() - u.mean()
        t, p = stats.ttest_ind(c, u)
        sig = "✓" if p < 0.05 and diff > 0 else ("✗" if p < 0.05 else "~")
        print(f"  {sector:25}: HMS={c.mean():+.2f}% No-HMS={u.mean():+.2f}% "
              f"diff={diff:+.2f} p={p:.3f} {sig}")
except FileNotFoundError:
    print("  ETF_Reference.csv not found — skipping sector breakdown.")

# ── VERDICT ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("OVERALL VERDICT (threshold=0.40)")
print("═" * 70)

best_row = next((r for r in sweep_results if r["threshold"] == 0.40), sweep_results[0])
diff_30  = best_row.get("diff_30d")
diff_60  = best_row.get("diff_60d")
diff_90  = best_row.get("diff_90d")
sig_30   = best_row.get("sig_30d")

print()
if diff_30 is not None:
    print(f"  30d: HMS-confirmed outperforms by {diff_30:+.2f} pp "
          f"({'SIGNIFICANT' if sig_30 else 'not significant'})")
if diff_60 is not None:
    print(f"  60d: {diff_60:+.2f} pp")
if diff_90 is not None:
    print(f"  90d: {diff_90:+.2f} pp")

print()
if diff_30 and diff_30 > 2.0 and sig_30:
    print("  VERDICT: ✓ IMPLEMENT HMS EARLY WARNING LAYER")
    print("    Add HMS >= 0.40 as a hard gate in the PureSim entry rule.")
elif diff_30 and diff_30 > 0.5 and sig_30:
    print("  VERDICT: ~ MARGINAL SUPPORT — monitor, don't implement yet")
elif diff_30 and diff_30 > 0:
    print("  VERDICT: ~ DIRECTIONAL SUPPORT — not statistically significant")
else:
    print("  VERDICT: ✗ HMS FILTER DOES NOT ADD ALPHA")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv("hms_backtest_results.csv", index=False)
pd.DataFrame(sweep_results).to_csv("hms_threshold_sweep.csv", index=False)

with open("hms_backtest_summary.txt", "w", encoding="utf-8") as f:
    f.write("ARGUS — HMS EARLY WARNING BACKTEST SUMMARY\n")
    f.write(f"Y2AI Research | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Data range: {DATE_FROM} → {DATE_TO}\n")
    f.write(f"Total signals: {len(df):,}\n\n")
    f.write("THRESHOLD SWEEP RESULTS:\n")
    for row in sweep_results:
        f.write(f"\n  Threshold {row['threshold']}:\n")
        for w in FORWARD_WINDOWS:
            d = row.get(f"diff_{w}d")
            p = row.get(f"p_{w}d")
            if d is not None:
                f.write(f"    {w}d: diff={d:+.3f} p={p:.4f}\n")

print("\nSaved: hms_backtest_results.csv")
print("Saved: hms_threshold_sweep.csv")
print("Saved: hms_backtest_summary.txt")
print(f"Done: {datetime.now().strftime('%H:%M:%S')}")
print("═" * 70)
