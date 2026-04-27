"""
BUG-003 follow-up — replicate the script's exact crossing logic on the
April 1 and April 25 trailing-3yr windows. For each ticker count crossings
of DM above 70. A ticker is dropped from output if signals==0 OR n<20.
Compare both windows to find which tickers exited the result set.
"""
import os
import time
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

ENTRY_DM = 70

print("Pulling dm_history (date,ticker,dm_smoothed)...")
rows = []
for y in range(2023, 2027):
    page, off = 10000, 0
    while True:
        attempt = 0; cur = page
        while True:
            try:
                r = (sb.table("dm_history")
                     .select("date,ticker,dm_smoothed")
                     .gte("date", f"{y}-01-01").lte("date", f"{y}-12-31")
                     .order("date").order("ticker")
                     .range(off, off + cur - 1).execute())
                break
            except Exception:
                attempt += 1
                if attempt >= 6: raise
                cur = max(500, cur // 2)
                time.sleep(2 ** attempt)
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < cur:
            break
        off += cur
    print(f"  {y}: cumulative {len(rows):,}")

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"]).dt.date
df["dm"]   = pd.to_numeric(df["dm_smoothed"], errors="coerce")
df = df.dropna(subset=["dm"]).sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"\nTotal: {len(df):,} rows | tickers: {df['ticker'].nunique()}")


def count_crossings(window_df, ticker):
    sub = window_df[window_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if len(sub) < 20:
        return None  # would be filtered by n<20
    sigs = 0
    for i in range(1, len(sub)):
        if sub.loc[i-1, "dm"] < ENTRY_DM <= sub.loc[i, "dm"]:
            sigs += 1
    return sigs


def universe_at(end_date):
    start_date = date(end_date.year - 3, end_date.month, end_date.day)
    w = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    print(f"\n  Window {start_date} → {end_date}: {len(w):,} rows | "
          f"{w['ticker'].nunique()} tickers in data")
    out = {}
    for t in w["ticker"].unique():
        s = count_crossings(w, t)
        if s is None or s == 0:
            continue
        out[t] = s
    return out, w["ticker"].nunique()


print("\n=== APRIL 1 (baseline) ===")
u_apr1, total_apr1 = universe_at(date(2026, 4, 1))
print(f"  Tickers with >=1 crossing & >=20 rows: {len(u_apr1)}")

print("\n=== APRIL 25 (Saturday) ===")
u_apr25, total_apr25 = universe_at(date(2026, 4, 25))
print(f"  Tickers with >=1 crossing & >=20 rows: {len(u_apr25)}")

print(f"\n  Delta: {len(u_apr25) - len(u_apr1):+}")

dropped = set(u_apr1) - set(u_apr25)
added   = set(u_apr25) - set(u_apr1)
print(f"  Tickers that fell out (in Apr1 set, not in Apr25 set): {len(dropped)}")
print(f"  Tickers added (in Apr25 set, not in Apr1 set):         {len(added)}")

# A ticker leaves the set when its last crossing fell out the back end.
# Show a sample with their last crossing date.
def last_crossing(window_df, ticker):
    sub = window_df[window_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    last = None
    for i in range(1, len(sub)):
        if sub.loc[i-1, "dm"] < ENTRY_DM <= sub.loc[i, "dm"]:
            last = sub.loc[i, "date"]
    return last

print(f"\n  Sample of tickers that dropped out (first 20):")
all_dm = df  # full history for these tickers
for t in sorted(dropped)[:20]:
    sub_all = all_dm[all_dm["ticker"] == t].sort_values("date").reset_index(drop=True)
    last_cross_anywhere = last_crossing(sub_all, t)
    sig_apr1 = u_apr1.get(t, 0)
    print(f"    {t:8}  Apr1 crossings={sig_apr1:3}  "
          f"last_crossing_in_history={last_cross_anywhere}")

pd.DataFrame({
    "ticker": sorted(dropped),
    "apr1_crossings": [u_apr1.get(t, 0) for t in sorted(dropped)],
    "last_crossing_anywhere": [
        last_crossing(df[df["ticker"] == t].sort_values("date").reset_index(drop=True), t)
        for t in sorted(dropped)
    ],
}).to_csv("_bug003_dropped_by_signals.csv", index=False)
print(f"\nSaved: _bug003_dropped_by_signals.csv ({len(dropped)} rows)")
