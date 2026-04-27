"""
Pull hms_daily, compute per-ticker 20-day HMS change, and attach it to the
existing wave_r_events.csv — saving wave_r_events_v2.csv.

Keeps all existing columns + adds hms_chg20. No re-running of the main
validator; everything else (dm_recovery, factor scoring) stays as-is.
"""
import os
import time
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Pull from two months before the events window so the 20-day rolling
# change has full support for the earliest events (2022-01-10).
DATE_FROM = "2021-11-01"

print(f"Pulling hms_daily from {DATE_FROM}...")
rows, offset, page = [], 0, 1000
while True:
    attempt = 0
    cur = page
    while True:
        try:
            r = (sb.table("hms_daily")
                 .select("date,ticker,hms_score")
                 .gte("date", DATE_FROM)
                 .order("date").order("ticker")
                 .range(offset, offset + cur - 1)
                 .execute())
            break
        except Exception as e:
            attempt += 1
            if attempt >= 6:
                raise
            cur = max(100, cur // 2)
            print(f"  [retry {attempt}] page={cur}, sleep {2**attempt}s")
            time.sleep(2 ** attempt)
    if not r.data:
        break
    rows.extend(r.data)
    if len(r.data) < cur:
        break
    offset += cur

hms = pd.DataFrame(rows)
hms["date"] = pd.to_datetime(hms["date"])
hms = hms.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"  Pulled {len(hms):,} hms_daily rows | {hms['ticker'].nunique()} tickers")

print("Computing hms_chg20 per ticker (row-based 20 back)...")
hms["hms_chg20"] = hms.groupby("ticker")["hms_score"].transform(lambda s: s - s.shift(20))

print("Loading wave_r_events.csv...")
events = pd.read_csv("wave_r_events.csv")
events["date"] = pd.to_datetime(events["date"])
print(f"  Events: {len(events):,}")

print("Merging hms_chg20 onto events...")
merged = events.merge(
    hms[["date", "ticker", "hms_chg20"]],
    on=["date", "ticker"],
    how="left",
)

missing = merged["hms_chg20"].isna().sum()
print(f"  Events with hms_chg20: {len(merged) - missing:,} | missing: {missing:,}")
print(f"  hms_chg20 stats: min={merged['hms_chg20'].min():.4f} "
      f"max={merged['hms_chg20'].max():.4f} "
      f"mean={merged['hms_chg20'].mean():.4f}")

merged.to_csv("wave_r_events_v2.csv", index=False)
print(f"\nSaved: wave_r_events_v2.csv ({len(merged):,} rows)")
