"""
BUG-003 probe — for each ticker, fetch min/max date and total rows from
dm_history, then simulate the n>=20 filter for the April 1 and April 25
3-year windows. Compare the two universes side by side.
"""
import os
import time
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Pull all dm_history (only need date+ticker for this analysis).
print("Pulling dm_history (date,ticker) — year-chunked...")
rows = []
years = list(range(2016, 2027))
for y in years:
    page, off = 5000, 0
    while True:
        attempt = 0
        cur = page
        while True:
            try:
                r = (sb.table("dm_history")
                     .select("date,ticker")
                     .gte("date", f"{y}-01-01").lte("date", f"{y}-12-31")
                     .order("date").order("ticker")
                     .range(off, off + cur - 1).execute())
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    raise
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
print(f"\nTotal: {len(df):,} rows | unique tickers: {df['ticker'].nunique()}")

today = date.today()
W1_START = today - timedelta(days=24) - timedelta(days=365 * 3)  # April 1's window start
W1_END   = today - timedelta(days=24)                            # April 1
W2_START = today - timedelta(days=365 * 3)                       # current window start
W2_END   = today

# Approximate to anchor on actual run dates
W1_END   = date(2026, 4, 1)
W1_START = date(W1_END.year - 3, W1_END.month, W1_END.day)
W2_END   = date(2026, 4, 25)
W2_START = date(W2_END.year - 3, W2_END.month, W2_END.day)

print(f"\nWindow 1 (April 1 baseline): {W1_START} → {W1_END}")
print(f"Window 2 (April 25 Saturday):  {W2_START} → {W2_END}")

w1 = df[(df["date"] >= W1_START) & (df["date"] <= W1_END)]
w2 = df[(df["date"] >= W2_START) & (df["date"] <= W2_END)]

# Count rows per ticker in each window
c1 = w1.groupby("ticker").size()
c2 = w2.groupby("ticker").size()

# Apply the n>=20 filter
pass1 = set(c1[c1 >= 20].index)
pass2 = set(c2[c2 >= 20].index)

print(f"\nTickers with >=20 rows in W1: {len(pass1):,}")
print(f"Tickers with >=20 rows in W2: {len(pass2):,}")
print(f"Delta:                          {len(pass2) - len(pass1):+,}")

dropped = pass1 - pass2
added   = pass2 - pass1
print(f"\nTickers in W1 but NOT in W2 (dropped): {len(dropped)}")
print(f"Tickers in W2 but NOT in W1 (added):    {len(added)}")

# For dropped tickers, why did they fail W2? Either no data, or n<20.
print("\n--- DROPPED TICKERS — why did they fail W2? ---")
no_data_w2 = []
low_data_w2 = []
for t in sorted(dropped):
    n2 = c2.get(t, 0)
    if n2 == 0:
        no_data_w2.append(t)
    else:
        low_data_w2.append((t, n2))

print(f"  No rows at all in W2: {len(no_data_w2)}")
print(f"  Some rows but <20:    {len(low_data_w2)}")

# Also: how recent is each ticker's last update?
print("\n--- LAST-DATE DISTRIBUTION FOR DROPPED TICKERS ---")
last_per_ticker = df.groupby("ticker")["date"].max()
last_for_dropped = last_per_ticker[last_per_ticker.index.isin(dropped)]
buckets = {
    "<= 2025-12-31":           (last_for_dropped <= date(2025, 12, 31)).sum(),
    "2026-01-01 to 2026-03-01":((last_for_dropped >= date(2026, 1, 1)) &
                                (last_for_dropped <= date(2026, 3, 1))).sum(),
    "2026-03-02 to 2026-04-01":((last_for_dropped >= date(2026, 3, 2)) &
                                (last_for_dropped <= date(2026, 4, 1))).sum(),
    "2026-04-02 to 2026-04-25":((last_for_dropped >= date(2026, 4, 2)) &
                                (last_for_dropped <= date(2026, 4, 25))).sum(),
}
for k, v in buckets.items():
    print(f"  Last DM row {k}: {v}")

# Sample first 15 dropped tickers with their last date
print("\n--- SAMPLE DROPPED TICKERS (first 15) ---")
for t in sorted(dropped)[:15]:
    last = last_per_ticker.get(t, "no rows")
    n2 = c2.get(t, 0)
    print(f"  {t:8} last_dm_date={last}  rows_in_W2={n2}")

# Save
pd.DataFrame({"ticker": sorted(dropped),
              "last_dm_date": [last_per_ticker.get(t) for t in sorted(dropped)],
              "rows_in_w2": [c2.get(t, 0) for t in sorted(dropped)]}
            ).to_csv("_bug003_dropped_tickers.csv", index=False)
print(f"\nSaved: _bug003_dropped_tickers.csv ({len(dropped)} rows)")
