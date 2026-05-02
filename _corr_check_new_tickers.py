"""Quick 60-day rolling correlation check: new tickers vs assigned ETF."""
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

PAIRS = [
    ("OGN",  "XLV"),
    ("KVUE", "XLP"),
    ("SOLV", "XLV"),
    ("GEHC", "XLV"),
    ("VLTO", "XLI"),
    ("TROX", "XLB"),
]
WINDOW = 60
THRESHOLD = 0.40


def load_closes(ticker):
    rows = []
    page = 0
    while True:
        resp = (supabase.table("price_history")
                .select("date,close")
                .eq("ticker", ticker)
                .order("date", desc=False)
                .range(page * 1000, (page + 1) * 1000 - 1)
                .execute())
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < 1000:
            break
        page += 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["close"] = df["close"].astype(float)
    return df["close"]


print(f"60-day rolling correlation (threshold {THRESHOLD})\n")
print(f"{'Pair':<14} {'Obs':>5} {'Latest':>8} {'Median':>8} {'Min':>8} {'%<thr':>7}  Flag")
print("-" * 70)

flagged = []
for ticker, etf in PAIRS:
    s_t = load_closes(ticker)
    s_e = load_closes(etf)
    if s_t is None or s_e is None:
        print(f"{ticker}/{etf:<8} missing price data")
        continue

    df = pd.concat([s_t, s_e], axis=1, keys=[ticker, etf]).dropna()
    rets = df.pct_change().dropna()
    if len(rets) < WINDOW + 5:
        print(f"{ticker}/{etf:<8} only {len(rets)} return obs (need >{WINDOW})")
        continue

    rolling = rets[ticker].rolling(WINDOW).corr(rets[etf]).dropna()
    latest = rolling.iloc[-1]
    median = rolling.median()
    minv = rolling.min()
    pct_below = (rolling < THRESHOLD).mean() * 100
    flag = "FLAG" if latest < THRESHOLD or median < THRESHOLD else ""
    if flag:
        flagged.append((ticker, etf, latest, median))

    print(f"{ticker}/{etf:<8} {len(rolling):>5d} {latest:>8.3f} {median:>8.3f} {minv:>8.3f} {pct_below:>6.1f}%  {flag}")

print()
if flagged:
    print(f"FLAGGED ({len(flagged)}): {[f[0] for f in flagged]}")
else:
    print("All pairs above threshold on latest + median.")
