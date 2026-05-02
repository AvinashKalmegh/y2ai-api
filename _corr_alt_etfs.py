"""Alt-ETF correlation search for OGN, SOLV, VLTO.

Compares each ticker against its current ETF and a list of candidate
replacements, on a 60-day rolling correlation basis. Backfills any
missing ETF from Polygon before running the comparison.
"""
import os
import time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")

CANDIDATES = {
    "OGN":  {"current": "XLV", "alts": ["IHE", "PPH", "XPH", "PJP"]},
    "SOLV": {"current": "XLV", "alts": ["IHI", "XHE"]},
    "VLTO": {"current": "XLI", "alts": ["PHO", "FIW", "CGW", "PAVE", "IYJ"]},
}
WINDOW = 60
THRESHOLD = 0.40
START_DATE = "2016-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


def load_closes(ticker):
    rows, page = [], 0
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


def backfill_polygon(ticker):
    print(f"  Fetching {ticker} from Polygon...")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{START_DATE}/{END_DATE}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    resp = requests.get(url, params=params, timeout=60)
    data = resp.json()
    if data.get("resultsCount", 0) == 0:
        print(f"    No data from Polygon for {ticker}")
        return False
    rows = []
    for r in data.get("results", []):
        try:
            d = datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
            rows.append({
                "date": d, "ticker": ticker,
                "open": float(r["o"]), "high": float(r["h"]),
                "low": float(r["l"]), "close": float(r["c"]),
                "volume": int(r["v"]),
            })
        except (ValueError, KeyError):
            continue
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        supabase.table("price_history").upsert(
            rows[i:i + BATCH], on_conflict="date,ticker"
        ).execute()
    print(f"    {len(rows)} rows uploaded ({rows[0]['date']} to {rows[-1]['date']})")
    return True


def ensure_in_db(ticker):
    s = load_closes(ticker)
    if s is not None and len(s) > 100:
        return s
    backfill_polygon(ticker)
    return load_closes(ticker)


def corr_stats(ticker_series, etf_series):
    df = pd.concat([ticker_series, etf_series], axis=1, keys=["t", "e"]).dropna()
    rets = df.pct_change().dropna()
    if len(rets) < WINDOW + 5:
        return None
    rolling = rets["t"].rolling(WINDOW).corr(rets["e"]).dropna()
    return {
        "obs": len(rolling),
        "latest": rolling.iloc[-1],
        "median": rolling.median(),
        "min": rolling.min(),
        "pct_below": (rolling < THRESHOLD).mean() * 100,
    }


print(f"Alt-ETF correlation search (window={WINDOW}, threshold={THRESHOLD})\n")

# Pre-fetch all needed ETFs
all_etfs = set()
for cfg in CANDIDATES.values():
    all_etfs.add(cfg["current"])
    all_etfs.update(cfg["alts"])

etf_series = {}
for etf in sorted(all_etfs):
    s = ensure_in_db(etf)
    etf_series[etf] = s
    if s is not None:
        print(f"  {etf}: {len(s)} rows ({s.index.min().date()} to {s.index.max().date()})")
    else:
        print(f"  {etf}: NO DATA")
    time.sleep(0.2)

print()
recommendations = {}
for ticker, cfg in CANDIDATES.items():
    t_series = load_closes(ticker)
    if t_series is None:
        print(f"{ticker}: missing price data\n")
        continue

    candidates_to_test = [cfg["current"]] + cfg["alts"]
    print(f"{ticker} (current ETF: {cfg['current']})")
    print(f"  {'ETF':<6} {'Obs':>5} {'Latest':>8} {'Median':>8} {'Min':>8} {'%<thr':>7}  Pass")
    rows = []
    for etf in candidates_to_test:
        e_series = etf_series.get(etf)
        if e_series is None:
            print(f"  {etf:<6}  no data")
            continue
        stats = corr_stats(t_series, e_series)
        if stats is None:
            print(f"  {etf:<6}  insufficient overlap")
            continue
        passes = stats["latest"] >= THRESHOLD and stats["median"] >= THRESHOLD
        flag = "PASS" if passes else "fail"
        marker = " *" if etf == cfg["current"] else ""
        print(f"  {etf:<6} {stats['obs']:>5d} {stats['latest']:>8.3f} "
              f"{stats['median']:>8.3f} {stats['min']:>8.3f} "
              f"{stats['pct_below']:>6.1f}%  {flag}{marker}")
        rows.append((etf, stats, passes))

    # Pick best: highest median among those that pass; if none pass, highest median overall
    passing = [r for r in rows if r[2]]
    pool = passing if passing else rows
    if pool:
        best = max(pool, key=lambda r: r[1]["median"])
        recommendations[ticker] = (best[0], best[2])
        status = "RECOMMEND" if best[2] else "BEST AVAILABLE (still below threshold)"
        print(f"  -> {status}: {best[0]} (median {best[1]['median']:.3f}, latest {best[1]['latest']:.3f})\n")
    else:
        print()

print("=" * 60)
print("SUMMARY:")
for ticker, cfg in CANDIDATES.items():
    if ticker in recommendations:
        etf, passed = recommendations[ticker]
        change = "KEEP" if etf == cfg["current"] else f"SWITCH to {etf}"
        verdict = "" if passed else " (still below threshold — manual review)"
        print(f"  {ticker}: current={cfg['current']}  ->  {change}{verdict}")
