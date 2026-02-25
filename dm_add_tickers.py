"""
Add new tickers to scanner_universe and backfill price history from Polygon.

Usage: python dm_add_tickers.py
"""
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client
import requests
from datetime import datetime

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")

TICKERS_TO_ADD = {
    # Quantum Computing → XLK
    "IONQ":  "Technology",
    "RGTI":  "Technology",
    "QUBT":  "Technology",
    "QBTS":  "Technology",
    # Semiconductors → SMH
    "ARM":   "Semiconductors",
    "ASML":  "Semiconductors",
    "MRVL":  "Semiconductors",
    # Technology → XLK
    "AXON":  "Technology",
    "GIB":   "Technology",
    "DXC":   "Technology",
    # Software → IGV
    "TEAM":  "Software",
    "ASAN":  "Software",
    "MNDY":  "Software",
    "ZI":    "Software",
    # Consumer Discretionary → XLY
    "MELI":  "Consumer Discretionary",
    "BKNG":  "Consumer Discretionary",
    # Consumer Staples → XLP
    "CCEP":  "Consumer Staples",
    # Health Care → XLV
    "ABBV":  "Health Care",
    "IDXX":  "Health Care",
    "CNC":   "Health Care",
    # Biotechnology → XBI
    "ALNY":  "Biotechnology",
    # Cybersecurity → HACK
    "GEN":   "Cybersecurity",
    "OKTA":  "Cybersecurity",
    "CYBR":  "Cybersecurity",
    "CHKP":  "Cybersecurity",
    "S":     "Cybersecurity",
    "TENB":  "Cybersecurity",
    "QLYS":  "Cybersecurity",
    # Energy → XLE
    "BKR":   "Energy",
    # Industrials → XLI
    "VRSK":  "Industrials",
    "TREX":  "Industrials",
    "AZEK":  "Industrials",
    "IBP":   "Industrials",
    "FERG":  "Industrials",
    "GMS":   "Industrials",
    # Communication Services → XLC
    "IPG":   "Communication Services",
    "WPP":   "Communication Services",
    "PUBGY": "Communication Services",
    # Financials → XLF
    "ABCB":  "Financials",
    "LPLA":  "Financials",
    "IBKR":  "Financials",
    "AMTD":  "Financials",
    "SF":    "Financials",
    # Real Estate → XLRE
    "VNO":   "Real Estate",
    "SLG":   "Real Estate",
    "HIW":   "Real Estate",
    "DEA":   "Real Estate",
    "CTRE":  "Real Estate",
}

# ── Step 1: Check existing universe ──
result = supabase.table("scanner_universe").select("ticker,sector").execute()
existing = {r['ticker']: r['sector'] for r in result.data}

print(f"Current universe: {len(existing)} tickers\n")
print("STATUS CHECK:")

to_add = []
for ticker, sector in TICKERS_TO_ADD.items():
    if ticker in existing:
        print(f"  {ticker}: Already exists (sector: {existing[ticker]})")
    else:
        print(f"  {ticker}: Missing — will add (sector: {sector})")
        to_add.append(ticker)

# ── Step 2: Add to scanner_universe ──
if not to_add:
    print("\nAll tickers already in scanner_universe.")
else:
    print(f"\nAdding {len(to_add)} tickers...")
    for ticker in to_add:
        sector = TICKERS_TO_ADD[ticker]
        supabase.table("scanner_universe").upsert(
            {"ticker": ticker, "sector": sector},
            on_conflict="ticker"
        ).execute()
        print(f"  Added {ticker} ({sector})")
    print(f"New universe size: {len(existing) + len(to_add)}")

# ── Step 3: Backfill prices from Polygon ──
if not POLYGON_API_KEY:
    print("\nNo POLYGON_API_KEY found. Skipping price backfill.")
    print("Set POLYGON_API_KEY in .env and run again, or use:")
    print("  python Dm_historical_pipeline.py fetch")
    exit()

START_DATE = "2016-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

print(f"\nPRICE BACKFILL ({START_DATE} to {END_DATE}):")
for ticker in TICKERS_TO_ADD:
    # Check existing price data
    count_result = supabase.table("price_history") \
        .select("date", count="exact") \
        .eq("ticker", ticker) \
        .execute()
    existing_count = count_result.count or 0

    if existing_count > 1000:
        print(f"  {ticker}: {existing_count} rows already — skipping")
        continue

    print(f"  {ticker}: {existing_count} rows — fetching from Polygon...")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{START_DATE}/{END_DATE}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}

    try:
        resp = requests.get(url, params=params)
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            print(f"    No data from Polygon for {ticker}")
            continue

        rows = []
        for r in data.get("results", []):
            try:
                date = datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "open": float(r["o"]),
                    "high": float(r["h"]),
                    "low": float(r["l"]),
                    "close": float(r["c"]),
                    "volume": int(r["v"])
                })
            except (ValueError, KeyError):
                continue

        # Batch upsert
        uploaded = 0
        BATCH = 500
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            supabase.table("price_history").upsert(
                batch, on_conflict="date,ticker"
            ).execute()
            uploaded += len(batch)

        print(f"    {uploaded} rows uploaded ({rows[0]['date']} to {rows[-1]['date']})")

    except Exception as e:
        print(f"    ERROR: {e}")

print("\nDone! Now run:")
print("  python Dm_historical_pipeline.py calculate --refresh-cache")
print("  python Dm_historical_pipeline.py summary")