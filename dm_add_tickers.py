"""
Check and add tickers to scanner_universe.
Usage: python add_tickers.py
"""
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

TICKERS_TO_CHECK = {
    "INFY": "Information Technology",
    "WIT": "Information Technology",
    "CTSH": "Information Technology",
    "HDB": "Financials",
    "IBM": "Information Technology",
    "ACN": "Information Technology",
    "IT": "Information Technology",
    "SAP": "Information Technology",
}

# Check existing
result = supabase.table("scanner_universe").select("ticker,sector").execute()
existing = {r['ticker']: r['sector'] for r in result.data}

print("STATUS CHECK:")
print("")
already_in = []
to_add = []

for ticker, sector in TICKERS_TO_CHECK.items():
    if ticker in existing:
        print(f"  {ticker}: ✓ Already exists (sector: {existing[ticker]})")
        already_in.append(ticker)
    else:
        print(f"  {ticker}: ❌ Missing - will add (sector: {sector})")
        to_add.append(ticker)

print("")

if not to_add:
    print("All tickers already in scanner_universe. Nothing to add.")
else:
    print(f"Adding {len(to_add)} tickers: {', '.join(to_add)}")
    
    for ticker in to_add:
        sector = TICKERS_TO_CHECK[ticker]
        supabase.table("scanner_universe").upsert(
            {"ticker": ticker, "sector": sector},
            on_conflict="ticker"
        ).execute()
        print(f"  ✓ Added {ticker} ({sector})")
    
    print("")
    print("Done! New universe size:", len(existing) + len(to_add))

# Also check price_history coverage
print("")
print("PRICE HISTORY CHECK:")
for ticker in TICKERS_TO_CHECK:
    result = supabase.table("price_history") \
        .select("date", count="exact") \
        .eq("ticker", ticker) \
        .execute()
    count = result.count or 0
    status = "✓ has data" if count > 1000 else ("⚠️ short" if count > 0 else "❌ no data")
    print(f"  {ticker}: {count} rows {status}")