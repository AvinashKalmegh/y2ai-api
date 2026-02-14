"""
Clean up scanner_universe: remove delisted tickers, update renamed ones.

Delisted (acquired/taken private):
  K    → Kellanova, acquired by Mars (Dec 11, 2025)
  SPR  → Spirit AeroSystems, acquired by Boeing (Dec 8, 2025)
  IPG  → Interpublic Group, merged into Omnicom (Nov 26, 2025)
  WBA  → Walgreens Boots Alliance, taken private by Sycamore (Aug 28, 2025)
  DAY  → Dayforce, acquired by Thoma Bravo (Feb 4, 2026)

Renamed (still active):
  FI   → FISV  (Fiserv, moved NYSE→NASDAQ, Nov 11, 2025)
  MMC  → MRSH  (Marsh McLennan, rebranded, Jan 14, 2026)

Usage: python dm_cleanup_tickers.py
"""
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

DELISTED = ["K", "SPR", "IPG", "WBA", "DAY"]

RENAMED = {
    "FI":  "FISV",   # Fiserv — moved to NASDAQ, same company
    "MMC": "MRSH",   # Marsh McLennan — rebranded
}

# ── Step 1: Check current state ──
result = supabase.table("scanner_universe").select("ticker,sector").execute()
existing = {r['ticker']: r['sector'] for r in result.data}
print(f"Current universe: {len(existing)} tickers\n")

# ── Step 2: Remove delisted tickers ──
print("DELISTED TICKERS:")
removed = 0
for ticker in DELISTED:
    if ticker in existing:
        supabase.table("scanner_universe").delete().eq("ticker", ticker).execute()
        print(f"  Removed {ticker} from scanner_universe")
        removed += 1
    else:
        print(f"  {ticker}: not found (already removed)")

# ── Step 3: Rename tickers ──
print(f"\nRENAMED TICKERS:")
renamed_count = 0
for old_ticker, new_ticker in RENAMED.items():
    old_exists = old_ticker in existing
    new_exists = new_ticker in existing

    if not old_exists and new_exists:
        print(f"  {old_ticker} → {new_ticker}: already updated")
        continue

    if not old_exists and not new_exists:
        print(f"  {old_ticker} → {new_ticker}: neither found, skipping")
        continue

    sector = existing[old_ticker]

    # Add new ticker with same sector
    if not new_exists:
        supabase.table("scanner_universe").upsert(
            {"ticker": new_ticker, "sector": sector},
            on_conflict="ticker"
        ).execute()
        print(f"  Added {new_ticker} ({sector})")

    # Remove old ticker
    supabase.table("scanner_universe").delete().eq("ticker", old_ticker).execute()
    print(f"  Removed old ticker {old_ticker}")

    # Update price_history: rename old ticker rows to new ticker
    # This preserves historical data under the new symbol
    print(f"  Updating price_history: {old_ticker} → {new_ticker}...")
    try:
        update_result = supabase.table("price_history") \
            .update({"ticker": new_ticker}) \
            .eq("ticker", old_ticker) \
            .execute()
        row_count = len(update_result.data) if update_result.data else 0
        print(f"    Updated {row_count} price rows")
    except Exception as e:
        print(f"    WARNING: price_history update failed: {e}")
        print(f"    May need manual SQL: UPDATE price_history SET ticker='{new_ticker}' WHERE ticker='{old_ticker}'")

    # Update dm_history too
    print(f"  Updating dm_history: {old_ticker} → {new_ticker}...")
    try:
        dm_result = supabase.table("dm_history") \
            .update({"ticker": new_ticker}) \
            .eq("ticker", old_ticker) \
            .execute()
        dm_count = len(dm_result.data) if dm_result.data else 0
        print(f"    Updated {dm_count} DM rows")
    except Exception as e:
        print(f"    WARNING: dm_history update failed: {e}")
        print(f"    May need manual SQL: UPDATE dm_history SET ticker='{new_ticker}' WHERE ticker='{old_ticker}'")

    # Update dm_latest
    try:
        supabase.table("dm_latest").delete().eq("ticker", old_ticker).execute()
        print(f"    Cleared {old_ticker} from dm_latest (will rebuild)")
    except Exception:
        pass

    renamed_count += 1

# ── Step 4: Verify ──
print(f"\n{'='*50}")
result2 = supabase.table("scanner_universe").select("ticker", count="exact").execute()
new_count = result2.count or len(result2.data)
print(f"Updated universe: {new_count} tickers")
print(f"  Removed {removed} delisted tickers")
print(f"  Renamed {renamed_count} tickers")

# Check if renamed tickers need price backfill
print(f"\nPRICE COVERAGE CHECK:")
for new_ticker in RENAMED.values():
    price_result = supabase.table("price_history") \
        .select("date", count="exact") \
        .eq("ticker", new_ticker) \
        .execute()
    count = price_result.count or 0
    status = "OK" if count > 1000 else ("short" if count > 0 else "NO DATA")
    print(f"  {new_ticker}: {count} rows [{status}]")

print(f"\nDone! Next steps:")
print(f"  python Dm_historical_pipeline.py calculate --refresh-cache")
print(f"  python Dm_historical_pipeline.py summary")