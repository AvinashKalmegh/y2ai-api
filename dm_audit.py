# """
# dm_audit.py
# Audit dm_history data in Supabase to find gaps, missing dates, ticker coverage issues.

# Usage:
#     python dm_audit.py              # Full audit
#     python dm_audit.py ticker NVDA  # Audit single ticker
# """

# import os
# import sys
# from datetime import datetime
# from dotenv import load_dotenv
# load_dotenv()

# from supabase import create_client

# SUPABASE_URL = os.getenv('SUPABASE_URL')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# def get_supabase():
#     return create_client(SUPABASE_URL, SUPABASE_KEY)


# def paginate_query(table, select, filters=None, order_col="date", limit=1000):
#     """Paginate through Supabase results."""
#     supabase = get_supabase()
#     all_rows = []
#     offset = 0
#     while True:
#         q = supabase.table(table).select(select).order(order_col)
#         if filters:
#             for f in filters:
#                 q = getattr(q, f[0])(f[1], f[2])
#         result = q.range(offset, offset + limit - 1).execute()
#         if not result.data:
#             break
#         all_rows.extend(result.data)
#         if len(result.data) < limit:
#             break
#         offset += limit
#     return all_rows


# def count_by_date_range():
#     """Get row counts per year from dm_history."""
#     supabase = get_supabase()
    
#     ranges = [
#         ("2016", "2016-01-01", "2016-12-31"),
#         ("2017", "2017-01-01", "2017-12-31"),
#         ("2018", "2018-01-01", "2018-12-31"),
#         ("2019", "2019-01-01", "2019-12-31"),
#         ("2020", "2020-01-01", "2020-12-31"),
#         ("2021", "2021-01-01", "2021-12-31"),
#         ("2022", "2022-01-01", "2022-12-31"),
#         ("2023", "2023-01-01", "2023-12-31"),
#         ("2024", "2024-01-01", "2024-12-31"),
#         ("2025", "2025-01-01", "2025-12-31"),
#         ("2026", "2026-01-01", "2026-12-31"),
#     ]
    
#     print("=" * 60)
#     print("DM_HISTORY: ROW COUNTS BY YEAR")
#     print("=" * 60)
    
#     total = 0
#     for label, start, end in ranges:
#         # Get count by fetching just dates
#         result = supabase.table("dm_history") \
#             .select("date", count="exact") \
#             .gte("date", start) \
#             .lte("date", end) \
#             .limit(1) \
#             .execute()
#         count = result.count if result.count else 0
#         total += count
#         if count > 0:
#             # Get first and last date
#             first = supabase.table("dm_history") \
#                 .select("date") \
#                 .gte("date", start) \
#                 .lte("date", end) \
#                 .order("date") \
#                 .limit(1) \
#                 .execute()
#             last = supabase.table("dm_history") \
#                 .select("date") \
#                 .gte("date", start) \
#                 .lte("date", end) \
#                 .order("date", desc=True) \
#                 .limit(1) \
#                 .execute()
#             first_date = first.data[0]["date"] if first.data else "?"
#             last_date = last.data[0]["date"] if last.data else "?"
#             print(f"  {label}: {count:>10,} rows  ({first_date} to {last_date})")
#         else:
#             print(f"  {label}: {count:>10,} rows")
    
#     print(f"  {'TOTAL':>4}: {total:>10,} rows")
#     print()


# def count_distinct_tickers_by_year():
#     """Count unique tickers per year."""
#     supabase = get_supabase()
    
#     ranges = [
#         ("2016-2019", "2016-01-01", "2019-12-31"),
#         ("2020-2023", "2020-01-01", "2023-12-31"),
#         ("2024-2026", "2024-01-01", "2026-12-31"),
#     ]
    
#     print("=" * 60)
#     print("DM_HISTORY: DISTINCT TICKERS BY PERIOD")
#     print("=" * 60)
    
#     for label, start, end in ranges:
#         # Sample unique tickers by fetching ticker column
#         all_tickers = set()
#         offset = 0
#         while True:
#             result = supabase.table("dm_history") \
#                 .select("ticker") \
#                 .gte("date", start) \
#                 .lte("date", end) \
#                 .order("ticker") \
#                 .range(offset, offset + 999) \
#                 .execute()
#             if not result.data:
#                 break
#             for r in result.data:
#                 all_tickers.add(r["ticker"])
#             if len(result.data) < 1000:
#                 break
#             offset += 1000
#             if offset > 500000:  # safety
#                 break
        
#         print(f"  {label}: {len(all_tickers)} tickers")
#     print()


# def count_trading_days():
#     """Count distinct trading days per year."""
#     supabase = get_supabase()
    
#     ranges = [
#         ("2016", "2016-01-01", "2016-12-31"),
#         ("2017", "2017-01-01", "2017-12-31"),
#         ("2018", "2018-01-01", "2018-12-31"),
#         ("2019", "2019-01-01", "2019-12-31"),
#         ("2020", "2020-01-01", "2020-12-31"),
#         ("2021", "2021-01-01", "2021-12-31"),
#         ("2022", "2022-01-01", "2022-12-31"),
#         ("2023", "2023-01-01", "2023-12-31"),
#         ("2024", "2024-01-01", "2024-12-31"),
#         ("2025", "2025-01-01", "2025-12-31"),
#         ("2026", "2026-01-01", "2026-12-31"),
#     ]
    
#     print("=" * 60)
#     print("DM_HISTORY: TRADING DAYS BY YEAR (expected ~252)")
#     print("=" * 60)
    
#     for label, start, end in ranges:
#         dates = set()
#         offset = 0
#         while True:
#             result = supabase.table("dm_history") \
#                 .select("date") \
#                 .gte("date", start) \
#                 .lte("date", end) \
#                 .order("date") \
#                 .range(offset, offset + 999) \
#                 .execute()
#             if not result.data:
#                 break
#             for r in result.data:
#                 dates.add(r["date"])
#             if len(result.data) < 1000:
#                 break
#             offset += 1000
#             if offset > 500000:
#                 break
        
#         if dates:
#             sorted_dates = sorted(dates)
#             print(f"  {label}: {len(dates):>4} days  ({sorted_dates[0]} to {sorted_dates[-1]})")
#         else:
#             print(f"  {label}:    0 days")
#     print()


# def check_price_history():
#     """Check price_history coverage for comparison."""
#     supabase = get_supabase()
    
#     ranges = [
#         ("2016", "2016-01-01", "2016-12-31"),
#         ("2017", "2017-01-01", "2017-12-31"),
#         ("2018", "2018-01-01", "2018-12-31"),
#         ("2019", "2019-01-01", "2019-12-31"),
#         ("2020", "2020-01-01", "2020-12-31"),
#     ]
    
#     print("=" * 60)
#     print("PRICE_HISTORY: ROW COUNTS BY YEAR")
#     print("=" * 60)
    
#     for label, start, end in ranges:
#         result = supabase.table("price_history") \
#             .select("date", count="exact") \
#             .gte("date", start) \
#             .lte("date", end) \
#             .limit(1) \
#             .execute()
#         count = result.count if result.count else 0
        
#         if count > 0:
#             first = supabase.table("price_history") \
#                 .select("date") \
#                 .gte("date", start) \
#                 .lte("date", end) \
#                 .order("date") \
#                 .limit(1) \
#                 .execute()
#             last = supabase.table("price_history") \
#                 .select("date") \
#                 .gte("date", start) \
#                 .lte("date", end) \
#                 .order("date", desc=True) \
#                 .limit(1) \
#                 .execute()
#             first_date = first.data[0]["date"] if first.data else "?"
#             last_date = last.data[0]["date"] if last.data else "?"
#             print(f"  {label}: {count:>10,} rows  ({first_date} to {last_date})")
#         else:
#             print(f"  {label}: {count:>10,} rows")
#     print()


# def audit_ticker(ticker):
#     """Audit a single ticker across price_history and dm_history."""
#     supabase = get_supabase()
    
#     print(f"=" * 60)
#     print(f"TICKER AUDIT: {ticker}")
#     print(f"=" * 60)
    
#     # Price history
#     result = supabase.table("price_history") \
#         .select("date", count="exact") \
#         .eq("ticker", ticker) \
#         .limit(1) \
#         .execute()
#     price_count = result.count if result.count else 0
    
#     if price_count > 0:
#         first = supabase.table("price_history") \
#             .select("date") \
#             .eq("ticker", ticker) \
#             .order("date") \
#             .limit(1) \
#             .execute()
#         last = supabase.table("price_history") \
#             .select("date") \
#             .eq("ticker", ticker) \
#             .order("date", desc=True) \
#             .limit(1) \
#             .execute()
#         print(f"  price_history: {price_count} rows ({first.data[0]['date']} to {last.data[0]['date']})")
#     else:
#         print(f"  price_history: 0 rows")
    
#     # DM history
#     result = supabase.table("dm_history") \
#         .select("date", count="exact") \
#         .eq("ticker", ticker) \
#         .limit(1) \
#         .execute()
#     dm_count = result.count if result.count else 0
    
#     if dm_count > 0:
#         first = supabase.table("dm_history") \
#             .select("date") \
#             .eq("ticker", ticker) \
#             .order("date") \
#             .limit(1) \
#             .execute()
#         last = supabase.table("dm_history") \
#             .select("date") \
#             .eq("ticker", ticker) \
#             .order("date", desc=True) \
#             .limit(1) \
#             .execute()
#         print(f"  dm_history:    {dm_count} rows ({first.data[0]['date']} to {last.data[0]['date']})")
#     else:
#         print(f"  dm_history:    0 rows")
    
#     # dm_latest
#     result = supabase.table("dm_latest") \
#         .select("date,dm_smoothed,phase,lead") \
#         .eq("ticker", ticker) \
#         .execute()
#     if result.data:
#         r = result.data[0]
#         print(f"  dm_latest:     date={r['date']}, DM={r['dm_smoothed']}, phase={r['phase']}, lead={r['lead']}")
#     else:
#         print(f"  dm_latest:     not found")
    
#     # Check for date gaps in dm_history (sample years)
#     if dm_count > 0:
#         print(f"\n  Rows per year:")
#         for year in range(2016, 2027):
#             result = supabase.table("dm_history") \
#                 .select("date", count="exact") \
#                 .eq("ticker", ticker) \
#                 .gte("date", f"{year}-01-01") \
#                 .lte("date", f"{year}-12-31") \
#                 .limit(1) \
#                 .execute()
#             ycount = result.count if result.count else 0
#             if ycount > 0:
#                 print(f"    {year}: {ycount} rows")
#     print()


# def audit_dm_latest():
#     """Check dm_latest for stale dates."""
#     supabase = get_supabase()
    
#     print("=" * 60)
#     print("DM_LATEST: STALE TICKERS")
#     print("=" * 60)
    
#     # Get the most common (latest) date
#     all_rows = []
#     offset = 0
#     while True:
#         result = supabase.table("dm_latest") \
#             .select("ticker,date") \
#             .order("date") \
#             .range(offset, offset + 999) \
#             .execute()
#         if not result.data:
#             break
#         all_rows.extend(result.data)
#         if len(result.data) < 1000:
#             break
#         offset += 1000
    
#     dates = {}
#     for r in all_rows:
#         d = r["date"]
#         if d not in dates:
#             dates[d] = []
#         dates[d].append(r["ticker"])
    
#     print(f"  Total tickers: {len(all_rows)}")
#     print(f"  Date distribution:")
#     for d in sorted(dates.keys()):
#         tickers = dates[d]
#         if len(tickers) <= 5:
#             print(f"    {d}: {len(tickers)} tickers - {', '.join(tickers)}")
#         else:
#             print(f"    {d}: {len(tickers)} tickers")
#     print()


# def find_short_tickers(threshold=1000):
#     """Find tickers with suspiciously few rows in price_history.
    
#     Full 10-year coverage = ~2500 rows. Anything under threshold
#     likely has a failed backfill or is a newer IPO.
#     """
#     supabase = get_supabase()
    
#     print("=" * 60)
#     print(f"PRICE_HISTORY: TICKERS WITH < {threshold} ROWS (possible backfill gaps)")
#     print("=" * 60)
    
#     # Get all tickers from scanner_universe
#     result = supabase.table("scanner_universe") \
#         .select("ticker,sector") \
#         .order("ticker") \
#         .execute()
#     universe = {r["ticker"]: r.get("sector", "?") for r in result.data} if result.data else {}
    
#     # Also include ETFs
#     from Dm_historical_pipeline import ALL_ETFS
#     for etf in ALL_ETFS:
#         if etf not in universe:
#             universe[etf] = "ETF"
    
#     short_tickers = []
#     no_data_tickers = []
    
#     for ticker in sorted(universe.keys()):
#         result = supabase.table("price_history") \
#             .select("date", count="exact") \
#             .eq("ticker", ticker) \
#             .limit(1) \
#             .execute()
#         count = result.count if result.count else 0
        
#         if count == 0:
#             no_data_tickers.append((ticker, universe[ticker]))
#         elif count < threshold:
#             # Get date range
#             first = supabase.table("price_history") \
#                 .select("date") \
#                 .eq("ticker", ticker) \
#                 .order("date") \
#                 .limit(1) \
#                 .execute()
#             last = supabase.table("price_history") \
#                 .select("date") \
#                 .eq("ticker", ticker) \
#                 .order("date", desc=True) \
#                 .limit(1) \
#                 .execute()
#             first_d = first.data[0]["date"] if first.data else "?"
#             last_d = last.data[0]["date"] if last.data else "?"
#             short_tickers.append((ticker, count, first_d, last_d, universe[ticker]))
    
#     if no_data_tickers:
#         print(f"\n  NO DATA ({len(no_data_tickers)} tickers):")
#         for ticker, sector in no_data_tickers:
#             print(f"    {ticker:>6} ({sector})")
    
#     if short_tickers:
#         print(f"\n  SHORT HISTORY ({len(short_tickers)} tickers, < {threshold} rows):")
#         print(f"    {'Ticker':>6}  {'Rows':>5}  {'First':>10}  {'Last':>10}  Sector")
#         print(f"    {'------':>6}  {'-----':>5}  {'----------':>10}  {'----------':>10}  ------")
#         for ticker, count, first_d, last_d, sector in sorted(short_tickers, key=lambda x: x[1]):
#             expected = 2520  # ~10 years
#             missing_pct = (1 - count / expected) * 100
#             print(f"    {ticker:>6}  {count:>5}  {first_d:>10}  {last_d:>10}  {sector}  ({missing_pct:.0f}% missing)")
#     else:
#         print("  All tickers have >= {threshold} rows.")
    
#     total_short = len(no_data_tickers) + len(short_tickers)
#     print(f"\n  Summary: {total_short} tickers need attention out of {len(universe)}")
#     print()


# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "ticker":
#         ticker = sys.argv[2].upper()
#         audit_ticker(ticker)
#     elif len(sys.argv) > 1 and sys.argv[1] == "short":
#         threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
#         find_short_tickers(threshold)
#     else:
#         print("\nRunning full DM data audit...\n")
#         count_by_date_range()
#         count_trading_days()
#         check_price_history()
#         audit_dm_latest()
#         find_short_tickers()
        
#         # Spot-check a few tickers
#         for t in ["NVDA", "AAPL", "SPOT", "CCJ", "MSTR"]:
#             audit_ticker(t)
        
#         print("Done.")
#         print("  python dm_audit.py ticker SYMBOL  - audit specific ticker")
#         print("  python dm_audit.py short 500      - find tickers with < 500 rows")


"""
Fill missing SPOT rows in DM_2024_2026 sheet.
Fetches SPOT data from Supabase dm_history and appends any dates 
missing from the Google Sheet.

Usage: python fill_spot_gaps.py
"""

import os
import time
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SPREADSHEET_NAME = "dm-history 2024-current"
TAB_NAME = "DM_2024_2026"

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

HISTORY_FIELDS = [
    'date', 'ticker', 'close', 'volume', 'return_20d',
    'etf_return_20d', 'spy_return_20d', 'vol_20d_avg',
    'vol_ratio', 'rel_str_etf', 'rel_str_spy', 'volume_z',
    'dm_raw', 'dm_smoothed', 'etf', 'etf_dm', 'lead', 'phase'
]


def main():
    # Connect to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Get all SPOT rows from Supabase for 2024+
    print("Fetching SPOT from Supabase dm_history...")
    result = supabase.table("dm_history") \
        .select(",".join(HISTORY_FIELDS)) \
        .eq("ticker", "SPOT") \
        .gte("date", "2024-01-01") \
        .order("date") \
        .execute()
    
    supa_rows = {r['date']: r for r in result.data}
    print(f"  Supabase has {len(supa_rows)} SPOT rows")
    
    # Connect to Google Sheets
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open(SPREADSHEET_NAME)
    sheet = spreadsheet.worksheet(TAB_NAME)
    
    # Find existing SPOT dates in sheet
    print("Scanning sheet for existing SPOT rows...")
    last_row = sheet.row_count
    all_tickers = sheet.col_values(2)  # Ticker column
    all_dates = sheet.col_values(1)    # Date column
    
    sheet_spot_dates = set()
    for i in range(1, len(all_tickers)):  # skip header
        if all_tickers[i] == "SPOT":
            sheet_spot_dates.add(all_dates[i])
    
    print(f"  Sheet has {len(sheet_spot_dates)} SPOT rows")
    
    # Find missing dates
    missing_dates = sorted(set(supa_rows.keys()) - sheet_spot_dates)
    
    if not missing_dates:
        print("  No missing dates. Sheet is current.")
        return
    
    print(f"  Missing {len(missing_dates)} dates: {', '.join(missing_dates)}")
    
    # Build rows to append
    rows_to_add = []
    for date in missing_dates:
        r = supa_rows[date]
        row = [r.get(f, '') for f in HISTORY_FIELDS]
        rows_to_add.append(row)
    
    # Append to sheet
    last_data_row = len(all_tickers)
    start_row = last_data_row + 1
    
    print(f"  Appending {len(rows_to_add)} rows at row {start_row}...")
    sheet.update(
        range_name=f'A{start_row}',
        values=rows_to_add,
        value_input_option='USER_ENTERED'
    )
    
    print("  Done! Missing SPOT rows filled.")
    print()
    for date in missing_dates:
        r = supa_rows[date]
        print(f"  Added: {date} DM={r['dm_smoothed']} phase={r['phase']}")


if __name__ == "__main__":
    main()