"""
Build a substitute ETF_Reference.csv for wave_r_test2_sector.py from
scanner_universe (the real per-ticker sector table).

Output columns expected by Test 2: Ticker, Sector_Name, Current_ETF.
Current_ETF is filled with a sector->sector-ETF mapping where known;
blank otherwise (Test 2 groups by Sector_Name, so Current_ETF is optional).
"""
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

rows, offset, page = [], 0, 1000
while True:
    r = sb.table("scanner_universe").select("ticker,sector").range(
        offset, offset + page - 1).execute()
    if not r.data:
        break
    rows.extend(r.data)
    if len(r.data) < page:
        break
    offset += page

df = pd.DataFrame(rows)
print(f"Loaded {len(df):,} tickers from scanner_universe")
print(f"Sectors: {sorted(df['sector'].dropna().unique())}")

# GICS sector -> canonical sector ETF (SPDR Select Sector). May not all exist
# in etf_flows_history, but Test 2 only needs Current_ETF for labelling.
SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Financials":             "XLF",
    "Financial Services":     "XLF",
    "Health Care":            "XLV",
    "Industrials":            "XLI",
    "Information Technology": "XLK",
    "Technology":             "XLK",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}

df["Current_ETF"] = df["sector"].map(SECTOR_TO_ETF).fillna("")
out = df.rename(columns={"ticker": "Ticker", "sector": "Sector_Name"})[
    ["Ticker", "Sector_Name", "Current_ETF"]]
out.to_csv("ETF_Reference.csv", index=False)
print(f"Wrote ETF_Reference.csv ({len(out):,} rows)")
