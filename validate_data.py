#!/usr/bin/env python3
"""
DATA VALIDATION: Compare GS vs Supabase
========================================
Pulls raw data from all MCI component tables and outputs in a format
that can be easily compared with Google Sheets.

Usage:
    python validate_data.py                    # Last 10 days
    python validate_data.py --days 30          # Last 30 days
    python validate_data.py --csv              # Export to CSV
    python validate_data.py --component vix    # Only VIX data
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from supabase import create_client, Client

logging.basicConfig(level=logging.WARNING)

# =============================================================================
# SUPABASE
# =============================================================================

def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY required")
    return create_client(url, key)

# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_vix_data(supabase: Client, days: int) -> pd.DataFrame:
    """Fetch VIX dial data."""
    response = supabase.table("vix_dial_daily") \
        .select("date, vix, ma_20, trend_20d, combined_regime") \
        .order("date", desc=True) \
        .limit(days) \
        .execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def fetch_credit_data(supabase: Client, days: int) -> pd.DataFrame:
    """Fetch Credit spread data."""
    response = supabase.table("credit_spread_daily") \
        .select("date, hy_spread, ig_spread, hy_20d_change, combined_regime") \
        .order("date", desc=True) \
        .limit(days) \
        .execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def fetch_breadth_data(supabase: Client, days: int) -> pd.DataFrame:
    """Fetch Breadth data."""
    response = supabase.table("breadth_daily") \
        .select("date, daily_breadth, breadth_20d, breadth_50d, breadth_momentum, advancers, decliners, regime") \
        .order("date", desc=True) \
        .limit(days) \
        .execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def fetch_pillar_data(supabase: Client, days: int) -> pd.DataFrame:
    """Fetch Pillar index data."""
    response = supabase.table("pillar_index_daily") \
        .select("date, infra_5d, enterprise_5d, macro_5d, financial_5d, productivity_5d, demand_5d") \
        .order("date", desc=True) \
        .limit(days) \
        .execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    
    # Calculate average 5D (as percentage)
    pillars = ["infra_5d", "enterprise_5d", "macro_5d", "financial_5d", "productivity_5d", "demand_5d"]
    df["avg_5d_pct"] = df[pillars].mean(axis=1) * 100
    
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def fetch_mci_data(supabase: Client, days: int) -> pd.DataFrame:
    """Fetch MCI data."""
    response = supabase.table("mci_daily") \
        .select("date, mci_score, regime, breadth_component, vix_component, credit_component, pillar_component") \
        .order("date", desc=True) \
        .limit(days) \
        .execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def fetch_price_history(supabase: Client, ticker: str, days: int) -> pd.DataFrame:
    """Fetch price history for a specific ticker."""
    response = supabase.table("price_history") \
        .select("date, ticker, open, high, low, close, volume") \
        .eq("ticker", ticker) \
        .order("date", desc=True) \
        .limit(days) \
        .execute()
    
    if not response.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_section(title: str):
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)


def display_vix(df: pd.DataFrame):
    print_section("VIX DATA (vix_dial_daily)")
    print("Compare with: FRED VIXCLS series")
    print()
    print(f"{'Date':<12} {'VIX':>8} {'MA20':>8} {'Trend20D':>10} {'Regime':<12}")
    print("-" * 60)
    for _, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        vix = row["vix"] if pd.notna(row["vix"]) else 0
        ma20 = row["ma_20"] if pd.notna(row["ma_20"]) else 0
        trend = row["trend_20d"] if pd.notna(row["trend_20d"]) else 0
        regime = row["combined_regime"] or ""
        print(f"{date:<12} {vix:>8.2f} {ma20:>8.2f} {trend:>+10.2f} {regime:<12}")


def display_credit(df: pd.DataFrame):
    print_section("CREDIT SPREAD DATA (credit_spread_daily)")
    print("Compare with: FRED BAMLH0A0HYM2 (HY) and BAMLC0A0CM (IG)")
    print()
    print(f"{'Date':<12} {'HY Spread':>10} {'IG Spread':>10} {'HY 20D Chg':>12} {'Regime':<12}")
    print("-" * 70)
    for _, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        hy = row["hy_spread"] if pd.notna(row["hy_spread"]) else 0
        ig = row["ig_spread"] if pd.notna(row["ig_spread"]) else 0
        hy_chg = row["hy_20d_change"] if pd.notna(row["hy_20d_change"]) else 0
        regime = row["combined_regime"] or ""
        print(f"{date:<12} {hy:>10.2f}% {ig:>10.2f}% {hy_chg:>+12.2f}% {regime:<12}")


def display_breadth(df: pd.DataFrame):
    print_section("BREADTH DATA (breadth_daily)")
    print("Compare with: GS Breadth calculation from GOOGLEFINANCE prices")
    print()
    print(f"{'Date':<12} {'Daily':>8} {'20D':>8} {'50D':>8} {'Momentum':>10} {'Adv':>5} {'Dec':>5} {'Regime':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        daily = row["daily_breadth"] if pd.notna(row["daily_breadth"]) else 0
        b20d = row["breadth_20d"] if pd.notna(row["breadth_20d"]) else 0
        b50d = row["breadth_50d"] if pd.notna(row["breadth_50d"]) else 0
        mom = row["breadth_momentum"] if pd.notna(row["breadth_momentum"]) else 0
        adv = int(row["advancers"]) if pd.notna(row["advancers"]) else 0
        dec = int(row["decliners"]) if pd.notna(row["decliners"]) else 0
        regime = row["regime"] or ""
        print(f"{date:<12} {daily:>7.1%} {b20d:>7.1%} {b50d:>7.1%} {mom:>+10.1%} {adv:>5} {dec:>5} {regime:<10}")


def display_pillar(df: pd.DataFrame):
    print_section("PILLAR 5D RETURNS (pillar_index_daily)")
    print("Compare with: GS Pillar_History sheet 5D momentum columns")
    print("Values shown as percentages")
    print()
    print(f"{'Date':<12} {'Infra':>8} {'Enter':>8} {'Macro':>8} {'Finan':>8} {'Prod':>8} {'Demand':>8} {'AVG':>8}")
    print("-" * 90)
    for _, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        infra = (row["infra_5d"] or 0) * 100
        enter = (row["enterprise_5d"] or 0) * 100
        macro = (row["macro_5d"] or 0) * 100
        finan = (row["financial_5d"] or 0) * 100
        prod = (row["productivity_5d"] or 0) * 100
        demand = (row["demand_5d"] or 0) * 100
        avg = row["avg_5d_pct"] if pd.notna(row["avg_5d_pct"]) else 0
        print(f"{date:<12} {infra:>+7.2f}% {enter:>+7.2f}% {macro:>+7.2f}% {finan:>+7.2f}% {prod:>+7.2f}% {demand:>+7.2f}% {avg:>+7.2f}%")


def display_mci(df: pd.DataFrame):
    print_section("MCI DATA (mci_daily)")
    print("Compare with: GS MCI_Dial HISTORY section")
    print()
    print(f"{'Date':<12} {'MCI':>8} {'Breadth':>8} {'VIX':>8} {'Credit':>8} {'Pillar':>8} {'Regime':<15}")
    print("-" * 80)
    for _, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        mci = row["mci_score"] if pd.notna(row["mci_score"]) else 0
        breadth = row["breadth_component"] if pd.notna(row["breadth_component"]) else 0
        vix = row["vix_component"] if pd.notna(row["vix_component"]) else 0
        credit = row["credit_component"] if pd.notna(row["credit_component"]) else 0
        pillar = row["pillar_component"] if pd.notna(row["pillar_component"]) else 0
        regime = row["regime"] or ""
        print(f"{date:<12} {mci:>+8.1f} {breadth:>+8.1f} {vix:>+8.1f} {credit:>+8.1f} {pillar:>+8.1f} {regime:<15}")


def display_prices(df: pd.DataFrame, ticker: str):
    print_section(f"PRICE HISTORY: {ticker} (price_history)")
    print("Compare with: Yahoo Finance or GOOGLEFINANCE")
    print()
    print(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print("-" * 70)
    for _, row in df.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        print(f"{date:<12} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} {row['close']:>10.2f} {row['volume']:>12,.0f}")

# =============================================================================
# CSV EXPORT
# =============================================================================

def export_to_csv(supabase: Client, days: int, output_dir: str = "."):
    """Export all data to CSV files for comparison."""
    
    print("Exporting data to CSV files...")
    
    vix_df = fetch_vix_data(supabase, days)
    if not vix_df.empty:
        vix_df.to_csv(f"{output_dir}/supabase_vix.csv", index=False)
        print(f"  ✓ supabase_vix.csv ({len(vix_df)} rows)")
    
    credit_df = fetch_credit_data(supabase, days)
    if not credit_df.empty:
        credit_df.to_csv(f"{output_dir}/supabase_credit.csv", index=False)
        print(f"  ✓ supabase_credit.csv ({len(credit_df)} rows)")
    
    breadth_df = fetch_breadth_data(supabase, days)
    if not breadth_df.empty:
        breadth_df.to_csv(f"{output_dir}/supabase_breadth.csv", index=False)
        print(f"  ✓ supabase_breadth.csv ({len(breadth_df)} rows)")
    
    pillar_df = fetch_pillar_data(supabase, days)
    if not pillar_df.empty:
        pillar_df.to_csv(f"{output_dir}/supabase_pillar.csv", index=False)
        print(f"  ✓ supabase_pillar.csv ({len(pillar_df)} rows)")
    
    mci_df = fetch_mci_data(supabase, days)
    if not mci_df.empty:
        mci_df.to_csv(f"{output_dir}/supabase_mci.csv", index=False)
        print(f"  ✓ supabase_mci.csv ({len(mci_df)} rows)")
    
    print("\nDone! Import these CSVs into GS to compare side-by-side.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Supabase data against GS")
    parser.add_argument("--days", type=int, default=10, help="Days of data to show")
    parser.add_argument("--csv", action="store_true", help="Export to CSV files")
    parser.add_argument("--component", type=str, choices=["vix", "credit", "breadth", "pillar", "mci", "all"], 
                        default="all", help="Which component to show")
    parser.add_argument("--ticker", type=str, help="Check price history for specific ticker (e.g., NVDA)")
    args = parser.parse_args()
    
    print()
    print("=" * 80)
    print(" DATA VALIDATION: Supabase vs Google Sheets")
    print("=" * 80)
    print(f" Days: {args.days}")
    print(" Use this output to compare with your GS sheets")
    print("=" * 80)
    
    supabase = get_supabase()
    
    if args.csv:
        export_to_csv(supabase, args.days)
        return
    
    if args.ticker:
        df = fetch_price_history(supabase, args.ticker.upper(), args.days)
        if not df.empty:
            display_prices(df, args.ticker.upper())
        else:
            print(f"No price data found for {args.ticker}")
        return
    
    # Show requested components
    if args.component in ["vix", "all"]:
        df = fetch_vix_data(supabase, args.days)
        if not df.empty:
            display_vix(df)
    
    if args.component in ["credit", "all"]:
        df = fetch_credit_data(supabase, args.days)
        if not df.empty:
            display_credit(df)
    
    if args.component in ["breadth", "all"]:
        df = fetch_breadth_data(supabase, args.days)
        if not df.empty:
            display_breadth(df)
    
    if args.component in ["pillar", "all"]:
        df = fetch_pillar_data(supabase, args.days)
        if not df.empty:
            display_pillar(df)
    
    if args.component in ["mci", "all"]:
        df = fetch_mci_data(supabase, args.days)
        if not df.empty:
            display_mci(df)
    
    print()
    print("=" * 80)
    print(" NEXT STEPS")
    print("=" * 80)
    print(" 1. Compare VIX values with FRED: https://fred.stlouisfed.org/series/VIXCLS")
    print(" 2. Compare Credit spreads with FRED: https://fred.stlouisfed.org/series/BAMLH0A0HYM2")
    print(" 3. Compare Pillar 5D returns with GS Pillar_History sheet")
    print(" 4. Run with --csv to export and import into GS for side-by-side comparison")
    print("=" * 80)

if __name__ == "__main__":
    main()