#!/usr/bin/env python3
"""
MCI FIX - Corrected Formulas
============================
Fixes:
1. PILLAR_THRESHOLD should be 3 (percentage), not 25
2. Pillar raw is stored as decimal (0.0194), needs *100 for percentage
3. Validates VIX data against expected values

Usage:
    python fix_mci.py
    python fix_mci.py --validate-only   # Just show diagnostics
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
import numpy as np

from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CORRECTED CONFIG (matching Google Sheets)
# =============================================================================

MCI_CONFIG = {
    # Weights (each component contributes up to ±25)
    "WEIGHT_BREADTH": 25,
    "WEIGHT_VIX": 25,
    "WEIGHT_CREDIT": 25,
    "WEIGHT_PILLAR": 25,
    
    # Thresholds for normalization
    "VIX_THRESHOLD": 3,       # VIX change of ±3 pts = ±25 score
    "CREDIT_THRESHOLD": 30,   # Credit change of ±30 bps = ±25 score
    "PILLAR_THRESHOLD": 3,    # FIXED: Pillar avg 5D of ±3% = ±25 score
    
    # Lookback periods
    "BREADTH_LOOKBACK": 5,
    "VIX_LOOKBACK": 10,
    "CREDIT_LOOKBACK": 10,
}

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
# VALIDATION / DIAGNOSTICS
# =============================================================================

def validate_data(supabase: Client, target_date: str = "2026-01-19"):
    """Show raw component data for a specific date to diagnose issues."""
    
    print()
    print("=" * 70)
    print(f"DIAGNOSTICS FOR {target_date}")
    print("=" * 70)
    
    config = MCI_CONFIG
    lookback = config["VIX_LOOKBACK"]
    
    # 1. VIX Data
    print("\n--- VIX DATA ---")
    vix_response = supabase.table("vix_dial_daily") \
        .select("date, vix") \
        .order("date", desc=True) \
        .limit(20) \
        .execute()
    
    if vix_response.data:
        vix_df = pd.DataFrame(vix_response.data)
        vix_df = vix_df.sort_values("date", ascending=False).reset_index(drop=True)
        
        # Find target date
        target_idx = None
        for i, row in vix_df.iterrows():
            if row["date"] == target_date:
                target_idx = i
                break
        
        if target_idx is not None and target_idx + lookback <= len(vix_df):
            current_vix = vix_df.iloc[target_idx]["vix"]
            prev_vix = vix_df.iloc[target_idx + lookback - 1]["vix"]
            prev_date = vix_df.iloc[target_idx + lookback - 1]["date"]
            vix_change = current_vix - prev_vix
            
            print(f"  Current ({target_date}): {current_vix:.2f}")
            print(f"  10D Ago ({prev_date}):   {prev_vix:.2f}")
            print(f"  Change:                  {vix_change:+.2f}")
            
            vix_norm = max(-1, min(1, -vix_change / config["VIX_THRESHOLD"]))
            vix_score = round(vix_norm * config["WEIGHT_VIX"], 1)
            print(f"  Calculated Score:        {vix_score}")
            print(f"  GS Expected:             -3.6")
        else:
            print(f"  Target date {target_date} not found or not enough history")
            print(f"  Available dates: {vix_df['date'].tolist()[:5]}...")
    
    # 2. Credit Data
    print("\n--- CREDIT DATA ---")
    credit_response = supabase.table("credit_spread_daily") \
        .select("date, hy_spread") \
        .order("date", desc=True) \
        .limit(20) \
        .execute()
    
    if credit_response.data:
        credit_df = pd.DataFrame(credit_response.data)
        credit_df = credit_df.sort_values("date", ascending=False).reset_index(drop=True)
        
        target_idx = None
        for i, row in credit_df.iterrows():
            if row["date"] == target_date:
                target_idx = i
                break
        
        if target_idx is not None and target_idx + lookback <= len(credit_df):
            current_spread = credit_df.iloc[target_idx]["hy_spread"]
            prev_spread = credit_df.iloc[target_idx + lookback - 1]["hy_spread"]
            prev_date = credit_df.iloc[target_idx + lookback - 1]["date"]
            spread_change_bps = (current_spread - prev_spread) * 100
            
            print(f"  Current ({target_date}): {current_spread:.2f}%")
            print(f"  10D Ago ({prev_date}):   {prev_spread:.2f}%")
            print(f"  Change:                  {spread_change_bps:+.0f} bps")
            
            credit_norm = max(-1, min(1, -spread_change_bps / config["CREDIT_THRESHOLD"]))
            credit_score = round(credit_norm * config["WEIGHT_CREDIT"], 1)
            print(f"  Calculated Score:        {credit_score}")
            print(f"  GS Expected:             10")
    
    # 3. Breadth Data
    print("\n--- BREADTH DATA ---")
    breadth_response = supabase.table("breadth_daily") \
        .select("date, breadth_20d") \
        .eq("date", target_date) \
        .execute()
    
    if breadth_response.data:
        breadth_20d = breadth_response.data[0]["breadth_20d"]
        print(f"  Breadth 20D: {breadth_20d:.2%}")
        
        breadth_norm = max(-1, min(1, (breadth_20d - 0.5) / 0.5))
        breadth_score = round(breadth_norm * config["WEIGHT_BREADTH"], 1)
        print(f"  Calculated Score: {breadth_score}")
        print(f"  GS Expected:      0")
    
    # 4. Pillar Data
    print("\n--- PILLAR DATA ---")
    pillar_response = supabase.table("pillar_index_daily") \
        .select("date, infra_5d, enterprise_5d, macro_5d, financial_5d, productivity_5d, demand_5d") \
        .eq("date", target_date) \
        .execute()
    
    if pillar_response.data:
        row = pillar_response.data[0]
        pillars = ["infra", "enterprise", "financial", "demand"]  
        
        print("  5D Returns (stored as decimal):")
        values = []
        for p in pillars:
            val = row.get(f"{p}_5d", 0) or 0
            values.append(val)
            print(f"    {p:15}: {val:.4f} ({val*100:.2f}%)")
        
        avg_5d_decimal = sum(values) / len(values)
        avg_5d_pct = avg_5d_decimal * 100
        
        print(f"\n  Average 5D (decimal): {avg_5d_decimal:.4f}")
        print(f"  Average 5D (percent): {avg_5d_pct:.2f}%")
        
        # CORRECTED formula: convert to %, divide by 3
        pillar_norm = max(-1, min(1, avg_5d_pct / config["PILLAR_THRESHOLD"]))
        pillar_score = round(pillar_norm * config["WEIGHT_PILLAR"], 1)
        print(f"  Calculated Score:     {pillar_score}")
        print(f"  GS Expected:          -16.2")
        print(f"  GS Raw:               -1.94%")
    
    print("\n" + "=" * 70)

# =============================================================================
# CORRECTED MCI BACKFILL
# =============================================================================

def backfill_mci_corrected(supabase: Client, days: int = 252) -> int:
    """Backfill MCI with CORRECTED formulas."""
    
    logger.info("=== MCI BACKFILL (CORRECTED FORMULAS) ===")
    
    config = MCI_CONFIG
    lookback = config["VIX_LOOKBACK"]
    
    # Load component data
    vix_data = supabase.table("vix_dial_daily") \
        .select("date, vix") \
        .order("date", desc=True) \
        .limit(days + 20) \
        .execute()
    
    credit_data = supabase.table("credit_spread_daily") \
        .select("date, hy_spread") \
        .order("date", desc=True) \
        .limit(days + 20) \
        .execute()
    
    breadth_data = supabase.table("breadth_daily") \
        .select("date, breadth_20d") \
        .order("date", desc=True) \
        .limit(days + 10) \
        .execute()
    
    pillar_data = supabase.table("pillar_index_daily") \
        .select("date, infra_5d, enterprise_5d, macro_5d, financial_5d, productivity_5d, demand_5d") \
        .order("date", desc=True) \
        .limit(days + 10) \
        .execute()
    
    if not all([vix_data.data, credit_data.data, breadth_data.data, pillar_data.data]):
        logger.error("Missing component data")
        return 0
    
    # Build lookups
    vix_lookup = {row["date"]: row["vix"] for row in vix_data.data}
    credit_lookup = {row["date"]: row["hy_spread"] for row in credit_data.data}
    breadth_lookup = {row["date"]: row["breadth_20d"] for row in breadth_data.data}
    pillar_lookup = {row["date"]: row for row in pillar_data.data}
    
    # Get dates from VIX (sorted descending)
    vix_dates = sorted(vix_lookup.keys(), reverse=True)
    
    logger.info(f"Processing {min(days, len(vix_dates) - lookback)} dates...")
    
    saved = 0
    
    for i, date in enumerate(vix_dates[:days]):
        if i + lookback >= len(vix_dates):
            break
        
        # Get 10D-ago date (using LOOKBACK - 1 to match GS)
        prev_date = vix_dates[i + lookback - 1]
        
        # Get raw values
        current_vix = vix_lookup.get(date)
        previous_vix = vix_lookup.get(prev_date)
        current_credit = credit_lookup.get(date)
        previous_credit = credit_lookup.get(prev_date)
        breadth_20d = breadth_lookup.get(date)
        pillar_row = pillar_lookup.get(date, {})
        
        # Skip if missing critical data
        if not all([current_vix, previous_vix]):
            continue
        
        # Default values
        if breadth_20d is None:
            breadth_20d = 0.5
        if current_credit is None or previous_credit is None:
            current_credit = previous_credit = 0
        
        # =================================================================
        # COMPONENT 1: BREADTH
        # Formula: (breadth_20d - 0.5) / 0.5 * 25, clamped to ±25
        # =================================================================
        breadth_norm = max(-1, min(1, (breadth_20d - 0.5) / 0.5))
        breadth_component = round(breadth_norm * config["WEIGHT_BREADTH"], 1)
        
        # =================================================================
        # COMPONENT 2: VIX TREND
        # Formula: -vix_change / 3 * 25, clamped to ±25
        # (negative because VIX drop is bullish)
        # =================================================================
        vix_change = current_vix - previous_vix
        vix_norm = max(-1, min(1, -vix_change / config["VIX_THRESHOLD"]))
        vix_component = round(vix_norm * config["WEIGHT_VIX"], 1)
        
        # =================================================================
        # COMPONENT 3: CREDIT TREND
        # Formula: -spread_change_bps / 30 * 25, clamped to ±25
        # (negative because tightening is bullish)
        # =================================================================
        spread_change_bps = (current_credit - previous_credit) * 100
        credit_norm = max(-1, min(1, -spread_change_bps / config["CREDIT_THRESHOLD"]))
        credit_component = round(credit_norm * config["WEIGHT_CREDIT"], 1)
        
        # =================================================================
        # COMPONENT 4: PILLAR MOMENTUM (CORRECTED!)
        # Formula: avg_5d_pct / 3 * 25, clamped to ±25
        # Raw values are stored as decimals (0.0194), need *100 for %
        # =================================================================
        pillars = ["infra", "enterprise", "financial", "demand"]  
        pillar_5d_values = [pillar_row.get(f"{p}_5d", 0) or 0 for p in pillars]
        avg_5d_decimal = sum(pillar_5d_values) / len(pillar_5d_values) if pillar_5d_values else 0
        avg_5d_pct = avg_5d_decimal * 100  # Convert to percentage
        
        # CORRECTED: divide by 3 (not 25)
        pillar_norm = max(-1, min(1, avg_5d_pct / config["PILLAR_THRESHOLD"]))
        pillar_component = round(pillar_norm * config["WEIGHT_PILLAR"], 1)
        
        # =================================================================
        # TOTAL MCI SCORE
        # =================================================================
        mci_score = breadth_component + vix_component + credit_component + pillar_component
        
        # Regime classification
        if mci_score >= 40:
            regime = "Melt-Up"
        elif mci_score >= 10:
            regime = "Extension"
        elif mci_score >= -10:
            regime = "Knife Edge"
        elif mci_score >= -40:
            regime = "Collapse Bias"
        else:
            regime = "Break Path"
        
        # Save to Supabase
        try:
            row = {
                "date": date,
                "mci_score": round(mci_score, 1),
                "regime": regime,
                "interpretation": f"MCI {mci_score:.1f} ({regime})",
                "breadth_component": breadth_component,
                "vix_component": vix_component,
                "credit_component": credit_component,
                "pillar_component": pillar_component,
                "breadth_raw": breadth_20d,
                "vix_raw": current_vix,
                "credit_raw": current_credit,
                "pillar_raw": avg_5d_pct  # Store as percentage now
            }
            supabase.table("mci_daily").upsert(row, on_conflict="date").execute()
            saved += 1
        except Exception as e:
            logger.warning(f"Save failed for {date}: {e}")
    
    logger.info(f"✓ Saved {saved} rows to mci_daily")
    return saved

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fix MCI with corrected formulas")
    parser.add_argument("--days", type=int, default=252, help="Days to backfill")
    parser.add_argument("--validate-only", action="store_true", help="Only show diagnostics")
    parser.add_argument("--date", type=str, default="2026-01-17", help="Date to validate")
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("MCI FIX - CORRECTED FORMULAS")
    print("=" * 60)
    print("Fixes applied:")
    print("  1. PILLAR_THRESHOLD: 25 → 3")
    print("  2. Pillar raw * 100 for percentage conversion")
    print("  3. Regime names match GS (Melt-Up, Extension, etc.)")
    print("=" * 60)
    print()
    
    supabase = get_supabase()
    logger.info("Connected to Supabase")
    
    # Always run validation first
    validate_data(supabase, args.date)
    
    if args.validate_only:
        print("\n[VALIDATE ONLY] No changes made.")
        return
    
    # Backfill with corrected formulas
    print()
    saved = backfill_mci_corrected(supabase, args.days)
    
    # Show result for target date
    print()
    print("=" * 60)
    print(f"VERIFICATION - Checking {args.date}")
    print("=" * 60)
    
    result = supabase.table("mci_daily") \
        .select("*") \
        .eq("date", args.date) \
        .execute()
    
    if result.data:
        row = result.data[0]
        print(f"\n  MCI Score:          {row['mci_score']}")
        print(f"  Regime:             {row['regime']}")
        print(f"  Breadth Component:  {row['breadth_component']}")
        print(f"  VIX Component:      {row['vix_component']}")
        print(f"  Credit Component:   {row['credit_component']}")
        print(f"  Pillar Component:   {row['pillar_component']}")
        
        print(f"\n  GS Expected:")
        print(f"    MCI: -9.8, Breadth: 0, VIX: -3.6, Credit: 10, Pillar: -16.2")
    
    print()
    print("=" * 60)
    print("Done!")

    main()