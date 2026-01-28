#!/usr/bin/env python3
"""
ARGUS-1 COMPLETE DIAL RESET & BACKFILL
======================================
Handles ALL 35 Supabase dial tables in correct dependency order.

Tables organized by level:
- Level 0: Foundation (price_history, history tables)
- Level 1: Base Dials (vix, credit, breadth, pillar, etc.)
- Level 2: Composite Dials (mci, amri, bubble_index, etc.)
- Level 3: Signals & Reports (dashboard, morning_brief, etc.)

Usage:
    python reset_all_dials.py                    # Full reset (clear all + backfill)
    python reset_all_dials.py --dry-run          # Preview changes
    python reset_all_dials.py --level 1          # Only reset Level 1 dials
    python reset_all_dials.py --tables mci vix   # Only reset specific tables
    python reset_all_dials.py --skip-clear       # Backfill without clearing

Author: Claude + Vikram
Date: 2026-01-21
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

from supabase import create_client, Client

# Add current directory to path for y2ai imports
import sys
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TABLE DEFINITIONS BY LEVEL
# =============================================================================

@dataclass
class TableDef:
    """Definition of a Supabase table."""
    name: str
    level: int
    python_module: str
    description: str
    has_backfill: bool = True
    dependencies: List[str] = None

# ALL TABLES organized by dependency level
ALL_TABLES = {
    # =========================================================================
    # LEVEL 0: FOUNDATION (Raw data - usually don't clear these)
    # =========================================================================
    "price_history": TableDef(
        name="price_history",
        level=0,
        python_module="stock_tracker",
        description="43 stock prices from TwelveData",
        has_backfill=True,
        dependencies=[]
    ),
    "vix_history": TableDef(
        name="vix_history",
        level=0,
        python_module="vix_dial",
        description="Raw VIX data from FRED",
        has_backfill=True,
        dependencies=[]
    ),
    "credit_spread_history": TableDef(
        name="credit_spread_history",
        level=0,
        python_module="credit_spread_dial",
        description="Raw credit spread data from FRED",
        has_backfill=True,
        dependencies=[]
    ),
    "financial_stress_history": TableDef(
        name="financial_stress_history",
        level=0,
        python_module="financial_stress_dial",
        description="Financial stress indicators",
        has_backfill=True,
        dependencies=[]
    ),
    "articles": TableDef(
        name="articles",
        level=0,
        python_module="nst_dial",
        description="News articles for sentiment",
        has_backfill=False,
        dependencies=[]
    ),
    "processed_articles": TableDef(
        name="processed_articles",
        level=0,
        python_module="nst_dial",
        description="Processed news articles",
        has_backfill=False,
        dependencies=[]
    ),
    
    # =========================================================================
    # LEVEL 1: BASE DIALS (Calculated from foundation data)
    # =========================================================================
    "vix_dial_daily": TableDef(
        name="vix_dial_daily",
        level=1,
        python_module="vix_dial",
        description="VIX analysis with Bollinger Bands",
        has_backfill=True,
        dependencies=["vix_history"]
    ),
    "credit_spread_daily": TableDef(
        name="credit_spread_daily",
        level=1,
        python_module="credit_spread_dial",
        description="Credit spread momentum",
        has_backfill=True,
        dependencies=["credit_spread_history"]
    ),
    "breadth_daily": TableDef(
        name="breadth_daily",
        level=1,
        python_module="breadth_dial",
        description="Market breadth (% above MA)",
        has_backfill=True,
        dependencies=["price_history"]
    ),
    "pillar_index_daily": TableDef(
        name="pillar_index_daily",
        level=1,
        python_module="pillar_index",
        description="6 pillar returns & momentum",
        has_backfill=True,
        dependencies=["price_history"]
    ),
    "correlation_daily": TableDef(
        name="correlation_daily",
        level=1,
        python_module="correlation_dial",
        description="Cross-asset correlation",
        has_backfill=True,
        dependencies=["price_history"]
    ),
    "macro_dial_daily": TableDef(
        name="macro_dial_daily",
        level=1,
        python_module="macro_dial",
        description="Macro economic indicators",
        has_backfill=True,
        dependencies=[]
    ),
    "labor_dial_daily": TableDef(
        name="labor_dial_daily",
        level=1,
        python_module="labor_dial",
        description="Labor market indicators",
        has_backfill=True,
        dependencies=[]
    ),
    "liquidity_dial_daily": TableDef(
        name="liquidity_dial_daily",
        level=1,
        python_module="liquidity_dial",
        description="Market liquidity indicators",
        has_backfill=True,
        dependencies=[]
    ),
    "financial_stress_daily": TableDef(
        name="financial_stress_daily",
        level=1,
        python_module="financial_stress_dial",
        description="Financial stress index",
        has_backfill=True,
        dependencies=["financial_stress_history"]
    ),
    "etf_dial_daily": TableDef(
        name="etf_dial_daily",
        level=1,
        python_module="etf_dial",
        description="ETF flow analysis",
        has_backfill=True,
        dependencies=["price_history"]
    ),
    "sentiment_dial_daily": TableDef(
        name="sentiment_dial_daily",
        level=1,
        python_module="sentiment_dial",
        description="Market sentiment",
        has_backfill=True,
        dependencies=[]
    ),
    "nst_dial_daily": TableDef(
        name="nst_dial_daily",
        level=1,
        python_module="nst_dial",
        description="News/Sentiment/Trends",
        has_backfill=True,
        dependencies=["articles"]
    ),
    "stock_flow_dial_daily": TableDef(
        name="stock_flow_dial_daily",
        level=1,
        python_module="stock_flow_dial",
        description="Stock flow analysis",
        has_backfill=True,
        dependencies=["price_history"]
    ),
    "macro_multipliers_daily": TableDef(
        name="macro_multipliers_daily",
        level=1,
        python_module="macro_multipliers",
        description="Macro multiplier effects",
        has_backfill=True,
        dependencies=[]
    ),
    
    # =========================================================================
    # LEVEL 2: COMPOSITE DIALS (Calculated from Level 1)
    # =========================================================================
    "mci_daily": TableDef(
        name="mci_daily",
        level=2,
        python_module="mci",
        description="Market Cycle Indicator",
        has_backfill=True,
        dependencies=["vix_dial_daily", "credit_spread_daily", "breadth_daily", "pillar_index_daily"]
    ),
    "amri_daily": TableDef(
        name="amri_daily",
        level=2,
        python_module="analytical.amri",
        description="ARGUS Master Regime Index",
        has_backfill=True,
        dependencies=["vix_dial_daily", "credit_spread_daily", "breadth_daily"]
    ),
    "bubble_index_daily": TableDef(
        name="bubble_index_daily",
        level=2,
        python_module="bubble_index",
        description="Bubble formation detection",
        has_backfill=True,
        dependencies=["price_history", "correlation_daily"]
    ),
    "bubble_overlay_daily": TableDef(
        name="bubble_overlay_daily",
        level=2,
        python_module="bubble_overlay_dial",
        description="Bubble overlay analysis",
        has_backfill=True,
        dependencies=["bubble_index_daily"]
    ),
    "cluster_dial_daily": TableDef(
        name="cluster_dial_daily",
        level=2,
        python_module="cluster_dial",
        description="Market clustering analysis",
        has_backfill=True,
        dependencies=["correlation_daily"]
    ),
    "flow_divergence_daily": TableDef(
        name="flow_divergence_daily",
        level=2,
        python_module="flow_divergence",
        description="Flow divergence signals",
        has_backfill=True,
        dependencies=["etf_dial_daily", "price_history"]
    ),
    "hypergraph_signals": TableDef(
        name="hypergraph_signals",
        level=2,
        python_module="hypergraph_dial",
        description="Hypergraph network signals",
        has_backfill=True,
        dependencies=["correlation_daily"]
    ),
    "fingerprint_daily": TableDef(
        name="fingerprint_daily",
        level=2,
        python_module="fingerprint_dial",
        description="Historical pattern matching",
        has_backfill=True,
        dependencies=["price_history"]
    ),
    
    # =========================================================================
    # LEVEL 3: SIGNALS & REPORTS (Uses all levels)
    # =========================================================================
    "signals_dial_daily": TableDef(
        name="signals_dial_daily",
        level=3,
        python_module="signals_dial",
        description="Consolidated signals",
        has_backfill=True,
        dependencies=["mci_daily", "amri_daily"]
    ),
    "daily_signals": TableDef(
        name="daily_signals",
        level=3,
        python_module="signals_dial",
        description="Daily signal snapshots",
        has_backfill=True,
        dependencies=["signals_dial_daily"]
    ),
    "shadow_portfolio_daily": TableDef(
        name="shadow_portfolio_daily",
        level=3,
        python_module="shadow_portfolio_dial",
        description="Shadow portfolio tracking",
        has_backfill=True,
        dependencies=["price_history", "signals_dial_daily"]
    ),
    "shadow_portfolio": TableDef(
        name="shadow_portfolio",
        level=3,
        python_module="shadow_portfolio_dial",
        description="Shadow portfolio positions",
        has_backfill=False,
        dependencies=[]
    ),
    "dashboard_daily": TableDef(
        name="dashboard_daily",
        level=3,
        python_module="dashboard",
        description="Dashboard snapshots",
        has_backfill=True,
        dependencies=["mci_daily", "amri_daily"]
    ),
    "morning_brief_daily": TableDef(
        name="morning_brief_daily",
        level=3,
        python_module="morning_brief",
        description="Morning brief generation",
        has_backfill=True,
        dependencies=["mci_daily", "signals_dial_daily"]
    ),
    "trends_history": TableDef(
        name="trends_history",
        level=3,
        python_module="trends_history_dial",
        description="Trends tracking",
        has_backfill=True,
        dependencies=["price_history"]
    ),
}

# Tables to skip clearing by default (foundation data)
PROTECTED_TABLES = ["price_history", "articles", "processed_articles"]


# =============================================================================
# SUPABASE CONNECTION
# =============================================================================

def get_supabase_client() -> Client:
    """Create Supabase client from environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
    
    return create_client(url, key)


# =============================================================================
# TABLE OPERATIONS
# =============================================================================

def get_table_count(supabase: Client, table: str) -> int:
    """Get row count for a table."""
    try:
        response = supabase.table(table).select("*", count="exact").limit(0).execute()
        return response.count if response.count else 0
    except:
        return -1

def clear_table(supabase: Client, table: str, dry_run: bool = False) -> int:
    """Clear all rows from a table. Returns rows deleted."""
    try:
        count = get_table_count(supabase, table)
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would delete {count} rows from {table}")
            return count
        
        if count == 0:
            logger.info(f"  - {table} already empty")
            return 0
        
        # Delete in batches (Supabase has limits)
        # Use date column which all dial tables have, fallback to id
        deleted = 0
        while True:
            # Try date first (most dial tables use this)
            try:
                response = supabase.table(table).delete().neq("date", "1900-01-01").execute()
            except:
                # Fallback to id
                try:
                    response = supabase.table(table).delete().neq("id", -999999).execute()
                except:
                    # Last resort - try gte on date
                    response = supabase.table(table).delete().gte("date", "1900-01-01").execute()
            
            batch = len(response.data) if response.data else 0
            deleted += batch
            if batch == 0:
                break
        
        logger.info(f"  ✓ Deleted {deleted} rows from {table}")
        return deleted
        
    except Exception as e:
        logger.error(f"  ✗ Failed to clear {table}: {e}")
        return -1


def get_tables_by_level(level: int) -> List[str]:
    """Get all table names for a specific level."""
    return [name for name, tdef in ALL_TABLES.items() if tdef.level == level]


def get_tables_in_dependency_order(include_levels: List[int] = None) -> List[str]:
    """Get tables sorted by dependency level (highest first for clearing)."""
    tables = []
    levels = include_levels or [3, 2, 1, 0]
    
    for level in sorted(levels, reverse=True):
        level_tables = get_tables_by_level(level)
        tables.extend(sorted(level_tables))
    
    return tables


def get_tables_for_backfill_order(include_levels: List[int] = None) -> List[str]:
    """Get tables sorted for backfill (lowest level first)."""
    tables = []
    levels = include_levels or [0, 1, 2, 3]
    
    for level in sorted(levels):
        level_tables = get_tables_by_level(level)
        # Filter to only tables with backfill capability
        backfillable = [t for t in level_tables if ALL_TABLES[t].has_backfill]
        tables.extend(sorted(backfillable))
    
    return tables


# =============================================================================
# BACKFILL FUNCTIONS
# =============================================================================

def backfill_vix_dial(supabase: Client, days: int) -> int:
    """Backfill VIX dial from FRED."""
    from dials.vix_dial import VixDialCalculator
    
    calc = VixDialCalculator()
    df = calc.fetch_from_fred(days=days + 50)
    
    if df.empty:
        return 0
    
    metrics_df = calc.calculate_metrics(df)
    saved = 0
    
    for _, row in metrics_df.head(days).iterrows():
        try:
            data = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "vix": float(row["vix"]),
                "ma_20": float(row["ma_20"]) if pd.notna(row["ma_20"]) else None,
                "std_dev_20": float(row["std_dev_20"]) if pd.notna(row["std_dev_20"]) else None,
                "upper_bb": float(row["upper_bb"]) if pd.notna(row["upper_bb"]) else None,
                "lower_bb": float(row["lower_bb"]) if pd.notna(row["lower_bb"]) else None,
                "trend_20d": float(row["trend_20d"]) if pd.notna(row["trend_20d"]) else None,
                "level_regime": row["level_regime"],
                "trend_regime": row["trend_regime"],
                "bb_regime": row["bb_regime"],
                "combined_regime": row["combined_regime"],
                "is_pre_shock": row["combined_regime"] == "Pre-Shock",
            }
            supabase.table("vix_dial_daily").upsert(data, on_conflict="date").execute()
            saved += 1
        except Exception as e:
            logger.warning(f"VIX save failed for {row['date']}: {e}")
    
    return saved


def backfill_credit_spread(supabase: Client, days: int) -> int:
    """Backfill Credit Spread from FRED."""
    from dials.credit_spread_dial import CreditSpreadCalculator
    
    calc = CreditSpreadCalculator()
    df = calc.fetch_history(limit=days + 100)
    
    if df.empty:
        return 0
    
    momentum_df = calc.calculate_momentum(df)
    saved = 0
    
    for _, row in momentum_df.head(days).iterrows():
        try:
            data = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "hy_spread": float(row["hy_spread"]),
                "ig_spread": float(row["ig_spread"]),
                "hy_20d_change": float(row["hy_20d_change"]) if pd.notna(row["hy_20d_change"]) else None,
                "ig_20d_change": float(row["ig_20d_change"]) if pd.notna(row["ig_20d_change"]) else None,
                "hy_60d_change": float(row["hy_60d_change"]) if pd.notna(row["hy_60d_change"]) else None,
                "ig_60d_change": float(row["ig_60d_change"]) if pd.notna(row["ig_60d_change"]) else None,
                "hy_regime": row["hy_regime"],
                "ig_regime": row["ig_regime"],
                "combined_regime": row["combined_regime"],
            }
            supabase.table("credit_spread_daily").upsert(data, on_conflict="date").execute()
            saved += 1
        except Exception as e:
            logger.warning(f"Credit save failed: {e}")
    
    return saved


def backfill_pillar_index(supabase: Client, days: int) -> int:
    """Backfill Pillar Index from price_history."""
    from dials.pillar_index import PillarIndexCalculator
    
    calc = PillarIndexCalculator()
    results = calc.calculate(days=days)
    
    if not results:
        return 0
    
    return calc.save_to_supabase(results)


def backfill_breadth(supabase: Client, days: int) -> int:
    """Backfill Breadth from price_history."""
    from dials.breadth_dial import BreadthCalculator
    
    calc = BreadthCalculator()
    results = calc.calculate_history(days=days)
    
    if not results:
        return 0
    
    saved = 0
    for data in results:
        try:
            row = {
                "date": data.date,
                "daily_breadth": data.daily_breadth,
                "breadth_20d": data.breadth_20d,
                "breadth_50d": data.breadth_50d,
                "breadth_momentum": data.breadth_momentum,
                "advancers": data.advancers,
                "decliners": data.decliners,
                "above_20d": data.above_20d,
                "above_50d": data.above_50d,
                "valid_tickers": data.valid_tickers,
                "regime": data.regime,
                "pillar_breadth": data.pillar_breadth
            }
            supabase.table("breadth_daily").upsert(row, on_conflict="date").execute()
            saved += 1
        except Exception as e:
            logger.warning(f"Breadth save failed: {e}")
    
    return saved


def backfill_mci(supabase: Client, days: int) -> int:
    """Backfill MCI from component tables."""
    from dials.mci import MCI_CONFIG
    
    # Load component data
    vix_data = supabase.table("vix_dial_daily").select("date, vix").order("date", desc=True).limit(days + 20).execute()
    credit_data = supabase.table("credit_spread_daily").select("date, hy_spread").order("date", desc=True).limit(days + 20).execute()
    breadth_data = supabase.table("breadth_daily").select("date, breadth_20d").order("date", desc=True).limit(days + 10).execute()
    pillar_data = supabase.table("pillar_index_daily").select("date, infra_5d, enterprise_5d, macro_5d, financial_5d, productivity_5d, demand_5d").order("date", desc=True).limit(days + 10).execute()
    
    if not all([vix_data.data, credit_data.data, breadth_data.data, pillar_data.data]):
        logger.error("Missing component data for MCI")
        return 0
    
    # Build lookups
    vix_lookup = {row["date"]: row["vix"] for row in vix_data.data}
    credit_lookup = {row["date"]: row["hy_spread"] for row in credit_data.data}
    breadth_lookup = {row["date"]: row["breadth_20d"] for row in breadth_data.data}
    pillar_lookup = {row["date"]: row for row in pillar_data.data}
    
    vix_dates = sorted(vix_lookup.keys(), reverse=True)
    config = MCI_CONFIG
    lookback = config["VIX_LOOKBACK"]
    
    saved = 0
    
    for i, date in enumerate(vix_dates[:days]):
        if i + lookback >= len(vix_dates):
            break
        
        prev_date = vix_dates[i + lookback - 1]
        
        current_vix = vix_lookup.get(date)
        previous_vix = vix_lookup.get(prev_date)
        current_credit = credit_lookup.get(date)
        previous_credit = credit_lookup.get(prev_date)
        breadth_20d = breadth_lookup.get(date, 0.5) or 0.5
        pillar_row = pillar_lookup.get(date, {})
        
        if not all([current_vix, previous_vix, current_credit, previous_credit]):
            continue
        
        # Calculate components
        breadth_norm = max(-1, min(1, (breadth_20d - 0.5) / 0.5))
        breadth_component = round(breadth_norm * config["WEIGHT_BREADTH"], 1)
        
        vix_change = current_vix - previous_vix
        vix_norm = max(-1, min(1, -vix_change / config["VIX_THRESHOLD"]))
        vix_component = round(vix_norm * config["WEIGHT_VIX"], 1)
        
        spread_change_bps = (current_credit - previous_credit) * 100
        credit_norm = max(-1, min(1, -spread_change_bps / config["CREDIT_THRESHOLD"]))
        credit_component = round(credit_norm * config["WEIGHT_CREDIT"], 1)
        
        pillar_5d = [pillar_row.get(f"{p}_5d", 0) or 0 for p in ["infra", "enterprise", "macro", "financial", "productivity", "demand"]]
        avg_5d = sum(pillar_5d) / len(pillar_5d)
        pillar_norm = max(-1, min(1, avg_5d * 1000 / config["PILLAR_THRESHOLD"]))
        pillar_component = round(pillar_norm * config["WEIGHT_PILLAR"], 1)
        
        mci_score = breadth_component + vix_component + credit_component + pillar_component
        
        if mci_score >= 30:
            regime = "Constructive"
        elif mci_score >= 0:
            regime = "Neutral"
        elif mci_score >= -30:
            regime = "Cautious"
        else:
            regime = "Defensive"
        
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
                "pillar_raw": avg_5d
            }
            supabase.table("mci_daily").upsert(row, on_conflict="date").execute()
            saved += 1
        except Exception as e:
            logger.warning(f"MCI save failed: {e}")
    
    return saved


def backfill_correlation(supabase: Client, days: int) -> int:
    """Backfill Correlation dial."""
    try:
        from dials.correlation_dial import CorrelationCalculator
        calc = CorrelationCalculator()
        data = calc.calculate()
        if data and calc.save_to_supabase(data):
            return 1
    except Exception as e:
        logger.warning(f"Correlation backfill failed: {e}")
    return 0


# Map table names to backfill functions
BACKFILL_FUNCTIONS = {
    "vix_dial_daily": backfill_vix_dial,
    "credit_spread_daily": backfill_credit_spread,
    "pillar_index_daily": backfill_pillar_index,
    "breadth_daily": backfill_breadth,
    "mci_daily": backfill_mci,
    "correlation_daily": backfill_correlation,
}


def run_backfill(supabase: Client, table: str, days: int, dry_run: bool = False) -> int:
    """Run backfill for a single table."""
    tdef = ALL_TABLES.get(table)
    
    if not tdef:
        logger.warning(f"Unknown table: {table}")
        return -1
    
    if not tdef.has_backfill:
        logger.info(f"  - {table} (no backfill available)")
        return 0
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would backfill {table}")
        return days
    
    backfill_func = BACKFILL_FUNCTIONS.get(table)
    
    if backfill_func:
        logger.info(f"  Backfilling {table}...")
        count = backfill_func(supabase, days)
        logger.info(f"  ✓ {table}: {count} rows")
        return count
    else:
        logger.info(f"  - {table} (backfill not implemented)")
        return 0


# =============================================================================
# MAIN OPERATIONS
# =============================================================================

def clear_tables(supabase: Client, tables: List[str], dry_run: bool = False, 
                 skip_protected: bool = True) -> Dict[str, int]:
    """Clear specified tables."""
    results = {}
    
    logger.info("=" * 60)
    logger.info("CLEARING TABLES")
    logger.info("=" * 60)
    
    for table in tables:
        if skip_protected and table in PROTECTED_TABLES:
            logger.info(f"  [PROTECTED] Skipping {table}")
            continue
        
        results[table] = clear_table(supabase, table, dry_run)
    
    return results


def backfill_tables(supabase: Client, tables: List[str], days: int, 
                    dry_run: bool = False) -> Dict[str, int]:
    """Backfill specified tables in dependency order."""
    results = {}
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKFILLING TABLES")
    logger.info("=" * 60)
    
    # Sort by level (lowest first)
    sorted_tables = sorted(tables, key=lambda t: ALL_TABLES.get(t, TableDef("", 99, "", "")).level)
    
    for table in sorted_tables:
        results[table] = run_backfill(supabase, table, days, dry_run)
    
    return results


def show_table_status(supabase: Client):
    """Show status of all tables."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TABLE STATUS")
    logger.info("=" * 60)
    
    for level in [0, 1, 2, 3]:
        tables = get_tables_by_level(level)
        logger.info(f"\n--- Level {level} ---")
        
        for table in sorted(tables):
            tdef = ALL_TABLES[table]
            count = get_table_count(supabase, table)
            status = "✓" if count > 0 else "○"
            bf = "📥" if tdef.has_backfill else "  "
            logger.info(f"  {status} {bf} {table:30} {count:>6} rows  ({tdef.description})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ARGUS-1 Complete Dial Reset & Backfill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Levels:
  0 = Foundation (price_history, etc.) - usually don't clear
  1 = Base Dials (vix, credit, breadth, pillar, etc.)
  2 = Composite (mci, amri, bubble_index, etc.)
  3 = Signals & Reports

Examples:
  python reset_all_dials.py --status              # Show all tables
  python reset_all_dials.py --dry-run             # Preview full reset
  python reset_all_dials.py --level 1 2           # Reset levels 1 and 2
  python reset_all_dials.py --tables mci vix      # Reset specific tables
  python reset_all_dials.py --skip-clear          # Backfill only
        """
    )
    
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--skip-clear", action="store_true", help="Skip clearing tables")
    parser.add_argument("--days", type=int, default=252, help="Days to backfill (default: 252)")
    parser.add_argument("--level", type=int, nargs="+", help="Only process specific levels (0-3)")
    parser.add_argument("--tables", nargs="+", help="Only process specific tables")
    parser.add_argument("--status", action="store_true", help="Show table status and exit")
    parser.add_argument("--include-protected", action="store_true", help="Include protected tables (price_history, etc.)")
    
    args = parser.parse_args()
    
    # Banner
    print()
    print("=" * 60)
    print("ARGUS-1 COMPLETE DIAL RESET & BACKFILL")
    print("=" * 60)
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tables:     {len(ALL_TABLES)} defined")
    print(f"Days:       {args.days}")
    print("=" * 60)
    print()
    
    # Connect
    try:
        supabase = get_supabase_client()
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        sys.exit(1)
    
    # Status only
    if args.status:
        show_table_status(supabase)
        return
    
    # Determine which tables to process
    if args.tables:
        tables = args.tables
    elif args.level:
        tables = []
        for level in args.level:
            tables.extend(get_tables_by_level(level))
    else:
        # All tables except Level 0 by default
        tables = get_tables_by_level(1) + get_tables_by_level(2) + get_tables_by_level(3)
    
    # Confirmation
    if not args.dry_run and not args.skip_clear:
        print(f"⚠️  Will clear {len(tables)} tables:")
        for t in tables[:10]:
            print(f"    - {t}")
        if len(tables) > 10:
            print(f"    ... and {len(tables) - 10} more")
        print()
        confirm = input("Type 'yes' to continue: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            sys.exit(1)
        print()
    
    # Clear tables
    if not args.skip_clear:
        clear_results = clear_tables(
            supabase, 
            get_tables_in_dependency_order(args.level),
            args.dry_run,
            not args.include_protected
        )
    else:
        logger.info("Skipping table clear")
        clear_results = {}
    
    # Backfill
    backfill_results = backfill_tables(
        supabase,
        tables,
        args.days,
        args.dry_run
    )
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if clear_results:
        cleared = sum(v for v in clear_results.values() if v > 0)
        print(f"Rows cleared:     {cleared}")
    
    saved = sum(v for v in backfill_results.values() if v > 0)
    print(f"Rows backfilled:  {saved}")
    
    failed = [k for k, v in backfill_results.items() if v < 0]
    if failed:
        print(f"Failed:           {', '.join(failed)}")
    
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")


if __name__ == "__main__":
    main()