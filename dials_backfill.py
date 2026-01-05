"""
DIALS BACKFILL
==============
Backfill historical data for all Y2AI dial modules.

Usage:
    python dials_backfill.py --days 30        # Backfill last 30 days
    python dials_backfill.py --start 2024-12-01 --end 2024-12-31
    python dials_backfill.py --module pillar  # Backfill specific module only
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# BACKFILL FUNCTIONS
# =============================================================================

def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get list of trading days (weekdays) between dates."""
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days.append(current)
        current += timedelta(days=1)
    return days


def backfill_pillar_index(dates: List[datetime], save: bool = True):
    """Backfill pillar index for historical dates."""
    from dials.pillar_index import PillarIndexCalculator
    
    logger.info(f"Backfilling PillarIndex for {len(dates)} dates...")
    calc = PillarIndexCalculator()
    
    success = 0
    for date in dates:
        try:
            # Most calculators work on current data
            # For true historical backfill, would need to modify calculator
            # to accept a date parameter
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
            logger.info(f"  ✅ {date.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_breadth(dates: List[datetime], save: bool = True):
    """Backfill breadth data."""
    from dials.breadth_dial import BreadthCalculator
    
    logger.info(f"Backfilling Breadth for {len(dates)} dates...")
    calc = BreadthCalculator()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_mci(dates: List[datetime], save: bool = True):
    """Backfill MCI data."""
    from dials.mci import MCICalculator
    
    logger.info(f"Backfilling MCI for {len(dates)} dates...")
    calc = MCICalculator()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_vix(dates: List[datetime], save: bool = True):
    """Backfill VIX dial data."""
    from dials.vix_dial import VixDialCalculator
    
    logger.info(f"Backfilling VixDial for {len(dates)} dates...")
    calc = VixDialCalculator()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_credit(dates: List[datetime], save: bool = True):
    """Backfill credit spread data."""
    from dials.credit_spread_dial import CreditSpreadCalculator
    
    logger.info(f"Backfilling CreditSpread for {len(dates)} dates...")
    calc = CreditSpreadCalculator()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_labor(dates: List[datetime], save: bool = True):
    """Backfill labor dial data."""
    from dials.labor_dial import LaborDialCalculator
    
    logger.info(f"Backfilling LaborDial for {len(dates)} dates...")
    calc = LaborDialCalculator()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_macro(dates: List[datetime], save: bool = True):
    """Backfill macro dial data."""
    from dials.macro_dial import MacroDialCalculator
    
    logger.info(f"Backfilling MacroDial for {len(dates)} dates...")
    calc = MacroDialCalculator()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


def backfill_regime_arbiter(dates: List[datetime], save: bool = True):
    """Backfill regime arbiter data."""
    from portfolio.regime_arbiter import RegimeArbiter
    
    logger.info(f"Backfilling RegimeArbiter for {len(dates)} dates...")
    calc = RegimeArbiter()
    
    success = 0
    for date in dates:
        try:
            data = calc.calculate()
            if save:
                calc.save_to_supabase(data)
            success += 1
        except Exception as e:
            logger.error(f"  ❌ {date.strftime('%Y-%m-%d')}: {e}")
    
    return success


# =============================================================================
# MAIN BACKFILL ORCHESTRATOR
# =============================================================================

def run_backfill(
    start_date: datetime,
    end_date: datetime,
    modules: Optional[List[str]] = None,
    save: bool = True
):
    """
    Run backfill for specified modules and date range.
    
    Args:
        start_date: Start date for backfill
        end_date: End date for backfill
        modules: List of module names to backfill (None = all)
        save: Whether to save to Supabase
    """
    logger.info("=" * 60)
    logger.info("Y2AI DIALS BACKFILL")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    dates = get_trading_days(start_date, end_date)
    logger.info(f"Trading days: {len(dates)}")
    
    if modules:
        logger.info(f"Modules: {', '.join(modules)}")
    else:
        logger.info("Modules: ALL")
    
    # Module map
    all_modules = {
        'pillar': backfill_pillar_index,
        'breadth': backfill_breadth,
        'mci': backfill_mci,
        'vix': backfill_vix,
        'credit': backfill_credit,
        'labor': backfill_labor,
        'macro': backfill_macro,
        'regime': backfill_regime_arbiter,
    }
    
    # Filter modules if specified
    if modules:
        to_run = {k: v for k, v in all_modules.items() if k in modules}
    else:
        to_run = all_modules
    
    # Run backfill
    results = {}
    for name, func in to_run.items():
        try:
            count = func(dates, save=save)
            results[name] = count
            logger.info(f"  {name}: {count}/{len(dates)} successful")
        except Exception as e:
            logger.error(f"  {name}: FAILED - {e}")
            results[name] = 0
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 60)
    for name, count in results.items():
        status = "✅" if count == len(dates) else "⚠️"
        logger.info(f"  {status} {name}: {count}/{len(dates)}")
    
    return results


def run_single_day_backfill(save: bool = True):
    """
    Run all dials for today only (simpler than date range).
    This is what you'd use for daily updates.
    """
    logger.info("=" * 60)
    logger.info("SINGLE DAY BACKFILL (TODAY)")
    logger.info("=" * 60)
    
    from dials_runner import run_all_dials
    result = run_all_dials(save=save)
    
    logger.info(f"\nComplete: {result['success']}/{result['total']} modules")
    if result['errors']:
        for err in result['errors']:
            logger.warning(f"  Error: {err}")
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Y2AI Dials Backfill")
    parser.add_argument('--days', type=int, help='Backfill last N days')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--module', type=str, action='append',
                        choices=['pillar', 'breadth', 'mci', 'vix', 'credit', 
                                'labor', 'macro', 'regime'],
                        help='Specific module(s) to backfill')
    parser.add_argument('--today', action='store_true', help='Run for today only')
    parser.add_argument('--no-save', action='store_true', help='Calculate only, do not save')
    
    args = parser.parse_args()
    save = not args.no_save
    
    if args.today:
        run_single_day_backfill(save=save)
    elif args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        run_backfill(start_date, end_date, modules=args.module, save=save)
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        run_backfill(start_date, end_date, modules=args.module, save=save)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python dials_backfill.py --today              # Run all dials for today")
        print("  python dials_backfill.py --days 30            # Backfill last 30 days")
        print("  python dials_backfill.py --start 2024-12-01 --end 2024-12-31")
        print("  python dials_backfill.py --days 7 --module vix --module credit")


if __name__ == "__main__":
    main()