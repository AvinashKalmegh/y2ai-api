"""
DIALS RUNNER
============
Orchestrates all Y2AI dial modules in correct sequence.

Run order (dependencies matter):
1. Foundation: PillarIndex, VixDial, CreditSpreadDial
2. Core: BreadthDial, MCI, MacroDial, SignalsDial
3. Additional: ClusterDial, LiquidityDial, LaborDial, ETFDial, SentimentDial
4. Flow: StockFlowDial, FlowDivergence
5. Multipliers: MacroMultipliers
6. Aggregation: RegimeArbiter, PortfolioTracker
7. Output: Dashboard, MorningBrief

Usage:
    python -m y2ai.dials_runner --all
    python -m y2ai.dials_runner --dials
    python -m y2ai.dials_runner --dashboard
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DIAL RUNNER
# =============================================================================

class DialsRunner:
    """
    Run all dial modules in sequence.
    """
    
    def __init__(self, save_to_supabase: bool = True):
        self.save = save_to_supabase
        self.results = {}
        self.errors = []
    
    def run_module(self, name: str, calculator_class, method: str = "calculate") -> bool:
        """Run a single module and optionally save."""
        try:
            logger.info(f"Running {name}...")
            calc = calculator_class()
            data = getattr(calc, method)()
            
            if self.save and hasattr(calc, 'save_to_supabase'):
                calc.save_to_supabase(data)
                logger.info(f"  ✅ {name} - saved")
            else:
                logger.info(f"  ✅ {name} - calculated")
            
            self.results[name] = data
            return True
            
        except Exception as e:
            logger.error(f"  ❌ {name} - {e}")
            self.errors.append(f"{name}: {e}")
            return False
    
    # =========================================================================
    # PHASE 1: FOUNDATION
    # =========================================================================
    
    def run_foundation(self) -> int:
        """Run foundation dials (price-based, no dependencies)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: FOUNDATION DIALS")
        logger.info("="*60)
        
        success = 0
        
        from dials.pillar_index import PillarIndexCalculator
        if self.run_module("PillarIndex", PillarIndexCalculator):
            success += 1
        
        from dials.vix_dial import VixDialCalculator
        if self.run_module("VixDial", VixDialCalculator):
            success += 1
        
        from dials.credit_spread_dial import CreditSpreadCalculator
        if self.run_module("CreditSpreadDial", CreditSpreadCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 2: CORE DIALS
    # =========================================================================
    
    def run_core(self) -> int:
        """Run core dials (depend on foundation)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: CORE DIALS")
        logger.info("="*60)
        
        success = 0
        
        from dials.breadth_dial import BreadthCalculator
        if self.run_module("BreadthDial", BreadthCalculator):
            success += 1
        
        from dials.mci import MCICalculator
        if self.run_module("MCI", MCICalculator):
            success += 1
        
        from dials.macro_dial import MacroDialCalculator
        if self.run_module("MacroDial", MacroDialCalculator):
            success += 1
        
        from dials.signals_dial import SignalsDialCalculator
        if self.run_module("SignalsDial", SignalsDialCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 3: ADDITIONAL DIALS
    # =========================================================================
    
    def run_additional(self) -> int:
        """Run additional dials (independent of core)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: ADDITIONAL DIALS")
        logger.info("="*60)
        
        success = 0
        
        from dials.cluster_dial import ClusterDialCalculator
        if self.run_module("ClusterDial", ClusterDialCalculator):
            success += 1
        
        from dials.liquidity_dial import LiquidityDialCalculator
        if self.run_module("LiquidityDial", LiquidityDialCalculator):
            success += 1
        
        from dials.labor_dial import LaborDialCalculator
        if self.run_module("LaborDial", LaborDialCalculator):
            success += 1
        
        from dials.etf_dial import ETFDialCalculator
        if self.run_module("ETFDial", ETFDialCalculator):
            success += 1
        
        from dials.sentiment_dial import SentimentDialCalculator
        if self.run_module("SentimentDial", SentimentDialCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 4: FLOW DIALS
    # =========================================================================
    
    def run_flow(self) -> int:
        """Run flow dials (depend on price data)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: FLOW DIALS")
        logger.info("="*60)
        
        success = 0
        
        from dials.stock_flow_dial import StockFlowCalculator
        if self.run_module("StockFlowDial", StockFlowCalculator):
            success += 1
        
        from dials.flow_divergence import FlowDivergenceCalculator
        if self.run_module("FlowDivergence", FlowDivergenceCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 5: MULTIPLIERS
    # =========================================================================
    
    def run_multipliers(self) -> int:
        """Run macro multipliers (depend on multiple dials)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: MULTIPLIERS")
        logger.info("="*60)
        
        success = 0
        
        from dials.macro_multipliers import MacroMultiplierCalculator
        if self.run_module("MacroMultipliers", MacroMultiplierCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 6: AGGREGATION
    # =========================================================================
    
    def run_aggregation(self) -> int:
        """Run aggregation modules (depend on all dials)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 6: AGGREGATION")
        logger.info("="*60)
        
        success = 0
        
        # RegimeArbiter uses arbitrate() not calculate()
        try:
            logger.info("Running RegimeArbiter...")
            from portfolio.regime_arbiter import RegimeArbiter
            calc = RegimeArbiter()
            data = calc.arbitrate()
            if self.save and hasattr(calc, 'save_to_supabase'):
                calc.save_to_supabase(data)
                logger.info("  ✅ RegimeArbiter - saved")
            else:
                logger.info("  ✅ RegimeArbiter - calculated")
            self.results["RegimeArbiter"] = data
            success += 1
        except Exception as e:
            logger.error(f"  ❌ RegimeArbiter - {e}")
            self.errors.append(f"RegimeArbiter: {e}")
        
        # PortfolioTracker uses calculate_daily() not calculate()
        try:
            logger.info("Running PortfolioTracker...")
            from portfolio.portfolio_tracker import PortfolioTracker
            calc = PortfolioTracker()
            data = calc.calculate_daily()
            if self.save and hasattr(calc, 'save_to_supabase'):
                calc.save_to_supabase(data)
                logger.info("  ✅ PortfolioTracker - saved")
            else:
                logger.info("  ✅ PortfolioTracker - calculated")
            self.results["PortfolioTracker"] = data
            success += 1
        except Exception as e:
            logger.error(f"  ❌ PortfolioTracker - {e}")
            self.errors.append(f"PortfolioTracker: {e}")
        
        return success
    
    # =========================================================================
    # PHASE 7: OUTPUT
    # =========================================================================
    
    def run_output(self) -> int:
        """Run output generators."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 7: OUTPUT")
        logger.info("="*60)
        
        success = 0
        
        from dials.dashboard import DashboardGenerator
        if self.run_module("Dashboard", DashboardGenerator, method="generate"):
            success += 1
        
        from dials.morning_brief import MorningBriefGenerator
        if self.run_module("MorningBrief", MorningBriefGenerator, method="generate"):
            success += 1
        
        return success
    
    # =========================================================================
    # RUN ALL
    # =========================================================================
    
    def run_all_dials(self) -> Dict:
        """Run all dial modules in sequence."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*60)
        logger.info("Y2AI DIALS RUNNER")
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        total_success = 0
        total_modules = 0
        
        # Phase 1: Foundation
        count = self.run_foundation()
        total_success += count
        total_modules += 3
        
        # Phase 2: Core
        count = self.run_core()
        total_success += count
        total_modules += 4
        
        # Phase 3: Additional
        count = self.run_additional()
        total_success += count
        total_modules += 5
        
        # Phase 4: Flow
        count = self.run_flow()
        total_success += count
        total_modules += 2
        
        # Phase 5: Multipliers
        count = self.run_multipliers()
        total_success += count
        total_modules += 1
        
        # Phase 6: Aggregation
        count = self.run_aggregation()
        total_success += count
        total_modules += 2
        
        # Phase 7: Output
        count = self.run_output()
        total_success += count
        total_modules += 2
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Completed: {total_success}/{total_modules} modules")
        logger.info(f"Duration: {duration:.1f}s")
        
        if self.errors:
            logger.info(f"\nErrors ({len(self.errors)}):")
            for err in self.errors:
                logger.info(f"  - {err}")
        
        return {
            "success": total_success,
            "total": total_modules,
            "errors": self.errors,
            "duration": duration
        }
    
    def run_dials_only(self) -> Dict:
        """Run only dial modules (no output generators)."""
        start_time = datetime.now()
        
        total_success = 0
        total_modules = 0
        
        # Phases 1-5
        total_success += self.run_foundation()
        total_modules += 3
        
        total_success += self.run_core()
        total_modules += 4
        
        total_success += self.run_additional()
        total_modules += 5
        
        total_success += self.run_flow()
        total_modules += 2
        
        total_success += self.run_multipliers()
        total_modules += 1
        
        # Phase 6: Aggregation
        total_success += self.run_aggregation()
        total_modules += 2
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": total_success,
            "total": total_modules,
            "errors": self.errors,
            "duration": duration
        }
    
    def run_dashboard_only(self) -> Dict:
        """Run only dashboard and morning brief."""
        total_success = self.run_output()
        
        return {
            "success": total_success,
            "total": 2,
            "errors": self.errors
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_all_dials(save: bool = True) -> Dict:
    """Run all dials (for orchestrator integration)."""
    runner = DialsRunner(save_to_supabase=save)
    return runner.run_all_dials()


def run_dials_only(save: bool = True) -> Dict:
    """Run dials without output generators."""
    runner = DialsRunner(save_to_supabase=save)
    return runner.run_dials_only()


def run_dashboard(save: bool = True) -> Dict:
    """Run dashboard and morning brief only."""
    runner = DialsRunner(save_to_supabase=save)
    return runner.run_dashboard_only()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Y2AI Dials Runner")
    parser.add_argument('--all', action='store_true', help='Run all dials + outputs')
    parser.add_argument('--dials', action='store_true', help='Run dials only (no outputs)')
    parser.add_argument('--dashboard', action='store_true', help='Run dashboard + brief only')
    parser.add_argument('--no-save', action='store_true', help='Calculate only, do not save')
    parser.add_argument('--phase', type=int, choices=[1,2,3,4,5,6,7], help='Run specific phase only')
    
    args = parser.parse_args()
    
    save = not args.no_save
    runner = DialsRunner(save_to_supabase=save)
    
    if args.phase:
        phases = {
            1: runner.run_foundation,
            2: runner.run_core,
            3: runner.run_additional,
            4: runner.run_flow,
            5: runner.run_multipliers,
            6: runner.run_aggregation,
            7: runner.run_output
        }
        phases[args.phase]()
    elif args.all:
        result = runner.run_all_dials()
        print(f"\n✅ Complete: {result['success']}/{result['total']} in {result['duration']:.1f}s")
    elif args.dials:
        result = runner.run_dials_only()
        print(f"\n✅ Complete: {result['success']}/{result['total']} in {result['duration']:.1f}s")
    elif args.dashboard:
        result = runner.run_dashboard_only()
        print(f"\n✅ Complete: {result['success']}/{result['total']}")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python dials_runner.py --all       # Run everything")
        print("  python dials_runner.py --dials     # Run dials only")
        print("  python dials_runner.py --dashboard # Run outputs only")
        print("  python dials_runner.py --phase 1   # Run foundation only")


if __name__ == "__main__":
    main()