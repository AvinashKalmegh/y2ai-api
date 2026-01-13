"""
DIALS RUNNER - COMPLETE
=======================
Orchestrates all Y2AI dial modules in correct sequence.

Run order (dependencies matter):
Phase 1: Foundation - PillarIndex, VixDial, CreditSpreadDial
Phase 2: Core - BreadthDial, MCI, MacroDial, SignalsDial, CorrelationDial
Phase 3: Additional - ClusterDial, LiquidityDial, LaborDial, ETFDial, SentimentDial
Phase 4: Flow - StockFlowDial, FlowDivergence
Phase 5: Multipliers - MacroMultipliers, FinancialStressDial
Phase 6: NST & Bubble - NSTDial, BubbleIndex, BubbleOverlayDial  <-- ADDED BubbleIndex
Phase 7: Analysis - HypergraphDial, FingerprintDial, TrendsHistoryDial
Phase 8: Portfolio - ShadowPortfolioDial, RegimeArbiter, PortfolioTracker
Phase 9: Output - Dashboard, MorningBrief

CHANGE LOG:
- 2026-01-13: Added BubbleIndex to Phase 6 to fix data corruption issue.
  The old orchestrator.py was writing garbage data (cape=181.77, credit=10000).
  Now BubbleIndex uses the correct 6-component formula.

Usage:
    python -m y2ai.dials_runner --all
    python -m y2ai.dials_runner --dials
    python -m y2ai.dials_runner --dashboard
    python -m y2ai.dials_runner --phase 6
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
    Total: 26 dial modules + 2 portfolio modules = 28 modules
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
                logger.info(f"  ✅ {name} - calculated (no save method)")
            
            self.results[name] = data
            return True
            
        except Exception as e:
            logger.error(f"  ❌ {name} - {e}")
            self.errors.append(f"{name}: {e}")
            return False
    
    # =========================================================================
    # PHASE 1: FOUNDATION (3 modules)
    # =========================================================================
    
    def run_phase1_foundation(self) -> int:
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
    # PHASE 2: CORE DIALS (5 modules)
    # =========================================================================
    
    def run_phase2_core(self) -> int:
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
        
        from dials.correlation_dial import CorrelationDialCalculator
        if self.run_module("CorrelationDial", CorrelationDialCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 3: ADDITIONAL DIALS (5 modules)
    # =========================================================================
    
    def run_phase3_additional(self) -> int:
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
    # PHASE 4: FLOW DIALS (2 modules)
    # =========================================================================
    
    def run_phase4_flow(self) -> int:
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
    # PHASE 5: MULTIPLIERS & STRESS (2 modules)
    # =========================================================================
    
    def run_phase5_multipliers(self) -> int:
        """Run macro multipliers and stress indicators."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: MULTIPLIERS & STRESS")
        logger.info("="*60)
        
        success = 0
        
        from dials.macro_multipliers import MacroMultiplierCalculator
        if self.run_module("MacroMultipliers", MacroMultiplierCalculator):
            success += 1
        
        from dials.financial_stress_dial import FinancialStressCalculator
        if self.run_module("FinancialStress", FinancialStressCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 6: NST & BUBBLE (3 modules) - UPDATED
    # =========================================================================
    
    def run_phase6_nst_bubble(self) -> int:
        """
        Run NST composite, Bubble Index, and bubble overlay detection.
        
        NOTE: BubbleIndex added 2026-01-13 to fix data corruption.
        Uses 6-component formula (valuation, growth, momentum, volatility,
        concentration, vix complacency) instead of old 3-indicator version.
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 6: NST & BUBBLE")
        logger.info("="*60)
        
        success = 0
        
        # NST Dial (Narrative Sentiment Tracker)
        from dials.nst_dial import NSTDialCalculator
        if self.run_module("NSTDial", NSTDialCalculator):
            success += 1
        
        # Bubble Index (6-component formula)
        # This is the CORRECT calculator - not the old 3-indicator version
        # that was writing garbage data (cape=181.77, credit=10000)
        try:
            logger.info("Running BubbleIndex...")
            from bubble_index import BubbleIndexCalculator
            calc = BubbleIndexCalculator()
            data = calc.calculate()
            
            if self.save:
                calc.save_to_supabase(data)
                logger.info(f"  ✅ BubbleIndex - saved (score={data.bubble_index:.1f}, regime={data.regime})")
            else:
                logger.info(f"  ✅ BubbleIndex - calculated (score={data.bubble_index:.1f})")
            
            self.results["BubbleIndex"] = data
            success += 1
            
        except Exception as e:
            logger.error(f"  ❌ BubbleIndex - {e}")
            self.errors.append(f"BubbleIndex: {e}")
        
        # Bubble Overlay (LPPLS, PSY, LZC detection)
        from dials.bubble_overlay_dial import BubbleOverlayCalculator
        if self.run_module("BubbleOverlay", BubbleOverlayCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 7: ANALYSIS (3 modules)
    # =========================================================================
    
    def run_phase7_analysis(self) -> int:
        """Run analysis modules (depend on multiple inputs)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 7: ANALYSIS")
        logger.info("="*60)
        
        success = 0
        
        from dials.hypergraph_dial import HypergraphDialCalculator
        if self.run_module("HypergraphDial", HypergraphDialCalculator):
            success += 1
        
        from dials.fingerprint_dial import FingerprintLibraryCalculator
        if self.run_module("FingerprintDial", FingerprintLibraryCalculator):
            success += 1
        
        from dials.trends_history_dial import TrendsHistoryCalculator
        if self.run_module("TrendsHistory", TrendsHistoryCalculator):
            success += 1
        
        return success
    
    # =========================================================================
    # PHASE 8: PORTFOLIO (3 modules)
    # =========================================================================
    
    def run_phase8_portfolio(self) -> int:
        """Run portfolio modules (depend on all dials)."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 8: PORTFOLIO")
        logger.info("="*60)
        
        success = 0
        
        # ShadowPortfolio dial
        from dials.shadow_portfolio_dial import ShadowPortfolioCalculator
        if self.run_module("ShadowPortfolio", ShadowPortfolioCalculator):
            success += 1
        
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
    # PHASE 9: OUTPUT (2 modules)
    # =========================================================================
    
    def run_phase9_output(self) -> int:
        """Run output generators."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 9: OUTPUT")
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
        """Run all 28 modules in sequence across 9 phases."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*60)
        logger.info("Y2AI DIALS RUNNER - COMPLETE")
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Total modules: 28 (26 dials + 2 portfolio)")
        logger.info("="*60)
        
        total_success = 0
        total_modules = 0
        
        # Module counts per phase (updated: phase 6 now has 3)
        phase_counts = [3, 5, 5, 2, 2, 3, 3, 3, 2]  # = 28
        
        # Phase 1: Foundation
        count = self.run_phase1_foundation()
        total_success += count
        total_modules += phase_counts[0]
        
        # Phase 2: Core
        count = self.run_phase2_core()
        total_success += count
        total_modules += phase_counts[1]
        
        # Phase 3: Additional
        count = self.run_phase3_additional()
        total_success += count
        total_modules += phase_counts[2]
        
        # Phase 4: Flow
        count = self.run_phase4_flow()
        total_success += count
        total_modules += phase_counts[3]
        
        # Phase 5: Multipliers & Stress
        count = self.run_phase5_multipliers()
        total_success += count
        total_modules += phase_counts[4]
        
        # Phase 6: NST & Bubble (now includes BubbleIndex)
        count = self.run_phase6_nst_bubble()
        total_success += count
        total_modules += phase_counts[5]
        
        # Phase 7: Analysis
        count = self.run_phase7_analysis()
        total_success += count
        total_modules += phase_counts[6]
        
        # Phase 8: Portfolio
        count = self.run_phase8_portfolio()
        total_success += count
        total_modules += phase_counts[7]
        
        # Phase 9: Output
        count = self.run_phase9_output()
        total_success += count
        total_modules += phase_counts[8]
        
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
        """Run dial modules only (phases 1-7, no portfolio or output)."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*60)
        logger.info("Y2AI DIALS RUNNER - DIALS ONLY")
        logger.info("="*60)
        
        total_success = 0
        total_modules = 0
        
        # Phases 1-7 (23 modules now, was 22)
        total_success += self.run_phase1_foundation()
        total_modules += 3
        
        total_success += self.run_phase2_core()
        total_modules += 5
        
        total_success += self.run_phase3_additional()
        total_modules += 5
        
        total_success += self.run_phase4_flow()
        total_modules += 2
        
        total_success += self.run_phase5_multipliers()
        total_modules += 2
        
        total_success += self.run_phase6_nst_bubble()
        total_modules += 3  # Was 2, now 3 with BubbleIndex
        
        total_success += self.run_phase7_analysis()
        total_modules += 3
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Dials complete: {total_success}/{total_modules} in {duration:.1f}s")
        
        return {
            "success": total_success,
            "total": total_modules,
            "errors": self.errors,
            "duration": duration
        }
    
    def run_portfolio_only(self) -> Dict:
        """Run portfolio modules only (phase 8)."""
        total_success = self.run_phase8_portfolio()
        
        return {
            "success": total_success,
            "total": 3,
            "errors": self.errors
        }
    
    def run_dashboard_only(self) -> Dict:
        """Run dashboard and morning brief only (phase 9)."""
        total_success = self.run_phase9_output()
        
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
    
    parser = argparse.ArgumentParser(description="Y2AI Dials Runner - All 28 Modules")
    parser.add_argument('--all', action='store_true', help='Run all dials + portfolio + outputs (28 modules)')
    parser.add_argument('--dials', action='store_true', help='Run dials only, phases 1-7 (23 modules)')
    parser.add_argument('--portfolio', action='store_true', help='Run portfolio modules only (3 modules)')
    parser.add_argument('--dashboard', action='store_true', help='Run dashboard + brief only (2 modules)')
    parser.add_argument('--no-save', action='store_true', help='Calculate only, do not save to Supabase')
    parser.add_argument('--phase', type=int, choices=[1,2,3,4,5,6,7,8,9], help='Run specific phase only')
    
    args = parser.parse_args()
    
    save = not args.no_save
    runner = DialsRunner(save_to_supabase=save)
    
    if args.phase:
        phases = {
            1: runner.run_phase1_foundation,
            2: runner.run_phase2_core,
            3: runner.run_phase3_additional,
            4: runner.run_phase4_flow,
            5: runner.run_phase5_multipliers,
            6: runner.run_phase6_nst_bubble,
            7: runner.run_phase7_analysis,
            8: runner.run_phase8_portfolio,
            9: runner.run_phase9_output
        }
        count = phases[args.phase]()
        print(f"\n✅ Phase {args.phase} complete: {count} modules")
    elif args.all:
        result = runner.run_all_dials()
        print(f"\n✅ Complete: {result['success']}/{result['total']} in {result['duration']:.1f}s")
        if result['errors']:
            print(f"⚠️  {len(result['errors'])} errors occurred")
    elif args.dials:
        result = runner.run_dials_only()
        print(f"\n✅ Complete: {result['success']}/{result['total']} in {result['duration']:.1f}s")
    elif args.portfolio:
        result = runner.run_portfolio_only()
        print(f"\n✅ Complete: {result['success']}/{result['total']}")
    elif args.dashboard:
        result = runner.run_dashboard_only()
        print(f"\n✅ Complete: {result['success']}/{result['total']}")
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("MODULE SUMMARY: 28 total (26 dials + 2 portfolio)")
        print("="*60)
        print("""
Phase 1 - Foundation (3):  PillarIndex, VixDial, CreditSpreadDial
Phase 2 - Core (5):        BreadthDial, MCI, MacroDial, SignalsDial, CorrelationDial
Phase 3 - Additional (5):  ClusterDial, LiquidityDial, LaborDial, ETFDial, SentimentDial
Phase 4 - Flow (2):        StockFlowDial, FlowDivergence
Phase 5 - Multipliers (2): MacroMultipliers, FinancialStress
Phase 6 - NST/Bubble (3):  NSTDial, BubbleIndex, BubbleOverlay  <-- BubbleIndex added
Phase 7 - Analysis (3):    HypergraphDial, FingerprintDial, TrendsHistory
Phase 8 - Portfolio (3):   ShadowPortfolio, RegimeArbiter, PortfolioTracker
Phase 9 - Output (2):      Dashboard, MorningBrief
""")
        print("Examples:")
        print("  python dials_runner.py --all       # Run everything (28 modules)")
        print("  python dials_runner.py --dials     # Run dials only (23 modules)")
        print("  python dials_runner.py --phase 6   # Run NST & Bubble only")
        print("  python dials_runner.py --no-save   # Dry run, no Supabase writes")


if __name__ == "__main__":
    main()