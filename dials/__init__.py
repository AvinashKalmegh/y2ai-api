"""
Y2AI Dials - Market condition indicators
========================================

PHASE 1 - FOUNDATION (3):
- PillarIndex: 6-pillar returns and momentum (foundation)
- VixDial: VIX with Bollinger Bands and Pre-Shock detection
- CreditSpreadDial: HY/IG spreads from FRED

PHASE 2 - CORE (5):
- BreadthDial: Market participation (20D, 50D breadth)
- MCI: Market Condition Index (-100 to +100)
- MacroDial: FRED macro indicators
- SignalsDial: Infrastructure signals + VETO alerts
- CorrelationDial: Pillar-level correlations (20D/60D)

PHASE 3 - ADDITIONAL (5):
- ClusterDial: Cluster consolidation / herding metrics
- LiquidityDial: Vol-of-Vol based liquidity assessment
- LaborDial: BLS labor market indicators
- ETFDial: ETF flow tracking
- SentimentDial: Market-data sentiment composite

PHASE 4 - FLOW (2):
- StockFlowDial: Accumulation/distribution tracking
- FlowDivergence: ETF vs Stock flow comparison

PHASE 5 - MULTIPLIERS (2):
- MacroMultipliers: Position sizing multipliers
- FinancialStressDial: STLFSI4 from FRED

PHASE 6 - NST & BUBBLE (2):
- NSTDial: Integrated news sentiment (NCI + NPD + EVI + Burst)
- BubbleOverlayDial: Academic bubble detection (PSY, LPPLS, LZC)

PHASE 7 - ANALYSIS (3):
- HypergraphDial: Contagion layer (cross-pillar risk)
- FingerprintDial: Historical episode matching
- TrendsHistoryDial: Daily metrics logging and trend analysis

PHASE 8 - PORTFOLIO (1 dial + 2 portfolio modules):
- ShadowPortfolioDial: Portfolio tracking and risk alerts
- (RegimeArbiter): portfolio module
- (PortfolioTracker): portfolio module

PHASE 9 - OUTPUTS (2):
- Dashboard: Aggregated view of all dials
- MorningBrief: Daily brief generator

TOTAL: 25 dial modules + 2 portfolio modules = 27 modules
"""

# Phase 1 - Foundation
from .pillar_index import PillarIndexCalculator, PillarDayData, PillarSignal, PILLAR_STOCKS
from .vix_dial import VixDialCalculator, VixData  # FIXED: was VixDialData
from .credit_spread_dial import CreditSpreadCalculator, CreditSpreadData

# Phase 2 - Core
from .breadth_dial import BreadthCalculator, BreadthData, PillarBreadth
# from .mci import MCICalculator, MCIData, MCIComponent  # TODO: mci.py is a fix script, not a dial module
from .macro_dial import MacroDialCalculator, MacroData  # FIXED: was MacroDialData
from .signals_dial import SignalsDialCalculator, SignalsData
from .correlation_dial import CorrelationDialCalculator, CorrelationDialData

# Phase 3 - Additional
from .cluster_dial import ClusterDialCalculator, ClusterData
from .liquidity_dial import LiquidityDialCalculator, LiquidityData
from .labor_dial import LaborDialCalculator, LaborDialData
from .etf_dial import ETFDialCalculator, ETFDialData
from .sentiment_dial import SentimentDialCalculator, SentimentData

# Phase 4 - Flow
from .stock_flow_dial import StockFlowCalculator, StockFlowDialData
from .flow_divergence import FlowDivergenceCalculator, FlowDivergenceData

# Phase 5 - Multipliers
from .macro_multipliers import MacroMultiplierCalculator, MultiplierData
from .financial_stress_dial import FinancialStressCalculator, FinancialStressData

# Phase 6 - NST & Bubble
from .nst_dial import NSTDialCalculator, NSTDialData
from .bubble_overlay_dial import BubbleOverlayCalculator, BubbleOverlayData

# Phase 7 - Analysis
from .hypergraph_dial import HypergraphDialCalculator, HypergraphData
from .fingerprint_dial import FingerprintLibraryCalculator, FingerprintData
from .trends_history_dial import TrendsHistoryCalculator, TrendsHistoryData

# Phase 8 - Portfolio (dial only; RegimeArbiter/PortfolioTracker in portfolio/)
from .shadow_portfolio_dial import ShadowPortfolioCalculator, ShadowPortfolioData

# Phase 9 - Outputs
from .dashboard import DashboardGenerator, DashboardData
from .morning_brief import MorningBriefGenerator, MorningBrief


__all__ = [
    # Phase 1 - Foundation
    "PillarIndexCalculator", "PillarDayData", "PillarSignal", "PILLAR_STOCKS",
    "VixDialCalculator", "VixData",  # FIXED
    "CreditSpreadCalculator", "CreditSpreadData",
    
    # Phase 2 - Core
    "BreadthCalculator", "BreadthData", "PillarBreadth",
    # "MCICalculator", "MCIData", "MCIComponent",  # TODO: mci.py is a fix script
    "MacroDialCalculator", "MacroData",  # FIXED
    "SignalsDialCalculator", "SignalsData",
    "CorrelationDialCalculator", "CorrelationDialData",
    
    # Phase 3 - Additional
    "ClusterDialCalculator", "ClusterData",
    "LiquidityDialCalculator", "LiquidityData",
    "LaborDialCalculator", "LaborDialData",
    "ETFDialCalculator", "ETFDialData",
    "SentimentDialCalculator", "SentimentData",
    
    # Phase 4 - Flow
    "StockFlowCalculator", "StockFlowDialData",
    "FlowDivergenceCalculator", "FlowDivergenceData",
    # "AttractorMassCalculator", "AttractorMassResult",  # TODO: not imported
    
    # Phase 5 - Multipliers
    "MacroMultiplierCalculator", "MultiplierData",
    "FinancialStressCalculator", "FinancialStressData",
    
    # Phase 6 - NST & Bubble
    "NSTDialCalculator", "NSTDialData",
    "BubbleOverlayCalculator", "BubbleOverlayData",
    
    # Phase 7 - Analysis
    "HypergraphDialCalculator", "HypergraphData",
    "FingerprintLibraryCalculator", "FingerprintData",
    "TrendsHistoryCalculator", "TrendsHistoryData",
    
    # Phase 8 - Portfolio dial
    "ShadowPortfolioCalculator", "ShadowPortfolioData",
    
    # Phase 9 - Outputs
    "DashboardGenerator", "DashboardData",
    "MorningBriefGenerator", "MorningBrief",
]