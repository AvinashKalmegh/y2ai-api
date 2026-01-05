"""
Y2AI Dials - Market condition indicators
========================================

CORE DIALS:
- PillarIndex: 6-pillar returns and momentum (foundation)
- BreadthDial: Market participation (20D, 50D breadth)
- MCI: Market Condition Index (-100 to +100)

MARKET DIALS:
- CreditSpreadDial: HY/IG spreads from FRED
- VixDial: VIX with Bollinger Bands and Pre-Shock detection
- SignalsDial: Infrastructure signals + VETO alerts
- MacroDial: FRED macro indicators

ADDITIONAL DIALS:
- ClusterDial: Cluster consolidation / herding metrics
- LiquidityDial: Vol-of-Vol based liquidity assessment
- ETFDial: ETF flow tracking
- LaborDial: BLS labor market indicators
- SentimentDial: News sentiment aggregation
- StockFlowDial: Accumulation/distribution tracking
- FlowDivergence: ETF vs Stock flow comparison
- MacroMultipliers: Position sizing multipliers

OUTPUTS:
- Dashboard: Aggregated view of all dials
- MorningBrief: Daily brief generator
"""

# Core dials
from .pillar_index import PillarIndexCalculator, PillarDayData, PillarSignal, PILLAR_STOCKS
from .breadth_dial import BreadthCalculator, BreadthData, PillarBreadth
from .mci import MCICalculator, MCIData, MCIComponent

# Market dials
from .credit_spread_dial import CreditSpreadCalculator, CreditSpreadData
from .signals_dial import SignalsDialCalculator, SignalsData

# Additional dials
from .cluster_dial import ClusterDialCalculator, ClusterData
from .liquidity_dial import LiquidityDialCalculator, LiquidityData
from .etf_dial import ETFDialCalculator, ETFDialData
from .labor_dial import LaborDialCalculator, LaborDialData
from .sentiment_dial import SentimentDialCalculator, SentimentData
from .stock_flow_dial import StockFlowCalculator, StockFlowDialData
from .flow_divergence import FlowDivergenceCalculator, FlowDivergenceData
from .macro_multipliers import MacroMultiplierCalculator, MultiplierData

# Outputs
from .dashboard import DashboardGenerator, DashboardData
from .morning_brief import MorningBriefGenerator, MorningBrief

__all__ = [
    # Core
    "PillarIndexCalculator", "PillarDayData", "PillarSignal", "PILLAR_STOCKS",
    "BreadthCalculator", "BreadthData", "PillarBreadth",
    "MCICalculator", "MCIData", "MCIComponent",
    # Market
    "CreditSpreadCalculator", "CreditSpreadData",
    "SignalsDialCalculator", "SignalsData",
    # Additional
    "ClusterDialCalculator", "ClusterData",
    "LiquidityDialCalculator", "LiquidityData",
    "ETFDialCalculator", "ETFDialData",
    "LaborDialCalculator", "LaborDialData",
    "SentimentDialCalculator", "SentimentData",
    "StockFlowCalculator", "StockFlowDialData",
    "FlowDivergenceCalculator", "FlowDivergenceData",
    "MacroMultiplierCalculator", "MultiplierData",
    # Outputs
    "DashboardGenerator", "DashboardData",
    "MorningBriefGenerator", "MorningBrief",
]
