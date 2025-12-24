"""
ARGUS-1 Analytical Layer

This package provides the analytical calculation layer for ARGUS-1,
computing AMRI, TTI, SAC, and regime determination from raw data
stored in Supabase tables.

Usage:
    from analytical import ARGUS1Calculator
    
    calc = ARGUS1Calculator()
    result = calc.run()
    
    print(result.regime)           # Regime.NORMAL, ELEVATED, TENSION, FRAGILE, BREAK
    print(result.amri.composite)   # 0-100 AMRI score
    print(result.tti.display)      # "3D to FRAGILE"
    print(result.format_summary()) # Full text summary
    
    # Store to Supabase
    calc.store_result(result)
    
    # Get dashboard-compatible cell values
    cells = result.format_dashboard_cells()

Components:
    - AMRI: Master regime index with S/B/C decomposition
    - TTI: Time-to-Instability projections
    - SAC: Shock Absorption Capacity
    - NST: News Sentiment Tracking (from daily_signals)
    - Contagion: Hypergraph contagion (from hypergraph_signals)
    - Fingerprints: Historical episode matching
    - Rotation: Pillar leadership tracking
    - Recovery: Safe re-entry signal detection
    - Events: Market calendar tracking

Data Sources:
    - bubble_index_daily: VIX, CAPE, Credit Spreads
    - hypergraph_signals: Contagion, stability, clusters
    - daily_signals: NLP signals, VETO triggers
    - stock_tracker_daily: Pillar performance
"""

from .config import (
    Regime,
    Authority,
    Confidence,
    AMRI_WEIGHTS,
    AMRI_THRESHOLDS,
    CONTAGION_THRESHOLDS,
    VIX_THRESHOLDS,
    PILLARS,
    TICKER_TO_PILLAR,
    HISTORICAL_EPISODES,
    DASHBOARD_CELLS,
)

from .amri import (
    AMRICalculator,
    AMRIResult,
    AMRIComponents,
    AMRIDecomposition,
)

from .tti import (
    TTICalculator,
    TTIResult,
)

from .sac import (
    SACCalculator,
    SACResult,
    SACComponents,
)

from .fingerprints import (
    FingerprintMatcher,
    FingerprintMatch,
)

from .rotation import (
    RotationTracker,
    RotationResult,
    RecoveryDetector,
    RecoveryResult,
)

from .events import (
    EventsTracker,
    EventsResult,
    MarketEvent,
)

from .calculator import (
    ARGUS1Calculator,
    ARGUS1Result,
    NSTResult,
    ContagionResult,
)


__all__ = [
    # Main calculator
    "ARGUS1Calculator",
    "ARGUS1Result",
    
    # Enums
    "Regime",
    "Authority", 
    "Confidence",
    
    # AMRI
    "AMRICalculator",
    "AMRIResult",
    "AMRIComponents",
    "AMRIDecomposition",
    
    # TTI
    "TTICalculator",
    "TTIResult",
    
    # SAC
    "SACCalculator",
    "SACResult",
    "SACComponents",
    
    # Fingerprints
    "FingerprintMatcher",
    "FingerprintMatch",
    
    # Rotation & Recovery
    "RotationTracker",
    "RotationResult",
    "RecoveryDetector",
    "RecoveryResult",
    
    # Events
    "EventsTracker",
    "EventsResult",
    "MarketEvent",
    
    # Supporting results
    "NSTResult",
    "ContagionResult",
    
    # Config
    "PILLARS",
    "TICKER_TO_PILLAR",
    "DASHBOARD_CELLS",
]

__version__ = "1.0.0"
