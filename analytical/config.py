"""
ARGUS-1 Analytical Configuration - CORRECTED
=============================================
All thresholds, weights, and constants for regime detection

FIXES APPLIED:
1. AMRI uses 4 components with equal 25% weights (not 5 components)
2. VIX is NOT a separate AMRI component
3. SRS and SDS weights corrected to 25% each

Based on Google Sheets AMRI_MASTER sheet formulas.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

class Regime(str, Enum):
    """Market regime states"""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    TENSION = "TENSION"
    FRAGILE = "FRAGILE"
    BREAK = "BREAK"


class Authority(str, Enum):
    """
    Authority hierarchy for regime determination.
    Higher authority overrides lower.
    """
    MARKET = "MARKET"        # Level 1: Prices/flows (lowest)
    STRUCTURAL = "STRUCTURAL"  # Level 2: Contagion topology
    NARRATIVE = "NARRATIVE"   # Level 3: NLP/VETO signals
    BREAK = "BREAK"          # Level 4: Hard break conditions (highest)


class Confidence(str, Enum):
    """Confidence level in regime assessment"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# =============================================================================
# AMRI WEIGHTS - CORRECTED TO MATCH GOOGLE SHEETS
# =============================================================================

# CORE AMRI: Equal weights across 4 components
# Formula: AMRI = CRS×0.25 + CCS×0.25 + SRS×0.25 + SDS×0.25
AMRI_WEIGHTS = {
    "CRS": 0.25,  # Correlation Regime Score
    "CCS": 0.25,  # Cluster Concentration Score
    "SRS": 0.25,  # Spread Regime Score
    "SDS": 0.25,  # Structural Divergence Score
}

# DEPRECATED - Do not use this anymore
# This was the old incorrect version with VIX as separate component
AMRI_WEIGHTS_ALT_DEPRECATED = {
    "CRS": 0.25,
    "CCS": 0.25,
    "SRS": 0.20,  # WRONG
    "VIX": 0.15,  # WRONG - VIX is not a component
    "SDS": 0.15,  # WRONG
}

# Enhanced AMRI weights (Core + Bubble Overlay)
ENHANCED_AMRI_WEIGHTS = {
    "CORE": 0.80,
    "OVERLAY": 0.20,
}

# AMRI Decomposition Weights (for S/B/C breakdown)
AMRI_S_WEIGHTS = {
    "CRS": 0.43,  # Correlation contribution
    "CCS": 0.57,  # Cluster contribution
}

AMRI_B_WEIGHTS = {
    "SDS": 0.55,  # Structural divergence
    "Bubble": 0.45,  # Bubble index component
}

AMRI_C_WEIGHTS = {
    "SRS": 0.15,  # Spread regime
    "NST": 0.35,  # News sentiment
    "Contagion": 0.50,  # Hypergraph contagion
}


# =============================================================================
# MCI CONFIGURATION - CORRECTED
# =============================================================================

MCI_CONFIG = {
    # Lookback periods (trading days)
    "BREADTH_LOOKBACK": 5,
    "VIX_LOOKBACK": 10,
    "CREDIT_LOOKBACK": 10,
    "PILLAR_LOOKBACK": 5,
    
    # Component weights (must sum to 100)
    "WEIGHT_BREADTH": 25,
    "WEIGHT_VIX": 25,
    "WEIGHT_CREDIT": 25,
    "WEIGHT_PILLAR": 25,
    
    # CORRECTED Thresholds
    "BREADTH_THRESHOLD": 10,  # +/- 10% breadth change = max score
    "VIX_THRESHOLD": 6,       # CORRECTED: +/- 6 VIX points = max score
    "CREDIT_THRESHOLD": 20,   # +/- 20 bps spread change = max score  
    "PILLAR_THRESHOLD": 3,    # +/- 3% pillar return = max score
}


# =============================================================================
# BUBBLE INDEX CONFIGURATION - CORRECTED
# =============================================================================

# 6-component Bubble Index formula from Google Sheets
BUBBLE_INDEX_WEIGHTS = {
    "VALUATION_EXTREME": 0.25,
    "GROWTH_DISCONNECT": 0.20,
    "MOMENTUM_MANIA": 0.20,
    "VOLATILITY_STRESS": 0.15,
    "CONCENTRATION_RISK": 0.10,
    "VIX_COMPLACENCY": 0.10,
}

# Volatility Stress sub-components
VOLATILITY_STRESS_WEIGHTS = {
    "BETA": 0.40,
    "CORRELATION": 0.30,
    "CREDIT_MOMENTUM": 0.30,
}


# =============================================================================
# REGIME THRESHOLDS
# =============================================================================

AMRI_THRESHOLDS = {
    "NORMAL": (0, 40),      # Safe zone
    "ELEVATED": (40, 55),   # CORRECTED: Changed from 60
    "TENSION": (55, 70),    # CORRECTED: Changed from 75
    "FRAGILE": (70, 85),    # CORRECTED: Changed from 90
    "BREAK": (85, 100),     # Critical
}

BUBBLE_INDEX_THRESHOLDS = {
    "NORMAL": (0, 30),
    "ELEVATED": (30, 45),
    "CAUTION": (45, 60),
    "HIGH": (60, 75),
    "CRITICAL": (75, 100),
}

MCI_THRESHOLDS = {
    "MELT_UP": (40, 101),
    "EXTENSION": (10, 40),
    "KNIFE_EDGE": (-10, 10),
    "COLLAPSE_BIAS": (-40, -10),
    "BREAK_PATH": (-101, -40),
}

CONTAGION_THRESHOLDS = {
    "STABLE": (0, 30),
    "TRANSITIONING": (30, 50),
    "FRAGMENTING": (50, 70),
    "CONTAGION": (70, 100),
}

VIX_THRESHOLDS = {
    "LOW": (0, 15),
    "NORMAL": (15, 20),
    "ELEVATED": (20, 25),
    "HIGH": (25, 35),
    "EXTREME": (35, 100),
}

CREDIT_SPREAD_THRESHOLDS = {
    "TIGHT": (0, 100),
    "NORMAL": (100, 150),
    "WIDE": (150, 250),
    "DISTRESSED": (250, 500),
    "CRISIS": (500, 2000),
}


# =============================================================================
# FRAGILITY MODEL (Boolean Logic Gates)
# =============================================================================

FRAGILITY_THRESHOLDS = {
    # Condition 1: Cluster Consolidation
    "CLUSTER_THRESHOLD": 7,  # clusters < 7 = ACTIVE
    
    # Condition 2: Pillar Divergence  
    "INFRA_BREADTH_HIGH": 0.55,  # Infra > 55%
    "ENT_BREADTH_LOW": 0.35,     # Ent < 35%
    
    # Condition 3: VIX Spike
    "VIX_SPIKE_THRESHOLD": 2.0,  # VIX 5D change > 2 pts
}


# =============================================================================
# PILLAR DEFINITIONS (6 Pillars from Google Sheets)
# =============================================================================

PILLARS = {
    "Infrastructure": {
        "description": "Infrastructure & Energy",
        "tickers": ["NVDA", "AMD", "AVGO", "TSM", "INTC", "MRVL", "ANET", 
                    "ARM", "SMCI", "VRT", "DELL", "HPE", "EQIX", "DLR", "AMT", "CEG"],
    },
    "Enterprise": {
        "description": "Enterprise Adoption",
        "tickers": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "PLTR", 
                    "SNOW", "DDOG", "MDB", "PANW", "CRWD", "ZS"],
    },
    "Macro": {
        "description": "Macro & Policy",
        "tickers": ["GS", "MS", "JPM", "V"],
    },
    "Financial": {
        "description": "Financial & Market",
        "tickers": ["BLK", "CME", "ICE", "SCHW"],
    },
    "Productivity": {
        "description": "Productivity & Labor",
        "tickers": ["ADBE", "INTU", "WDAY"],
    },
    "Demand": {
        "description": "Demand Dynamics",
        "tickers": ["AAPL", "TSLA", "NFLX"],
    },
}

# Flatten ticker to pillar mapping
TICKER_TO_PILLAR = {}
for pillar, info in PILLARS.items():
    for ticker in info["tickers"]:
        TICKER_TO_PILLAR[ticker] = pillar


# =============================================================================
# TTI (Time-to-Instability) THRESHOLDS
# =============================================================================

TTI_THRESHOLDS = {
    "IMMINENT": (1, 2),
    "DAYS": (3, 7),
    "WEEKS": (8, 30),
    "MONTHS": (31, 90),
    "DISTANT": (91, 365),
}

TTI_LOOKBACK_DAYS = 10


# =============================================================================
# UPDATE SCHEDULING
# =============================================================================

UPDATE_INTERVALS = {
    Regime.NORMAL: 24,
    Regime.ELEVATED: 12,
    Regime.TENSION: 6,
    Regime.FRAGILE: 4,
    Regime.BREAK: 2,
}

STALENESS_WARNING_HOURS = 24
STALENESS_CRITICAL_HOURS = 48


# Historical episodes for validation (required by other modules)
HISTORICAL_EPISODES = {
    "Y2K_BUBBLE": {
        "start": "1998-01-01",
        "end": "2000-03-10",
        "type": "bubble",
    },
    "DOTCOM_CRASH": {
        "start": "2000-03-10",
        "end": "2002-10-09",
        "type": "crash",
    },
    "HOUSING_BUBBLE": {
        "start": "2005-01-01",
        "end": "2007-10-09",
        "type": "bubble",
    },
    "GFC_CRASH": {
        "start": "2007-10-09",
        "end": "2009-03-09",
        "type": "crash",
    },
    "COVID_CRASH": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "type": "crash",
    },
    "AI_CYCLE": {
        "start": "2023-01-01",
        "end": None,
        "type": "expansion",
    },
}


DASHBOARD_CELLS = {
    "AMRI": "B2",
    "BUBBLE_INDEX": "B3",
    "MCI": "B4",
    "VIX": "B5",
    "REGIME": "B6",
}


# SAC (Systematic Allocation Calculator) weights
SAC_WEIGHTS = {
    "AMRI_buffer": 0.30,
    "Bubble_buffer": 0.25,
    "Contagion_buffer": 0.20,
    "Correlations_buffer": 0.15,
    "Breadth_buffer": 0.10,
}

# Dashboard cell references
DASHBOARD_CELLS = {
    "AMRI": "B2",
    "BUBBLE_INDEX": "B3",
    "MCI": "B4",
    "VIX": "B5",
    "REGIME": "B6",
}

# Historical episodes for validation
HISTORICAL_EPISODES = {
    "Y2K_BUBBLE": {"start": "1998-01-01", "end": "2000-03-10", "type": "bubble"},
    "DOTCOM_CRASH": {"start": "2000-03-10", "end": "2002-10-09", "type": "crash"},
    "HOUSING_BUBBLE": {"start": "2005-01-01", "end": "2007-10-09", "type": "bubble"},
    "GFC_CRASH": {"start": "2007-10-09", "end": "2009-03-09", "type": "crash"},
    "COVID_CRASH": {"start": "2020-02-19", "end": "2020-03-23", "type": "crash"},
    "AI_CYCLE": {"start": "2023-01-01", "end": None, "type": "expansion"},
}