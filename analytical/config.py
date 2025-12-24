"""
ARGUS-1 Analytical Configuration
All thresholds, weights, and constants for regime detection

Based on handoff documentation and vik.py formulas.
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
# AMRI WEIGHTS (from vik.py documentation)
# =============================================================================

# AMRI Composite: AMRI = 0.23*CRS + 0.31*CCS + 0.15*SRS + 0.31*SDS
AMRI_WEIGHTS = {
    "CRS": 0.23,  # Correlation Regime Score (coincident)
    "CCS": 0.31,  # Cluster Concentration Score (leading)
    "SRS": 0.15,  # Spread Regime Score (lagging)
    "SDS": 0.31,  # Structural Divergence Score (leading)
}

# Alternative weights from handoff doc
# AMRI = CRS×0.25 + CCS×0.25 + SRS×0.20 + VIX×0.15 + SDS×0.15
AMRI_WEIGHTS_ALT = {
    "CRS": 0.25,
    "CCS": 0.25,
    "SRS": 0.20,
    "VIX": 0.15,
    "SDS": 0.15,
}

# AMRI Decomposition Weights
# AMRI-S (Structural capacity to break)
AMRI_S_WEIGHTS = {
    "CRS": 0.43,  # Correlation contribution
    "CCS": 0.57,  # Cluster contribution
}

# AMRI-B (Behavioral pressure building)
AMRI_B_WEIGHTS = {
    "SDS": 0.55,  # Structural divergence
    "Bubble": 0.45,  # Bubble index component
}

# AMRI-C (Catalyst risk)
AMRI_C_WEIGHTS = {
    "SRS": 0.15,  # Spread regime
    "NST": 0.35,  # News sentiment
    "Contagion": 0.50,  # Hypergraph contagion
}


# =============================================================================
# CONTAGION FORMULA (from hypergraph docs)
# =============================================================================

# Contagion = (cross_pillar_ratio × 50) + (1 - stability) × 30 + (1 - 1/avg_size) × 20
CONTAGION_WEIGHTS = {
    "cross_pillar": 50,
    "instability": 30,
    "size_factor": 20,
}


# =============================================================================
# SAC (Shock Absorption Capacity) WEIGHTS
# =============================================================================

# SAC = AMRI_buffer×0.30 + Bubble_buffer×0.20 + Contagion_buffer×0.25 + 
#       Correlations_buffer×0.15 + Breadth_buffer×0.10
SAC_WEIGHTS = {
    "AMRI_buffer": 0.30,
    "Bubble_buffer": 0.20,
    "Contagion_buffer": 0.25,
    "Correlations_buffer": 0.15,
    "Breadth_buffer": 0.10,
}


# =============================================================================
# REGIME THRESHOLDS
# =============================================================================

# AMRI thresholds for regime determination
AMRI_THRESHOLDS = {
    "NORMAL": (0, 40),      # Safe zone
    "ELEVATED": (40, 60),   # Watchful
    "TENSION": (60, 75),    # Caution
    "FRAGILE": (75, 90),    # Danger
    "BREAK": (90, 100),     # Critical
}

# Contagion score thresholds (0-100)
CONTAGION_THRESHOLDS = {
    "STABLE": (0, 30),
    "TRANSITIONING": (30, 50),
    "FRAGMENTING": (50, 70),
    "CONTAGION": (70, 100),
}

# VIX thresholds
VIX_THRESHOLDS = {
    "LOW": (0, 15),
    "NORMAL": (15, 20),
    "ELEVATED": (20, 25),
    "HIGH": (25, 35),
    "EXTREME": (35, 100),
}

# Credit spread thresholds (basis points)
CREDIT_SPREAD_THRESHOLDS = {
    "TIGHT": (0, 100),
    "NORMAL": (100, 150),
    "WIDE": (150, 250),
    "DISTRESSED": (250, 500),
    "CRISIS": (500, 2000),
}


# =============================================================================
# FRAGILITY MODEL (Boolean Logic Gates from vik.py)
# =============================================================================

# Condition 1: Structural Homogeneity - clusters < threshold
FRAGILITY_CLUSTER_THRESHOLD = 7

# Condition 2: Divergence Trap - divergence > threshold
FRAGILITY_DIVERGENCE_THRESHOLD = 0.20

# Condition 3: Volatility Ignition - VIX delta > threshold
FRAGILITY_VIX_DELTA_THRESHOLD = 2.0

# Combined logic:
# C1 AND C2 AND C3 → CRITICAL_BREAK
# C1 AND C2 → TENSION
# Otherwise → CLEAR


# =============================================================================
# TTI (Time-to-Instability) THRESHOLDS
# =============================================================================

TTI_THRESHOLDS = {
    "IMMINENT": (1, 2),      # 1-2 days
    "DAYS": (3, 7),          # 3-7 days
    "WEEKS": (8, 30),        # 8-30 days
    "MONTHS": (31, 90),      # 1-3 months
    "DISTANT": (91, 365),    # 3+ months
}

# Rate of change window for TTI calculation
TTI_LOOKBACK_DAYS = 10


# =============================================================================
# NST (News Sentiment Tracking) THRESHOLDS
# =============================================================================

# VETO trigger thresholds
VETO_THRESHOLDS = {
    "burst_count": 20,
    "veto_triggers": 25,
    "evi_score": 85,
    "thesis_swing": 15,  # Absolute change from previous day
}

# NCI (Narrative Coherence Index) interpretation
NCI_THRESHOLDS = {
    "COHERENT": (70, 100),
    "MIXED": (40, 70),
    "FRACTURED": (0, 40),
}


# =============================================================================
# HISTORICAL EPISODES FOR FINGERPRINT MATCHING
# =============================================================================

@dataclass
class HistoricalEpisode:
    """Historical market episode for fingerprint matching"""
    name: str
    start_date: str
    end_date: str
    peak_amri: float
    peak_contagion: float
    peak_vix: float
    days_to_break: int
    pattern_type: str  # "FAST_BREAK", "SLOW_GRIND", "V_RECOVERY"
    characteristics: Dict[str, float]


HISTORICAL_EPISODES = [
    HistoricalEpisode(
        name="COVID_CRASH_2020",
        start_date="2020-02-19",
        end_date="2020-03-23",
        peak_amri=95,
        peak_contagion=92,
        peak_vix=82.69,
        days_to_break=23,
        pattern_type="FAST_BREAK",
        characteristics={
            "cluster_collapse_speed": 0.95,
            "cross_pillar_surge": 0.88,
            "vix_acceleration": 0.92,
        }
    ),
    HistoricalEpisode(
        name="TECH_CRASH_2022",
        start_date="2021-11-19",
        end_date="2022-10-12",
        peak_amri=78,
        peak_contagion=65,
        peak_vix=36.45,
        days_to_break=327,
        pattern_type="SLOW_GRIND",
        characteristics={
            "cluster_collapse_speed": 0.35,
            "cross_pillar_surge": 0.45,
            "vix_acceleration": 0.40,
        }
    ),
    HistoricalEpisode(
        name="SVB_CRISIS_2023",
        start_date="2023-03-08",
        end_date="2023-03-15",
        peak_amri=72,
        peak_contagion=78,
        peak_vix=26.52,
        days_to_break=5,
        pattern_type="FAST_BREAK",
        characteristics={
            "cluster_collapse_speed": 0.82,
            "cross_pillar_surge": 0.75,
            "vix_acceleration": 0.65,
        }
    ),
    HistoricalEpisode(
        name="AUG_2024_UNWIND",
        start_date="2024-07-31",
        end_date="2024-08-05",
        peak_amri=68,
        peak_contagion=71,
        peak_vix=65.73,
        days_to_break=3,
        pattern_type="FAST_BREAK",
        characteristics={
            "cluster_collapse_speed": 0.88,
            "cross_pillar_surge": 0.70,
            "vix_acceleration": 0.95,
        }
    ),
    HistoricalEpisode(
        name="DOT_COM_2000",
        start_date="2000-03-10",
        end_date="2002-10-09",
        peak_amri=88,
        peak_contagion=72,
        peak_vix=45.74,
        days_to_break=944,
        pattern_type="SLOW_GRIND",
        characteristics={
            "cluster_collapse_speed": 0.25,
            "cross_pillar_surge": 0.55,
            "vix_acceleration": 0.45,
        }
    ),
]


# =============================================================================
# PILLAR DEFINITIONS (6 Pillars)
# =============================================================================

PILLARS = {
    "COMPUTE": {
        "description": "GPU/Chip makers",
        "tickers": ["NVDA", "AMD", "AVGO", "QCOM", "INTC", "MU", "MRVL"],
    },
    "INFRASTRUCTURE": {
        "description": "Data centers, cooling, power",
        "tickers": ["EQIX", "DLR", "VRT", "ANET", "KEYS"],
    },
    "FOUNDATION_MODELS": {
        "description": "AI model developers",
        "tickers": ["GOOGL", "MSFT", "META", "AMZN", "ORCL"],
    },
    "AI_NATIVE": {
        "description": "Pure-play AI companies",
        "tickers": ["PLTR", "AI", "PATH", "SNOW", "MDB"],
    },
    "PRODUCTIVITY": {
        "description": "AI-enhanced enterprise",
        "tickers": ["CRM", "NOW", "ADBE", "WDAY", "INTU"],
    },
    "DEMAND": {
        "description": "End-user demand proxies",
        "tickers": ["TSLA", "UBER", "ABNB", "SPOT", "ROKU"],
    },
}

# Flatten to ticker -> pillar mapping
TICKER_TO_PILLAR = {}
for pillar, info in PILLARS.items():
    for ticker in info["tickers"]:
        TICKER_TO_PILLAR[ticker] = pillar


# =============================================================================
# UPDATE SCHEDULING
# =============================================================================

# Update intervals in hours based on regime
UPDATE_INTERVALS = {
    Regime.NORMAL: 24,
    Regime.ELEVATED: 12,
    Regime.TENSION: 6,
    Regime.FRAGILE: 4,
    Regime.BREAK: 2,
}

# Staleness thresholds
STALENESS_WARNING_HOURS = 24
STALENESS_CRITICAL_HOURS = 48


# =============================================================================
# DASHBOARD CELL MAPPINGS (for Google Sheets compatibility)
# =============================================================================

DASHBOARD_CELLS = {
    # AMRI section
    "AMRI": "B59",
    "REGIME": "B60",
    "AUTHORITY": "B61",
    "CONFIDENCE": "B62",
    "AMRI_S": "B63",
    "AMRI_B": "B64",
    "TTI": "B65",
    "TTI_BINDING": "B66",
    "AMRI_C": "B67",
    
    # SAC section
    "SAC": "B68",
    "SAC_WEAKEST": "B69",
    
    # Bubble section
    "BUBBLE": "B70",
    "RECOVERY": "B71",
    "ROTATION": "B72",
    "LEADER": "B73",
    "LAGGARD": "B74",
    
    # Contagion section
    "CONTAGION": "B80",
    "CONTAGION_REGIME": "B81",
    "STABILITY": "B82",
    "HYPEREDGE_COUNT": "B83",
    "DATA_AGE": "B84",
    
    # NST section
    "NST": "B85",
    "VETO_ACTIVE": "B86",
    "NCI": "B87",
    "NPD": "B88",
    "BURST": "B89",
    
    # Fingerprint section
    "FINGERPRINT_EPISODE": "B90",
    "FINGERPRINT_MATCH": "B91",
    "FINGERPRINT_QUALITY": "B92",
    "FINGERPRINT_PATTERN": "B93",
    
    # Events section
    "EVENT_STATUS": "B95",
    "NEXT_EVENT": "B96",
    "DAYS_TO_EVENT": "B97",
}
