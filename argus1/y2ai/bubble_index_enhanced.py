"""
Y2AI BUBBLE INDEX SERVICE (ENHANCED)
With robust error handling, fallback values, and health tracking

Formula Reference (from Santa Fe paper):
Bifurcation Score = 0.6*BI - 0.2*VI - 0.2*CS

Where:
- BI = Bubble Index (valuation extremeness, 0-100)
- VI = VIX z-score (volatility regime)  
- CS = Credit Spreads z-score (financial stress)

Regimes:
- INFRASTRUCTURE: score > +0.5
- ADOPTION: +0.2 to +0.5
- TRANSITION: -0.2 to +0.2
- BUBBLE_WARNING: < -0.2
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

import pandas as pd
import numpy as np

# Import resilience module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.resilience import (
    resilient_call,
    with_fallback,
    get_http_session,
    get_health_tracker,
    RetryExhaustedError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FALLBACK VALUES
# =============================================================================

# Historical median/typical values used when APIs fail
FALLBACK_VALUES = {
    "vix": 18.0,           # Long-term median VIX
    "cape": 28.0,          # Recent decade average CAPE
    "credit_ig": 100.0,    # Typical IG spread (bps)
    "credit_hy": 350.0,    # Typical HY spread (bps)
}

# Historical data for z-score calculation (last 5 years approximation)
HISTORICAL_STATS = {
    "vix": {"mean": 19.5, "std": 7.5},
    "cape": {"mean": 30.0, "std": 5.0},
    "credit": {"mean": 150.0, "std": 50.0},
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class BubbleIndexReading:
    """Complete bubble index reading with all components"""
    date: str
    
    # Raw values
    vix: float
    cape: float
    credit_spread_ig: float
    credit_spread_hy: float
    
    # Z-scores
    vix_zscore: float
    cape_zscore: float
    credit_zscore: float
    
    # Bubble Index
    bubble_index: float  # 0-100 scale
    
    # Unified bifurcation score
    bifurcation_score: float  # -1 to +1 typically
    
    # Regime interpretation
    regime: str  # INFRASTRUCTURE, ADOPTION, TRANSITION, BUBBLE_WARNING
    
    # Data quality indicators
    data_sources: Dict[str, str] = field(default_factory=dict)  # source -> "live" or "fallback"
    
    # Metadata
    calculated_at: str = ""
    
    def __post_init__(self):
        if not self.calculated_at:
            self.calculated_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def data_quality_score(self) -> float:
        """Score from 0-1 indicating how much live data we have"""
        if not self.data_sources:
            return 1.0
        live_count = sum(1 for v in self.data_sources.values() if v == "live")
        return live_count / len(self.data_sources)


# =============================================================================
# DATA FETCHERS (Enhanced)
# =============================================================================

class VIXFetcher:
    """Fetch VIX data with fallback strategies"""
    
    def __init__(self):
        self._session = get_http_session()
        self._cached_history: Optional[pd.Series] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)
    
    @resilient_call(
        service_name="yfinance_vix",
        max_retries=3,
        base_delay=1.0,
        use_circuit_breaker=True,
        use_rate_limiter=False,
    )
    def _fetch_vix_yfinance(self, period: str = "1d") -> pd.DataFrame:
        """Fetch VIX from yfinance with resilience"""
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        return vix.history(period=period)
    
    def get_current_vix(self) -> Tuple[float, str]:
        """
        Get current VIX value.
        
        Returns:
            Tuple of (value, source) where source is "live" or "fallback"
        """
        try:
            hist = self._fetch_vix_yfinance("1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1]), "live"
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
        
        logger.info("Using fallback VIX value")
        return FALLBACK_VALUES["vix"], "fallback"
    
    def get_vix_history(self, months: int = 60) -> pd.Series:
        """Get VIX history for z-score calculation with caching"""
        # Check cache
        if self._cached_history is not None and self._cache_time:
            if datetime.utcnow() - self._cache_time < self._cache_duration:
                return self._cached_history
        
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            end = datetime.now()
            start = end - timedelta(days=months * 30)
            hist = vix.history(start=start, end=end)
            
            if not hist.empty:
                self._cached_history = hist['Close']
                self._cache_time = datetime.utcnow()
                return self._cached_history
        except Exception as e:
            logger.warning(f"VIX history fetch failed: {e}")
        
        # Return synthetic history based on historical stats
        logger.info("Using synthetic VIX history")
        mean = HISTORICAL_STATS["vix"]["mean"]
        std = HISTORICAL_STATS["vix"]["std"]
        return pd.Series([mean + np.random.normal(0, std) for _ in range(months)])


class CAPEFetcher:
    """Fetch Shiller CAPE ratio with multiple fallback strategies"""
    
    def __init__(self):
        self._session = get_http_session()
        self.shiller_url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    
    @resilient_call(
        service_name="shiller_cape",
        max_retries=2,
        base_delay=2.0,
        use_circuit_breaker=True,
        use_rate_limiter=False,
    )
    def _fetch_shiller_data(self) -> pd.DataFrame:
        """Fetch Shiller data from Yale with resilience"""
        return pd.read_excel(self.shiller_url, sheet_name='Data', skiprows=7)
    
    def _calculate_cape_from_spy(self) -> Optional[float]:
        """Calculate approximate CAPE from SPY P/E as fallback"""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            pe = spy.info.get('trailingPE')
            if pe:
                # CAPE is typically 1.3-1.5x trailing P/E
                return pe * 1.4
        except Exception as e:
            logger.warning(f"SPY P/E fallback failed: {e}")
        return None
    
    def get_current_cape(self) -> Tuple[float, str]:
        """
        Get current CAPE ratio.
        
        Returns:
            Tuple of (value, source) where source is "live", "spy_estimate", or "fallback"
        """
        # Strategy 1: Shiller data
        try:
            df = self._fetch_shiller_data()
            df = df.dropna(subset=[df.columns[10]])  # CAPE column
            cape = float(df.iloc[-1, 10])
            return cape, "live"
        except Exception as e:
            logger.warning(f"Shiller CAPE fetch failed: {e}")
        
        # Strategy 2: Calculate from SPY
        spy_cape = self._calculate_cape_from_spy()
        if spy_cape:
            return spy_cape, "spy_estimate"
        
        # Strategy 3: Fallback
        logger.info("Using fallback CAPE value")
        return FALLBACK_VALUES["cape"], "fallback"
    
    def get_cape_history(self, months: int = 60) -> pd.Series:
        """Get CAPE history for z-score calculation"""
        try:
            df = self._fetch_shiller_data()
            df = df.dropna(subset=[df.columns[10]])
            cape_series = df.iloc[-months:, 10].astype(float)
            return pd.Series(cape_series.values)
        except Exception as e:
            logger.warning(f"CAPE history fetch failed: {e}")
        
        # Return synthetic history
        mean = HISTORICAL_STATS["cape"]["mean"]
        std = HISTORICAL_STATS["cape"]["std"]
        return pd.Series([mean + np.random.normal(0, std) for _ in range(months)])


class CreditSpreadFetcher:
    """Fetch credit spreads from FRED with fallbacks"""
    
    def __init__(self):
        self._session = get_http_session()
        self.api_key = os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # FRED series IDs
        self.ig_series = "BAMLC0A0CM"  # Investment Grade OAS
        self.hy_series = "BAMLH0A0HYM2"  # High Yield OAS
    
    @resilient_call(
        service_name="fred",
        max_retries=3,
        base_delay=1.0,
        use_circuit_breaker=True,
        use_rate_limiter=True,
    )
    def _fetch_fred_series(self, series_id: str, limit: int = 1) -> Optional[float]:
        """Fetch a FRED series with resilience"""
        if not self.api_key:
            raise ValueError("FRED API key not configured")
        
        response = self._session.get(
            self.base_url,
            params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        obs = data.get("observations", [])
        
        if obs and obs[0].get("value") != ".":
            return float(obs[0]["value"])
        return None
    
    def get_current_spreads(self) -> Tuple[Tuple[float, float], str]:
        """
        Get current IG and HY spreads in basis points.
        
        Returns:
            Tuple of ((ig_spread, hy_spread), source)
        """
        ig, hy = None, None
        source = "live"
        
        # Try to fetch IG spread
        try:
            ig_pct = self._fetch_fred_series(self.ig_series)
            if ig_pct:
                ig = ig_pct * 100  # Convert to bps
        except Exception as e:
            logger.warning(f"FRED IG spread fetch failed: {e}")
        
        # Try to fetch HY spread
        try:
            hy_pct = self._fetch_fred_series(self.hy_series)
            if hy_pct:
                hy = hy_pct * 100  # Convert to bps
        except Exception as e:
            logger.warning(f"FRED HY spread fetch failed: {e}")
        
        # Apply fallbacks as needed
        if ig is None:
            ig = FALLBACK_VALUES["credit_ig"]
            source = "partial_fallback" if hy else "fallback"
        if hy is None:
            hy = FALLBACK_VALUES["credit_hy"]
            source = "partial_fallback" if source == "live" else "fallback"
        
        return (ig, hy), source
    
    def get_composite_spread(self) -> Tuple[float, str]:
        """Get composite credit spread (weighted IG + HY)"""
        (ig, hy), source = self.get_current_spreads()
        composite = 0.6 * ig + 0.4 * hy
        return composite, source
    
    def get_spread_history(self, months: int = 60) -> pd.Series:
        """Get credit spread history for z-score"""
        if not self.api_key:
            # Return synthetic history
            mean = HISTORICAL_STATS["credit"]["mean"]
            std = HISTORICAL_STATS["credit"]["std"]
            return pd.Series([mean + np.random.normal(0, std) for _ in range(months)])
        
        try:
            response = self._session.get(
                self.base_url,
                params={
                    "series_id": self.ig_series,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": months * 4  # Weekly data
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                values = []
                for o in data.get("observations", []):
                    if o["value"] != ".":
                        values.append(float(o["value"]) * 100)
                if values:
                    return pd.Series(values[::-1])
        except Exception as e:
            logger.warning(f"Spread history fetch failed: {e}")
        
        # Return synthetic history
        mean = HISTORICAL_STATS["credit"]["mean"]
        std = HISTORICAL_STATS["credit"]["std"]
        return pd.Series([mean + np.random.normal(0, std) for _ in range(months)])


# =============================================================================
# BUBBLE INDEX CALCULATOR (Enhanced)
# =============================================================================

class BubbleIndexCalculator:
    """
    Calculate the Y2AI Bubble Index and Bifurcation Score.
    
    Enhanced with:
    - Graceful degradation when data sources fail
    - Data quality tracking
    - Fallback to historical/typical values
    """
    
    def __init__(self):
        self.vix_fetcher = VIXFetcher()
        self.cape_fetcher = CAPEFetcher()
        self.credit_fetcher = CreditSpreadFetcher()
        
        # Cached history for z-scores
        self._vix_history: Optional[pd.Series] = None
        self._cape_history: Optional[pd.Series] = None
        self._credit_history: Optional[pd.Series] = None
    
    def _calculate_zscore(self, value: float, history: pd.Series) -> float:
        """Calculate z-score relative to rolling history"""
        if history is None or len(history) == 0:
            return 0.0
        
        mean = history.mean()
        std = history.std()
        
        if std == 0 or np.isnan(std):
            return 0.0
        
        return (value - mean) / std
    
    def _load_history(self):
        """Load historical data for z-score calculations"""
        if self._vix_history is None:
            self._vix_history = self.vix_fetcher.get_vix_history(60)
        if self._cape_history is None:
            self._cape_history = self.cape_fetcher.get_cape_history(60)
        if self._credit_history is None:
            self._credit_history = self.credit_fetcher.get_spread_history(60)
    
    def calculate_bubble_index(self, cape: float) -> float:
        """
        Calculate simplified Bubble Index (0-100 scale).
        
        Maps CAPE to 0-100 scale where:
        - CAPE 15 → Index 20 (cheap)
        - CAPE 25 → Index 50 (fair value)
        - CAPE 35 → Index 80 (expensive)
        - CAPE 45+ → Index 95+ (extreme)
        """
        bubble_index = (cape - 15) * 2.5 + 20
        bubble_index = max(0, min(100, bubble_index))
        return round(bubble_index, 1)
    
    def calculate_bifurcation_score(
        self,
        bubble_index: float,
        vix_zscore: float,
        credit_zscore: float
    ) -> float:
        """
        Calculate unified Bifurcation Score.
        
        Formula: Bifurcation = 0.6*BI_normalized - 0.2*VI - 0.2*CS
        """
        # Normalize bubble index to -1 to +1 range
        bi_normalized = (bubble_index - 50) / 50
        
        # Calculate bifurcation score
        score = 0.6 * bi_normalized - 0.2 * vix_zscore - 0.2 * credit_zscore
        
        return round(score, 3)
    
    def determine_regime(self, bifurcation_score: float, vix: float) -> str:
        """
        Determine current market regime.
        
        Regimes:
        - INFRASTRUCTURE: Strong infrastructure cycle (score > +0.5)
        - ADOPTION: Healthy adoption phase (score +0.2 to +0.5)
        - TRANSITION: Watching for regime change (score -0.2 to +0.2)
        - BUBBLE_WARNING: Elevated risk (score < -0.2)
        """
        # High VIX overrides other signals
        if vix > 30:
            return "TRANSITION"
        
        if bifurcation_score > 0.5:
            return "INFRASTRUCTURE"
        elif bifurcation_score > 0.2:
            return "ADOPTION"
        elif bifurcation_score > -0.2:
            return "TRANSITION"
        else:
            return "BUBBLE_WARNING"
    
    def calculate(self) -> BubbleIndexReading:
        """
        Calculate complete bubble index reading.
        
        This method always returns a result, using fallbacks when necessary.
        The data_sources field indicates which values are live vs fallback.
        """
        logger.info("Calculating Y2AI Bubble Index...")
        
        # Track data sources
        data_sources = {}
        
        # Load historical data for z-scores
        self._load_history()
        
        # Fetch current values with source tracking
        vix, vix_source = self.vix_fetcher.get_current_vix()
        data_sources["vix"] = vix_source
        
        cape, cape_source = self.cape_fetcher.get_current_cape()
        data_sources["cape"] = cape_source
        
        (ig, hy), credit_source = self.credit_fetcher.get_current_spreads()
        data_sources["credit"] = credit_source
        credit_composite = 0.6 * ig + 0.4 * hy
        
        logger.info(f"  VIX: {vix:.2f} ({vix_source})")
        logger.info(f"  CAPE: {cape:.2f} ({cape_source})")
        logger.info(f"  Credit (IG/HY): {ig:.0f}/{hy:.0f} bps ({credit_source})")
        
        # Calculate z-scores
        vix_z = self._calculate_zscore(vix, self._vix_history)
        cape_z = self._calculate_zscore(cape, self._cape_history)
        credit_z = self._calculate_zscore(credit_composite, self._credit_history)
        
        logger.info(f"  VIX z-score: {vix_z:.2f}")
        logger.info(f"  CAPE z-score: {cape_z:.2f}")
        logger.info(f"  Credit z-score: {credit_z:.2f}")
        
        # Calculate indices
        bubble_index = self.calculate_bubble_index(cape)
        bifurcation = self.calculate_bifurcation_score(bubble_index, vix_z, credit_z)
        regime = self.determine_regime(bifurcation, vix)
        
        logger.info(f"  Bubble Index: {bubble_index}")
        logger.info(f"  Bifurcation Score: {bifurcation}")
        logger.info(f"  Regime: {regime}")
        
        reading = BubbleIndexReading(
            date=datetime.now().strftime("%Y-%m-%d"),
            vix=round(vix, 2),
            cape=round(cape, 2),
            credit_spread_ig=round(ig, 0),
            credit_spread_hy=round(hy, 0),
            vix_zscore=round(vix_z, 2),
            cape_zscore=round(cape_z, 2),
            credit_zscore=round(credit_z, 2),
            bubble_index=bubble_index,
            bifurcation_score=bifurcation,
            regime=regime,
            data_sources=data_sources,
            calculated_at=datetime.utcnow().isoformat()
        )
        
        # Log data quality
        quality = reading.data_quality_score
        if quality < 1.0:
            logger.warning(f"  Data quality: {quality:.0%} (some fallback values used)")
        else:
            logger.info(f"  Data quality: {quality:.0%} (all live data)")
        
        return reading
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status of all data fetchers"""
        return {
            "vix": get_health_tracker("yfinance_vix").to_dict(),
            "cape": get_health_tracker("shiller_cape").to_dict(),
            "credit": get_health_tracker("fred").to_dict(),
        }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    calculator = BubbleIndexCalculator()
    reading = calculator.calculate()
    
    print(f"\n{'='*60}")
    print("Y2AI BUBBLE INDEX READING (ENHANCED)")
    print(f"{'='*60}")
    print(f"Date: {reading.date}")
    print(f"Data Quality: {reading.data_quality_score:.0%}")
    print()
    
    print("RAW VALUES:")
    print(f"  VIX:           {reading.vix} ({reading.data_sources.get('vix', 'unknown')})")
    print(f"  CAPE:          {reading.cape} ({reading.data_sources.get('cape', 'unknown')})")
    print(f"  Credit IG:     {reading.credit_spread_ig} bps")
    print(f"  Credit HY:     {reading.credit_spread_hy} bps ({reading.data_sources.get('credit', 'unknown')})")
    print()
    
    print("Z-SCORES:")
    print(f"  VIX:           {reading.vix_zscore:+.2f}")
    print(f"  CAPE:          {reading.cape_zscore:+.2f}")
    print(f"  Credit:        {reading.credit_zscore:+.2f}")
    print()
    
    print("INDICES:")
    print(f"  Bubble Index:  {reading.bubble_index} / 100")
    print(f"  Bifurcation:   {reading.bifurcation_score:+.3f}")
    print(f"  Regime:        {reading.regime}")
    print(f"{'='*60}")
