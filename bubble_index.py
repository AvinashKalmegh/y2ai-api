"""
Y2AI BUBBLE INDEX SERVICE
Replaces Google Sheets formulas for VIX, CAPE, Credit Spreads, and Bifurcation Score

Formula Reference (from your Santa Fe paper):
BI_t = 0.30*CAPE_z + 0.20*PS_z + 0.25*MCGDP_z + 0.15*DY_inv_z + 0.10*CG_z

Bifurcation Score = 0.6*BI - 0.2*VI - 0.2*CS
Where:
- BI = Bubble Index (valuation extremeness)
- VI = VIX z-score (volatility regime)  
- CS = Credit Spreads z-score (financial stress)
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # Bubble Index components
    bubble_index: float  # 0-100 scale
    
    # Unified bifurcation score
    bifurcation_score: float  # -1 to +1 typically
    
    # Regime interpretation
    regime: str  # INFRASTRUCTURE, ADOPTION, TRANSITION, BUBBLE_WARNING
    
    # Metadata
    calculated_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# DATA FETCHERS
# =============================================================================

class VIXFetcher:
    """Fetch VIX data from CBOE or Yahoo Finance"""
    
    def get_current_vix(self) -> float:
        """Get current VIX value"""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"VIX fetch error: {e}")
        
        # Fallback: return typical VIX if fetch fails
        logger.warning("Using fallback VIX value")
        return 15.0
    
    def get_vix_history(self, months: int = 60) -> pd.Series:
        """Get VIX history for z-score calculation"""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            end = datetime.now()
            start = end - timedelta(days=months * 30)
            hist = vix.history(start=start, end=end)
            return hist['Close']
        except Exception as e:
            logger.error(f"VIX history error: {e}")
            return pd.Series([15.0] * months)


class CAPEFetcher:
    """Fetch Shiller CAPE ratio"""
    
    def __init__(self):
        # Shiller data URL (updated monthly)
        self.shiller_url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    
    def get_current_cape(self) -> float:
        """Get current CAPE ratio"""
        try:
            # Try to fetch from Shiller data
            df = pd.read_excel(self.shiller_url, sheet_name='Data', skiprows=7)
            df = df.dropna(subset=[df.columns[10]])  # CAPE column
            cape = float(df.iloc[-1, 10])  # Most recent CAPE
            return cape
        except Exception as e:
            logger.warning(f"Shiller fetch error: {e}, using fallback")
        
        # Fallback: use alternative source or estimate
        try:
            # Alternative: calculate from SPY P/E
            import yfinance as yf
            spy = yf.Ticker("SPY")
            pe = spy.info.get('trailingPE', 20)
            # CAPE is typically higher than trailing P/E
            return pe * 1.4
        except:
            return 30.0  # Fallback value
    
    def get_cape_history(self, months: int = 60) -> pd.Series:
        """Get CAPE history for z-score calculation"""
        try:
            df = pd.read_excel(self.shiller_url, sheet_name='Data', skiprows=7)
            df = df.dropna(subset=[df.columns[10]])
            cape_series = df.iloc[-months:, 10].astype(float)
            return pd.Series(cape_series.values)
        except Exception as e:
            logger.error(f"CAPE history error: {e}")
            return pd.Series([25.0] * months)


class CreditSpreadFetcher:
    """Fetch credit spreads from FRED"""
    
    def __init__(self):
        self.api_key = os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # FRED series IDs
        self.ig_series = "BAMLC0A0CM"  # Investment Grade OAS
        self.hy_series = "BAMLH0A0HYM2"  # High Yield OAS
    
    def _fetch_fred_series(self, series_id: str, limit: int = 1) -> Optional[float]:
        """Fetch a FRED series"""
        if not self.api_key:
            logger.warning("FRED API key not set")
            return None
        
        try:
            response = requests.get(
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
            
            if response.status_code == 200:
                data = response.json()
                obs = data.get("observations", [])
                if obs:
                    return float(obs[0]["value"])
        except Exception as e:
            logger.error(f"FRED fetch error for {series_id}: {e}")
        
        return None
    
    def get_current_spreads(self) -> Tuple[float, float]:
        """Get current IG and HY spreads in basis points"""
        ig = self._fetch_fred_series(self.ig_series)
        hy = self._fetch_fred_series(self.hy_series)
        
        # Fallback values if fetch fails
        if ig is None:
            ig = 100.0  # Typical IG spread
        if hy is None:
            hy = 350.0  # Typical HY spread
        
        # FRED returns in percentage points, convert to basis points
        return ig * 100, hy * 100
    
    def get_composite_spread(self) -> float:
        """Get composite credit spread (weighted IG + HY)"""
        ig, hy = self.get_current_spreads()
        # Weight: 60% IG, 40% HY
        return 0.6 * ig + 0.4 * hy
    
    def get_spread_history(self, months: int = 60) -> pd.Series:
        """Get credit spread history for z-score"""
        if not self.api_key:
            return pd.Series([120.0] * months)
        
        try:
            response = requests.get(
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
                values = [float(o["value"]) * 100 for o in data.get("observations", [])]
                return pd.Series(values[::-1])  # Reverse to chronological
        except Exception as e:
            logger.error(f"Spread history error: {e}")
        
        return pd.Series([120.0] * months)


# =============================================================================
# BUBBLE INDEX CALCULATOR
# =============================================================================

class BubbleIndexCalculator:
    """
    Calculate the Y2AI Bubble Index and Bifurcation Score
    
    Replaces the Google Sheets formulas in cells B1, B23, B28
    """
    
    def __init__(self):
        self.vix_fetcher = VIXFetcher()
        self.cape_fetcher = CAPEFetcher()
        self.credit_fetcher = CreditSpreadFetcher()
        
        # Historical data for z-score calculation
        self._vix_history = None
        self._cape_history = None
        self._credit_history = None
    
    def _calculate_zscore(self, value: float, history: pd.Series) -> float:
        """Calculate z-score relative to rolling history"""
        if history is None or len(history) == 0:
            return 0.0
        
        mean = history.mean()
        std = history.std()
        
        if std == 0:
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
        Calculate simplified Bubble Index (0-100 scale)
        
        Full formula: BI = 0.30*CAPE_z + 0.20*PS_z + 0.25*MCGDP_z + 0.15*DY_z + 0.10*CG_z
        
        Simplified (using CAPE as primary proxy):
        Maps CAPE to 0-100 scale where:
        - CAPE 15 → Index 20 (cheap)
        - CAPE 25 → Index 50 (fair value)
        - CAPE 35 → Index 80 (expensive)
        - CAPE 45+ → Index 95+ (extreme)
        """
        # Linear mapping from CAPE to Bubble Index
        # Formula: BI = (CAPE - 15) * 2.5 + 20
        # Clamped to 0-100 range
        
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
        Calculate unified Bifurcation Score
        
        Formula: Bifurcation = 0.6*BI_normalized - 0.2*VI - 0.2*CS
        
        Interpretation:
        - Score > +0.5: Infrastructure cycle (healthy growth)
        - Score +0.2 to +0.5: Adoption phase
        - Score -0.2 to +0.2: Transition zone
        - Score < -0.2: Bubble warning
        
        Note: High VIX and wide credit spreads REDUCE the score (negative contribution)
        """
        # Normalize bubble index to -1 to +1 range
        # BI 0 → -1, BI 50 → 0, BI 100 → +1
        bi_normalized = (bubble_index - 50) / 50
        
        # Calculate bifurcation score
        score = 0.6 * bi_normalized - 0.2 * vix_zscore - 0.2 * credit_zscore
        
        return round(score, 2)
    
    def determine_regime(self, bifurcation_score: float, vix: float) -> str:
        """
        Determine current market regime
        
        Regimes:
        - INFRASTRUCTURE: Strong infrastructure cycle (score > +0.5)
        - ADOPTION: Healthy adoption phase (score +0.2 to +0.5)
        - TRANSITION: Watching for regime change (score -0.2 to +0.2)
        - BUBBLE_WARNING: Elevated risk (score < -0.2)
        """
        if vix > 30:
            return "TRANSITION"  # High volatility overrides
        
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
        Calculate complete bubble index reading
        
        This is the main entry point that replaces your Google Sheets calculations.
        """
        logger.info("Calculating Y2AI Bubble Index...")
        
        # Load historical data for z-scores
        self._load_history()
        
        # Fetch current values
        vix = self.vix_fetcher.get_current_vix()
        cape = self.cape_fetcher.get_current_cape()
        ig, hy = self.credit_fetcher.get_current_spreads()
        credit_composite = 0.6 * ig + 0.4 * hy
        
        logger.info(f"  VIX: {vix:.2f}")
        logger.info(f"  CAPE: {cape:.2f}")
        logger.info(f"  Credit (IG/HY): {ig:.0f}/{hy:.0f} bps")
        
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
        
        return BubbleIndexReading(
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
            calculated_at=datetime.utcnow().isoformat()
        )


# =============================================================================
# STORAGE INTEGRATION
# =============================================================================

def store_reading(reading: BubbleIndexReading):
    """Store reading in Supabase"""
    from .storage import get_storage
    
    storage = get_storage()
    if hasattr(storage, 'client') and storage.is_connected():
        storage.client.table("bubble_index_daily").upsert(
            reading.to_dict(),
            on_conflict="date"
        ).execute()
        logger.info(f"Stored reading for {reading.date}")


def get_latest_reading() -> Optional[BubbleIndexReading]:
    """Get most recent reading from Supabase"""
    from .storage import get_storage
    
    storage = get_storage()
    if hasattr(storage, 'client') and storage.is_connected():
        result = storage.client.table("bubble_index_daily")\
            .select("*")\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        if result.data:
            return BubbleIndexReading(**result.data[0])
    
    return None


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    calculator = BubbleIndexCalculator()
    reading = calculator.calculate()
    
    print(f"\n{'='*60}")
    print("Y2AI BUBBLE INDEX READING")
    print(f"{'='*60}")
    print(f"Date: {reading.date}")
    print(f"")
    print(f"RAW VALUES:")
    print(f"  VIX:           {reading.vix}")
    print(f"  CAPE:          {reading.cape}")
    print(f"  Credit IG:     {reading.credit_spread_ig} bps")
    print(f"  Credit HY:     {reading.credit_spread_hy} bps")
    print(f"")
    print(f"Z-SCORES:")
    print(f"  VIX:           {reading.vix_zscore:+.2f}")
    print(f"  CAPE:          {reading.cape_zscore:+.2f}")
    print(f"  Credit:        {reading.credit_zscore:+.2f}")
    print(f"")
    print(f"INDICES:")
    print(f"  Bubble Index:  {reading.bubble_index} / 100")
    print(f"  Bifurcation:   {reading.bifurcation_score:+.2f}")
    print(f"  Regime:        {reading.regime}")
    print(f"{'='*60}")
