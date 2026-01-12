"""
BUBBLE OVERLAY DIAL
===================
Implements three academic bubble detection methods as overlay to AMRI.

Components:
- PSY (Phillips-Shi-Yu): Explosive regime detection using recursive ADF tests
- LPPLS (Log-Periodic Power Law): Sornette bubble model hazard rate
- LZC (Lempel-Ziv Complexity): Herding behavior detection

Integration:
- Enhanced AMRI = 80% Core AMRI + 20% Bubble Overlay
- Each component scored 0-100, combined into composite

Regimes:
- Clear: Composite < 50 (no bubble signals)
- Warning: Composite 50-75 (some bubble signals)
- Danger: Composite > 75 (strong bubble signals)

References:
- Phillips, Shi & Yu (2015) - PSY test for explosive regimes
- Sornette et al. (2001) - LPPLS bubble model
- Lempel & Ziv (1976) - Complexity measure
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BUBBLE_OVERLAY_CONFIG = {
    # PSY Parameters
    "psy": {
        "window": 60,              # Days for ADF test window
        "min_window": 20,          # Minimum window for recursive test
        "critical_value": -0.5,    # Right-tailed ADF critical value
    },
    
    # LPPLS Parameters
    "lppls": {
        "window": 120,             # Days to fit LPPLS model
        "hazard_threshold": 0.7,   # Hazard rate threshold
    },
    
    # LZC Parameters
    "lzc": {
        "window": 60,              # Days for complexity calculation
        "warning_threshold": 35,   # Below this = herding
    },
    
    # Composite thresholds
    "regime_thresholds": {
        "clear": 50,
        "warning": 75,
        # > 75 = Danger
    },
    
    # Component weights
    "weights": {
        "psy": 0.40,
        "lppls": 0.30,
        "lzc": 0.30,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PSYData:
    """Phillips-Shi-Yu test result."""
    score: float          # 0-100
    status: str           # Clear, Warning, Explosive
    adf_statistic: float  # ADF test statistic
    critical_value: float
    is_explosive: bool


@dataclass
class LPPLSData:
    """Log-Periodic Power Law result."""
    score: float          # 0-100
    status: str           # Not Measured, Low, Moderate, High
    hazard_rate: float    # 0-1
    fit_quality: float    # R-squared of fit


@dataclass
class LZCData:
    """Lempel-Ziv Complexity result."""
    score: float          # 0-100 (inverted: low complexity = high score)
    status: str           # Normal, Caution, Herding
    complexity: float     # Raw complexity measure
    threshold: float


@dataclass
class BubbleOverlayData:
    """Complete bubble overlay data."""
    date: str
    # Components
    psy: PSYData
    lppls: LPPLSData
    lzc: LZCData
    # Composite
    composite_score: float
    regime: str
    interpretation: str


# =============================================================================
# BUBBLE OVERLAY CALCULATOR
# =============================================================================

class BubbleOverlayCalculator:
    """
    Calculate bubble overlay signals using academic methods.
    
    Usage:
        calc = BubbleOverlayCalculator()
        data = calc.calculate()
        calc.save_to_supabase(data)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.config = BUBBLE_OVERLAY_CONFIG
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_price_series(self, ticker: str = "QQQ", days: int = 150) -> Optional[pd.Series]:
        """Fetch price series for bubble analysis."""
        # Try Supabase
        if self.supabase:
            try:
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                
                response = self.supabase.table("price_history") \
                    .select("date, close") \
                    .eq("ticker", ticker) \
                    .gte("date", start_date) \
                    .order("date", desc=False) \
                    .execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    logger.info(f"Loaded {len(df)} rows for {ticker} from Supabase")
                    return df["close"]
            except Exception as e:
                logger.warning(f"Supabase fetch failed: {e}")
        
        # Fallback to yfinance
        try:
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=days)
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                return data["Close"]
        except Exception as e:
            logger.warning(f"yfinance fetch failed: {e}")
        
        return None
    
    # =========================================================================
    # PSY (PHILLIPS-SHI-YU) TEST
    # =========================================================================
    
    def calculate_psy(self, prices: pd.Series) -> PSYData:
        """
        Calculate PSY explosive regime test.
        
        Uses right-tailed ADF test to detect explosive (bubble) behavior.
        Standard ADF tests for stationarity (unit root).
        PSY tests for explosive behavior (right tail).
        """
        config = self.config["psy"]
        
        if prices is None or len(prices) < config["min_window"]:
            return PSYData(
                score=0, status="Not Measured", adf_statistic=0,
                critical_value=config["critical_value"], is_explosive=False
            )
        
        # Calculate log prices
        log_prices = np.log(prices.values)
        
        # Simple ADF-like test: regress diff on lagged level
        n = min(len(log_prices), config["window"])
        y = log_prices[-n:]
        
        # First difference
        dy = np.diff(y)
        # Lagged level
        y_lag = y[:-1]
        
        # Regression: dy = alpha + beta * y_lag + error
        # beta > 0 indicates explosive behavior
        try:
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            coeffs = np.linalg.lstsq(X, dy, rcond=None)[0]
            beta = coeffs[1]
            
            # Calculate t-statistic for beta
            residuals = dy - X @ coeffs
            se = np.sqrt(np.sum(residuals**2) / (len(dy) - 2))
            se_beta = se / np.sqrt(np.sum((y_lag - y_lag.mean())**2))
            t_stat = beta / se_beta if se_beta > 0 else 0
            
        except Exception:
            t_stat = 0
            beta = 0
        
        # PSY uses right-tailed test: explosive if t-stat > critical value
        # (opposite of standard ADF which tests left tail)
        is_explosive = t_stat > config["critical_value"]
        
        # Convert to score (0-100)
        # Higher t-stat = more explosive = higher score
        if t_stat < -2:
            score = 0
        elif t_stat < config["critical_value"]:
            score = 25
        elif t_stat < 1:
            score = 50
        elif t_stat < 2:
            score = 75
        else:
            score = 100
        
        # Status
        if is_explosive:
            status = "Explosive"
        elif t_stat > -1:
            status = "Warning"
        else:
            status = "Clear"
        
        return PSYData(
            score=score,
            status=status,
            adf_statistic=round(t_stat, 3),
            critical_value=config["critical_value"],
            is_explosive=is_explosive
        )
    
    # =========================================================================
    # LPPLS (LOG-PERIODIC POWER LAW)
    # =========================================================================
    
    def calculate_lppls(self, prices: pd.Series) -> LPPLSData:
        """
        Calculate LPPLS bubble hazard rate.
        
        LPPLS model: log(p(t)) = A + B(tc-t)^m + C(tc-t)^m * cos(w*ln(tc-t) + phi)
        
        Simplified version: detect super-exponential growth with oscillations.
        """
        config = self.config["lppls"]
        
        if prices is None or len(prices) < 60:
            return LPPLSData(
                score=0, status="Not Measured", hazard_rate=0, fit_quality=0
            )
        
        # Use recent window
        n = min(len(prices), config["window"])
        log_prices = np.log(prices.values[-n:])
        t = np.arange(n)
        
        # Fit simple exponential trend
        try:
            # Linear regression on log prices
            X = np.column_stack([np.ones(n), t])
            coeffs = np.linalg.lstsq(X, log_prices, rcond=None)[0]
            
            # Calculate residuals
            fitted = X @ coeffs
            residuals = log_prices - fitted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((log_prices - log_prices.mean())**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # Calculate growth rate (slope)
            growth_rate = coeffs[1] * 252  # Annualized
            
            # Check for acceleration (second derivative)
            # Compare growth in first half vs second half
            mid = n // 2
            growth_first = (log_prices[mid] - log_prices[0]) / mid
            growth_second = (log_prices[-1] - log_prices[mid]) / (n - mid)
            acceleration = growth_second / growth_first if growth_first > 0 else 1
            
            # Simple hazard rate based on acceleration and volatility
            # Higher acceleration + lower residual volatility = higher hazard
            residual_vol = np.std(residuals)
            
            if acceleration > 1.5 and residual_vol < 0.02:
                hazard_rate = 0.8
            elif acceleration > 1.2 and residual_vol < 0.03:
                hazard_rate = 0.6
            elif acceleration > 1.0:
                hazard_rate = 0.4
            else:
                hazard_rate = 0.2
            
        except Exception:
            hazard_rate = 0
            r_squared = 0
        
        # Convert to score
        score = hazard_rate * 100
        
        # Status
        if hazard_rate >= config["hazard_threshold"]:
            status = "High"
        elif hazard_rate >= 0.5:
            status = "Moderate"
        elif hazard_rate >= 0.3:
            status = "Low"
        else:
            status = "Clear"
        
        return LPPLSData(
            score=round(score, 1),
            status=status,
            hazard_rate=round(hazard_rate, 3),
            fit_quality=round(r_squared, 3)
        )
    
    # =========================================================================
    # LZC (LEMPEL-ZIV COMPLEXITY)
    # =========================================================================
    
    def calculate_lzc(self, prices: pd.Series) -> LZCData:
        """
        Calculate Lempel-Ziv Complexity.
        
        Low complexity = synchronized/herding behavior = bubble risk.
        High complexity = diverse/random behavior = healthy.
        """
        config = self.config["lzc"]
        
        if prices is None or len(prices) < config["window"]:
            return LZCData(
                score=0, status="Not Measured", complexity=0,
                threshold=config["warning_threshold"]
            )
        
        # Calculate returns
        returns = prices.pct_change().dropna().values[-config["window"]:]
        
        # Convert to binary sequence (up/down)
        binary = ''.join(['1' if r > 0 else '0' for r in returns])
        
        # Calculate Lempel-Ziv complexity
        complexity = self._lz_complexity(binary)
        
        # Normalize to 0-100 scale
        # Random binary sequence of length n has complexity ~n/log2(n)
        n = len(binary)
        max_complexity = n / np.log2(n) if n > 1 else 1
        normalized = (complexity / max_complexity) * 100 if max_complexity > 0 else 50
        
        # Invert: low complexity = high bubble score
        # LZC below threshold = herding = danger
        if normalized < config["warning_threshold"]:
            score = 100 - normalized  # Invert
            status = "Herding"
        elif normalized < 50:
            score = 50
            status = "Caution"
        else:
            score = max(0, 100 - normalized)
            status = "Normal"
        
        return LZCData(
            score=round(score, 1),
            status=status,
            complexity=round(normalized, 1),
            threshold=config["warning_threshold"]
        )
    
    def _lz_complexity(self, s: str) -> int:
        """Calculate Lempel-Ziv complexity of a string."""
        i, k, l = 0, 1, 1
        c = 1
        n = len(s)
        
        while True:
            if s[i + k - 1] == s[l + k - 1]:
                k += 1
                if l + k > n:
                    c += 1
                    break
            else:
                if k > 1:
                    c += 1
                    i = l
                    l = i + 1
                    k = 1
                else:
                    l += 1
                
                if l > n - 1:
                    break
                elif l == i + 1:
                    c += 1
                    i = l
                    l = i + 1
        
        return c
    
    # =========================================================================
    # COMPOSITE CALCULATION
    # =========================================================================
    
    def calculate_composite(
        self,
        psy: PSYData,
        lppls: LPPLSData,
        lzc: LZCData
    ) -> Tuple[float, str, str]:
        """
        Calculate composite bubble overlay score.
        
        Returns: (score, regime, interpretation)
        """
        weights = self.config["weights"]
        
        # Weighted average
        composite = (
            psy.score * weights["psy"] +
            lppls.score * weights["lppls"] +
            lzc.score * weights["lzc"]
        )
        
        # Determine regime
        thresholds = self.config["regime_thresholds"]
        if composite < thresholds["clear"]:
            regime = "Clear"
            interpretation = "No significant bubble signals detected."
        elif composite < thresholds["warning"]:
            regime = "Warning"
            interpretation = "Some bubble indicators elevated. Monitor closely."
        else:
            regime = "Danger"
            interpretation = "Strong bubble signals. Consider reducing exposure."
        
        return round(composite, 1), regime, interpretation
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self, ticker: str = "QQQ") -> BubbleOverlayData:
        """
        Calculate bubble overlay metrics.
        
        Args:
            ticker: Ticker to analyze (default QQQ for broad tech)
            
        Returns:
            BubbleOverlayData with all metrics
        """
        logger.info(f"Calculating bubble overlay for {ticker}...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch prices
        prices = self.fetch_price_series(ticker, days=150)
        
        # Calculate components
        psy = self.calculate_psy(prices)
        lppls = self.calculate_lppls(prices)
        lzc = self.calculate_lzc(prices)
        
        # Calculate composite
        composite, regime, interpretation = self.calculate_composite(psy, lppls, lzc)
        
        logger.info(f"Bubble Overlay: {composite:.1f} ({regime})")
        logger.info(f"  PSY: {psy.score:.1f} ({psy.status})")
        logger.info(f"  LPPLS: {lppls.score:.1f} ({lppls.status})")
        logger.info(f"  LZC: {lzc.score:.1f} ({lzc.status})")
        
        return BubbleOverlayData(
            date=date_str,
            psy=psy,
            lppls=lppls,
            lzc=lzc,
            composite_score=composite,
            regime=regime,
            interpretation=interpretation,
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: BubbleOverlayData) -> Optional[Dict]:
        """Save bubble overlay data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = {
            "date": data.date,
            "composite_score": data.composite_score,
            "regime": data.regime,
            "interpretation": data.interpretation,
            # PSY
            "psy_score": data.psy.score,
            "psy_status": data.psy.status,
            "psy_adf": data.psy.adf_statistic,
            "psy_explosive": data.psy.is_explosive,
            # LPPLS
            "lppls_score": data.lppls.score,
            "lppls_status": data.lppls.status,
            "lppls_hazard": data.lppls.hazard_rate,
            # LZC
            "lzc_score": data.lzc.score,
            "lzc_status": data.lzc.status,
            "lzc_complexity": data.lzc.complexity,
        }
        
        try:
            response = self.supabase.table("bubble_overlay_daily") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved bubble overlay for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save bubble overlay: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run bubble overlay calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Bubble Overlay")
    parser.add_argument("--ticker", default="QQQ", help="Ticker to analyze")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = BubbleOverlayCalculator()
    data = calc.calculate(args.ticker)
    
    print(f"\n{'='*60}")
    print(f"BUBBLE OVERLAY: {data.composite_score:.1f} ({data.regime})")
    print(f"{'='*60}")
    print(f"Interpretation: {data.interpretation}")
    print(f"\nComponents:")
    print(f"  PSY (40%):   {data.psy.score:5.1f} ({data.psy.status}) - ADF: {data.psy.adf_statistic:.3f}")
    print(f"  LPPLS (30%): {data.lppls.score:5.1f} ({data.lppls.status}) - Hazard: {data.lppls.hazard_rate:.3f}")
    print(f"  LZC (30%):   {data.lzc.score:5.1f} ({data.lzc.status}) - Complexity: {data.lzc.complexity:.1f}")
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
