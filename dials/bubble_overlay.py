"""
BUBBLE OVERLAY
==============
Implements three academic bubble detection methods matching Google Sheets:
- PSY (Phillips-Shi-Yu): Explosive regime detection via recursive ADF
- LPPLS (Log-Periodic Power Law): Super-exponential growth detection
- LZC (Lempel-Ziv Complexity): Herding behavior detection

Based on:
- Phillips, Shi & Yu (2015) - PSY Test
- Sornette et al. (2001) - LPPLS
- Lempel & Ziv (1976) - LZC
"""

import os
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION (matches GS BUBBLE_OVERLAY_CONFIG)
# =============================================================================

BUBBLE_CONFIG = {
    # PSY Parameters
    "psy_window": 60,           # Days for ADF test window
    "psy_min_window": 20,       # Minimum window for recursive test
    "psy_critical_value": -0.5, # Right-tailed ADF critical value
    
    # LPPLS Parameters
    "lppls_window": 120,        # Days to fit LPPLS model
    "lppls_hazard_threshold": 0.7,
    
    # LZC Parameters
    "lzc_window": 60,           # Days for complexity calculation
    "lzc_warning_threshold": 35,
    
    # Composite thresholds
    "composite_warning": 50,
    "composite_danger": 75
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PSYResult:
    score: float
    status: str
    detail: str
    max_adf: float = 0.0

@dataclass
class LZCResult:
    score: float
    status: str
    detail: str
    raw_complexity: float = 0.0

@dataclass
class LPPLSResult:
    score: float
    status: str
    detail: str
    acceleration: float = 0.0
    vol_ratio: float = 0.0
    growth: float = 0.0

@dataclass
class BubbleOverlayData:
    date: str
    psy_score: float
    psy_status: str
    lppls_score: float
    lppls_status: str
    lzc_score: float
    lzc_status: str
    composite: float
    regime: str
    psy_detail: str = ""
    lppls_detail: str = ""
    lzc_detail: str = ""
    psy_adf: float = 0.0
    lppls_hazard: float = 0.0
    lzc_complexity: float = 0.0


# =============================================================================
# PSY (PHILLIPS-SHI-YU) IMPLEMENTATION
# =============================================================================

def calculate_adf(prices: list) -> Optional[float]:
    """
    Calculate Augmented Dickey-Fuller test statistic.
    Tests for unit root (non-stationarity).
    Right-tailed version detects explosive behavior.
    """
    n = len(prices)
    if n < 10:
        return None
    
    # Calculate returns (first differences)
    returns = [prices[i] - prices[i-1] for i in range(1, n)]
    
    # Lagged prices
    lagged_prices = prices[:-1]
    
    n_ret = len(returns)
    
    # Calculate means
    mean_x = sum(lagged_prices) / n_ret
    mean_y = sum(returns) / n_ret
    
    # Calculate beta (slope) via OLS
    numerator = sum((lagged_prices[i] - mean_x) * (returns[i] - mean_y) for i in range(n_ret))
    denominator = sum((lagged_prices[i] - mean_x) ** 2 for i in range(n_ret))
    
    if denominator == 0:
        return None
    
    beta = numerator / denominator
    alpha = mean_y - beta * mean_x
    
    # Calculate residuals and standard error
    sse = sum((returns[i] - (alpha + beta * lagged_prices[i])) ** 2 for i in range(n_ret))
    mse = sse / (n_ret - 2) if n_ret > 2 else sse
    se_beta = math.sqrt(mse / denominator) if denominator > 0 else 0
    
    if se_beta == 0:
        return None
    
    # ADF statistic
    adf_stat = beta / se_beta
    return adf_stat


def calculate_psy(prices: list) -> PSYResult:
    """
    PSY Test - Recursive right-tailed ADF.
    Returns score 0-100 indicating explosive behavior likelihood.
    """
    config = BUBBLE_CONFIG
    
    if not prices or len(prices) < config["psy_window"]:
        return PSYResult(0, "Not Measured", "Insufficient data")
    
    # Use last psy_window prices, reversed to oldest-first
    price_window = prices[:config["psy_window"]]
    price_window = [p for p in price_window if p and p > 0]
    price_window = price_window[::-1]  # Oldest to newest
    
    if len(price_window) < config["psy_min_window"]:
        return PSYResult(0, "Not Measured", "Insufficient valid prices")
    
    # Recursive ADF tests
    explosive_count = 0
    total_tests = 0
    max_adf = -999
    
    for end_idx in range(config["psy_min_window"], len(price_window) + 1):
        window_prices = price_window[:end_idx]
        adf_stat = calculate_adf(window_prices)
        
        if adf_stat is not None:
            total_tests += 1
            if adf_stat > config["psy_critical_value"]:
                explosive_count += 1
            if adf_stat > max_adf:
                max_adf = adf_stat
    
    if total_tests == 0:
        return PSYResult(0, "Not Measured", "ADF calculation failed")
    
    # Score based on proportion of explosive windows
    explosive_ratio = explosive_count / total_tests
    score = min(100, explosive_ratio * 100)
    
    # Determine status
    status = "Clear"
    if score > 50:
        status = "DANGER"
    elif score > 25:
        status = "WARNING"
    
    return PSYResult(
        score=round(score, 1),
        status=status,
        detail=f"{explosive_count}/{total_tests} windows explosive, max ADF: {max_adf:.2f}",
        max_adf=max_adf
    )


# =============================================================================
# LZC (LEMPEL-ZIV COMPLEXITY) IMPLEMENTATION
# =============================================================================

def calculate_lz_complexity(sequence: list) -> float:
    """
    Lempel-Ziv Complexity.
    Measures the complexity/randomness of a binary sequence.
    Lower values = more predictable = potential herding.
    """
    if not sequence or len(sequence) == 0:
        return 0
    
    n = len(sequence)
    complexity = 1
    k = 1
    l = 1
    k_max = 1
    
    while k + l <= n:
        # Check if substring exists in previous part
        substring = ''.join(str(s) for s in sequence[k:k+l])
        search_space = ''.join(str(s) for s in sequence[:k+l-1])
        
        if substring in search_space:
            l += 1
        else:
            if l > k_max:
                k_max = l
            complexity += 1
            k += l
            l = 1
    
    # Normalize by theoretical maximum complexity
    max_complexity = n / math.log2(n) if n > 1 else 1
    normalized_complexity = (complexity / max_complexity) * 100
    
    return normalized_complexity


def calculate_lzc(returns: list) -> LZCResult:
    """
    LZC Score for market returns.
    Converts returns to binary sequence and calculates complexity.
    """
    config = BUBBLE_CONFIG
    
    if not returns or len(returns) < config["lzc_window"]:
        return LZCResult(0, "Not Measured", "Insufficient data")
    
    # Filter valid returns
    valid_returns = [r for r in returns[:config["lzc_window"]] if r is not None and not math.isnan(r)]
    
    if len(valid_returns) < 20:
        return LZCResult(0, "Not Measured", "Insufficient return data")
    
    # Convert to binary: 1 if positive, 0 if negative
    binary_sequence = [1 if r >= 0 else 0 for r in valid_returns]
    
    # Calculate complexity
    complexity = calculate_lz_complexity(binary_sequence)
    
    # For LZC, LOWER is more concerning (herding)
    # Invert for scoring: high score = concerning
    inverted_score = max(0, 100 - complexity)
    
    # Determine status
    status = "Clear"
    if complexity < config["lzc_warning_threshold"]:
        status = "WARNING"
    if complexity < 25:
        status = "DANGER"
    
    return LZCResult(
        score=round(inverted_score, 1),
        status=status,
        detail=f"Raw complexity: {complexity:.1f}%, {len(valid_returns)} days analyzed",
        raw_complexity=complexity
    )


# =============================================================================
# LPPLS (LOG-PERIODIC POWER LAW) IMPLEMENTATION
# =============================================================================

def calculate_lppls(prices: list) -> LPPLSResult:
    """
    Simplified LPPLS Hazard Indicator.
    Full LPPLS requires nonlinear optimization; this is a proxy
    based on super-exponential growth detection.
    """
    config = BUBBLE_CONFIG
    
    if not prices or len(prices) < config["lppls_window"]:
        return LPPLSResult(0, "Not Measured", "Insufficient data")
    
    # Get valid prices, reversed to oldest-first
    price_window = [p for p in prices[:config["lppls_window"]] if p and p > 0]
    price_window = price_window[::-1]  # Oldest to newest
    
    if len(price_window) < 60:
        return LPPLSResult(0, "Not Measured", "Insufficient price data")
    
    # Calculate log prices
    log_prices = [math.log(p) for p in price_window]
    n = len(log_prices)
    
    # Linear regression on log prices
    sum_x = sum(range(n))
    sum_y = sum(log_prices)
    sum_xy = sum(i * log_prices[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate residuals
    residuals = [log_prices[i] - (intercept + slope * i) for i in range(n)]
    
    # Check if residuals are accelerating (convex pattern = super-exponential)
    first_half = residuals[:n//2]
    second_half = residuals[n//2:]
    
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    
    # Acceleration metric
    acceleration = avg_second - avg_first
    
    # Check for increasing volatility
    first_vol = math.sqrt(sum(r*r for r in first_half) / len(first_half))
    second_vol = math.sqrt(sum(r*r for r in second_half) / len(second_half))
    vol_ratio = second_vol / (first_vol + 0.001)
    
    # Combine into hazard score
    hazard_score = 0
    
    if acceleration > 0:
        hazard_score += min(50, acceleration * 500)
    
    if vol_ratio > 1.2:
        hazard_score += min(30, (vol_ratio - 1) * 30)
    
    # Growth rate component
    annualized_growth = slope * 252
    if annualized_growth > 0.5:
        hazard_score += min(20, (annualized_growth - 0.5) * 40)
    
    hazard_score = min(100, max(0, hazard_score))
    
    # Determine status
    status = "Clear"
    if hazard_score >= 70:
        status = "DANGER"
    elif hazard_score >= 40:
        status = "WARNING"
    
    return LPPLSResult(
        score=round(hazard_score, 1),
        status=status,
        detail=f"Accel: {acceleration:.3f}, VolRatio: {vol_ratio:.2f}, Growth: {annualized_growth*100:.1f}%",
        acceleration=acceleration,
        vol_ratio=vol_ratio,
        growth=annualized_growth
    )


# =============================================================================
# BUBBLE OVERLAY CALCULATOR
# =============================================================================

class BubbleOverlayCalculator:
    """
    Calculate bubble overlay metrics using PSY, LPPLS, and LZC.
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    def fetch_pillar_index(self, days: int = 150) -> pd.DataFrame:
        """Fetch pillar index data for calculations."""
        if not self.supabase:
            return pd.DataFrame()
        
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            response = self.supabase.table("pillar_index_daily") \
                .select("date, infra_index, infra_return") \
                .gte("date", start_date) \
                .order("date", desc=True) \
                .limit(days) \
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df["date"] = pd.to_datetime(df["date"])
                logger.info(f"Loaded {len(df)} rows from pillar_index_daily")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch pillar_index: {e}")
        
        return pd.DataFrame()
    
    def calculate(self) -> BubbleOverlayData:
        """Calculate all bubble overlay metrics."""
        logger.info("Calculating bubble overlay...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch data
        df = self.fetch_pillar_index()
        
        if df.empty:
            logger.warning("No pillar index data available")
            return BubbleOverlayData(
                date=date_str,
                psy_score=0, psy_status="Not Measured",
                lppls_score=0, lppls_status="Not Measured",
                lzc_score=0, lzc_status="Not Measured",
                composite=0, regime="Unknown"
            )
        
        # Extract prices and returns (data is newest-first)
        prices = df["infra_index"].tolist()
        returns = df["infra_return"].tolist()
        
        # Calculate PSY
        psy_result = calculate_psy(prices)
        logger.info(f"PSY: {psy_result.score} ({psy_result.status})")
        
        # Calculate LZC
        lzc_result = calculate_lzc(returns)
        logger.info(f"LZC: {lzc_result.score} ({lzc_result.status})")
        
        # Calculate LPPLS
        lppls_result = calculate_lppls(prices)
        logger.info(f"LPPLS: {lppls_result.score} ({lppls_result.status})")
        
        # Calculate composite (average of three scores)
        composite = (psy_result.score + lzc_result.score + lppls_result.score) / 3
        
        # Determine regime
        if composite > BUBBLE_CONFIG["composite_danger"]:
            regime = "DANGER"
        elif composite > BUBBLE_CONFIG["composite_warning"]:
            regime = "WARNING"
        else:
            regime = "Clear"
        
        return BubbleOverlayData(
            date=date_str,
            psy_score=psy_result.score,
            psy_status=psy_result.status,
            lppls_score=lppls_result.score,
            lppls_status=lppls_result.status,
            lzc_score=lzc_result.score,
            lzc_status=lzc_result.status,
            composite=round(composite, 1),
            regime=regime,
            psy_detail=psy_result.detail,
            lppls_detail=lppls_result.detail,
            lzc_detail=lzc_result.detail,
            psy_adf=psy_result.max_adf,
            lppls_hazard=lppls_result.score,
            lzc_complexity=lzc_result.raw_complexity
        )
    
    def save_to_supabase(self, data: BubbleOverlayData) -> bool:
        """Save bubble overlay data to Supabase."""
        if not self.supabase:
            return False
        
        row = {
            "date": data.date,
            "psy_score": data.psy_score,
            "psy_status": data.psy_status,
            "psy_adf": data.psy_adf,
            "psy_explosive": data.psy_score > 25,
            "lppls_score": data.lppls_score,
            "lppls_status": data.lppls_status,
            "lppls_hazard": data.lppls_hazard,
            "lzc_score": data.lzc_score,
            "lzc_status": data.lzc_status,
            "lzc_complexity": data.lzc_complexity,
            "composite_score": data.composite,
            "regime": data.regime,
            "interpretation": f"PSY: {data.psy_detail}, LPPLS: {data.lppls_detail}, LZC: {data.lzc_detail}"
        }
        
        try:
            self.supabase.table("bubble_overlay_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bubble Overlay")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = BubbleOverlayCalculator()
    
    print(f"\n{'='*60}")
    print("BUBBLE OVERLAY (PSY + LPPLS + LZC)")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"PSY Score:   {data.psy_score:>6.1f}  ({data.psy_status})")
    print(f"LPPLS Score: {data.lppls_score:>6.1f}  ({data.lppls_status})")
    print(f"LZC Score:   {data.lzc_score:>6.1f}  ({data.lzc_status})")
    print(f"{'='*40}")
    print(f"COMPOSITE:   {data.composite:>6.1f}  ({data.regime})")
    print(f"{'='*40}")
    
    print(f"\nPSY Detail: {data.psy_detail}")
    print(f"LPPLS Detail: {data.lppls_detail}")
    print(f"LZC Detail: {data.lzc_detail}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")
        else:
            print(f"\n❌ Failed to save")


if __name__ == "__main__":
    main()