"""
CREDIT SPREAD DIAL
==================
Credit spread momentum system using FRED data.
Tracks HY and IG spread changes with regime classification.

FRED Series:
- BAMLH0A0HYM2: ICE BofA US High Yield OAS
- BAMLC0A0CM: ICE BofA US Corporate OAS (Investment Grade)

Momentum Windows:
- 20-day change (primary)
- 60-day change (trend)

HY Spread Regime Thresholds (20D change in percentage points):
- Crisis: > 1.5
- Stressed: > 0.7
- Fragile: > 0.3
- Caution: > 0.1
- Healthy: <= 0.1

IG Spread Regime Thresholds (20D change):
- Crisis: > 0.5
- Stressed: > 0.25
- Fragile: > 0.1
- Caution: > 0.05
- Healthy: <= 0.05

Note: Spreads WIDENING (positive change) = Risk / Bearish
      Spreads TIGHTENING (negative change) = Bullish
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

import requests
import pandas as pd
import numpy as np

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CREDIT_CONFIG = {
    # FRED API
    "fred_api_key": os.getenv("FRED_API_KEY"),
    "fred_base_url": "https://api.stlouisfed.org/fred/series/observations",
    
    # FRED Series IDs
    "series": {
        "HY_Spread": "BAMLH0A0HYM2",   # High Yield OAS
        "IG_Spread": "BAMLC0A0CM"       # Investment Grade OAS
    },
    
    # Momentum periods
    "momentum_periods": {
        "short": 20,   # 20 trading days
        "long": 60     # 60 trading days
    },
    
    # HY Regime thresholds (20D change in percentage points)
    "hy_thresholds": {
        "crisis": 1.5,
        "stressed": 0.7,
        "fragile": 0.3,
        "caution": 0.1
    },
    
    # IG Regime thresholds (20D change)
    "ig_thresholds": {
        "crisis": 0.5,
        "stressed": 0.25,
        "fragile": 0.1,
        "caution": 0.05
    }
}

# Regime order (worst to best)
REGIME_ORDER = ["Crisis", "Stressed", "Fragile", "Caution", "Healthy"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CreditSpreadData:
    """Daily credit spread data."""
    date: str
    # Current spreads
    hy_spread: float = 0.0
    ig_spread: float = 0.0
    # 20D changes
    hy_20d_change: float = 0.0
    ig_20d_change: float = 0.0
    # 60D changes
    hy_60d_change: float = 0.0
    ig_60d_change: float = 0.0
    # Regimes
    hy_regime: str = "Unknown"
    ig_regime: str = "Unknown"
    combined_regime: str = "Unknown"
    # Interpretation
    interpretation: str = ""


# =============================================================================
# CREDIT SPREAD CALCULATOR
# =============================================================================

class CreditSpreadCalculator:
    """
    Calculate credit spread momentum and regime.
    
    Usage:
        calc = CreditSpreadCalculator()
        calc.fetch_history()
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
        
        self.config = CREDIT_CONFIG
        self.fred_key = self.config["fred_api_key"]
        
        # Cache for spread history
        self._history_df: Optional[pd.DataFrame] = None
    
    # =========================================================================
    # FRED DATA FETCHING
    # =========================================================================
    
    def fetch_fred_series(self, series_id: str, limit: int = 400) -> pd.DataFrame:
        """
        Fetch a series from FRED API.
        
        Args:
            series_id: FRED series ID
            limit: Number of observations
            
        Returns:
            DataFrame with Date and Value columns
        """
        if not self.fred_key:
            logger.warning("FRED_API_KEY not set")
            return pd.DataFrame()
        
        url = f"{self.config['fred_base_url']}"
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df[["date", "value"]].dropna()
                    df.columns = ["date", series_id]
                    return df
                    
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
        
        return pd.DataFrame()
    
    def fetch_history(self, limit: int = 400) -> pd.DataFrame:
        """
        Fetch both HY and IG spread history from FRED.
        
        Returns:
            DataFrame with Date, HY_Spread, IG_Spread columns
        """
        logger.info("Fetching credit spread history from FRED...")
        
        series = self.config["series"]
        
        # Fetch HY
        hy_df = self.fetch_fred_series(series["HY_Spread"], limit)
        
        # Fetch IG  
        ig_df = self.fetch_fred_series(series["IG_Spread"], limit)
        
        if hy_df.empty and ig_df.empty:
            # Try Supabase fallback
            return self._fetch_from_supabase()
        
        # Merge on date
        if not hy_df.empty and not ig_df.empty:
            df = pd.merge(hy_df, ig_df, on="date", how="inner")
            df.columns = ["date", "hy_spread", "ig_spread"]
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} days of spread history")
            self._history_df = df
            return df
        
        return pd.DataFrame()
    
    def _fetch_from_supabase(self) -> pd.DataFrame:
        """Fetch from Supabase as fallback."""
        if not self.supabase:
            return pd.DataFrame()
        
        try:
            response = self.supabase.table("credit_spread_history") \
                .select("date, hy_spread, ig_spread") \
                .order("date", desc=True) \
                .limit(400) \
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df["date"] = pd.to_datetime(df["date"])
                return df
                
        except Exception as e:
            logger.warning(f"Supabase fallback failed: {e}")
        
        return pd.DataFrame()
    
    # =========================================================================
    # MOMENTUM CALCULATION
    # =========================================================================
    
    def calculate_momentum(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate spread momentum and regime for each day.
        
        Args:
            df: Spread history DataFrame. If None, uses cached or fetches.
            
        Returns:
            DataFrame with momentum and regime columns
        """
        if df is None:
            df = self._history_df if self._history_df is not None else self.fetch_history()
        
        if df.empty:
            return pd.DataFrame()
        
        periods = self.config["momentum_periods"]
        
        # Calculate changes (positive = spreads widening = risk)
        df["hy_20d_change"] = df["hy_spread"] - df["hy_spread"].shift(-periods["short"])
        df["ig_20d_change"] = df["ig_spread"] - df["ig_spread"].shift(-periods["short"])
        df["hy_60d_change"] = df["hy_spread"] - df["hy_spread"].shift(-periods["long"])
        df["ig_60d_change"] = df["ig_spread"] - df["ig_spread"].shift(-periods["long"])
        
        # Calculate regimes
        df["hy_regime"] = df["hy_20d_change"].apply(self._get_hy_regime)
        df["ig_regime"] = df["ig_20d_change"].apply(self._get_ig_regime)
        df["combined_regime"] = df.apply(
            lambda row: self._get_combined_regime(row["hy_regime"], row["ig_regime"]),
            axis=1
        )
        
        return df
    
    def _get_hy_regime(self, change: float) -> str:
        """Get HY regime from 20D change."""
        if pd.isna(change):
            return "Unknown"
        
        thresholds = self.config["hy_thresholds"]
        
        if change > thresholds["crisis"]:
            return "Crisis"
        elif change > thresholds["stressed"]:
            return "Stressed"
        elif change > thresholds["fragile"]:
            return "Fragile"
        elif change > thresholds["caution"]:
            return "Caution"
        else:
            return "Healthy"
    
    def _get_ig_regime(self, change: float) -> str:
        """Get IG regime from 20D change."""
        if pd.isna(change):
            return "Unknown"
        
        thresholds = self.config["ig_thresholds"]
        
        if change > thresholds["crisis"]:
            return "Crisis"
        elif change > thresholds["stressed"]:
            return "Stressed"
        elif change > thresholds["fragile"]:
            return "Fragile"
        elif change > thresholds["caution"]:
            return "Caution"
        else:
            return "Healthy"
    
    def _get_combined_regime(self, hy_regime: str, ig_regime: str) -> str:
        """Get combined regime (worst of HY and IG)."""
        if hy_regime == "Unknown" or ig_regime == "Unknown":
            return "Unknown"
        
        hy_rank = REGIME_ORDER.index(hy_regime) if hy_regime in REGIME_ORDER else 4
        ig_rank = REGIME_ORDER.index(ig_regime) if ig_regime in REGIME_ORDER else 4
        
        return REGIME_ORDER[min(hy_rank, ig_rank)]
    
    def _get_interpretation(self, regime: str, hy_change: float, ig_change: float) -> str:
        """Get human-readable interpretation."""
        direction = "widening" if hy_change > 0 else "tightening"
        
        interpretations = {
            "Crisis": f"Credit crisis - spreads {direction} sharply. Maximum defensive posture.",
            "Stressed": f"Credit stress - spreads {direction}. Reduce risk exposure significantly.",
            "Fragile": f"Credit fragile - spreads {direction}. Monitor closely, trim positions.",
            "Caution": f"Credit caution - spreads slightly {direction}. Maintain awareness.",
            "Healthy": f"Credit healthy - spreads stable/tightening. Normal operations."
        }
        return interpretations.get(regime, "Unknown credit regime")
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self, date: str = None) -> CreditSpreadData:
        """
        Calculate credit spread data for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses latest.
            
        Returns:
            CreditSpreadData with all metrics
        """
        logger.info("Calculating credit spread data...")
        
        # Get momentum data
        df = self.calculate_momentum()
        
        if df.empty:
            return CreditSpreadData(
                date=date or datetime.now().strftime("%Y-%m-%d"),
                combined_regime="Unknown",
                interpretation="No credit spread data available"
            )
        
        # Get specific date or latest
        if date:
            row = df[df["date"] == pd.to_datetime(date)]
            if row.empty:
                row = df.iloc[[0]]  # Fall back to latest
        else:
            row = df.iloc[[0]]
        
        row = row.iloc[0]
        date_str = row["date"].strftime("%Y-%m-%d")
        
        result = CreditSpreadData(
            date=date_str,
            hy_spread=row["hy_spread"],
            ig_spread=row["ig_spread"],
            hy_20d_change=row["hy_20d_change"] if pd.notna(row["hy_20d_change"]) else 0,
            ig_20d_change=row["ig_20d_change"] if pd.notna(row["ig_20d_change"]) else 0,
            hy_60d_change=row["hy_60d_change"] if pd.notna(row["hy_60d_change"]) else 0,
            ig_60d_change=row["ig_60d_change"] if pd.notna(row["ig_60d_change"]) else 0,
            hy_regime=row["hy_regime"],
            ig_regime=row["ig_regime"],
            combined_regime=row["combined_regime"],
            interpretation=self._get_interpretation(
                row["combined_regime"],
                row["hy_20d_change"] if pd.notna(row["hy_20d_change"]) else 0,
                row["ig_20d_change"] if pd.notna(row["ig_20d_change"]) else 0
            )
        )
        
        logger.info(f"Credit: HY={result.hy_spread:.2f} ({result.hy_regime}), IG={result.ig_spread:.2f} ({result.ig_regime})")
        
        return result
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_history_to_supabase(self, df: pd.DataFrame = None) -> int:
        """Save spread history to Supabase."""
        if not self.supabase:
            return 0
        
        if df is None:
            df = self._history_df
        
        if df is None or df.empty:
            return 0
        
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "hy_spread": row["hy_spread"],
                "ig_spread": row["ig_spread"]
            })
        
        try:
            self.supabase.table("credit_spread_history") \
                .upsert(rows, on_conflict="date") \
                .execute()
            return len(rows)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return 0
    
    def save_to_supabase(self, data: CreditSpreadData) -> bool:
        """Save credit spread data to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("credit_spread_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved credit spread for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def get_history(self, days: int = 30) -> List[CreditSpreadData]:
        """Get credit spread history."""
        df = self.calculate_momentum()
        
        if df.empty:
            return []
        
        results = []
        for _, row in df.head(days).iterrows():
            results.append(CreditSpreadData(
                date=row["date"].strftime("%Y-%m-%d"),
                hy_spread=row["hy_spread"],
                ig_spread=row["ig_spread"],
                hy_20d_change=row["hy_20d_change"] if pd.notna(row["hy_20d_change"]) else 0,
                ig_20d_change=row["ig_20d_change"] if pd.notna(row["ig_20d_change"]) else 0,
                hy_regime=row["hy_regime"],
                ig_regime=row["ig_regime"],
                combined_regime=row["combined_regime"]
            ))
        
        return results


# =============================================================================
# SUPABASE TABLES
# =============================================================================

CREATE_TABLES_SQL = """
-- Credit spread history (raw FRED data)
CREATE TABLE IF NOT EXISTS credit_spread_history (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    hy_spread FLOAT,
    ig_spread FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_credit_history_date ON credit_spread_history(date DESC);

-- Credit spread daily (calculated metrics)
CREATE TABLE IF NOT EXISTS credit_spread_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    hy_spread FLOAT,
    ig_spread FLOAT,
    hy_20d_change FLOAT,
    ig_20d_change FLOAT,
    hy_60d_change FLOAT,
    ig_60d_change FLOAT,
    hy_regime VARCHAR(20),
    ig_regime VARCHAR(20),
    combined_regime VARCHAR(20),
    interpretation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_credit_daily_date ON credit_spread_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run credit spread calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Credit Spread Dial")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from FRED")
    parser.add_argument("--history", type=int, default=0, help="Show N days of history")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = CreditSpreadCalculator()
    
    print(f"\n{'='*60}")
    print("CREDIT SPREAD DIAL")
    print(f"{'='*60}\n")
    
    if args.fetch:
        df = calc.fetch_history()
        print(f"Fetched {len(df)} days of spread history")
        
        if args.save and not df.empty:
            saved = calc.save_history_to_supabase(df)
            print(f"Saved {saved} rows to Supabase")
    
    if args.history > 0:
        history = calc.get_history(args.history)
        print(f"Last {len(history)} days:\n")
        for cs in history[:10]:
            print(f"  {cs.date}: HY={cs.hy_spread:.2f} ({cs.hy_regime}), IG={cs.ig_spread:.2f} ({cs.ig_regime})")
    else:
        data = calc.calculate()
        
        print(f"Date: {data.date}")
        print(f"\n{'='*40}")
        print("CURRENT SPREADS")
        print(f"{'='*40}")
        print(f"  High Yield (HY):       {data.hy_spread:.2f}%")
        print(f"  Investment Grade (IG): {data.ig_spread:.2f}%")
        
        print(f"\n{'='*40}")
        print("20-DAY MOMENTUM")
        print(f"{'='*40}")
        print(f"  HY Change: {data.hy_20d_change:+.2f}pp ({data.hy_regime})")
        print(f"  IG Change: {data.ig_20d_change:+.2f}pp ({data.ig_regime})")
        
        print(f"\n{'='*40}")
        print("60-DAY MOMENTUM")
        print(f"{'='*40}")
        print(f"  HY Change: {data.hy_60d_change:+.2f}pp")
        print(f"  IG Change: {data.ig_60d_change:+.2f}pp")
        
        print(f"\n{'='*40}")
        print(f"COMBINED REGIME: {data.combined_regime}")
        print(f"{'='*40}")
        print(f"\n{data.interpretation}")
        
        if args.save:
            if calc.save_to_supabase(data):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
