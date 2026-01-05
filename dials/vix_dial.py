"""
VIX REGIME DIAL
===============
Full volatility analysis with Bollinger Bands and Pre-Shock detection.

Detects:
- Volatility compression (calm before storm) → Pre-Shock
- Volatility expansion (risk-off)
- Volatility spikes (crisis)
- Volatility collapse (post-panic relief)

Components:
1. Level Regime - Direct VIX reading
2. Trend Regime - 20-day slope
3. BB Regime - Bollinger Band position (Compression/Normal/Expansion)
4. Combined Regime - Final classification

Level Thresholds:
- Crisis: VIX > 40
- Fragile: VIX > 30
- Caution: VIX > 20
- Healthy: VIX <= 20

Trend Thresholds (20D change):
- Crisis: > +10
- Fragile: > +5
- Caution: > +2
- Healthy: <= +2

BB Position:
- Expansion: VIX > Upper BB (2σ)
- Compression: VIX < Lower BB (2σ) → Pre-Shock warning!
- Normal: Between bands

FRED Series: VIXCLS
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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

VIX_CONFIG = {
    # FRED API
    "fred_api_key": os.getenv("FRED_API_KEY"),
    "fred_series": "VIXCLS",
    
    # Bollinger Band parameters
    "bb_period": 20,
    "bb_std": 2,
    
    # Level thresholds
    "level_thresholds": {
        "crisis": 40,
        "fragile": 30,
        "caution": 20
    },
    
    # Trend thresholds (20D change)
    "trend_thresholds": {
        "crisis": 10,
        "fragile": 5,
        "caution": 2
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VixData:
    """Daily VIX data with all metrics."""
    date: str
    vix: float = 0.0
    # Moving averages
    ma_20: float = 0.0
    std_dev_20: float = 0.0
    # Bollinger Bands
    upper_bb: float = 0.0
    lower_bb: float = 0.0
    # Trend
    trend_20d: float = 0.0
    # Regimes
    level_regime: str = "Unknown"
    trend_regime: str = "Unknown"
    bb_regime: str = "Unknown"
    combined_regime: str = "Unknown"
    # Flags
    is_pre_shock: bool = False
    interpretation: str = ""


# =============================================================================
# VIX CALCULATOR
# =============================================================================

class VixDialCalculator:
    """
    Calculate VIX metrics and regime.
    
    Usage:
        calc = VixDialCalculator()
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
        
        self.config = VIX_CONFIG
        self._history_df: Optional[pd.DataFrame] = None
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_from_fred(self, days: int = 400) -> pd.DataFrame:
        """Fetch VIX history from FRED API."""
        api_key = self.config["fred_api_key"]
        
        if not api_key:
            logger.warning("FRED_API_KEY not set")
            return pd.DataFrame()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": self.config["fred_series"],
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df = df[df["value"] != "."]  # Filter missing
                    df["date"] = pd.to_datetime(df["date"])
                    df["vix"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df[["date", "vix"]].dropna()
                    df = df.sort_values("date", ascending=False).reset_index(drop=True)
                    
                    logger.info(f"Fetched {len(df)} VIX observations from FRED")
                    self._history_df = df
                    return df
                    
        except Exception as e:
            logger.error(f"Failed to fetch VIX from FRED: {e}")
        
        return pd.DataFrame()
    
    def fetch_from_supabase(self) -> pd.DataFrame:
        """Fetch VIX history from Supabase as fallback."""
        if not self.supabase:
            return pd.DataFrame()
        
        try:
            response = self.supabase.table("vix_history") \
                .select("date, close") \
                .order("date", desc=True) \
                .limit(400) \
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.rename(columns={"close": "vix"})
                return df
                
        except Exception as e:
            logger.warning(f"Supabase VIX fetch failed: {e}")
        
        return pd.DataFrame()
    
    def fetch_history(self, days: int = 400) -> pd.DataFrame:
        """Fetch VIX history (tries FRED first, then Supabase)."""
        df = self.fetch_from_fred(days)
        
        if df.empty:
            df = self.fetch_from_supabase()
        
        self._history_df = df
        return df
    
    # =========================================================================
    # CALCULATIONS
    # =========================================================================
    
    def calculate_metrics(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate all VIX metrics including Bollinger Bands.
        
        Args:
            df: VIX history DataFrame. If None, fetches.
            
        Returns:
            DataFrame with all calculated metrics
        """
        if df is None:
            df = self._history_df if self._history_df is not None else self.fetch_history()
        
        if df.empty:
            return pd.DataFrame()
        
        period = self.config["bb_period"]
        std_mult = self.config["bb_std"]
        
        # Calculate rolling metrics (data is sorted desc)
        # For rolling calculations, we need to reverse, calculate, then reverse back
        df_asc = df.sort_values("date").reset_index(drop=True)
        
        # 20-day MA and StdDev
        df_asc["ma_20"] = df_asc["vix"].rolling(window=period).mean()
        df_asc["std_dev_20"] = df_asc["vix"].rolling(window=period).std()
        
        # Bollinger Bands
        df_asc["upper_bb"] = df_asc["ma_20"] + std_mult * df_asc["std_dev_20"]
        df_asc["lower_bb"] = df_asc["ma_20"] - std_mult * df_asc["std_dev_20"]
        
        # 20-day trend (current - 20 days ago)
        df_asc["trend_20d"] = df_asc["vix"] - df_asc["vix"].shift(period)
        
        # Sort back to descending
        df = df_asc.sort_values("date", ascending=False).reset_index(drop=True)
        
        # Calculate regimes
        df["level_regime"] = df["vix"].apply(self._get_level_regime)
        df["trend_regime"] = df["trend_20d"].apply(self._get_trend_regime)
        df["bb_regime"] = df.apply(self._get_bb_regime, axis=1)
        df["combined_regime"] = df.apply(self._get_combined_regime, axis=1)
        df["is_pre_shock"] = df["combined_regime"] == "Pre-Shock"
        
        return df
    
    def _get_level_regime(self, vix: float) -> str:
        """Get regime based on VIX level."""
        if pd.isna(vix):
            return "Unknown"
        
        t = self.config["level_thresholds"]
        
        if vix > t["crisis"]:
            return "Crisis"
        elif vix > t["fragile"]:
            return "Fragile"
        elif vix > t["caution"]:
            return "Caution"
        else:
            return "Healthy"
    
    def _get_trend_regime(self, trend: float) -> str:
        """Get regime based on 20D trend."""
        if pd.isna(trend):
            return "Unknown"
        
        t = self.config["trend_thresholds"]
        
        if trend > t["crisis"]:
            return "Crisis"
        elif trend > t["fragile"]:
            return "Fragile"
        elif trend > t["caution"]:
            return "Caution"
        else:
            return "Healthy"
    
    def _get_bb_regime(self, row) -> str:
        """Get Bollinger Band regime."""
        vix = row.get("vix")
        upper = row.get("upper_bb")
        lower = row.get("lower_bb")
        
        if pd.isna(vix) or pd.isna(upper) or pd.isna(lower):
            return "Unknown"
        
        if vix > upper:
            return "Expansion"
        elif vix < lower:
            return "Compression"
        else:
            return "Normal"
    
    def _get_combined_regime(self, row) -> str:
        """Get combined regime from all components."""
        level = row.get("level_regime", "Unknown")
        trend = row.get("trend_regime", "Unknown")
        bb = row.get("bb_regime", "Unknown")
        
        if level == "Unknown":
            return "Unknown"
        
        # Priority order
        if level == "Crisis" or trend == "Crisis":
            return "Crisis"
        elif bb == "Expansion":
            return "Fragile"
        elif level == "Fragile" or trend == "Fragile":
            return "Fragile"
        elif bb == "Compression":
            return "Pre-Shock"  # KEY: Calm before the storm
        elif level == "Caution" or trend == "Caution":
            return "Caution"
        else:
            return "Healthy"
    
    def _get_interpretation(self, regime: str, vix: float, bb_regime: str) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            "Crisis": f"VIX Crisis ({vix:.1f}) - Maximum defensive posture, hedge aggressively",
            "Fragile": f"VIX Fragile ({vix:.1f}) - Elevated risk, reduce exposure",
            "Pre-Shock": f"VIX Compression ({vix:.1f}) - ALERT: Calm before storm, volatility spike likely",
            "Caution": f"VIX Caution ({vix:.1f}) - Elevated but manageable, maintain awareness",
            "Healthy": f"VIX Healthy ({vix:.1f}) - Normal volatility, standard operations"
        }
        return interpretations.get(regime, f"VIX {vix:.1f} - Unknown regime")
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self, date: str = None) -> VixData:
        """
        Calculate VIX data for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses latest.
            
        Returns:
            VixData with all metrics
        """
        logger.info("Calculating VIX metrics...")
        
        df = self.calculate_metrics()
        
        if df.empty:
            return VixData(
                date=date or datetime.now().strftime("%Y-%m-%d"),
                combined_regime="Unknown",
                interpretation="No VIX data available"
            )
        
        # Get specific date or latest
        if date:
            row = df[df["date"] == pd.to_datetime(date)]
            if row.empty:
                row = df.iloc[[0]]
        else:
            row = df.iloc[[0]]
        
        row = row.iloc[0]
        date_str = row["date"].strftime("%Y-%m-%d")
        
        result = VixData(
            date=date_str,
            vix=row["vix"],
            ma_20=row["ma_20"] if pd.notna(row["ma_20"]) else 0,
            std_dev_20=row["std_dev_20"] if pd.notna(row["std_dev_20"]) else 0,
            upper_bb=row["upper_bb"] if pd.notna(row["upper_bb"]) else 0,
            lower_bb=row["lower_bb"] if pd.notna(row["lower_bb"]) else 0,
            trend_20d=row["trend_20d"] if pd.notna(row["trend_20d"]) else 0,
            level_regime=row["level_regime"],
            trend_regime=row["trend_regime"],
            bb_regime=row["bb_regime"],
            combined_regime=row["combined_regime"],
            is_pre_shock=row["combined_regime"] == "Pre-Shock",
            interpretation=self._get_interpretation(
                row["combined_regime"], 
                row["vix"],
                row["bb_regime"]
            )
        )
        
        logger.info(f"VIX: {result.vix:.1f} ({result.combined_regime})")
        
        return result
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_history_to_supabase(self, df: pd.DataFrame = None) -> int:
        """Save VIX history to Supabase."""
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
                "close": row["vix"]
            })
        
        try:
            self.supabase.table("vix_history") \
                .upsert(rows, on_conflict="date") \
                .execute()
            return len(rows)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
            return 0
    
    def save_to_supabase(self, data: VixData) -> bool:
        """Save VIX dial data to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("vix_dial_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved VIX dial for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def get_history(self, days: int = 30) -> List[VixData]:
        """Get VIX dial history."""
        df = self.calculate_metrics()
        
        if df.empty:
            return []
        
        results = []
        for _, row in df.head(days).iterrows():
            results.append(VixData(
                date=row["date"].strftime("%Y-%m-%d"),
                vix=row["vix"],
                ma_20=row["ma_20"] if pd.notna(row["ma_20"]) else 0,
                trend_20d=row["trend_20d"] if pd.notna(row["trend_20d"]) else 0,
                level_regime=row["level_regime"],
                bb_regime=row["bb_regime"],
                combined_regime=row["combined_regime"],
                is_pre_shock=row["combined_regime"] == "Pre-Shock"
            ))
        
        return results


# =============================================================================
# SUPABASE TABLES
# =============================================================================

CREATE_TABLES_SQL = """
-- VIX history (raw data)
CREATE TABLE IF NOT EXISTS vix_history (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    close FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vix_history_date ON vix_history(date DESC);

-- VIX dial daily (calculated metrics)
CREATE TABLE IF NOT EXISTS vix_dial_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    vix FLOAT,
    ma_20 FLOAT,
    std_dev_20 FLOAT,
    upper_bb FLOAT,
    lower_bb FLOAT,
    trend_20d FLOAT,
    level_regime VARCHAR(20),
    trend_regime VARCHAR(20),
    bb_regime VARCHAR(20),
    combined_regime VARCHAR(20),
    is_pre_shock BOOLEAN,
    interpretation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vix_dial_date ON vix_dial_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run VIX dial calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VIX Dial")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from FRED")
    parser.add_argument("--history", type=int, default=0, help="Show N days of history")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = VixDialCalculator()
    
    print(f"\n{'='*60}")
    print("VIX REGIME DIAL")
    print(f"{'='*60}\n")
    
    if args.fetch:
        df = calc.fetch_from_fred()
        print(f"Fetched {len(df)} days of VIX history")
        
        if args.save and not df.empty:
            saved = calc.save_history_to_supabase(df)
            print(f"Saved {saved} rows to Supabase")
    
    if args.history > 0:
        history = calc.get_history(args.history)
        print(f"Last {len(history)} days:\n")
        for v in history[:10]:
            pre_shock = " ⚠️ PRE-SHOCK" if v.is_pre_shock else ""
            print(f"  {v.date}: VIX={v.vix:.1f} ({v.combined_regime}){pre_shock}")
    else:
        data = calc.calculate()
        
        print(f"Date: {data.date}")
        print(f"\n{'='*40}")
        print(f"VIX: {data.vix:.2f}")
        print(f"COMBINED REGIME: {data.combined_regime}")
        if data.is_pre_shock:
            print("⚠️  PRE-SHOCK WARNING: Volatility compression detected!")
        print(f"{'='*40}")
        
        print(f"\n{data.interpretation}")
        
        print(f"\n{'='*40}")
        print("COMPONENT REGIMES")
        print(f"{'='*40}")
        print(f"  Level Regime: {data.level_regime}")
        print(f"  Trend Regime: {data.trend_regime}")
        print(f"  BB Regime:    {data.bb_regime}")
        
        print(f"\n{'='*40}")
        print("BOLLINGER BANDS")
        print(f"{'='*40}")
        print(f"  20-Day MA:    {data.ma_20:.2f}")
        print(f"  Std Dev (20): {data.std_dev_20:.2f}")
        print(f"  Upper BB:     {data.upper_bb:.2f}")
        print(f"  Lower BB:     {data.lower_bb:.2f}")
        print(f"  20D Trend:    {data.trend_20d:+.2f}")
        
        print(f"\n{'='*40}")
        print("REGIME GUIDE")
        print(f"{'='*40}")
        print("  Healthy:   VIX < 20, stable")
        print("  Caution:   VIX 20-30 or rising")
        print("  Fragile:   VIX > 30 or BB expansion")
        print("  Crisis:    VIX > 40 or spiking")
        print("  Pre-Shock: VIX below lower BB (compression)")
        
        if args.save:
            if calc.save_to_supabase(data):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
