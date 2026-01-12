"""
MACRO DIAL
==========
Aggregated macro indicators from FRED for regime classification.

FRED Series:
- VIXCLS: CBOE Volatility Index
- BAMLH0A0HYM2: ICE BofA US High Yield OAS
- BAMLC0A0CM: ICE BofA US Corporate (IG) OAS
- DGS10: 10-Year Treasury
- T10Y2Y: 10Y-2Y Spread (Yield Curve)
- UMCSENT: Consumer Sentiment

VIX Regime Thresholds:
- Complacent: < 15
- Calm: 15-20
- Elevated: 20-25
- High: 25-30
- Very High: 30-40
- Extreme: > 40

Credit Regime Thresholds (HY OAS):
- Tight: < 3%
- Normal: 3-4%
- Elevated: 4-5%
- Stressed: 5-7%
- High Stress: 7-10%
- Crisis: > 10%

Historical Context:
- 2008 HY Peak: ~20%
- 2020 HY Peak: ~11%
- Normal range: 3-5%
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

MACRO_CONFIG = {
    "fred_api_key": os.getenv("FRED_API_KEY"),
    "fred_base_url": "https://api.stlouisfed.org/fred/series/observations",
    
    # FRED Series
    "series": {
        "VIX": "VIXCLS",
        "HY_SPREAD": "BAMLH0A0HYM2",
        "IG_SPREAD": "BAMLC0A0CM",
        "TREASURY_10Y": "DGS10",
        "YIELD_CURVE": "T10Y2Y",
        "CONSUMER_SENTIMENT": "UMCSENT"
    },
    
    # VIX thresholds
    "vix_thresholds": {
        "complacent": 15,
        "calm": 20,
        "elevated": 25,
        "high": 30,
        "very_high": 40
    },
    
    # HY spread thresholds (percentage points)
    "hy_thresholds": {
        "tight": 3,
        "normal": 4,
        "elevated": 5,
        "stressed": 7,
        "high_stress": 10
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MacroData:
    """Daily macro data."""
    date: str
    # Raw values
    vix: float = 0.0
    hy_spread: float = 0.0
    ig_spread: float = 0.0
    hy_ig_diff: float = 0.0
    treasury_10y: float = 0.0
    yield_curve: float = 0.0
    consumer_sentiment: float = 0.0
    # Regimes
    vix_regime: str = "Unknown"
    credit_regime: str = "Unknown"
    combined_regime: str = "Unknown"
    # Flags
    yield_curve_inverted: bool = False
    interpretation: str = ""


# =============================================================================
# MACRO CALCULATOR
# =============================================================================

class MacroDialCalculator:
    """
    Calculate macro indicators and regime.
    
    Usage:
        calc = MacroDialCalculator()
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
        
        self.config = MACRO_CONFIG
    
    # =========================================================================
    # FRED DATA FETCHING
    # =========================================================================
    
    def fetch_fred_series(self, series_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch a series from FRED API."""
        api_key = self.config["fred_api_key"]
        
        if not api_key:
            logger.warning("FRED_API_KEY not set")
            return pd.DataFrame()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc"
        }
        
        try:
            response = requests.get(
                self.config["fred_base_url"], 
                params=params, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df = df[df["value"] != "."]
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df[["date", "value"]].dropna()
                    df.columns = ["date", series_id]
                    return df
                    
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
        
        return pd.DataFrame()
    
    def fetch_all_series(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Fetch all macro series from FRED."""
        logger.info("Fetching macro data from FRED...")
        
        results = {}
        
        for name, series_id in self.config["series"].items():
            df = self.fetch_fred_series(series_id, days)
            if not df.empty:
                results[name] = df
                logger.info(f"Fetched {len(df)} rows for {name}")
        
        return results
    
    def merge_series(self, series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all series by date."""
        if not series_dict:
            return pd.DataFrame()
        
        # Start with first series
        dfs = list(series_dict.values())
        merged = dfs[0].copy()
        
        # Merge remaining
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="date", how="outer")
        
        merged = merged.sort_values("date", ascending=False).reset_index(drop=True)
        
        # Rename columns
        rename_map = {v: k.lower() for k, v in self.config["series"].items()}
        merged = merged.rename(columns=rename_map)
        
        return merged
    
    # =========================================================================
    # REGIME CLASSIFICATION
    # =========================================================================
    
    def classify_vix_regime(self, vix: float) -> str:
        """Classify VIX regime."""
        if pd.isna(vix):
            return "Unknown"
        
        t = self.config["vix_thresholds"]
        
        if vix < t["complacent"]:
            return "Complacent"
        elif vix < t["calm"]:
            return "Calm"
        elif vix < t["elevated"]:
            return "Elevated"
        elif vix < t["high"]:
            return "High"
        elif vix < t["very_high"]:
            return "Very High"
        else:
            return "Extreme"
    
    def classify_credit_regime(self, hy_spread: float) -> str:
        """Classify credit regime based on HY spread."""
        if pd.isna(hy_spread):
            return "Unknown"
        
        t = self.config["hy_thresholds"]
        
        if hy_spread < t["tight"]:
            return "Tight"
        elif hy_spread < t["normal"]:
            return "Normal"
        elif hy_spread < t["elevated"]:
            return "Elevated"
        elif hy_spread < t["stressed"]:
            return "Stressed"
        elif hy_spread < t["high_stress"]:
            return "High Stress"
        else:
            return "Crisis"
    
    def classify_combined_regime(self, vix_regime: str, credit_regime: str) -> str:
        """Classify combined macro regime."""
        # Priority order (worst first)
        vix_order = ["Extreme", "Very High", "High", "Elevated", "Calm", "Complacent"]
        credit_order = ["Crisis", "High Stress", "Stressed", "Elevated", "Normal", "Tight"]
        
        vix_rank = vix_order.index(vix_regime) if vix_regime in vix_order else 5
        credit_rank = credit_order.index(credit_regime) if credit_regime in credit_order else 5
        
        # Map to combined regime
        worst_rank = min(vix_rank, credit_rank)
        
        if worst_rank == 0:
            return "Crisis"
        elif worst_rank == 1:
            return "High Stress"
        elif worst_rank == 2:
            return "Stressed"
        elif worst_rank == 3:
            return "Elevated"
        elif worst_rank == 4:
            return "Normal"
        else:
            return "Calm"
    
    def _get_interpretation(self, combined: str, vix: float, hy: float, yc: float) -> str:
        """Get human-readable interpretation."""
        parts = []
        
        # VIX commentary
        if vix > 30:
            parts.append(f"VIX elevated at {vix:.1f}")
        elif vix < 15:
            parts.append(f"VIX complacent at {vix:.1f}")
        
        # Credit commentary
        if hy > 5:
            parts.append(f"HY spreads stressed at {hy:.2f}%")
        elif hy < 3:
            parts.append(f"HY spreads tight at {hy:.2f}%")
        
        # Yield curve
        if yc < 0:
            parts.append("⚠️ Yield curve inverted")
        
        base = {
            "Crisis": "Macro crisis - maximum defensive posture",
            "High Stress": "High macro stress - significant risk reduction needed",
            "Stressed": "Macro stress - reduce exposure, raise hedges",
            "Elevated": "Elevated macro risk - maintain awareness",
            "Normal": "Normal macro conditions - standard operations",
            "Calm": "Calm macro environment - favorable conditions"
        }
        
        main = base.get(combined, "Unknown macro regime")
        
        if parts:
            return f"{main}. {'. '.join(parts)}."
        return main
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self, date: str = None) -> MacroData:
        """
        Calculate macro data for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses latest.
            
        Returns:
            MacroData with all metrics
        """
        logger.info("Calculating macro data...")
        
        # Fetch all series
        series = self.fetch_all_series(days=60)
        
        if not series:
            return MacroData(
                date=date or datetime.now().strftime("%Y-%m-%d"),
                combined_regime="Unknown",
                interpretation="No macro data available"
            )
        
        # Merge series
        df = self.merge_series(series)
        
        if df.empty:
            return MacroData(
                date=date or datetime.now().strftime("%Y-%m-%d"),
                combined_regime="Unknown",
                interpretation="Failed to merge macro data"
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
        
        # Extract values
        vix = row.get("vix", 0) or 0
        hy = row.get("hy_spread", 0) or 0
        ig = row.get("ig_spread", 0) or 0
        t10y = row.get("treasury_10y", 0) or 0
        yc = row.get("yield_curve", 0) or 0
        sent = row.get("consumer_sentiment", 0) or 0
        
        # Calculate derived
        hy_ig_diff = hy - ig if hy and ig else 0
        
        # Classify regimes
        vix_regime = self.classify_vix_regime(vix)
        credit_regime = self.classify_credit_regime(hy)
        combined_regime = self.classify_combined_regime(vix_regime, credit_regime)
        
        result = MacroData(
            date=date_str,
            vix=vix,
            hy_spread=hy,
            ig_spread=ig,
            hy_ig_diff=hy_ig_diff,
            treasury_10y=t10y,
            yield_curve=yc,
            consumer_sentiment=sent,
            vix_regime=vix_regime,
            credit_regime=credit_regime,
            combined_regime=combined_regime,
            yield_curve_inverted=yc < 0 if pd.notna(yc) else False,
            interpretation=self._get_interpretation(combined_regime, vix, hy, yc)
        )
        
        logger.info(f"Macro: VIX={vix:.1f} ({vix_regime}), HY={hy:.2f}% ({credit_regime})")
        
        return result
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, data: MacroData) -> bool:
        """Save macro data to Supabase."""
        import math
        import numpy as np
        
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        # Convert numpy/NaN/inf values for JSON compatibility
        for key, value in row.items():
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                # Convert numpy types to Python native types
                row[key] = value.item() if hasattr(value, 'item') else bool(value) if isinstance(value, np.bool_) else float(value)
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                row[key] = None
            elif isinstance(value, bool):
                row[key] = bool(value)  # Ensure Python bool
        
        try:
            self.supabase.table("macro_dial_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            logger.info(f"Saved macro dial for {data.date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def get_history(self, days: int = 30) -> List[MacroData]:
        """Get macro history from Supabase."""
        if not self.supabase:
            return []
        
        try:
            response = self.supabase.table("macro_dial_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(days) \
                .execute()
            
            return [MacroData(**row) for row in response.data]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


# =============================================================================
# SUPABASE TABLE
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS macro_dial_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    vix FLOAT,
    hy_spread FLOAT,
    ig_spread FLOAT,
    hy_ig_diff FLOAT,
    treasury_10y FLOAT,
    yield_curve FLOAT,
    consumer_sentiment FLOAT,
    vix_regime VARCHAR(20),
    credit_regime VARCHAR(20),
    combined_regime VARCHAR(20),
    yield_curve_inverted BOOLEAN,
    interpretation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_macro_dial_date ON macro_dial_daily(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run macro dial calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Macro Dial")
    parser.add_argument("--history", type=int, default=0, help="Show N days of history")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = MacroDialCalculator()
    
    print(f"\n{'='*60}")
    print("MACRO DIAL")
    print(f"{'='*60}\n")
    
    if args.history > 0:
        history = calc.get_history(args.history)
        print(f"Last {len(history)} days:\n")
        for m in history[:10]:
            yc_flag = " ⚠️" if m.yield_curve_inverted else ""
            print(f"  {m.date}: VIX={m.vix:.1f} HY={m.hy_spread:.2f}% ({m.combined_regime}){yc_flag}")
    else:
        data = calc.calculate()
        
        print(f"Date: {data.date}")
        print(f"\n{'='*40}")
        print(f"COMBINED REGIME: {data.combined_regime}")
        if data.yield_curve_inverted:
            print("⚠️  YIELD CURVE INVERTED")
        print(f"{'='*40}")
        
        print(f"\n{data.interpretation}")
        
        print(f"\n{'='*40}")
        print("VOLATILITY")
        print(f"{'='*40}")
        print(f"  VIX:         {data.vix:.2f}")
        print(f"  VIX Regime:  {data.vix_regime}")
        
        print(f"\n{'='*40}")
        print("CREDIT")
        print(f"{'='*40}")
        print(f"  HY Spread:     {data.hy_spread:.2f}%")
        print(f"  IG Spread:     {data.ig_spread:.2f}%")
        print(f"  HY-IG Diff:    {data.hy_ig_diff:.2f}%")
        print(f"  Credit Regime: {data.credit_regime}")
        
        print(f"\n{'='*40}")
        print("RATES")
        print(f"{'='*40}")
        print(f"  10Y Treasury:  {data.treasury_10y:.2f}%")
        print(f"  Yield Curve:   {data.yield_curve:.2f}%")
        
        print(f"\n{'='*40}")
        print("SENTIMENT")
        print(f"{'='*40}")
        print(f"  Consumer:      {data.consumer_sentiment:.1f}")
        
        if args.save:
            if calc.save_to_supabase(data):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()