"""
LABOR DIAL
==========
Tracks BLS labor market indicators for recession early warning.

FRED Series:
- ICSA: Weekly initial jobless claims
- CCSA: Weekly continuing claims
- UNRATE: Monthly unemployment rate
- PAYEMS: Monthly nonfarm payrolls (thousands)
- CES0500000003: Monthly avg hourly earnings

Thresholds:
Initial Claims:
- Healthy: <= 225,000
- Caution: <= 275,000
- Fragile: <= 350,000
- Crisis: > 350,000

Unemployment Rate:
- Healthy: <= 4.5%
- Caution: <= 5.5%
- Fragile: <= 7.0%
- Crisis: > 7.0%

Payrolls Change (MoM):
- Healthy: >= +150K
- Caution: >= +75K
- Fragile: >= 0
- Crisis: < 0
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

import requests
import pandas as pd

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

LABOR_CONFIG = {
    "fred_api_key": os.getenv("FRED_API_KEY"),
    "fred_base_url": "https://api.stlouisfed.org/fred/series/observations",
    
    "series": {
        "initial_claims": "ICSA",           # Weekly initial jobless claims
        "continuing_claims": "CCSA",        # Weekly continuing claims
        "unemployment_rate": "UNRATE",      # Monthly unemployment rate
        "payrolls": "PAYEMS",               # Monthly nonfarm payrolls
        "avg_hourly_earnings": "CES0500000003"  # Monthly avg hourly earnings
    },
    
    "thresholds": {
        "initial_claims": {
            "healthy": 225000,
            "caution": 275000,
            "fragile": 350000,
            "crisis": 500000
        },
        "continuing_claims": {
            "healthy": 1700000,
            "caution": 2000000,
            "fragile": 2500000,
            "crisis": 3500000
        },
        "unemployment_rate": {
            "healthy": 4.5,
            "caution": 5.5,
            "fragile": 7.0,
            "crisis": 9.0
        },
        "payrolls_change": {  # MoM change in thousands
            "healthy": 150,
            "caution": 75,
            "fragile": 0,
            "crisis": -100
        }
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LaborIndicator:
    """Single labor indicator data."""
    name: str
    value: float
    previous: float
    change: float
    date: str
    regime: str
    score: int


@dataclass
class LaborDialData:
    """Complete labor dial data."""
    date: str
    # Key indicators
    initial_claims: Optional[float]
    continuing_claims: Optional[float]
    unemployment_rate: Optional[float]
    payrolls: Optional[float]
    payrolls_change: Optional[float]
    # Regimes
    claims_regime: str
    unemployment_regime: str
    payrolls_regime: str
    combined_regime: str
    # Composite score
    labor_score: float
    interpretation: str


# =============================================================================
# LABOR CALCULATOR
# =============================================================================

class LaborDialCalculator:
    """
    Calculate labor market indicators.
    
    Usage:
        calc = LaborDialCalculator()
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
        
        self.config = LABOR_CONFIG
        self.fred_key = self.config["fred_api_key"]
    
    def fetch_fred_series(self, series_id: str, limit: int = 12) -> Optional[Dict]:
        """Fetch series from FRED."""
        if not self.fred_key:
            logger.warning("FRED_API_KEY not set")
            return None
        
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit
        }
        
        try:
            response = requests.get(self.config["fred_base_url"], params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                if observations:
                    latest = observations[0]
                    previous = observations[1] if len(observations) > 1 else latest
                    
                    value = float(latest["value"]) if latest["value"] != "." else None
                    prev_value = float(previous["value"]) if previous["value"] != "." else None
                    
                    if value is not None:
                        return {
                            "value": value,
                            "previous": prev_value,
                            "change": value - prev_value if prev_value else 0,
                            "date": latest["date"]
                        }
        except Exception as e:
            logger.warning(f"Failed to fetch {series_id}: {e}")
        
        return None
    
    def get_regime(self, indicator: str, value: float) -> tuple:
        """Get regime and score for indicator."""
        thresholds = self.config["thresholds"].get(indicator, {})
        
        if not thresholds:
            return "Unknown", 50
        
        # For claims and unemployment, higher is worse
        if indicator in ["initial_claims", "continuing_claims", "unemployment_rate"]:
            if value <= thresholds.get("healthy", 0):
                return "Healthy", 0
            elif value <= thresholds.get("caution", 0):
                return "Caution", 25
            elif value <= thresholds.get("fragile", 0):
                return "Fragile", 60
            else:
                return "Crisis", 100
        
        # For payrolls change, higher is better
        elif indicator == "payrolls_change":
            if value >= thresholds.get("healthy", 0):
                return "Healthy", 0
            elif value >= thresholds.get("caution", 0):
                return "Caution", 25
            elif value >= thresholds.get("fragile", 0):
                return "Fragile", 60
            else:
                return "Crisis", 100
        
        return "Unknown", 50
    
    def calculate(self) -> LaborDialData:
        """Main calculation: fetch and process labor data."""
        logger.info("Calculating labor market indicators...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch all series
        series = self.config["series"]
        
        initial_claims_data = self.fetch_fred_series(series["initial_claims"])
        continuing_claims_data = self.fetch_fred_series(series["continuing_claims"])
        unemployment_data = self.fetch_fred_series(series["unemployment_rate"])
        payrolls_data = self.fetch_fred_series(series["payrolls"])
        
        # Extract values
        initial_claims = initial_claims_data["value"] if initial_claims_data else None
        continuing_claims = continuing_claims_data["value"] if continuing_claims_data else None
        unemployment_rate = unemployment_data["value"] if unemployment_data else None
        payrolls = payrolls_data["value"] if payrolls_data else None
        payrolls_change = payrolls_data["change"] if payrolls_data else None
        
        # Calculate regimes
        claims_regime, claims_score = self.get_regime("initial_claims", initial_claims) if initial_claims else ("Unknown", 50)
        unemployment_regime, unemp_score = self.get_regime("unemployment_rate", unemployment_rate) if unemployment_rate else ("Unknown", 50)
        payrolls_regime, payrolls_score = self.get_regime("payrolls_change", payrolls_change) if payrolls_change else ("Unknown", 50)
        
        # Combined regime (worst of the three)
        regimes = [claims_regime, unemployment_regime, payrolls_regime]
        regime_order = ["Crisis", "Fragile", "Caution", "Healthy", "Unknown"]
        
        combined_regime = "Healthy"
        for r in regime_order:
            if r in regimes:
                combined_regime = r
                break
        
        # Labor score (average of scores)
        scores = [s for s in [claims_score, unemp_score, payrolls_score] if s is not None]
        labor_score = sum(scores) / len(scores) if scores else 50
        
        # Interpretation
        interpretation = self._get_interpretation(combined_regime, initial_claims, unemployment_rate, payrolls_change)
        
        result = LaborDialData(
            date=date_str,
            initial_claims=initial_claims,
            continuing_claims=continuing_claims,
            unemployment_rate=unemployment_rate,
            payrolls=payrolls,
            payrolls_change=payrolls_change,
            claims_regime=claims_regime,
            unemployment_regime=unemployment_regime,
            payrolls_regime=payrolls_regime,
            combined_regime=combined_regime,
            labor_score=round(labor_score, 1),
            interpretation=interpretation
        )
        
        logger.info(f"Labor Regime: {combined_regime}, Score: {labor_score:.1f}")
        
        return result
    
    def _get_interpretation(self, regime: str, claims: float, unemp: float, payrolls_chg: float) -> str:
        """Get human-readable interpretation."""
        parts = []
        
        if claims:
            parts.append(f"Initial claims: {claims:,.0f}")
        if unemp:
            parts.append(f"Unemployment: {unemp:.1f}%")
        if payrolls_chg:
            parts.append(f"Payrolls MoM: {payrolls_chg:+,.0f}K")
        
        base = {
            "Healthy": "Labor market strong - low recession risk",
            "Caution": "Labor market softening - monitor trends",
            "Fragile": "Labor market stressed - elevated recession risk",
            "Crisis": "Labor market crisis - recession likely"
        }.get(regime, "Unknown labor conditions")
        
        return f"{base}. {', '.join(parts)}"
    
    def save_to_supabase(self, data: LaborDialData) -> bool:
        """Save labor data to Supabase."""
        if not self.supabase:
            return False
        
        row = asdict(data)
        
        try:
            self.supabase.table("labor_dial_daily") \
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
    
    parser = argparse.ArgumentParser(description="Labor Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = LaborDialCalculator()
    
    print(f"\n{'='*60}")
    print("LABOR DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"COMBINED REGIME: {data.combined_regime}")
    print(f"LABOR SCORE: {data.labor_score:.1f}/100")
    print(f"{'='*40}")
    
    print(f"\n{'='*40}")
    print("INDICATORS")
    print(f"{'='*40}")
    if data.initial_claims:
        print(f"  Initial Claims:     {data.initial_claims:,.0f} ({data.claims_regime})")
    if data.continuing_claims:
        print(f"  Continuing Claims:  {data.continuing_claims:,.0f}")
    if data.unemployment_rate:
        print(f"  Unemployment Rate:  {data.unemployment_rate:.1f}% ({data.unemployment_regime})")
    if data.payrolls:
        print(f"  Nonfarm Payrolls:   {data.payrolls:,.0f}K")
    if data.payrolls_change:
        print(f"  Payrolls Change:    {data.payrolls_change:+,.0f}K MoM ({data.payrolls_regime})")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
