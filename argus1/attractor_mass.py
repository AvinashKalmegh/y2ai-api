"""
ATTRACTOR MASS CALCULATOR (FlowOS)
==================================
Measures the "gravitational pull" of attractor stocks within their bubble ecosystems.
Higher Attractor Mass = stronger narrative concentration + flow dominance.

Formula:
    Attractor Mass = NarrativeDensity (30%) + VolumeConcentration (40%) + PriceMomentum (30%)

Components:
1. NarrativeDensity (30%): % of theme news mentioning the attractor specifically
   - Source: NST module (nst_mentions table)
   - High density = attractor dominating the narrative

2. VolumeConcentration (40%): Attractor's share of pillar dollar volume
   - Measures capital flow concentration into the attractor
   - High concentration = money gravitating to the attractor

3. PriceMomentum (30%): Attractor's relative strength vs pillar peers
   - 20D return Z-score relative to pillar average
   - High momentum = attractor outperforming its ecosystem

Bubble Types:
    - AI/Compute: NVDA (attractor) vs SMH components
    - Energy/Grid: CEG (attractor) vs nuclear/utility peers
    - Crypto: COIN (attractor) vs crypto-adjacent stocks
    - Clean Energy: ENPH (attractor) vs solar/clean peers

Usage:
    from dials.attractor_mass import AttractorMassCalculator
    
    calc = AttractorMassCalculator()
    result = calc.calculate('AI/Compute')
    print(f"Attractor Mass: {result.attractor_mass}")
    print(f"Signal: {result.signal}")
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Attractor tickers per bubble type (matches NST)
ATTRACTOR_TICKERS = {
    'AI/Compute': 'NVDA',
    'Energy/Grid': 'CEG',
    'Crypto': 'COIN',
    'Clean Energy': 'ENPH',
}

# Peer universe for each bubble type (for volume concentration & momentum)
BUBBLE_UNIVERSES = {
    'AI/Compute': {
        'attractor': 'NVDA',
        'peers': ['AMD', 'AVGO', 'TSM', 'ASML', 'MU', 'INTC', 'QCOM', 'ARM', 
                  'MRVL', 'KLAC', 'LRCX', 'AMAT', 'SMCI', 'VRT', 'DELL'],
    },
    'Energy/Grid': {
        'attractor': 'CEG',
        'peers': ['VST', 'NRG', 'DUK', 'SO', 'NEE', 'AEP', 'XEL', 'ETR',
                  'SMR', 'OKLO', 'NNE', 'LEU'],
    },
    'Crypto': {
        'attractor': 'COIN',
        'peers': ['MSTR', 'RIOT', 'MARA', 'CLSK', 'HUT', 'BITF', 'HIVE',
                  'SQ', 'PYPL', 'HOOD'],
    },
    'Clean Energy': {
        'attractor': 'ENPH',
        'peers': ['SEDG', 'FSLR', 'RUN', 'NOVA', 'ARRY', 'MAXN', 'JKS',
                  'CSIQ', 'DQ', 'SPWR'],
    },
}

# Component weights
ATTRACTOR_MASS_WEIGHTS = {
    'narrative_density': 0.30,
    'volume_concentration': 0.40,
    'price_momentum': 0.30,
}

# Signal thresholds (0-100 scale)
SIGNAL_THRESHOLDS = {
    'EXTREME': 80,      # Attractor Mass >= 80 = extreme concentration
    'HIGH': 65,         # 65-80 = high concentration
    'MODERATE': 50,     # 50-65 = moderate
    'LOW': 35,          # 35-50 = low
    # < 35 = DISPERSED (capital/narrative spread across ecosystem)
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AttractorMassResult:
    """Complete Attractor Mass calculation result."""
    bubble_type: str
    attractor_ticker: str
    date: str
    
    # Final score
    attractor_mass: float  # 0-100 scale
    signal: str            # EXTREME, HIGH, MODERATE, LOW, DISPERSED
    
    # Component scores (0-100 each)
    narrative_density_score: float
    volume_concentration_score: float
    price_momentum_score: float
    
    # Raw metrics
    narrative_density_raw: float    # 0-1 ratio
    volume_concentration_raw: float # 0-1 ratio
    price_momentum_zscore: float    # z-score
    
    # Context
    attractor_dollar_volume_m: float
    universe_dollar_volume_m: float
    attractor_return_20d: float
    universe_avg_return_20d: float
    
    interpretation: str


@dataclass
class ComponentData:
    """Intermediate data for a single component."""
    raw_value: float
    scaled_score: float  # 0-100
    weight: float
    weighted_score: float


# =============================================================================
# ATTRACTOR MASS CALCULATOR
# =============================================================================

class AttractorMassCalculator:
    """
    Calculate Attractor Mass for FlowOS bubble detection.
    
    Usage:
        calc = AttractorMassCalculator()
        result = calc.calculate('AI/Compute')
        calc.save_to_supabase(result)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator with optional Supabase connection."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.weights = ATTRACTOR_MASS_WEIGHTS
        self._price_cache = {}
    
    # -------------------------------------------------------------------------
    # COMPONENT 1: Narrative Density (30%)
    # -------------------------------------------------------------------------
    
    def get_narrative_density(self, bubble_type: str, days: int = 7) -> Tuple[float, float]:
        """
        Get NarrativeDensity from NST module.
        
        Returns:
            Tuple of (raw_density 0-1, scaled_score 0-100)
        """
        # Try multiple import paths to handle different execution contexts
        nst_get_density = None
        
        # Try 1: Direct import from argus1 (when running from y2ai package)
        try:
            from argus1.nst import get_narrative_density as nst_get_density
        except ImportError:
            pass
        
        # Try 2: Import from y2ai.argus1 (when running as module)
        if nst_get_density is None:
            try:
                from y2ai.argus1.nst import get_narrative_density as nst_get_density
            except ImportError:
                pass
        
        # Try 3: Import from nst package directly
        if nst_get_density is None:
            try:
                from nst import get_narrative_density as nst_get_density
            except ImportError:
                pass
        
        # Try 4: Relative import
        if nst_get_density is None:
            try:
                import sys
                import os
                # Add parent directory to path
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from argus1.nst import get_narrative_density as nst_get_density
            except ImportError:
                pass
        
        if nst_get_density is None:
            logger.warning("NST module not available, using defaults")
            return 0.5, 50.0
        
        try:
            result = nst_get_density(bubble_type, days)
            
            if result.status == 'NO_DATA':
                logger.warning(f"No NST data for {bubble_type}, using default")
                return 0.5, 50.0
            
            # raw is 0-1, scaled is 0-100
            return result.narrative_density, result.scaled
            
        except Exception as e:
            logger.error(f"Error getting narrative density: {e}")
            return 0.5, 50.0
    
    # -------------------------------------------------------------------------
    # COMPONENT 2: Volume Concentration (40%)
    # -------------------------------------------------------------------------
    
    def get_volume_concentration(self, bubble_type: str, days: int = 5) -> Tuple[float, float, float, float]:
        """
        Calculate attractor's share of universe dollar volume.
        
        Returns:
            Tuple of (raw_concentration 0-1, scaled_score 0-100, 
                     attractor_vol_m, universe_vol_m)
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available")
            return 0.5, 50.0, 0, 0
        
        universe = BUBBLE_UNIVERSES.get(bubble_type)
        if not universe:
            logger.warning(f"Unknown bubble type: {bubble_type}")
            return 0.5, 50.0, 0, 0
        
        attractor = universe['attractor']
        all_tickers = [attractor] + universe['peers']
        
        try:
            # Fetch data for all tickers
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)  # Buffer for trading days
            
            dollar_volumes = {}
            
            for ticker in all_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if len(hist) >= days:
                        # Calculate average daily dollar volume
                        hist['dollar_volume'] = hist['Close'] * hist['Volume']
                        avg_dv = hist['dollar_volume'].tail(days).mean()
                        dollar_volumes[ticker] = avg_dv / 1_000_000  # Convert to millions
                except Exception as e:
                    logger.debug(f"Failed to fetch {ticker}: {e}")
                    continue
            
            if attractor not in dollar_volumes:
                logger.warning(f"Could not fetch attractor {attractor}")
                return 0.5, 50.0, 0, 0
            
            attractor_vol = dollar_volumes[attractor]
            universe_vol = sum(dollar_volumes.values())
            
            if universe_vol == 0:
                return 0.5, 50.0, attractor_vol, universe_vol
            
            # Raw concentration (0-1)
            concentration = attractor_vol / universe_vol
            
            # Scale to 0-100
            # Expected range: 10% (dispersed) to 60% (extreme concentration)
            # Map: 0.10 -> 0, 0.35 -> 50, 0.60 -> 100
            if concentration <= 0.10:
                scaled = 0
            elif concentration >= 0.60:
                scaled = 100
            else:
                scaled = (concentration - 0.10) / (0.60 - 0.10) * 100
            
            return concentration, scaled, attractor_vol, universe_vol
            
        except Exception as e:
            logger.error(f"Error calculating volume concentration: {e}")
            return 0.5, 50.0, 0, 0
    
    # -------------------------------------------------------------------------
    # COMPONENT 3: Price Momentum (30%)
    # -------------------------------------------------------------------------
    
    def get_price_momentum(self, bubble_type: str, lookback: int = 20) -> Tuple[float, float, float, float]:
        """
        Calculate attractor's momentum relative to peers.
        
        Returns:
            Tuple of (zscore, scaled_score 0-100, 
                     attractor_return, universe_avg_return)
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available")
            return 0.0, 50.0, 0, 0
        
        universe = BUBBLE_UNIVERSES.get(bubble_type)
        if not universe:
            return 0.0, 50.0, 0, 0
        
        attractor = universe['attractor']
        all_tickers = [attractor] + universe['peers']
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback + 10)
            
            returns = {}
            
            for ticker in all_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if len(hist) >= lookback:
                        # Calculate lookback-day return
                        start_price = hist['Close'].iloc[-lookback]
                        end_price = hist['Close'].iloc[-1]
                        ret = (end_price - start_price) / start_price
                        returns[ticker] = ret
                except Exception as e:
                    logger.debug(f"Failed to fetch {ticker}: {e}")
                    continue
            
            if attractor not in returns or len(returns) < 3:
                return 0.0, 50.0, 0, 0
            
            attractor_return = returns[attractor]
            peer_returns = [r for t, r in returns.items() if t != attractor]
            
            if not peer_returns:
                return 0.0, 50.0, attractor_return, 0
            
            # Calculate z-score
            peer_mean = np.mean(peer_returns)
            peer_std = np.std(peer_returns)
            
            if peer_std == 0:
                zscore = 0
            else:
                zscore = (attractor_return - peer_mean) / peer_std
            
            # Scale z-score to 0-100
            # z = -2 -> 0, z = 0 -> 50, z = +2 -> 100
            scaled = 50 + (zscore * 25)
            scaled = max(0, min(100, scaled))
            
            return zscore, scaled, attractor_return, peer_mean
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return 0.0, 50.0, 0, 0
    
    # -------------------------------------------------------------------------
    # MAIN CALCULATION
    # -------------------------------------------------------------------------
    
    def calculate(self, bubble_type: str) -> AttractorMassResult:
        """
        Calculate complete Attractor Mass for a bubble type.
        
        Args:
            bubble_type: One of 'AI/Compute', 'Energy/Grid', 'Crypto', 'Clean Energy'
        
        Returns:
            AttractorMassResult with all components and final score
        """
        logger.info(f"Calculating Attractor Mass for {bubble_type}...")
        
        attractor = ATTRACTOR_TICKERS.get(bubble_type, 'UNKNOWN')
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Get component scores
        narrative_raw, narrative_scaled = self.get_narrative_density(bubble_type)
        vol_raw, vol_scaled, attractor_vol, universe_vol = self.get_volume_concentration(bubble_type)
        momentum_z, momentum_scaled, attractor_ret, universe_ret = self.get_price_momentum(bubble_type)
        
        # Calculate weighted Attractor Mass
        attractor_mass = (
            narrative_scaled * self.weights['narrative_density'] +
            vol_scaled * self.weights['volume_concentration'] +
            momentum_scaled * self.weights['price_momentum']
        )
        
        # Determine signal
        if attractor_mass >= SIGNAL_THRESHOLDS['EXTREME']:
            signal = 'EXTREME'
        elif attractor_mass >= SIGNAL_THRESHOLDS['HIGH']:
            signal = 'HIGH'
        elif attractor_mass >= SIGNAL_THRESHOLDS['MODERATE']:
            signal = 'MODERATE'
        elif attractor_mass >= SIGNAL_THRESHOLDS['LOW']:
            signal = 'LOW'
        else:
            signal = 'DISPERSED'
        
        # Build interpretation
        interpretation = self._build_interpretation(
            bubble_type, attractor, signal, attractor_mass,
            narrative_scaled, vol_scaled, momentum_scaled,
            vol_raw, momentum_z
        )
        
        result = AttractorMassResult(
            bubble_type=bubble_type,
            attractor_ticker=attractor,
            date=date_str,
            attractor_mass=round(attractor_mass, 1),
            signal=signal,
            narrative_density_score=round(narrative_scaled, 1),
            volume_concentration_score=round(vol_scaled, 1),
            price_momentum_score=round(momentum_scaled, 1),
            narrative_density_raw=round(narrative_raw, 3),
            volume_concentration_raw=round(vol_raw, 3),
            price_momentum_zscore=round(momentum_z, 2),
            attractor_dollar_volume_m=round(attractor_vol, 1),
            universe_dollar_volume_m=round(universe_vol, 1),
            attractor_return_20d=round(attractor_ret * 100, 2) if attractor_ret else 0,
            universe_avg_return_20d=round(universe_ret * 100, 2) if universe_ret else 0,
            interpretation=interpretation
        )
        
        logger.info(f"Attractor Mass: {result.attractor_mass} ({signal})")
        
        return result
    
    def _build_interpretation(
        self, bubble_type: str, attractor: str, signal: str, mass: float,
        narrative: float, volume: float, momentum: float,
        vol_raw: float, momentum_z: float
    ) -> str:
        """Build human-readable interpretation."""
        
        parts = [f"{attractor} Attractor Mass: {mass:.1f} ({signal})"]
        
        # Identify dominant component
        components = [
            ('Narrative', narrative),
            ('Volume', volume),
            ('Momentum', momentum)
        ]
        dominant = max(components, key=lambda x: x[1])
        weakest = min(components, key=lambda x: x[1])
        
        if signal in ['EXTREME', 'HIGH']:
            parts.append(f"Strong gravitational pull driven by {dominant[0].lower()} ({dominant[1]:.0f}).")
            parts.append(f"{attractor} capturing {vol_raw*100:.0f}% of {bubble_type} dollar volume.")
        elif signal == 'MODERATE':
            parts.append(f"Moderate concentration with {dominant[0].lower()} leading.")
        else:
            parts.append(f"Capital dispersed across {bubble_type} ecosystem.")
            parts.append(f"Weakest component: {weakest[0]} ({weakest[1]:.0f}).")
        
        if momentum_z > 1.5:
            parts.append(f"Momentum strongly positive (z={momentum_z:.1f}).")
        elif momentum_z < -1.5:
            parts.append(f"Momentum lagging peers (z={momentum_z:.1f}).")
        
        return " ".join(parts)
    
    # -------------------------------------------------------------------------
    # STORAGE
    # -------------------------------------------------------------------------
    
    def save_to_supabase(self, result: AttractorMassResult) -> bool:
        """Save Attractor Mass result to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return False
        
        row = {
            'date': result.date,
            'bubble_type': result.bubble_type,
            'attractor_ticker': result.attractor_ticker,
            'attractor_mass': result.attractor_mass,
            'signal': result.signal,
            'narrative_density_score': result.narrative_density_score,
            'volume_concentration_score': result.volume_concentration_score,
            'price_momentum_score': result.price_momentum_score,
            'narrative_density_raw': result.narrative_density_raw,
            'volume_concentration_raw': result.volume_concentration_raw,
            'price_momentum_zscore': result.price_momentum_zscore,
            'attractor_dollar_volume_m': result.attractor_dollar_volume_m,
            'universe_dollar_volume_m': result.universe_dollar_volume_m,
            'attractor_return_20d': result.attractor_return_20d,
            'universe_avg_return_20d': result.universe_avg_return_20d,
            'interpretation': result.interpretation,
        }
        
        try:
            self.supabase.table('attractor_mass_daily') \
                .upsert(row, on_conflict='date,bubble_type') \
                .execute()
            logger.info(f"Saved Attractor Mass for {result.bubble_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False
    
    def calculate_all(self, save: bool = False) -> Dict[str, AttractorMassResult]:
        """Calculate Attractor Mass for all bubble types."""
        results = {}
        
        for bubble_type in ATTRACTOR_TICKERS.keys():
            result = self.calculate(bubble_type)
            results[bubble_type] = result
            
            if save:
                self.save_to_supabase(result)
        
        return results


# =============================================================================
# SQL SCHEMA
# =============================================================================

ATTRACTOR_MASS_SCHEMA = """
-- Attractor Mass daily tracking for FlowOS
CREATE TABLE IF NOT EXISTS attractor_mass_daily (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    bubble_type TEXT NOT NULL,
    attractor_ticker TEXT NOT NULL,
    
    -- Final score
    attractor_mass NUMERIC(5,2),
    signal TEXT,
    
    -- Component scores (0-100)
    narrative_density_score NUMERIC(5,2),
    volume_concentration_score NUMERIC(5,2),
    price_momentum_score NUMERIC(5,2),
    
    -- Raw metrics
    narrative_density_raw NUMERIC(5,3),
    volume_concentration_raw NUMERIC(5,3),
    price_momentum_zscore NUMERIC(5,2),
    
    -- Context
    attractor_dollar_volume_m NUMERIC(12,2),
    universe_dollar_volume_m NUMERIC(12,2),
    attractor_return_20d NUMERIC(7,2),
    universe_avg_return_20d NUMERIC(7,2),
    
    interpretation TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(date, bubble_type)
);

-- Index for querying
CREATE INDEX IF NOT EXISTS idx_attractor_mass_bubble_date 
ON attractor_mass_daily(bubble_type, date DESC);

-- View for latest readings
CREATE OR REPLACE VIEW attractor_mass_latest AS
SELECT DISTINCT ON (bubble_type) *
FROM attractor_mass_daily
ORDER BY bubble_type, date DESC;
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run Attractor Mass calculation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlowOS Attractor Mass Calculator")
    parser.add_argument('--bubble', default='AI/Compute',
                       choices=['AI/Compute', 'Energy/Grid', 'Crypto', 'Clean Energy', 'all'],
                       help='Bubble type to calculate')
    parser.add_argument('--save', action='store_true',
                       help='Save results to Supabase')
    parser.add_argument('--schema', action='store_true',
                       help='Print SQL schema and exit')
    
    args = parser.parse_args()
    
    if args.schema:
        print(ATTRACTOR_MASS_SCHEMA)
        return
    
    calc = AttractorMassCalculator()
    
    print(f"\n{'='*60}")
    print("FLOWOS ATTRACTOR MASS CALCULATOR")
    print(f"{'='*60}\n")
    
    if args.bubble == 'all':
        results = calc.calculate_all(save=args.save)
        
        print(f"{'Bubble Type':<15} {'Attractor':<8} {'Mass':>6} {'Signal':<10}")
        print("-" * 45)
        
        for bubble_type, result in results.items():
            print(f"{bubble_type:<15} {result.attractor_ticker:<8} {result.attractor_mass:>6.1f} {result.signal:<10}")
        
        print()
        for result in results.values():
            print(f"\n{result.bubble_type}:")
            print(f"  Narrative Density:    {result.narrative_density_score:>5.1f} (raw: {result.narrative_density_raw:.2f})")
            print(f"  Volume Concentration: {result.volume_concentration_score:>5.1f} (raw: {result.volume_concentration_raw:.1%})")
            print(f"  Price Momentum:       {result.price_momentum_score:>5.1f} (z: {result.price_momentum_zscore:+.2f})")
            print(f"  → {result.interpretation}")
    else:
        result = calc.calculate(args.bubble)
        
        print(f"Bubble Type: {result.bubble_type}")
        print(f"Attractor: {result.attractor_ticker}")
        print(f"Date: {result.date}")
        print()
        
        print(f"{'='*40}")
        print(f"ATTRACTOR MASS: {result.attractor_mass:.1f} ({result.signal})")
        print(f"{'='*40}")
        print()
        
        print("Components (0-100 scale):")
        print(f"  Narrative Density (30%):    {result.narrative_density_score:>5.1f}")
        print(f"    └─ Raw: {result.narrative_density_raw:.2f} ({result.narrative_density_raw*100:.0f}% attractor mentions)")
        print(f"  Volume Concentration (40%): {result.volume_concentration_score:>5.1f}")
        print(f"    └─ Raw: {result.volume_concentration_raw:.1%} of ${result.universe_dollar_volume_m:.0f}M universe")
        print(f"  Price Momentum (30%):       {result.price_momentum_score:>5.1f}")
        print(f"    └─ Z-score: {result.price_momentum_zscore:+.2f} ({result.attractor_ticker}: {result.attractor_return_20d:+.1f}% vs peers: {result.universe_avg_return_20d:+.1f}%)")
        print()
        
        print(f"Interpretation: {result.interpretation}")
        
        if args.save:
            if calc.save_to_supabase(result):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()