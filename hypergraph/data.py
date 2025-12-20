"""
hypergraph/data.py
Step 1: Data Ingestion using yfinance
Supports multiple bubble universes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default: AI Infrastructure (for backward compatibility)
DEFAULT_TICKERS = [
    # Infrastructure & Energy (16)
    'NVDA', 'AMD', 'AVGO', 'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC',
    'MRVL', 'MU', 'QCOM', 'ARM', 'SMCI', 'DELL', 'VRT', 'CEG',
    # Enterprise Adoption (13)
    'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'CRM', 'NOW', 
    'SNOW', 'PLTR', 'MDB', 'DDOG', 'NET', 'CRWD',
    # Financial & Market (4)
    'GS', 'MS', 'BLK', 'COIN',
    # Macro & Policy (4)
    'LMT', 'RTX', 'GD', 'NOC',
    # Productivity & Labor (3)
    'ADBE', 'INTU', 'WDAY',
    # Demand Dynamics (3)
    'TSLA', 'UBER', 'ABNB',
]

DEFAULT_PILLAR_MAP = {
    # Infrastructure & Energy (16)
    'NVDA': 'Infrastructure & Energy', 'AMD': 'Infrastructure & Energy',
    'AVGO': 'Infrastructure & Energy', 'TSM': 'Infrastructure & Energy',
    'ASML': 'Infrastructure & Energy', 'AMAT': 'Infrastructure & Energy',
    'LRCX': 'Infrastructure & Energy', 'KLAC': 'Infrastructure & Energy',
    'MRVL': 'Infrastructure & Energy', 'MU': 'Infrastructure & Energy',
    'QCOM': 'Infrastructure & Energy', 'ARM': 'Infrastructure & Energy',
    'SMCI': 'Infrastructure & Energy', 'DELL': 'Infrastructure & Energy',
    'VRT': 'Infrastructure & Energy', 'CEG': 'Infrastructure & Energy',
    # Enterprise Adoption (13)
    'MSFT': 'Enterprise Adoption', 'GOOGL': 'Enterprise Adoption',
    'AMZN': 'Enterprise Adoption', 'META': 'Enterprise Adoption',
    'ORCL': 'Enterprise Adoption', 'CRM': 'Enterprise Adoption',
    'NOW': 'Enterprise Adoption', 'SNOW': 'Enterprise Adoption',
    'PLTR': 'Enterprise Adoption', 'MDB': 'Enterprise Adoption',
    'DDOG': 'Enterprise Adoption', 'NET': 'Enterprise Adoption',
    'CRWD': 'Enterprise Adoption',
    # Financial & Market (4)
    'GS': 'Financial & Market', 'MS': 'Financial & Market',
    'BLK': 'Financial & Market', 'COIN': 'Financial & Market',
    # Macro & Policy (4)
    'LMT': 'Macro & Policy', 'RTX': 'Macro & Policy',
    'GD': 'Macro & Policy', 'NOC': 'Macro & Policy',
    # Productivity & Labor (3)
    'ADBE': 'Productivity & Labor', 'INTU': 'Productivity & Labor',
    'WDAY': 'Productivity & Labor',
    # Demand Dynamics (3)
    'TSLA': 'Demand Dynamics', 'UBER': 'Demand Dynamics',
    'ABNB': 'Demand Dynamics',
}

# Backward compatibility aliases
TICKERS = DEFAULT_TICKERS
PILLAR_MAP = DEFAULT_PILLAR_MAP


def build_pillar_map(universe: Dict) -> Dict[str, str]:
    """
    Build ticker -> pillar mapping from universe definition
    
    Args:
        universe: Universe dict with 'pillars' key
    
    Returns:
        Dict mapping ticker to pillar name
    """
    pillar_map = {}
    for pillar_name, tickers in universe['pillars'].items():
        for ticker in tickers:
            pillar_map[ticker] = pillar_name
    return pillar_map


def get_universe(universe_id: str = 'ai_infra') -> Dict:
    """
    Load a universe by ID
    
    Args:
        universe_id: 'ai_infra', 'crypto', 'nuclear', etc.
    
    Returns:
        Universe dict with tickers, pillars, etc.
    """
    from .universes import ALL_UNIVERSES
    
    if universe_id not in ALL_UNIVERSES:
        available = list(ALL_UNIVERSES.keys())
        raise ValueError(f"Unknown universe: {universe_id}. Available: {available}")
    
    return ALL_UNIVERSES[universe_id]


def fetch_price_data(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """
    Fetch daily close prices for all tickers using yfinance
    
    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        tickers: List of tickers (default: DEFAULT_TICKERS)
    
    Returns:
        DataFrame with dates as index, tickers as columns, close prices as values
    """
    import yfinance as yf
    
    tickers = tickers or DEFAULT_TICKERS
    logger.info(f"Fetching price data for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Download all at once (faster)
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False
    )
    
    # Extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            prices = data
    else:
        prices = data  # Single ticker case
    
    # Ensure DataFrame format
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    # Filter to only requested tickers that have data
    available = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    
    if missing:
        logger.warning(f"Missing data for: {missing}")
    
    prices = prices[available]
    
    logger.info(f"Fetched {len(prices)} trading days")
    logger.info(f"Tickers with data: {len(prices.columns)}")
    
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from prices
    
    Args:
        prices: DataFrame of close prices (dates × tickers)
    
    Returns:
        DataFrame of daily returns (dates × tickers)
    """
    returns = prices.pct_change(fill_method=None).dropna()
    return returns


def get_returns_data(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """
    Main function: fetch prices and compute returns
    
    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        tickers: List of tickers (default: DEFAULT_TICKERS)
    
    Returns:
        DataFrame of daily returns
    """
    prices = fetch_price_data(start_date, end_date, tickers)
    returns = compute_returns(prices)
    
    logger.info(f"Returns shape: {returns.shape}")
    logger.info(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    return returns


def get_universe_returns(universe_id: str, start_date: str, end_date: str) -> tuple:
    """
    Get returns data for a specific universe
    
    Args:
        universe_id: 'ai_infra', 'crypto', 'nuclear'
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
    
    Returns:
        Tuple of (returns DataFrame, pillar_map dict, universe dict)
    """
    universe = get_universe(universe_id)
    tickers = universe['tickers']
    pillar_map = build_pillar_map(universe)
    
    logger.info(f"Loading universe: {universe['name']}")
    logger.info(f"Stage: {universe['stage']}")
    
    returns = get_returns_data(start_date, end_date, tickers)
    
    return returns, pillar_map, universe


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with last 60 days
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("TESTING MULTIPLE UNIVERSES")
    print("=" * 60)
    
    for universe_id in ['ai_infra', 'crypto', 'nuclear']:
        print(f"\n--- {universe_id.upper()} ---")
        try:
            returns, pillar_map, universe = get_universe_returns(universe_id, start, end)
            print(f"  Name: {universe['name']}")
            print(f"  Stage: {universe['stage']}")
            print(f"  Tickers: {len(returns.columns)}")
            print(f"  Pillars: {len(universe['pillars'])}")
            print(f"  Days: {len(returns)}")
        except Exception as e:
            print(f"  Error: {e}")

