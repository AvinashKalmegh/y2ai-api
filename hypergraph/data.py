"""
hypergraph/data.py
Step 1: Data Ingestion - 4-PILLAR VERSION
Supports multiple bubble universes

UPDATED 2026-02-02: 4-Pillar Configuration (52 tickers)
Pillars:
- Infrastructure & Energy (22 stocks)
- Enterprise Demand (17 stocks)
- Consumer Demand (7 stocks)
- Capital (6 stocks)

Data source: Supabase price_history (synced from TwelveData)
Fallback: yfinance
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 4-PILLAR UNIVERSE (GS ALIGNED)
# UPDATED 2026-02-02: Synced with Google Sheets Universe tab
# =============================================================================

# All 52 tickers
DEFAULT_TICKERS = [
    # Infrastructure & Energy (22)
    'EQIX', 'DLR', 'TSM', 'ASML', 'NVDA', 'AMD', 'MU', 'AVGO',
    'VRT', 'CEG', 'NRG', 'LRCX', 'AMAT', 'SMCI', 'JKS', 'RUN',
    'FSLR', 'ENPH', 'INTC', 'NXPI', 'QCOM', 'ON',
    # Enterprise Demand (17)
    'MSFT', 'AMZN', 'GOOGL', 'SNOW', 'PLTR', 'ADBE', 'ORCL',
    'MDB', 'DDOG', 'ZS', 'NET', 'PANW', 'CRWD', 'CRM', 'NOW',
    'WDAY', 'ADP',
    # Consumer Demand (7)
    'TSLA', 'SHOP', 'UBER', 'ABNB', 'META', 'NFLX', 'SPOT',
    # Capital (6)
    'GS', 'MS', 'BX', 'KKR', 'APO', 'ARES',
]

# Pillar mapping (ticker -> pillar name)
DEFAULT_PILLAR_MAP = {
    # Infrastructure & Energy (22)
    'EQIX': 'Infrastructure & Energy', 'DLR': 'Infrastructure & Energy',
    'TSM': 'Infrastructure & Energy', 'ASML': 'Infrastructure & Energy',
    'NVDA': 'Infrastructure & Energy', 'AMD': 'Infrastructure & Energy',
    'MU': 'Infrastructure & Energy', 'AVGO': 'Infrastructure & Energy',
    'VRT': 'Infrastructure & Energy', 'CEG': 'Infrastructure & Energy',
    'NRG': 'Infrastructure & Energy', 'LRCX': 'Infrastructure & Energy',
    'AMAT': 'Infrastructure & Energy', 'SMCI': 'Infrastructure & Energy',
    'JKS': 'Infrastructure & Energy', 'RUN': 'Infrastructure & Energy',
    'FSLR': 'Infrastructure & Energy', 'ENPH': 'Infrastructure & Energy',
    'INTC': 'Infrastructure & Energy', 'NXPI': 'Infrastructure & Energy',
    'QCOM': 'Infrastructure & Energy', 'ON': 'Infrastructure & Energy',
    # Enterprise Demand (17)
    'MSFT': 'Enterprise Demand', 'AMZN': 'Enterprise Demand',
    'GOOGL': 'Enterprise Demand', 'SNOW': 'Enterprise Demand',
    'PLTR': 'Enterprise Demand', 'ADBE': 'Enterprise Demand',
    'ORCL': 'Enterprise Demand', 'MDB': 'Enterprise Demand',
    'DDOG': 'Enterprise Demand', 'ZS': 'Enterprise Demand',
    'NET': 'Enterprise Demand', 'PANW': 'Enterprise Demand',
    'CRWD': 'Enterprise Demand', 'CRM': 'Enterprise Demand',
    'NOW': 'Enterprise Demand', 'WDAY': 'Enterprise Demand',
    'ADP': 'Enterprise Demand',
    # Consumer Demand (7)
    'TSLA': 'Consumer Demand', 'SHOP': 'Consumer Demand',
    'UBER': 'Consumer Demand', 'ABNB': 'Consumer Demand',
    'META': 'Consumer Demand', 'NFLX': 'Consumer Demand',
    'SPOT': 'Consumer Demand',
    # Capital (6)
    'GS': 'Capital', 'MS': 'Capital',
    'BX': 'Capital', 'KKR': 'Capital',
    'APO': 'Capital', 'ARES': 'Capital',
}

# Backward compatibility aliases
TICKERS = DEFAULT_TICKERS
PILLAR_MAP = DEFAULT_PILLAR_MAP


# =============================================================================
# SUPABASE CLIENT
# =============================================================================

def get_supabase_client():
    """Get Supabase client."""
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY not set")
            return None
        
        return create_client(url, key)
    except ImportError:
        logger.warning("supabase package not installed")
        return None


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data_supabase(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """
    Fetch daily close prices from Supabase price_history table.
    
    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        tickers: List of tickers (default: DEFAULT_TICKERS)
    
    Returns:
        DataFrame with dates as index, tickers as columns, close prices as values
    """
    tickers = tickers or DEFAULT_TICKERS
    logger.info(f"Fetching price data from Supabase for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    client = get_supabase_client()
    if not client:
        logger.warning("No Supabase client, falling back to yfinance")
        return fetch_price_data_yfinance(start_date, end_date, tickers)
    
    try:
        # Paginate to handle Supabase 1000-row limit
        all_data = []
        batch_size = 1000
        offset = 0
        
        while True:
            response = client.table("price_history") \
                .select("date, ticker, close") \
                .in_("ticker", tickers) \
                .gte("date", start_date) \
                .lte("date", end_date) \
                .order("date", desc=True) \
                .range(offset, offset + batch_size - 1) \
                .execute()
            
            if not response.data:
                break
                
            all_data.extend(response.data)
            
            if len(response.data) < batch_size:
                break
                
            offset += batch_size
        
        if not all_data:
            logger.warning("No data from Supabase, falling back to yfinance")
            return fetch_price_data_yfinance(start_date, end_date, tickers)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        
        # Pivot to get tickers as columns
        prices = df.pivot_table(index="date", columns="ticker", values="close")
        prices = prices.sort_index()
        
        # Filter to only requested tickers that have data
        available = [t for t in tickers if t in prices.columns]
        missing = [t for t in tickers if t not in prices.columns]
        
        if missing:
            logger.warning(f"Missing data for: {missing}")
        
        prices = prices[available]
        
        logger.info(f"Fetched {len(prices)} trading days from Supabase")
        logger.info(f"Tickers with data: {len(prices.columns)}")
        
        return prices
        
    except Exception as e:
        logger.error(f"Supabase fetch error: {e}")
        logger.warning("Falling back to yfinance")
        return fetch_price_data_yfinance(start_date, end_date, tickers)


def fetch_price_data_yfinance(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """
    Fetch daily close prices using yfinance (fallback).
    
    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        tickers: List of tickers (default: DEFAULT_TICKERS)
    
    Returns:
        DataFrame with dates as index, tickers as columns, close prices as values
    """
    import yfinance as yf
    
    tickers = tickers or DEFAULT_TICKERS
    logger.info(f"Fetching price data from yfinance for {len(tickers)} tickers...")
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
    
    logger.info(f"Fetched {len(prices)} trading days from yfinance")
    logger.info(f"Tickers with data: {len(prices.columns)}")
    
    return prices


def fetch_price_data(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """
    Main function: fetch prices (Supabase first, yfinance fallback).
    
    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        tickers: List of tickers (default: DEFAULT_TICKERS)
    
    Returns:
        DataFrame with dates as index, tickers as columns, close prices as values
    """
    return fetch_price_data_supabase(start_date, end_date, tickers)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from prices.
    
    Args:
        prices: DataFrame of close prices (dates × tickers)
    
    Returns:
        DataFrame of daily returns (dates × tickers)
    """
    returns = prices.pct_change(fill_method=None).dropna()
    return returns


def get_returns_data(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """
    Main function: fetch prices and compute returns.
    
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


# =============================================================================
# UNIVERSE SUPPORT (for multi-bubble analysis)
# =============================================================================

def build_pillar_map(universe: Dict) -> Dict[str, str]:
    """Build ticker -> pillar mapping from universe definition."""
    pillar_map = {}
    for pillar_name, tickers in universe['pillars'].items():
        for ticker in tickers:
            pillar_map[ticker] = pillar_name
    return pillar_map


def get_universe(universe_id: str = 'ai_infra') -> Dict:
    """
    Load a universe by ID.
    
    Args:
        universe_id: 'ai_infra', 'crypto', 'nuclear', etc.
    
    Returns:
        Universe dict with tickers, pillars, etc.
    """
    # Default AI Infrastructure universe (4-pillar)
    if universe_id == 'ai_infra':
        return {
            'name': 'AI Infrastructure',
            'stage': 'MATURE',
            'tickers': DEFAULT_TICKERS,
            'pillars': {
                'Infrastructure & Energy': [
                    'EQIX', 'DLR', 'TSM', 'ASML', 'NVDA', 'AMD', 'MU', 'AVGO',
                    'VRT', 'CEG', 'NRG', 'LRCX', 'AMAT', 'SMCI', 'JKS', 'RUN',
                    'FSLR', 'ENPH', 'INTC', 'NXPI', 'QCOM', 'ON'
                ],
                'Enterprise Demand': [
                    'MSFT', 'AMZN', 'GOOGL', 'SNOW', 'PLTR', 'ADBE', 'ORCL',
                    'MDB', 'DDOG', 'ZS', 'NET', 'PANW', 'CRWD', 'CRM', 'NOW',
                    'WDAY', 'ADP'
                ],
                'Consumer Demand': [
                    'TSLA', 'SHOP', 'UBER', 'ABNB', 'META', 'NFLX', 'SPOT'
                ],
                'Capital': [
                    'GS', 'MS', 'BX', 'KKR', 'APO', 'ARES'
                ]
            }
        }
    
    # Try loading from universes module
    try:
        from .universes import ALL_UNIVERSES
        if universe_id in ALL_UNIVERSES:
            return ALL_UNIVERSES[universe_id]
    except ImportError:
        pass
    
    raise ValueError(f"Unknown universe: {universe_id}")


def get_universe_returns(universe_id: str, start_date: str, end_date: str) -> tuple:
    """
    Get returns data for a specific universe.
    
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
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("HYPERGRAPH DATA MODULE (4-PILLAR VERSION)")
    print("=" * 60)
    print(f"Date range: {start} to {end}")
    print(f"Total tickers: {len(DEFAULT_TICKERS)}")
    print()
    
    # Count by pillar
    pillar_counts = {}
    for ticker, pillar in PILLAR_MAP.items():
        pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
    
    print("Pillars:")
    for pillar, count in pillar_counts.items():
        print(f"  {pillar}: {count} tickers")
    
    print()
    print("Testing data fetch...")
    
    returns = get_returns_data(start, end)
    print(f"\nReturns shape: {returns.shape}")
    print(f"Tickers: {list(returns.columns)[:10]}...")