"""
hypergraph/correlation.py
Step 2: Rolling correlation matrix
Step 3: Threshold to adjacency matrix
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rolling_correlations(returns: pd.DataFrame, window: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Compute rolling correlation matrix for each date
    
    Args:
        returns: DataFrame of daily returns (dates × tickers)
        window: Rolling window size (default 20 trading days)
    
    Returns:
        Dict of {date_string: correlation_matrix}
    """
    correlations = {}
    dates = returns.index[window-1:]  # Need 'window' days of history
    
    logger.info(f"Computing {len(dates)} correlation matrices (window={window})...")
    
    for i, date in enumerate(dates):
        # Get window of returns ending at this date
        start_idx = i
        end_idx = i + window
        window_returns = returns.iloc[start_idx:end_idx]
        
        # Compute correlation matrix
        corr_matrix = window_returns.corr()
        
        # Store with date string key
        date_str = date.strftime('%Y-%m-%d')
        correlations[date_str] = corr_matrix
    
    logger.info(f"Computed {len(correlations)} correlation matrices")
    return correlations


def correlation_to_adjacency(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
    """
    Convert correlation matrix to binary adjacency matrix
    
    Args:
        corr_matrix: Correlation matrix (tickers × tickers)
        threshold: Correlation threshold for "connected" (default 0.7)
    
    Returns:
        Binary numpy array (1 = connected, 0 = not connected)
    """
    # Absolute correlation above threshold
    adjacency = (corr_matrix.abs() >= threshold).astype(int)
    
    # Remove self-connections (diagonal)
    np.fill_diagonal(adjacency.values, 0)
    
    return adjacency.values


def get_adjacency_stats(adjacency: np.ndarray, tickers: list) -> dict:
    """
    Get statistics about the adjacency matrix
    
    Args:
        adjacency: Binary adjacency matrix
        tickers: List of ticker symbols
    
    Returns:
        Dict with stats
    """
    n_tickers = len(tickers)
    n_edges = adjacency.sum() // 2  # Undirected, so divide by 2
    max_edges = n_tickers * (n_tickers - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0
    
    # Degree per ticker (how many connections)
    degrees = adjacency.sum(axis=1)
    
    return {
        'n_tickers': n_tickers,
        'n_edges': n_edges,
        'max_edges': max_edges,
        'density': round(density, 3),
        'avg_degree': round(degrees.mean(), 1),
        'max_degree': int(degrees.max()),
        'most_connected': tickers[degrees.argmax()]
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from .data import get_returns_data
    from datetime import datetime, timedelta
    
    # Get test data
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print("Fetching returns data...")
    returns = get_returns_data(start, end)
    
    print("\nComputing correlations...")
    correlations = compute_rolling_correlations(returns, window=20)
    
    # Get latest correlation matrix
    latest_date = max(correlations.keys())
    latest_corr = correlations[latest_date]
    
    print(f"\nLatest correlation matrix ({latest_date}):")
    print(f"  Shape: {latest_corr.shape}")
    print(f"  Sample (NVDA correlations):")
    nvda_corr = latest_corr['NVDA'].sort_values(ascending=False)
    print(nvda_corr.head(6))
    
    # Convert to adjacency
    print("\nConverting to adjacency (threshold=0.7)...")
    adjacency = correlation_to_adjacency(latest_corr, threshold=0.7)
    
    stats = get_adjacency_stats(adjacency, list(returns.columns))
    print(f"\nAdjacency stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")