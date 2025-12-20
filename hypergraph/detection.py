"""
hypergraph/detection.py
Step 4: Detect hyperedges (cliques) from adjacency matrix
Step 5: Map hyperedges to pillars
"""

import numpy as np
import networkx as nx
from typing import List, Set, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_hyperedges(adjacency: np.ndarray, tickers: list, min_size: int = 3) -> List[Set[str]]:
    """
    Find all maximal cliques (hyperedges) of size >= min_size
    
    A clique is a group where every stock is connected to every other.
    This is the core hyperedge detection.
    
    Args:
        adjacency: Binary adjacency matrix
        tickers: List of ticker symbols
        min_size: Minimum clique size (default 3)
    
    Returns:
        List of sets, e.g. [{'NVDA', 'AMD', 'AVGO'}, {'MSFT', 'GOOGL', 'AMZN'}]
    """
    # Build networkx graph from adjacency matrix
    G = nx.Graph()
    G.add_nodes_from(tickers)
    
    # Add edges where adjacency = 1
    n = len(tickers)
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j] == 1:
                G.add_edge(tickers[i], tickers[j])
    
    # Find all maximal cliques
    all_cliques = list(nx.find_cliques(G))
    
    # Filter by minimum size
    hyperedges = [set(c) for c in all_cliques if len(c) >= min_size]
    
    # Sort by size (largest first)
    hyperedges.sort(key=len, reverse=True)
    
    logger.info(f"Found {len(hyperedges)} hyperedges (size >= {min_size})")
    
    return hyperedges


def get_hyperedge_pillars(hyperedge: Set[str], pillar_map: Dict[str, str]) -> Set[str]:
    """
    Get which pillars a hyperedge spans
    
    Args:
        hyperedge: Set of tickers
        pillar_map: Dict mapping ticker -> pillar name
    
    Returns:
        Set of pillar names
    """
    pillars = set()
    for ticker in hyperedge:
        if ticker in pillar_map:
            pillars.add(pillar_map[ticker])
    return pillars


def is_cross_pillar(hyperedge: Set[str], pillar_map: Dict[str, str]) -> bool:
    """Check if hyperedge spans multiple pillars"""
    pillars = get_hyperedge_pillars(hyperedge, pillar_map)
    return len(pillars) > 1


def analyze_hyperedges(hyperedges: List[Set[str]], pillar_map: Dict[str, str]) -> Dict:
    """
    Analyze hyperedge structure
    
    Args:
        hyperedges: List of hyperedges (sets of tickers)
        pillar_map: Dict mapping ticker -> pillar name
    
    Returns:
        Dict with analysis results
    """
    if not hyperedges:
        return {
            'count': 0,
            'sizes': [],
            'avg_size': 0,
            'max_size': 0,
            'cross_pillar_count': 0,
            'largest': set(),
            'largest_pillars': set()
        }
    
    sizes = [len(h) for h in hyperedges]
    cross_pillar = [h for h in hyperedges if is_cross_pillar(h, pillar_map)]
    largest = hyperedges[0]  # Already sorted by size
    
    return {
        'count': len(hyperedges),
        'sizes': sizes,
        'avg_size': round(sum(sizes) / len(sizes), 2),
        'max_size': max(sizes),
        'cross_pillar_count': len(cross_pillar),
        'largest': largest,
        'largest_pillars': get_hyperedge_pillars(largest, pillar_map)
    }


def print_hyperedge_report(hyperedges: List[Set[str]], pillar_map: Dict[str, str]):
    """Print formatted hyperedge report"""
    analysis = analyze_hyperedges(hyperedges, pillar_map)
    
    print(f"\n{'='*60}")
    print("HYPEREDGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total hyperedges (size >= 3): {analysis['count']}")
    print(f"Average size: {analysis['avg_size']}")
    print(f"Max size: {analysis['max_size']}")
    print(f"Cross-pillar hyperedges: {analysis['cross_pillar_count']}")
    
    if hyperedges:
        print(f"\n{'='*60}")
        print("LARGEST HYPEREDGES")
        print(f"{'='*60}")
        
        for i, h in enumerate(hyperedges[:5]):  # Top 5
            pillars = get_hyperedge_pillars(h, pillar_map)
            cross = "⚠️ CROSS-PILLAR" if len(pillars) > 1 else ""
            print(f"\n#{i+1} Size {len(h)}: {sorted(h)} {cross}")
            print(f"    Pillars: {pillars}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from .data import get_returns_data, PILLAR_MAP
    from .correlation import compute_rolling_correlations, correlation_to_adjacency
    from datetime import datetime, timedelta
    
    # Get test data
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print("Fetching returns data...")
    returns = get_returns_data(start, end)
    tickers = list(returns.columns)
    
    print("Computing correlations...")
    correlations = compute_rolling_correlations(returns, window=20)
    
    # Get latest
    latest_date = max(correlations.keys())
    latest_corr = correlations[latest_date]
    
    print(f"\nAnalyzing {latest_date}...")
    
    # Test different thresholds
    for threshold in [0.7, 0.6, 0.5]:
        print(f"\n{'='*60}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*60}")
        
        adjacency = correlation_to_adjacency(latest_corr, threshold=threshold)
        hyperedges = find_hyperedges(adjacency, tickers, min_size=3)
        print_hyperedge_report(hyperedges, PILLAR_MAP)