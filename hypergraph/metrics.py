"""
hypergraph/metrics.py
Step 7: Calculate daily metrics for Supabase
Step 8: Regime classification
"""

from typing import List, Set, Dict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_daily_metrics(
    hyperedges: List[Set[str]],
    evolution: Dict,
    pillar_map: Dict[str, str],
    prev_metrics: Dict = None
) -> Dict:
    """
    Calculate all hypergraph metrics for a single day
    
    Args:
        hyperedges: List of hyperedges for this day
        evolution: Evolution dict from match_hyperedges
        pillar_map: Ticker -> pillar mapping
        prev_metrics: Previous day's metrics (for growth rate)
    
    Returns:
        Dict with all metrics for Supabase
    """
    if not hyperedges:
        return {
            'hyperedge_count': 0,
            'avg_hyperedge_size': 0,
            'max_hyperedge_size': 0,
            'cross_pillar_count': 0,
            'cross_pillar_ratio': 0,
            'stability_score': 0,
            'growth_rate_1d': 0,
            'contagion_score': 0,
            'regime': 'UNKNOWN',
            'largest_hyperedge_tickers': [],
            'bridge_stocks': []
        }
    
    # Basic counts
    sizes = [len(h) for h in hyperedges]
    hyperedge_count = len(hyperedges)
    avg_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    
    # Cross-pillar analysis
    cross_pillar = []
    for h in hyperedges:
        pillars = set(pillar_map.get(t, 'Unknown') for t in h)
        if len(pillars) > 1:
            cross_pillar.append(h)
    
    cross_pillar_count = len(cross_pillar)
    cross_pillar_ratio = cross_pillar_count / hyperedge_count if hyperedge_count > 0 else 0
    
    # Stability from evolution
    if evolution:
        total = evolution['stable_count'] + evolution['evolved_count'] + evolution['birth_count']
        stability_score = evolution['stable_count'] / total if total > 0 else 0
    else:
        stability_score = 0
    
    # Growth rate
    if prev_metrics and prev_metrics.get('hyperedge_count', 0) > 0:
        growth_rate_1d = (hyperedge_count - prev_metrics['hyperedge_count']) / prev_metrics['hyperedge_count']
    else:
        growth_rate_1d = 0
    
    # Contagion score (0-100)
    # Based on: cross-pillar ratio, max size, and how many pillars are connected
    pillars_connected = set()
    for h in hyperedges:
        for t in h:
            pillars_connected.add(pillar_map.get(t, 'Unknown'))
    
    pillar_coverage = len(pillars_connected) / 6  # 6 pillars total
    size_factor = min(max_size / 20, 1)  # Normalize to 20-stock max
    contagion_score = round((cross_pillar_ratio * 40 + pillar_coverage * 30 + size_factor * 30), 1)
    
    # Bridge stocks (appear in most hyperedges)
    stock_counts = {}
    for h in hyperedges:
        for t in h:
            stock_counts[t] = stock_counts.get(t, 0) + 1
    
    sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
    bridge_stocks = [s[0] for s in sorted_stocks[:5]]  # Top 5
    
    # Largest hyperedge
    largest = max(hyperedges, key=len)
    
    # Regime classification
    regime = classify_regime(
        hyperedge_count=hyperedge_count,
        growth_rate=growth_rate_1d,
        stability=stability_score,
        cross_pillar_ratio=cross_pillar_ratio,
        contagion_score=contagion_score
    )
    
    return {
        'hyperedge_count': hyperedge_count,
        'avg_hyperedge_size': round(avg_size, 2),
        'max_hyperedge_size': max_size,
        'cross_pillar_count': cross_pillar_count,
        'cross_pillar_ratio': round(cross_pillar_ratio, 3),
        'stability_score': round(stability_score, 3),
        'growth_rate_1d': round(growth_rate_1d, 3),
        'contagion_score': contagion_score,
        'regime': regime,
        'largest_hyperedge_tickers': sorted(list(largest)),
        'bridge_stocks': bridge_stocks
    }


def classify_regime(
    hyperedge_count: int,
    growth_rate: float,
    stability: float,
    cross_pillar_ratio: float,
    contagion_score: float
) -> str:
    """
    Classify hypergraph regime
    
    Regimes:
    - STABLE: Mature, steady coordination
    - FORMING: New coordination emerging
    - ACCELERATING: Coordination growing fast
    - FRAGMENTING: Coordination breaking down
    - CONTAGION: High cross-pillar risk
    """
    
    # Priority: Contagion overrides others
    if contagion_score > 70 and cross_pillar_ratio > 0.8:
        return "CONTAGION"
    
    # Fragmentation
    if stability < 0.15 and growth_rate < 0:
        return "FRAGMENTING"
    
    # Forming (new clusters, unstable)
    if growth_rate > 0.15 and stability < 0.3:
        return "FORMING"
    
    # Accelerating (growing and stabilizing)
    if growth_rate > 0.1 and stability >= 0.3:
        return "ACCELERATING"
    
    # Stable
    if abs(growth_rate) <= 0.1 and stability >= 0.4:
        return "STABLE"
    
    return "TRANSITIONING"


def print_metrics_report(metrics: Dict, date: str):
    """Print formatted metrics report"""
    print(f"\n{'='*60}")
    print(f"HYPERGRAPH METRICS: {date}")
    print(f"{'='*60}")
    print(f"Regime: {metrics['regime']}")
    print(f"Hyperedge Count: {metrics['hyperedge_count']}")
    print(f"Avg Size: {metrics['avg_hyperedge_size']}")
    print(f"Max Size: {metrics['max_hyperedge_size']}")
    print(f"Cross-Pillar Ratio: {metrics['cross_pillar_ratio']:.1%}")
    print(f"Stability Score: {metrics['stability_score']:.1%}")
    print(f"Growth Rate (1D): {metrics['growth_rate_1d']:+.1%}")
    print(f"Contagion Score: {metrics['contagion_score']}/100")
    print(f"Bridge Stocks: {metrics['bridge_stocks']}")
    print(f"Largest Hyperedge: {metrics['largest_hyperedge_tickers']}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from .data import get_returns_data, PILLAR_MAP
    from .correlation import compute_rolling_correlations, correlation_to_adjacency
    from .detection import find_hyperedges
    from .evolution import match_hyperedges
    from datetime import datetime, timedelta
    
    # Get test data
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print("Fetching returns data...")
    returns = get_returns_data(start, end)
    tickers = list(returns.columns)
    
    print("Computing correlations...")
    correlations = compute_rolling_correlations(returns, window=20)
    
    dates = sorted(correlations.keys())
    threshold = 0.6
    
    print(f"\nCalculating metrics for last 5 days...")
    
    prev_hyperedges = None
    prev_metrics = None
    
    # Process last 5 days
    for date in dates[-6:]:
        corr = correlations[date]
        adjacency = correlation_to_adjacency(corr, threshold=threshold)
        hyperedges = find_hyperedges(adjacency, tickers, min_size=3)
        
        if prev_hyperedges is not None:
            evolution = match_hyperedges(prev_hyperedges, hyperedges)
            metrics = calculate_daily_metrics(hyperedges, evolution, PILLAR_MAP, prev_metrics)
            print_metrics_report(metrics, date)
            prev_metrics = metrics
        
        prev_hyperedges = hyperedges