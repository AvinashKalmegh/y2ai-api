"""
hypergraph/evolution.py
Step 6: Track hyperedge evolution across time
"""

from typing import List, Set, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hyperedge_similarity(h1: Set[str], h2: Set[str]) -> float:
    """
    Jaccard similarity between two hyperedges
    
    Args:
        h1, h2: Sets of tickers
    
    Returns:
        Float 0-1 (1 = identical)
    """
    if not h1 or not h2:
        return 0.0
    intersection = len(h1 & h2)
    union = len(h1 | h2)
    return intersection / union if union > 0 else 0.0


def match_hyperedges(prev_edges: List[Set[str]], curr_edges: List[Set[str]], 
                     threshold: float = 0.5) -> Dict:
    """
    Match hyperedges between two time periods
    
    Args:
        prev_edges: Hyperedges from previous period
        curr_edges: Hyperedges from current period
        threshold: Similarity threshold for matching (default 0.5)
    
    Returns:
        Dict with births, deaths, stable, evolved
    """
    matched_prev = set()
    matched_curr = set()
    stable = []
    evolved = []
    
    # Find matches
    for i, curr in enumerate(curr_edges):
        best_match = -1
        best_sim = 0
        
        for j, prev in enumerate(prev_edges):
            if j in matched_prev:
                continue
            sim = hyperedge_similarity(prev, curr)
            if sim > best_sim and sim >= threshold:
                best_sim = sim
                best_match = j
        
        if best_match >= 0:
            matched_prev.add(best_match)
            matched_curr.add(i)
            
            if best_sim == 1.0:
                stable.append(curr)
            else:
                evolved.append({
                    'from': prev_edges[best_match],
                    'to': curr,
                    'similarity': best_sim
                })
    
    # Births = current edges not matched
    births = [curr_edges[i] for i in range(len(curr_edges)) if i not in matched_curr]
    
    # Deaths = previous edges not matched
    deaths = [prev_edges[j] for j in range(len(prev_edges)) if j not in matched_prev]
    
    return {
        'stable': stable,
        'evolved': evolved,
        'births': births,
        'deaths': deaths,
        'stable_count': len(stable),
        'evolved_count': len(evolved),
        'birth_count': len(births),
        'death_count': len(deaths)
    }


def calculate_stability(evolution_history: List[Dict], lookback: int = 5) -> float:
    """
    Calculate stability score from evolution history
    
    Args:
        evolution_history: List of evolution dicts
        lookback: How many periods to consider
    
    Returns:
        Float 0-1 (1 = perfectly stable)
    """
    if not evolution_history:
        return 0.0
    
    recent = evolution_history[-lookback:]
    
    total_edges = 0
    total_stable = 0
    
    for evo in recent:
        total_edges += evo['stable_count'] + evo['evolved_count'] + evo['birth_count']
        total_stable += evo['stable_count']
    
    return total_stable / total_edges if total_edges > 0 else 0.0


def classify_regime(
    hyperedge_count: int,
    prev_count: int,
    stability: float,
    cross_pillar_ratio: float
) -> str:
    """
    Classify hypergraph regime based on metrics
    
    Args:
        hyperedge_count: Current number of hyperedges
        prev_count: Previous period count
        stability: Stability score (0-1)
        cross_pillar_ratio: Ratio of cross-pillar hyperedges
    
    Returns:
        Regime string
    """
    growth_rate = (hyperedge_count - prev_count) / prev_count if prev_count > 0 else 0
    
    if growth_rate > 0.2 and stability < 0.4:
        return "FORMING"  # New coordination emerging rapidly
    elif growth_rate > 0.1 and stability >= 0.4:
        return "ACCELERATING"  # Coordination growing and stabilizing
    elif growth_rate < -0.1 and stability < 0.4:
        return "FRAGMENTING"  # Coordination breaking down
    elif abs(growth_rate) <= 0.1 and stability >= 0.5:
        return "STABLE"  # Mature coordination
    elif cross_pillar_ratio > 0.8:
        return "CONTAGION"  # High cross-pillar = systemic risk
    else:
        return "TRANSITIONING"


def print_evolution_report(evolution: Dict, date: str):
    """Print formatted evolution report"""
    print(f"\n{'='*60}")
    print(f"EVOLUTION REPORT: {date}")
    print(f"{'='*60}")
    print(f"Stable hyperedges: {evolution['stable_count']}")
    print(f"Evolved hyperedges: {evolution['evolved_count']}")
    print(f"New births: {evolution['birth_count']}")
    print(f"Deaths: {evolution['death_count']}")
    
    if evolution['births']:
        print(f"\n🆕 NEW HYPEREDGES:")
        for h in evolution['births'][:3]:
            print(f"    {sorted(h)}")
    
    if evolution['deaths']:
        print(f"\n💀 DISSOLVED HYPEREDGES:")
        for h in evolution['deaths'][:3]:
            print(f"    {sorted(h)}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from .data import get_returns_data, PILLAR_MAP
    from .correlation import compute_rolling_correlations, correlation_to_adjacency
    from .detection import find_hyperedges, analyze_hyperedges
    from datetime import datetime, timedelta
    
    # Get test data
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print("Fetching returns data...")
    returns = get_returns_data(start, end)
    tickers = list(returns.columns)
    
    print("Computing correlations...")
    correlations = compute_rolling_correlations(returns, window=20)
    
    # Get sorted dates
    dates = sorted(correlations.keys())
    
    print(f"\nTracking evolution across {len(dates)} days...")
    print(f"Threshold: 0.6")
    
    threshold = 0.6
    evolution_history = []
    prev_hyperedges = None
    
    for date in dates[-10:]:  # Last 10 days
        corr = correlations[date]
        adjacency = correlation_to_adjacency(corr, threshold=threshold)
        hyperedges = find_hyperedges(adjacency, tickers, min_size=3)
        
        if prev_hyperedges is not None:
            evolution = match_hyperedges(prev_hyperedges, hyperedges)
            evolution_history.append(evolution)
            print_evolution_report(evolution, date)
        
        prev_hyperedges = hyperedges
    
    # Calculate overall stability
    stability = calculate_stability(evolution_history)
    print(f"\n{'='*60}")
    print(f"OVERALL STABILITY: {stability:.2f}")
    print(f"{'='*60}")