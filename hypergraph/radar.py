"""
hypergraph/radar.py
Bubble Radar - Multi-bubble detection and cross-bubble analysis
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from .data import get_universe_returns, build_pillar_map
from .correlation import compute_rolling_correlations, correlation_to_adjacency
from .detection import find_hyperedges
from .evolution import match_hyperedges
from .metrics import calculate_daily_metrics
from .universes import ALL_UNIVERSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_universe(
    universe_id: str,
    days_back: int = 60,
    threshold: float = 0.6,
    window: int = 20
) -> Optional[Dict]:
    """
    Run hypergraph analysis on a single universe
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    try:
        # Get data
        returns, pillar_map, universe = get_universe_returns(universe_id, start_date, end_date)
        
        if len(returns) < window + 2:
            logger.warning(f"{universe_id}: Not enough data ({len(returns)} days)")
            return None
        
        tickers = list(returns.columns)
        
        # Compute correlations
        correlations = compute_rolling_correlations(returns, window=window)
        dates = sorted(correlations.keys())
        
        # Process each day
        prev_hyperedges = None
        prev_metrics = None
        latest_metrics = None
        
        for date in dates:
            corr = correlations[date]
            adjacency = correlation_to_adjacency(corr, threshold=threshold)
            hyperedges = find_hyperedges(adjacency, tickers, min_size=3)
            
            if prev_hyperedges is not None:
                evolution = match_hyperedges(prev_hyperedges, hyperedges)
                metrics = calculate_daily_metrics(hyperedges, evolution, pillar_map, prev_metrics)
                prev_metrics = metrics
                latest_metrics = metrics
                latest_metrics['date'] = date
            
            prev_hyperedges = hyperedges
        
        if latest_metrics:
            latest_metrics['universe_id'] = universe_id
            latest_metrics['universe_name'] = universe['name']
            latest_metrics['stage'] = universe['stage']
            latest_metrics['ticker_count'] = len(tickers)
            latest_metrics['pillar_count'] = len(universe['pillars'])
        
        return latest_metrics
        
    except Exception as e:
        logger.error(f"{universe_id}: Error - {e}")
        return None


def find_cross_bubble_stocks() -> Dict[str, List[str]]:
    """
    Find stocks that appear in multiple universes (bridge stocks between bubbles)
    """
    stock_bubbles = {}
    
    for universe_id, universe in ALL_UNIVERSES.items():
        for ticker in universe['tickers']:
            if ticker not in stock_bubbles:
                stock_bubbles[ticker] = []
            stock_bubbles[ticker].append(universe_id)
    
    # Filter to stocks in 2+ universes
    cross_bubble = {k: v for k, v in stock_bubbles.items() if len(v) > 1}
    return cross_bubble


def run_bubble_radar(
    universes: List[str] = None,
    days_back: int = 60,
    threshold: float = 0.6,
    window: int = 20
) -> Dict:
    """
    Run hypergraph analysis on all bubbles and compare
    """
    universes = universes or list(ALL_UNIVERSES.keys())
    
    print("=" * 70)
    print("                      BUBBLE RADAR")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Universes: {', '.join(universes)}")
    print("=" * 70)
    
    results = {}
    
    # Analyze each universe
    for universe_id in universes:
        print(f"\n{'─' * 70}")
        print(f"Analyzing: {universe_id.upper()}")
        print('─' * 70)
        
        metrics = analyze_universe(universe_id, days_back, threshold, window)
        
        if metrics:
            results[universe_id] = metrics
            
            # Print summary
            regime_emoji = {
                'CONTAGION': '🔴',
                'FRAGMENTING': '🟠',
                'TRANSITIONING': '🟡',
                'ACCELERATING': '📈',
                'STABLE': '🟢'
            }.get(metrics['regime'], '⚪')
            
            print(f"  Name:           {metrics['universe_name']}")
            print(f"  Stage:          {metrics['stage']}")
            print(f"  Regime:         {regime_emoji} {metrics['regime']}")
            print(f"  Contagion:      {metrics['contagion_score']:.1f}/100")
            print(f"  Stability:      {metrics['stability_score']:.1%}")
            print(f"  Cross-Pillar:   {metrics['cross_pillar_ratio']:.1%}")
            print(f"  Hyperedges:     {metrics['hyperedge_count']}")
            print(f"  Bridge Stocks:  {', '.join(metrics['bridge_stocks'][:3])}")
        else:
            print(f"  ⚠️  Analysis failed or insufficient data")
    
    # Cross-bubble analysis
    print(f"\n{'=' * 70}")
    print("CROSS-BUBBLE ANALYSIS")
    print('=' * 70)
    
    cross_stocks = find_cross_bubble_stocks()
    if cross_stocks:
        print("\nStocks appearing in multiple bubbles (contagion channels):")
        for ticker, bubbles in sorted(cross_stocks.items(), key=lambda x: -len(x[1])):
            print(f"  {ticker}: {' ↔ '.join(bubbles)}")
    else:
        print("\n  No cross-bubble stocks detected")
    
    # Radar Summary
    print(f"\n{'=' * 70}")
    print("RADAR SUMMARY")
    print('=' * 70)
    
    print(f"\n{'Bubble':<20} {'Stage':<18} {'Regime':<15} {'Contagion':<12} {'Risk'}")
    print('─' * 70)
    
    for universe_id, metrics in results.items():
        stage = metrics['stage']
        regime = metrics['regime']
        contagion = metrics['contagion_score']
        
        # Risk assessment
        if regime == 'CONTAGION' and contagion > 70:
            risk = '🔴 HIGH'
        elif regime in ['FRAGMENTING', 'CONTAGION'] or contagion > 50:
            risk = '🟠 ELEVATED'
        elif regime == 'ACCELERATING':
            risk = '🟡 WATCH'
        else:
            risk = '🟢 NORMAL'
        
        print(f"{metrics['universe_name']:<20} {stage:<18} {regime:<15} {contagion:<12.1f} {risk}")
    
    print('─' * 70)
    
    # Store results
    radar_result = {
        'timestamp': datetime.now().isoformat(),
        'universes': results,
        'cross_bubble_stocks': cross_stocks,
    }
    
    return radar_result


if __name__ == "__main__":
    run_bubble_radar()
