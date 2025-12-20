"""
hypergraph/pipeline.py
Main orchestration pipeline
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

from .data import get_returns_data, PILLAR_MAP
from .correlation import compute_rolling_correlations, correlation_to_adjacency
from .detection import find_hyperedges
from .evolution import match_hyperedges
from .metrics import calculate_daily_metrics, print_metrics_report
from .storage import store_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_hypergraph_analysis(
    days_back: int = 60,
    threshold: float = 0.6,
    window: int = 20,
    store_to_supabase: bool = True,
    print_output: bool = True
) -> dict:
    """
    Run complete hypergraph analysis pipeline
    """
    logger.info("=" * 60)
    logger.info("HYPERGRAPH ANALYSIS PIPELINE")
    logger.info("=" * 60)
    
    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Threshold: {threshold}, Window: {window}")
    
    # Step 1: Get returns
    logger.info("\n[1/5] Fetching price data...")
    returns = get_returns_data(start_date, end_date)
    tickers = list(returns.columns)
    logger.info(f"Got {len(tickers)} tickers")
    
    # Step 2: Compute correlations
    logger.info("\n[2/5] Computing correlation matrices...")
    correlations = compute_rolling_correlations(returns, window=window)
    
    dates = sorted(correlations.keys())
    logger.info(f"Generated {len(dates)} correlation matrices")
    
    # Step 3-5: Process each day
    logger.info("\n[3/5] Detecting hyperedges...")
    logger.info("[4/5] Tracking evolution...")
    logger.info("[5/5] Calculating metrics...")
    
    prev_hyperedges = None
    prev_metrics = None
    latest_metrics = None
    
    for date in dates:
        corr = correlations[date]
        adjacency = correlation_to_adjacency(corr, threshold=threshold)
        hyperedges = find_hyperedges(adjacency, tickers, min_size=3)
        
        if prev_hyperedges is not None:
            evolution = match_hyperedges(prev_hyperedges, hyperedges)
            metrics = calculate_daily_metrics(hyperedges, evolution, PILLAR_MAP, prev_metrics)
            
            # Store to Supabase
            if store_to_supabase:
                store_metrics(date, metrics, threshold, window)
            
            prev_metrics = metrics
            latest_metrics = metrics
            latest_metrics['date'] = date
        
        prev_hyperedges = hyperedges
    
    # Print final report
    if print_output and latest_metrics:
        print("\n" + "=" * 60)
        print("LATEST HYPERGRAPH STATUS")
        print("=" * 60)
        print_metrics_report(latest_metrics, latest_metrics['date'])
        
        # Risk assessment
        print("\n" + "-" * 60)
        print("RISK ASSESSMENT")
        print("-" * 60)
        
        regime = latest_metrics['regime']
        contagion = latest_metrics['contagion_score']
        stability = latest_metrics['stability_score']
        
        if regime == "CONTAGION":
            print("⚠️  CONTAGION REGIME - Cross-pillar risk elevated")
            print("    Diversification within AI theme failing")
        elif regime == "FRAGMENTING":
            print("📉 FRAGMENTING - Coordination breaking down")
            print("    Watch for continued MCI deterioration")
        elif regime == "ACCELERATING":
            print("📈 ACCELERATING - Coordination building")
            print("    Watch for bubble formation signals")
        elif regime == "STABLE":
            print("✅ STABLE - Mature coordination")
            print("    Normal market structure")
        else:
            print(f"⚡ {regime} - Structure in flux")
        
        print(f"\n    Contagion Score: {contagion}/100 {'🔴' if contagion > 70 else '🟡' if contagion > 50 else '🟢'}")
        print(f"    Stability: {stability:.1%} {'🔴' if stability < 0.1 else '🟡' if stability < 0.3 else '🟢'}")
        print(f"    Bridge Stocks: {', '.join(latest_metrics['bridge_stocks'][:3])}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return latest_metrics


def run_daily():
    """Daily update function - call from scheduler"""
    return run_hypergraph_analysis(
        days_back=60,
        threshold=0.6,
        window=20,
        store_to_supabase=True,
        print_output=False
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hypergraph Analysis Pipeline')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--threshold', type=float, default=0.6, help='Correlation threshold')
    parser.add_argument('--window', type=int, default=20, help='Rolling window size')
    parser.add_argument('--no-store', action='store_true', help='Skip Supabase storage')
    
    args = parser.parse_args()
    
    run_hypergraph_analysis(
        days_back=args.days,
        threshold=args.threshold,
        window=args.window,
        store_to_supabase=not args.no_store
    )
