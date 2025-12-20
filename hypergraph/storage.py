"""
hypergraph/storage.py
Step 9: Store metrics to Supabase
"""

import os
from datetime import datetime
from typing import Dict, Optional
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supabase_client():
    """Get Supabase client"""
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


def store_metrics(date: str, metrics: Dict, threshold: float = 0.6, window: int = 20) -> bool:
    """
    Store hypergraph metrics to Supabase
    
    Args:
        date: Date string 'YYYY-MM-DD'
        metrics: Metrics dict from calculate_daily_metrics
        threshold: Correlation threshold used
        window: Rolling window size used
    
    Returns:
        True if successful
    """
    client = get_supabase_client()
    if not client:
        logger.warning("No Supabase client, skipping storage")
        return False
    
    try:
        record = {
            'date': date,
            'hyperedge_count': metrics['hyperedge_count'],
            'avg_hyperedge_size': metrics['avg_hyperedge_size'],
            'max_hyperedge_size': metrics['max_hyperedge_size'],
            'cross_pillar_count': metrics['cross_pillar_count'],
            'cross_pillar_ratio': metrics['cross_pillar_ratio'],
            'stability_score': metrics['stability_score'],
            'growth_rate_1d': metrics['growth_rate_1d'],
            'contagion_score': metrics['contagion_score'],
            'regime': metrics['regime'],
            'largest_hyperedge_tickers': metrics['largest_hyperedge_tickers'],
            'bridge_stocks': metrics['bridge_stocks'],
            'threshold_used': threshold,
            'window_size': window,
            'calculated_at': datetime.utcnow().isoformat()
        }
        
        client.table('hypergraph_signals').upsert(
            record,
            on_conflict='date'
        ).execute()
        
        logger.info(f"Stored hypergraph metrics for {date}")
        return True
        
    except Exception as e:
        logger.error(f"Storage error: {e}")
        return False


def get_latest_metrics() -> Optional[Dict]:
    """Get most recent hypergraph metrics"""
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table('hypergraph_signals')\
            .select('*')\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return None