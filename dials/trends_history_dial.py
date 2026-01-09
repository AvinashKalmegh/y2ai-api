"""
TRENDS HISTORY DIAL
===================
Log daily snapshots of all key metrics for historical analysis.

Tracked Metrics:
- AMRI + Enhanced AMRI + Break Probability
- MCI + MCI Regime
- Bubble Index
- Cluster Count
- Correlations (20D)
- Breadth (20D)
- VIX
- Credit Spreads
- Infrastructure Breadth
- Enterprise Breadth
- Thesis Balance
- Hypergraph (Contagion, Stability, Cross-Pillar)

Features:
- Daily snapshots
- 7-day change calculations
- Regime distribution tracking
- Historical trend analysis
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DailyMetrics:
    """Daily metrics snapshot."""
    date: str
    # AMRI
    amri: float
    amri_regime: str
    enhanced_amri: float
    break_probability: float
    # MCI
    mci: float
    mci_regime: str
    # Market
    bubble_index: float
    clusters: int
    correlations_20d: float
    breadth_20d: float
    vix: float
    credit_spreads: float
    # Pillar breadth
    infra_breadth: float
    enterprise_breadth: float
    # Signals
    thesis_balance: float
    # Hypergraph
    hg_contagion: float
    hg_stability: float
    hg_cross_pillar: float


@dataclass
class TrendChange:
    """7-day trend change data."""
    metric: str
    current: float
    week_ago: float
    change: float
    direction: str


@dataclass
class TrendsHistoryData:
    """Complete trends history data."""
    date: str
    current: DailyMetrics
    changes_7d: List[TrendChange]
    regime_distribution: Dict[str, int]


# =============================================================================
# TRENDS HISTORY CALCULATOR
# =============================================================================

class TrendsHistoryCalculator:
    """
    Log and analyze daily metric trends.
    
    Usage:
        calc = TrendsHistoryCalculator()
        data = calc.calculate()
        calc.save_to_supabase(data)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize calculator."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    # =========================================================================
    # DATA COLLECTION
    # =========================================================================
    
    def collect_daily_metrics(self) -> DailyMetrics:
        """Collect all key metrics from various tables."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        metrics = DailyMetrics(
            date=date_str,
            amri=0, amri_regime="Unknown", enhanced_amri=0, break_probability=0,
            mci=0, mci_regime="Unknown",
            bubble_index=0, clusters=0, correlations_20d=0, breadth_20d=0,
            vix=0, credit_spreads=0,
            infra_breadth=0, enterprise_breadth=0,
            thesis_balance=0,
            hg_contagion=0, hg_stability=0, hg_cross_pillar=0
        )
        
        if not self.supabase:
            return metrics
        
        try:
            # AMRI
            response = self.supabase.table("amri_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                d = response.data[0]
                metrics.amri = float(d.get("amri", 0))
                metrics.amri_regime = d.get("regime", "Unknown")
                metrics.enhanced_amri = float(d.get("enhanced_amri", 0))
                metrics.break_probability = float(d.get("break_probability", 0))
            
            # MCI
            response = self.supabase.table("mci_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                d = response.data[0]
                metrics.mci = float(d.get("mci", 0))
                metrics.mci_regime = d.get("regime", "Unknown")
            
            # Bubble Index
            response = self.supabase.table("bubble_index_daily") \
                .select("bubble_index") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.bubble_index = float(response.data[0].get("bubble_index", 0))
            
            # Clusters
            response = self.supabase.table("cluster_dial_daily") \
                .select("cluster_count") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.clusters = int(response.data[0].get("cluster_count", 0))
            
            # Correlations
            response = self.supabase.table("correlation_daily") \
                .select("corr_20d") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.correlations_20d = float(response.data[0].get("corr_20d", 0))
            
            # Breadth
            response = self.supabase.table("breadth_daily") \
                .select("breadth_20d") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.breadth_20d = float(response.data[0].get("breadth_20d", 0))
            
            # VIX
            response = self.supabase.table("vix_history") \
                .select("close") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.vix = float(response.data[0].get("close", 0))
            
            # Credit Spreads
            response = self.supabase.table("credit_spread_daily") \
                .select("hy_spread") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.credit_spreads = float(response.data[0].get("hy_spread", 0))
            
            # Signals (thesis balance)
            response = self.supabase.table("signals_dial_daily") \
                .select("thesis_balance") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                metrics.thesis_balance = float(response.data[0].get("thesis_balance", 0))
            
            # Hypergraph
            response = self.supabase.table("hypergraph_signals") \
                .select("contagion_score, stability_score, cross_pillar_ratio") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                d = response.data[0]
                metrics.hg_contagion = float(d.get("contagion_score", 0))
                metrics.hg_stability = float(d.get("stability_score", 0))
                metrics.hg_cross_pillar = float(d.get("cross_pillar_ratio", 0))
            
            # Dashboard for pillar breadth
            response = self.supabase.table("dashboard_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            if response.data:
                d = response.data[0]
                metrics.infra_breadth = float(d.get("infra_breadth", 0))
                metrics.enterprise_breadth = float(d.get("enterprise_breadth", 0))
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================
    
    def get_7day_changes(self, current: DailyMetrics) -> List[TrendChange]:
        """Calculate 7-day changes for key metrics."""
        changes = []
        
        if not self.supabase:
            return changes
        
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        try:
            response = self.supabase.table("trends_history") \
                .select("*") \
                .lte("date", week_ago) \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if not response.data:
                return changes
            
            old = response.data[0]
            
            # Define metrics to track
            metric_defs = [
                ("AMRI", current.amri, old.get("amri", 0), 2),
                ("Break Prob", current.break_probability, old.get("break_probability", 0), 3),
                ("Bubble Index", current.bubble_index, old.get("bubble_index", 0), 3),
                ("Clusters", current.clusters, old.get("clusters", 0), 1),
                ("Breadth 20D", current.breadth_20d * 100, old.get("breadth_20d", 0) * 100, 5),
            ]
            
            for name, curr_val, old_val, threshold in metric_defs:
                change = curr_val - old_val
                
                if name == "Clusters":
                    if change > threshold:
                        direction = "↑ Diversifying"
                    elif change < -threshold:
                        direction = "↓ Consolidating"
                    else:
                        direction = "→ Stable"
                elif name == "Breadth 20D":
                    if change > threshold:
                        direction = "↑ Improving"
                    elif change < -threshold:
                        direction = "↓ Weakening"
                    else:
                        direction = "→ Stable"
                else:
                    if change > threshold:
                        direction = "↑ Rising"
                    elif change < -threshold:
                        direction = "↓ Falling"
                    else:
                        direction = "→ Stable"
                
                changes.append(TrendChange(
                    metric=name,
                    current=curr_val,
                    week_ago=old_val,
                    change=change,
                    direction=direction
                ))
            
        except Exception as e:
            logger.error(f"Error calculating 7-day changes: {e}")
        
        return changes
    
    def get_regime_distribution(self, days: int = 30) -> Dict[str, int]:
        """Get regime distribution for last N days."""
        distribution = {"Normal": 0, "Elevated": 0, "Fragile": 0, "Critical": 0}
        
        if not self.supabase:
            return distribution
        
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            response = self.supabase.table("trends_history") \
                .select("amri_regime") \
                .gte("date", cutoff) \
                .execute()
            
            if response.data:
                for row in response.data:
                    regime = row.get("amri_regime", "Unknown")
                    if regime in distribution:
                        distribution[regime] += 1
            
        except Exception as e:
            logger.error(f"Error getting regime distribution: {e}")
        
        return distribution
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> TrendsHistoryData:
        """
        Calculate trends history data.
        
        Returns:
            TrendsHistoryData with current metrics and trends
        """
        logger.info("Calculating trends history...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Collect current metrics
        current = self.collect_daily_metrics()
        
        # Calculate 7-day changes
        changes = self.get_7day_changes(current)
        
        # Get regime distribution
        distribution = self.get_regime_distribution(30)
        
        logger.info(f"Trends: AMRI={current.amri:.1f}, MCI={current.mci:.1f}, Bubble={current.bubble_index:.1f}")
        
        return TrendsHistoryData(
            date=date_str,
            current=current,
            changes_7d=changes,
            regime_distribution=distribution
        )
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_to_supabase(self, data: TrendsHistoryData) -> Optional[Dict]:
        """Save trends history to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return None
        
        record = asdict(data.current)
        
        try:
            response = self.supabase.table("trends_history") \
                .upsert(record, on_conflict="date") \
                .execute()
            
            if response.data:
                logger.info(f"Saved trends for {data.date}")
                return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save trends: {e}")
        
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run trends history logging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trends History")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    calc = TrendsHistoryCalculator()
    data = calc.calculate()
    
    print(f"\n{'='*60}")
    print(f"TRENDS HISTORY - {data.date}")
    print(f"{'='*60}")
    
    print(f"\nCurrent Readings:")
    print(f"  AMRI: {data.current.amri:.1f} ({data.current.amri_regime})")
    print(f"  Enhanced AMRI: {data.current.enhanced_amri:.1f}")
    print(f"  Break Probability: {data.current.break_probability:.1f}")
    print(f"  MCI: {data.current.mci:.1f} ({data.current.mci_regime})")
    print(f"  Bubble Index: {data.current.bubble_index:.1f}")
    print(f"  Clusters: {data.current.clusters}")
    print(f"  Breadth 20D: {data.current.breadth_20d*100:.1f}%")
    print(f"  VIX: {data.current.vix:.2f}")
    
    if data.changes_7d:
        print(f"\n7-Day Changes:")
        for change in data.changes_7d:
            print(f"  {change.metric}: {change.current:.1f} ({change.direction}, {change.change:+.1f})")
    
    print(f"\nRegime Distribution (30D):")
    for regime, count in data.regime_distribution.items():
        print(f"  {regime}: {count} days")
    
    print(f"{'='*60}\n")
    
    if args.save:
        result = calc.save_to_supabase(data)
        if result:
            print("Saved to Supabase")
    
    return data


if __name__ == "__main__":
    main()
