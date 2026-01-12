"""
CLUSTER DIAL
============
Measures cluster consolidation as a leading indicator of regime change.
Fewer clusters = higher systemic risk (herding behavior).

Algorithm:
- Calculate 60-day correlation matrix for all 43 stocks
- Convert to distance matrix (distance = 1 - correlation)
- Run hierarchical clustering with threshold 0.7
- Count resulting clusters

Regime Thresholds (based on cluster count):
- Healthy: >= 10 clusters (diversified, low systemic risk)
- Caution: 7-9 clusters (some consolidation)
- Fragile: 4-6 clusters (high consolidation)
- Crisis: < 4 clusters (extreme herding, systemic risk)

Also reports:
- Average correlation across all pairs
- Cluster composition by pillar
- Concentration metrics (HHI, top cluster %)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CLUSTER_CONFIG = {
    "lookback_days": 60,          # Days for correlation calculation
    "cluster_threshold": 0.7,     # Distance threshold for clustering
    "min_tickers": 10,            # Minimum tickers required
    
    # Regime thresholds (cluster count)
    "regime_thresholds": {
        "healthy": 10,    # >= 10 clusters
        "caution": 7,     # 7-9 clusters
        "fragile": 4,     # 4-6 clusters
        # < 4 = Crisis
    }
}

# Pillar mapping
PILLAR_STOCKS = {
    "Infrastructure": ["TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT",
                      "CEG", "NRG", "EQIX", "DLR", "KLAC", "LRCX", "AMAT", "QCOM"],
    "Enterprise": ["MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR",
                  "ADBE", "ORCL", "MDB", "DDOG", "ZS"],
    "Productivity": ["NET", "CRWD", "PANW"],
    "Demand": ["TSLA", "SHOP", "UBER"],
    "Macro": ["NXPI", "ON", "SMCI", "ARM"],
    "Financial": ["GS", "MS", "JKS", "FSLR"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClusterData:
    """Cluster analysis results."""
    date: str
    cluster_count: int
    regime: str
    avg_correlation: float
    # Concentration metrics
    hhi: float                    # Herfindahl-Hirschman Index
    top_cluster_pct: float        # % of stocks in largest cluster
    # Cluster details
    cluster_sizes: List[int]
    cluster_composition: Dict     # Pillar breakdown per cluster
    valid_tickers: int
    interpretation: str


# =============================================================================
# CLUSTER CALCULATOR
# =============================================================================

class ClusterDialCalculator:
    """
    Calculate cluster consolidation metrics.
    
    Usage:
        calc = ClusterDialCalculator()
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
        
        self.config = CLUSTER_CONFIG
        
        # Build ticker to pillar mapping
        self.ticker_to_pillar = {}
        for pillar, tickers in PILLAR_STOCKS.items():
            for ticker in tickers:
                self.ticker_to_pillar[ticker] = pillar
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers."""
        return list(self.ticker_to_pillar.keys())
    
    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    
    def fetch_price_history(self, days: int = None) -> pd.DataFrame:
        """Fetch price history for correlation calculation."""
        if days is None:
            days = self.config["lookback_days"]
        
        tickers = self.get_all_tickers()
        
        # Primary: Supabase
        if self.supabase:
            try:
                start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
                
                response = self.supabase.table("price_history") \
                    .select("date, ticker, close") \
                    .in_("ticker", tickers) \
                    .gte("date", start_date) \
                    .order("date", desc=True) \
                    .execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    df["date"] = pd.to_datetime(df["date"])
                    logger.info(f"Loaded {len(df)} rows from Supabase")
                    return df
            except Exception as e:
                logger.warning(f"Supabase fetch failed: {e}")
        
        # Fallback: yfinance
        logger.warning("Falling back to yfinance")
        try:
            import yfinance as yf
            
            all_data = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    for date, row in hist.iterrows():
                        all_data.append({
                            "date": date,
                            "ticker": ticker,
                            "close": row["Close"]
                        })
                except:
                    pass
            
            if all_data:
                return pd.DataFrame(all_data)
        except ImportError:
            pass
        
        return pd.DataFrame()
    
    # =========================================================================
    # CORRELATION CALCULATION
    # =========================================================================
    
    def calculate_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns matrix."""
        # Pivot to get prices per ticker per day
        pivot = price_df.pivot_table(
            index="date",
            columns="ticker",
            values="close"
        ).sort_index()
        
        # Calculate returns
        returns = pivot.pct_change().dropna()
        
        # Keep only last N days
        lookback = self.config["lookback_days"]
        if len(returns) > lookback:
            returns = returns.tail(lookback)
        
        return returns
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate correlation matrix."""
        # Drop tickers with too many NaN
        valid_returns = returns.dropna(axis=1, thresh=len(returns) * 0.8)
        
        if valid_returns.empty:
            return pd.DataFrame(), []
        
        corr_matrix = valid_returns.corr()
        valid_tickers = list(corr_matrix.columns)
        
        return corr_matrix, valid_tickers
    
    # =========================================================================
    # CLUSTERING
    # =========================================================================
    
    def run_clustering(self, corr_matrix: pd.DataFrame, threshold: float = None) -> List[List[str]]:
        """
        Run hierarchical clustering on correlation matrix.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Distance threshold for cluster formation
            
        Returns:
            List of clusters (each cluster is list of tickers)
        """
        if threshold is None:
            threshold = self.config["cluster_threshold"]
        
        if corr_matrix.empty or len(corr_matrix) < 2:
            return []
        
        # Convert correlation to distance (distance = 1 - correlation)
        distance_matrix = 1 - corr_matrix.values
        
        # Ensure symmetric and no negative distances
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, 2)
        
        # Convert to condensed form for scipy
        try:
            condensed = squareform(distance_matrix)
            
            # Hierarchical clustering (average linkage)
            Z = linkage(condensed, method='average')
            
            # Form clusters at threshold
            cluster_labels = fcluster(Z, threshold, criterion='distance')
            
            # Group tickers by cluster
            tickers = list(corr_matrix.columns)
            clusters = {}
            
            for ticker, label in zip(tickers, cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(ticker)
            
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return [[t] for t in corr_matrix.columns]  # Each ticker is own cluster
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def analyze_clusters(self, clusters: List[List[str]]) -> Dict:
        """Analyze cluster composition by pillar."""
        analysis = {}
        
        for i, cluster in enumerate(clusters):
            pillar_counts = {}
            
            for ticker in cluster:
                pillar = self.ticker_to_pillar.get(ticker, "Unknown")
                pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
            
            analysis[f"cluster_{i+1}"] = {
                "size": len(cluster),
                "tickers": cluster,
                "pillars": pillar_counts
            }
        
        return analysis
    
    def calculate_concentration(self, clusters: List[List[str]], total_tickers: int) -> Tuple[float, float]:
        """
        Calculate concentration metrics.
        
        Returns:
            Tuple of (HHI, top_cluster_percentage)
        """
        if not clusters or total_tickers == 0:
            return 0, 0
        
        # Cluster sizes as percentages
        sizes = [len(c) / total_tickers for c in clusters]
        
        # HHI (sum of squared market shares)
        hhi = sum(s ** 2 for s in sizes)
        
        # Top cluster percentage
        top_pct = max(sizes) if sizes else 0
        
        return hhi, top_pct
    
    def get_regime(self, cluster_count: int) -> str:
        """Determine regime from cluster count."""
        thresholds = self.config["regime_thresholds"]
        
        if cluster_count >= thresholds["healthy"]:
            return "Healthy"
        elif cluster_count >= thresholds["caution"]:
            return "Caution"
        elif cluster_count >= thresholds["fragile"]:
            return "Fragile"
        else:
            return "Crisis"
    
    def get_interpretation(self, regime: str, cluster_count: int, avg_corr: float) -> str:
        """Get human-readable interpretation."""
        interpretations = {
            "Healthy": f"Diversified market with {cluster_count} distinct clusters. Low systemic risk.",
            "Caution": f"Some consolidation ({cluster_count} clusters). Monitor for herding.",
            "Fragile": f"High consolidation ({cluster_count} clusters). Elevated systemic risk.",
            "Crisis": f"Extreme herding ({cluster_count} clusters). Systemic crisis conditions."
        }
        base = interpretations.get(regime, "Unknown regime")
        return f"{base} Avg correlation: {avg_corr:.2f}"
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate(self) -> ClusterData:
        """
        Main calculation: compute cluster metrics.
        
        Returns:
            ClusterData with all metrics
        """
        logger.info("Calculating cluster metrics...")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Fetch price data
        price_df = self.fetch_price_history()
        
        if price_df.empty:
            logger.warning("No price data available")
            return ClusterData(
                date=date_str,
                cluster_count=0,
                regime="Unknown",
                avg_correlation=0,
                hhi=0,
                top_cluster_pct=0,
                cluster_sizes=[],
                cluster_composition={},
                valid_tickers=0,
                interpretation="No data available"
            )
        
        # Calculate returns
        returns = self.calculate_returns(price_df)
        
        # Calculate correlation matrix
        corr_matrix, valid_tickers = self.calculate_correlation_matrix(returns)
        
        if len(valid_tickers) < self.config["min_tickers"]:
            logger.warning(f"Only {len(valid_tickers)} valid tickers, need {self.config['min_tickers']}")
            return ClusterData(
                date=date_str,
                cluster_count=0,
                regime="Unknown",
                avg_correlation=0,
                hhi=0,
                top_cluster_pct=0,
                cluster_sizes=[],
                cluster_composition={},
                valid_tickers=len(valid_tickers),
                interpretation="Insufficient data"
            )
        
        # Calculate average correlation
        upper_tri = np.triu(corr_matrix.values, k=1)
        avg_correlation = upper_tri[upper_tri != 0].mean()
        
        # Run clustering
        clusters = self.run_clustering(corr_matrix)
        cluster_count = len(clusters)
        
        # Analyze clusters
        cluster_composition = self.analyze_clusters(clusters)
        cluster_sizes = [len(c) for c in clusters]
        
        # Calculate concentration
        hhi, top_pct = self.calculate_concentration(clusters, len(valid_tickers))
        
        # Determine regime
        regime = self.get_regime(cluster_count)
        interpretation = self.get_interpretation(regime, cluster_count, avg_correlation)
        
        result = ClusterData(
            date=date_str,
            cluster_count=cluster_count,
            regime=regime,
            avg_correlation=round(avg_correlation, 3),
            hhi=round(hhi, 4),
            top_cluster_pct=round(top_pct, 3),
            cluster_sizes=cluster_sizes,
            cluster_composition=cluster_composition,
            valid_tickers=len(valid_tickers),
            interpretation=interpretation
        )
        
        logger.info(f"Clusters: {cluster_count}, Regime: {regime}, Avg Corr: {avg_correlation:.2f}")
        
        return result
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, data: ClusterData) -> bool:
        """Save cluster data to Supabase."""
        if not self.supabase:
            return False
        
        row = {
            "date": data.date,
            "cluster_count": data.cluster_count,
            "regime": data.regime,
            "avg_correlation": data.avg_correlation,
            "hhi": data.hhi,
            "top_cluster_pct": data.top_cluster_pct,
            "cluster_sizes": data.cluster_sizes,
            "cluster_composition": data.cluster_composition,
            "valid_tickers": data.valid_tickers,
            "interpretation": data.interpretation
        }
        
        try:
            self.supabase.table("cluster_dial_daily") \
                .upsert(row, on_conflict="date") \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster Dial")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    calc = ClusterDialCalculator()
    
    print(f"\n{'='*60}")
    print("CLUSTER DIAL")
    print(f"{'='*60}\n")
    
    data = calc.calculate()
    
    print(f"Date: {data.date}")
    print(f"\n{'='*40}")
    print(f"CLUSTER COUNT: {data.cluster_count}")
    print(f"REGIME: {data.regime}")
    print(f"{'='*40}")
    
    print(f"\nAvg Correlation: {data.avg_correlation:.3f}")
    print(f"HHI (concentration): {data.hhi:.4f}")
    print(f"Top Cluster %: {data.top_cluster_pct*100:.1f}%")
    print(f"Valid Tickers: {data.valid_tickers}")
    
    if data.cluster_sizes:
        print(f"\nCluster Sizes: {data.cluster_sizes}")
    
    print(f"\n{data.interpretation}")
    
    if args.save:
        if calc.save_to_supabase(data):
            print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
