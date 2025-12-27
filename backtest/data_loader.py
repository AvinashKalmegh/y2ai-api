"""
Historical Data Loader for ARGUS-1 Backtesting

Loads historical data from Supabase tables and creates point-in-time snapshots
for backtesting the ARGUS-1 regime detection system.

Data Sources:
- bubble_index_daily: VIX, CAPE, credit spreads, bifurcation score
- hypergraph_signals: contagion, stability, clusters
- daily_signals: NLP signals, VETO triggers
- stock_tracker_daily: pillar performance, rotation
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PointInTimeData:
    """All data available as of a specific date"""
    date: str
    
    # Bubble Index data
    vix: float = 0.0
    cape: float = 0.0
    credit_spread_ig: float = 0.0
    credit_spread_hy: float = 0.0
    bubble_index: float = 50.0
    bifurcation_score: float = 0.0
    bubble_regime: str = "TRANSITION"
    
    # Hypergraph data
    contagion_score: float = 50.0
    stability_score: float = 0.5
    hyperedge_count: int = 10
    cross_pillar_ratio: float = 0.3
    contagion_regime: str = "STABLE"
    
    # NLP signals
    veto_triggers: int = 0
    thesis_balance: float = 0.0
    nci_score: float = 50.0
    npd_score: float = 0.0
    burst_count: int = 0
    evi_score: float = 50.0
    signal_regime: str = "NEUTRAL"
    
    # Stock tracker (pillar data)
    y2ai_index_today: float = 0.0
    y2ai_index_5day: float = 0.0
    y2ai_index_ytd: float = 0.0
    best_pillar: str = ""
    worst_pillar: str = ""
    pillar_spread: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if we have minimum required data"""
        return self.vix > 0 and self.bubble_index > 0


class HistoricalDataLoader:
    """
    Load historical data from Supabase for backtesting.
    Creates point-in-time snapshots that simulate what ARGUS-1
    would have seen on any given historical date.
    """
    
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        if not self.client:
            self._init_client()
        
        # Cache DataFrames
        self._bubble_df: Optional[pd.DataFrame] = None
        self._hypergraph_df: Optional[pd.DataFrame] = None
        self._signals_df: Optional[pd.DataFrame] = None
        self._tracker_df: Optional[pd.DataFrame] = None
    
    def _init_client(self):
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if url and key:
                self.client = create_client(url, key)
                logger.info("Supabase client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Supabase: {e}")
            self.client = None
    
    def load_all_data(self, start_date: str, end_date: str) -> bool:
        """
        Load all historical data for the specified date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            True if data loaded successfully
        """
        logger.info(f"Loading historical data from {start_date} to {end_date}")
        
        try:
            self._load_bubble_index(start_date, end_date)
            self._load_hypergraph(start_date, end_date)
            self._load_daily_signals(start_date, end_date)
            self._load_stock_tracker(start_date, end_date)
            
            logger.info(f"Loaded: bubble={len(self._bubble_df) if self._bubble_df is not None else 0}, "
                       f"hypergraph={len(self._hypergraph_df) if self._hypergraph_df is not None else 0}, "
                       f"signals={len(self._signals_df) if self._signals_df is not None else 0}, "
                       f"tracker={len(self._tracker_df) if self._tracker_df is not None else 0}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _load_bubble_index(self, start_date: str, end_date: str):
        """Load bubble_index_daily table"""
        if not self.client:
            self._bubble_df = pd.DataFrame()
            return
        
        result = self.client.table("bubble_index_daily")\
            .select("*")\
            .gte("date", start_date)\
            .lte("date", end_date)\
            .order("date")\
            .execute()
        
        if result.data:
            self._bubble_df = pd.DataFrame(result.data)
            self._bubble_df['date'] = pd.to_datetime(self._bubble_df['date']).dt.date.astype(str)
            self._bubble_df = self._bubble_df.set_index('date')
        else:
            self._bubble_df = pd.DataFrame()
    
    def _load_hypergraph(self, start_date: str, end_date: str):
        """Load hypergraph_signals table"""
        if not self.client:
            self._hypergraph_df = pd.DataFrame()
            return
        
        result = self.client.table("hypergraph_signals")\
            .select("*")\
            .gte("date", start_date)\
            .lte("date", end_date)\
            .order("date")\
            .execute()
        
        if result.data:
            self._hypergraph_df = pd.DataFrame(result.data)
            self._hypergraph_df['date'] = pd.to_datetime(self._hypergraph_df['date']).dt.date.astype(str)
            self._hypergraph_df = self._hypergraph_df.set_index('date')
        else:
            self._hypergraph_df = pd.DataFrame()
    
    def _load_daily_signals(self, start_date: str, end_date: str):
        """Load daily_signals table"""
        if not self.client:
            self._signals_df = pd.DataFrame()
            return
        
        result = self.client.table("daily_signals")\
            .select("*")\
            .gte("date", start_date)\
            .lte("date", end_date)\
            .order("date")\
            .execute()
        
        if result.data:
            self._signals_df = pd.DataFrame(result.data)
            self._signals_df['date'] = pd.to_datetime(self._signals_df['date']).dt.date.astype(str)
            # Group by date and take latest if multiple rows per day
            self._signals_df = self._signals_df.groupby('date').last()
        else:
            self._signals_df = pd.DataFrame()
    
    def _load_stock_tracker(self, start_date: str, end_date: str):
        """Load stock_tracker_daily table"""
        if not self.client:
            self._tracker_df = pd.DataFrame()
            return
        
        result = self.client.table("stock_tracker_daily")\
            .select("*")\
            .gte("date", start_date)\
            .lte("date", end_date)\
            .order("date")\
            .execute()
        
        if result.data:
            self._tracker_df = pd.DataFrame(result.data)
            self._tracker_df['date'] = pd.to_datetime(self._tracker_df['date']).dt.date.astype(str)
            self._tracker_df = self._tracker_df.set_index('date')
        else:
            self._tracker_df = pd.DataFrame()
    
    def get_point_in_time(self, date: str) -> PointInTimeData:
        """
        Get all data available as of a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
        
        Returns:
            PointInTimeData with all available signals
        """
        pit = PointInTimeData(date=date)
        
        # Get bubble index data (exact date or most recent before)
        if self._bubble_df is not None and len(self._bubble_df) > 0:
            if date in self._bubble_df.index:
                row = self._bubble_df.loc[date]
            else:
                # Get most recent date before target
                prior_dates = [d for d in self._bubble_df.index if d <= date]
                if prior_dates:
                    row = self._bubble_df.loc[max(prior_dates)]
                else:
                    row = None
            
            if row is not None:
                pit.vix = float(row.get('vix', 0) or 0)
                pit.cape = float(row.get('cape', 0) or 0)
                pit.credit_spread_ig = float(row.get('credit_spread_ig', 0) or 0)
                pit.credit_spread_hy = float(row.get('credit_spread_hy', 0) or 0)
                pit.bubble_index = float(row.get('bubble_index', 50) or 50)
                pit.bifurcation_score = float(row.get('bifurcation_score', 0) or 0)
                pit.bubble_regime = str(row.get('regime', 'TRANSITION') or 'TRANSITION')
        
        # Get hypergraph data
        if self._hypergraph_df is not None and len(self._hypergraph_df) > 0:
            if date in self._hypergraph_df.index:
                row = self._hypergraph_df.loc[date]
            else:
                prior_dates = [d for d in self._hypergraph_df.index if d <= date]
                if prior_dates:
                    row = self._hypergraph_df.loc[max(prior_dates)]
                else:
                    row = None
            
            if row is not None:
                pit.contagion_score = float(row.get('contagion_score', 50) or 50)
                pit.stability_score = float(row.get('stability_score', 0.5) or 0.5)
                pit.hyperedge_count = int(row.get('hyperedge_count', 10) or 10)
                pit.cross_pillar_ratio = float(row.get('cross_pillar_ratio', 0.3) or 0.3)
                pit.contagion_regime = str(row.get('regime', 'STABLE') or 'STABLE')
        
        # Get daily signals
        if self._signals_df is not None and len(self._signals_df) > 0:
            if date in self._signals_df.index:
                row = self._signals_df.loc[date]
            else:
                prior_dates = [d for d in self._signals_df.index if d <= date]
                if prior_dates:
                    row = self._signals_df.loc[max(prior_dates)]
                else:
                    row = None
            
            if row is not None:
                pit.veto_triggers = int(row.get('veto_triggers', 0) or 0)
                pit.thesis_balance = float(row.get('thesis_balance', 0) or 0)
                pit.nci_score = float(row.get('nci_score', 50) or 50)
                pit.npd_score = float(row.get('npd_score', 0) or 0)
                pit.burst_count = int(row.get('burst_count', 0) or 0)
                pit.evi_score = float(row.get('evi_score', 50) or 50)
                pit.signal_regime = str(row.get('signal_regime', 'NEUTRAL') or 'NEUTRAL')
        
        # Get stock tracker data
        if self._tracker_df is not None and len(self._tracker_df) > 0:
            if date in self._tracker_df.index:
                row = self._tracker_df.loc[date]
            else:
                prior_dates = [d for d in self._tracker_df.index if d <= date]
                if prior_dates:
                    row = self._tracker_df.loc[max(prior_dates)]
                else:
                    row = None
            
            if row is not None:
                pit.y2ai_index_today = float(row.get('y2ai_index_today', 0) or 0)
                pit.y2ai_index_5day = float(row.get('y2ai_index_5day', 0) or 0)
                pit.y2ai_index_ytd = float(row.get('y2ai_index_ytd', 0) or 0)
                pit.best_pillar = str(row.get('best_pillar', '') or '')
                pit.worst_pillar = str(row.get('worst_pillar', '') or '')
        
        return pit
    
    def get_available_dates(self) -> List[str]:
        """Get list of dates where we have bubble index data"""
        if self._bubble_df is None or len(self._bubble_df) == 0:
            return []
        return sorted(self._bubble_df.index.tolist())
    
    def get_date_range(self) -> Tuple[str, str]:
        """Get min and max dates with data"""
        dates = self.get_available_dates()
        if not dates:
            return ("", "")
        return (min(dates), max(dates))
