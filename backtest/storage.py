"""
Backtest Storage Module

Saves backtest results to Supabase:
- Breadth divergence signals
- Portfolio backtest runs and daily returns
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BACKTEST_SCHEMA_SQL = """
-- Backtest run metadata
CREATE TABLE IF NOT EXISTS backtest_runs (
    id SERIAL PRIMARY KEY,
    run_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    config JSONB,
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT
);

-- Breadth divergence signals
CREATE TABLE IF NOT EXISTS backtest_divergences (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id),
    signal_date DATE NOT NULL,
    infra_breadth FLOAT NOT NULL,
    enterprise_breadth FLOAT NOT NULL,
    divergence_gap FLOAT NOT NULL,
    fwd_5d_delta FLOAT,
    fwd_10d_delta FLOAT,
    fwd_20d_delta FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Breadth divergence summary
CREATE TABLE IF NOT EXISTS backtest_divergence_summary (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id),
    total_divergences INTEGER,
    avg_5d_delta FLOAT,
    avg_10d_delta FLOAT,
    avg_20d_delta FLOAT,
    pct_negative_5d FLOAT,
    pct_negative_10d FLOAT,
    pct_negative_20d FLOAT,
    period_start DATE,
    period_end DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio backtest daily values
CREATE TABLE IF NOT EXISTS backtest_portfolio_daily (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id),
    trade_date DATE NOT NULL,
    equal_weight_value FLOAT,
    rotation_value FLOAT,
    spy_value FLOAT,
    equal_weight_return FLOAT,
    rotation_return FLOAT,
    spy_return FLOAT,
    top_pillars JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio backtest metrics
CREATE TABLE IF NOT EXISTS backtest_portfolio_metrics (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id),
    strategy VARCHAR(50) NOT NULL,
    total_return FLOAT,
    cagr FLOAT,
    volatility FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    best_day FLOAT,
    worst_day FLOAT,
    total_days INTEGER,
    alpha_vs_spy FLOAT,
    period_start DATE,
    period_end DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_divergences_run ON backtest_divergences(run_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_run ON backtest_portfolio_daily(run_id);
"""


class BacktestStorage:
    """Storage operations for backtest results"""
    
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.client = None
        
        if self.url and self.key:
            try:
                from supabase import create_client
                self.client = create_client(self.url, self.key)
                logger.info("Backtest storage initialized")
            except ImportError:
                logger.warning("supabase-py not installed")
            except Exception as e:
                logger.error(f"Supabase connection error: {e}")
    
    def is_connected(self) -> bool:
        return self.client is not None
    
    def create_run(self, run_type: str, config: Dict = None) -> Optional[int]:
        """Create a new backtest run and return its ID"""
        if not self.is_connected():
            return None
        
        try:
            result = self.client.table("backtest_runs").insert({
                "run_type": run_type,
                "config": config or {},
                "status": "running"
            }).execute()
            
            if result.data:
                run_id = result.data[0]["id"]
                logger.info(f"Created backtest run {run_id} ({run_type})")
                return run_id
        except Exception as e:
            logger.error(f"Failed to create run: {e}")
        return None
    
    def complete_run(self, run_id: int, status: str = "completed", error: str = None):
        """Mark a run as completed or failed"""
        if not self.is_connected():
            return
        
        try:
            update_data = {
                "completed_at": datetime.now().isoformat(),
                "status": status
            }
            if error:
                update_data["error_message"] = error
            
            self.client.table("backtest_runs").update(
                update_data
            ).eq("id", run_id).execute()
        except Exception as e:
            logger.error(f"Failed to complete run: {e}")
    
    def save_divergences(self, run_id: int, divergences: List) -> int:
        """Save divergence signals to Supabase"""
        if not self.is_connected() or not divergences:
            return 0
        
        try:
            rows = [{
                "run_id": run_id,
                "signal_date": d.date,
                "infra_breadth": d.infra_breadth,
                "enterprise_breadth": d.enterprise_breadth,
                "divergence_gap": d.divergence_gap,
                "fwd_5d_delta": d.fwd_5d,
                "fwd_10d_delta": d.fwd_10d,
                "fwd_20d_delta": d.fwd_20d
            } for d in divergences]
            
            result = self.client.table("backtest_divergences").insert(rows).execute()
            count = len(result.data) if result.data else 0
            logger.info(f"Saved {count} divergences to Supabase")
            return count
        except Exception as e:
            logger.error(f"Failed to save divergences: {e}")
            return 0
    
    def save_divergence_summary(self, run_id: int, summary: Dict, 
                                 period_start: str, period_end: str):
        """Save divergence summary statistics"""
        if not self.is_connected():
            return
        
        try:
            self.client.table("backtest_divergence_summary").insert({
                "run_id": run_id,
                "total_divergences": summary.get("total_divergences", 0),
                "avg_5d_delta": summary.get("avg_5d_delta"),
                "avg_10d_delta": summary.get("avg_10d_delta"),
                "avg_20d_delta": summary.get("avg_20d_delta"),
                "pct_negative_5d": summary.get("pct_negative_5d"),
                "pct_negative_10d": summary.get("pct_negative_10d"),
                "pct_negative_20d": summary.get("pct_negative_20d"),
                "period_start": period_start,
                "period_end": period_end
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def save_portfolio_daily(self, run_id: int, daily_data: List[Dict]) -> int:
        """Save daily portfolio values"""
        if not self.is_connected() or not daily_data:
            return 0
        
        try:
            rows = [{
                "run_id": run_id,
                "trade_date": d["date"],
                "equal_weight_value": d.get("equal_weight_value"),
                "rotation_value": d.get("rotation_value"),
                "spy_value": d.get("spy_value"),
                "equal_weight_return": d.get("equal_weight_return"),
                "rotation_return": d.get("rotation_return"),
                "spy_return": d.get("spy_return"),
                "top_pillars": d.get("top_pillars")
            } for d in daily_data]
            
            result = self.client.table("backtest_portfolio_daily").insert(rows).execute()
            count = len(result.data) if result.data else 0
            logger.info(f"Saved {count} daily records to Supabase")
            return count
        except Exception as e:
            logger.error(f"Failed to save portfolio daily: {e}")
            return 0
    
    def save_portfolio_metrics(self, run_id: int, strategy: str, 
                                metrics, period_start: str, period_end: str,
                                alpha_vs_spy: float = None):
        """Save portfolio performance metrics"""
        if not self.is_connected():
            return
        
        try:
            m = asdict(metrics) if hasattr(metrics, '__dataclass_fields__') else metrics
            
            self.client.table("backtest_portfolio_metrics").insert({
                "run_id": run_id,
                "strategy": strategy,
                "total_return": m.get("total_return"),
                "cagr": m.get("cagr"),
                "volatility": m.get("volatility"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "max_drawdown": m.get("max_drawdown"),
                "win_rate": m.get("win_rate"),
                "best_day": m.get("best_day"),
                "worst_day": m.get("worst_day"),
                "total_days": m.get("total_days"),
                "alpha_vs_spy": alpha_vs_spy,
                "period_start": period_start,
                "period_end": period_end
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


_storage_instance = None

def get_backtest_storage() -> BacktestStorage:
    """Get singleton storage instance"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = BacktestStorage()
    return _storage_instance


if __name__ == "__main__":
    print("Run this SQL in Supabase:\n")
    print(BACKTEST_SCHEMA_SQL)
