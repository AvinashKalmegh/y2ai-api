"""
SHADOW PORTFOLIO TRACKER
========================
Live portfolio tracking with pillar-based weights and macro multiplier.

Strategy (Option B: Pillar-weighted with macro multiplier):
- Starting capital: $1,000,000
- Pillar weights based on signals:
  - LEADING: 1.5x weight
  - NEUTRAL: 1.0x weight  
  - WEAKENING: 0.5x weight
  - STRESSED: 0.0x weight (no allocation)
- Macro multiplier from AMRI reduces equity exposure
- Benchmarks: SPY, Equal-weight universe

Metrics:
- Daily NAV
- Cumulative return
- Alpha vs SPY
- Alpha vs equal-weight
- Maximum drawdown
- Equity/cash exposure
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PORTFOLIO_CONFIG = {
    "starting_capital": 1_000_000,
    "start_date": "2025-12-18",
    "benchmarks": ["SPY", "QQQ"],
    
    # Signal weights for pillar allocation
    "signal_weights": {
        "LEADING": 1.5,
        "NEUTRAL": 1.0,
        "WEAKENING": 0.5,
        "STRESSED": 0.0
    }
}

# Pillar to stock mapping (matches Google Sheets)
PILLAR_STOCKS = {
    "Infrastructure & Energy": [
        "TSM", "ASML", "NVDA", "AMD", "MU", "INTC", "AVGO", "VRT",
        "CEG", "NRG", "EQIX", "DLR", "SMCI", "AMAT", "LRCX", "QCOM"
    ],
    "Enterprise Adoption": [
        "MSFT", "AMZN", "GOOGL", "META", "CRM", "NOW", "SNOW", "PLTR",
        "ORCL", "MDB", "DDOG", "ZS", "NET"
    ],
    "Productivity & Labor": ["CRM", "NOW", "ADBE"],
    "Demand Dynamics": ["TSLA", "SHOP", "UBER"],
    "Financial & Market": ["GS", "MS", "COIN", "HOOD"],
    "Macro & Policy": ["FSLR", "ENPH", "JKS", "RUN"]
}

# AMRI thresholds for macro multiplier
AMRI_MULTIPLIER_MAP = {
    # (min_amri, max_amri): multiplier
    (0, 30): 1.0,      # Normal - full equity
    (30, 50): 0.75,    # Elevated - 75% equity
    (50, 70): 0.50,    # Tension - 50% equity
    (70, 85): 0.30,    # Fragile - 30% equity
    (85, 100): 0.10    # Break - 10% equity
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioPosition:
    """Single position in portfolio."""
    ticker: str
    pillar: str
    pillar_weight: float
    stock_weight: float
    shares: float
    price: float
    value: float
    daily_return: float
    weighted_return: float
    signal: str  # LEADING, NEUTRAL, etc.


@dataclass 
class PortfolioSnapshot:
    """Daily portfolio snapshot."""
    date: str
    nav: float
    daily_return: float
    cumulative_return: float
    spy_return: float
    spy_cumulative: float
    alpha_vs_spy: float
    ew_return: float
    ew_cumulative: float
    alpha_vs_ew: float
    equity_exposure: float
    cash_exposure: float
    macro_multiplier: float
    amri_score: float
    regime: str
    drawdown: float
    positions: List[PortfolioPosition] = None


# =============================================================================
# PORTFOLIO TRACKER
# =============================================================================

class PortfolioTracker:
    """
    Track shadow portfolio performance.
    
    Usage:
        tracker = PortfolioTracker()
        snapshot = tracker.calculate_daily()
        tracker.save_to_supabase(snapshot)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize tracker."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        self.config = PORTFOLIO_CONFIG
    
    # =========================================================================
    # PILLAR SIGNALS
    # =========================================================================
    
    def get_pillar_signals(self) -> Dict[str, str]:
        """
        Get current pillar signals from Supabase.
        
        Returns:
            Dict of pillar name -> signal (LEADING, NEUTRAL, etc.)
        """
        if not self.supabase:
            # Default all to NEUTRAL
            return {pillar: "NEUTRAL" for pillar in PILLAR_STOCKS.keys()}
        
        try:
            # Get from pillar_index_daily
            response = self.supabase.table("pillar_index_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                row = response.data[0]
                signals = {}
                
                # Determine signal based on momentum
                pillar_map = {
                    "Infrastructure & Energy": ("infra_5d", "infra_1m", "infra_3m"),
                    "Enterprise Adoption": ("enterprise_5d", "enterprise_1m", "enterprise_3m"),
                    "Macro & Policy": ("macro_5d", "macro_1m", "macro_3m"),
                    "Financial & Market": ("financial_5d", "financial_1m", "financial_3m"),
                    "Productivity & Labor": ("productivity_5d", "productivity_1m", "productivity_3m"),
                    "Demand Dynamics": ("demand_5d", "demand_1m", "demand_3m")
                }
                
                for pillar, (m5d_col, m1m_col, m3m_col) in pillar_map.items():
                    m5d = row.get(m5d_col, 0) or 0
                    m1m = row.get(m1m_col, 0) or 0
                    m3m = row.get(m3m_col, 0) or 0
                    
                    if m5d > 0 and m1m > 0 and m3m > 0:
                        signals[pillar] = "LEADING"
                    elif m5d < 0 and m1m < 0 and m3m < 0:
                        signals[pillar] = "STRESSED"
                    elif m5d < 0 or m1m < 0:
                        signals[pillar] = "WEAKENING"
                    else:
                        signals[pillar] = "NEUTRAL"
                
                return signals
                
        except Exception as e:
            logger.warning(f"Failed to get pillar signals: {e}")
        
        return {pillar: "NEUTRAL" for pillar in PILLAR_STOCKS.keys()}
    
    def calculate_pillar_weights(self, signals: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate normalized pillar weights from signals.
        
        Args:
            signals: Dict of pillar -> signal
            
        Returns:
            Dict of pillar -> normalized weight (sums to 1.0)
        """
        raw_weights = {}
        total_raw = 0
        
        for pillar, signal in signals.items():
            weight = self.config["signal_weights"].get(signal, 1.0)
            raw_weights[pillar] = weight
            total_raw += weight
        
        # Normalize
        if total_raw > 0:
            return {p: w / total_raw for p, w in raw_weights.items()}
        else:
            # All stressed - equal weight
            n = len(signals)
            return {p: 1/n for p in signals.keys()}
    
    # =========================================================================
    # MACRO MULTIPLIER
    # =========================================================================
    
    def get_macro_multiplier(self) -> Tuple[float, float, str]:
        """
        Get macro multiplier from AMRI score.
        
        Returns:
            Tuple of (multiplier, amri_score, regime)
        """
        if not self.supabase:
            return 1.0, 30, "Normal"
        
        try:
            response = self.supabase.table("amri_daily") \
                .select("amri_score, regime") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                amri = response.data[0].get("amri_score", 30)
                regime = response.data[0].get("regime", "Normal")
                
                # Get multiplier based on AMRI
                multiplier = 1.0
                for (low, high), mult in AMRI_MULTIPLIER_MAP.items():
                    if low <= amri < high:
                        multiplier = mult
                        break
                
                return multiplier, amri, regime
                
        except Exception as e:
            logger.warning(f"Failed to get macro multiplier: {e}")
        
        return 1.0, 30, "Normal"
    
    # =========================================================================
    # PRICE DATA
    # =========================================================================
    
    def get_stock_returns(self, date: str = None) -> Dict[str, float]:
        """
        Get daily returns for all stocks.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses latest.
            
        Returns:
            Dict of ticker -> daily return
        """
        all_tickers = []
        for stocks in PILLAR_STOCKS.values():
            all_tickers.extend(stocks)
        all_tickers = list(set(all_tickers))
        
        # Try Supabase
        if self.supabase:
            try:
                if date:
                    # Get specific date and previous
                    response = self.supabase.table("price_history") \
                        .select("date, ticker, close") \
                        .in_("ticker", all_tickers) \
                        .lte("date", date) \
                        .order("date", desc=True) \
                        .execute()
                else:
                    response = self.supabase.table("price_history") \
                        .select("date, ticker, close") \
                        .in_("ticker", all_tickers) \
                        .order("date", desc=True) \
                        .execute()
                
                if response.data:
                    df = pd.DataFrame(response.data)
                    pivot = df.pivot_table(
                        index="date",
                        columns="ticker",
                        values="close"
                    ).sort_index()
                    
                    if len(pivot) >= 2:
                        returns = pivot.iloc[-1] / pivot.iloc[-2] - 1
                        return returns.to_dict()
                        
            except Exception as e:
                logger.warning(f"Supabase price fetch failed: {e}")
        
        # Fall back to yfinance
        if YFINANCE_AVAILABLE:
            returns = {}
            for ticker in all_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d")
                    if len(hist) >= 2:
                        returns[ticker] = hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1
                except:
                    pass
            return returns
        
        return {}
    
    def get_benchmark_return(self, ticker: str = "SPY") -> float:
        """Get benchmark daily return."""
        # Try Supabase
        if self.supabase:
            try:
                response = self.supabase.table("etf_history") \
                    .select("date, close") \
                    .eq("ticker", ticker) \
                    .order("date", desc=True) \
                    .limit(2) \
                    .execute()
                
                if response.data and len(response.data) >= 2:
                    return response.data[0]["close"] / response.data[1]["close"] - 1
            except:
                pass
        
        # Fall back to yfinance
        if YFINANCE_AVAILABLE:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if len(hist) >= 2:
                    return hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1
            except:
                pass
        
        return 0
    
    # =========================================================================
    # PORTFOLIO CALCULATION
    # =========================================================================
    
    def calculate_portfolio_return(
        self,
        pillar_weights: Dict[str, float],
        stock_returns: Dict[str, float],
        macro_multiplier: float
    ) -> Tuple[float, List[PortfolioPosition]]:
        """
        Calculate portfolio return for the day.
        
        Args:
            pillar_weights: Dict of pillar -> weight
            stock_returns: Dict of ticker -> return
            macro_multiplier: Equity exposure multiplier
            
        Returns:
            Tuple of (portfolio_return, positions)
        """
        positions = []
        total_return = 0
        
        for pillar, pillar_weight in pillar_weights.items():
            stocks = PILLAR_STOCKS.get(pillar, [])
            n_stocks = len(stocks)
            
            if n_stocks == 0 or pillar_weight == 0:
                continue
            
            stock_weight = pillar_weight / n_stocks
            
            for ticker in stocks:
                ret = stock_returns.get(ticker, 0)
                weighted_ret = ret * stock_weight * macro_multiplier
                
                positions.append(PortfolioPosition(
                    ticker=ticker,
                    pillar=pillar,
                    pillar_weight=pillar_weight,
                    stock_weight=stock_weight,
                    shares=0,  # Calculated separately if needed
                    price=0,
                    value=0,
                    daily_return=ret,
                    weighted_return=weighted_ret,
                    signal=""
                ))
                
                total_return += weighted_ret
        
        return total_return, positions
    
    def calculate_equal_weight_return(self, stock_returns: Dict[str, float]) -> float:
        """Calculate equal-weight universe return."""
        returns = [r for r in stock_returns.values() if r is not None and not np.isnan(r)]
        return np.mean(returns) if returns else 0
    
    # =========================================================================
    # MAIN CALCULATION
    # =========================================================================
    
    def calculate_daily(self, date: str = None) -> PortfolioSnapshot:
        """
        Calculate daily portfolio snapshot.
        
        Args:
            date: Date to calculate. If None, uses latest.
            
        Returns:
            PortfolioSnapshot with all metrics
        """
        logger.info("Calculating daily portfolio snapshot...")
        
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        
        # Get pillar signals and weights
        signals = self.get_pillar_signals()
        pillar_weights = self.calculate_pillar_weights(signals)
        
        # Get macro multiplier
        macro_mult, amri, regime = self.get_macro_multiplier()
        
        # Get stock returns
        stock_returns = self.get_stock_returns(date)
        
        # Calculate portfolio return
        portfolio_return, positions = self.calculate_portfolio_return(
            pillar_weights, stock_returns, macro_mult
        )
        
        # Calculate benchmarks
        spy_return = self.get_benchmark_return("SPY")
        ew_return = self.calculate_equal_weight_return(stock_returns)
        
        # Get previous NAV
        prev_nav = self._get_previous_nav()
        nav = prev_nav * (1 + portfolio_return)
        
        # Calculate cumulative returns
        start_capital = self.config["starting_capital"]
        cum_return = (nav / start_capital) - 1
        
        # Get previous SPY/EW cumulative
        prev_spy_cum = self._get_previous_cumulative("spy")
        prev_ew_cum = self._get_previous_cumulative("ew")
        
        spy_cum = (1 + prev_spy_cum) * (1 + spy_return) - 1
        ew_cum = (1 + prev_ew_cum) * (1 + ew_return) - 1
        
        # Calculate drawdown
        peak_nav = self._get_peak_nav()
        if nav > peak_nav:
            peak_nav = nav
        drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        
        # Exposure
        equity_exposure = macro_mult
        cash_exposure = 1 - macro_mult
        
        snapshot = PortfolioSnapshot(
            date=date_str,
            nav=nav,
            daily_return=portfolio_return,
            cumulative_return=cum_return,
            spy_return=spy_return,
            spy_cumulative=spy_cum,
            alpha_vs_spy=cum_return - spy_cum,
            ew_return=ew_return,
            ew_cumulative=ew_cum,
            alpha_vs_ew=cum_return - ew_cum,
            equity_exposure=equity_exposure,
            cash_exposure=cash_exposure,
            macro_multiplier=macro_mult,
            amri_score=amri,
            regime=regime,
            drawdown=drawdown,
            positions=positions
        )
        
        logger.info(f"Portfolio NAV: ${nav:,.0f} ({cum_return*100:+.2f}%)")
        logger.info(f"Alpha vs SPY: {snapshot.alpha_vs_spy*100:+.2f}%")
        
        return snapshot
    
    def _get_previous_nav(self) -> float:
        """Get previous day's NAV."""
        if self.supabase:
            try:
                response = self.supabase.table("portfolio_nav_daily") \
                    .select("nav") \
                    .order("date", desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    return response.data[0]["nav"]
            except:
                pass
        
        return self.config["starting_capital"]
    
    def _get_previous_cumulative(self, benchmark: str) -> float:
        """Get previous cumulative return for benchmark."""
        if self.supabase:
            try:
                col = "spy_cumulative" if benchmark == "spy" else "ew_cumulative"
                response = self.supabase.table("portfolio_nav_daily") \
                    .select(col) \
                    .order("date", desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    return response.data[0][col]
            except:
                pass
        
        return 0
    
    def _get_peak_nav(self) -> float:
        """Get peak NAV for drawdown calculation."""
        if self.supabase:
            try:
                response = self.supabase.table("portfolio_nav_daily") \
                    .select("nav") \
                    .order("nav", desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    return response.data[0]["nav"]
            except:
                pass
        
        return self.config["starting_capital"]
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def save_to_supabase(self, snapshot: PortfolioSnapshot) -> bool:
        """Save portfolio snapshot to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return False
        
        # Save NAV row (without positions)
        nav_row = {
            "date": snapshot.date,
            "nav": snapshot.nav,
            "daily_return": snapshot.daily_return,
            "cumulative_return": snapshot.cumulative_return,
            "spy_return": snapshot.spy_return,
            "spy_cumulative": snapshot.spy_cumulative,
            "alpha_vs_spy": snapshot.alpha_vs_spy,
            "ew_return": snapshot.ew_return,
            "ew_cumulative": snapshot.ew_cumulative,
            "alpha_vs_ew": snapshot.alpha_vs_ew,
            "equity_exposure": snapshot.equity_exposure,
            "cash_exposure": snapshot.cash_exposure,
            "macro_multiplier": snapshot.macro_multiplier,
            "amri_score": snapshot.amri_score,
            "regime": snapshot.regime,
            "drawdown": snapshot.drawdown
        }
        
        try:
            self.supabase.table("portfolio_nav_daily") \
                .upsert(nav_row, on_conflict="date") \
                .execute()
            
            # Save positions separately
            if snapshot.positions:
                position_rows = []
                for pos in snapshot.positions:
                    position_rows.append({
                        "date": snapshot.date,
                        "ticker": pos.ticker,
                        "pillar": pos.pillar,
                        "pillar_weight": pos.pillar_weight,
                        "stock_weight": pos.stock_weight,
                        "daily_return": pos.daily_return,
                        "weighted_return": pos.weighted_return
                    })
                
                # Delete existing positions for date
                self.supabase.table("shadow_portfolio") \
                    .delete() \
                    .eq("date", snapshot.date) \
                    .execute()
                
                # Insert new positions
                self.supabase.table("shadow_portfolio") \
                    .insert(position_rows) \
                    .execute()
            
            logger.info(f"Saved portfolio data for {snapshot.date}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
            return False
    
    def get_history(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get portfolio history from Supabase."""
        if not self.supabase:
            return []
        
        try:
            response = self.supabase.table("portfolio_nav_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(days) \
                .execute()
            
            return [PortfolioSnapshot(**row, positions=None) for row in response.data]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


# =============================================================================
# SUPABASE TABLES
# =============================================================================

CREATE_TABLES_SQL = """
-- Portfolio NAV daily
CREATE TABLE IF NOT EXISTS portfolio_nav_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    nav FLOAT,
    daily_return FLOAT,
    cumulative_return FLOAT,
    spy_return FLOAT,
    spy_cumulative FLOAT,
    alpha_vs_spy FLOAT,
    ew_return FLOAT,
    ew_cumulative FLOAT,
    alpha_vs_ew FLOAT,
    equity_exposure FLOAT,
    cash_exposure FLOAT,
    macro_multiplier FLOAT,
    amri_score FLOAT,
    regime VARCHAR(30),
    drawdown FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_nav_date ON portfolio_nav_daily(date DESC);

-- Shadow portfolio positions
CREATE TABLE IF NOT EXISTS shadow_portfolio (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    pillar VARCHAR(50),
    pillar_weight FLOAT,
    stock_weight FLOAT,
    daily_return FLOAT,
    weighted_return FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_shadow_portfolio_date ON shadow_portfolio(date DESC);
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run portfolio tracker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shadow Portfolio Tracker")
    parser.add_argument("--history", type=int, default=0, help="Show N days of history")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    args = parser.parse_args()
    
    tracker = PortfolioTracker()
    
    print(f"\n{'='*60}")
    print("SHADOW PORTFOLIO TRACKER")
    print(f"{'='*60}\n")
    
    if args.history > 0:
        history = tracker.get_history(args.history)
        print(f"Last {len(history)} days:\n")
        
        for snap in history[:10]:
            print(f"  {snap.date}: NAV=${snap.nav:,.0f} ({snap.cumulative_return*100:+.2f}%) | Alpha: {snap.alpha_vs_spy*100:+.2f}%")
    else:
        snapshot = tracker.calculate_daily()
        
        print(f"Date: {snapshot.date}")
        print(f"\n{'='*40}")
        print("PORTFOLIO METRICS")
        print(f"{'='*40}")
        print(f"  NAV:              ${snapshot.nav:,.0f}")
        print(f"  Daily Return:     {snapshot.daily_return*100:+.2f}%")
        print(f"  Cumulative:       {snapshot.cumulative_return*100:+.2f}%")
        
        print(f"\n{'='*40}")
        print("VS BENCHMARKS")
        print(f"{'='*40}")
        print(f"  SPY Return:       {snapshot.spy_return*100:+.2f}%")
        print(f"  SPY Cumulative:   {snapshot.spy_cumulative*100:+.2f}%")
        print(f"  Alpha vs SPY:     {snapshot.alpha_vs_spy*100:+.2f}%")
        print(f"\n  EW Return:        {snapshot.ew_return*100:+.2f}%")
        print(f"  EW Cumulative:    {snapshot.ew_cumulative*100:+.2f}%")
        print(f"  Alpha vs EW:      {snapshot.alpha_vs_ew*100:+.2f}%")
        
        print(f"\n{'='*40}")
        print("EXPOSURE")
        print(f"{'='*40}")
        print(f"  Equity:           {snapshot.equity_exposure*100:.0f}%")
        print(f"  Cash:             {snapshot.cash_exposure*100:.0f}%")
        print(f"  Macro Multiplier: {snapshot.macro_multiplier:.2f}")
        print(f"  AMRI Score:       {snapshot.amri_score:.1f}")
        print(f"  Regime:           {snapshot.regime}")
        
        print(f"\n{'='*40}")
        print("RISK")
        print(f"{'='*40}")
        print(f"  Drawdown:         {snapshot.drawdown*100:.2f}%")
        
        if args.save:
            if tracker.save_to_supabase(snapshot):
                print(f"\n✅ Saved to Supabase")


if __name__ == "__main__":
    main()
