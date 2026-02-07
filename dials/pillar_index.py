# """
# PILLAR INDEX CALCULATOR
# =======================
# Calculates daily returns and momentum for each of the 6 pillars.
# Foundation module - MCI, BreadthDial, and PortfolioTracker depend on this.

# Pillars:
# - Infrastructure & Energy (16 stocks)
# - Enterprise Adoption (13 stocks)
# - Productivity & Labor (3 stocks)
# - Demand Dynamics (3 stocks)
# - Macro & Policy (4 stocks)
# - Financial & Market (4 stocks)

# Output:
# - Daily returns per pillar
# - Cumulative index (starting at 100)
# - Momentum: 5D, 1M (21d), 3M (63d), 6M (126d), YTD
# """

# import os
# import logging
# from datetime import datetime, timedelta
# from typing import Dict, List, Optional, Tuple
# from dataclasses import dataclass, field, asdict
# import json
# from dotenv import load_dotenv

# load_dotenv()

# import pandas as pd
# import numpy as np

# # Try yfinance, but it may be blocked
# try:
#     import yfinance as yf
#     YFINANCE_AVAILABLE = True
# except ImportError:
#     YFINANCE_AVAILABLE = False

# from supabase import create_client, Client

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Pillar mapping (matches Google Sheets Universe)
# PILLAR_MAP = {
#     "Infrastructure & Energy": "Infrastructure",
#     "Enterprise Adoption": "Enterprise", 
#     "Macro & Policy": "Macro",
#     "Financial & Market": "Financial",
#     "Productivity & Labor": "Productivity",
#     "Demand Dynamics": "Demand"
# }

# # Full pillar definitions with tickers and weights
# # UPDATED 2026-01-14: Synced with Google Sheets Universe tab
# PILLAR_STOCKS = {
#     "Infrastructure & Energy": {
#         "tickers": ["TSM", "ASML", "NVDA", "AMD", "MU", "AVGO", "VRT", 
#                    "CEG", "NRG", "LRCX", "AMAT", "SMCI", "JKS", "RUN", "FSLR", "ENPH"],
#         "weights": None  # Equal weight if None
#     },
#     "Enterprise Adoption": {
#         "tickers": ["MSFT", "AMZN", "GOOGL", "SNOW", "PLTR", "ADBE", "ORCL",
#                    "MDB", "DDOG", "ZS", "NET", "PANW", "CRWD"],
#         "weights": None
#     },
#     "Productivity & Labor": {
#         "tickers": ["META", "CRM", "NOW"],
#         "weights": None
#     },
#     "Demand Dynamics": {
#         "tickers": ["TSLA", "SHOP", "UBER"],
#         "weights": None
#     },
#     "Macro & Policy": {
#         "tickers": ["INTC", "NXPI", "QCOM", "ON"],
#         "weights": None
#     },
#     "Financial & Market": {
#         "tickers": ["EQIX", "DLR", "GS", "MS"],
#         "weights": None
#     }
# }

# # Ticker weights from Google Sheets Universe (inverse market-cap style)
# # UPDATED 2026-01-14: Synced with Google Sheets Universe Column L
# TICKER_WEIGHTS = {
#     "TSM": 0.023256,
#     "ASML": 0.023810,
#     "NVDA": 0.024390,
#     "AMD": 0.025000,
#     "MU": 0.025641,
#     "INTC": 0.026316,
#     "AVGO": 0.027027,
#     "VRT": 0.027778,
#     "CEG": 0.028571,
#     "NRG": 0.029412,
#     "EQIX": 0.030303,
#     "DLR": 0.031250,
#     "MSFT": 0.032258,
#     "AMZN": 0.033333,
#     "GOOGL": 0.034483,
#     "META": 0.035714,
#     "CRM": 0.037037,
#     "NOW": 0.038462,
#     "SNOW": 0.040000,
#     "PLTR": 0.041667,
#     "ADBE": 0.043478,
#     "ORCL": 0.045455,
#     "MDB": 0.047619,
#     "DDOG": 0.050000,
#     "ZS": 0.052632,
#     "NET": 0.055556,
#     "NXPI": 0.058824,
#     "QCOM": 0.062500,
#     "ON": 0.066667,
#     "LRCX": 0.071429,
#     "AMAT": 0.076923,
#     "TSLA": 0.083333,
#     "SHOP": 0.090909,
#     "UBER": 0.100000,
#     "PANW": 0.111111,
#     "CRWD": 0.125000,
#     "SMCI": 0.142857,
#     "JKS": 0.166667,
#     "RUN": 0.200000,
#     "FSLR": 0.250000,
#     "ENPH": 0.333333,
#     "GS": 0.500000,
#     "MS": 1.000000,
# }

# # Momentum periods (trading days)
# MOMENTUM_PERIODS = {
#     "5D": 5,
#     "1M": 21,
#     "3M": 63,
#     "6M": 126
# }

# # Short names for database columns
# PILLAR_SHORT_NAMES = ["Infrastructure", "Enterprise", "Macro", "Financial", "Productivity", "Demand"]


# # =============================================================================
# # DATA CLASSES
# # =============================================================================

# @dataclass
# class PillarDayData:
#     """Single day's pillar data."""
#     date: str
#     # Daily returns
#     infra_return: float = 0.0
#     enterprise_return: float = 0.0
#     macro_return: float = 0.0
#     financial_return: float = 0.0
#     productivity_return: float = 0.0
#     demand_return: float = 0.0
#     # Index values (cumulative, starting at 100)
#     infra_index: float = 100.0
#     enterprise_index: float = 100.0
#     macro_index: float = 100.0
#     financial_index: float = 100.0
#     productivity_index: float = 100.0
#     demand_index: float = 100.0
#     # 5D momentum
#     infra_5d: float = 0.0
#     enterprise_5d: float = 0.0
#     macro_5d: float = 0.0
#     financial_5d: float = 0.0
#     productivity_5d: float = 0.0
#     demand_5d: float = 0.0
#     # 1M momentum
#     infra_1m: float = 0.0
#     enterprise_1m: float = 0.0
#     macro_1m: float = 0.0
#     financial_1m: float = 0.0
#     productivity_1m: float = 0.0
#     demand_1m: float = 0.0
#     # 3M momentum
#     infra_3m: float = 0.0
#     enterprise_3m: float = 0.0
#     macro_3m: float = 0.0
#     financial_3m: float = 0.0
#     productivity_3m: float = 0.0
#     demand_3m: float = 0.0
#     # 6M momentum
#     infra_6m: float = 0.0
#     enterprise_6m: float = 0.0
#     macro_6m: float = 0.0
#     financial_6m: float = 0.0
#     productivity_6m: float = 0.0
#     demand_6m: float = 0.0
#     # YTD momentum
#     infra_ytd: float = 0.0
#     enterprise_ytd: float = 0.0
#     macro_ytd: float = 0.0
#     financial_ytd: float = 0.0
#     productivity_ytd: float = 0.0
#     demand_ytd: float = 0.0


# @dataclass
# class PillarSignal:
#     """Pillar signal for portfolio decisions."""
#     pillar: str
#     index_value: float
#     momentum_5d: float
#     momentum_1m: float
#     momentum_3m: float
#     signal: str  # LEADING, NEUTRAL, WEAKENING, STRESSED
#     strength: float  # 0-100


# # =============================================================================
# # PILLAR INDEX CALCULATOR
# # =============================================================================

# class PillarIndexCalculator:
#     """
#     Calculates pillar indices and momentum from price data.
    
#     Usage:
#         calc = PillarIndexCalculator()
#         results = calc.calculate(days=252)
#         calc.save_to_supabase(results)
#     """
    
#     def __init__(self, supabase_url: str = None, supabase_key: str = None):
#         """Initialize with optional Supabase credentials."""
#         self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
#         self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
#         self.supabase: Optional[Client] = None
        
#         if self.supabase_url and self.supabase_key:
#             self.supabase = create_client(self.supabase_url, self.supabase_key)
    
#     def get_all_tickers(self) -> List[str]:
#         """Get all tickers across all pillars."""
#         all_tickers = []
#         for pillar_data in PILLAR_STOCKS.values():
#             all_tickers.extend(pillar_data["tickers"])
#         return list(set(all_tickers))
    
#     def fetch_price_history(self, days: int = 252) -> pd.DataFrame:
#         """
#         Fetch price history for all tickers.
        
#         Data flow: Supabase (updated by price_sync.py) → yfinance fallback
        
#         Args:
#             days: Number of trading days to fetch
            
#         Returns:
#             DataFrame with columns: Date, Ticker, Close
#         """
#         tickers = self.get_all_tickers()
#         logger.info(f"Fetching price history for {len(tickers)} tickers, {days} days")
        
#         # Primary: Supabase (should have fresh data from price_sync.py)
#         if self.supabase:
#             df = self._fetch_from_supabase(tickers, days)
#             if df is not None and len(df) > 0:
#                 # Check if data is recent (within 3 trading days)
#                 latest_date = df["date"].max()
#                 days_old = (datetime.now() - pd.to_datetime(latest_date)).days
#                 if days_old <= 4:  # Allow for weekends
#                     logger.info(f"Loaded {len(df)} rows from Supabase (latest: {latest_date.strftime('%Y-%m-%d')})")
#                     return df
#                 else:
#                     logger.warning(f"Supabase data is {days_old} days old - run price_sync.py first!")
        
#         # Fallback: yfinance (if Supabase empty or stale)
#         logger.warning("Supabase data unavailable, falling back to yfinance")
#         if YFINANCE_AVAILABLE:
#             return self._fetch_from_yfinance(tickers, days)
        
#         raise RuntimeError("No price data source available. Run price_sync.py first!")
    
#     def _fetch_from_twelvedata(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
#         """Fetch from TwelveData API (same source as Google Sheets)."""
#         try:
#             from .twelvedata_client import TwelveDataClient
            
#             client = TwelveDataClient()
#             if not client.api_key:
#                 logger.info("TwelveData API key not set, skipping")
#                 return None
            
#             logger.info("Fetching from TwelveData...")
#             df = client.fetch_batch_time_series(tickers, outputsize=days)
            
#             if not df.empty:
#                 logger.info(f"Fetched {len(df)} rows from TwelveData")
#                 return df
                
#         except Exception as e:
#             logger.warning(f"TwelveData fetch failed: {e}")
        
#         return None
    
#     def _fetch_from_supabase(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
#         """Fetch from Supabase price_history table with pagination."""
#         try:
#             start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
            
#             # Paginate to handle Supabase 1000-row server limit
#             all_data = []
#             batch_size = 1000
#             offset = 0
            
#             while True:
#                 response = self.supabase.table("price_history") \
#                     .select("date, ticker, close") \
#                     .in_("ticker", tickers) \
#                     .gte("date", start_date) \
#                     .order("date", desc=True) \
#                     .range(offset, offset + batch_size - 1) \
#                     .execute()
                
#                 if not response.data:
#                     break
                    
#                 all_data.extend(response.data)
                
#                 if len(response.data) < batch_size:
#                     break  # Last page
                    
#                 offset += batch_size
            
#             if all_data:
#                 logger.info(f"Fetched {len(all_data)} total rows via pagination")
#                 df = pd.DataFrame(all_data)
#                 df["date"] = pd.to_datetime(df["date"])
#                 return df
#         except Exception as e:
#             logger.warning(f"Supabase fetch failed: {e}")
        
#         return None
    
#     def _fetch_from_yfinance(self, tickers: List[str], days: int) -> pd.DataFrame:
#         """Fetch from Yahoo Finance."""
#         logger.info("Fetching from Yahoo Finance...")
        
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=days * 2)  # Extra buffer for weekends
        
#         all_data = []
        
#         for ticker in tickers:
#             try:
#                 stock = yf.Ticker(ticker)
#                 hist = stock.history(start=start_date, end=end_date)
                
#                 if len(hist) > 0:
#                     for date, row in hist.iterrows():
#                         all_data.append({
#                             "date": date.strftime("%Y-%m-%d"),
#                             "ticker": ticker,
#                             "close": row["Close"]
#                         })
#             except Exception as e:
#                 logger.warning(f"Failed to fetch {ticker}: {e}")
        
#         df = pd.DataFrame(all_data)
#         df["date"] = pd.to_datetime(df["date"])
#         logger.info(f"Fetched {len(df)} rows from yfinance")
#         return df
    
#     def calculate_pillar_returns(self, price_df: pd.DataFrame) -> Dict[str, pd.Series]:
#         """
#         Calculate daily weighted returns for each pillar.
#         Uses TICKER_WEIGHTS from Google Sheets Universe for weighted averaging.
        
#         Args:
#             price_df: DataFrame with Date, Ticker, Close columns
            
#         Returns:
#             Dict of pillar name -> Series of daily returns indexed by date
#         """
#         # Pivot to get prices per ticker per day
#         pivot = price_df.pivot_table(
#             index="date", 
#             columns="ticker", 
#             values="close"
#         ).sort_index()
        
#         # Calculate daily returns
#         returns = pivot.pct_change()
        
#         pillar_returns = {}
        
#         for pillar_full, pillar_data in PILLAR_STOCKS.items():
#             pillar_short = PILLAR_MAP[pillar_full]
#             tickers = pillar_data["tickers"]
            
#             # Get returns for tickers in this pillar
#             available = [t for t in tickers if t in returns.columns]
            
#             if not available:
#                 logger.warning(f"No tickers available for {pillar_short}")
#                 pillar_returns[pillar_short] = pd.Series(0, index=returns.index)
#                 continue
            
#             pillar_rets = returns[available]
            
#             # Use TICKER_WEIGHTS for weighted average (matching Google Sheets)
#             weights = np.array([TICKER_WEIGHTS.get(t, 1.0) for t in available])
#             weights = weights / weights.sum()  # Normalize within pillar
#             pillar_returns[pillar_short] = (pillar_rets * weights).sum(axis=1)
        
#         return pillar_returns
        
#         return pillar_returns
    
#     def build_cumulative_index(self, pillar_returns: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
#         """
#         Build cumulative index starting at 100.
        
#         Args:
#             pillar_returns: Dict of pillar -> daily return series
            
#         Returns:
#             Dict of pillar -> cumulative index series
#         """
#         pillar_index = {}
        
#         for pillar, returns in pillar_returns.items():
#             # Start at 100, compound returns
#             index_values = (1 + returns.fillna(0)).cumprod() * 100
#             pillar_index[pillar] = index_values
        
#         return pillar_index
    
#     def calculate_momentum(
#         self, 
#         pillar_index: Dict[str, pd.Series],
#         periods: Dict[str, int] = None
#     ) -> Dict[str, Dict[str, pd.Series]]:
#         """
#         Calculate momentum for various periods.
        
#         Args:
#             pillar_index: Dict of pillar -> index series
#             periods: Dict of period name -> lookback days
            
#         Returns:
#             Dict of pillar -> Dict of period -> momentum series
#         """
#         if periods is None:
#             periods = MOMENTUM_PERIODS
        
#         momentum = {pillar: {} for pillar in pillar_index}
        
#         for pillar, index in pillar_index.items():
#             for period_name, lookback in periods.items():
#                 # Momentum = (current - lookback) / lookback
#                 shifted = index.shift(lookback)
#                 mom = (index - shifted) / shifted
#                 momentum[pillar][period_name] = mom
            
#             # YTD momentum
#             ytd_start = self._get_ytd_start_date(index.index)
#             if ytd_start and ytd_start in index.index:
#                 ytd_start_val = index[ytd_start]
#                 momentum[pillar]["YTD"] = (index - ytd_start_val) / ytd_start_val
#             else:
#                 momentum[pillar]["YTD"] = pd.Series(0, index=index.index)
        
#         return momentum
    
#     def _get_ytd_start_date(self, dates: pd.DatetimeIndex) -> Optional[str]:
#         """Find first trading day of current year."""
#         current_year = datetime.now().year
#         year_dates = [d for d in dates if d.year == current_year]
#         if year_dates:
#             return min(year_dates)
#         return None
    
#     def calculate(self, days: int = 252) -> List[PillarDayData]:
#         """
#         Main calculation: fetch data and compute all pillar metrics.
        
#         Args:
#             days: Number of trading days to calculate
            
#         Returns:
#             List of PillarDayData objects, newest first
#         """
#         logger.info(f"Calculating pillar index for {days} days")
        
#         # Fetch price history
#         price_df = self.fetch_price_history(days)
        
#         # Calculate pillar returns
#         pillar_returns = self.calculate_pillar_returns(price_df)
        
#         # Build cumulative index
#         pillar_index = self.build_cumulative_index(pillar_returns)
        
#         # Calculate momentum
#         momentum = self.calculate_momentum(pillar_index)
        
#         # Build results
#         results = []
        
#         # Get all dates (sorted descending - newest first)
#         all_dates = sorted(pillar_returns["Infrastructure"].index, reverse=True)
        
#         for date in all_dates[:days]:
#             date_str = date.strftime("%Y-%m-%d")
            
#             data = PillarDayData(date=date_str)
            
#             # Daily returns
#             data.infra_return = pillar_returns["Infrastructure"].get(date, 0) or 0
#             data.enterprise_return = pillar_returns["Enterprise"].get(date, 0) or 0
#             data.macro_return = pillar_returns["Macro"].get(date, 0) or 0
#             data.financial_return = pillar_returns["Financial"].get(date, 0) or 0
#             data.productivity_return = pillar_returns["Productivity"].get(date, 0) or 0
#             data.demand_return = pillar_returns["Demand"].get(date, 0) or 0
            
#             # Index values
#             data.infra_index = pillar_index["Infrastructure"].get(date, 100) or 100
#             data.enterprise_index = pillar_index["Enterprise"].get(date, 100) or 100
#             data.macro_index = pillar_index["Macro"].get(date, 100) or 100
#             data.financial_index = pillar_index["Financial"].get(date, 100) or 100
#             data.productivity_index = pillar_index["Productivity"].get(date, 100) or 100
#             data.demand_index = pillar_index["Demand"].get(date, 100) or 100
            
#             # 5D momentum
#             data.infra_5d = momentum["Infrastructure"]["5D"].get(date, 0) or 0
#             data.enterprise_5d = momentum["Enterprise"]["5D"].get(date, 0) or 0
#             data.macro_5d = momentum["Macro"]["5D"].get(date, 0) or 0
#             data.financial_5d = momentum["Financial"]["5D"].get(date, 0) or 0
#             data.productivity_5d = momentum["Productivity"]["5D"].get(date, 0) or 0
#             data.demand_5d = momentum["Demand"]["5D"].get(date, 0) or 0
            
#             # 1M momentum
#             data.infra_1m = momentum["Infrastructure"]["1M"].get(date, 0) or 0
#             data.enterprise_1m = momentum["Enterprise"]["1M"].get(date, 0) or 0
#             data.macro_1m = momentum["Macro"]["1M"].get(date, 0) or 0
#             data.financial_1m = momentum["Financial"]["1M"].get(date, 0) or 0
#             data.productivity_1m = momentum["Productivity"]["1M"].get(date, 0) or 0
#             data.demand_1m = momentum["Demand"]["1M"].get(date, 0) or 0
            
#             # 3M momentum
#             data.infra_3m = momentum["Infrastructure"]["3M"].get(date, 0) or 0
#             data.enterprise_3m = momentum["Enterprise"]["3M"].get(date, 0) or 0
#             data.macro_3m = momentum["Macro"]["3M"].get(date, 0) or 0
#             data.financial_3m = momentum["Financial"]["3M"].get(date, 0) or 0
#             data.productivity_3m = momentum["Productivity"]["3M"].get(date, 0) or 0
#             data.demand_3m = momentum["Demand"]["3M"].get(date, 0) or 0
            
#             # 6M momentum
#             data.infra_6m = momentum["Infrastructure"]["6M"].get(date, 0) or 0
#             data.enterprise_6m = momentum["Enterprise"]["6M"].get(date, 0) or 0
#             data.macro_6m = momentum["Macro"]["6M"].get(date, 0) or 0
#             data.financial_6m = momentum["Financial"]["6M"].get(date, 0) or 0
#             data.productivity_6m = momentum["Productivity"]["6M"].get(date, 0) or 0
#             data.demand_6m = momentum["Demand"]["6M"].get(date, 0) or 0
            
#             # YTD momentum
#             data.infra_ytd = momentum["Infrastructure"]["YTD"].get(date, 0) or 0
#             data.enterprise_ytd = momentum["Enterprise"]["YTD"].get(date, 0) or 0
#             data.macro_ytd = momentum["Macro"]["YTD"].get(date, 0) or 0
#             data.financial_ytd = momentum["Financial"]["YTD"].get(date, 0) or 0
#             data.productivity_ytd = momentum["Productivity"]["YTD"].get(date, 0) or 0
#             data.demand_ytd = momentum["Demand"]["YTD"].get(date, 0) or 0
            
#             results.append(data)
        
#         logger.info(f"Calculated {len(results)} days of pillar data")
#         return results
    
#     def get_pillar_signals(self, data: List[PillarDayData] = None) -> List[PillarSignal]:
#         """
#         Generate pillar signals based on momentum.
        
#         Signal logic:
#         - LEADING: 5D > 0 AND 1M > 0 AND 3M > 0
#         - NEUTRAL: Mixed signals
#         - WEAKENING: 5D < 0 OR 1M < 0
#         - STRESSED: 5D < 0 AND 1M < 0 AND 3M < 0
        
#         Returns:
#             List of PillarSignal objects for most recent date
#         """
#         if data is None:
#             data = self.calculate(days=30)
        
#         if not data:
#             return []
        
#         latest = data[0]  # Most recent
        
#         signals = []
        
#         pillar_mapping = [
#             ("Infrastructure", latest.infra_index, latest.infra_5d, latest.infra_1m, latest.infra_3m),
#             ("Enterprise", latest.enterprise_index, latest.enterprise_5d, latest.enterprise_1m, latest.enterprise_3m),
#             ("Macro", latest.macro_index, latest.macro_5d, latest.macro_1m, latest.macro_3m),
#             ("Financial", latest.financial_index, latest.financial_5d, latest.financial_1m, latest.financial_3m),
#             ("Productivity", latest.productivity_index, latest.productivity_5d, latest.productivity_1m, latest.productivity_3m),
#             ("Demand", latest.demand_index, latest.demand_5d, latest.demand_1m, latest.demand_3m),
#         ]
        
#         for pillar, index_val, m5d, m1m, m3m in pillar_mapping:
#             # Determine signal
#             if m5d > 0 and m1m > 0 and m3m > 0:
#                 signal = "LEADING"
#                 strength = min(100, (m5d + m1m + m3m) * 100 / 3 + 50)
#             elif m5d < 0 and m1m < 0 and m3m < 0:
#                 signal = "STRESSED"
#                 strength = max(0, 50 + (m5d + m1m + m3m) * 100 / 3)
#             elif m5d < 0 or m1m < 0:
#                 signal = "WEAKENING"
#                 strength = 25 + (m3m * 50 if m3m > 0 else 0)
#             else:
#                 signal = "NEUTRAL"
#                 strength = 50
            
#             signals.append(PillarSignal(
#                 pillar=pillar,
#                 index_value=index_val,
#                 momentum_5d=m5d,
#                 momentum_1m=m1m,
#                 momentum_3m=m3m,
#                 signal=signal,
#                 strength=strength
#             ))
        
#         return signals
    
#     def save_to_supabase(self, data: List[PillarDayData]) -> int:
#         """
#         Save pillar data to Supabase.
        
#         Args:
#             data: List of PillarDayData objects
            
#         Returns:
#             Number of rows upserted
#         """
#         if not self.supabase:
#             logger.warning("Supabase not configured")
#             return 0
        
#         rows = []
#         for d in data:
#             row = asdict(d)
#             # Convert NaN to None
#             for k, v in row.items():
#                 if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
#                     row[k] = None
#             rows.append(row)
        
#         try:
#             # Upsert (insert or update on conflict)
#             response = self.supabase.table("pillar_index_daily") \
#                 .upsert(rows, on_conflict="date") \
#                 .execute()
            
#             logger.info(f"Upserted {len(rows)} rows to pillar_index_daily")
#             return len(rows)
            
#         except Exception as e:
#             logger.error(f"Supabase save failed: {e}")
#             return 0
    
#     def get_latest(self) -> Optional[PillarDayData]:
#         """Get most recent pillar data from Supabase."""
#         if not self.supabase:
#             return None
        
#         try:
#             response = self.supabase.table("pillar_index_daily") \
#                 .select("*") \
#                 .order("date", desc=True) \
#                 .limit(1) \
#                 .execute()
            
#             if response.data:
#                 return PillarDayData(**response.data[0])
#         except Exception as e:
#             logger.error(f"Failed to get latest: {e}")
        
#         return None


# # =============================================================================
# # SUPABASE TABLE CREATION
# # =============================================================================

# CREATE_TABLE_SQL = """
# CREATE TABLE IF NOT EXISTS pillar_index_daily (
#     id SERIAL PRIMARY KEY,
#     date DATE UNIQUE NOT NULL,
#     -- Daily returns
#     infra_return FLOAT,
#     enterprise_return FLOAT,
#     macro_return FLOAT,
#     financial_return FLOAT,
#     productivity_return FLOAT,
#     demand_return FLOAT,
#     -- Index values (cumulative, starting at 100)
#     infra_index FLOAT,
#     enterprise_index FLOAT,
#     macro_index FLOAT,
#     financial_index FLOAT,
#     productivity_index FLOAT,
#     demand_index FLOAT,
#     -- 5D momentum
#     infra_5d FLOAT,
#     enterprise_5d FLOAT,
#     macro_5d FLOAT,
#     financial_5d FLOAT,
#     productivity_5d FLOAT,
#     demand_5d FLOAT,
#     -- 1M momentum
#     infra_1m FLOAT,
#     enterprise_1m FLOAT,
#     macro_1m FLOAT,
#     financial_1m FLOAT,
#     productivity_1m FLOAT,
#     demand_1m FLOAT,
#     -- 3M momentum
#     infra_3m FLOAT,
#     enterprise_3m FLOAT,
#     macro_3m FLOAT,
#     financial_3m FLOAT,
#     productivity_3m FLOAT,
#     demand_3m FLOAT,
#     -- 6M momentum
#     infra_6m FLOAT,
#     enterprise_6m FLOAT,
#     macro_6m FLOAT,
#     financial_6m FLOAT,
#     productivity_6m FLOAT,
#     demand_6m FLOAT,
#     -- YTD momentum
#     infra_ytd FLOAT,
#     enterprise_ytd FLOAT,
#     macro_ytd FLOAT,
#     financial_ytd FLOAT,
#     productivity_ytd FLOAT,
#     demand_ytd FLOAT,
#     -- Metadata
#     created_at TIMESTAMPTZ DEFAULT NOW()
# );

# -- Index for faster lookups
# CREATE INDEX IF NOT EXISTS idx_pillar_index_date ON pillar_index_daily(date DESC);
# """


# # =============================================================================
# # CLI & TESTING
# # =============================================================================

# def main():
#     """Run pillar index calculation."""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Calculate Pillar Index")
#     parser.add_argument("--days", type=int, default=30, help="Days to calculate")
#     parser.add_argument("--save", action="store_true", help="Save to Supabase")
#     parser.add_argument("--signals", action="store_true", help="Show pillar signals")
#     args = parser.parse_args()
    
#     calc = PillarIndexCalculator()
    
#     print(f"\n{'='*60}")
#     print("PILLAR INDEX CALCULATOR")
#     print(f"{'='*60}\n")
    
#     # Calculate
#     data = calc.calculate(days=args.days)
    
#     if not data:
#         print("No data calculated!")
#         return
    
#     # Show latest
#     latest = data[0]
#     print(f"Date: {latest.date}")
#     print(f"\n{'='*40}")
#     print("DAILY RETURNS")
#     print(f"{'='*40}")
#     print(f"  Infrastructure: {latest.infra_return*100:+.2f}%")
#     print(f"  Enterprise:     {latest.enterprise_return*100:+.2f}%")
#     print(f"  Macro:          {latest.macro_return*100:+.2f}%")
#     print(f"  Financial:      {latest.financial_return*100:+.2f}%")
#     print(f"  Productivity:   {latest.productivity_return*100:+.2f}%")
#     print(f"  Demand:         {latest.demand_return*100:+.2f}%")
    
#     print(f"\n{'='*40}")
#     print("INDEX VALUES (Base 100)")
#     print(f"{'='*40}")
#     print(f"  Infrastructure: {latest.infra_index:.1f}")
#     print(f"  Enterprise:     {latest.enterprise_index:.1f}")
#     print(f"  Macro:          {latest.macro_index:.1f}")
#     print(f"  Financial:      {latest.financial_index:.1f}")
#     print(f"  Productivity:   {latest.productivity_index:.1f}")
#     print(f"  Demand:         {latest.demand_index:.1f}")
    
#     print(f"\n{'='*40}")
#     print("MOMENTUM (5D / 1M / 3M)")
#     print(f"{'='*40}")
#     print(f"  Infrastructure: {latest.infra_5d*100:+.1f}% / {latest.infra_1m*100:+.1f}% / {latest.infra_3m*100:+.1f}%")
#     print(f"  Enterprise:     {latest.enterprise_5d*100:+.1f}% / {latest.enterprise_1m*100:+.1f}% / {latest.enterprise_3m*100:+.1f}%")
#     print(f"  Macro:          {latest.macro_5d*100:+.1f}% / {latest.macro_1m*100:+.1f}% / {latest.macro_3m*100:+.1f}%")
#     print(f"  Financial:      {latest.financial_5d*100:+.1f}% / {latest.financial_1m*100:+.1f}% / {latest.financial_3m*100:+.1f}%")
#     print(f"  Productivity:   {latest.productivity_5d*100:+.1f}% / {latest.productivity_1m*100:+.1f}% / {latest.productivity_3m*100:+.1f}%")
#     print(f"  Demand:         {latest.demand_5d*100:+.1f}% / {latest.demand_1m*100:+.1f}% / {latest.demand_3m*100:+.1f}%")
    
#     if args.signals:
#         signals = calc.get_pillar_signals(data)
#         print(f"\n{'='*40}")
#         print("PILLAR SIGNALS")
#         print(f"{'='*40}")
#         for sig in signals:
#             print(f"  {sig.pillar:15} {sig.signal:12} (strength: {sig.strength:.0f})")
    
#     if args.save:
#         saved = calc.save_to_supabase(data)
#         print(f"\n✅ Saved {saved} rows to Supabase")


# if __name__ == "__main__":
#     main()





"""
PILLAR INDEX CALCULATOR
=======================
Calculates daily returns and momentum for each pillar.
Foundation module - MCI, BreadthDial, and PortfolioTracker depend on this.

UPDATED 2026-02-02: 4-Pillar Configuration (52 tickers)
Pillars:
- Infrastructure & Energy (22 stocks)
- Enterprise Demand (17 stocks)
- Consumer Demand (7 stocks)
- Capital (6 stocks)

Deprecated (empty, kept for DB compatibility):
- Macro & Policy
- Productivity & Labor

Output:
- Daily returns per pillar
- Cumulative index (starting at 100)
- Momentum: 5D, 1M (21d), 3M (63d), 6M (126d), YTD
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np

# Try yfinance, but it may be blocked
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - 4 PILLARS (GS ALIGNED)
# UPDATED 2026-02-02: Synced with Google Sheets Universe tab
# =============================================================================

# Pillar mapping (full name -> short name for database columns)
# NOTE: Maps to existing DB column names for compatibility
PILLAR_MAP = {
    "Infrastructure & Energy": "Infrastructure",  # → infra_*
    "Enterprise Demand": "Enterprise",            # → enterprise_* (was Enterprise Adoption)
    "Macro & Policy": "Macro",                    # → macro_* (DEPRECATED - empty)
    "Capital": "Financial",                       # → financial_* (was Financial & Market)
    "Productivity & Labor": "Productivity",       # → productivity_* (DEPRECATED - empty)
    "Consumer Demand": "Demand"                   # → demand_* (was Demand Dynamics)
}

# Full pillar definitions with tickers
# UPDATED 2026-02-02: Synced with Google Sheets Universe tab (4 pillars, 52 tickers)
# NOTE: Macro & Productivity kept with empty tickers for backward compatibility
PILLAR_STOCKS = {
    "Infrastructure & Energy": {
        "tickers": ["EQIX", "DLR", "TSM", "ASML", "NVDA", "AMD", "MU", "AVGO", 
                   "VRT", "CEG", "NRG", "LRCX", "AMAT", "SMCI", "JKS", "RUN", 
                   "FSLR", "ENPH", "INTC", "NXPI", "QCOM", "ON"],
        "weights": None  # Equal weight
    },
    "Enterprise Demand": {
        "tickers": ["MSFT", "AMZN", "GOOGL", "SNOW", "PLTR", "ADBE", "ORCL", 
                   "MDB", "DDOG", "ZS", "NET", "PANW", "CRWD", "CRM", "NOW", 
                   "WDAY", "ADP"],
        "weights": None
    },
    "Productivity & Labor": {
        "tickers": [],  # DEPRECATED - merged into Consumer/Enterprise
        "weights": None
    },
    "Consumer Demand": {
        "tickers": ["TSLA", "SHOP", "UBER", "ABNB", "META", "NFLX", "SPOT"],
        "weights": None
    },
    "Macro & Policy": {
        "tickers": [],  # DEPRECATED - merged into Infrastructure
        "weights": None
    },
    "Capital": {
        "tickers": ["GS", "MS", "BX", "KKR", "APO", "ARES"],
        "weights": None
    }
}

# Ticker weights from Google Sheets Universe (Column L)
# UPDATED 2026-02-02: Added new tickers with estimated weights
TICKER_WEIGHTS = {
    # Infrastructure & Energy (22)
    "EQIX": 0.033333, "DLR": 0.034483, "TSM": 0.035714, "ASML": 0.037037,
    "NVDA": 0.038462, "AMD": 0.040000, "MU": 0.041667, "AVGO": 0.043478,
    "VRT": 0.045455, "CEG": 0.047619, "NRG": 0.050000, "LRCX": 0.052632,
    "AMAT": 0.055556, "SMCI": 0.058824, "JKS": 0.062500, "RUN": 0.066667,
    "FSLR": 0.071429, "ENPH": 0.076923, "INTC": 0.083333, "NXPI": 0.090909,
    "QCOM": 0.100000, "ON": 0.111111,
    # Enterprise Demand (17)
    "MSFT": 0.021277, "AMZN": 0.021739, "GOOGL": 0.022222, "SNOW": 0.022727,
    "PLTR": 0.023256, "ADBE": 0.023810, "ORCL": 0.024390, "MDB": 0.025000,
    "DDOG": 0.025641, "ZS": 0.026316, "NET": 0.027027, "PANW": 0.027778,
    "CRWD": 0.028571, "CRM": 0.029412, "NOW": 0.030303, 
    "WDAY": 0.031000, "ADP": 0.031500,
    # Consumer Demand (7)
    "TSLA": 0.019231, "SHOP": 0.019608, "UBER": 0.020000, "META": 0.020833,
    "ABNB": 0.020500, "NFLX": 0.020600, "SPOT": 0.020700,
    # Capital (6)
    "GS": 0.050000, "MS": 0.050000, 
    "BX": 0.050000, "KKR": 0.050000, "APO": 0.050000, "ARES": 0.050000,
}

# Momentum periods (trading days)
MOMENTUM_PERIODS = {
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126
}

# Short names for database columns (keep all 6 for DB compatibility)
PILLAR_SHORT_NAMES = ["Infrastructure", "Enterprise", "Macro", "Financial", "Productivity", "Demand"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PillarDayData:
    """Single day's pillar data."""
    date: str
    # Daily returns
    infra_return: float = 0.0
    enterprise_return: float = 0.0
    macro_return: float = 0.0
    financial_return: float = 0.0
    productivity_return: float = 0.0
    demand_return: float = 0.0
    # Index values (cumulative, starting at 100)
    infra_index: float = 100.0
    enterprise_index: float = 100.0
    macro_index: float = 100.0
    financial_index: float = 100.0
    productivity_index: float = 100.0
    demand_index: float = 100.0
    # 5D momentum
    infra_5d: float = 0.0
    enterprise_5d: float = 0.0
    macro_5d: float = 0.0
    financial_5d: float = 0.0
    productivity_5d: float = 0.0
    demand_5d: float = 0.0
    # 1M momentum
    infra_1m: float = 0.0
    enterprise_1m: float = 0.0
    macro_1m: float = 0.0
    financial_1m: float = 0.0
    productivity_1m: float = 0.0
    demand_1m: float = 0.0
    # 3M momentum
    infra_3m: float = 0.0
    enterprise_3m: float = 0.0
    macro_3m: float = 0.0
    financial_3m: float = 0.0
    productivity_3m: float = 0.0
    demand_3m: float = 0.0
    # 6M momentum
    infra_6m: float = 0.0
    enterprise_6m: float = 0.0
    macro_6m: float = 0.0
    financial_6m: float = 0.0
    productivity_6m: float = 0.0
    demand_6m: float = 0.0
    # YTD momentum
    infra_ytd: float = 0.0
    enterprise_ytd: float = 0.0
    macro_ytd: float = 0.0
    financial_ytd: float = 0.0
    productivity_ytd: float = 0.0
    demand_ytd: float = 0.0


@dataclass
class PillarSignal:
    """Pillar signal for portfolio decisions."""
    pillar: str
    index_value: float
    momentum_5d: float
    momentum_1m: float
    momentum_3m: float
    signal: str  # LEADING, NEUTRAL, WEAKENING, STRESSED
    strength: float  # 0-100


# =============================================================================
# PILLAR INDEX CALCULATOR
# =============================================================================

class PillarIndexCalculator:
    """
    Calculates pillar indices and momentum from price data.
    
    Usage:
        calc = PillarIndexCalculator()
        results = calc.calculate(days=252)
        calc.save_to_supabase(results)
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize with optional Supabase credentials."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.supabase: Optional[Client] = None
        
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers across all pillars."""
        all_tickers = []
        for pillar_data in PILLAR_STOCKS.values():
            all_tickers.extend(pillar_data["tickers"])
        return list(set(all_tickers))
    
    def fetch_price_history(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch price history for all tickers.
        
        Data flow: Supabase (updated by price_sync.py) → yfinance fallback
        """
        tickers = self.get_all_tickers()
        logger.info(f"Fetching price history for {len(tickers)} tickers, {days} days")
        
        # Primary: Supabase (should have fresh data from price_sync.py)
        if self.supabase:
            df = self._fetch_from_supabase(tickers, days)
            if df is not None and len(df) > 0:
                latest_date = df["date"].max()
                days_old = (datetime.now() - pd.to_datetime(latest_date)).days
                if days_old <= 4:  # Allow for weekends
                    logger.info(f"Loaded {len(df)} rows from Supabase (latest: {latest_date.strftime('%Y-%m-%d')})")
                    return df
                else:
                    logger.warning(f"Supabase data is {days_old} days old - run price_sync.py first!")
        
        # Fallback: yfinance (if Supabase empty or stale)
        logger.warning("Supabase data unavailable, falling back to yfinance")
        if YFINANCE_AVAILABLE:
            return self._fetch_from_yfinance(tickers, days)
        
        raise RuntimeError("No price data source available. Run price_sync.py first!")
    
    def _fetch_from_twelvedata(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
        """Fetch from TwelveData API (same source as Google Sheets)."""
        try:
            from .twelvedata_client import TwelveDataClient
            
            client = TwelveDataClient()
            if not client.api_key:
                logger.info("TwelveData API key not set, skipping")
                return None
            
            logger.info("Fetching from TwelveData...")
            df = client.fetch_batch_time_series(tickers, outputsize=days)
            
            if not df.empty:
                logger.info(f"Fetched {len(df)} rows from TwelveData")
                return df
                
        except Exception as e:
            logger.warning(f"TwelveData fetch failed: {e}")
        
        return None
    
    def _fetch_from_supabase(self, tickers: List[str], days: int) -> Optional[pd.DataFrame]:
        """Fetch from Supabase price_history table with pagination."""
        try:
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
            
            all_data = []
            batch_size = 1000
            offset = 0
            
            while True:
                response = self.supabase.table("price_history") \
                    .select("date, ticker, close") \
                    .in_("ticker", tickers) \
                    .gte("date", start_date) \
                    .order("date", desc=True) \
                    .range(offset, offset + batch_size - 1) \
                    .execute()
                
                if not response.data:
                    break
                    
                all_data.extend(response.data)
                
                if len(response.data) < batch_size:
                    break
                    
                offset += batch_size
            
            if all_data:
                logger.info(f"Fetched {len(all_data)} total rows via pagination")
                df = pd.DataFrame(all_data)
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logger.warning(f"Supabase fetch failed: {e}")
        
        return None
    
    def _fetch_from_yfinance(self, tickers: List[str], days: int) -> pd.DataFrame:
        """Fetch from Yahoo Finance."""
        logger.info("Fetching from Yahoo Finance...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)
        
        all_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) > 0:
                    for date, row in hist.iterrows():
                        all_data.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "close": row["Close"]
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Fetched {len(df)} rows from yfinance")
        return df
    
    def calculate_pillar_returns(self, price_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate daily weighted returns for each pillar."""
        pivot = price_df.pivot_table(
            index="date", 
            columns="ticker", 
            values="close"
        ).sort_index()
        
        returns = pivot.pct_change()
        
        pillar_returns = {}
        
        for pillar_full, pillar_data in PILLAR_STOCKS.items():
            pillar_short = PILLAR_MAP[pillar_full]
            tickers = pillar_data["tickers"]
            
            # Handle deprecated pillars with empty tickers
            if not tickers:
                pillar_returns[pillar_short] = pd.Series(0, index=returns.index)
                continue
            
            available = [t for t in tickers if t in returns.columns]
            
            if not available:
                logger.warning(f"No tickers available for {pillar_short}")
                pillar_returns[pillar_short] = pd.Series(0, index=returns.index)
                continue
            
            pillar_rets = returns[available]
            
            weights = np.array([TICKER_WEIGHTS.get(t, 1.0) for t in available])
            weights = weights / weights.sum()
            pillar_returns[pillar_short] = (pillar_rets * weights).sum(axis=1)
        
        return pillar_returns
    
    def build_cumulative_index(self, pillar_returns: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Build cumulative index starting at 100."""
        pillar_index = {}
        
        for pillar, returns in pillar_returns.items():
            index_values = (1 + returns.fillna(0)).cumprod() * 100
            pillar_index[pillar] = index_values
        
        return pillar_index
    
    def calculate_momentum(
        self, 
        pillar_index: Dict[str, pd.Series],
        periods: Dict[str, int] = None
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Calculate momentum for various periods."""
        if periods is None:
            periods = MOMENTUM_PERIODS
        
        momentum = {pillar: {} for pillar in pillar_index}
        
        for pillar, index in pillar_index.items():
            for period_name, lookback in periods.items():
                shifted = index.shift(lookback)
                mom = (index - shifted) / shifted
                momentum[pillar][period_name] = mom
            
            ytd_start = self._get_ytd_start_date(index.index)
            if ytd_start and ytd_start in index.index:
                ytd_start_val = index[ytd_start]
                momentum[pillar]["YTD"] = (index - ytd_start_val) / ytd_start_val
            else:
                momentum[pillar]["YTD"] = pd.Series(0, index=index.index)
        
        return momentum
    
    def _get_ytd_start_date(self, dates: pd.DatetimeIndex) -> Optional[str]:
        """Find first trading day of current year."""
        current_year = datetime.now().year
        year_dates = [d for d in dates if d.year == current_year]
        if year_dates:
            return min(year_dates)
        return None
    
    def calculate(self, days: int = 252) -> List[PillarDayData]:
        """Main calculation: fetch data and compute all pillar metrics."""
        logger.info(f"Calculating pillar index for {days} days")
        
        price_df = self.fetch_price_history(days)
        pillar_returns = self.calculate_pillar_returns(price_df)
        pillar_index = self.build_cumulative_index(pillar_returns)
        momentum = self.calculate_momentum(pillar_index)
        
        results = []
        all_dates = sorted(pillar_returns["Infrastructure"].index, reverse=True)
        
        for date in all_dates[:days]:
            date_str = date.strftime("%Y-%m-%d")
            
            data = PillarDayData(date=date_str)
            
            # Daily returns
            data.infra_return = pillar_returns["Infrastructure"].get(date, 0) or 0
            data.enterprise_return = pillar_returns["Enterprise"].get(date, 0) or 0
            data.macro_return = pillar_returns["Macro"].get(date, 0) or 0
            data.financial_return = pillar_returns["Financial"].get(date, 0) or 0
            data.productivity_return = pillar_returns["Productivity"].get(date, 0) or 0
            data.demand_return = pillar_returns["Demand"].get(date, 0) or 0
            
            # Index values
            data.infra_index = pillar_index["Infrastructure"].get(date, 100) or 100
            data.enterprise_index = pillar_index["Enterprise"].get(date, 100) or 100
            data.macro_index = pillar_index["Macro"].get(date, 100) or 100
            data.financial_index = pillar_index["Financial"].get(date, 100) or 100
            data.productivity_index = pillar_index["Productivity"].get(date, 100) or 100
            data.demand_index = pillar_index["Demand"].get(date, 100) or 100
            
            # 5D momentum
            data.infra_5d = momentum["Infrastructure"]["5D"].get(date, 0) or 0
            data.enterprise_5d = momentum["Enterprise"]["5D"].get(date, 0) or 0
            data.macro_5d = momentum["Macro"]["5D"].get(date, 0) or 0
            data.financial_5d = momentum["Financial"]["5D"].get(date, 0) or 0
            data.productivity_5d = momentum["Productivity"]["5D"].get(date, 0) or 0
            data.demand_5d = momentum["Demand"]["5D"].get(date, 0) or 0
            
            # 1M momentum
            data.infra_1m = momentum["Infrastructure"]["1M"].get(date, 0) or 0
            data.enterprise_1m = momentum["Enterprise"]["1M"].get(date, 0) or 0
            data.macro_1m = momentum["Macro"]["1M"].get(date, 0) or 0
            data.financial_1m = momentum["Financial"]["1M"].get(date, 0) or 0
            data.productivity_1m = momentum["Productivity"]["1M"].get(date, 0) or 0
            data.demand_1m = momentum["Demand"]["1M"].get(date, 0) or 0
            
            # 3M momentum
            data.infra_3m = momentum["Infrastructure"]["3M"].get(date, 0) or 0
            data.enterprise_3m = momentum["Enterprise"]["3M"].get(date, 0) or 0
            data.macro_3m = momentum["Macro"]["3M"].get(date, 0) or 0
            data.financial_3m = momentum["Financial"]["3M"].get(date, 0) or 0
            data.productivity_3m = momentum["Productivity"]["3M"].get(date, 0) or 0
            data.demand_3m = momentum["Demand"]["3M"].get(date, 0) or 0
            
            # 6M momentum
            data.infra_6m = momentum["Infrastructure"]["6M"].get(date, 0) or 0
            data.enterprise_6m = momentum["Enterprise"]["6M"].get(date, 0) or 0
            data.macro_6m = momentum["Macro"]["6M"].get(date, 0) or 0
            data.financial_6m = momentum["Financial"]["6M"].get(date, 0) or 0
            data.productivity_6m = momentum["Productivity"]["6M"].get(date, 0) or 0
            data.demand_6m = momentum["Demand"]["6M"].get(date, 0) or 0
            
            # YTD momentum
            data.infra_ytd = momentum["Infrastructure"]["YTD"].get(date, 0) or 0
            data.enterprise_ytd = momentum["Enterprise"]["YTD"].get(date, 0) or 0
            data.macro_ytd = momentum["Macro"]["YTD"].get(date, 0) or 0
            data.financial_ytd = momentum["Financial"]["YTD"].get(date, 0) or 0
            data.productivity_ytd = momentum["Productivity"]["YTD"].get(date, 0) or 0
            data.demand_ytd = momentum["Demand"]["YTD"].get(date, 0) or 0
            
            results.append(data)
        
        logger.info(f"Calculated {len(results)} days of pillar data")
        return results
    
    def get_pillar_signals(self, data: List[PillarDayData] = None) -> List[PillarSignal]:
        """Generate pillar signals based on momentum."""
        if data is None:
            data = self.calculate(days=30)
        
        if not data:
            return []
        
        latest = data[0]
        signals = []
        
        # Only include active pillars (skip deprecated Macro/Productivity)
        pillar_mapping = [
            ("Infrastructure", latest.infra_index, latest.infra_5d, latest.infra_1m, latest.infra_3m),
            ("Enterprise", latest.enterprise_index, latest.enterprise_5d, latest.enterprise_1m, latest.enterprise_3m),
            ("Financial", latest.financial_index, latest.financial_5d, latest.financial_1m, latest.financial_3m),
            ("Demand", latest.demand_index, latest.demand_5d, latest.demand_1m, latest.demand_3m),
        ]
        
        for pillar, index_val, m5d, m1m, m3m in pillar_mapping:
            if m5d > 0 and m1m > 0 and m3m > 0:
                signal = "LEADING"
                strength = min(100, (m5d + m1m + m3m) * 100 / 3 + 50)
            elif m5d < 0 and m1m < 0 and m3m < 0:
                signal = "STRESSED"
                strength = max(0, 50 + (m5d + m1m + m3m) * 100 / 3)
            elif m5d < 0 or m1m < 0:
                signal = "WEAKENING"
                strength = 25 + (m3m * 50 if m3m > 0 else 0)
            else:
                signal = "NEUTRAL"
                strength = 50
            
            signals.append(PillarSignal(
                pillar=pillar,
                index_value=index_val,
                momentum_5d=m5d,
                momentum_1m=m1m,
                momentum_3m=m3m,
                signal=signal,
                strength=strength
            ))
        
        return signals
    
    def save_to_supabase(self, data: List[PillarDayData]) -> int:
        """Save pillar data to Supabase."""
        if not self.supabase:
            logger.warning("Supabase not configured")
            return 0
        
        rows = []
        for d in data:
            row = asdict(d)
            # Convert numpy types to native Python types
            for k, v in row.items():
                if isinstance(v, (np.int64, np.int32)):
                    row[k] = int(v)
                elif isinstance(v, (np.float64, np.float32, float)):
                    if np.isnan(v) or np.isinf(v):
                        row[k] = None
                    else:
                        row[k] = float(v)
            rows.append(row)
        
        try:
            response = self.supabase.table("pillar_index_daily") \
                .upsert(rows, on_conflict="date") \
                .execute()
            
            logger.info(f"Upserted {len(rows)} rows to pillar_index_daily")
            return len(rows)
            
        except Exception as e:
            logger.error(f"Supabase save failed: {e}")
            return 0
    
    def get_latest(self) -> Optional[PillarDayData]:
        """Get most recent pillar data from Supabase."""
        if not self.supabase:
            return None
        
        try:
            response = self.supabase.table("pillar_index_daily") \
                .select("*") \
                .order("date", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                return PillarDayData(**response.data[0])
        except Exception as e:
            logger.error(f"Failed to get latest: {e}")
        
        return None


# =============================================================================
# SUPABASE TABLE CREATION
# =============================================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pillar_index_daily (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    infra_return FLOAT, enterprise_return FLOAT, macro_return FLOAT,
    financial_return FLOAT, productivity_return FLOAT, demand_return FLOAT,
    infra_index FLOAT, enterprise_index FLOAT, macro_index FLOAT,
    financial_index FLOAT, productivity_index FLOAT, demand_index FLOAT,
    infra_5d FLOAT, enterprise_5d FLOAT, macro_5d FLOAT,
    financial_5d FLOAT, productivity_5d FLOAT, demand_5d FLOAT,
    infra_1m FLOAT, enterprise_1m FLOAT, macro_1m FLOAT,
    financial_1m FLOAT, productivity_1m FLOAT, demand_1m FLOAT,
    infra_3m FLOAT, enterprise_3m FLOAT, macro_3m FLOAT,
    financial_3m FLOAT, productivity_3m FLOAT, demand_3m FLOAT,
    infra_6m FLOAT, enterprise_6m FLOAT, macro_6m FLOAT,
    financial_6m FLOAT, productivity_6m FLOAT, demand_6m FLOAT,
    infra_ytd FLOAT, enterprise_ytd FLOAT, macro_ytd FLOAT,
    financial_ytd FLOAT, productivity_ytd FLOAT, demand_ytd FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pillar_index_date ON pillar_index_daily(date DESC);
"""


# =============================================================================
# CLI & TESTING
# =============================================================================

def main():
    """Run pillar index calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Pillar Index")
    parser.add_argument("--days", type=int, default=30, help="Days to calculate")
    parser.add_argument("--save", action="store_true", help="Save to Supabase")
    parser.add_argument("--signals", action="store_true", help="Show pillar signals")
    args = parser.parse_args()
    
    calc = PillarIndexCalculator()
    
    print(f"\n{'='*60}")
    print("PILLAR INDEX CALCULATOR (4-PILLAR VERSION)")
    print(f"{'='*60}")
    print(f"Pillars: Infrastructure, Enterprise, Demand, Financial")
    print(f"Total tickers: {len(calc.get_all_tickers())}")
    print()
    
    data = calc.calculate(days=args.days)
    
    if not data:
        print("No data calculated!")
        return
    
    latest = data[0]
    print(f"Date: {latest.date}")
    print(f"\n{'='*40}")
    print("INDEX VALUES (Base 100)")
    print(f"{'='*40}")
    print(f"  Infrastructure: {latest.infra_index:.1f}")
    print(f"  Enterprise:     {latest.enterprise_index:.1f}")
    print(f"  Financial:      {latest.financial_index:.1f}")
    print(f"  Demand:         {latest.demand_index:.1f}")
    
    print(f"\n{'='*40}")
    print("MOMENTUM (5D / 1M / 3M)")
    print(f"{'='*40}")
    print(f"  Infrastructure: {latest.infra_5d*100:+.1f}% / {latest.infra_1m*100:+.1f}% / {latest.infra_3m*100:+.1f}%")
    print(f"  Enterprise:     {latest.enterprise_5d*100:+.1f}% / {latest.enterprise_1m*100:+.1f}% / {latest.enterprise_3m*100:+.1f}%")
    print(f"  Financial:      {latest.financial_5d*100:+.1f}% / {latest.financial_1m*100:+.1f}% / {latest.financial_3m*100:+.1f}%")
    print(f"  Demand:         {latest.demand_5d*100:+.1f}% / {latest.demand_1m*100:+.1f}% / {latest.demand_3m*100:+.1f}%")
    
    if args.signals:
        signals = calc.get_pillar_signals(data)
        print(f"\n{'='*40}")
        print("PILLAR SIGNALS")
        print(f"{'='*40}")
        for sig in signals:
            print(f"  {sig.pillar:15} {sig.signal:12} (strength: {sig.strength:.0f})")
    
    if args.save:
        saved = calc.save_to_supabase(data)
        print(f"\n✅ Saved {saved} rows to Supabase")


if __name__ == "__main__":
    main()