#!/usr/bin/env python3
"""
Historical Data Backfill for ARGUS-1 Backtesting

Pulls historical VIX, CAPE, and credit spreads to populate
bubble_index_daily table for backtesting.

Data Sources:
- VIX: Yahoo Finance (^VIX)
- CAPE: Yale Shiller dataset
- Credit Spreads: FRED (BAMLC0A0CM, BAMLH0A0HYM2)

Usage:
    python -m backtest.backfill_historical --days 365
    python -m backtest.backfill_historical --start 2024-01-01 --end 2024-12-31
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataBackfill:
    """Backfill historical market data for backtesting"""
    
    # Thresholds for bubble index calculation
    VIX_THRESHOLDS = {"low": 12, "normal": 20, "elevated": 25, "high": 35}
    CAPE_THRESHOLDS = {"low": 15, "normal": 25, "elevated": 30, "high": 35}
    SPREAD_THRESHOLDS = {"low": 100, "normal": 150, "elevated": 200, "high": 300}
    
    def __init__(self):
        self.client = None
        self._init_client()
        
        # DataFrames
        self.vix_df: Optional[pd.DataFrame] = None
        self.cape_df: Optional[pd.DataFrame] = None
        self.spreads_df: Optional[pd.DataFrame] = None
    
    def _init_client(self):
        """Initialize Supabase client"""
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if url and key:
                self.client = create_client(url, key)
                logger.info("Supabase client initialized")
            else:
                logger.warning("SUPABASE_URL or SUPABASE_KEY not set")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
    
    def fetch_vix(self, start_date: str, end_date: str) -> bool:
        """Fetch VIX data from Yahoo Finance"""
        try:
            import yfinance as yf
            
            logger.info(f"Fetching VIX from {start_date} to {end_date}...")
            
            vix = yf.Ticker("^VIX")
            df = vix.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.error("No VIX data returned")
                return False
            
            self.vix_df = pd.DataFrame({
                'date': df.index.date,
                'vix': df['Close'].values
            })
            self.vix_df['date'] = self.vix_df['date'].astype(str)
            self.vix_df = self.vix_df.set_index('date')
            
            logger.info(f"Loaded {len(self.vix_df)} VIX records")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return False
    
    def fetch_cape(self, start_date: str, end_date: str) -> bool:
        """Fetch CAPE ratio from Yale Shiller dataset"""
        try:
            logger.info("Fetching CAPE from Yale Shiller...")
            
            # Yale Shiller CAPE data URL
            url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
            
            try:
                df = pd.read_excel(url, sheet_name="Data", skiprows=7)
                
                # Clean up the data
                df = df[['Date', 'CAPE']].dropna()
                df.columns = ['date', 'cape']
                
                # Convert date (format: YYYY.MM)
                df['date'] = pd.to_datetime(df['date'].astype(str).str.replace('.', '-') + '-01')
                
                # Filter to date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                
                # Resample to daily (forward fill monthly values)
                df = df.set_index('date')
                df = df.resample('D').ffill()
                df = df.reset_index()
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                
                self.cape_df = df.set_index('date')
                logger.info(f"Loaded {len(self.cape_df)} CAPE records")
                return True
                
            except Exception as e:
                logger.warning(f"Could not fetch from Yale: {e}")
                logger.info("Using fallback CAPE estimate...")
                
                # Fallback: Use approximate current CAPE
                # As of late 2024, CAPE is around 35-38
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                self.cape_df = pd.DataFrame({
                    'date': dates.strftime('%Y-%m-%d'),
                    'cape': 36.0  # Approximate current value
                }).set_index('date')
                
                return True
                
        except Exception as e:
            logger.error(f"Error fetching CAPE: {e}")
            return False
    
    def fetch_credit_spreads(self, start_date: str, end_date: str) -> bool:
        """Fetch credit spreads from FRED"""
        try:
            logger.info("Fetching credit spreads from FRED...")
            
            try:
                from pandas_datareader import data as pdr
                
                # ICE BofA Investment Grade spread
                ig = pdr.DataReader('BAMLC0A0CM', 'fred', start_date, end_date)
                # ICE BofA High Yield spread  
                hy = pdr.DataReader('BAMLH0A0HYM2', 'fred', start_date, end_date)
                
                # Combine and convert to basis points
                spreads = pd.DataFrame({
                    'credit_spread_ig': ig.iloc[:, 0] * 100,  # Convert to bps
                    'credit_spread_hy': hy.iloc[:, 0] * 100
                })
                
                spreads = spreads.ffill().bfill()
                spreads.index = spreads.index.strftime('%Y-%m-%d')
                
                self.spreads_df = spreads
                logger.info(f"Loaded {len(self.spreads_df)} spread records")
                return True
                
            except Exception as e:
                logger.warning(f"FRED fetch failed: {e}")
                logger.info("Using Yahoo Finance for spread proxy...")
                
                # Fallback: Use HYG price movements to estimate spread changes
                import yfinance as yf
                
                hyg = yf.Ticker("HYG").history(start=start_date, end=end_date)
                
                if not hyg.empty:
                    # HYG inverse price proxy for spreads
                    # When HYG drops, spreads widen
                    # Base spread ~350 bps, adjust by HYG return from start
                    hyg_returns = (hyg['Close'] / hyg['Close'].iloc[0] - 1) * 100
                    
                    # Each 1% drop in HYG ≈ 30 bps wider spread
                    spread_adjustment = -hyg_returns * 30
                    
                    dates = hyg.index.strftime('%Y-%m-%d')
                    
                    self.spreads_df = pd.DataFrame({
                        'credit_spread_ig': 100 + (spread_adjustment * 0.3).values,
                        'credit_spread_hy': 350 + spread_adjustment.values
                    }, index=dates)
                    
                    logger.info(f"Estimated spreads from HYG: {len(self.spreads_df)} records")
                    return True
                else:
                    # Ultimate fallback
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    self.spreads_df = pd.DataFrame({
                        'credit_spread_ig': 100,
                        'credit_spread_hy': 350
                    }, index=dates.strftime('%Y-%m-%d'))
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Error fetching spreads: {e}")
            return False
    
    def calculate_bubble_index(self, vix: float, cape: float, 
                               spread_hy: float) -> Tuple[float, str, float]:
        """
        Calculate bubble index and regime.
        
        Returns:
            (bubble_index, regime, bifurcation_score)
        """
        # Normalize each component to 0-100
        vix_score = self._normalize_vix(vix)
        cape_score = self._normalize_cape(cape)
        spread_score = self._normalize_spread(spread_hy)
        
        # Composite bubble index (weighted average)
        # Higher = more bubbly/risky
        bubble_index = (vix_score * 0.3 + cape_score * 0.4 + spread_score * 0.3)
        
        # Determine regime
        if bubble_index < 30:
            regime = "INFRASTRUCTURE"
        elif bubble_index < 50:
            regime = "ADOPTION"
        elif bubble_index < 70:
            regime = "TRANSITION"
        else:
            regime = "BUBBLE_WARNING"
        
        # Bifurcation score (simplified)
        # Measures how far from "normal" we are
        vix_z = (vix - 18) / 5  # Mean ~18, std ~5
        cape_z = (cape - 25) / 8  # Mean ~25, std ~8
        
        bifurcation = 0.5 + 0.25 * np.tanh(vix_z) + 0.25 * np.tanh(cape_z)
        
        return round(bubble_index, 1), regime, round(bifurcation, 2)
    
    def _normalize_vix(self, vix: float) -> float:
        """Normalize VIX to 0-100 scale"""
        if vix < 12:
            return vix * 2  # 0-24
        elif vix < 20:
            return 24 + (vix - 12) * 3  # 24-48
        elif vix < 30:
            return 48 + (vix - 20) * 3  # 48-78
        else:
            return min(100, 78 + (vix - 30) * 2)
    
    def _normalize_cape(self, cape: float) -> float:
        """Normalize CAPE to 0-100 scale"""
        if cape < 15:
            return cape * 2  # 0-30
        elif cape < 25:
            return 30 + (cape - 15) * 2  # 30-50
        elif cape < 35:
            return 50 + (cape - 25) * 3  # 50-80
        else:
            return min(100, 80 + (cape - 35) * 2)
    
    def _normalize_spread(self, spread: float) -> float:
        """Normalize HY spread to 0-100 scale (inverted - tight spreads = low score)"""
        # Typical HY spread: 300-500 bps normal, >600 stressed
        if spread < 300:
            return 20 + (spread / 300) * 20  # 20-40
        elif spread < 500:
            return 40 + ((spread - 300) / 200) * 30  # 40-70
        else:
            return min(100, 70 + ((spread - 500) / 300) * 30)
    
    def calculate_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling z-scores for each metric"""
        window = 252  # ~1 year
        
        # VIX z-score
        vix_mean = df['vix'].rolling(window, min_periods=20).mean()
        vix_std = df['vix'].rolling(window, min_periods=20).std()
        df['vix_zscore'] = ((df['vix'] - vix_mean) / vix_std).fillna(0).round(2)
        
        # CAPE z-score
        cape_mean = df['cape'].rolling(window, min_periods=20).mean()
        cape_std = df['cape'].rolling(window, min_periods=20).std()
        df['cape_zscore'] = ((df['cape'] - cape_mean) / cape_std).fillna(0).round(2)
        
        # Credit z-score
        spread_mean = df['credit_spread_hy'].rolling(window, min_periods=20).mean()
        spread_std = df['credit_spread_hy'].rolling(window, min_periods=20).std()
        df['credit_zscore'] = ((df['credit_spread_hy'] - spread_mean) / spread_std).fillna(0).round(2)
        
        return df
    
    def build_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build complete dataset for backfill"""
        logger.info("Building combined dataset...")
        
        # Get all dates from VIX (most complete)
        if self.vix_df is None or len(self.vix_df) == 0:
            logger.error("No VIX data available")
            return pd.DataFrame()
        
        df = self.vix_df.copy()
        
        # Merge CAPE
        if self.cape_df is not None:
            df = df.join(self.cape_df, how='left')
            df['cape'] = df['cape'].ffill().fillna(36.0)
        else:
            df['cape'] = 36.0
        
        # Merge spreads
        if self.spreads_df is not None:
            df = df.join(self.spreads_df, how='left')
            df['credit_spread_ig'] = df['credit_spread_ig'].ffill().fillna(100)
            df['credit_spread_hy'] = df['credit_spread_hy'].ffill().fillna(350)
        else:
            df['credit_spread_ig'] = 100
            df['credit_spread_hy'] = 350
        
        # Calculate z-scores
        df = self.calculate_zscores(df)
        
        # Calculate bubble index for each row
        bubble_data = []
        for idx, row in df.iterrows():
            bi, regime, bif = self.calculate_bubble_index(
                row['vix'], row['cape'], row['credit_spread_hy']
            )
            bubble_data.append({
                'date': idx,
                'bubble_index': bi,
                'regime': regime,
                'bifurcation_score': bif
            })
        
        bubble_df = pd.DataFrame(bubble_data).set_index('date')
        df = df.join(bubble_df)
        
        # Add calculated_at timestamp
        df['calculated_at'] = datetime.now().isoformat()
        
        # Reset index for output
        df = df.reset_index()
        
        logger.info(f"Built dataset with {len(df)} records")
        return df
    
    def save_to_supabase(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Save data to bubble_index_daily table"""
        if self.client is None:
            logger.error("No Supabase client available")
            return 0
        
        if df.empty:
            logger.warning("No data to save")
            return 0
        
        logger.info(f"Saving {len(df)} records to bubble_index_daily...")
        
        # Prepare records
        records = []
        for _, row in df.iterrows():
            records.append({
                'date': row['date'],
                'vix': float(row['vix']),
                'cape': float(row['cape']),
                'credit_spread_ig': float(row['credit_spread_ig']),
                'credit_spread_hy': float(row['credit_spread_hy']),
                'vix_zscore': float(row['vix_zscore']),
                'cape_zscore': float(row['cape_zscore']),
                'credit_zscore': float(row['credit_zscore']),
                'bubble_index': float(row['bubble_index']),
                'bifurcation_score': float(row['bifurcation_score']),
                'regime': row['regime'],
                'calculated_at': row['calculated_at']
            })
        
        # Upsert in batches
        saved = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                result = self.client.table("bubble_index_daily")\
                    .upsert(batch, on_conflict="date")\
                    .execute()
                saved += len(batch)
                logger.info(f"  Saved batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as e:
                logger.error(f"  Error saving batch: {e}")
        
        logger.info(f"Total saved: {saved} records")
        return saved
    
    def run(self, start_date: str, end_date: str) -> bool:
        """Run full backfill process"""
        logger.info("=" * 60)
        logger.info("HISTORICAL DATA BACKFILL")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 60)
        
        # Fetch all data sources
        if not self.fetch_vix(start_date, end_date):
            return False
        
        self.fetch_cape(start_date, end_date)  # Optional
        self.fetch_credit_spreads(start_date, end_date)  # Optional
        
        # Build combined dataset
        df = self.build_dataset(start_date, end_date)
        
        if df.empty:
            logger.error("Failed to build dataset")
            return False
        
        # Save to Supabase
        saved = self.save_to_supabase(df)
        
        logger.info("=" * 60)
        logger.info(f"Backfill complete: {saved} records saved")
        logger.info("=" * 60)
        
        return saved > 0


def main():
    parser = argparse.ArgumentParser(description="Backfill historical data for ARGUS-1 backtesting")
    parser.add_argument("--days", type=int, default=365, help="Number of days to backfill (default: 365)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    # Run backfill
    backfill = HistoricalDataBackfill()
    success = backfill.run(start_date, end_date)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())