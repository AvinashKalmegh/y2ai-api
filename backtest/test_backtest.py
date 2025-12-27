#!/usr/bin/env python3
"""
Test script for ARGUS-1 Backtesting Module

Run with:
    cd y2ai
    python -m backtest.test_backtest
    
Or with specific dates:
    python -m backtest.test_backtest 2024-10-01 2024-12-01
"""

import os
import sys
from datetime import datetime, timedelta

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import BacktestRunner, ReportGenerator
from backtest.data_loader import HistoricalDataLoader


def check_available_data():
    """Check what data is available in Supabase"""
    print("Checking available data in Supabase...")
    
    loader = HistoricalDataLoader()
    
    # Load a wide date range to find what's available
    loader.load_all_data("2020-01-01", "2025-12-31")
    
    dates = loader.get_available_dates()
    
    if not dates:
        print("  No data found in bubble_index_daily table!")
        return None, None
    
    min_date = min(dates)
    max_date = max(dates)
    
    print(f"  Available data: {min_date} to {max_date} ({len(dates)} days)")
    
    return min_date, max_date


def main():
    print("=" * 60)
    print("ARGUS-1 BACKTEST TEST")
    print("=" * 60)
    
    # Check available data first
    min_date, max_date = check_available_data()
    
    if not min_date:
        print("\nNo historical data found. Make sure:")
        print("  1. SUPABASE_URL and SUPABASE_KEY are set")
        print("  2. bubble_index_daily table has data")
        print("  3. Run: python -m orchestrator --indicators")
        return 1
    
    # Set date range
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        # Use available data range (leave 60 days buffer at end for outcomes)
        start_date = min_date
        end_dt = datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=60)
        end_date = max(start_date, end_dt.strftime("%Y-%m-%d"))
    
    print(f"\nBacktest Period: {start_date} to {end_date}")
    print("-" * 60)
    
    # Validate date range
    if start_date >= end_date:
        print(f"\nNot enough data for backtesting.")
        print(f"Need at least 60 days of data + 60 days buffer for outcomes.")
        print(f"Current data: {min_date} to {max_date}")
        return 1
    
    # Run backtest
    runner = BacktestRunner()
    
    try:
        results = runner.run(
            start_date=start_date,
            end_date=end_date,
            benchmark="SPY"
        )
        
        # Print summary
        print("\n")
        print(results.summary())
        
        # Export results
        results.export_csv("backtest_signals.csv")
        
        # Generate HTML report
        ReportGenerator.save_html(results.metrics, "backtest_report.html")
        
        # Generate JSON report
        ReportGenerator.save_json(results.metrics, "backtest_report.json")
        
        print("\nFiles generated:")
        print("  - backtest_signals.csv")
        print("  - backtest_report.html")
        print("  - backtest_report.json")
        
    except Exception as e:
        print(f"\nError running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())