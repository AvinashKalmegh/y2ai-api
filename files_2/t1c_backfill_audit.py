"""
DATA BACKFILL TASK — PRIORITY 2
=================================
Backfill all signal tables to January 1, 2016.
Target: ~1.3M additional rows across all tables.

Run this script to verify current coverage before and after backfill.
The actual backfill work requires your data pipeline — this script
audits what is there and what is missing.

Tables to backfill:
  1. macro_history       — currently 2016+, verify coverage
  2. etf_flows_history   — currently 2016+, verify coverage
  3. insider_transactions — currently 2014+, verify 2016 coverage
  4. short_volume        — currently 2020+, NEEDS backfill to 2016
  5. short_interest      — currently 2018+, NEEDS backfill to 2016

Target for all tables: 2016-01-01 through present.
"""

import os, sys
import pandas as pd
from datetime import date
from dotenv import load_dotenv

try:
    from supabase import create_client
except ImportError:
    print("ERROR: pip install supabase pandas python-dotenv")
    sys.exit(1)

load_dotenv()


def get_client():
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be in .env")
    return create_client(url, key)


TABLES = [
    {
        'name':      'dm_history',
        'date_col':  'date',
        'ticker_col':'ticker',
        'target_start': '2016-01-01',
        'notes':     'Primary signal — production data, should be complete',
    },
    {
        'name':      'hms_daily',
        'date_col':  'date',
        'ticker_col':'ticker',
        'target_start': '2016-01-01',
        'notes':     'HMS signal — backfill to 2016 requested',
    },
    {
        'name':      'macro_history',
        'date_col':  'date',
        'ticker_col': None,
        'target_start': '2016-01-01',
        'notes':     'Macro regime — verify 2016 coverage',
    },
    {
        'name':      'etf_flows_history',
        'date_col':  'date',
        'ticker_col':'ticker',
        'target_start': '2016-01-01',
        'notes':     'ETF flows — verify 2016 coverage',
    },
    {
        'name':      'insider_transactions',
        'date_col':  'filing_date',
        'ticker_col':'ticker',
        'target_start': '2016-01-01',
        'notes':     'Insider data — currently 2014+, verify 2016 coverage',
    },
    {
        'name':      'short_volume',
        'date_col':  'date',
        'ticker_col':'ticker',
        'target_start': '2016-01-01',
        'notes':     'NEEDS BACKFILL — currently only 2020+',
    },
    {
        'name':      'short_interest',
        'date_col':  'settlement_date',
        'ticker_col':'ticker',
        'target_start': '2016-01-01',
        'notes':     'NEEDS BACKFILL — currently only 2018+',
    },
]


def audit_table(client, table_info):
    """Audit coverage for a single table."""
    name = table_info['name']
    date_col = table_info['date_col']
    ticker_col = table_info['ticker_col']
    target_start = table_info['target_start']

    print(f"\n  TABLE: {name}")
    print(f"  Target start: {target_start}")
    print(f"  Notes: {table_info['notes']}")

    try:
        # Get earliest date
        r_min = (client.table(name)
                 .select(date_col)
                 .order(date_col, desc=False)
                 .limit(1)
                 .execute())

        # Get latest date
        r_max = (client.table(name)
                 .select(date_col)
                 .order(date_col, desc=True)
                 .limit(1)
                 .execute())

        # Get row count
        r_count = (client.table(name)
                   .select('*', count='exact')
                   .execute())

        earliest = r_min.data[0][date_col] if r_min.data else 'NO DATA'
        latest   = r_max.data[0][date_col] if r_max.data else 'NO DATA'
        count    = r_count.count if hasattr(r_count, 'count') else 'UNKNOWN'

        # Check gap — rows between 2016-01-01 and target start
        gap_needed = earliest > target_start if earliest != 'NO DATA' else True

        print(f"  Earliest date:  {earliest}")
        print(f"  Latest date:    {latest}")
        print(f"  Total rows:     {count:,}" if isinstance(count, int) else f"  Total rows: {count}")
        print(f"  Gap to fill:    {'YES — backfill needed from ' + target_start + ' to ' + str(earliest) if gap_needed else 'NO — coverage OK'}")

        if ticker_col:
            r_tickers = (client.table(name)
                         .select(ticker_col)
                         .execute())
            unique_tickers = len(set(r[ticker_col] for r in r_tickers.data)) if r_tickers.data else 0
            print(f"  Unique tickers: {unique_tickers}")

        return {
            'table':     name,
            'earliest':  earliest,
            'latest':    latest,
            'rows':      count,
            'gap':       gap_needed,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {'table': name, 'error': str(e)}


def main():
    print("\nDATA BACKFILL AUDIT")
    print("Target: All signal tables covered from 2016-01-01")
    print("="*60)

    client = get_client()
    results = []

    for table_info in TABLES:
        r = audit_table(client, table_info)
        results.append(r)

    print(f"\n{'='*60}")
    print("BACKFILL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Table':<25} {'Earliest':>12} {'Latest':>12} {'Rows':>10} {'Gap?':>6}")
    print(f"{'-'*65}")
    for r in results:
        if 'error' in r:
            print(f"{r['table']:<25} ERROR")
            continue
        rows_str = f"{r['rows']:,}" if isinstance(r['rows'], int) else str(r['rows'])
        gap_str  = 'YES ⚠' if r['gap'] else 'OK'
        print(f"{r['table']:<25} {str(r['earliest']):>12} {str(r['latest']):>12} "
              f"{rows_str:>10} {gap_str:>6}")

    print(f"\n{'='*60}")
    print("BACKFILL PRIORITIES:")
    print("  1. short_volume — backfill 2016-2019 (4 years missing)")
    print("  2. short_interest — backfill 2016-2017 (2 years missing)")
    print("  3. hms_daily — extend back to 2016 if not already there")
    print("  4. All others — verify and patch gaps if found")
    print("\nOnce backfill is complete, re-run this script to confirm.")
    print("Then run t1c_sector_runs.py and t1c_ai_formation.py.")


if __name__ == '__main__':
    main()
