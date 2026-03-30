"""
T1C SECTOR-BY-SECTOR SIMULATION — v3 (with Sharpe)
====================================================
Runs production spec for each sector cluster.
Reports CAGR, max drawdown, hit rate, Sharpe, Sortino, Calmar.
"""

import sys
sys.path.insert(0, '.')
from t1c_sim_core_v3 import *

SECTORS = {
    'Semiconductors':    ['NVDA', 'AMD', 'AVGO', 'MRVL', 'KLAC', 'ARM'],
    'Software_Cloud':    ['MSFT', 'GOOGL', 'META', 'CRM', 'NOW', 'WDAY', 'ADBE', 'INTU'],
    'Infrastructure':    ['VRT', 'EQIX', 'IRM', 'DLR'],
    'Nuclear_Energy':    ['CEG', 'DNN'],
    'Biotech':           ['MRNA'],
    'Full_Universe':     ['NVDA','AMD','TSM','AVGO','QCOM',
                          'MSFT','GOOGL','META','AMZN','ORCL',
                          'CRM','NOW','WDAY','ADBE','INTU',
                          'VRT','EQIX','IRM','DLR','CEG',
                          'ALAB','MRVL','SMCI','ARM','KLAC',
                          'DNN','MRNA','COST'],
}

BASE_CONFIG = {
    'DM_TABLE': 'dm_history', 'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_TABLE': 'hms_daily', 'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'DM_ENTRY_MIN': 65, 'MA5_RISING': True, 'MA20_EXIT': 50, 'HMS_MIN': None,
    'SIZING_MODE': 'dm_tiered',
    'TIER_90': 0.080, 'TIER_80': 0.065, 'TIER_70': 0.050, 'TIER_65': 0.035,
    'INITIAL_NAV': 10_000_000, 'START_DATE': '2016-01-01', 'END_DATE': '2026-03-15',
}

def main():
    print("\nT1C SECTOR-BY-SECTOR SIMULATION — v3 (Sharpe enabled)")
    print("Production Spec: DM>=65, 5d MA rising, Exit MA20<50, DM-tiered sizing\n")

    results = []
    for sector_name, tickers in SECTORS.items():
        print(f"\n{'='*60}")
        print(f"  SECTOR: {sector_name} | {', '.join(tickers)}")
        print(f"{'='*60}")
        config = {**BASE_CONFIG, 'UNIVERSE': tickers}
        try:
            r = run_simulation(config)
            r['sector'] = sector_name
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'sector': sector_name, 'error': str(e)})

    print(f"\n{'='*95}")
    print("SECTOR RESULTS — PRODUCTION SPEC (with Sharpe)")
    print(f"{'='*95}")
    print(f"{'Sector':<22} {'CAGR':>7} {'MaxDD':>7} {'HitRate':>8} "
          f"{'Trades':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'FinalNAV':>13}")
    print(f"{'-'*95}")

    for r in results:
        if 'error' in r:
            print(f"{r['sector']:<22} ERROR: {r['error']}")
            continue
        print(f"{r['sector']:<22} "
              f"{r['cagr']:>6.1f}% "
              f"{r['max_drawdown']:>6.1f}% "
              f"{r['hit_rate']:>7.1f}% "
              f"{r['total_trades']:>7} "
              f"{str(r['sharpe']):>7} "
              f"{str(r['sortino']):>8} "
              f"{str(r['calmar']):>7} "
              f"${r['final_nav']:>12,.0f}")

    print(f"{'='*95}")

if __name__ == '__main__':
    main()
