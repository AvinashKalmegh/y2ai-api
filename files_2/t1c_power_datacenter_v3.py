"""
T1C POWER & DATA CENTER SECTOR SIMULATIONS — v3 (with Sharpe)
===============================================================
"""

import sys
sys.path.insert(0, '.')
from t1c_sim_core_v3 import *

SECTORS = {
    'Power_Energy': [
        'CEG','VST','NRG','ETR','EXC','PCG','NEE','AES','DNN','UEC','CCJ',
    ],
    'Data_Center_AI_Infra': [
        'VRT','EQIX','DLR','IRM','SMCI','ALAB','NVDA','AMD','AVGO','MRVL','ARM','QCOM',
    ],
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
    print("\nT1C POWER & DATA CENTER — v3 (Sharpe enabled)")
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
            print(f"  Yearly NAV:")
            for e in r['nav_log']:
                print(f"    {e['date']}: ${e['nav']:,.0f} ({e['return_pct']:+.1f}%)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'sector': sector_name, 'error': str(e)})

    print(f"\n{'='*90}")
    print("POWER & DATA CENTER RESULTS — PRODUCTION SPEC (with Sharpe)")
    print(f"{'='*90}")
    print(f"{'Sector':<25} {'CAGR':>7} {'MaxDD':>7} {'HitRate':>8} "
          f"{'Trades':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'FinalNAV':>13}")
    print(f"{'-'*90}")

    for r in results:
        if 'error' in r:
            print(f"{r['sector']:<25} ERROR: {r['error']}")
            continue
        print(f"{r['sector']:<25} "
              f"{r['cagr']:>6.1f}% "
              f"{r['max_drawdown']:>6.1f}% "
              f"{r['hit_rate']:>7.1f}% "
              f"{r['total_trades']:>7} "
              f"{str(r['sharpe']):>7} "
              f"{str(r['sortino']):>8} "
              f"{str(r['calmar']):>7} "
              f"${r['final_nav']:>12,.0f}")

    print(f"{'='*90}")
    print("\nReference: Full_Universe = 19.3% CAGR, Sharpe TBD (re-run with v3)")

if __name__ == '__main__':
    main()
