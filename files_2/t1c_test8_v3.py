"""
T1C TEST 8 v3 — DM-BASED POSITION SIZING (with Sharpe/Sortino/Calmar)
Full 528-ticker universe, production spec.
"""
import sys
sys.path.insert(0, '.')
from t1c_sim_core_v3 import *

CONFIG_LOCAL = {
    'DM_TABLE': 'dm_history', 'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_TABLE': 'hms_daily', 'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',

    'DM_ENTRY_MIN': 65,
    'MA5_RISING': True,
    'MA20_EXIT': 50,
    'HMS_MIN': None,

    'SIZING_MODE': 'dm_tiered',
    'TIER_90': 0.080,
    'TIER_80': 0.065,
    'TIER_70': 0.050,
    'TIER_65': 0.035,

    'INITIAL_NAV': 1_000_000,
    'START_DATE': '2016-01-01',
    'END_DATE': '2026-03-20',
}

def main():
    print("\nTEST 8 v3 - DM SIZING | Full Universe | Sharpe/Sortino/Calmar")
    print("Production Spec: DM>=65, 5d MA rising, Exit MA20<50, DM-tiered sizing")
    print("Period: 2016-01-01 to 2026-03-20\n")

    result = run_simulation(CONFIG_LOCAL)

    print(f"\n{'='*65}")
    print("TEST 8 — DM SIZING (PRODUCTION SPEC) — RISK-ADJUSTED METRICS")
    print(f"{'='*65}")
    print(f"  CAGR:            {result['cagr']:.1f}%")
    print(f"  Total Return:    {result['total_return']:.1f}%")
    print(f"  Final NAV:       ${result['final_nav']:,.0f}")
    print(f"  Max Drawdown:    {result['max_drawdown']:.1f}%")
    print(f"  Hit Rate:        {result['hit_rate']:.1f}%")
    print(f"  Total Trades:    {result['total_trades']}")
    print(f"  Avg Return:      {result['avg_return']:.1f}%")
    print(f"  Avg Hold:        {result['avg_hold_days']:.0f} days")
    print(f"\n  -- RISK-ADJUSTED --")
    print(f"  Sharpe Ratio:    {result['sharpe']}")
    print(f"  Sortino Ratio:   {result['sortino']}")
    print(f"  Calmar Ratio:    {result['calmar']}")
    print(f"\n  YEARLY NAV:")
    for e in result.get('nav_log', []):
        print(f"    {e['date']}: ${e['nav']:,.0f} ({e['return_pct']:+.1f}%)")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
