"""
T1C 17-MONTH COMPARABLE PERIOD SIMULATION — v3
================================================
Exact same period as the PM's track record:
  October 1, 2024 → February 28, 2026

Full universe production spec.
Purpose: apples-to-apples comparison with PM's 56.85% return
over the same 17-month period.

Production spec:
  - DM >= 65, 5-day MA rising
  - Exit: 20-day MA < 50
  - Sizing: DM-tiered (8/6.5/5/3.5%)
  - HMS gate: DISABLED
"""

import sys
sys.path.insert(0, '.')
from t1c_sim_core_v3 import *

CONFIG_LOCAL = {
    'DM_TABLE': 'dm_history', 'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_TABLE': 'hms_daily', 'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',

    # Full 28-ticker production universe
    'UNIVERSE': [
        'NVDA','AMD','TSM','AVGO','QCOM','MSFT','GOOGL','META','AMZN','ORCL',
        'CRM','NOW','WDAY','ADBE','INTU','VRT','EQIX','IRM','DLR','CEG',
        'ALAB','MRVL','SMCI','ARM','KLAC','DNN','MRNA','COST'
    ],

    # Entry/exit — production spec
    'DM_ENTRY_MIN':    65,
    'MA5_RISING':      True,
    'MA20_EXIT':       50,
    'HMS_MIN':         None,

    # Sizing — DM-tiered
    'SIZING_MODE':     'dm_tiered',
    'TIER_90':         0.080,
    'TIER_80':         0.065,
    'TIER_70':         0.050,
    'TIER_65':         0.035,

    # EXACT same period as the PM's track record
    'INITIAL_NAV':     10_000_000,
    'START_DATE':      '2024-10-01',
    'END_DATE':        '2026-02-28',
}


def main():
    print("\nT1C 17-MONTH COMPARABLE PERIOD SIMULATION")
    print("Period: October 1, 2024 → February 28, 2026")
    print("Matches: PM track record period (56.85% return)")
    print("Production Spec: DM>=65, 5d MA rising, Exit MA20<50, DM-tiered sizing\n")

    result = run_simulation(CONFIG_LOCAL)

    # Calculate annualized return for the 17-month period
    months = 17
    total_ret = result['total_return']
    annualized = ((1 + total_ret/100) ** (12/months) - 1) * 100

    print(f"\n{'='*65}")
    print("17-MONTH COMPARABLE PERIOD RESULTS")
    print(f"{'='*65}")
    print(f"  Period:          Oct 1, 2024 → Feb 28, 2026 (17 months)")
    print(f"  Total Return:    {result['total_return']:+.2f}%")
    print(f"  Annualized:      {annualized:+.1f}%")
    print(f"  Final NAV:       ${result['final_nav']:,.0f}")
    print(f"  Max Drawdown:    {result['max_drawdown']:.1f}%")
    print(f"  Hit Rate:        {result['hit_rate']:.1f}%")
    print(f"  Total Trades:    {result['total_trades']}")
    print(f"  Avg Return:      {result['avg_return']:.1f}%")
    print(f"  Avg Hold:        {result['avg_hold_days']:.0f} days")
    print(f"\n  ── RISK-ADJUSTED ──────────────────────────────────")
    print(f"  Sharpe Ratio:    {result['sharpe']}")
    print(f"  Sortino Ratio:   {result['sortino']}")
    print(f"  Calmar Ratio:    {result['calmar']}")
    print(f"\n  ── APPLES-TO-APPLES COMPARISON ────────────────────")
    print(f"  PM total return:       +56.85%  (Oct 2024 → Feb 2026)")
    print(f"  ARGUS total return:    {result['total_return']:+.2f}%  (same period)")
    print(f"  PM annualized:         ~40%")
    print(f"  ARGUS annualized:      {annualized:+.1f}%")
    print(f"  Category benchmark:    ~18%  (PM's stated benchmark)")
    print(f"\n  YEARLY NAV:")
    for e in result['nav_log']:
        print(f"    {e['date']}: ${e['nav']:,.0f} ({e['return_pct']:+.1f}%)")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
