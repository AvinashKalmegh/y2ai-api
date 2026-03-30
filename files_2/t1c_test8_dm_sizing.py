"""
T1C TEST 8 — DM-BASED POSITION SIZING
Entry:  DM >= 65 AND 5-day MA rising (HMS DISABLED)
Exit:   20-day MA drops below 50
Sizing: Position size scales with DM score at entry:
        DM >= 90: 8% of portfolio
        DM 80-89: 6.5% of portfolio
        DM 70-79: 5% of portfolio (baseline)
        DM 65-69: 3.5% of portfolio
Question: Does concentrating capital in highest-DM entries
improve CAGR vs equal-weight Test 2 (53.8% CAGR)?
Key comparison: same entry/exit as Test 2, only sizing differs.
If CAGR improves — DM score is a reliable conviction indicator.
If CAGR declines — high DM at entry is a late signal (already extended).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,

    # DM-based sizing tiers
    'DM_SIZING': True,
    'DM_TIERS': [
        (90, 0.080),   # DM >= 90 → 8% allocation
        (80, 0.065),   # DM 80-89 → 6.5% allocation
        (70, 0.050),   # DM 70-79 → 5% allocation (baseline)
        (65, 0.035),   # DM 65-69 → 3.5% allocation
    ],

    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 8 — DM SIZING | DM 65 | HMS DISABLED | Exit MA20 < 50 | DM-tiered allocation")
