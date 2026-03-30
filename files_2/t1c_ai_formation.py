"""
T1C AI FORMATION PERIOD SIMULATION
=====================================
Runs production spec starting January 2023 to capture the full
AI infrastructure cycle from inception through March 2026.

Production spec:
  - DM >= 65, 5-day MA rising
  - Exit: 20-day MA < 50
  - Sizing: DM-tiered (8/6.5/5/3.5%)
  - HMS gate: DISABLED
  - Period: 2023-2026
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 10_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,
    'DM_SIZING': True,
    'DM_TIERS': [
        (90, 0.080),
        (80, 0.065),
        (70, 0.050),
        (65, 0.035),
    ],
    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2023, 'END_YEAR': 2026,
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("AI FORMATION | DM 65 | DM-tiered sizing | 2023-2026")
