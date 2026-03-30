"""
T1C TEST 9 — MACRO REGIME SIZING
Entry:  DM >= 65 AND 5-day MA rising (HMS DISABLED)
Exit:   20-day MA drops below 50
Sizing: DM-tiered (same as Test 8) PLUS macro stress multiplier:
        When VIX > 25 AND credit spreads elevated: cap all positions at 3%
        When VIX > 20 (moderate stress): cap all positions at 4%
        When VIX <= 20 (low stress): DM-tiered sizing applies normally
Question: Does adding macro regime awareness to the DM-tiered sizing
reduce the 2022 drawdown without hurting overall CAGR?
Compare directly against Test 8 (52.7% CAGR, -29.0% MaxDD).
Macro table: macro_history (2016+)
Expected columns: date, vix_close, credit_spread (or hy_spread, ig_spread)
Developer: confirm exact column names before running.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,

    # DM-tiered sizing (same as Test 8)
    'DM_SIZING': True,
    'DM_TIERS': [
        (90, 0.080),
        (80, 0.065),
        (70, 0.050),
        (65, 0.035),
    ],

    # Macro stress override
    'MACRO_SIZING': True,
    'MACRO_TABLE':           'macro_history',
    'MACRO_DATE_COL':        'date',
    'MACRO_VIX_COL':         'vix_close',       # confirm column name
    'MACRO_SPREAD_COL':      'credit_spread',    # confirm column name (HY or IG spread)
    'MACRO_HIGH_STRESS_VIX': 25,    # VIX above this = high stress, cap at 3%
    'MACRO_MED_STRESS_VIX':  20,    # VIX above this = medium stress, cap at 4%
    'MACRO_HIGH_STRESS_CAP': 0.030, # 3% max per position in high stress
    'MACRO_MED_STRESS_CAP':  0.040, # 4% max per position in medium stress

    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 9 — MACRO REGIME SIZING | DM 65 | DM-tiered + VIX cap | Exit MA20 < 50")
