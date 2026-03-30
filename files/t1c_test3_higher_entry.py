"""
T1C TEST 3 — HIGHER ENTRY THRESHOLD
Entry:  DM >= 70 AND 5-day MA rising AND HMS >= 0.45
Exit:   20-day MA drops below 50
Question: Does requiring stronger DM conviction at entry improve hit rate?
Compare hit rate and CAGR vs Test 1 baseline.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'ENTRY_DM': 70, 'EXIT_MA20': 50, 'HMS_MIN': 0.45,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,
    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_corev2.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 3 — HIGHER ENTRY | DM 70 | HMS 0.45 | Exit MA20 < 50")
