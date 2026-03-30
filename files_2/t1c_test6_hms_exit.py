"""
T1C TEST 6 — HMS AS EXIT SIGNAL (not entry gate)
Entry:  DM >= 65 AND 5-day MA rising (NO HMS gate on entry)
Exit:   20-day MA drops below 50 OR HMS drops below 0.35
Question: Does HMS add value when used to detect deterioration
rather than confirm entry? Test 2 proved HMS hurts as entry gate.
Does it help as an exit accelerator?
Note: HMS exit fires when either condition triggers — whichever comes first.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'HMS_EXIT_MIN': 0.35,   # exit if HMS drops below this
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,
    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 6 — HMS EXIT SIGNAL | DM 65 | No HMS entry gate | Exit MA20<50 OR HMS<0.35")
