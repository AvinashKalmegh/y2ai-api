"""
T1C TEST 11 — SHORT INTEREST FILTER
Entry:  DM >= 65 AND 5-day MA rising AND short interest < 15% of float
Exit:   20-day MA drops below 50
Sizing: DM-tiered (same as Test 8)
Question: Does filtering out heavily shorted names improve results?
High short interest with rising DM could mean squeeze setup (positive)
or it could mean smart money is bearish (negative). This test determines
which effect dominates across the decade.
Compare hit rate and CAGR vs Test 8.
Short interest table: short_interest (2018+)
Expected columns: date, ticker, short_interest_pct (% of float shorted)
Developer: confirm exact column name. Threshold is 15% — names above
this level are excluded from entry consideration.
Note: short_interest only goes back to 2018. Results for 2016-2017
will use DM-tiered sizing without the short interest filter (pass all).
Flag this in output if possible.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,

    # DM-tiered sizing
    'DM_SIZING': True,
    'DM_TIERS': [
        (90, 0.080),
        (80, 0.065),
        (70, 0.050),
        (65, 0.035),
    ],

    # Short interest filter
    'SHORT_FILTER': True,
    'SHORT_TABLE':        'short_interest',
    'SHORT_DATE_COL':     'settlement_date',
    'SHORT_TICKER_COL':   'ticker',
    'SHORT_PCT_COL':      'short_pct_outstanding', # confirm column name
    'SHORT_MAX_PCT':      15.0,  # block entry if short interest > 15% of float
    'SHORT_START_YEAR':   2018,  # filter only applies from this year forward

    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 11 — SHORT INTEREST FILTER | DM 65 | Short interest < 15% | DM-tiered sizing")
