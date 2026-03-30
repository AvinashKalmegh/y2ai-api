"""
T1C TEST 10 — INSIDER TRANSACTION FILTER
Entry:  DM >= 65 AND 5-day MA rising AND no insider SELLING in prior 30 days
Exit:   20-day MA drops below 50
Sizing: DM-tiered (same as Test 8)
Question: Does filtering out names with recent insider selling
improve hit rate and reduce false positives?
Insider buying is not required — only insider SELLING blocks entry.
This is a universe quality filter, not a confirmation gate.
Compare hit rate and MaxDD vs Test 8 (52.7% CAGR, -29.0% MaxDD).
Insider table: insider_transactions (2014+)
Expected columns: date, ticker, transaction_type (BUY/SELL), shares, value
Developer: confirm exact column names. Filter on transaction_type = 'SELL'
within 30 days prior to entry date.
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

    # Insider filter
    'INSIDER_FILTER': True,
    'INSIDER_TABLE':          'insider_transactions',
    'INSIDER_DATE_COL':       'filing_date',
    'INSIDER_TICKER_COL':     'ticker',
    'INSIDER_TYPE_COL':       'transaction_type', # confirm column name
    'INSIDER_SELL_VALUE':     'Sale',              # confirm exact value
    'INSIDER_LOOKBACK_DAYS':  30,

    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 10 — INSIDER FILTER | DM 65 | No insider selling 30d | DM-tiered sizing")
