"""
T1C TEST 7 — ETF FLOWS POSITION SIZING
Entry:  DM >= 65 AND 5-day MA rising (HMS DISABLED)
Exit:   20-day MA drops below 50
Sizing: 7% per position when sector ETF shows INFLOW
        3% per position when sector ETF shows OUTFLOW
        5% per position when NEUTRAL (default)
Question: Does aligning position size with institutional sector
flows (AP creation/redemption) improve CAGR or reduce drawdown?
Compare against Test 2 baseline (same entry/exit, equal weight 5%).
ETF flows table: etf_flows_history (2016+)
Expected columns: date, etf_ticker, flow_direction (INFLOW/OUTFLOW/NEUTRAL)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'INITIAL_CAPITAL': 1_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,

    # ETF flow sizing — maps sector ETF to position size multiplier
    'ETF_FLOW_SIZING': True,
    'ETF_FLOW_TABLE':  'etf_flows_history',
    'ETF_FLOW_DATE_COL':      'date',
    'ETF_FLOW_TICKER_COL':    'ticker',
    'ETF_FLOW_DIR_COL':       'flow_signal',  # Inflow / Outflow / Neutral
    'ETF_INFLOW_ALLOC_PCT':   0.07,   # 7% of portfolio per position on inflow
    'ETF_OUTFLOW_ALLOC_PCT':  0.03,   # 3% of portfolio per position on outflow
    'ETF_NEUTRAL_ALLOC_PCT':  0.05,   # 5% default

    # Sector ETF mapping — adjust if column names differ
    # Maps stock sectors to ETF tickers in etf_flows_history
    'SECTOR_ETF_MAP': {
        'Technology':     'XLK',
        'Semiconductors': 'SMH',
        'Software':       'IGV',
        'Financials':     'XLF',
        'Energy':         'XLE',
        'Industrials':    'XLI',
        'Healthcare':     'XLV',
        'Default':        'SPY',
    },

    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026, 'END_DATE': '2026-03-20',
}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())
if __name__ == '__main__': run("TEST 7 — ETF FLOW SIZING | DM 65 | HMS DISABLED | Exit MA20 < 50 | ETF-based sizing")
