"""
STRATEGY 2 — HIGH CONVICTION ONLY
===================================
Raises the entry bar significantly.
Only enter when institutional accumulation is very strong (CF >= 75).
Expect fewer trades, longer holds, higher quality entries.

Question: Does requiring stronger conviction produce better risk-adjusted
returns, or does it miss too many good entries?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core_v1 import run

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    'DM_TABLE':      'dm_history',
    'DM_DATE_COL':   'date',
    'DM_TICKER_COL': 'ticker',
    'DM_PRICE_COL':  'close',
    'DM_SCORE_COL':  'dm_smoothed',

    # ── Strategy rules ────────────────────────────────────────────────────
    'ENTRY_DM_MIN':            75,       # higher bar — CF score >= 75
    'ENTRY_MA5_RISING':        True,
    'ENTRY_PRICE_ABOVE_MA20':  False,
    'EXIT_MA20_THRESHOLD':     50,

    # ── Sizing ────────────────────────────────────────────────────────────
    'SIZING_MODE':    'equal_weight',

    # ── Portfolio ─────────────────────────────────────────────────────────
    'INITIAL_CAPITAL': 1_000_000,
    'MAX_POSITIONS':   20,
    'MIN_POSITION':    500,

    # ── Date range ────────────────────────────────────────────────────────
    'START_YEAR': 2016,
    'END_YEAR':   2026,
}

if __name__ == '__main__':
    run("STRATEGY 2 — HIGH CONVICTION | CF>=75 | 5d MA rising | Exit MA20<50 | Equal weight", CONFIG)
