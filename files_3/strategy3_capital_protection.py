"""
STRATEGY 3 — CAPITAL PROTECTION
=================================
Same entry as Strategy 1 but exits earlier.
Exits when 20d MA drops below 55 instead of 50.
Prioritizes protecting gains over holding for maximum return.

Question: Does exiting earlier protect capital in bear markets without
sacrificing too much CAGR in bull markets?
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
    'ENTRY_DM_MIN':            65,
    'ENTRY_MA5_RISING':        True,
    'ENTRY_PRICE_ABOVE_MA20':  False,
    'EXIT_MA20_THRESHOLD':     55,       # tighter exit — leaves earlier

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
    run("STRATEGY 3 — CAPITAL PROTECTION | CF>=65 | 5d MA rising | Exit MA20<55 | Equal weight", CONFIG)
