"""
STRATEGY 1 — PURE SIGNAL
=========================
The cleanest possible test of the CF signal.
No overlays, no complexity. Entry on CF score alone with 5d MA confirmation.
Exit when CF deteriorates. Equal weight sizing.

Question: What does the raw CF signal produce when nothing else interferes?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core_v1 import run

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    'DM_TABLE':      'dm_history',      # adjust to your Supabase table name
    'DM_DATE_COL':   'date',
    'DM_TICKER_COL': 'ticker',
    'DM_PRICE_COL':  'close',
    'DM_SCORE_COL':  'dm_smoothed',     # adjust to your DM score column

    # ── Strategy rules ────────────────────────────────────────────────────
    'ENTRY_DM_MIN':            65,       # CF score >= 65
    'ENTRY_MA5_RISING':        True,     # 5d MA of CF must be rising
    'ENTRY_PRICE_ABOVE_MA20':  False,    # no price filter
    'EXIT_MA20_THRESHOLD':     50,       # exit when 20d MA < 50

    # ── Sizing ────────────────────────────────────────────────────────────
    'SIZING_MODE':    'equal_weight',    # cash / remaining slots

    # ── Portfolio ─────────────────────────────────────────────────────────
    'INITIAL_CAPITAL': 1_000_000,
    'MAX_POSITIONS':   20,
    'MIN_POSITION':    500,

    # ── Date range ────────────────────────────────────────────────────────
    'START_YEAR': 2016,
    'END_YEAR':   2026,
}

if __name__ == '__main__':
    run("STRATEGY 1 — PURE SIGNAL | CF>=65 | 5d MA rising | Exit MA20<50 | Equal weight", CONFIG)
