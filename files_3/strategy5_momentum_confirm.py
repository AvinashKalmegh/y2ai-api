"""
STRATEGY 5 — MOMENTUM CONFIRMATION
=====================================
Requires both CF signal AND price momentum before entering.
Two independent signals must agree:
  1. CF score >= 65 and 5d MA rising (institutional flow)
  2. Price is above its 20-day price MA (price momentum confirms)

Question: Does requiring price confirmation reduce false entries enough
to improve Sharpe, or does it add lag that reduces CAGR?
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
    'ENTRY_PRICE_ABOVE_MA20':  True,     # price must be above 20d price MA
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
    run("STRATEGY 5 — MOMENTUM CONFIRM | CF>=65 | 5d MA rising | Price>MA20 | Exit MA20<50 | Equal weight", CONFIG)
