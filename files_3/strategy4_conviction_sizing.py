"""
STRATEGY 4 — CONVICTION SIZING
================================
Same entry and exit as Strategy 1.
Position size scales with CF score at entry — higher conviction = larger position.
Tests whether the CF score predicts forward return magnitude.

Tiers:
  CF >= 90  → 8% of initial capital
  CF 80-89  → 6.5% of initial capital
  CF 70-79  → 5% of initial capital
  CF 65-69  → 3.5% of initial capital

Question: Does the CF score at entry predict forward returns well enough
to justify concentrating capital in higher-scored entries?
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
    'EXIT_MA20_THRESHOLD':     50,

    # ── Sizing — DM-tiered as % of initial capital ────────────────────────
    'SIZING_MODE':    'dm_tiered',
    # Tiers defined in sim_core_v1.get_allocation():
    #   CF >= 90  → 8% of initial capital
    #   CF 80-89  → 6.5%
    #   CF 70-79  → 5%
    #   CF 65-69  → 3.5%

    # ── Portfolio ─────────────────────────────────────────────────────────
    'INITIAL_CAPITAL': 1_000_000,
    'MAX_POSITIONS':   20,
    'MIN_POSITION':    500,

    # ── Date range ────────────────────────────────────────────────────────
    'START_YEAR': 2016,
    'END_YEAR':   2026,
}

if __name__ == '__main__':
    run("STRATEGY 4 — CONVICTION SIZING | CF>=65 | 5d MA rising | Exit MA20<50 | DM-tiered", CONFIG)
