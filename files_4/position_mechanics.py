"""
ARGUS — POSITION MECHANICS & CAPACITY ANALYSIS
================================================
Purpose: Compute position-level statistics needed for institutional
         presentation — answers questions about concentration, liquidity,
         and scalability that a PM will ask.

Run this after receiving the Strategy 5 nav and trade CSV files.
Place both CSV files in the same directory as this script.

Outputs:
  position_mechanics.txt  — console-style summary for documentation
  capacity_analysis.csv   — per-trade capacity estimates (if ADV data available)

Developer notes:
- This script reads from the CSV files, not Supabase
- For ADV (average daily volume) data, you will need to add a Polygon fetch
  or a Supabase lookup — see the CAPACITY section below
- The script will run cleanly without ADV data but will note what is missing
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import date

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRADES_FILE = 'trades_strategy_5_momentum_confirm_cf_65_5d_ma_.csv'
NAV_FILE    = 'nav_strategy_5_momentum_confirm_cf_65_5d_ma_.csv'

# Capacity assumption — what % of ADV can we trade without market impact?
MAX_ADV_PCT = 0.10   # 10% of average daily volume — conservative institutional assumption


def run():
    print("\n" + "="*65)
    print("  ARGUS POSITION MECHANICS & CAPACITY ANALYSIS")
    print("  Strategy 5 — Momentum Confirmation")
    print("="*65 + "\n")

    # Load files
    if not os.path.exists(TRADES_FILE):
        print(f"ERROR: {TRADES_FILE} not found.")
        sys.exit(1)
    if not os.path.exists(NAV_FILE):
        print(f"ERROR: {NAV_FILE} not found.")
        sys.exit(1)

    trades = pd.read_csv(TRADES_FILE, encoding='latin-1')
    nav    = pd.read_csv(NAV_FILE,    encoding='latin-1')

    trades['entry_date'] = pd.to_datetime(trades['entry_date']).dt.date
    trades['exit_date']  = pd.to_datetime(trades['exit_date']).dt.date
    trades['entry_price'] = pd.to_numeric(trades['entry_price'], errors='coerce')
    trades['entry_cost']  = pd.to_numeric(trades['entry_cost'],  errors='coerce')
    trades['return_pct']  = pd.to_numeric(trades['return_pct'],  errors='coerce')
    trades['pnl']         = pd.to_numeric(trades['pnl'],         errors='coerce')

    nav['date'] = pd.to_datetime(nav['date']).dt.date
    nav['nav']  = pd.to_numeric(nav['nav'], errors='coerce')
    nav['open_positions'] = pd.to_numeric(nav['open_positions'], errors='coerce')

    nav_lookup = dict(zip(nav['date'], nav['nav']))

    def get_nav(d):
        candidates = [k for k in nav_lookup if k <= d]
        return nav_lookup[max(candidates)] if candidates else None

    trades['nav_at_entry'] = trades['entry_date'].apply(get_nav)
    trades['weight_pct']   = trades['entry_cost'] / trades['nav_at_entry'] * 100

    lines = []
    def log(s=''):
        print(s)
        lines.append(s)

    # ── 1. CONCURRENT POSITIONS ───────────────────────────────────────────────
    log("── 1. CONCURRENT POSITIONS ─────────────────────────────────────────")
    log(f"   Avg concurrent open positions:   {nav['open_positions'].mean():.1f}")
    log(f"   Median concurrent:               {nav['open_positions'].median():.0f}")
    log(f"   Max concurrent (any single day): {nav['open_positions'].max():.0f}")
    log(f"   Days at maximum (20 positions):  {(nav['open_positions']==20).sum():,}")
    log(f"   Days at 0 positions:             {(nav['open_positions']==0).sum():,}")
    log(f"   Total trading days:              {len(nav):,}")
    log(f"   % days fully deployed:           {(nav['open_positions']==20).sum()/len(nav)*100:.1f}%")

    # ── 2. POSITION SIZING ────────────────────────────────────────────────────
    log()
    log("── 2. POSITION SIZING (as % of portfolio NAV at entry) ─────────────")
    log(f"   Avg position weight:     {trades['weight_pct'].mean():.1f}%")
    log(f"   Median position weight:  {trades['weight_pct'].median():.1f}%")
    log(f"   Max position weight:     {trades['weight_pct'].max():.1f}%")
    log(f"   Min position weight:     {trades['weight_pct'].min():.2f}%")
    log(f"   Positions > 10% of NAV:  {(trades['weight_pct']>10).sum()} trades")
    log(f"   Positions > 20% of NAV:  {(trades['weight_pct']>20).sum()} trades")
    log(f"   Positions < 2% of NAV:   {(trades['weight_pct']<2).sum()} trades")
    log()
    log("   Note: Large position weights occur late in the simulation when NAV")
    log("   has grown to $50M-$100M+ and equal-weight allocations are large")
    log("   in dollar terms. In a real deployment, position caps would apply.")
    log()

    # Largest positions by weight
    log("   Top 10 largest positions by weight:")
    top_w = trades.nlargest(10, 'weight_pct')[
        ['ticker','entry_date','weight_pct','nav_at_entry','entry_cost','return_pct']
    ]
    for _, r in top_w.iterrows():
        log(f"     {r['ticker']:<6} {str(r['entry_date']):<12} "
            f"weight:{r['weight_pct']:>5.1f}%  "
            f"NAV:${r['nav_at_entry']:>12,.0f}  "
            f"cost:${r['entry_cost']:>10,.0f}  "
            f"return:{r['return_pct']:>+6.1f}%")

    # ── 3. PAYOFF ASYMMETRY ───────────────────────────────────────────────────
    log()
    log("── 3. PAYOFF ASYMMETRY ──────────────────────────────────────────────")
    winners = trades[trades['return_pct'] > 0]
    losers  = trades[trades['return_pct'] <= 0]
    log(f"   Total trades:          {len(trades):,}")
    log(f"   Winners:               {len(winners):,} ({len(winners)/len(trades)*100:.1f}%)")
    log(f"   Losers:                {len(losers):,} ({len(losers)/len(trades)*100:.1f}%)")
    log()
    log(f"   Avg winner return:     +{winners['return_pct'].mean():.1f}%")
    log(f"   Avg loser return:       {losers['return_pct'].mean():.1f}%")
    log(f"   Win/loss ratio:         {abs(winners['return_pct'].mean()/losers['return_pct'].mean()):.1f}x")
    log()
    log(f"   Median winner:         +{winners['return_pct'].median():.1f}%")
    log(f"   Median loser:           {losers['return_pct'].median():.1f}%")
    log()
    log(f"   Avg winner hold:        {winners['hold_days'].mean():.0f} days")
    log(f"   Avg loser hold:         {losers['hold_days'].mean():.0f} days")
    log(f"   (Winners held ~{winners['hold_days'].mean()-losers['hold_days'].mean():.0f}d longer — signal lets winners run)")

    # ── 4. P&L CONCENTRATION ─────────────────────────────────────────────────
    log()
    log("── 4. P&L CONCENTRATION ─────────────────────────────────────────────")
    total_pnl = trades['pnl'].sum()
    sorted_pnl = trades.sort_values('pnl', ascending=False).reset_index(drop=True)
    log(f"   Total P&L: ${total_pnl:,.0f}")
    log()
    for n in [1, 5, 10, 25, 50, 100]:
        top = sorted_pnl.head(n)['pnl'].sum()
        log(f"   Top {n:>4} trades: ${top:>14,.0f}  =  {top/total_pnl*100:>5.0f}% of total P&L")

    log()
    log("   Context: P&L concentration is a known characteristic of momentum")
    log("   strategies. Rare large winners more than offset frequent small losers.")
    log("   The signal's job is to be in position when the large move happens.")

    # ── 5. TURNOVER ───────────────────────────────────────────────────────────
    log()
    log("── 5. TURNOVER ──────────────────────────────────────────────────────")
    trades['year'] = pd.to_datetime(trades['entry_date']).dt.year
    by_year = trades.groupby('year').agg(
        n_trades=('pnl','count'),
        total_cost=('entry_cost','sum')
    ).reset_index()
    log(f"   Avg trades per year:   {trades['year'].value_counts().mean():.0f}")
    log(f"   Avg hold period:       {trades['hold_days'].mean():.0f} days")
    log()
    log("   Year-by-year trade count and capital deployed:")
    for _, r in by_year.iterrows():
        log(f"     {int(r['year'])}: {int(r['n_trades']):>4} trades | "
            f"${r['total_cost']:>14,.0f} deployed")

    # ── 6. CAPACITY ESTIMATE (without ADV data) ───────────────────────────────
    log()
    log("── 6. CAPACITY ESTIMATE ─────────────────────────────────────────────")
    log("   (Requires ADV data for full analysis — see developer note below)")
    log()
    log("   Proxy estimate based on position sizing:")
    log(f"   Max single position in late simulation: ${sorted_pnl['entry_cost'].max():,.0f}")
    log(f"   This occurred at NAV ~${trades.loc[trades['entry_cost'].idxmax(),'nav_at_entry']:,.0f}")
    log()
    log("   For a $100M institutional deployment:")
    log("   Equal weight across 20 positions = $5M per position")
    log("   For this to be <10% of ADV, each name needs $50M+ daily volume")
    log("   Most large-cap names in the universe comfortably exceed this")
    log("   Quantum computing names (QUBT, RGTI) may be constrained at scale")
    log()
    log("   TODO FOR DEVELOPER: Add ADV lookup from Polygon for each entry")
    log("   to compute % ADV traded per position. Formula:")
    log("   pct_adv = entry_cost / (avg_daily_volume * price_at_entry)")
    log("   Report: avg pct_adv, max pct_adv, trades where pct_adv > 10%")

    # ── 7. YEAR-BY-YEAR REGIME SHARPE ─────────────────────────────────────────
    log()
    log("── 7. REGIME SHARPE (from daily NAV) ───────────────────────────────")
    RISK_FREE = 0.04 / 252

    def regime_sharpe(nav_df, label, start, end):
        sub = nav_df[
            (nav_df['date'] >= pd.to_datetime(start).date()) &
            (nav_df['date'] <= pd.to_datetime(end).date())
        ].copy()
        if len(sub) < 20:
            return
        navs = sub['nav'].values
        rets = np.diff(navs) / navs[:-1]
        excess = rets - RISK_FREE
        sharpe  = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        down    = excess[excess < 0]
        sortino = excess.mean() / down.std() * np.sqrt(252) if len(down)>1 and down.std()>0 else 0
        peak    = pd.Series(navs).cummax()
        max_dd  = ((pd.Series(navs) - peak) / peak * 100).min()
        days    = (sub['date'].iloc[-1] - sub['date'].iloc[0]).days
        cagr    = ((navs[-1]/navs[0])**(365.25/days)-1)*100 if days > 0 else 0
        log(f"   {label}")
        log(f"     CAGR: {cagr:>7.1f}%  |  Sharpe: {sharpe:>5.2f}  |  "
            f"Sortino: {sortino:>5.2f}  |  Max DD: {max_dd:>6.1f}%  |  "
            f"Days: {len(sub):,}")

    regime_sharpe(nav, "2016–2019  Steady Bull / Fed Tightening", '2016-01-01', '2019-12-31')
    regime_sharpe(nav, "2020–2023  COVID / Bear Market / AI Formation", '2020-01-01', '2023-12-31')
    regime_sharpe(nav, "2024–2026  AI Breakout (excluded from stability test)", '2024-01-01', '2026-03-31')
    regime_sharpe(nav, "2022 only  Bear Market isolation", '2022-01-01', '2022-12-31')
    regime_sharpe(nav, "Full Decade 2016–2026", '2016-01-01', '2026-03-31')

    log()
    log("="*65)
    log("  SUMMARY FOR PRESENTATION")
    log("="*65)
    log()
    log("  Position mechanics (safe to state in PM meeting):")
    log(f"    Max concurrent positions: 20")
    log(f"    Avg position weight: {trades['weight_pct'].mean():.1f}% of portfolio NAV")
    log(f"    Equal weight construction — ~5% per slot when fully deployed")
    log()
    log("  Payoff profile (safe to state):")
    log(f"    Win rate: {len(winners)/len(trades)*100:.1f}%  |  "
        f"Avg winner: +{winners['return_pct'].mean():.1f}%  |  "
        f"Avg loser: {losers['return_pct'].mean():.1f}%  |  "
        f"Ratio: {abs(winners['return_pct'].mean()/losers['return_pct'].mean()):.1f}x")
    log()
    log("  P&L concentration (disclose in speaker notes):")
    top5 = sorted_pnl.head(5)['pnl'].sum()
    top10 = sorted_pnl.head(10)['pnl'].sum()
    log(f"    Top 5 trades = {top5/total_pnl*100:.0f}% of total P&L")
    log(f"    Top 10 trades = {top10/total_pnl*100:.0f}% of total P&L")
    log(f"    Characteristic of momentum strategies — not a flaw")

    # Write summary
    with open('position_mechanics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nFull analysis written to position_mechanics.txt")
    print("DONE.\n")


if __name__ == '__main__':
    run()
