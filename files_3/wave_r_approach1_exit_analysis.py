"""
═══════════════════════════════════════════════════════════════════════════════
WAVE R — APPROACH 1: RETROSPECTIVE EXIT ANALYSIS
Y2AI Research | April 2026
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS ANSWERS:
    For every exit that already fired in PureSim since March 23, 2026 —
    what was HMS doing at the time of exit?

    If HMS was stable or rising when MA20 triggered the exit (Wave R false exit):
        → What did the price do in the 30 days AFTER we exited?
        → Did we exit too early because of hedging noise?

    If HMS was also declining (genuine exit):
        → Did the price continue falling as expected?
        → Was the exit correct?

    The gap between these two groups tells you whether the Wave R filter
    would have materially improved PureSim returns historically.

VERDICT LOGIC:
    Wave R false exits that RECOVERED after exit → filter would have helped
    Wave R false exits that CONTINUED FALLING  → filter would not have helped
    Genuine exits that CONTINUED FALLING       → exits were correct
    Genuine exits that RECOVERED               → exits were unlucky but correct

INPUTS (all already available locally):
    wave_r_events_v2.csv         — classified events with hms_chg20
    puresim_closed_trades.csv    — PureSim closed trade history (see note below)

    NOTE: puresim_closed_trades.csv needs to be exported from your
    Supabase trade log or Google Sheets PureSim tab.
    Required columns: exit_date, ticker, entry_price, exit_price,
                      pnl_pct, exit_reason, entry_date

    If the file is not available, the script runs in DEMO MODE using
    the known closed PureSim trades from the chronicle.

OUTPUTS:
    wave_r_exit_analysis.csv     — every exit with Wave R classification
    wave_r_exit_summary.txt      — plain English verdict
    wave_r_exit_chart.txt        — ASCII comparison table

RUN:
    python wave_r_approach1_exit_analysis.py

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
HMS_STABLE    = 0.05   # |hms_chg20| < this = HMS stable (Wave R)
HMS_DECLINE   = 0.05   # hms_chg20 <= -this = HMS declining (genuine exit)
MATCH_WINDOW  = 10     # days to search for Wave R signal around exit date
RECOVERY_DAYS = 30     # days post-exit to measure price/DM performance

# ── DEMO MODE — known PureSim closed trades from chronicle ───────────────────
# These are the actual closed trades from the ARGUS chronicle.
# Replace with CSV export for full analysis.
DEMO_TRADES = [
    # Trades closed via MA20 exit rule (the ones we're testing)
    {"exit_date": "2026-04-22", "ticker": "AMAT", "entry_date": "2026-03-23",
     "entry_price": 397.81, "exit_price": 394.33, "pnl_pct": -0.9,
     "exit_reason": "MA20_EXIT"},
    # Trades closed via T2/T3 rules (not MA20 — different logic)
    {"exit_date": "2026-04-17", "ticker": "PAYC", "entry_date": "2026-04-11",
     "entry_price": 113.59, "exit_price": 93.19, "pnl_pct": -17.1,
     "exit_reason": "T3_LOSS_LIMIT"},
    {"exit_date": "2026-04-21", "ticker": "COIN", "entry_date": "2026-04-10",
     "entry_price": 169.02, "exit_price": 142.13, "pnl_pct": -15.9,
     "exit_reason": "T3_LOSS_LIMIT"},
    {"exit_date": "2026-04-21", "ticker": "SHW", "entry_date": "2026-02-10",
     "entry_price": 364.65, "exit_price": 343.93, "pnl_pct": 5.68,
     "exit_reason": "T2_PHASE_CONFIRM"},
    {"exit_date": "2026-04-17", "ticker": "RJF", "entry_date": "2026-02-10",
     "entry_price": 158.48, "exit_price": 161.19, "pnl_pct": 1.71,
     "exit_reason": "T2_PHASE_CONFIRM"},
    {"exit_date": "2026-04-15", "ticker": "LPLA", "entry_date": "2026-02-10",
     "entry_price": 360.58, "exit_price": 408.62, "pnl_pct": 13.33,
     "exit_reason": "T2_PHASE_CONFIRM"},
]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("═" * 70)
print("WAVE R — APPROACH 1: RETROSPECTIVE EXIT ANALYSIS")
print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("═" * 70 + "\n")

# Load Wave R events (with hms_chg20 added by Avinash's script)
print("Loading wave_r_events_v2.csv...")
try:
    events = pd.read_csv("wave_r_events_v2.csv")
    events["date"] = pd.to_datetime(events["date"])
    has_hms20 = "hms_chg20" in events.columns
    print(f"  {len(events):,} events | hms_chg20 available: {has_hms20}")
    if not has_hms20:
        print("  ⚠ hms_chg20 not found — using hms_chg5 instead")
        events["hms_chg20"] = events["hms_chg5"]
except FileNotFoundError:
    print("  ⚠ wave_r_events_v2.csv not found — trying wave_r_events.csv")
    try:
        events = pd.read_csv("wave_r_events.csv")
        events["date"] = pd.to_datetime(events["date"])
        events["hms_chg20"] = events["hms_chg5"]
        print(f"  Loaded wave_r_events.csv: {len(events):,} events (using hms_chg5)")
    except FileNotFoundError:
        print("  ERROR: No wave_r_events file found. Run wave_r_validator.py first.")
        exit(1)

# Build lookup: (ticker, date) → {hms_chg20, dm_score, hms_score}
event_lookup = {}
for _, row in events.iterrows():
    event_lookup[(row["ticker"], row["date"].date())] = {
        "hms_chg20":  row.get("hms_chg20"),
        "hms_chg5":   row.get("hms_chg5"),
        "dm_score":   row.get("dm_score"),
        "hms_score":  row.get("hms_score"),
        "dm_chg5":    row.get("dm_chg5"),
    }

# Load DM history for post-exit performance
print("Building DM pivot for post-exit analysis...")
dm_pivot = events.pivot_table(
    index="date", columns="ticker", values="dm_score"
)
print(f"  DM pivot: {len(dm_pivot)} dates × {len(dm_pivot.columns)} tickers\n")

# Load PureSim closed trades
print("Loading PureSim closed trades...")
try:
    trades = pd.read_csv("puresim_closed_trades.csv")
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    print(f"  Loaded: {len(trades):,} closed trades")
    demo_mode = False
except FileNotFoundError:
    print("  puresim_closed_trades.csv not found — using DEMO MODE")
    print("  (Known PureSim closed trades from ARGUS chronicle)")
    trades = pd.DataFrame(DEMO_TRADES)
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    demo_mode = True

print(f"  Trades to analyze: {len(trades)}")
print(f"  Demo mode: {demo_mode}\n")

# ── CLASSIFY EACH EXIT ────────────────────────────────────────────────────────
print("Classifying each exit with Wave R signal...")

def get_hms_at_exit(ticker, exit_date):
    """
    Find the HMS change reading closest to the exit date.
    Searches within ±MATCH_WINDOW days.
    Returns (hms_chg20, dm_score, hms_score, offset_days) or None.
    """
    for offset in range(0, MATCH_WINDOW + 1):
        for direction in ([0] if offset == 0 else [1, -1]):
            check_date = (exit_date + timedelta(days=offset * direction)).date()
            signal = event_lookup.get((ticker, check_date))
            if signal and signal["hms_chg20"] is not None:
                return signal, offset * direction
    return None, None

def classify_exit(hms_chg20):
    """Classify the HMS reading at exit time."""
    if hms_chg20 is None:
        return "NO_SIGNAL"
    if hms_chg20 > HMS_STABLE:
        return "WAVE_R_STRONG"   # HMS rising while MA20 triggered — false exit?
    if abs(hms_chg20) <= HMS_STABLE:
        return "WAVE_R"          # HMS flat — possibly false exit
    if hms_chg20 <= -HMS_DECLINE:
        return "GENUINE_EXIT"    # HMS also declining — correct exit
    return "AMBIGUOUS"

results = []
for _, trade in trades.iterrows():
    ticker    = trade["ticker"]
    exit_date = trade["exit_date"]

    signal, offset = get_hms_at_exit(ticker, exit_date)
    hms_chg20 = signal["hms_chg20"] if signal else None
    classification = classify_exit(hms_chg20)

    # Measure DM 30 days post-exit
    dm_at_exit = signal["dm_score"] if signal else None
    dm_30d_post = None
    dm_recovery = None

    if dm_at_exit and ticker in dm_pivot.columns:
        fwd_date = exit_date + timedelta(days=int(RECOVERY_DAYS * 7/5))
        candidates = dm_pivot.index[
            (dm_pivot.index >= fwd_date) &
            (dm_pivot.index <= fwd_date + timedelta(days=7))
        ]
        if len(candidates) > 0:
            dm_30d_post = dm_pivot[ticker].get(candidates[0])
            if dm_30d_post and not pd.isna(dm_30d_post) and dm_at_exit:
                dm_recovery = round(float(dm_30d_post - dm_at_exit), 2)

    results.append({
        "ticker":         ticker,
        "entry_date":     trade["entry_date"].date(),
        "exit_date":      exit_date.date(),
        "entry_price":    trade.get("entry_price"),
        "exit_price":     trade.get("exit_price"),
        "pnl_pct":        trade.get("pnl_pct"),
        "exit_reason":    trade.get("exit_reason", "UNKNOWN"),
        "hms_chg20_at_exit": round(hms_chg20, 4) if hms_chg20 else None,
        "dm_at_exit":     round(dm_at_exit, 1) if dm_at_exit else None,
        "hms_at_exit":    round(signal["hms_score"], 3) if signal else None,
        "offset_days":    offset,
        "classification": classification,
        "is_wave_r":      classification in ("WAVE_R", "WAVE_R_STRONG"),
        "is_genuine":     classification == "GENUINE_EXIT",
        "dm_30d_post":    round(dm_30d_post, 1) if dm_30d_post and not pd.isna(dm_30d_post) else None,
        "dm_recovery_30d": dm_recovery,
        "filter_verdict": None,  # filled below
    })

df = pd.DataFrame(results)

# ── FILTER VERDICT ────────────────────────────────────────────────────────────
# For Wave R exits: did the stock recover after we exited?
# "Recover" = DM went UP 30 days after exit (we left too early)
# "Continue falling" = DM went DOWN (exit was actually fine)
for i, row in df.iterrows():
    if row["is_wave_r"] and row["dm_recovery_30d"] is not None:
        if row["dm_recovery_30d"] > 5:
            df.at[i, "filter_verdict"] = "FILTER_WOULD_HELP"
        elif row["dm_recovery_30d"] < -5:
            df.at[i, "filter_verdict"] = "FILTER_IRRELEVANT"
        else:
            df.at[i, "filter_verdict"] = "NEUTRAL"
    elif row["is_genuine"] and row["dm_recovery_30d"] is not None:
        if row["dm_recovery_30d"] < -5:
            df.at[i, "filter_verdict"] = "EXIT_CORRECT"
        elif row["dm_recovery_30d"] > 5:
            df.at[i, "filter_verdict"] = "EXIT_UNLUCKY"
        else:
            df.at[i, "filter_verdict"] = "NEUTRAL"

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("RESULTS — EXIT CLASSIFICATION")
print("═" * 70)

print(f"\n{'Ticker':6} {'Exit Date':12} {'Exit Reason':20} {'HMS chg20':10} "
      f"{'Class':18} {'P&L':7} {'DM Rcv 30d':10} {'Verdict'}")
print("-" * 95)

for _, r in df.sort_values("classification").iterrows():
    hms_str = f"{r['hms_chg20_at_exit']:+.3f}" if r["hms_chg20_at_exit"] else "N/A"
    rec_str = f"{r['dm_recovery_30d']:+.1f}" if r["dm_recovery_30d"] else "pending"
    pnl_str = f"{r['pnl_pct']:+.1f}%" if pd.notna(r.get("pnl_pct")) else "N/A"
    flag = "⚠" if r["is_wave_r"] else ("✓" if r["is_genuine"] else "—")
    verdict = r["filter_verdict"] or "—"
    print(f"  {r['ticker']:4} {str(r['exit_date']):12} "
          f"{str(r['exit_reason'])[:18]:20} {hms_str:10} "
          f"{flag} {r['classification']:16} {pnl_str:7} {rec_str:10} {verdict}")

# ── SUMMARY STATISTICS ────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SUMMARY")
print("═" * 70)

total      = len(df)
wave_r_n   = df["is_wave_r"].sum()
genuine_n  = df["is_genuine"].sum()
no_signal  = (df["classification"] == "NO_SIGNAL").sum()

print(f"\nTotal exits analyzed:  {total}")
print(f"Wave R (HMS stable):   {wave_r_n} ({wave_r_n/total*100:.0f}%)")
print(f"Genuine exit:          {genuine_n} ({genuine_n/total*100:.0f}%)")
print(f"No signal:             {no_signal} ({no_signal/total*100:.0f}%)")

# Only MA20 exits matter for the Wave R filter
# T2/T3 exits are rule-based and bypass the exit check differently
ma20_exits = df[df["exit_reason"] == "MA20_EXIT"]
if len(ma20_exits) > 0:
    print(f"\nMA20 exits specifically (what Wave R filter would affect):")
    wr_ma20 = ma20_exits[ma20_exits["is_wave_r"]]
    ge_ma20 = ma20_exits[ma20_exits["is_genuine"]]
    print(f"  Wave R: {len(wr_ma20)} | Genuine: {len(ge_ma20)}")

# Wave R filter verdict
filter_help    = (df["filter_verdict"] == "FILTER_WOULD_HELP").sum()
filter_irrel   = (df["filter_verdict"] == "FILTER_IRRELEVANT").sum()
exit_correct   = (df["filter_verdict"] == "EXIT_CORRECT").sum()
exit_unlucky   = (df["filter_verdict"] == "EXIT_UNLUCKY").sum()

if wave_r_n > 0:
    print(f"\nWave R exits post-exit DM movement:")
    print(f"  Filter would have helped (DM recovered):   {filter_help}")
    print(f"  Filter irrelevant (DM kept falling):       {filter_irrel}")

    wr_recoveries = df[df["is_wave_r"] & df["dm_recovery_30d"].notna()]["dm_recovery_30d"]
    ge_recoveries = df[df["is_genuine"] & df["dm_recovery_30d"].notna()]["dm_recovery_30d"]

    if len(wr_recoveries) > 0:
        print(f"\n  Wave R avg DM change 30d post-exit: {wr_recoveries.mean():+.2f}")
    if len(ge_recoveries) > 0:
        print(f"  Genuine avg DM change 30d post-exit: {ge_recoveries.mean():+.2f}")

    if len(wr_recoveries) > 0 and len(ge_recoveries) > 0:
        diff = wr_recoveries.mean() - ge_recoveries.mean()
        print(f"  Difference: {diff:+.2f} points")
        print(f"\n  INTERPRETATION:")
        if diff > 5 and filter_help > filter_irrel:
            print(f"  ✓ FILTER WOULD HAVE HELPED — Wave R exits recovered more than genuine exits.")
            print(f"    Build Shadow_WaveR portfolio (Approach 2).")
            print(f"    Add HMS check before MA20 exit in PureSim and Shadow_Rotation.")
        elif diff > 0:
            print(f"  ~ MARGINAL BENEFIT — Wave R exits recovered slightly more.")
            print(f"    Directional support but sample too small ({len(wr_recoveries)} Wave R exits).")
            print(f"    Wait for 30+ MA20 exits before implementing filter.")
        else:
            print(f"  ~ NO BENEFIT — Wave R exits did not recover more than genuine exits.")
            print(f"    Do not implement Wave R as exit filter for PureSim/Shadow_Rotation.")
            print(f"    Wave R remains a monitoring signal only.")

print(f"\n{'═'*70}")
print("NEXT STEP RECOMMENDATION")
print("═" * 70)

if demo_mode:
    print(f"""
  Currently running in DEMO MODE with {total} known closed trades.
  Sample size is too small for a statistically reliable conclusion.

  To get the real answer:
  1. Export PureSim closed trade history from Supabase or Google Sheets
     as puresim_closed_trades.csv with columns:
     exit_date, ticker, entry_price, exit_price, pnl_pct,
     exit_reason, entry_date
  2. Rerun this script — it will switch to full mode automatically.
  3. At 20+ MA20 exits the result becomes statistically meaningful.
  4. At 50+ MA20 exits you can make a production decision with confidence.

  Current PureSim run started March 23, 2026.
  Estimated time to 20 MA20 exits: roughly Day 60-70 (mid-May 2026).
""")
else:
    ma20_count = len(df[df["exit_reason"] == "MA20_EXIT"])
    if ma20_count < 10:
        print(f"""
  {ma20_count} MA20 exits so far — still too few for a reliable conclusion.
  Rerun this script monthly as more exits accumulate.
  Target: 20+ MA20 exits for directional signal, 50+ for production decision.
""")
    elif filter_help > filter_irrel:
        print(f"""
  ✓ Evidence supports building Shadow_WaveR (Approach 2).
  Build the parallel portfolio and let it run for 60 days.
""")
    else:
        print(f"""
  ~ Insufficient evidence to implement Wave R exit filter.
  Continue monitoring. Rerun in 30 days.
""")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv("wave_r_exit_analysis.csv", index=False)
print(f"Saved: wave_r_exit_analysis.csv ({len(df)} exits)\n")

# Write summary
with open("wave_r_exit_summary.txt", "w") as f:
    f.write("WAVE R — APPROACH 1: RETROSPECTIVE EXIT ANALYSIS\n")
    f.write(f"Y2AI Research | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total exits: {total}\n")
    f.write(f"Wave R exits: {wave_r_n} ({wave_r_n/total*100:.0f}%)\n")
    f.write(f"Genuine exits: {genuine_n} ({genuine_n/total*100:.0f}%)\n\n")
    f.write("INDIVIDUAL EXITS:\n")
    for _, r in df.iterrows():
        hms_str = f"{r['hms_chg20_at_exit']:+.3f}" if r["hms_chg20_at_exit"] else "N/A"
        f.write(f"  {r['ticker']:6} | {str(r['exit_date']):12} | "
                f"{str(r['exit_reason'])[:18]:18} | HMS20: {hms_str:8} | "
                f"Class: {r['classification']:18} | "
                f"P&L: {r['pnl_pct']:+.1f}%\n"
                if pd.notna(r.get("pnl_pct")) else
                f"  {r['ticker']:6} | {str(r['exit_date']):12} | "
                f"{str(r['exit_reason'])[:18]:18} | HMS20: {hms_str:8} | "
                f"Class: {r['classification']}\n")

print("Saved: wave_r_exit_summary.txt")
print(f"\nDone: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 70)
