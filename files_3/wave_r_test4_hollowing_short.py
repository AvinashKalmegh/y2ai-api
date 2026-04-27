"""
WAVE R TEST 4 — HOLLOWING SHORT BACKTEST REFINEMENT
Y2AI Research | April 2026

Takes the existing Hollowing Short trade history and asks:
Did Wave R contamination hurt our short entries?

For every Hollowing Short trade entry, checks whether the ticker
had a Wave R classification on or near the entry date.

If Wave R entries underperformed Genuine Exit entries in Hollowing Short:
  → Wave R filter would have materially improved historical returns
  → Implement as mandatory pre-entry check

If Wave R and Genuine entries performed similarly:
  → Wave R filter adds caution but isn't a hard gate for short entries

INPUTS:
  wave_r_events.csv       — all classified events (corrected threshold)
  hollowing_short_trades.csv — export from your trade log
      Required columns: entry_date, ticker, entry_price, exit_price,
                        pnl_pct, exit_reason
      (export from Hollowing_Short sheet or Supabase trade table)

OUTPUTS:
  wave_r_hs_refinement.csv
  wave_r_hs_summary.txt

RUN:     python wave_r_test4_hollowing_short.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta

HMS_STABLE  = 0.05
HMS_DECLINE = 0.05
MATCH_WINDOW_DAYS = 10  # look for Wave R signal within ±10 days of entry

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading files...")
events = pd.read_csv("wave_r_events.csv")
events["date"] = pd.to_datetime(events["date"])

try:
    trades = pd.read_csv("hollowing_short_trades.csv")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    print(f"  Trade log loaded: {len(trades):,} trades")
    has_trades = True
except FileNotFoundError:
    print("  ⚠ hollowing_short_trades.csv not found")
    print("  Export Hollowing Short trade log from Supabase or Google Sheets")
    print("  Required columns: entry_date, ticker, entry_price, exit_price, pnl_pct, exit_reason")
    print("\n  Running diagnostic mode — showing what Wave R coverage looks like")
    print("  across current Hollowing Short positions instead.\n")
    has_trades = False

# ── RECLASSIFY ────────────────────────────────────────────────────────────────
def reclassify(hms_chg):
    if pd.isna(hms_chg):       return "UNKNOWN"
    if hms_chg > HMS_STABLE:   return "WAVE_R_STRONG"
    if abs(hms_chg) <= HMS_STABLE: return "WAVE_R"
    if hms_chg <= -HMS_DECLINE: return "GENUINE_EXIT"
    return "AMBIGUOUS"

events["classification"] = events["hms_chg5"].apply(reclassify)

# Build Wave R lookup: {(ticker, date): classification}
wave_r_lookup = {}
for _, row in events.iterrows():
    wave_r_lookup[(row["ticker"], row["date"].date())] = row["classification"]

if not has_trades:
    # ── DIAGNOSTIC MODE — current HS positions ────────────────────────────────
    # Manually define current Hollowing Short book for diagnostic
    current_hs = [
        {"ticker": "BLDR", "entry_date": "2026-03-23"},
        {"ticker": "LDOS", "entry_date": "2026-03-23"},
        {"ticker": "WDAY", "entry_date": "2026-03-23"},
        {"ticker": "ARE",  "entry_date": "2026-03-23"},
        {"ticker": "BAH",  "entry_date": "2026-03-23"},
        {"ticker": "ADBE", "entry_date": "2026-03-23"},
        {"ticker": "DXCM", "entry_date": "2026-03-23"},
        {"ticker": "CRM",  "entry_date": "2026-03-23"},
        {"ticker": "MSI",  "entry_date": "2026-04-20"},
        {"ticker": "KR",   "entry_date": "2026-04-20"},
        {"ticker": "MOS",  "entry_date": "2026-04-17"},
        {"ticker": "CF",   "entry_date": "2026-04-20"},
        {"ticker": "VRSK", "entry_date": "2026-03-23"},
        {"ticker": "CSGP", "entry_date": "2026-03-23"},
    ]
    trades = pd.DataFrame(current_hs)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["pnl_pct"] = None
    trades["exit_reason"] = "OPEN"
    print("Using current Hollowing Short book (14 positions) for diagnostic.\n")

# ── MATCH WAVE R CLASSIFICATION TO EACH TRADE ENTRY ──────────────────────────
def get_wave_r_class(ticker, entry_date):
    """Check for Wave R signal within ±10 days of entry."""
    for offset in range(-MATCH_WINDOW_DAYS, MATCH_WINDOW_DAYS + 1):
        check_date = (entry_date + timedelta(days=offset)).date()
        cls = wave_r_lookup.get((ticker, check_date))
        if cls and cls != "UNKNOWN":
            return cls, offset
    return "NO_SIGNAL", None

print("Matching Wave R classifications to trade entries...")
trade_results = []
for _, trade in trades.iterrows():
    cls, offset = get_wave_r_class(trade["ticker"], trade["entry_date"])
    trade_results.append({
        **trade.to_dict(),
        "wave_r_class": cls,
        "wave_r_offset_days": offset,
        "is_wave_r": cls in ("WAVE_R", "WAVE_R_STRONG"),
        "is_genuine": cls == "GENUINE_EXIT",
    })

df_trades = pd.DataFrame(trade_results)

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
print("\n═" * 35)
print("HOLLOWING SHORT — WAVE R CLASSIFICATION AT ENTRY")
print("═" * 70)

class_counts = df_trades["wave_r_class"].value_counts()
print(f"\nClassification of {len(df_trades)} trade entries:")
for cls, n in class_counts.items():
    pct = n / len(df_trades) * 100
    print(f"  {cls:20}: {n:4} ({pct:.1f}%)")

# Wave R contamination rate
wave_r_pct = df_trades["is_wave_r"].mean() * 100
genuine_pct = df_trades["is_genuine"].mean() * 100
print(f"\nWave R contamination rate: {wave_r_pct:.1f}% of short entries")
print(f"Genuine exit entries:       {genuine_pct:.1f}% of short entries")

if has_trades and "pnl_pct" in df_trades.columns and df_trades["pnl_pct"].notna().sum() > 5:
    # Performance comparison
    wr_trades = df_trades[df_trades["is_wave_r"] & df_trades["pnl_pct"].notna()]
    ge_trades = df_trades[df_trades["is_genuine"] & df_trades["pnl_pct"].notna()]

    print(f"\nPERFORMANCE COMPARISON — Wave R entries vs Genuine Exit entries:")
    print(f"  Wave R entries:    n={len(wr_trades)} | avg P&L={wr_trades['pnl_pct'].mean():+.2f}%")
    print(f"  Genuine entries:   n={len(ge_trades)} | avg P&L={ge_trades['pnl_pct'].mean():+.2f}%")

    if len(wr_trades) >= 5 and len(ge_trades) >= 5:
        t, p = stats.ttest_ind(wr_trades["pnl_pct"], ge_trades["pnl_pct"])
        direction = "Genuine > Wave R ✓" if ge_trades["pnl_pct"].mean() > wr_trades["pnl_pct"].mean() else "Wave R > Genuine ✗"
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  p={p:.4f} | {sig} | {direction}")

        if ge_trades["pnl_pct"].mean() > wr_trades["pnl_pct"].mean() and p < 0.05:
            improvement = ge_trades["pnl_pct"].mean() - wr_trades["pnl_pct"].mean()
            print(f"\n  ✓ FILTER VALIDATED: Excluding Wave R entries would have")
            print(f"    improved avg short P&L by {improvement:+.2f} percentage points.")
            print(f"    RECOMMENDATION: Implement Wave R as mandatory pre-entry gate.")
        else:
            print(f"\n  ~ FILTER NOT VALIDATED at this sample size.")
            print(f"    Wave R is a caution flag, not a hard gate, for short entries.")
else:
    # Diagnostic output — show classification per position
    print(f"\nCURRENT POSITION CLASSIFICATIONS:")
    print(f"{'Ticker':8} {'Entry Date':12} {'Wave R Class':20} {'Offset':8} {'Action'}")
    print("-" * 65)
    for _, r in df_trades.sort_values("wave_r_class").iterrows():
        action = "⚠ REVIEW" if r["is_wave_r"] else ("✓ CONFIRMED" if r["is_genuine"] else "— NO DATA")
        od = r["wave_r_offset_days"]
        offset_str = f"{int(od):+d}d" if pd.notna(od) else "N/A"
        print(f"  {r['ticker']:6} {str(r['entry_date'].date()):12} "
              f"{r['wave_r_class']:20} {offset_str:8} {action}")

# Exit reason analysis
if has_trades and "exit_reason" in df_trades.columns:
    print(f"\nEXIT REASON BY WAVE R CLASS:")
    exit_analysis = df_trades.groupby(["wave_r_class", "exit_reason"]).size().unstack(fill_value=0)
    print(exit_analysis.to_string())

# Save
df_trades.to_csv("wave_r_hs_refinement.csv", index=False)

# Write summary
with open("wave_r_hs_summary.txt", "w") as f:
    f.write("WAVE R — HOLLOWING SHORT REFINEMENT\n")
    f.write(f"Y2AI Research | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Trades analyzed: {len(df_trades)}\n")
    f.write(f"Wave R contamination rate: {wave_r_pct:.1f}%\n")
    f.write(f"Genuine exit entries: {genuine_pct:.1f}%\n\n")
    f.write("POSITIONS FLAGGED AS WAVE R (review before entering short):\n")
    wave_r_positions = df_trades[df_trades["is_wave_r"]]
    for _, r in wave_r_positions.iterrows():
        f.write(f"  {r['ticker']:6} | Entry: {str(r['entry_date'].date())} "
                f"| Class: {r['wave_r_class']}\n")
    f.write("\nPOSITIONS CONFIRMED AS GENUINE EXIT (strongest short candidates):\n")
    genuine_positions = df_trades[df_trades["is_genuine"]]
    for _, r in genuine_positions.iterrows():
        f.write(f"  {r['ticker']:6} | Entry: {str(r['entry_date'].date())}\n")

print(f"\nSaved: wave_r_hs_refinement.csv")
print(f"Saved: wave_r_hs_summary.txt")
print("\nNOTE: For full performance analysis, export trade log from Supabase/Sheets")
print("      as hollowing_short_trades.csv with columns:")
print("      entry_date, ticker, entry_price, exit_price, pnl_pct, exit_reason")
