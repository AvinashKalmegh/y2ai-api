"""
WAVE R TEST 4 — HOLLOWING SHORT BACKTEST REFINEMENT
Y2AI Research | April 2026

Of all Hollowing Short trades ever made, how many had Wave R contamination
at entry? Did those contaminated entries perform worse than clean
genuine-exit entries?

If yes → Wave R filter should be a mandatory hard gate before entry.
If no  → Wave R is a yellow flag only, not a hard rule.

INPUTS:
    wave_r_events.csv              — 120,070 classified events
    hollowing_short_trades.tsv     — trade export (optional; diagnostic mode falls back
                                      to the current 14-position book with live P&L)

OUTPUTS:
    wave_r_hs_refinement.csv
    wave_r_hs_summary.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta

HMS_STABLE        = 0.05
HMS_DECLINE       = 0.05
MATCH_WINDOW_DAYS = 10
MIN_TRADES        = 5

# ── LOAD WAVE R EVENTS ────────────────────────────────────────────────────────
print("Loading wave_r_events.csv...")
try:
    events = pd.read_csv("wave_r_events.csv")
    events["date"] = pd.to_datetime(events["date"])
    print(f"  Loaded: {len(events):,} events | "
          f"{events['ticker'].nunique()} tickers | "
          f"{events['date'].min().date()} → {events['date'].max().date()}")
except FileNotFoundError:
    print("ERROR: wave_r_events.csv not found.")
    print("Run wave_r_validator first to generate this file.")
    raise SystemExit(1)

# ── RECLASSIFY WITH CORRECT HMS THRESHOLD ─────────────────────────────────────
def reclassify(hms_chg):
    if pd.isna(hms_chg):
        return "UNKNOWN"
    if hms_chg > HMS_STABLE:
        return "WAVE_R_STRONG"       # DM down, HMS rising
    if abs(hms_chg) <= HMS_STABLE:
        return "WAVE_R"               # DM down, HMS flat
    if hms_chg <= -HMS_DECLINE:
        return "GENUINE_EXIT"         # both down
    return "AMBIGUOUS"

events["classification"] = events["hms_chg5"].apply(reclassify)

# Build (ticker, date) → signal dict for O(1) lookup
print("Building lookup index...")
wave_r_lookup = {}
for _, row in events.iterrows():
    wave_r_lookup[(row["ticker"], row["date"].date())] = {
        "classification": row["classification"],
        "dm_score":       row["dm_score"],
        "hms_score":      row["hms_score"],
        "hms_chg5":       row["hms_chg5"],
    }
print(f"  Index built: {len(wave_r_lookup):,} (ticker, date) pairs\n")

# ── LOAD TRADE HISTORY ────────────────────────────────────────────────────────
print("Loading hollowing_short_trades.tsv...")
has_trades = False
try:
    trades = pd.read_csv("hollowing_short_trades.tsv", sep="\t")

    # Normalize column names from the Y2AI blotter schema.
    rename = {
        "Ticker":       "ticker",
        "Entry_Date":   "entry_date",
        "Entry_Price":  "entry_price",
        "Current_Price":"exit_price",   # proxy for open positions
        "P&L_%":        "pnl_pct",
        "Exit_Reason":  "exit_reason",
        "Status":       "status",
    }
    trades = trades.rename(columns={k: v for k, v in rename.items() if k in trades.columns})

    # Blotter mixes US ("2/10/2026") and ISO ("2026-04-10") date formats.
    # format="mixed" lets pandas pick per-cell instead of inferring once.
    trades["entry_date"] = pd.to_datetime(
        trades["entry_date"], errors="coerce", format="mixed", dayfirst=False
    )

    # Blotter stores P&L as a fraction (e.g. 0.0328 means +3.28%). Convert to percent.
    trades["pnl_pct"] = pd.to_numeric(trades["pnl_pct"], errors="coerce") * 100

    # Mark ACTIVE rows with an exit_reason of OPEN so the "open positions"
    # branch still triggers and closed trades drive the stat test.
    if "status" in trades.columns:
        trades["exit_reason"] = trades["exit_reason"].fillna("").astype(str)
        trades.loc[trades["status"].str.upper() == "ACTIVE", "exit_reason"] = "OPEN"

    required_cols = ["entry_date", "ticker", "pnl_pct"]
    missing = [c for c in required_cols if c not in trades.columns]
    if missing:
        print(f"ERROR: missing required columns: {missing}")
        print(f"Found columns: {list(trades.columns)}")
        raise SystemExit(1)

    # Drop rows with unparseable entry_date
    before = len(trades)
    trades = trades.dropna(subset=["entry_date"])
    if len(trades) < before:
        print(f"  Dropped {before - len(trades)} rows with unparseable entry_date")

    print(f"  Loaded: {len(trades):,} trades | "
          f"{trades['ticker'].nunique()} unique tickers | "
          f"{trades['entry_date'].min().date()} → {trades['entry_date'].max().date()}")
    if "status" in trades.columns:
        open_n   = (trades["exit_reason"] == "OPEN").sum()
        closed_n = len(trades) - open_n
        print(f"  Status: {open_n} ACTIVE (open) | {closed_n} CLOSED")
    has_trades = True

except FileNotFoundError:
    print("  hollowing_short_trades.tsv not found.")
    print("  Running in DIAGNOSTIC MODE using current 14 live positions (with P&L).\n")

    current_hs = [
        {"ticker": "BLDR", "entry_date": "2026-03-23", "pnl_pct": 30.1, "exit_reason": "OPEN"},
        {"ticker": "LDOS", "entry_date": "2026-03-23", "pnl_pct": 21.3, "exit_reason": "OPEN"},
        {"ticker": "WDAY", "entry_date": "2026-03-23", "pnl_pct": 15.7, "exit_reason": "OPEN"},
        {"ticker": "ARE",  "entry_date": "2026-03-23", "pnl_pct": 15.9, "exit_reason": "OPEN"},
        {"ticker": "BAH",  "entry_date": "2026-03-23", "pnl_pct":  9.9, "exit_reason": "OPEN"},
        {"ticker": "ADBE", "entry_date": "2026-03-23", "pnl_pct":  6.6, "exit_reason": "OPEN"},
        {"ticker": "DXCM", "entry_date": "2026-03-23", "pnl_pct":  8.2, "exit_reason": "OPEN"},
        {"ticker": "CRM",  "entry_date": "2026-03-23", "pnl_pct":  3.3, "exit_reason": "OPEN"},
        {"ticker": "MSI",  "entry_date": "2026-04-20", "pnl_pct": -1.2, "exit_reason": "OPEN"},
        {"ticker": "KR",   "entry_date": "2026-04-20", "pnl_pct": -1.4, "exit_reason": "OPEN"},
        {"ticker": "MOS",  "entry_date": "2026-04-17", "pnl_pct": -2.0, "exit_reason": "OPEN"},
        {"ticker": "CF",   "entry_date": "2026-04-20", "pnl_pct": -4.6, "exit_reason": "OPEN"},
        {"ticker": "VRSK", "entry_date": "2026-03-23", "pnl_pct": -7.5, "exit_reason": "OPEN"},
        {"ticker": "CSGP", "entry_date": "2026-03-23", "pnl_pct": -4.3, "exit_reason": "OPEN"},
    ]
    trades = pd.DataFrame(current_hs)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])

# ── CLASSIFY EACH TRADE AT ENTRY ──────────────────────────────────────────────
def get_classification(ticker, entry_date):
    """Find closest Wave R signal within ±MATCH_WINDOW_DAYS of entry_date."""
    for offset in range(0, MATCH_WINDOW_DAYS + 1):
        dirs = [0] if offset == 0 else [1, -1]
        for direction in dirs:
            check_date = (entry_date + timedelta(days=offset * direction)).date()
            signal = wave_r_lookup.get((ticker, check_date))
            if signal and signal["classification"] != "UNKNOWN":
                return signal["classification"], offset * direction, signal
    return "NO_SIGNAL", None, {}

print("Classifying each trade entry...")
results = []
for _, trade in trades.iterrows():
    cls, offset, signal = get_classification(trade["ticker"], trade["entry_date"])
    results.append({
        **trade.to_dict(),
        "wave_r_class":      cls,
        "offset_days":       offset,
        "dm_at_entry":       signal.get("dm_score"),
        "hms_at_entry":      signal.get("hms_score"),
        "hms_chg5_at_entry": signal.get("hms_chg5"),
        "is_wave_r":         cls in ("WAVE_R", "WAVE_R_STRONG"),
        "is_genuine":        cls == "GENUINE_EXIT",
        "no_signal":         cls == "NO_SIGNAL",
    })

df = pd.DataFrame(results)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("WAVE R TEST 4 — HOLLOWING SHORT BACKTEST REFINEMENT")
print("═" * 70)

total = len(df)
wr_n  = int(df["is_wave_r"].sum())
ge_n  = int(df["is_genuine"].sum())
ns_n  = int(df["no_signal"].sum())

print(f"\nTrade entries classified: {total}")
print(f"  Wave R (contaminated):   {wr_n:4} ({wr_n/total*100:.1f}%)")
print(f"  Genuine Exit (clean):    {ge_n:4} ({ge_n/total*100:.1f}%)")
print(f"  No signal found:         {ns_n:4} ({ns_n/total*100:.1f}%)")

print(f"\n{'Ticker':6} {'Entry':12} {'Wave R Class':20} "
      f"{'DM':>6} {'HMS chg':>8} {'P&L':>7} {'Exit Reason'}")
print("-" * 75)
for _, r in df.sort_values(["wave_r_class", "ticker"]).iterrows():
    flag = "⚠" if r["is_wave_r"] else ("✓" if r["is_genuine"] else "—")
    dm_str  = f"{r['dm_at_entry']:.1f}"       if pd.notna(r.get("dm_at_entry"))       else "N/A"
    hms_str = f"{r['hms_chg5_at_entry']:.3f}" if pd.notna(r.get("hms_chg5_at_entry")) else "N/A"
    pnl_str = f"{r['pnl_pct']:+.1f}%"         if pd.notna(r.get("pnl_pct"))          else "OPEN"
    exit_r  = str(r.get("exit_reason", ""))[:15]
    print(f"  {r['ticker']:4} {str(r['entry_date'].date()):12} "
          f"{flag} {r['wave_r_class']:18} "
          f"{dm_str:>6} {hms_str:>8} {pnl_str:>7} {exit_r}")

# ── PERFORMANCE COMPARISON ────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("PERFORMANCE COMPARISON — Wave R vs Genuine Exit")
print("═" * 70)

wr_trades = df[df["is_wave_r"]  & df["pnl_pct"].notna()]
ge_trades = df[df["is_genuine"] & df["pnl_pct"].notna()]
ns_trades = df[df["no_signal"]  & df["pnl_pct"].notna()]

groups = [
    ("Wave R (contaminated)", wr_trades),
    ("Genuine Exit (clean)",  ge_trades),
    ("No signal",             ns_trades),
]

for label, grp in groups:
    if len(grp) == 0:
        print(f"\n  {label}: no completed trades")
        continue
    wins = int((grp["pnl_pct"] > 0).sum())
    print(f"\n  {label}:")
    print(f"    Trades:    {len(grp)}")
    print(f"    Avg P&L:   {grp['pnl_pct'].mean():+.2f}%")
    print(f"    Median:    {grp['pnl_pct'].median():+.2f}%")
    print(f"    Win rate:  {wins}/{len(grp)} ({wins/len(grp)*100:.0f}%)")
    print(f"    Best:      {grp['pnl_pct'].max():+.2f}%")
    print(f"    Worst:     {grp['pnl_pct'].min():+.2f}%")

if len(wr_trades) >= MIN_TRADES and len(ge_trades) >= MIN_TRADES:
    t, p = stats.ttest_ind(wr_trades["pnl_pct"], ge_trades["pnl_pct"])
    ge_better = ge_trades["pnl_pct"].mean() > wr_trades["pnl_pct"].mean()
    improvement = ge_trades["pnl_pct"].mean() - wr_trades["pnl_pct"].mean()
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"

    print(f"\n{'-'*70}")
    print("STATISTICAL TEST (t-test, Wave R vs Genuine Exit):")
    print(f"  p = {p:.4f} | {sig}")
    print(f"  Genuine Exit avg: {ge_trades['pnl_pct'].mean():+.2f}%")
    print(f"  Wave R avg:       {wr_trades['pnl_pct'].mean():+.2f}%")
    print(f"  Difference:       {improvement:+.2f} percentage points")

    print("\nVERDICT:")
    if ge_better and p < 0.05:
        print(f"  ✓ FILTER VALIDATED — Genuine exits outperformed Wave R entries")
        print(f"    by {improvement:+.2f} percentage points (statistically significant).")
        print("    RECOMMENDATION: Implement Wave R as a MANDATORY HARD GATE.")
    elif ge_better and p >= 0.05:
        print(f"  ~ DIRECTIONAL SUPPORT — Genuine exits outperformed Wave R entries")
        print(f"    by {improvement:+.2f} points but not statistically significant")
        print("    RECOMMENDATION: Wave R is a YELLOW FLAG — apply caution.")
    else:
        print("  ~ NOT VALIDATED — Wave R entries performed similarly or better.")
        print("    RECOMMENDATION: theoretical caution flag only.")
else:
    print(f"\n  Insufficient trades for statistical test "
          f"(Wave R: {len(wr_trades)}, Genuine: {len(ge_trades)}, need {MIN_TRADES} each)")

# ── BY EXIT REASON ────────────────────────────────────────────────────────────
if "exit_reason" in df.columns and df["exit_reason"].notna().sum() > 0:
    closed = df[df["exit_reason"] != "OPEN"]
    if len(closed) > 0:
        print(f"\n{'-'*70}")
        print("EXIT REASON BREAKDOWN BY WAVE R CLASS:")
        try:
            pivot = closed.groupby(["wave_r_class", "exit_reason"]).size().unstack(fill_value=0)
            print(pivot.to_string())
        except Exception:
            pass

# ── OPEN POSITIONS FLAGGED AS WAVE R ──────────────────────────────────────────
open_wave_r = df[df["is_wave_r"] & (df.get("exit_reason", "") == "OPEN")]
if len(open_wave_r) > 0:
    print(f"\n{'-'*70}")
    print("OPEN POSITIONS FLAGGED AS WAVE R — REVIEW BEFORE ADDING TO:")
    for _, r in open_wave_r.iterrows():
        pnl = r["pnl_pct"]
        pnl_str = f"{pnl:+.1f}%" if pd.notna(pnl) else "N/A"
        print(f"  {r['ticker']:6} | Class: {r['wave_r_class']:20} | Current P&L: {pnl_str}")
    print("  These are not automatic exits — but do not add to these positions.")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv("wave_r_hs_refinement.csv", index=False)

with open("wave_r_hs_summary.txt", "w", encoding="utf-8") as f:
    f.write("WAVE R — HOLLOWING SHORT BACKTEST REFINEMENT\n")
    f.write(f"Y2AI Research | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total trades analyzed: {total}\n")
    f.write(f"Wave R contamination rate: {wr_n/total*100:.1f}% ({wr_n} trades)\n")
    f.write(f"Genuine exit rate: {ge_n/total*100:.1f}% ({ge_n} trades)\n")
    f.write(f"No signal: {ns_n/total*100:.1f}% ({ns_n} trades)\n\n")

    if len(wr_trades) >= MIN_TRADES and len(ge_trades) >= MIN_TRADES:
        f.write("PERFORMANCE:\n")
        f.write(f"  Genuine Exit avg P&L: {ge_trades['pnl_pct'].mean():+.2f}%\n")
        f.write(f"  Wave R avg P&L:       {wr_trades['pnl_pct'].mean():+.2f}%\n")
        f.write(f"  Difference: {ge_trades['pnl_pct'].mean()-wr_trades['pnl_pct'].mean():+.2f} pts\n\n")

    f.write("WAVE R POSITIONS (review before adding to):\n")
    for _, r in df[df["is_wave_r"]].iterrows():
        pnl = r.get("pnl_pct")
        line = f"  {r['ticker']:6} | Entry: {str(r['entry_date'].date())} | Class: {r['wave_r_class']}"
        if pd.notna(pnl):
            line += f" | P&L: {pnl:+.1f}%"
        f.write(line + "\n")

    f.write("\nGENUINE EXIT POSITIONS (highest conviction shorts):\n")
    for _, r in df[df["is_genuine"]].iterrows():
        pnl = r.get("pnl_pct")
        line = f"  {r['ticker']:6} | Entry: {str(r['entry_date'].date())}"
        if pd.notna(pnl):
            line += f" | P&L: {pnl:+.1f}%"
        f.write(line + "\n")

print("\nSaved: wave_r_hs_refinement.csv")
print("Saved: wave_r_hs_summary.txt")

if not has_trades:
    print(f"\n{'═'*70}")
    print("NEXT STEP: export the complete Hollowing Short trade history as")
    print("  hollowing_short_trades.tsv with columns:")
    print("    entry_date, ticker, entry_price, exit_price, pnl_pct, exit_reason")
    print("  and rerun. The verdict will then be statistically definitive.")
    print("═" * 70)
