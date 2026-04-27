"""
═══════════════════════════════════════════════════════════════════════════════
WAVE R VALIDATOR v3.0 — Y2AI Research
Fully vectorized. No row-by-row loops. Should complete in 2-5 minutes.
═══════════════════════════════════════════════════════════════════════════════

HYPOTHESIS:
    DM declining + HMS flat/rising = Wave R (institutional hedge, not exit)
    DM declining + HMS declining   = Genuine exit (conviction changing)

    Wave R candidates should show DM recovery vs continued decline.

FIVE-FACTOR CONFIRMATION (all via pandas merge — no loops):
    1. DM/HMS divergence           — primary signal
    2. Macro-pillar ETF flow_z     — high = TLT/GLD inflow = hedging regime
    3. short_z_score               — elevated = short pressure
    4. Macro vix/credit z          — low = cheap hedge environment
    5. No significant insider selling

DATA: All from Supabase. No external API calls.

REQUIREMENTS:
    pip install supabase pandas numpy scipy python-dotenv

SETUP: .env with SUPABASE_URL and SUPABASE_KEY

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
DM_DECLINE        = 10    # DM 5-day drop to qualify (DM is 0-100 scale)
HMS_STABLE        = 0.05  # |HMS 5-day change| < this = stable (HMS is 0-1 scale)
HMS_DECLINE       = 0.05  # HMS change <= -this = declining (HMS is 0-1 scale)
MACRO_FLOW_Z      =  0.5  # Macro pillar daily flow_z_score >= this = hedge-on regime
SHORT_Z           =  0.5  # short_z_score >= this = elevated short pressure
VIX_Z             = -0.5  # vix_z_score <= this = cheap options
CREDIT_Z          = -0.5  # credit_z_score <= this = tight spreads
FORWARD_WINDOWS   = [30, 60, 90]
DATE_FROM = "2022-01-01"
TIME_PERIODS = {
    "covid_tail": ("2022-01-01", "2022-12-31"),
    "ai_cycle":   ("2023-01-01", "2026-12-31"),
}

# ── SUPABASE ──────────────────────────────────────────────────────────────────

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


_ORDER_COLS = {
    "hms_daily":            ["date", "ticker"],
    "dm_history":           ["date", "ticker"],
    "etf_flows_history":    ["date", "ticker"],
    "short_volume":         ["date", "ticker"],
    "macro_history":        ["date"],
    "insider_transactions": ["transaction_date", "ticker"],
}


def pull_table(sb: Client, table: str, select: str = "*",
               date_col: str = None, date_from: str = None) -> pd.DataFrame:
    """Paginated pull with stable ordering + retries on transient errors.

    Optional server-side date filter: pass `date_col` and `date_from` to
    slice the table before pagination begins.
    """
    import time
    rows, offset, page = [], 0, 1000
    order_cols = _ORDER_COLS.get(table, [])

    while True:
        attempt, cur = 0, page
        while True:
            try:
                q = sb.table(table).select(select)
                if date_col and date_from:
                    q = q.gte(date_col, date_from)
                for col in order_cols:
                    q = q.order(col)
                r = q.range(offset, offset + cur - 1).execute()
                break
            except Exception as e:
                msg, etype = str(e), type(e).__name__
                attempt += 1
                transient = (
                    "57014" in msg or "502" in msg or "503" in msg or
                    "504" in msg or "timeout" in msg.lower() or
                    "Bad Gateway" in msg or "ConnectionTerminated" in msg or
                    "RemoteProtocolError" in etype or "ReadError" in etype or
                    "ConnectError" in etype or "ConnectTimeout" in etype or
                    "ReadTimeout" in etype
                )
                if attempt >= 6 or not transient:
                    raise
                cur = max(100, cur // 2)
                print(f"    [retry {attempt}] {etype}: page={cur}, "
                      f"sleep {2**attempt}s")
                time.sleep(2 ** attempt)

        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < cur:
            break
        offset += cur

    return pd.DataFrame(rows)


# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────

def load(sb: Client) -> dict:
    print("Loading from Supabase...\n")
    d = {}

    print("  hms_daily...")
    df = pull_table(sb, "hms_daily",
                    select="date,ticker,hms_score",
                    date_col="date", date_from=DATE_FROM)
    df["date"] = pd.to_datetime(df["date"])
    d["hms"] = df.sort_values(["ticker","date"]).reset_index(drop=True)
    print(f"    {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    print("  dm_history...")
    df = pull_table(sb, "dm_history",
                    select="date,ticker,dm_smoothed",
                    date_col="date", date_from=DATE_FROM)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"dm_smoothed": "dm_score"})
    d["dm"] = df.sort_values(["ticker","date"]).reset_index(drop=True)
    print(f"    {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    print("  etf_flows_history (Macro pillar for hedge signal)...")
    df = pull_table(sb, "etf_flows_history",
                    select="date,ticker,flow_z_score,flow_signal,pillar",
                    date_col="date", date_from=DATE_FROM)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["pillar"] == "Macro"]
    d["etf_flows"] = df
    print(f"    {len(df):,} Macro-pillar rows | ETFs: {sorted(df['ticker'].unique().tolist())}")

    print("  short_volume...")
    df = pull_table(sb, "short_volume",
                    select="date,ticker,short_z_score,short_ratio",
                    date_col="date", date_from=DATE_FROM)
    df["date"] = pd.to_datetime(df["date"])
    d["short_vol"] = df
    print(f"    {len(df):,} rows | {df['ticker'].nunique()} tickers")

    print("  macro_history...")
    df = pull_table(sb, "macro_history",
                    select="date,vix_close,vix_z_score,credit_z_score",
                    date_col="date", date_from=DATE_FROM)
    df["date"] = pd.to_datetime(df["date"])
    d["macro"] = df
    print(f"    {len(df):,} rows")

    print("  insider_transactions...")
    df = pull_table(sb, "insider_transactions",
                    select="transaction_date,ticker,is_significant,shares_changed",
                    date_col="transaction_date", date_from=DATE_FROM)
    df["date"] = pd.to_datetime(df["transaction_date"])
    d["insider"] = df
    print(f"    {len(df):,} rows | {df['ticker'].nunique()} tickers\n")

    return d


# ── STEP 2: DIVERGENCE EVENTS ─────────────────────────────────────────────────

def find_events(d: dict) -> pd.DataFrame:
    print("Finding DM/HMS divergence events (vectorized)...")

    hms = d["hms"].copy()
    dm  = d["dm"].copy()

    # 5-day rolling change — groupby transform
    hms["hms_chg5"] = hms.groupby("ticker")["hms_score"].transform(
        lambda x: x - x.shift(5))
    dm["dm_chg5"]   = dm.groupby("ticker")["dm_score"].transform(
        lambda x: x - x.shift(5))
    dm["dm_chg10"]  = dm.groupby("ticker")["dm_score"].transform(
        lambda x: x - x.shift(10))

    # Merge
    merged = pd.merge(
        dm[["date","ticker","dm_score","dm_chg5","dm_chg10"]],
        hms[["date","ticker","hms_score","hms_chg5"]],
        on=["date","ticker"], how="inner"
    )

    # Filter: meaningful DM decline
    ev = merged[merged["dm_chg5"] <= -DM_DECLINE].copy()
    print(f"  DM decline events: {len(ev):,} | tickers: {ev['ticker'].nunique()}")

    # Classify
    conditions = [
        ev["hms_chg5"] >  HMS_STABLE,
        ev["hms_chg5"].abs() <= HMS_STABLE,
        ev["hms_chg5"] <= -HMS_DECLINE,
    ]
    choices = ["WAVE_R_STRONG", "WAVE_R", "GENUINE_EXIT"]
    ev["classification"] = np.select(conditions, choices, default="AMBIGUOUS")

    # Drop unknowns from HMS merge gaps
    ev = ev.dropna(subset=["hms_chg5"])

    # Time period
    def period(dt):
        for name, (s, e) in TIME_PERIODS.items():
            if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
                return name
        return "other"
    ev["time_period"] = ev["date"].apply(period)

    print(f"\n  Classification breakdown:")
    for cls, cnt in ev["classification"].value_counts().items():
        print(f"    {cls:20}: {cnt:7,} ({cnt/len(ev)*100:.1f}%)")

    print(f"\n  By time period:")
    pt = ev.groupby(["time_period","classification"]).size().unstack(fill_value=0)
    print(pt.to_string())

    return ev


# ── STEP 3: SCORE FIVE FACTORS (VECTORIZED) ───────────────────────────────────

def score_events(ev: pd.DataFrame, d: dict) -> pd.DataFrame:
    print("\nScoring five confirmation factors (vectorized merge)...")

    scored = ev.copy()

    # ── Factor 2: Macro-pillar ETF flow (hedge regime) ────────────────────────
    # Aggregate flow_z_score across Macro-pillar ETFs (TLT, GLD) per date.
    # Positive z = elevated volume into safe-havens = market-wide hedging.
    macro_etf = (d["etf_flows"]
                 .groupby("date", as_index=False)
                 .agg(macro_flow_z=("flow_z_score", "max"),
                      macro_flow_signal=("flow_signal",
                                         lambda s: "Inflow" if "Inflow" in set(s) else
                                                   ("Outflow" if "Outflow" in set(s) else "Neutral"))))
    scored = scored.merge(macro_etf, on="date", how="left")
    scored["f2_wave_r"] = scored["macro_flow_z"] >= MACRO_FLOW_Z

    # ── Factor 3: Short Volume Z-Score ────────────────────────────────────────
    sv = d["short_vol"][["date","ticker","short_z_score","short_ratio"]].copy()
    sv = sv.rename(columns={"short_z_score":"short_z"})
    scored = scored.merge(sv, on=["date","ticker"], how="left")
    scored["f3_wave_r"] = scored["short_z"] >= SHORT_Z

    # ── Factor 4: Macro Regime ────────────────────────────────────────────────
    macro = d["macro"][["date","vix_z_score","credit_z_score","vix_close"]].copy()
    macro = macro.rename(columns={"vix_z_score":"vix_z",
                                  "credit_z_score":"credit_z"})
    scored = scored.merge(macro, on="date", how="left")
    scored["f4_wave_r"] = (
        (scored["vix_z"]    <= VIX_Z) &
        (scored["credit_z"] <= CREDIT_Z)
    )

    # ── Factor 5: Insider Selling ─────────────────────────────────────────────
    # Flag any significant insider sell within ±10 days of event.
    # Uses merge_asof (O(n log n)) instead of a ticker-only cartesian join,
    # which previously exploded to hundreds of millions of rows for popular tickers.
    ins = d["insider"].copy()
    ins = ins[
        (ins["is_significant"] == True) &
        (ins["shares_changed"] < 0)
    ][["date","ticker"]].copy()
    ins["has_sig_sell"] = True

    # merge_asof requires both sides sorted by the `on` key (date).
    # The `by="ticker"` partitioning is handled by pandas internally.
    ins_sorted = ins.rename(columns={"date":"insider_date"}).sort_values("insider_date")
    scored_sorted = scored[["date","ticker"]].sort_values("date")

    near_fwd = pd.merge_asof(
        scored_sorted,
        ins_sorted,
        left_on="date", right_on="insider_date",
        by="ticker",
        tolerance=pd.Timedelta(days=10),
        direction="nearest",
    )
    near_fwd["insider_sell_nearby"] = near_fwd["has_sig_sell"].fillna(False)

    scored = scored.merge(
        near_fwd[["date","ticker","insider_sell_nearby"]],
        on=["date","ticker"], how="left"
    )
    scored["insider_sell_nearby"] = scored["insider_sell_nearby"].fillna(False)
    scored["f5_wave_r"] = ~scored["insider_sell_nearby"]

    # ── Composite Wave R Score ────────────────────────────────────────────────
    factor_cols = ["f2_wave_r","f3_wave_r","f4_wave_r","f5_wave_r"]

    # Factor 1 from classification
    scored["f1_wave_r"] = scored["classification"].isin(["WAVE_R","WAVE_R_STRONG"])

    all_factors = ["f1_wave_r"] + factor_cols
    scored["factors_confirmed"] = scored[all_factors].sum(axis=1)
    scored["factors_available"] = scored[all_factors].notna().sum(axis=1)
    scored["wave_r_score"]      = scored["factors_confirmed"] / scored["factors_available"].clip(lower=1)
    scored["wave_r_confirmed"]  = scored["factors_confirmed"] >= 3

    print(f"\n  Scoring complete. Summary:")
    for cls in ["WAVE_R","WAVE_R_STRONG","GENUINE_EXIT"]:
        sub = scored[scored["classification"] == cls]
        if len(sub) > 0:
            conf = sub["wave_r_confirmed"].sum()
            print(f"  {cls:20}: {len(sub):6,} | "
                  f"multi-factor confirmed: {conf:5,} ({conf/len(sub)*100:.1f}%)")

    return scored


# ── STEP 4: FORWARD DM RECOVERY (VECTORIZED) ──────────────────────────────────

def measure_performance(scored: pd.DataFrame, d: dict) -> pd.DataFrame:
    print("\nMeasuring forward DM recovery (vectorized)...")

    dm = d["dm"][["date","ticker","dm_score"]].copy()

    result = scored[["date","ticker","classification",
                     "wave_r_confirmed","wave_r_score",
                     "factors_confirmed","dm_score","hms_score",
                     "time_period"]].copy()

    for window in FORWARD_WINDOWS:
        # Create a shifted lookup: for each row, find DM ~window trading days later
        # Approximate: window trading days ≈ window * 7/5 calendar days
        cal_days = int(window * 7 / 5)

        # Create forward DM table with offset date
        dm_fwd = dm.copy()
        dm_fwd["date_event"] = dm_fwd["date"] - pd.Timedelta(days=cal_days)
        dm_fwd = dm_fwd.rename(columns={
            "date": f"date_fwd_{window}d",
            "dm_score": f"dm_fwd_{window}d"
        })

        # Merge: match event date to forward date (within 3-day tolerance)
        # Use merge_asof for nearest-date matching
        result = result.sort_values("date")
        dm_fwd = dm_fwd.sort_values("date_event")

        merged_fwd = pd.merge_asof(
            result[["date","ticker"]].sort_values("date"),
            dm_fwd[["date_event","ticker",f"dm_fwd_{window}d"]].sort_values("date_event"),
            left_on="date",
            right_on="date_event",
            by="ticker",
            tolerance=pd.Timedelta(days=5),
            direction="nearest"
        )

        result = result.merge(
            merged_fwd[["date","ticker",f"dm_fwd_{window}d"]],
            on=["date","ticker"], how="left"
        )

        result[f"dm_recovery_{window}d"] = (
            result[f"dm_fwd_{window}d"] - result["dm_score"]
        ).round(2)

        n_valid = result[f"dm_recovery_{window}d"].notna().sum()
        print(f"  {window}d window: {n_valid:,} events with forward data")

    return result


# ── STEP 5: STATISTICS ────────────────────────────────────────────────────────

def analyze(perf: pd.DataFrame) -> dict:
    print("\n" + "═"*70)
    print("STATISTICAL ANALYSIS")
    print("═"*70)

    wr = perf[perf["classification"].isin(["WAVE_R","WAVE_R_STRONG"])]
    ge = perf[perf["classification"] == "GENUINE_EXIT"]

    print(f"\n  Wave R: {len(wr):,} | Genuine Exit: {len(ge):,}")

    if len(wr) < 20:
        print(f"  ⚠ Low Wave R count. Try lowering DM_DECLINE to 7.")

    results = {}

    for w in FORWARD_WINDOWS:
        col = f"dm_recovery_{w}d"
        wr_v = wr[col].dropna()
        ge_v = ge[col].dropna()

        if len(wr_v) < 10 or len(ge_v) < 10:
            print(f"\n  {w}d: insufficient data (wr={len(wr_v)}, ge={len(ge_v)})")
            continue

        t, p = stats.ttest_ind(wr_v, ge_v)
        effect = (wr_v.mean() - ge_v.mean()) / (pd.concat([wr_v,ge_v]).std()+1e-9)

        results[w] = {
            "wr_mean":   round(wr_v.mean(), 2),
            "wr_med":    round(wr_v.median(), 2),
            "wr_n":      len(wr_v),
            "ge_mean":   round(ge_v.mean(), 2),
            "ge_med":    round(ge_v.median(), 2),
            "ge_n":      len(ge_v),
            "t":         round(t, 3),
            "p":         round(p, 4),
            "effect":    round(effect, 3),
            "sig":       p < 0.05,
            "direction": wr_v.mean() > ge_v.mean(),
        }

        d = "✓" if results[w]["direction"] else "✗"
        s = "SIGNIFICANT" if results[w]["sig"] else "not sig"
        print(f"\n  {w}-day DM recovery:")
        print(f"    Wave R:  avg={results[w]['wr_mean']:+.2f} "
              f"med={results[w]['wr_med']:+.2f} n={len(wr_v):,}")
        print(f"    Genuine: avg={results[w]['ge_mean']:+.2f} "
              f"med={results[w]['ge_med']:+.2f} n={len(ge_v):,}")
        print(f"    p={p:.4f} | {s} | effect={effect:.3f} | {d} WR recovers more")

    # Multi-factor confirmed
    confirmed  = wr[wr["wave_r_confirmed"] == True]
    nconfirmed = wr[wr["wave_r_confirmed"] != True]
    print(f"\n  Multi-factor confirmed (3+ factors) vs single-factor:")
    for w in FORWARD_WINDOWS:
        col = f"dm_recovery_{w}d"
        cv  = confirmed[col].dropna()
        ncv = nconfirmed[col].dropna()
        gev = ge[col].dropna()
        if len(cv) > 5:
            print(f"    {w}d: Confirmed={cv.mean():+.2f} | "
                  f"Single={ncv.mean():+.2f} | Genuine={gev.mean():+.2f}")

    # By time period
    print(f"\n  By time period (30d recovery):")
    for period in TIME_PERIODS:
        wrp = wr[wr["time_period"]==period]["dm_recovery_30d"].dropna()
        gep = ge[ge["time_period"]==period]["dm_recovery_30d"].dropna()
        if len(wrp) > 5 and len(gep) > 5:
            print(f"    {period:15}: WR={wrp.mean():+.2f} "
                  f"GE={gep.mean():+.2f} diff={wrp.mean()-gep.mean():+.2f} "
                  f"(n_wr={len(wrp)}, n_ge={len(gep)})")

    return results


# ── STEP 6: SUMMARY ───────────────────────────────────────────────────────────

def write_summary(ev, scored, perf, stat_results):
    total = len(ev)
    wr_n  = len(ev[ev["classification"].isin(["WAVE_R","WAVE_R_STRONG"])])
    ge_n  = len(ev[ev["classification"] == "GENUINE_EXIT"])

    lines = [
        "═"*70,
        "WAVE R VALIDATION SUMMARY — Y2AI Research",
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "═"*70,
        "",
        "HYPOTHESIS: DM declining + HMS stable = Wave R (hedge, not exit)",
        "            Wave R candidates should recover DM vs genuine exits.",
        "",
        "DATA:",
        f"  Range:    {ev['date'].min().date()} → {ev['date'].max().date()}",
        f"  Tickers:  {ev['ticker'].nunique():,}",
        f"  Events:   {total:,} total | {wr_n:,} Wave R | {ge_n:,} Genuine",
        "",
        "FORWARD DM RECOVERY:",
    ]

    supported = significant = 0
    for w, r in stat_results.items():
        d_ = "✓" if r["direction"] else "✗"
        s  = "SIGNIFICANT" if r["sig"] else "not sig"
        if r["direction"]: supported += 1
        if r["sig"]: significant += 1
        lines.append(f"  {w}d: WR={r['wr_mean']:+.2f} vs GE={r['ge_mean']:+.2f} "
                    f"| p={r['p']:.4f} | {s} | {d_}")

    lines.append("")
    lines.append("VERDICT:")
    n = max(len(stat_results), 1)

    if supported == n and significant >= n//2:
        lines += [
            "  ✓✓ STRONG SUPPORT — Wave R confirmed.",
            "  NEXT: Add HMS/DM divergence flag to Step 2B morning workflow.",
            "  Require HMS confirmation before entering Hollowing Short",
            "  positions where DM declines but HMS is stable.",
        ]
    elif supported >= n//2:
        lines += [
            "  ~  PARTIAL SUPPORT — direction correct, not fully significant.",
            "  NEXT: Extend lookback, lower DM_DECLINE threshold to 7, rerun.",
        ]
    else:
        lines += [
            "  ✗  NOT SUPPORTED at current thresholds.",
            "  NEXT: Adjust thresholds and retry. Wave R is theoretical only.",
        ]

    lines += [
        "",
        "BY TIME PERIOD:",
    ]
    for period in TIME_PERIODS:
        sub  = ev[ev["time_period"]==period]
        wr_p = len(sub[sub["classification"].isin(["WAVE_R","WAVE_R_STRONG"])])
        ge_p = len(sub[sub["classification"]=="GENUINE_EXIT"])
        lines.append(f"  {period:15}: Wave R={wr_p:,} | Genuine={ge_p:,}")

    lines += [
        "",
        "NEXT STEPS:",
        "  1. Send summary to Vikram for framework decision",
        "  2. If supported: implement HMS/DM flag in Hollowing Short scanner",
        "  3. Validate high-score Wave R dates with Calypso colleague",
        "  4. Add Wave R to Great Hollowing book chapter",
        "  5. Rerun in 90 days with new data",
        "",
        "═"*70,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    with open("wave_r_summary.txt","w") as f:
        f.write(text)
    print("Saved: wave_r_summary.txt")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print("═"*70)
    print("WAVE R VALIDATOR v3.0 — Y2AI Research")
    print("Fully vectorized. Expected runtime: 2-5 minutes.")
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*70 + "\n")

    sb = get_supabase()
    d  = load(sb)

    ev = find_events(d)
    ev.to_csv("wave_r_events.csv", index=False)
    print(f"Saved: wave_r_events.csv ({len(ev):,} rows)")

    scored = score_events(ev, d)
    scored.to_csv("wave_r_scored.csv", index=False)
    print(f"Saved: wave_r_scored.csv ({len(scored):,} rows)")

    perf = measure_performance(scored, d)
    perf.to_csv("wave_r_performance.csv", index=False)
    print(f"Saved: wave_r_performance.csv ({len(perf):,} rows)")

    stat_results = analyze(perf)
    write_summary(ev, scored, perf, stat_results)

    elapsed = (datetime.now() - t0).seconds
    print(f"\nCompleted in {elapsed}s | {datetime.now().strftime('%H:%M:%S')}")
    print("═"*70)


if __name__ == "__main__":
    main()
