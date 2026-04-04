"""
STRATEGY 5 HIT RATE ANALYSIS — Python conversion of runPureSimHitRateAnalysis.gs
==================================================================================
Applies Strategy 5 entry/exit conditions to DM history CSV files
and calculates per-ticker signal counts, hit rates, and avg returns.

Strategy 5 Rules:
  Entry: DM (EMA5) >= 65 AND 5d MA of DM is rising
  Exit:  20d MA of DM drops below 50
  Hit:   Position achieves 10%+ return before exit

Uses CSV files from files_5 folder: DM_2016_2019.csv, DM_2020_2023.csv, DM_2024_2026.csv
"""
import pandas as pd
import os

# ── CSV sources ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_5    = os.path.join(SCRIPT_DIR, '..', 'files_5')

DM_FILES = [
    (os.path.join(FILES_5, 'DM_2016_2019.csv'), '2016-2019'),
    (os.path.join(FILES_5, 'DM_2020_2023.csv'), '2020-2023'),
    (os.path.join(FILES_5, 'DM_2024_2026.csv'), '2024-2026'),
]

# ── Tickers to evaluate ─────────────────────────────────────────────────────
PURESIM_8 = ['AMD', 'ARM', 'DLR', 'EQIX', 'INTU', 'KLAC', 'MRVL', 'VRT']
COMPARISON = ['APP', 'META', 'PLTR', 'VST', 'TEAM', 'CRWD', 'NVDA', 'TSLA',
              'DNN', 'CEG', 'MU', 'UUUU', 'LEU']
TICKERS = PURESIM_8 + COMPARISON

# ── Strategy 5 parameters ───────────────────────────────────────────────────
ENTRY_DM_MIN = 65
EXIT_MA20    = 50
HIT_RETURN   = 0.10   # 10%
FORWARD_DAYS = 90


def find_col(columns, candidates):
    cols_upper = [c.upper() for c in columns]
    for name in candidates:
        if name.upper() in cols_upper:
            return columns[cols_upper.index(name.upper())]
    return None


def load_data():
    """Load all DM CSVs and extract ticker data."""
    all_rows = {tk: [] for tk in TICKERS}
    ticker_set = set(TICKERS)

    for filepath, label in DM_FILES:
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        print(f"  Loading {label}...")
        df = pd.read_csv(filepath, low_memory=False)

        date_col   = find_col(df.columns, ['Date', 'DATE', 'Trade_Date'])
        ticker_col = find_col(df.columns, ['Ticker', 'TICKER', 'Symbol'])
        close_col  = find_col(df.columns, ['Close', 'CLOSE'])
        dm_col     = find_col(df.columns, ['DM', 'DM_Score', 'DM_Smoothed', 'dm_smoothed'])

        if not all([date_col, ticker_col, close_col, dm_col]):
            print(f"    Column detection failed. Available: {df.columns.tolist()[:10]}")
            continue

        df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
        df = df[df[ticker_col].isin(ticker_set)]
        df[date_col]  = pd.to_datetime(df[date_col], errors='coerce')
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df[dm_col]    = pd.to_numeric(df[dm_col], errors='coerce')
        df = df.dropna(subset=[date_col, close_col, dm_col])
        df = df[df[close_col] > 0]
        df = df[df[dm_col] > 0]

        for _, row in df.iterrows():
            tk = row[ticker_col]
            if tk in all_rows:
                all_rows[tk].append({
                    'date':  row[date_col].strftime('%Y-%m-%d'),
                    'close': float(row[close_col]),
                    'dm':    float(row[dm_col]),
                })

        print(f"    Loaded from {label}")

    # Sort each ticker by date
    for tk in TICKERS:
        all_rows[tk].sort(key=lambda x: x['date'])

    return all_rows


def analyze_ticker(ticker, rows):
    """Compute signals and hit rate for one ticker."""
    n = len(rows)

    if n < 25:
        return {
            'ticker': ticker, 'dataPoints': n, 'signals': 0, 'hits': 0,
            'hitRate': 0, 'avgReturn': 0, 'bestReturn': 0, 'worstReturn': 0,
            'firstDate': 'N/A', 'lastDate': 'N/A', 'verdict': 'INSUFFICIENT DATA'
        }

    # Compute rolling MAs
    dm5  = [0.0] * n
    dm20 = [0.0] * n

    for i in range(n):
        # 5d MA
        start5 = max(0, i - 4)
        dm5[i] = sum(rows[j]['dm'] for j in range(start5, i + 1)) / (i - start5 + 1)

        # 20d MA
        start20 = max(0, i - 19)
        dm20[i] = sum(rows[j]['dm'] for j in range(start20, i + 1)) / (i - start20 + 1)

    # Find entry signals
    signals = 0
    hits = 0
    total_return = 0.0
    best_return = -999.0
    worst_return = 999.0

    i = 1
    while i < n:
        dm = rows[i]['dm']
        ma5 = dm5[i]
        ma5_prev = dm5[i - 1]

        # Entry conditions
        if dm < ENTRY_DM_MIN or ma5 <= ma5_prev or dm20[i] < EXIT_MA20:
            i += 1
            continue

        signals += 1
        entry_close = rows[i]['close']

        # Find exit
        exit_return = 0.0
        for j in range(i + 1, min(i + FORWARD_DAYS + 1, n)):
            exit_return = (rows[j]['close'] - entry_close) / entry_close

            if dm20[j] < EXIT_MA20:
                break
            if j == min(i + FORWARD_DAYS, n - 1):
                break

        total_return += exit_return
        if exit_return >= HIT_RETURN:
            hits += 1
        if exit_return > best_return:
            best_return = exit_return
        if exit_return < worst_return:
            worst_return = exit_return

        # Skip 20 days to avoid overlapping signals
        i += 20
        continue

    hit_rate = round(hits / signals * 100) if signals > 0 else 0
    avg_return = round(total_return / signals * 100, 1) if signals > 0 else 0

    if signals == 0:
        verdict = 'NO SIGNALS'
    elif signals < 3:
        verdict = 'THIN (<3 signals)'
    elif hit_rate >= 65:
        verdict = 'STRONG -- KEEP'
    elif hit_rate >= 50:
        verdict = 'QUALIFIED -- KEEP'
    elif hit_rate >= 40:
        verdict = 'BORDERLINE -- REVIEW'
    else:
        verdict = 'BELOW THRESHOLD -- REMOVE'

    return {
        'ticker':      ticker,
        'dataPoints':  n,
        'signals':     signals,
        'hits':        hits,
        'hitRate':     hit_rate,
        'avgReturn':   avg_return,
        'bestReturn':  round(best_return * 100, 1) if best_return > -999 else 0,
        'worstReturn': round(worst_return * 100, 1) if worst_return < 999 else 0,
        'firstDate':   rows[0]['date'],
        'lastDate':    rows[-1]['date'],
        'verdict':     verdict,
    }


def main():
    sep = '=' * 65
    print(f"\n{sep}")
    print("STRATEGY 5 HIT RATE ANALYSIS -- GS DM HISTORY")
    print(f"Entry: DM >= {ENTRY_DM_MIN} + 5d MA rising")
    print(f"Exit:  20d MA < {EXIT_MA20}")
    print(f"Hit:   {HIT_RETURN*100:.0f}% return within {FORWARD_DAYS} days")
    print(sep)

    print("\nLoading data...")
    all_rows = load_data()

    # Analyze all tickers
    results = []
    for ticker in TICKERS:
        results.append(analyze_ticker(ticker, all_rows[ticker]))

    # ── Print PureSim 8 ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("RESULTS -- PURESIM 8")
    print(sep)
    print(f"{'Ticker':<7}| {'DtPts':>5} | {'Signals':>7} | {'Hits':>4} | {'Hit%':>5} | {'AvgRet':>7} | {'Best':>6} | {'Worst':>6} | Verdict")
    print(f"{'------':<7}|{'------':>6} |{'--------':>8} |{'-----':>5} |{'------':>6} |{'--------':>8} |{'-------':>7} |{'-------':>7} |--------")

    for r in results:
        if r['ticker'] in PURESIM_8:
            print(f"{r['ticker']:<7}| {r['dataPoints']:>5} | {r['signals']:>7} | {r['hits']:>4} | {r['hitRate']:>4}% | {r['avgReturn']:>6.1f}% | {r['bestReturn']:>5.1f}% | {r['worstReturn']:>5.1f}% | {r['verdict']}")

    # ── Print Comparison ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("RESULTS -- COMPARISON TICKERS")
    print(sep)
    print(f"{'Ticker':<7}| {'DtPts':>5} | {'Signals':>7} | {'Hits':>4} | {'Hit%':>5} | {'AvgRet':>7} | Verdict")
    print(f"{'------':<7}|{'------':>6} |{'--------':>8} |{'-----':>5} |{'------':>6} |{'--------':>8} |--------")

    for r in results:
        if r['ticker'] in COMPARISON:
            print(f"{r['ticker']:<7}| {r['dataPoints']:>5} | {r['signals']:>7} | {r['hits']:>4} | {r['hitRate']:>4}% | {r['avgReturn']:>6.1f}% | {r['verdict']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY -- PURESIM 8")
    print(sep)

    ps8_results = [r for r in results if r['ticker'] in PURESIM_8]
    strong     = [r['ticker'] for r in ps8_results if r['hitRate'] >= 65 and r['signals'] >= 3]
    qualified  = [r['ticker'] for r in ps8_results if 50 <= r['hitRate'] < 65 and r['signals'] >= 3]
    borderline = [r['ticker'] for r in ps8_results if 40 <= r['hitRate'] < 50 and r['signals'] >= 3]
    below      = [r['ticker'] for r in ps8_results if r['hitRate'] < 40 and r['signals'] >= 3]
    thin       = [r['ticker'] for r in ps8_results if r['signals'] < 3]

    print(f"STRONG (65%+, 3+ signals):      {', '.join(strong) or 'None'}")
    print(f"QUALIFIED (50-65%, 3+ signals):  {', '.join(qualified) or 'None'}")
    print(f"BORDERLINE (40-50%):             {', '.join(borderline) or 'None'}")
    print(f"BELOW THRESHOLD (<40%):          {', '.join(below) or 'None'}")
    print(f"THIN (<3 signals):               {', '.join(thin) or 'None'}")
    print(sep)

    # ── Save to CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    outfile = os.path.join(SCRIPT_DIR, 'puresim_hit_rate_results.csv')
    df.to_csv(outfile, index=False)
    print(f"\nSaved: {outfile}")


if __name__ == '__main__':
    main()
