"""
T1C DEFENSE / DOD SECTOR SIMULATION
======================================
Runs production spec (Test 8) for the Defense / DOD sector cluster.

Production spec:
  - DM >= 65, 5-day MA rising
  - Exit: 20-day MA < 50
  - Sizing: DM-tiered (8/6.5/5/3.5%)
  - HMS gate: DISABLED
  - Period: 2016-2026
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFENSE_TICKERS = [
    'LMT', 'RTX', 'NOC', 'GD', 'BA', 'LHX', 'HII',
    'LDOS', 'BAH', 'SAIC', 'CACI', 'PLTR', 'AXON', 'KTOS', 'RCAT',
]

CONFIG = {
    'INITIAL_CAPITAL': 10_000_000,
    'ENTRY_DM': 65, 'EXIT_MA20': 50, 'HMS_MIN': None,
    'MAX_POSITIONS': 20, 'MIN_POSITION': 500,
    'DM_SIZING': True,
    'DM_TIERS': [
        (90, 0.080),
        (80, 0.065),
        (70, 0.050),
        (65, 0.035),
    ],
    'DM_TABLE': 'dm_history', 'HMS_TABLE': 'hms_daily',
    'DM_DATE_COL': 'date', 'DM_TICKER_COL': 'ticker',
    'DM_CLOSE_COL': 'close', 'DM_SCORE_COL': 'dm_smoothed',
    'HMS_DATE_COL': 'date', 'HMS_TICKER_COL': 'ticker',
    'HMS_SCORE_COL': 'hms_score',
    'START_YEAR': 2016, 'END_YEAR': 2026,
}

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())


def main():
    print("\nT1C DEFENSE / DOD SECTOR SIMULATION")
    print("Production Spec: DM>=65, 5d MA rising, Exit MA20<50, DM-tiered sizing")
    print(f"Universe: {', '.join(DEFENSE_TICKERS)}")
    print("Period: 2016-2026\n")

    try:
        client = get_client()
        print("Connected to Supabase.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    nav = CONFIG['INITIAL_CAPITAL']
    open_positions = {}
    year_results = []

    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        dm_df = load_dm(client, year)
        hms_df = load_hms(client, year)

        if dm_df.empty:
            continue

        dm_df = dm_df[dm_df['ticker'].isin(DEFENSE_TICKERS)].copy()
        if not hms_df.empty:
            hms_df = hms_df[hms_df['ticker'].isin(DEFENSE_TICKERS)].copy()

        if dm_df.empty:
            continue

        signals_df = compute_signals(dm_df, hms_df)

        try:
            result = run_year_v2(year, signals_df, nav, open_positions)
        except NameError:
            result = run_year(year, signals_df, nav, open_positions)

        if result is None:
            continue

        print_year(result)
        year_results.append(result)
        nav = result['final_nav']
        open_positions = result['open_positions']

    if year_results:
        print_summary(year_results, "SECTOR: Defense_DOD")

    print(f"\nReference: Full_Universe (28 tickers) = 19.3% CAGR, $69.7M final NAV")
    print("DONE.\n")


if __name__ == '__main__':
    main()
