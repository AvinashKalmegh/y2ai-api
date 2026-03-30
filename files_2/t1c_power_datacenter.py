"""
T1C POWER & DATA CENTER SECTOR SIMULATIONS
============================================
Runs production spec (Test 8) for two new sector clusters:
  1. Power / Energy Infrastructure
  2. Data Center / AI Infrastructure

Production spec:
  - DM >= 65, 5-day MA rising
  - Exit: 20-day MA < 50
  - Sizing: DM-tiered (8/6.5/5/3.5%)
  - HMS gate: DISABLED
  - Period: 2016-2026
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SECTORS = {
    'Power_Energy': [
        'CEG', 'VST', 'NRG', 'ETR', 'EXC', 'PCG',
        'NEE', 'AES', 'DNN', 'UEC', 'CCJ',
    ],
    'Data_Center_AI_Infra': [
        'VRT', 'EQIX', 'DLR', 'IRM', 'SMCI',
        'NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'QCOM',
    ],
}

BASE_CONFIG = {
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

# Load sim_core
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't1c_sim_core.py'), encoding='utf-8').read())


def run_sector(sector_name, tickers):
    """Run simulation for a single sector, filtering to only these tickers."""
    global CONFIG
    CONFIG = {**BASE_CONFIG}

    print(f"\n{'='*60}")
    print(f"  SECTOR: {sector_name} ({len(tickers)} tickers: {', '.join(tickers)})")
    print(f"{'='*60}")

    try:
        client = get_client()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    nav = CONFIG['INITIAL_CAPITAL']
    open_positions = {}
    year_results = []

    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        dm_df = load_dm(client, year)
        hms_df = load_hms(client, year)

        if dm_df.empty:
            continue

        # Filter to sector tickers only
        dm_df = dm_df[dm_df['ticker'].isin(tickers)].copy()
        if not hms_df.empty:
            hms_df = hms_df[hms_df['ticker'].isin(tickers)].copy()

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
        print_summary(year_results, f"SECTOR: {sector_name}")

    return year_results


def main():
    print("\nT1C POWER & DATA CENTER SECTOR SIMULATIONS")
    print("Production Spec: DM>=65, 5d MA rising, Exit MA20<50, DM-tiered sizing")
    print("Period: 2016-2026\n")

    all_results = {}
    for sector_name, tickers in SECTORS.items():
        results = run_sector(sector_name, tickers)
        if results:
            all_results[sector_name] = results

    # Summary
    print(f"\n{'='*80}")
    print("POWER & DATA CENTER SECTOR RESULTS")
    print(f"{'='*80}")
    print(f"{'Sector':<25} {'CAGR':>8} {'MaxDD':>8} {'Trades':>7} {'HitRate':>8} {'FinalNAV':>16}")
    print(f"{'-'*75}")

    for sector_name, results in all_results.items():
        if not results:
            continue
        first_nav = results[0]['starting_nav']
        last_nav = results[-1]['final_nav']
        years = len(results)
        cagr = ((last_nav / first_nav) ** (1.0 / years) - 1) * 100 if years > 0 else 0
        max_dd = min(r['year_return'] for r in results)
        total_trades = sum(r['n_trades'] for r in results)
        avg_hit = sum(r['hit_rate'] for r in results) / len(results) if results else 0
        print(f"{sector_name:<25} {cagr:>7.1f}% {max_dd:>7.1f}% {total_trades:>7} {avg_hit:>7.1f}% ${last_nav:>14,.0f}")

    print(f"{'='*80}")
    print("Reference: Full_Universe (28 tickers) = 19.3% CAGR, $69.7M final NAV")
    print("DONE.\n")


if __name__ == '__main__':
    main()
