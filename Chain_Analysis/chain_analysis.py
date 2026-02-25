"""
Chain Analysis — Full Universe Pipeline
All 5 Deliverables merged and runnable.
February 19, 2026

Deliverable 1: Full Universe Chain Map
Deliverable 2: Daily Chain Monitor
Deliverable 3: Propagation Timing Backtest
Deliverable 4: Strand Persistence Analysis
Deliverable 5: Strand Emergence Scanner (Bubble Radar)

Usage:
    python chain_analysis.py full           # Run all deliverables (1-5)
    python chain_analysis.py monitor        # Run daily monitor only (2)
    python chain_analysis.py backtest       # Run chain map + backtest (1+3)
    python chain_analysis.py emergence      # Run strand emergence scanner (5)

Requirements:
    pip install gspread google-auth pandas numpy scipy
"""

import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from collections import Counter
import time


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Credentials file — matches your existing pipeline setup
CREDENTIALS_FILE = 'credentials.json'

# Source spreadsheet (read DM data)
SOURCE_SPREADSHEET_ID = '1uozeMDJwQxj6dTjA_LG0kKx1U2AoSFfMI9MdA48uMMA'

# Output spreadsheet (write chain analysis results)
OUTPUT_SPREADSHEET_ID = '1nvuP8wC18Fza8C6cmVQj81akflOB-L3dzKtPH65mQ8Q'

# Sheet names
DM_HISTORY_TAB = 'DM_2024_2026'
DM_LATEST_TAB  = 'DM_Latest'

# Output sheet names
MONITOR_TAB    = 'Chain_Monitor'
BACKTEST_TAB   = 'Chain_Backtest_Results'
UNIVERSE_TAB   = 'Chain_Full_Universe'
STRAND_TAB     = 'Strand_Analysis'
EMERGENCE_TAB  = 'Strand_Emergence_All'
CLUSTER_TAB    = 'Strand_Emergence_Clusters'

# DM calculation constants
CORR_THRESHOLD = 0.75      # was 0.65 — tighter reduces spurious edges
CORR_WINDOW    = 60        # Days for correlation matrix
MAX_LAG        = 5         # Max lead/lag days to test
EXHAUST_ABOVE  = 70        # DM must be above this before exhaust
EXHAUST_BELOW  = 50        # DM must cross below this for exhaust
MIN_DAYS_ABOVE = 10        # Min days above threshold before exhaust counts
SATELLITE_STRESS = 40      # DM below this = satellite in stress

# Strand emergence constants
EMERGENCE_CURRENT_WINDOW = 30
EMERGENCE_PRIOR_START    = 60
EMERGENCE_PRIOR_END      = 30
EMERGENCE_THRESHOLD      = 0.80   # was 0.75 — require very strong new correlation
PRIOR_THRESHOLD          = 0.15   # was 0.20 — must have been truly uncorrelated before
CLUSTER_MIN_STRANDS      = 20     # was 5 — with 538 tickers, avg is ~9 strands per ticker

# Critical nodes for daily monitor
CRITICAL_NODES = {
    'TSLA': {'role': 'TOGGLE',          'alert_below': 35, 'alert_above': 65},
    'APP':  {'role': 'FIREWALL_BYPASS', 'alert_above': 85, 'alert_below': 40},
    'PLTR': {'role': 'MUTATION_ORIGIN', 'alert_below': 50, 'alert_above': None},
    'MSTR': {'role': 'MUTATION_ORIGIN', 'alert_below': 50, 'alert_above': None},
    'CEG':  {'role': 'TERMINAL_BASIN',  'alert_below': 30, 'alert_above': None},
    'NVDA': {'role': 'CORE_MASS',       'alert_below': 60, 'alert_above': None},
}


# ============================================================
# GOOGLE SHEETS CONNECTION
# ============================================================

_sheets_client = None

def get_sheets_client():
    global _sheets_client
    if _sheets_client is None:
        creds = Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=SCOPES
        )
        _sheets_client = gspread.authorize(creds)
    return _sheets_client


# ============================================================
# DATA LOADING
# ============================================================

def load_dm_history():
    """
    Load DM_2024_2026 into wide-format DataFrame.
    Returns: index=Date, columns=Ticker, values=DM
    """
    print("Loading DM history from Google Sheets...")
    client = get_sheets_client()
    sh = client.open_by_key(SOURCE_SPREADSHEET_ID)
    ws = sh.worksheet(DM_HISTORY_TAB)

    data = ws.get_all_values()
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    # Find correct column names
    # DM_2024_2026 has 18 columns: Date, Ticker, Close, Volume, ...
    # DM is column N (index 13), header may be 'DM' or 'dm_smoothed'
    ticker_col = None
    dm_col = None
    date_col = None
    close_col = None

    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('ticker',): ticker_col = col
        elif cl in ('dm', 'dm_smoothed', 'dm_raw'): dm_col = col
        elif cl in ('date',): date_col = col
        elif cl in ('close',): close_col = col

    # Fallback: use positional if headers don't match
    if not ticker_col: ticker_col = df.columns[1]
    if not dm_col: dm_col = df.columns[13]  # Column N
    if not date_col: date_col = df.columns[0]
    if not close_col: close_col = df.columns[2]

    print(f"  Using columns: Date={date_col}, Ticker={ticker_col}, DM={dm_col}")

    df[dm_col] = pd.to_numeric(df[dm_col], errors='coerce')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if close_col:
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
    df = df.dropna(subset=[date_col, dm_col, ticker_col])

    # Pivot to wide format
    dm_wide = df.pivot_table(
        index=date_col,
        columns=ticker_col,
        values=dm_col,
        aggfunc='last'
    )

    # Also build a close price wide table for backtest
    close_wide = None
    if close_col:
        close_wide = df.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=close_col,
            aggfunc='last'
        )

    print(f"  Loaded: {len(dm_wide)} trading days, {len(dm_wide.columns)} tickers")
    print(f"  Date range: {dm_wide.index.min().strftime('%Y-%m-%d')} to {dm_wide.index.max().strftime('%Y-%m-%d')}")
    return dm_wide, close_wide


def load_dm_latest():
    """
    Load DM_Latest sheet.
    Returns dict: {ticker: {'dm': float, 'close': float}}
    DM_Latest columns: Date(0), Ticker(1), DM(2), Phase(3), Lead(4),
    ETF_Flow(5), DM_7d_Ago(6), DM_Change(7), Volume_Z(8),
    RelStr_ETF(9), RelStr_SPY(10), Close(11), ETF(12)
    """
    print("Loading DM_Latest...")
    client = get_sheets_client()
    sh = client.open_by_key(SOURCE_SPREADSHEET_ID)
    ws = sh.worksheet(DM_LATEST_TAB)
    data = ws.get_all_values()

    result = {}
    for row in data[1:]:
        if len(row) < 12:
            continue
        ticker = row[1].strip()
        try:
            dm = float(row[2])
            close = float(row[11]) if row[11] else 0
            phase = row[3] if len(row) > 3 else ''
            if ticker:
                result[ticker] = {'dm': dm, 'close': close, 'phase': phase}
        except (ValueError, IndexError):
            continue

    print(f"  DM_Latest loaded: {len(result)} tickers")
    return result


# ============================================================
# WRITE RESULTS TO SHEETS
# ============================================================

def write_dataframe_to_sheet(df, tab_name, overwrite=True):
    """Write a pandas DataFrame to a Google Sheet tab. Handles large datasets."""
    if df.empty:
        print(f"  Skipping {tab_name} — empty DataFrame")
        return

    client = get_sheets_client()
    sh = client.open_by_key(OUTPUT_SPREADSHEET_ID)

    rows_needed = len(df) + 10
    cols_needed = max(len(df.columns) + 2, 20)

    try:
        ws = sh.worksheet(tab_name)
        if overwrite:
            ws.clear()
            # Resize sheet to fit the data
            if ws.row_count < rows_needed:
                ws.resize(rows=rows_needed, cols=cols_needed)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=max(rows_needed, 1000), cols=cols_needed)

    header = df.columns.tolist()
    rows = df.fillna('').astype(str).values.tolist()

    # Write in batches to avoid API limits
    all_data = [header] + rows
    batch_size = 500
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        start_row = i + 1
        end_row = start_row + len(batch) - 1
        end_col = chr(ord('A') + len(header) - 1) if len(header) <= 26 else 'Z'
        cell_range = f'A{start_row}:{end_col}{end_row}'
        ws.update(range_name=cell_range, values=batch, value_input_option='USER_ENTERED')
        if i + batch_size < len(all_data):
            time.sleep(2)  # Rate limit courtesy

    print(f"  Written {len(rows)} rows to {tab_name}")


def append_row_to_sheet(row_dict, tab_name):
    """Append a single row to a sheet (for daily monitor log)."""
    client = get_sheets_client()
    sh = client.open_by_key(OUTPUT_SPREADSHEET_ID)

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=20)
        ws.append_row(list(row_dict.keys()), value_input_option='USER_ENTERED')

    ws.append_row(list(row_dict.values()), value_input_option='USER_ENTERED')


# ============================================================
# DELIVERABLE 1 — FULL UNIVERSE CHAIN MAP
# ============================================================

def build_correlation_matrix(dm_wide, window=None):
    """
    Calculate Pearson correlation between all ticker pairs
    using the most recent `window` days.
    """
    if window is None:
        window = CORR_WINDOW

    print(f"Building correlation matrix (last {window} days)...")
    recent = dm_wide.tail(window)

    # Drop tickers with too many NaN values
    min_obs = int(window * 0.8)
    recent = recent.dropna(axis=1, thresh=min_obs)

    corr_matrix = recent.corr(method='pearson')
    print(f"  Correlation matrix: {corr_matrix.shape[0]} tickers")
    return corr_matrix


def get_lead_direction(series_a, series_b, max_lag=None):
    """
    Determine which ticker leads the other using lag correlation.
    Returns: (direction, lag_days, correlation)
    """
    if max_lag is None:
        max_lag = MAX_LAG

    best_lag = 0
    best_corr = 0
    direction = 'NONE'

    # Test A leads B (A's move predicts B shifted forward)
    for lag in range(1, max_lag + 1):
        corr = series_a.corr(series_b.shift(lag))
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
            direction = 'A_LEADS_B'

    # Test B leads A
    for lag in range(1, max_lag + 1):
        corr = series_b.corr(series_a.shift(lag))
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
            direction = 'B_LEADS_A'

    return direction, best_lag, best_corr


def classify_tickers(corr_matrix, dm_latest, dm_wide, corr_threshold=None):
    """
    Build directed graph from correlation matrix.
    Classify each ticker by role in the chain architecture.
    Returns: (classifications_df, edges_list)
    """
    if corr_threshold is None:
        corr_threshold = CORR_THRESHOLD

    print("Building directed graph and classifying tickers...")

    tickers = corr_matrix.columns.tolist()
    out_degree = {t: 0 for t in tickers}
    in_degree = {t: 0 for t in tickers}
    edges = []

    # Use last 120 days for lag calculation (performance optimization)
    dm_recent = dm_wide.tail(120)

    total_pairs = 0
    start_time = time.time()

    for i, ticker_a in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            pct = i / len(tickers) * 100
            print(f"  Processing ticker {i}/{len(tickers)} ({pct:.0f}%) "
                  f"[{elapsed:.0f}s elapsed, {len(edges)} edges found]")

        for ticker_b in tickers[i + 1:]:
            corr = corr_matrix.loc[ticker_a, ticker_b]
            if abs(corr) < corr_threshold:
                continue

            total_pairs += 1

            # Use recent data for lag computation (faster)
            if ticker_a not in dm_recent.columns or ticker_b not in dm_recent.columns:
                continue

            series_a = dm_recent[ticker_a].dropna()
            series_b = dm_recent[ticker_b].dropna()

            if len(series_a) < 30 or len(series_b) < 30:
                continue

            direction, lag, lead_corr = get_lead_direction(series_a, series_b)

            if direction == 'A_LEADS_B':
                out_degree[ticker_a] += 1
                in_degree[ticker_b] += 1
                edges.append((ticker_a, ticker_b, lag, lead_corr))
            elif direction == 'B_LEADS_A':
                out_degree[ticker_b] += 1
                in_degree[ticker_a] += 1
                edges.append((ticker_b, ticker_a, lag, lead_corr))

    elapsed = time.time() - start_time
    print(f"  Found {total_pairs} correlated pairs, {len(edges)} directed edges [{elapsed:.0f}s]")

    # Classify each ticker using percentile-based thresholds
    # With 17K+ edges, fixed thresholds are too loose
    out_values = sorted(out_degree.values(), reverse=True)
    in_values = sorted(in_degree.values(), reverse=True)
    n = len(out_values)

    # Top 5% out-degree = mutation origin candidates
    origin_threshold = out_values[max(0, int(n * 0.05) - 1)]
    # Top 5% in-degree with low out = firewall/basin candidates
    absorber_threshold = in_values[max(0, int(n * 0.10) - 1)]

    print(f"\n  Dynamic thresholds: Origin out_degree >= {origin_threshold}, "
          f"Absorber in_degree >= {absorber_threshold}")

    classifications = []
    for ticker in tickers:
        out_d = out_degree[ticker]
        in_d = in_degree[ticker]
        dm = dm_latest.get(ticker, {}).get('dm', 50)

        if out_d >= origin_threshold and dm > 60:
            classification = 'MUTATION_ORIGIN'
        elif in_d >= absorber_threshold and out_d <= max(5, origin_threshold // 10):
            classification = 'FIREWALL'
        elif in_d >= absorber_threshold // 2 and out_d <= 2:
            classification = 'TERMINAL_BASIN'
        elif out_d <= 2 and in_d <= 2:
            classification = 'ISOLATED'
        else:
            classification = 'SATELLITE'

        classifications.append({
            'Ticker':                ticker,
            'Classification':        classification,
            'Out_Degree':            out_d,
            'In_Degree':             in_d,
            'DM_Current':            round(dm, 2),
            'Phase':                 dm_latest.get(ticker, {}).get('phase', ''),
            'Scan_Date':             datetime.now().strftime('%Y-%m-%d'),
        })

    clf_df = pd.DataFrame(classifications)
    clf_df = clf_df.sort_values('Out_Degree', ascending=False)

    # Print summary
    print(f"\n  CLASSIFICATION SUMMARY:")
    for cls in ['MUTATION_ORIGIN', 'FIREWALL', 'TERMINAL_BASIN', 'SATELLITE', 'ISOLATED']:
        count = len(clf_df[clf_df['Classification'] == cls])
        print(f"    {cls}: {count}")

    return clf_df, edges


# ============================================================
# DELIVERABLE 2 — DAILY CHAIN MONITOR
# ============================================================

def run_daily_monitor(dm_latest=None):
    """
    Check critical nodes against DM_Latest. Append one row to Chain_Monitor.
    Can also be run standalone (loads dm_latest if not provided).
    """
    print("Running daily chain monitor...")

    if dm_latest is None:
        dm_latest = load_dm_latest()

    today = datetime.now().strftime('%Y-%m-%d')
    alerts = []
    row = {'Date': today}

    for ticker, config in CRITICAL_NODES.items():
        dm = dm_latest.get(ticker, {}).get('dm', None)
        row[f'{ticker}_DM'] = round(dm, 1) if dm is not None else 'N/A'

        if dm is None:
            continue

        if config.get('alert_below') and dm < config['alert_below']:
            msg = f"{ticker} ({config['role']}) DM={dm:.1f} BELOW {config['alert_below']}"
            alerts.append(msg)

        if config.get('alert_above') and dm > config['alert_above']:
            msg = f"{ticker} ({config['role']}) DM={dm:.1f} ABOVE {config['alert_above']}"
            alerts.append(msg)

    # Check for systemic: 2+ origins weakening simultaneously
    origin_alerts = [a for a in alerts if 'MUTATION_ORIGIN' in a and 'BELOW' in a]

    row['Alert_Count'] = len(alerts)
    row['Alerts_Text'] = ' | '.join(alerts) if alerts else 'NONE'
    row['Cascade_Risk'] = 'YES' if any('TSLA' in a and 'BELOW' in a for a in alerts) else 'NO'
    row['Systemic_Signal'] = 'YES' if len(origin_alerts) >= 2 else 'NO'

    # Write to sheet
    append_row_to_sheet(row, MONITOR_TAB)

    if alerts:
        print(f"  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    {a}")
    else:
        print("  No alerts today.")

    if row['Cascade_Risk'] == 'YES':
        print("  *** CASCADE_RISK — TSLA portfolio contagion pathway open ***")
    if row['Systemic_Signal'] == 'YES':
        print("  *** SYSTEMIC_SIGNAL — multiple origins weakening ***")

    return row


# ============================================================
# DELIVERABLE 3 — PROPAGATION TIMING BACKTEST
# ============================================================

def find_exhaust_events(dm_wide, ticker, above_threshold=None,
                        below_threshold=None, min_days_above=None):
    """
    Find dates where ticker DM crossed from sustained high to below threshold.
    Returns list of exhaust event dates.
    """
    if above_threshold is None: above_threshold = EXHAUST_ABOVE
    if below_threshold is None: below_threshold = EXHAUST_BELOW
    if min_days_above is None: min_days_above = MIN_DAYS_ABOVE

    series = dm_wide[ticker].dropna()
    events = []
    above_count = 0
    was_above = False

    for date, dm in series.items():
        if dm > above_threshold:
            above_count += 1
            was_above = True
        elif dm < below_threshold and was_above and above_count >= min_days_above:
            events.append(date)
            above_count = 0
            was_above = False
        else:
            if dm < above_threshold:
                was_above = False
                above_count = 0

    return events


def run_propagation_backtest(dm_wide, edges, dm_latest,
                              lookahead_days=None,
                              satellite_stress_threshold=None):
    """
    For each origin-satellite pair, find exhaust events and measure
    whether satellite DM drops below threshold within lookahead window.
    """
    if lookahead_days is None: lookahead_days = [10, 20, 30]
    if satellite_stress_threshold is None: satellite_stress_threshold = SATELLITE_STRESS

    print("Running propagation timing backtest...")
    results = []

    # Get unique origins (tickers that lead others)
    origins = list(set([e[0] for e in edges]))
    print(f"  Testing {len(origins)} origins against their satellites...")

    for idx, origin in enumerate(origins):
        if idx % 20 == 0 and idx > 0:
            print(f"  Processing origin {idx}/{len(origins)}...")

        # Get satellites (direct followers of this origin)
        satellites = [(follower, lag, corr)
                      for leader, follower, lag, corr in edges
                      if leader == origin]

        if not satellites or origin not in dm_wide.columns:
            continue

        # Find exhaust events for this origin
        exhaust_events = find_exhaust_events(dm_wide, origin)

        if len(exhaust_events) < 2:
            continue  # too few events

        for satellite, lag_days, corr in satellites:
            if satellite not in dm_wide.columns:
                continue

            sat_series = dm_wide[satellite]
            hit_counts = {d: 0 for d in lookahead_days}
            total_events = len(exhaust_events)
            lead_times = []
            magnitudes = []

            for event_date in exhaust_events:
                sat_dm_at_event = sat_series.get(event_date, None)
                if sat_dm_at_event is None or pd.isna(sat_dm_at_event):
                    total_events -= 1
                    continue

                for days in lookahead_days:
                    end_date = event_date + timedelta(days=days)
                    window = sat_series[event_date:end_date]
                    if any(window < satellite_stress_threshold):
                        hit_counts[days] += 1

                        if days == max(lookahead_days):
                            stress_dates = window[window < satellite_stress_threshold]
                            if len(stress_dates) > 0:
                                lead = (stress_dates.index[0] - event_date).days
                                lead_times.append(lead)
                                magnitude = sat_dm_at_event - stress_dates.iloc[0]
                                magnitudes.append(magnitude)

            if total_events == 0:
                continue

            hit_rates = {d: hit_counts[d] / total_events for d in lookahead_days}
            avg_lead = np.mean(lead_times) if lead_times else None
            avg_magnitude = np.mean(magnitudes) if magnitudes else None
            tradeable = hit_rates[max(lookahead_days)] > 0.65 and (avg_lead or 0) > 5

            result_row = {
                'Origin':              origin,
                'Satellite':           satellite,
                'Lag_Days':            lag_days,
                'Correlation':         round(corr, 3),
                'Exhaust_Events':      total_events,
            }

            for d in lookahead_days:
                result_row[f'Hit_Rate_{d}d'] = round(hit_rates[d], 3)

            result_row['Avg_Lead_Days'] = round(avg_lead, 1) if avg_lead else None
            result_row['Avg_Magnitude'] = round(avg_magnitude, 1) if avg_magnitude else None
            result_row['False_Positive_Rate'] = round(1 - hit_rates[max(lookahead_days)], 3)
            result_row['Tradeable'] = 'YES' if tradeable else 'NO'
            result_row['Origin_DM'] = round(dm_latest.get(origin, {}).get('dm', 0), 1)
            result_row['Satellite_DM'] = round(dm_latest.get(satellite, {}).get('dm', 0), 1)
            result_row['Scan_Date'] = datetime.now().strftime('%Y-%m-%d')

            results.append(result_row)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(f'Hit_Rate_{max(lookahead_days)}d', ascending=False)

    tradeable_count = len(results_df[results_df['Tradeable'] == 'YES']) if not results_df.empty else 0
    print(f"  Backtest complete: {len(results_df)} pairs evaluated, {tradeable_count} tradeable")
    return results_df


# ============================================================
# DELIVERABLE 4 — STRAND PERSISTENCE ANALYSIS
# ============================================================

def run_strand_persistence(dm_wide, edges, dm_latest,
                            cycle_above=70, cycle_below=40,
                            min_strand_strength=0.70, min_cycles=3):
    """
    Identify strands — propagation relationships that persist across
    multiple DM cycles. More reliable than one-off chain correlations.

    A cycle = one complete BUILD to EXHAUST sequence
    (DM above cycle_above then below cycle_below).

    Strand strength = consistent cycles / total cycles observed.
    """
    print("Running strand persistence analysis...")

    results = []

    for leader, follower, lag, corr in edges:
        if leader not in dm_wide.columns or follower not in dm_wide.columns:
            continue

        leader_series = dm_wide[leader].dropna()
        follower_series = dm_wide[follower].dropna()

        # Find all complete cycles for the leader
        cycles = []
        in_cycle = False
        cycle_start = None
        above_count = 0

        for date, dm in leader_series.items():
            if not in_cycle and dm > cycle_above:
                in_cycle = True
                cycle_start = date
                above_count = 1
            elif in_cycle and dm > cycle_above:
                above_count += 1
            elif in_cycle and dm < cycle_below and above_count >= 5:
                cycles.append({'start': cycle_start, 'end': date})
                in_cycle = False
                above_count = 0
            elif in_cycle and dm < cycle_above:
                # Dropped below high threshold but not yet below low
                pass

        if len(cycles) < 2:
            continue

        # For each cycle, check if the same propagation direction holds
        consistent = 0
        total_checked = 0

        for cycle in cycles:
            # Get data during this cycle
            cycle_leader = leader_series[cycle['start']:cycle['end']]
            cycle_follower = follower_series[cycle['start']:cycle['end']]

            # Need enough overlap
            common_dates = cycle_leader.index.intersection(cycle_follower.index)
            if len(common_dates) < 10:
                continue

            total_checked += 1

            # Test if leader still leads follower in this cycle
            cl = cycle_leader.reindex(common_dates)
            cf = cycle_follower.reindex(common_dates)

            direction, _, _ = get_lead_direction(cl, cf, max_lag=MAX_LAG)

            if direction == 'A_LEADS_B':
                consistent += 1

        if total_checked < min_cycles:
            continue

        strength = consistent / total_checked

        if strength >= min_strand_strength:
            results.append({
                'Ticker_A':          leader,
                'Ticker_B':          follower,
                'Direction':         f'{leader} -> {follower}',
                'Strand_Strength':   round(strength, 3),
                'Cycles_Observed':   total_checked,
                'Consistent_Cycles': consistent,
                'Avg_Lag_Days':      lag,
                'Correlation':       round(corr, 3),
                'DM_A':              round(dm_latest.get(leader, {}).get('dm', 0), 1),
                'DM_B':              round(dm_latest.get(follower, {}).get('dm', 0), 1),
                'Scan_Date':         datetime.now().strftime('%Y-%m-%d'),
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Strand_Strength', ascending=False)

    print(f"  Strand analysis complete: {len(results_df)} persistent strands found "
          f"(strength >= {min_strand_strength}, cycles >= {min_cycles})")
    return results_df


# ============================================================
# DELIVERABLE 5 — STRAND EMERGENCE SCANNER (Bubble Radar)
# ============================================================

def run_strand_emergence_scanner(dm_wide, dm_latest,
                                  current_window=None,
                                  prior_window_start=None,
                                  prior_window_end=None,
                                  emergence_threshold=None,
                                  prior_threshold=None,
                                  cluster_min_strands=None):
    """
    Detect new strand formations — earliest signal of bubble emergence.
    Scans for ticker pairs that were uncorrelated 30-60 days ago
    but are now correlated in the last 30 days.
    """
    if current_window is None: current_window = EMERGENCE_CURRENT_WINDOW
    if prior_window_start is None: prior_window_start = EMERGENCE_PRIOR_START
    if prior_window_end is None: prior_window_end = EMERGENCE_PRIOR_END
    if emergence_threshold is None: emergence_threshold = EMERGENCE_THRESHOLD
    if prior_threshold is None: prior_threshold = PRIOR_THRESHOLD
    if cluster_min_strands is None: cluster_min_strands = CLUSTER_MIN_STRANDS

    print("Running strand emergence scanner...")
    print(f"  Current window:  last {current_window} days")
    print(f"  Prior window:    {prior_window_start}-{prior_window_end} days ago")

    total_days = len(dm_wide)

    # Slice the two time windows
    current_data = dm_wide.tail(current_window)
    prior_data = dm_wide.iloc[
        max(0, total_days - prior_window_start):
        max(0, total_days - prior_window_end)
    ]

    # Drop tickers with insufficient data in either window
    min_observations = int(current_window * 0.8)
    valid_current = current_data.dropna(axis=1, thresh=min_observations).columns
    valid_prior = prior_data.dropna(axis=1, thresh=min_observations).columns
    valid_tickers = sorted(set(valid_current) & set(valid_prior))

    print(f"  Valid tickers: {len(valid_tickers)}")

    current_corr = current_data[valid_tickers].corr()
    prior_corr = prior_data[valid_tickers].corr()

    # Find newly forming strands
    new_strands = []
    for i, ticker_a in enumerate(valid_tickers):
        for ticker_b in valid_tickers[i + 1:]:
            try:
                c_corr = current_corr.loc[ticker_a, ticker_b]
                p_corr = prior_corr.loc[ticker_a, ticker_b]
            except KeyError:
                continue

            if np.isnan(c_corr) or np.isnan(p_corr):
                continue

            # New strand: was uncorrelated, now coordinating
            if abs(c_corr) >= emergence_threshold and abs(p_corr) < prior_threshold:
                dm_a = dm_latest.get(ticker_a, {}).get('dm', None)
                dm_b = dm_latest.get(ticker_b, {}).get('dm', None)

                new_strands.append({
                    'Ticker_A':     ticker_a,
                    'Ticker_B':     ticker_b,
                    'Current_Corr': round(c_corr, 3),
                    'Prior_Corr':   round(p_corr, 3),
                    'Corr_Change':  round(c_corr - p_corr, 3),
                    'DM_A':         round(dm_a, 1) if dm_a else None,
                    'DM_B':         round(dm_b, 1) if dm_b else None,
                    'Avg_DM':       round((dm_a + dm_b) / 2, 1) if dm_a and dm_b else None,
                    'Scan_Date':    datetime.now().strftime('%Y-%m-%d'),
                })

    print(f"  New strands found: {len(new_strands)}")

    if not new_strands:
        print("  No new strand formations detected.")
        return pd.DataFrame(), pd.DataFrame()

    new_strands_df = pd.DataFrame(new_strands)
    new_strands_df = new_strands_df.sort_values('Corr_Change', ascending=False)

    # Group into clusters
    all_tickers_in_strands = (
        list(new_strands_df['Ticker_A']) +
        list(new_strands_df['Ticker_B'])
    )
    ticker_counts = Counter(all_tickers_in_strands)

    cluster_tickers = {t: count for t, count in ticker_counts.items()
                       if count >= cluster_min_strands}

    clusters = []
    for ticker, strand_count in sorted(cluster_tickers.items(),
                                        key=lambda x: x[1], reverse=True):
        dm = dm_latest.get(ticker, {}).get('dm', None)
        partners = list(
            new_strands_df[new_strands_df['Ticker_A'] == ticker]['Ticker_B']
        ) + list(
            new_strands_df[new_strands_df['Ticker_B'] == ticker]['Ticker_A']
        )

        clusters.append({
            'Ticker':      ticker,
            'New_Strands': strand_count,
            'Partners':    ', '.join(sorted(set(partners))[:8]),
            'DM_Current':  round(dm, 1) if dm else None,
            'Signal':      'BUBBLE_EMERGING' if strand_count >= 5 else 'WATCH',
            'Scan_Date':   datetime.now().strftime('%Y-%m-%d'),
        })

    clusters_df = pd.DataFrame(clusters)

    # Print summary
    if not clusters_df.empty:
        print(f"\n  STRAND EMERGENCE CLUSTERS ({cluster_min_strands}+ new strands):")
        print(f"  {'TICKER':<8} {'STRANDS':<10} {'DM':<8} {'SIGNAL':<18} {'PARTNERS'}")
        print(f"  {'-' * 70}")
        for _, row in clusters_df.iterrows():
            print(f"  {row['Ticker']:<8} {row['New_Strands']:<10} "
                  f"{str(row['DM_Current']):<8} {row['Signal']:<18} "
                  f"{str(row['Partners'])[:40]}")
    else:
        print("  No clusters detected (no ticker in 3+ new strands).")

    return new_strands_df, clusters_df


# ============================================================
# MAIN — RUN MODES
# ============================================================

def run_full_analysis():
    """Run all five deliverables in sequence."""
    print("=" * 60)
    print("CHAIN ANALYSIS — FULL RUN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load data
    dm_wide, close_wide = load_dm_history()
    dm_latest = load_dm_latest()

    # Deliverable 1 — Full Universe Chain Map
    print("\n" + "=" * 60)
    print("[DELIVERABLE 1] Full Universe Chain Map")
    print("=" * 60)
    corr_matrix = build_correlation_matrix(dm_wide)
    classifications, edges = classify_tickers(corr_matrix, dm_latest, dm_wide)
    write_dataframe_to_sheet(classifications, UNIVERSE_TAB)

    # Deliverable 2 — Daily Chain Monitor
    print("\n" + "=" * 60)
    print("[DELIVERABLE 2] Daily Chain Monitor")
    print("=" * 60)
    run_daily_monitor(dm_latest)

    # Deliverable 3 — Propagation Timing Backtest
    print("\n" + "=" * 60)
    print("[DELIVERABLE 3] Propagation Timing Backtest")
    print("=" * 60)
    backtest_results = run_propagation_backtest(dm_wide, edges, dm_latest)
    write_dataframe_to_sheet(backtest_results, BACKTEST_TAB)

    tradeable = backtest_results[backtest_results['Tradeable'] == 'YES'] if not backtest_results.empty else pd.DataFrame()
    print(f"\n  Tradeable signals: {len(tradeable)}")
    if not tradeable.empty:
        print(tradeable[['Origin', 'Satellite', 'Hit_Rate_30d', 'Avg_Lead_Days']].head(10).to_string(index=False))

    # Deliverable 4 — Strand Persistence Analysis
    print("\n" + "=" * 60)
    print("[DELIVERABLE 4] Strand Persistence Analysis")
    print("=" * 60)
    strand_results = run_strand_persistence(dm_wide, edges, dm_latest)
    write_dataframe_to_sheet(strand_results, STRAND_TAB)

    # Deliverable 5 — Strand Emergence Scanner
    print("\n" + "=" * 60)
    print("[DELIVERABLE 5] Strand Emergence Scanner")
    print("=" * 60)
    new_strands, clusters = run_strand_emergence_scanner(dm_wide, dm_latest)
    if not new_strands.empty:
        write_dataframe_to_sheet(new_strands, EMERGENCE_TAB)
    if not clusters.empty:
        write_dataframe_to_sheet(clusters, CLUSTER_TAB)

    # Final summary
    print("\n" + "=" * 60)
    print("CHAIN ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Chain Map:     {len(classifications)} tickers classified, {len(edges)} edges")
    print(f"  Backtest:      {len(backtest_results)} pairs, {len(tradeable)} tradeable")
    print(f"  Strands:       {len(strand_results)} persistent strands")
    print(f"  Emergence:     {len(new_strands)} new strands, {len(clusters)} clusters")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def run_monitor_only():
    """Run just the daily monitor — fast, use every morning."""
    print("=" * 60)
    print("CHAIN MONITOR — DAILY CHECK")
    print("=" * 60)
    run_daily_monitor()


def run_backtest_only():
    """Run chain map + backtest without monitor or emergence scan."""
    print("=" * 60)
    print("CHAIN ANALYSIS — MAP + BACKTEST")
    print("=" * 60)

    dm_wide, close_wide = load_dm_history()
    dm_latest = load_dm_latest()

    print("\n[STEP 1] Building Chain Map...")
    corr_matrix = build_correlation_matrix(dm_wide)
    classifications, edges = classify_tickers(corr_matrix, dm_latest, dm_wide)
    write_dataframe_to_sheet(classifications, UNIVERSE_TAB)

    print("\n[STEP 2] Running Backtest...")
    backtest_results = run_propagation_backtest(dm_wide, edges, dm_latest)
    write_dataframe_to_sheet(backtest_results, BACKTEST_TAB)

    tradeable = backtest_results[backtest_results['Tradeable'] == 'YES'] if not backtest_results.empty else pd.DataFrame()
    print(f"\n  Tradeable signals: {len(tradeable)}")
    if not tradeable.empty:
        print(tradeable[['Origin', 'Satellite', 'Hit_Rate_30d', 'Avg_Lead_Days']].head(10).to_string(index=False))


def run_emergence_only():
    """Run just the strand emergence scanner."""
    print("=" * 60)
    print("STRAND EMERGENCE SCANNER")
    print("=" * 60)

    dm_wide, _ = load_dm_history()
    dm_latest = load_dm_latest()

    new_strands, clusters = run_strand_emergence_scanner(dm_wide, dm_latest)
    if not new_strands.empty:
        write_dataframe_to_sheet(new_strands, EMERGENCE_TAB)
    if not clusters.empty:
        write_dataframe_to_sheet(clusters, CLUSTER_TAB)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chain Analysis Pipeline')
    parser.add_argument('command', nargs='?', default='full',
                        choices=['full', 'monitor', 'backtest', 'emergence'],
                        help='Which analysis to run (default: full)')

    args = parser.parse_args()

    if args.command == 'full':
        run_full_analysis()
    elif args.command == 'monitor':
        run_monitor_only()
    elif args.command == 'backtest':
        run_backtest_only()
    elif args.command == 'emergence':
        run_emergence_only()