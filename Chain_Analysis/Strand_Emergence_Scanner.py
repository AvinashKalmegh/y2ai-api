# DELIVERABLE 5 — STRAND EMERGENCE SCANNER (New Bubble Radar)
# Addition to Chain_Analysis_Developer_Brief.md
# February 19, 2026

# ============================================================
# WHAT THIS DOES — PLAIN ENGLISH
# ============================================================

# When a new bubble forms, the sequence is always:
#   1. PE activity and narrative appear (news, announcements)
#   2. Institutional capital starts moving quietly (dark matter rises)
#   3. Tickers in the theme start correlating with each other (strands form)
#   4. DM scores elevate across the cluster
#   5. Price moves — retail notices
#
# Steps 1-3 happen BEFORE price moves.
# This scanner catches step 3 — the moment tickers start
# moving together for the first time.
#
# For quantum computing: if IonQ, Rigetti, D-Wave, IBM, Google
# were all uncorrelated last month and now suddenly show
# correlation emerging, that's institutional coordination beginning.
# That's your earliest possible signal.
#
# Two signals combined = high confidence new bubble:
#   Signal A: Strand emergence in a ticker cluster (this scanner)
#   Signal B: PE announcements in that sector (future integration)
#
# Either alone could be noise. Both together is signal.

# ============================================================
# WHAT TO BUILD
# ============================================================

# Weekly scan (run every Sunday evening):
# 1. Take all tickers in DM_2024_2026
# 2. For each ticker, calculate correlations in TWO windows:
#      - Last 30 days (current)
#      - 30-60 days ago (prior period)
# 3. Find pairs where:
#      - Prior correlation was LOW (< 0.40) — previously uncorrelated
#      - Current correlation is HIGH (> 0.65) — now coordinating
#      - This is a NEW STRAND FORMING
# 4. Group new strands by sector/theme
# 5. Flag any sector/theme where 3+ new strands appeared this week

# ============================================================
# PYTHON CODE — ADD TO chain_analysis.py
# ============================================================

def run_strand_emergence_scanner(dm_wide, dm_latest,
                                  current_window=30,
                                  prior_window_start=60,
                                  prior_window_end=30,
                                  emergence_threshold=0.65,
                                  prior_threshold=0.40,
                                  cluster_min_strands=3):
    """
    Detect new strand formations — earliest signal of bubble emergence.

    Scans for ticker pairs that were uncorrelated 30-60 days ago
    but are now correlated in the last 30 days. Groups by theme
    to identify emerging clusters.

    Parameters:
        dm_wide:              Full DM history DataFrame (dates x tickers)
        dm_latest:            Current DM scores dict
        current_window:       Days to look back for current correlation
        prior_window_start:   Days ago to start prior window
        prior_window_end:     Days ago to end prior window
        emergence_threshold:  Min correlation to count as NEW strand (0.65)
        prior_threshold:      Max correlation to count as PREVIOUSLY isolated (0.40)
        cluster_min_strands:  Min new strands to flag a cluster (3)

    Returns:
        new_strands_df:   All newly formed strand pairs
        clusters_df:      Grouped clusters with 3+ new strands (the signal)
    """

    print("Running strand emergence scanner...")
    print(f"  Current window:  last {current_window} days")
    print(f"  Prior window:    {prior_window_start}-{prior_window_end} days ago")

    total_days = len(dm_wide)

    # Slice the two time windows
    current_data = dm_wide.tail(current_window)
    prior_data   = dm_wide.iloc[
        max(0, total_days - prior_window_start) :
        max(0, total_days - prior_window_end)
    ]

    # Drop tickers with insufficient data in either window
    min_observations = int(current_window * 0.8)
    valid_current = current_data.dropna(axis=1, thresh=min_observations).columns
    valid_prior   = prior_data.dropna(axis=1, thresh=min_observations).columns
    valid_tickers = list(set(valid_current) & set(valid_prior))

    print(f"  Valid tickers (sufficient data in both windows): {len(valid_tickers)}")

    current_corr = current_data[valid_tickers].corr()
    prior_corr   = prior_data[valid_tickers].corr()

    # Find newly forming strands
    new_strands = []
    tickers = valid_tickers

    for i, ticker_a in enumerate(tickers):
        for ticker_b in tickers[i+1:]:
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
                    'Ticker_A':       ticker_a,
                    'Ticker_B':       ticker_b,
                    'Current_Corr':   round(c_corr, 3),
                    'Prior_Corr':     round(p_corr, 3),
                    'Corr_Change':    round(c_corr - p_corr, 3),
                    'DM_A':           dm_a,
                    'DM_B':           dm_b,
                    'Avg_DM':         round((dm_a + dm_b) / 2, 1) if dm_a and dm_b else None,
                    'Scan_Date':      datetime.now().strftime('%Y-%m-%d'),
                })

    print(f"  New strands found: {len(new_strands)}")

    if not new_strands:
        print("  No new strand formations detected this week.")
        return pd.DataFrame(), pd.DataFrame()

    new_strands_df = pd.DataFrame(new_strands)
    new_strands_df = new_strands_df.sort_values('Corr_Change', ascending=False)

    # Group into clusters — find tickers appearing in 3+ new strands
    # This identifies themes/sectors coordinating for the first time
    from collections import Counter

    all_tickers_in_strands = (
        list(new_strands_df['Ticker_A']) +
        list(new_strands_df['Ticker_B'])
    )
    ticker_counts = Counter(all_tickers_in_strands)

    # Flag tickers appearing in cluster_min_strands or more new strands
    cluster_tickers = {t: count for t, count in ticker_counts.items()
                       if count >= cluster_min_strands}

    clusters = []
    for ticker, strand_count in sorted(cluster_tickers.items(),
                                        key=lambda x: x[1], reverse=True):
        dm = dm_latest.get(ticker, {}).get('dm', None)

        # Find all partners for this ticker
        partners = list(
            new_strands_df[new_strands_df['Ticker_A'] == ticker]['Ticker_B']
        ) + list(
            new_strands_df[new_strands_df['Ticker_B'] == ticker]['Ticker_A']
        )

        clusters.append({
            'Ticker':         ticker,
            'New_Strands':    strand_count,
            'Partners':       ', '.join(partners[:5]),  # top 5
            'DM_Current':     dm,
            'Signal':         'BUBBLE_EMERGING' if strand_count >= 5 else 'WATCH',
            'Scan_Date':      datetime.now().strftime('%Y-%m-%d'),
        })

    clusters_df = pd.DataFrame(clusters)

    # Print summary
    print(f"\n  STRAND EMERGENCE CLUSTERS (3+ new strands):")
    print(f"  {'TICKER':<8} {'NEW_STRANDS':<14} {'DM':<8} {'SIGNAL':<20} {'PARTNERS'}")
    print(f"  {'-'*70}")
    for _, row in clusters_df.iterrows():
        print(f"  {row['Ticker']:<8} {row['New_Strands']:<14} "
              f"{str(row['DM_Current']):<8} {row['Signal']:<20} {row['Partners'][:40]}")

    return new_strands_df, clusters_df


# ============================================================
# ADD TO run_full_analysis() IN chain_analysis.py
# ============================================================

# After Deliverable 3, add:
#
#   print("\n[DELIVERABLE 5] Strand Emergence Scanner")
#   new_strands, clusters = run_strand_emergence_scanner(dm_wide, dm_latest)
#   if len(new_strands) > 0:
#       write_dataframe_to_sheet(new_strands, 'Strand_Emergence_All')
#   if len(clusters) > 0:
#       write_dataframe_to_sheet(clusters, 'Strand_Emergence_Clusters')

# ============================================================
# OUTPUT SHEETS
# ============================================================

# Sheet: Strand_Emergence_All
#   Every new strand pair detected this week
#   Columns: Ticker_A, Ticker_B, Current_Corr, Prior_Corr,
#            Corr_Change, DM_A, DM_B, Avg_DM, Scan_Date

# Sheet: Strand_Emergence_Clusters
#   THE SIGNAL — tickers in 3+ new strands simultaneously
#   Columns: Ticker, New_Strands, Partners, DM_Current, Signal, Scan_Date
#   Signal values:
#     BUBBLE_EMERGING = 5+ new strands (strong signal)
#     WATCH           = 3-4 new strands (monitor)

# ============================================================
# HOW TO USE THE OUTPUT
# ============================================================

# 1. Check Strand_Emergence_Clusters every Monday morning
# 2. Any ticker showing BUBBLE_EMERGING is worth investigating
# 3. Cross-reference with PE announcement news for that sector
# 4. If both strand emergence AND PE activity = high confidence early bubble
# 5. Run FlowOS Gate 2 on the cluster to confirm capital flows

# For quantum computing specifically:
# Search Strand_Emergence_All for these tickers:
#   IONQ, RGTI, QUBT, QBTS, IBM, GOOGL, MSFT (quantum divisions)
# If 3+ of them appear in new strands together = quantum bubble forming

# ============================================================
# SCHEDULE
# ============================================================

# Run: Every Sunday evening after market close
# Time: ~5-10 minutes (faster than full chain analysis)
# Function: run_strand_emergence_scanner(dm_wide, dm_latest)
# This is SEPARATE from the daily monitor (which runs every morning)
