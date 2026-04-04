"""
HMS SUSTAINED ELEVATION ANALYSIS
==================================
For each ticker, examines HMS behavior from T-0 through Entry to determine
whether the initial elevation was sustained, cyclical, or a brief spike.

Outputs: hms_sustained_elevation.csv and hms_sustained_elevation.xlsx
"""
import pandas as pd
import os

# T-0 and Entry dates from ticker_timeline_results
TICKERS = [
    ('DELL', '2023-02-01', '2024-03-07', 14.72),
    ('WDC',  '2018-07-06', '2019-07-16', 15.47),
    ('PCG',  '2021-08-11', '2022-09-07', 17.75),
    ('LRCX', '2018-08-27', '2019-09-26', 10.38),
    ('TRGP', '2016-01-21', '2016-09-23', -3.05),
    ('VZ',   '2017-09-20', '2018-10-25',  1.10),
    ('ADSK', '2016-02-16', '2016-09-02',  2.10),
]

HMS_FILES = ['HMS_2016_2019.csv', 'HMS_2020_2022.csv', 'HMS_Daily.csv']


def classify_pattern(days_above_65_pct, peak, mean):
    """Classify the HMS elevation pattern."""
    if days_above_65_pct >= 10 and peak >= 0.75:
        return 'STRONG_SUSTAINED'
    elif days_above_65_pct >= 5 and peak >= 0.70:
        return 'CYCLICAL_SUSTAINED'
    elif days_above_65_pct >= 2 and peak >= 0.65:
        return 'MODERATE'
    elif peak >= 0.60:
        return 'WEAK_BRIEF'
    else:
        return 'MINIMAL'


def main():
    print("\n=== HMS SUSTAINED ELEVATION ANALYSIS ===\n")

    # Load all HMS data
    print("Loading HMS data...")
    hms_parts = []
    for f in HMS_FILES:
        path = f
        if not os.path.exists(path):
            print(f"  WARNING: {f} not found, skipping")
            continue
        df = pd.read_csv(path, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'])
        hms_parts.append(df)
        print(f"  Loaded {f}: {len(df):,} rows")

    hms = pd.concat(hms_parts, ignore_index=True)
    print(f"  Combined: {len(hms):,} rows\n")

    summary_rows = []
    monthly_rows = []

    for ticker, t0_str, entry_str, return_pct in TICKERS:
        t0 = pd.Timestamp(t0_str)
        entry = pd.Timestamp(entry_str)

        sub = hms[hms['Ticker'].str.upper() == ticker.upper()].sort_values('Date')
        after_t0 = sub[(sub['Date'] >= t0) & (sub['Date'] <= entry)].copy()

        print(f"--- {ticker} | T-0: {t0_str} -> Entry: {entry_str} | {len(after_t0)} trading days ---")

        if len(after_t0) == 0:
            print(f"  No HMS data in window\n")
            summary_rows.append({
                'Ticker': ticker, 'T0_Date': t0_str, 'Entry_Date': entry_str,
                'Return_Pct': return_pct, 'Trading_Days': 0,
                'HMS_Mean': '', 'HMS_Max': '', 'HMS_Min': '',
                'Days_Above_060': 0, 'Days_Above_065': 0,
                'Days_Above_070': 0, 'Days_Above_075': 0, 'Days_Above_080': 0,
                'Pct_Above_060': 0, 'Pct_Above_065': 0,
                'Pct_Above_070': 0, 'Pct_Above_075': 0, 'Pct_Above_080': 0,
                'Pattern': 'NO_DATA',
            })
            continue

        scores = after_t0['HMS_Score']
        n = len(after_t0)

        counts = {
            '060': len(after_t0[scores >= 0.60]),
            '065': len(after_t0[scores >= 0.65]),
            '070': len(after_t0[scores >= 0.70]),
            '075': len(after_t0[scores >= 0.75]),
            '080': len(after_t0[scores >= 0.80]),
        }
        pcts = {k: round(v / n * 100, 1) for k, v in counts.items()}

        pattern = classify_pattern(pcts['065'], scores.max(), scores.mean())

        print(f"  HMS: min={scores.min():.4f}  mean={scores.mean():.4f}  max={scores.max():.4f}")
        print(f"  >=0.60: {counts['060']} ({pcts['060']}%)  >=0.65: {counts['065']} ({pcts['065']}%)  "
              f">=0.70: {counts['070']} ({pcts['070']}%)  >=0.80: {counts['080']} ({pcts['080']}%)")
        print(f"  Pattern: {pattern}")

        summary_rows.append({
            'Ticker': ticker, 'T0_Date': t0_str, 'Entry_Date': entry_str,
            'Return_Pct': return_pct, 'Trading_Days': n,
            'HMS_Mean': round(scores.mean(), 4),
            'HMS_Max': round(scores.max(), 4),
            'HMS_Min': round(scores.min(), 4),
            'Days_Above_060': counts['060'], 'Days_Above_065': counts['065'],
            'Days_Above_070': counts['070'], 'Days_Above_075': counts['075'],
            'Days_Above_080': counts['080'],
            'Pct_Above_060': pcts['060'], 'Pct_Above_065': pcts['065'],
            'Pct_Above_070': pcts['070'], 'Pct_Above_075': pcts['075'],
            'Pct_Above_080': pcts['080'],
            'Pattern': pattern,
        })

        # Monthly breakdown
        after_t0 = after_t0.copy()
        after_t0['Month'] = after_t0['Date'].dt.to_period('M')
        monthly = after_t0.groupby('Month')['HMS_Score'].agg(['mean', 'max', 'min', 'count'])

        for period, row in monthly.iterrows():
            monthly_rows.append({
                'Ticker': ticker,
                'Month': str(period),
                'HMS_Mean': round(row['mean'], 4),
                'HMS_Max': round(row['max'], 4),
                'HMS_Min': round(row['min'], 4),
                'Trading_Days': int(row['count']),
            })
            print(f"    {period}  avg={row['mean']:.4f}  peak={row['max']:.4f}  days={int(row['count'])}")

        print()

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    summary_df.to_csv('hms_sustained_elevation.csv', index=False)
    print(f"Saved: hms_sustained_elevation.csv")

    # Excel with both sheets
    try:
        with pd.ExcelWriter('hms_sustained_elevation.xlsx', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            monthly_df.to_excel(writer, sheet_name='Monthly_Detail', index=False)
            for sheet_name in ['Summary', 'Monthly_Detail']:
                ws = writer.sheets[sheet_name]
                for col in ws.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col) + 4
                    ws.column_dimensions[col[0].column_letter].width = min(max_len, 40)
        print(f"Saved: hms_sustained_elevation.xlsx (2 sheets: Summary + Monthly_Detail)")
    except Exception as e:
        print(f"Excel export skipped: {e}")
        monthly_df.to_csv('hms_sustained_elevation_monthly.csv', index=False)
        print(f"Saved: hms_sustained_elevation_monthly.csv (fallback)")

    print(f"\n=== COMPLETE ===\n")
    print("PATTERN SUMMARY:")
    print(summary_df[['Ticker', 'HMS_Mean', 'HMS_Max', 'Pct_Above_065', 'Pattern', 'Return_Pct']].to_string(index=False))


if __name__ == '__main__':
    main()
