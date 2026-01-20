from nst import NSTStorage, calculate_daily_summary, ATTRACTOR_TICKERS

storage = NSTStorage()

# Get unique dates from nst_mentions
result = storage.client.table('nst_mentions').select('published_at').execute()

# Extract unique dates
dates = set()
for row in result.data:
    pub = row.get('published_at')
    if pub:
        dates.add(pub[:10])  # YYYY-MM-DD

print(f'Found {len(dates)} unique dates')
print(f'Aggregating for {len(ATTRACTOR_TICKERS)} bubble types...\n')

for date_str in sorted(dates):
    for bubble_type in ATTRACTOR_TICKERS.keys():
        summary = calculate_daily_summary(storage, date_str, bubble_type)
        if summary.total_mentions > 0:
            print(f'{date_str} | {bubble_type}: {summary.total_mentions} mentions, density={summary.narrative_density:.3f}')

print('\nDone!')