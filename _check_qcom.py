import os
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client, Client

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
TICKER = 'QCOM'

def check_qcom_supabase():
    """Pull QCOM's most recent DM and the 11-day-prior DM for velocity check."""

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    response = (supabase.table('dm_history')
                .select('date, ticker, close, dm_smoothed, dm_raw, phase, etf, etf_dm, lead')
                .eq('ticker', TICKER)
                .order('date', desc=True)
                .limit(15)
                .execute())

    rows = response.data
    if not rows:
        print(f'{TICKER}: no data returned from Supabase')
        return

    latest = rows[0]
    prev_day = rows[1] if len(rows) > 1 else latest
    eleven_days_ago = rows[11] if len(rows) > 11 else rows[-1]

    dm_change_1d = latest["dm_smoothed"] - prev_day["dm_smoothed"]

    print(f'{TICKER} — Supabase data through {latest["date"]}')
    print(f'  Latest DM (smoothed): {latest["dm_smoothed"]:.2f}')
    print(f'  Latest DM (raw):      {latest["dm_raw"]:.2f}')
    print(f'  DM 1-day change:      {dm_change_1d:+.2f}')
    print(f'  Phase:                {latest["phase"]}')
    print(f'  Close:                ${latest["close"]:.2f}')
    print(f'  ETF:                  {latest["etf"]}  (ETF DM {latest["etf_dm"]:.2f}, lead {latest["lead"]:+.2f})')
    print()
    print(f'  11 trading days ago ({eleven_days_ago["date"]}):')
    print(f'    DM (smoothed):      {eleven_days_ago["dm_smoothed"]:.2f}')
    print(f'    Close:              ${eleven_days_ago["close"]:.2f}')
    print()

    velocity = latest["dm_smoothed"] - eleven_days_ago["dm_smoothed"]
    price_change = (latest["close"] - eleven_days_ago["close"]) / eleven_days_ago["close"] * 100
    print(f'  11-day DM velocity:   {velocity:+.2f} points')
    print(f'  11-day price chg:     {price_change:+.2f}%')

if __name__ == '__main__':
    check_qcom_supabase()
