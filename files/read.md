Please run 4 simulation tests and send back the output.

ATTACHED FILES (5 files — keep all in the same folder):
  t1c_sim_core.py        ← shared engine, do not run directly
  t1c_test1_baseline.py  ← Test 1
  t1c_test2_dm_only.py   ← Test 2
  t1c_test3_higher_entry.py ← Test 3
  t1c_test4_tighter_exit.py ← Test 4


STEP 1 — INSTALL
  pip install supabase pandas numpy python-dotenv


STEP 2 — CREATE .env FILE
  SUPABASE_URL=https://your-project.supabase.co
  SUPABASE_KEY=your-service-role-key


STEP 3 — CONFIRM COLUMN NAMES (important — do this before running anything)
  Open t1c_test1_baseline.py and find the CONFIG block at the top.
  Confirm these match the actual Supabase table and column names:

    DM_TABLE      = 'dm_daily'     (table name for daily DM data)
    DM_DATE_COL   = 'date'         (date column)
    DM_TICKER_COL = 'ticker'       (ticker column)
    DM_CLOSE_COL  = 'close'        (closing price column)
    DM_SCORE_COL  = 'dm'           (the smoothed DM EMA5 score column)

    HMS_TABLE     = 'hms_daily'    (table name for HMS data)
    HMS_DATE_COL  = 'date'
    HMS_TICKER_COL= 'ticker'
    HMS_SCORE_COL = 'hms_score'    (the 4-component HMS score column)

  Update any names that don't match in ALL FOUR test files before running.
  This is the only thing that needs changing.


STEP 4 — RUN THE 4 TESTS (run one at a time)

  python t1c_test1_baseline.py
  python t1c_test2_dm_only.py
  python t1c_test3_higher_entry.py
  python t1c_test4_tighter_exit.py

  Each test runs the full decade 2016-2026 and prints a yearly NAV table.
  Expected run time: 2-5 minutes per test.


STEP 5 — WHAT TO WATCH FOR
  The script automatically checks for zero-day holds (a position opened
  and closed on the same day). If more than 5% of trades are zero-day,
  the script prints a WARNING and stops.

  If WARNING fires:
    - Stop that test
    - Send back what you have so far
    - Do not continue to the next test
    - We will investigate and fix before continuing

  If no WARNING:
    - Run all 4 tests and send back all 4 outputs


WHAT TO SEND BACK
  The full console output for each test — do not summarize.
  I need to see the complete yearly table for each test.
  Label each output clearly: TEST 1, TEST 2, TEST 3, TEST 4.

  The table will look like this:
    Year   NAV            Ret%    Trades   Hit%   AvgHold  ZeroD
    2016   $1,320,000    +32.0%      45   62.2%      47d      2
    2017   $1,750,000    +32.6%      52   65.4%      51d      1
    ...
    FINAL NAV:   $X,XXX,XXX
    TOTAL RETURN: +XXX.X%
    CAGR:         XX.X% per year


IF YOU HIT ERRORS
  Supabase connection error — check .env file
  Column not found error — check CONFIG column names in Step 3
  Any other error — send the full error message