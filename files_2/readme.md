Send the developer these four files along with this note — replace the t1c_sim_core.py he already has with the new version, the other three are new additions:
What to tell him:
Replace t1c_sim_core.py with the new version — it now supports HMS exit signals and ETF flow sizing in addition to the original logic.
Run in sequence:
python t1c_test5_lower_entry.py
python t1c_test6_hms_exit.py
python t1c_test7_etf_sizing.py
Important note for Test 7: Before running it, confirm the etf_flows_history table column name for flow direction — is it flow_direction, direction, or something else? Check and update ETF_FLOW_DIR_COL in the CONFIG block if different. Also confirm the ETF tickers in that table match XLK, SMH, SPY etc. If the table structure is different, send back the schema before running Test 7.
What each test answers:

Test 5 — does DM 60 add winners or just noise vs Test 2?
Test 6 — does HMS work better as exit signal than entry gate?
Test 7 — does ETF flow sizing improve returns or reduce 2022 drawdown?

Same output format as before — full yearly table, CAGR, max drawdown. Send back raw output for all three.