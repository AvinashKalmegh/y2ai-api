Re-run all previous simulations using the v3 scripts:Use t1c_sector_runs_v3.py — replaces the original sector runs. Reports Sharpe, Sortino, Calmar alongside CAGR and max drawdown for all 8 sectors including Full Universe.Use t1c_ai_formation_v3.py — replaces the AI formation period run. Same 2023-2026 period.Use t1c_power_datacenter_v3.py — replaces the power and data center run.Use t1c_defense_dod_v3.py — replaces the defense/DOD run (also the first run of that sector).All v3 scripts import from t1c_sim_core_v3.py — make sure that file is in the same directory. The Tests 1-11 scripts also need to import from t1c_sim_core_v3 instead of t1c_sim_core to get Sharpe. Update the import line at the top of each test script from from t1c_sim_core import * to `froT1c sim core v3PY DownloadT1c sector runs v3PY DownloadT1c ai formation v3PY DownloadT1c power datacenter v3PY DownloadT1c defense dod v3PY


Use t1c_sector_runs_v3.py — replaces the original sector runs. Reports Sharpe, Sortino, Calmar alongside CAGR and max drawdown for all 8 sectors including Full Universe.
Use t1c_ai_formation_v3.py — replaces the AI formation period run. Same 2023-2026 period.
Use t1c_power_datacenter_v3.py — replaces the power and data center run.
Use t1c_defense_dod_v3.py — replaces the defense/DOD run (also the first run of that sector).
All v3 scripts import from t1c_sim_core_v3.py — make sure that file is in the same directory. The Tests 1-11 scripts also need to import from t1c_sim_core_v3 instead of t1c_sim_core to get Sharpe. Update the import line at the top of each test script from from t1c_sim_core import * to from t1c_sim_core_v3 import * and re-run all 11 tests.
Report Sharpe, Sortino, and Calmar for ev

