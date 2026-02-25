# Chain Analysis — Python Developer Brief
**Issued:** February 19, 2026
**Priority:** High — deliver tonight
**Developer:** Pathang (or assigned Python developer)

---

## CONTEXT

We ran a chain analysis on our 28-ticker V4 universe and found remarkable
structural architecture — firewall hierarchies, contagion pathways, and
critical toggle points. The results were descriptive but not yet tradeable.
We now need this running on the full universe (544 tickers) and converted
into a daily monitoring signal with a backtested predictive trigger.

This brief covers three deliverables in priority order.

---

## DATA SOURCES

**DM History (primary):**
- Google Sheet ID: `1GiLsxrgW-nssIhuUGYzn7SHYzcNjP4yg_7p1G2nPNiE`
- Sheet name: `DM_2024_2026`
- Columns: Date, Ticker, DM_Raw (or DM), Close
- Coverage: ~2 years daily, ~541 tickers
- Access: Use Google Sheets API with service account credentials

**DM Latest (for daily monitor):**
- Same spreadsheet, sheet name: `DM_Latest`
- Columns: Ticker (col B, index 1), DM (col C, index 2), Close (col L, index 11)
- Updated nightly by the Python pipeline

**Output target:**
- Write results back to the same Google Sheet
- New sheets: `Chain_Monitor`, `Chain_Backtest_Results`

---

## DELIVERABLE 1 — FULL UNIVERSE CHAIN MAP
**Priority: High | Estimated effort: 2-3 hours**

### What it does
Run the correlation-based chain analysis on all 544 tickers in DM_2024_2026.
Identify which tickers are mutation origins, which are firewalls, which are
satellites, and which are terminal basins.

### Methodology
A chain is a sequence of tickers where DM moves propagate from one to the next.

**Step 1 — Build correlation matrix**
For each pair of tickers, calculate rolling 60-day Pearson correlation of
DM_Raw values. Use the most recent 60 days of data.

**Step 2 — Build directed graph**
For each ticker pair (A, B) with correlation > 0.65:
- Determine direction: which ticker's DM move leads by 1-5 days
- Use Granger causality or simple lag correlation to determine direction
- Edge: A → B means A's DM change predicts B's DM change 1-5 days later

**Step 3 — Classify each ticker**
- MUTATION_ORIGIN: high out-degree (causes many others), DM > 70
- FIREWALL: high in-degree from origins, does NOT propagate to next tier
- SATELLITE: receives from origins/firewalls, limited out-degree
- TERMINAL_BASIN: high in-degree, near-zero out-degree, typically low DM
- ISOLATED: no significant correlations in either direction

**Step 4 — Identify chain sequences**
For each mutation origin, trace the full propagation path:
`ORIGIN → [intermediate nodes] → TERMINAL_BASIN`
Record: chain length, blocked % at each firewall, spreading % past each node

### Output — Sheet: `Chain_Full_Universe`
| Ticker | Classification | Out_Degree | In_Degree | Primary_Origin |
| Longest_Chain_Position | Firewall_Effectiveness | Sector | DM_Current |

### Key findings we already know from V4 (validate these hold in full universe)
- TSLA at DM 41 is a portfolio toggle — below 30 triggers cascade
- APP defeats VRT firewall by routing contagion around it
- Nuclear cluster (CEG, VST, NRG) is a terminal basin receiving from all directions
- PLTR, APP, MSTR are active mutation origins

---

## DELIVERABLE 2 — DAILY CHAIN MONITOR
**Priority: High | Estimated effort: 1-2 hours**

### What it does
Every morning after DM_Latest updates, check the five critical nodes from
the V4 analysis and any high-degree nodes found in Deliverable 1. Flag
state changes that require attention.

### Nodes to monitor daily
```python
CRITICAL_NODES = {
    'TSLA':  {'role': 'TOGGLE',          'alert_below': 35, 'alert_above': 65},
    'APP':   {'role': 'FIREWALL_BYPASS', 'alert_above': 85, 'alert_below': 40},
    'PLTR':  {'role': 'MUTATION_ORIGIN', 'alert_below': 50},
    'MSTR':  {'role': 'MUTATION_ORIGIN', 'alert_below': 50},
    'CEG':   {'role': 'TERMINAL_BASIN',  'alert_below': 30},
    'NVDA':  {'role': 'CORE_MASS',       'alert_below': 60},
}
```
Add top 10 highest out-degree nodes from Deliverable 1 to this list.

### Alert conditions
- TSLA crosses below 35: `CASCADE_RISK — portfolio contagion pathway open`
- TSLA crosses above 65: `FIREWALL_RESTORED — cascade risk reduced`
- APP crosses above 85: `BYPASS_ACTIVE — VRT firewall circumvented`
- Any mutation origin drops below 50: `ORIGIN_WEAKENING — monitor satellites`
- Two or more origins weaken simultaneously: `SYSTEMIC_SIGNAL`

### Output — Sheet: `Chain_Monitor`
One row appended daily:
| Date | TSLA_DM | APP_DM | PLTR_DM | MSTR_DM | CEG_DM | NVDA_DM |
| Alert_Count | Alerts_Text | Cascade_Risk | Systemic_Signal |

---

## DELIVERABLE 3 — PROPAGATION TIMING BACKTEST
**Priority: Medium | Estimated effort: 3-4 hours**
**This is the one that converts chain analysis from descriptive to tradeable**

### The hypothesis
When an anchor ticker (mutation origin, high DM) enters EXHAUST phase
(DM drops below 50 after being above 70), do its satellite tickers follow
into stress (DM drops below 40) within 20-30 days?

If yes: anchor weakens today → short satellites now → cover in 3-4 weeks.
That's a lead-time trading signal.

### Methodology

**Step 1 — Identify all EXHAUST events in DM_2024_2026**
For each mutation origin identified in Deliverable 1:
- Find every date where DM crossed from above 70 to below 50
- These are your EXHAUST events
- Minimum: 10 days above 70 before the crossing (filters noise)

**Step 2 — Identify satellites for each origin**
Use the chain graph from Deliverable 1:
- Satellites = tickers at position 2-3 downstream from the origin
- Minimum correlation 0.65 with lag 1-5 days

**Step 3 — Measure satellite response**
For each EXHAUST event:
- Record satellite DM on the event date
- Measure: did satellite DM drop below 40 within 10, 20, 30 days?
- Record: time to satellite stress (days), magnitude of satellite DM drop

**Step 4 — Calculate statistics**
For each origin-satellite pair:
- Hit rate: % of EXHAUST events followed by satellite stress within 30 days
- Avg lead time: average days between origin EXHAUST and satellite stress
- Avg magnitude: average satellite DM drop
- False positive rate: EXHAUST events NOT followed by satellite stress

**Step 5 — Position dependence**
Critical question from V4 analysis: firewalls only work at positions 1-2.
By position 3+, upstream mutation already tipped the chain.
Test: does hit rate change significantly between position 1 and position 3?

### Output — Sheet: `Chain_Backtest_Results`
| Origin | Satellite | Chain_Position | Exhaust_Events | Hit_Rate_10d |
| Hit_Rate_20d | Hit_Rate_30d | Avg_Lead_Days | Avg_Magnitude |
| False_Positive_Rate | Tradeable (Y/N based on hit_rate > 65%) |

### What makes it tradeable
Hit rate > 65% AND avg lead time > 5 days = actionable signal.
The lead time matters — a 1-day lead has no practical trading value.
A 15-day lead means you can enter the satellite short after origin EXHAUST
and have two weeks before the move happens.

---

## DELIVERABLE 4 — STRAND ANALYSIS ON FULL UNIVERSE
**Priority: Medium | Schedule separately if tonight is too much**

### What it does
A strand is a chain that persists across multiple DM cycles — meaning the
propagation relationship between two tickers holds not just once but
repeatedly across different market regimes. Strands are more reliable than
one-off chain correlations.

### Methodology
For each ticker pair with a directed edge in the chain graph:
- Count how many distinct DM cycles show the same propagation direction
- A cycle = one complete BUILD → EXHAUST sequence (DM above 70 then below 40)
- Strand strength = cycles showing consistent direction / total cycles observed

### Output — Sheet: `Strand_Analysis`
| Ticker_A | Ticker_B | Direction | Strand_Strength | Cycles_Observed |
| Consistent_Cycles | Avg_Lag_Days | Sector_A | Sector_B |

Filter: only show strands with strength > 0.70 and cycles_observed >= 3.

---

## TECHNICAL NOTES

**Performance:**
- DM_2024_2026 has ~541 tickers × ~500 trading days = ~270,000 rows
- Full correlation matrix is 541×541 = ~292,000 pairs
- Use pandas + numpy for vectorized operations
- Granger causality is expensive — consider lag correlation as faster proxy

**Lag correlation as Granger proxy:**
```python
def get_lag_correlation(series_a, series_b, max_lag=5):
    best_lag, best_corr = 0, 0
    for lag in range(1, max_lag+1):
        corr = series_a.corr(series_b.shift(lag))
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr
# Direction: if corr(A, B.shift(lag)) > corr(B, A.shift(lag)) → A leads B
```

**Writing back to Google Sheets:**
Use gspread or google-api-python-client.
Append-only for Chain_Monitor (daily log).
Overwrite for Chain_Full_Universe and Chain_Backtest_Results (static analysis).

**Runtime estimate:**
- Deliverable 1 (full universe map): ~10-15 min
- Deliverable 2 (daily monitor): <1 min per run
- Deliverable 3 (backtest): ~20-30 min
- Deliverable 4 (strand analysis): ~15-20 min

---

## QUESTIONS FOR DEVELOPER

1. Do you have the Google Sheets service account credentials already configured?
2. Can you confirm DM_2024_2026 is accessible and the column schema matches above?
3. For Deliverable 3, let us know if the number of EXHAUST events per origin
   is too sparse for statistical significance — we may need to adjust thresholds.

---

## PRIORITY ORDER FOR TONIGHT

1. Deliverable 2 (Daily Monitor) — simplest, highest immediate value
2. Deliverable 3 (Propagation Backtest) — converts theory to tradeable signal
3. Deliverable 1 (Full Universe Map) — needed for 3 but can run overnight
4. Deliverable 4 (Strand Analysis) — schedule separately if time is short

The backtest (Deliverable 3) is the most important output.
Everything else supports it.
