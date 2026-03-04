# Universe Scanner Rules

Pipeline that discovers, scores, and evaluates all liquid US equities.
Runs weekly (Saturday/Sunday) before DM calculation.

---

## Phase 1: Liquidity Filter

Source: `universe_scanner.py`

### Inclusion Criteria

| Rule | Threshold |
|------|-----------|
| Security types | Common Stock (CS) + ADRs (ADRC) |
| Market cap | >= $1B |
| Avg daily dollar volume (20-day) | >= $5M |
| Volume data requirement | At least 10 of 20 trading days with data |
| Exchanges | NYSE, NASDAQ, AMEX only |

### Valid Exchange Codes

| Code | Exchange |
|------|----------|
| XNYS | NYSE |
| XNAS | NASDAQ |
| XNMS | NASDAQ Global Select |
| XNCM | NASDAQ Capital Market |
| XNGS | NASDAQ Global Market |
| XASE | AMEX |

**Excluded:** All OTC, pink sheets, and non-major exchanges.

### Pipeline Steps

1. Fetch all active US stock tickers from Polygon (CS + ADRC)
2. Filter by valid exchange codes
3. Calculate 20-day average dollar volume using grouped daily endpoint
4. Filter by minimum dollar volume ($5M)
5. Fetch market cap via Polygon ticker details for volume-filtered tickers
6. Filter by minimum market cap ($1B)
7. Upsert results to `universe_tickers` table in Supabase

### Output Table: `universe_tickers`

| Column | Description |
|--------|-------------|
| ticker | Stock symbol (primary key) |
| name | Company name |
| exchange | Primary exchange code |
| market_cap | Latest market cap |
| avg_dollar_volume | 20-day average daily dollar volume |
| sic_code | SIC industry code |
| sic_description | SIC industry description |
| first_seen | Date ticker first entered universe |
| last_seen | Date ticker last passed filter |

---

## Phase 2: Dark Matter (DM) Calculation

Source: `universe_scanner_dm.py`

### DM Formula

```
DM_raw = (Rel_Str_ETF * 0.50) + (Rel_Str_SPY * 0.30) + (Volume_Z * 0.20)
DM_smoothed = EMA(DM_raw, span=5)
```

Clamped to 0-100 range.

### Component Details

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Rel Strength vs Sector ETF | 50% | `50 + (ticker_20d_return - etf_20d_return) * 500`, clamped 0-100 |
| Rel Strength vs SPY | 30% | `50 + (ticker_20d_return - spy_20d_return) * 500`, clamped 0-100 |
| Volume Z-Score | 20% | `(5d_avg_vol / baseline_avg_vol - 0.5) * 66.67`, clamped 0-100 |

### Parameters

| Parameter | Value |
|-----------|-------|
| Return period | 20 days (19 intervals) |
| EMA smoothing span | 5 |
| Volume average period | 20 days |
| Volume baseline window | 60 calendar days (excluding recent 5 days) |
| Rel strength scale factor | 500 |
| History days fetched | 280 (260 target + 20 buffer) |
| Data retained | 260 trading days |

### Sector ETF Mapping

| Sector | ETF |
|--------|-----|
| Technology | XLK |
| Semiconductors | SMH |
| Software | IGV |
| Consumer Discretionary | XLY |
| Consumer Staples | XLP |
| Health Care | XLV |
| Biotechnology | XBI |
| Financials | XLF |
| Industrials | XLI |
| Materials | XLB |
| Real Estate | XLRE |
| Utilities | XLU |
| Energy | XLE |
| Communication Services | XLC |
| Nuclear / Uranium | URA |
| Clean Energy | TAN |
| Cybersecurity | HACK |
| Aerospace & Defense / Defense | ITA |
| Transportation | IYT |
| **Default (unmapped)** | **SPY** |

### Sector Assignment Priority

1. Curated sector from `scanner_universe` table
2. SIC code mapping (built-in SIC-to-sector ranges)
3. Unmapped -> defaults to SPY as benchmark

### Output Table: `universe_dm_daily`

| Column | Description |
|--------|-------------|
| ticker | Stock symbol |
| date | Trading date |
| dm_raw | Raw DM score (0-100) |
| dm_smoothed | EMA-5 smoothed DM score (0-100) |
| close | Closing price |
| volume | Daily volume |

Primary key: `(ticker, date)`

---

## Phase 3: Crossings & Outcomes

Source: `universe_scanner_crossings.py`

### Crossing Detection

A **crossing** is detected when:
- `dm_smoothed` on day T >= 70 **AND**
- `dm_smoothed` on day T-1 < 70

This captures the upward cross of the 70 threshold.

### Outcome Evaluation

| Parameter | Value |
|-----------|-------|
| Crossing threshold | DM >= 70 |
| Hit definition | 10%+ max return within 90 calendar days |
| Evaluation windows | 30d, 60d, 90d max returns |
| Minimum forward data | 90 calendar days of price data required |

### Candidate Rules

A non-preferred ticker becomes a **candidate** when:

| Rule | Threshold |
|------|-----------|
| Not in Preferred 28 | Must be a new discovery |
| Evaluated crossings | >= 10 |
| Hit rate | >= 50% |
| Market cap | >= $2B |

### Degradation Rules

A Preferred 28 ticker is **flagged for degradation** when:

| Rule | Threshold |
|------|-----------|
| Must be in Preferred 28 | Existing member only |
| Evaluated crossings | >= 5 |
| Hit rate | < 40% |

### Era Consistency Check

Tickers are flagged as **inconsistent** when hit rate varies by more than 20 percentage points across eras:

| Era | Date Range |
|-----|------------|
| Pre-COVID | 2016-01-01 to 2019-12-31 |
| COVID + Recovery | 2020-01-01 to 2022-12-31 |
| AI Cycle | 2023-01-01 to 2026-12-31 |

### Ranked List Minimum

Tickers need **>= 3 evaluated crossings** to appear in the full ranked list and CSV export.

### Output Table: `universe_crossings`

| Column | Description |
|--------|-------------|
| id | Auto-increment ID |
| ticker | Stock symbol |
| cross_date | Date of upward crossing |
| entry_dm | DM score at crossing |
| entry_price | Price at crossing |
| max_price_90d | Max price in 90d window |
| max_return_30d | Max return in 30d window |
| max_return_60d | Max return in 60d window |
| max_return_90d | Max return in 90d window |
| is_hit | True if max_return_90d >= 10% |
| evaluated | True once outcome is calculated |

Primary key: `(ticker, cross_date)`

---

## Preferred 28

The curated benchmark set used for validation and degradation checks:

```
CEG  CRWD  TSM  APP  VRT  MU  NVDA  CCJ
DNN  PLTR  TSLA TTD  MSTR UEC LEU   HAL
WDC  ENPH  UUUU PDD  NCLH FCX RCL   LVS
PSKY MRNA  WYNN SMR
```

---

## Weekly Pipeline Sequence

```
Saturday/Sunday:
  1. universe_scanner.py filter        -> update universe_tickers
  2. universe_scanner_dm.py weekly     -> update universe_dm_daily
  3. universe_scanner_crossings.py all -> detect + evaluate + report
  4. universe_scanner_export.py        -> CSV export (optional --sheets)
```

---

## Report Sections (v3)

1. Summary (ticker counts, DM distribution, crossing totals)
2. Crossing Distribution (tickers by evaluated crossing count)
3. Preferred 28 Status (STRONG / OK / DEGRADED / LOW DATA / NO DATA)
4. New Candidates (non-preferred, 50%+, 10+ crossings, $2B+)
5. Degradation Flags (preferred below 40%)
6. Era Consistency Check (>20pp spread across eras)
7. Era Coverage (data availability per era)
8. Full Ranked List (all tickers with 3+ crossings)
9. Approaching Threshold (DM 60-69)
10. Top 20 DM Discovery (highest DM not in Preferred 28)
