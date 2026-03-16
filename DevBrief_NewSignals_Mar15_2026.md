# Developer Brief — New Signal Collection Tasks
## Y2AI Research | March 15, 2026 | Priority: This Week

---

## CONTEXT

We are expanding the Y2AI signal collection stack with five new signals identified
through SIGINT framework analysis. These sit upstream of the existing DM signal —
meaning they detect institutional positioning before it shows in capital flows.

The existing Margin Stress Proxy brief (DTCC + H.8) covers the funding stress layer.
This brief covers the five additional collection tasks.

Priority order is listed below. Build in sequence.

---

## TASK 1 — OPTIONS SKEW (Priority 1 — High Value, Polygon Already Paid)

### What it is
Put/call ratio measures direction. Skew measures fear — specifically the cost
differential between out-of-the-money puts and out-of-the-money calls. When skew
steepens sharply on a name, institutions are paying up for tail protection before
any price signal fires.

### Why it matters
Skew often precedes DM deterioration by 2-5 days. It sits at Stage 2 of our causal
chain (Order Flow Imbalance) and is complementary to the existing put/call ratio signal.

### Data source
Polygon.io options chain — already in our stack. Pull for each ticker in the universe.

### Computation
```python
# For each ticker daily:
# Get options chain for current expiry (30-day)
# OTM Put = strike 5% below current price
# OTM Call = strike 5% above current price
# Skew = IV(OTM Put) - IV(OTM Call)
# Positive skew = puts more expensive = fear signal
# Skew Z-score vs 30-day rolling window = normalized signal

skew_raw = iv_otm_put - iv_otm_call
skew_zscore = (skew_raw - skew_30d_mean) / skew_30d_std

# Alert threshold: skew_zscore > 2.0 = elevated fear
# Combine with DM direction for signal confirmation
```

### Supabase table
```sql
CREATE TABLE options_skew (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    skew_raw DECIMAL(6,4),
    iv_otm_put DECIMAL(6,4),
    iv_otm_call DECIMAL(6,4),
    skew_zscore DECIMAL(6,2),
    signal VARCHAR(20),  -- NORMAL / ELEVATED / EXTREME
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);
```

### Pull schedule
Daily at 4:30 PM ET after market close. Same cadence as EOD DM.

---

## TASK 2 — ETF CREATION/REDEMPTION UNITS (Priority 2)

### What it is
When authorized participants (APs) create or redeem large ETF baskets, they signal
directional intent before price moves. AP redeeming $500M of QQQ shares is
institutional distribution happening in the creation/redemption mechanism, not
the secondary market. Sits upstream of DM.

### Why it matters
ETF flows lead sector DM by 1-3 days in our framework. This is the earliest
public signal of institutional sector repositioning.

### Data source
ETF sponsor daily holdings files — public, free, updated daily.

Key ETFs to track (maps to our eight economic forces):
- QQQ (Enterprise Demand)
- XLK (Technology)
- SMH (Semiconductors)
- XLF (Capital/Financials)
- XLE (Energy)
- XLI (Infrastructure)
- ARKK (Innovation/Growth)
- SPY (Baseline)

### Where to get it
Each ETF sponsor publishes daily holdings CSV:
- iShares: ishares.com/us/products/etf-investments → Holdings tab → CSV download
- Invesco (QQQ): invesco.com/us/financial-products/etfs/product-detail
- State Street (SPY, XL series): ssga.com/us/en/individual/etfs

### Computation
```python
# Compare shares outstanding day over day
# Large single-day decrease = AP redemption = institutional exit signal
# Large single-day increase = AP creation = institutional entry signal

redemption_pct = (shares_outstanding_yesterday - shares_outstanding_today) / shares_outstanding_yesterday * 100

# Alert threshold: redemption > 2% in single day = significant outflow
# Combine with sector DM direction for confirmation
```

### Supabase table
```sql
CREATE TABLE etf_flows (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    etf_ticker VARCHAR(10) NOT NULL,
    shares_outstanding BIGINT,
    shares_change INTEGER,
    flow_pct DECIMAL(6,2),
    flow_direction VARCHAR(10),  -- INFLOW / OUTFLOW / NEUTRAL
    flow_signal VARCHAR(20),     -- NORMAL / NOTABLE / SIGNIFICANT
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, etf_ticker)
);
```

### Pull schedule
Daily at 8:00 AM ET — ETF sponsors publish previous day's data overnight.

---

## TASK 3 — HARD TO BORROW RATES (Priority 3)

### What it is
When short interest rises, the cost to borrow shares rises with it. A spike in
borrow cost before short interest shows up in public data is an early warning
that sophisticated shorts are positioning. Distinct from short interest — this
is a leading indicator of short interest.

### Why it matters
Sits at Stage 3 of our causal chain (Execution Fragmentation). Sophisticated
shorts establish borrow before executing, so borrow cost spikes precede
short volume spikes.

### Data source
Interactive Brokers publishes a public hard-to-borrow list daily:
https://www.ibkr.com/en/trading/short-selling

Also available from Ortex and S3 Partners (paid) but start with IBKR free data.

### Computation
```python
# Daily borrow rate per ticker (annualized %)
# Normal: < 1% annualized
# Elevated: 1-5% annualized
# Hard to borrow: > 5% annualized
# Extreme: > 25% annualized = major short squeeze risk or insider knowledge

# Z-score vs 30-day rolling average for each ticker
borrow_zscore = (borrow_rate_today - borrow_30d_mean) / borrow_30d_std

# Alert: borrow_zscore > 2.5 AND DM declining = institutional short building
```

### Supabase table
```sql
CREATE TABLE borrow_rates (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    borrow_rate_annualized DECIMAL(8,4),
    borrow_zscore DECIMAL(6,2),
    classification VARCHAR(20),  -- NORMAL / ELEVATED / HTB / EXTREME
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);
```

### Pull schedule
Daily at 7:00 AM ET — IBKR publishes overnight.

---

## TASK 4 — CROSS-ASSET DIVERGENCE / CDS SPREADS (Priority 4)

### What it is
When a company's equity DM is rising but its credit default swap (CDS) spread
is simultaneously widening, the bond market is disagreeing with the equity market.
Bond traders are typically more informed than equity traders on credit deterioration.
This divergence is a specific signal type not currently in the stack.

### Why it matters
CDS divergence from DM is one of the earliest warnings of structural deterioration.
The bond market caught Enron, Lehman, SVB, and every major corporate credit event
before equity signals fired.

### Data source
Start with Markit/IHS Markit CDS data — available through our Moody's MCP connector.
Alternatively FRED publishes aggregate investment grade and high yield CDS indices.
For individual names, Bloomberg or Refinitiv are the clean sources — but check
Markit first through the existing MCP connection.

Focus on names in our Hollowing Short portfolio and any name in our universe
with DM above 70 (to catch divergence where equity signal is bullish but credit
is bearish).

### Computation
```python
# Daily for each covered ticker:
# equity_signal = DM_EMA5 (our existing score)
# credit_signal = CDS_spread_zscore (normalized vs 90-day)

# Divergence = equity bullish (DM > 65) AND credit bearish (CDS z > 1.5)
# OR equity bearish (DM < 35) AND credit bullish (CDS z < -1.5)

divergence_flag = (dm_ema5 > 65 and cds_zscore > 1.5) or \
                  (dm_ema5 < 35 and cds_zscore < -1.5)
```

### Note
This task depends on confirming what's available through the Moody's MCP connector.
Check that first before building a separate data pipeline.

---

## TASK 5 — SPECIAL REPO RATES (Priority 5 — Research First)

### What it is
In the repo market, most securities trade at the "general collateral" rate. When
a specific security trades at a significantly lower rate (called "special"), it means
that security is in high demand as collateral — usually because it's being heavily
shorted and dealers need to deliver it. Special repo rates on individual securities
precede short squeeze conditions.

### Why it matters
More granular than the aggregate H.8 data in the MSP brief. Catches name-specific
stress before it shows in aggregate metrics.

### Data source
DTCC GCF Repo Index (public), SOFR term rates (NY Fed), and SIFMA repo data.
Individual special rates are harder to get without a Bloomberg terminal.

### Action for now
Research what's publicly available before committing to build. This is a Phase 4+
item. Flag for discussion when MSP and options skew are stable.

---

## INTEGRATION NOTES

All new signals integrate with the existing Supabase backend and appear in the
ARGUS morning workflow. The integration points:

**Morning workflow addition (after gauge readings):**
```
Options Skew Alert:    [NORMAL / ELEVATED / EXTREME] — top 3 flagged names
ETF Flow Alert:        [INFLOW / OUTFLOW] — largest single-day moves
Borrow Rate Alert:     [names with borrow_zscore > 2.5]
Cross-Asset Divergence:[names where equity and credit disagree]
```

**FlowMap T2 additions:**
Options skew appears alongside existing put/call ratio in the conviction depth panel.
Borrow rate appears as a new row in the seven-signal stack.

**Sizing modifier:**
When Options Skew is EXTREME + DM declining + Borrow rate elevated on the same name:
reduce position sizing by 25% regardless of DM reading. This is the institutional
exit signature — don't fight it.

---

## DELIVERY SCHEDULE

| Task | Estimated Hours | Target |
|------|----------------|--------|
| Options Skew (Polygon) | 3-4 hours | Tuesday |
| ETF Creation/Redemption | 4-6 hours | Wednesday |
| Hard to Borrow (IBKR) | 3-4 hours | Thursday |
| CDS Divergence (Moody's MCP) | 2-3 hours | Friday |
| Special Repo Rates | Research only | Next week |

Start with Options Skew — highest value, data source already in stack.

---

*Y2AI Research | Signal Collection Expansion Brief | March 15, 2026*
*Framework: SIGINT causal chain — Stage 2 (Order Flow Imbalance) through Stage 3 (Execution Fragmentation)*
*Validated against: Calypso institutional insight (March 13, 2026), SIGINT framework review (March 15, 2026)*
