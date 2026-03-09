# Intraday Capital Flow Computation via Polygon API
## Developer Brief — March 9, 2026

### OBJECTIVE

Compute intraday DM (Capital Flow) scores using Polygon.io price and volume data 
so that signal readings update during the trading day, not just at EOD. The current 
system computes DM once daily after market close. This addition gives us mid-day 
signal reads — critical for detecting marketplace impact, earnings reactions, and 
cascade propagation in real time.

### WHAT DM REQUIRES (no formula disclosed — just the data inputs)

DM computation needs three inputs per ticker per day:

1. **Ticker 20-day return** — closing price today vs closing price 20 trading days ago
2. **Sector ETF 20-day return** — same calculation for the sector ETF benchmark
3. **SPY 20-day return** — same calculation for SPY
4. **Volume Z-score** — today's volume vs the 20-day average volume, normalized

For intraday computation, we replace "closing price today" with "current price" 
(last trade or VWAP). The 20-day lookback prices remain the same (historical closes). 
Only today's price and volume are live.

### POLYGON API CALLS NEEDED

**1. Current/Last Trade Price**
```
GET https://api.polygon.io/v2/last/trade/{ticker}
```
Returns the most recent trade price. Call this for each ticker + its sector ETF + SPY.

**2. Previous Day Close (for the 20-day lookback)**
Already in our database from the EOD pipeline. No new API call needed for historical 
closes. Use DM_2024_2026 or the price history table.

**3. Intraday Volume**
```
GET https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}
```
The `v` field in the response gives cumulative volume for the day so far.
Compare against the 20-day average daily volume (already computed in EOD pipeline).

**4. Batch Snapshots (preferred — one call for all tickers)**
```
GET https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers
```
Returns last trade price and cumulative volume for ALL tickers in one call.
This is the efficient path — one API call replaces 585 individual calls.
Filter the response to our 585-ticker universe + sector ETFs + SPY.

### COMPUTATION FLOW

```
1. Pull snapshot (one Polygon call)
2. For each ticker in Scanner_Universe:
   a. current_price = snapshot last trade price
   b. price_20d_ago = from price history table (already stored)
   c. ticker_return_20d = (current_price - price_20d_ago) / price_20d_ago
   d. etf_return_20d = same calc for sector ETF (ETF mapping from DV_LAYER_MAP)
   e. spy_return_20d = same calc for SPY
   f. volume_today = snapshot cumulative volume
   g. volume_20d_avg = from volume history (already stored)
   h. volume_zscore = (volume_today - volume_20d_avg) / volume_20d_std
   i. Compute DM using production formula (in dm_calculator.py)
   j. Apply EMA5 smoothing against prior 4 days of DM history
3. Write results to DM_Intraday table (new table, same schema as DM_Latest)
4. Timestamp each row with scan time
```

### NEW DATABASE TABLE

```sql
CREATE TABLE dm_intraday (
    scan_time       TIMESTAMP NOT NULL,
    ticker          VARCHAR(10) NOT NULL,
    dm_score        DECIMAL(5,1),
    dm_ema5         DECIMAL(5,1),
    current_price   DECIMAL(10,2),
    volume_today    BIGINT,
    volume_zscore   DECIMAL(5,2),
    return_20d      DECIMAL(8,4),
    etf_return_20d  DECIMAL(8,4),
    spy_return_20d  DECIMAL(8,4),
    PRIMARY KEY (scan_time, ticker)
);
```

### SCHEDULING

Run at three fixed times during the trading day:
- **10:30 AM ET** — 1 hour after open, volume has stabilized
- **1:00 PM ET** — midday read
- **3:30 PM ET** — 30 min before close, near-final read

The EOD pipeline remains unchanged. These intraday reads are ADDITIONAL, not replacements.
The EOD DM in DM_Latest is still the official production signal.

### IMPORTANT NOTES

1. **Volume Z-score will be noisy early in the day.** At 10:30 AM, only ~30% of daily 
   volume has traded. The Z-score will be artificially low. Consider scaling:
   `adjusted_volume = volume_today * (390 / minutes_since_open)` to annualize.
   Or just accept the noise and use the 1:00 PM and 3:30 PM reads as more reliable.

2. **EMA5 smoothing uses prior 4 DAYS of EOD DM.** The intraday read is EMA5 where 
   the prior 4 values are yesterday's DM, day-before, etc. from DM_Latest. Only 
   today's value is intraday. This means the EMA5 won't whipsaw — it's anchored 
   by 4 days of stable EOD readings.

3. **Sector ETF mapping** is in DV_LAYER_MAP. The developer already has this from 
   the EOD pipeline. Same mapping applies intraday.

4. **Polygon rate limits:** The snapshot endpoint is 5 calls/minute on the free tier, 
   unlimited on paid. We're on paid. One snapshot call per scan time = 3 calls/day. 
   Well within limits.

5. **The production DM formula is in dm_calculator.py** (or the equivalent module). 
   The developer already has this. The intraday computation uses the SAME formula — 
   only the price and volume inputs change from EOD closes to current values.

6. **Priority tickers for first deployment:** Start with the 18 Marketplace Watchlist 
   tickers + the 26 FlowOS production universe. That's 44 tickers. Expand to full 
   585 after confirming the pipeline works.

### FIRST USE CASE — MONDAY MARCH 9

Run the 10:30 AM scan on the 18 Marketplace Watchlist tickers. Compare intraday DM 
against Friday's baseline DM in the Marketplace_Watch spreadsheet. This tells us 
whether the Anthropic Marketplace announcement is moving institutional flow before 
the EOD pipeline catches it.

Specifically watch:
- UPWK (Friday DM 11.5, hypothesis: STRONGLY BEARISH)
- GTLB (Friday DM 5.7, hypothesis: bounce possible)
- ACN (Friday DM 13.7, hypothesis: hollowing confirmed)
- HUBS (Friday DM 90.0, hypothesis: test of which SaaS survives)

### DELIVERABLES

1. Python script: `intraday_dm_scanner.py`
2. New Supabase table: `dm_intraday`
3. Cron schedule: 10:30, 13:00, 15:30 ET on trading days
4. First run: Monday March 9, 10:30 AM ET, 18 marketplace tickers

### POLYGON API KEY

Already configured in the environment from the February price fix. 
Same key, same endpoint, same authentication.
