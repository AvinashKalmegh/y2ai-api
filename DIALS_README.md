# Y2AI Dials Module

Market condition indicators and regime detection system for the Y2AI/ARGUS-1 platform.

## Overview

The Dials module provides 20 market condition calculators organized into 7 phases:

| Phase | Modules | Purpose |
|-------|---------|---------|
| 1. Foundation | PillarIndex, VixDial, CreditSpreadDial | Price-based indicators |
| 2. Core | BreadthDial, MCI, MacroDial, SignalsDial | Market breadth & conditions |
| 3. Additional | ClusterDial, LiquidityDial, LaborDial, ETFDial, SentimentDial | Extended indicators |
| 4. Flow | StockFlowDial, FlowDivergence | Volume/flow analysis |
| 5. Multipliers | MacroMultipliers | Position sizing |
| 6. Aggregation | RegimeArbiter, PortfolioTracker | Unified regime & portfolio |
| 7. Output | Dashboard, MorningBrief | Reports & alerts |

## Quick Start

```bash
# Run all dials (after market close)
python -m y2ai.orchestrator --dials

# Or run directly
python dials_runner.py --all

# Run specific phase
python dials_runner.py --phase 1

# Run dashboard only
python dials_runner.py --dashboard
```

## Installation

```bash
# Required packages
pip install pandas numpy yfinance scipy
pip install supabase httpx
pip install fredapi  # For FRED data

# Set environment variables
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-key"
export FRED_API_KEY="your-fred-api-key"
```

## Module Reference

### Foundation Dials

#### PillarIndex (`dials/pillar_index.py`)
Calculates cumulative returns and momentum for 6 pillars (43 stocks).

**Pillars:**
- Infrastructure & Energy (16 stocks)
- Enterprise Adoption (13 stocks)
- Productivity & Labor (3 stocks)
- Demand Dynamics (3 stocks)
- Macro & Policy (4 stocks)
- Financial & Market (4 stocks)

**Outputs:**
- Cumulative index per pillar
- 5D, 1M, 3M momentum
- Pillar signals (LEADING, NEUTRAL, WEAKENING, LAGGING)

```python
from dials import PillarIndexCalculator
calc = PillarIndexCalculator()
data = calc.calculate()
print(data.pillar_signals)
```

#### VixDial (`dials/vix_dial.py`)
VIX analysis with Bollinger Bands and pre-shock detection.

**Regimes:**
- Healthy: VIX < 15
- Caution: VIX 15-20
- Fragile: VIX 20-30
- Crisis: VIX > 30
- Pre-Shock: Low VIX + compressed bands (warning signal)

**Outputs:**
- VIX level, 20D average, trend
- Bollinger Band position
- Combined regime assessment

#### CreditSpreadDial (`dials/credit_spread_dial.py`)
High-yield and investment-grade credit spreads from FRED.

**Data sources:**
- BAMLH0A0HYM2: HY OAS
- BAMLC0A0CM: IG OAS

**Regimes:**
- Healthy: Spreads normal, tightening
- Caution: Spreads widening slightly
- Fragile: Spreads elevated
- Crisis: Spreads blowing out

### Core Dials

#### BreadthDial (`dials/breadth_dial.py`)
Market breadth across the 43-stock universe.

**Metrics:**
- Daily breadth (% advancing)
- 5D, 20D, 50D moving averages
- Pillar-level breadth breakdown

**Thresholds:**
- Healthy: > 60% above 20D MA
- Caution: 40-60%
- Fragile: 20-40%
- Stressed: < 20%

#### MCI (`dials/mci.py`)
Market Condition Index (-100 to +100).

**Components:**
- Pillar momentum (40%)
- Breadth (25%)
- VIX trend (20%)
- Credit trend (15%)

**Regimes:**
- Melt-Up: > 40
- Extension: 20 to 40
- Knife Edge: -20 to 20
- Collapse Bias: -40 to -20
- Break Path: < -40

#### MacroDial (`dials/macro_dial.py`)
FRED macro indicators.

**Indicators:**
- GDP growth
- Inflation rate
- Fed funds rate
- Unemployment
- PMI (Manufacturing & Services)

#### SignalsDial (`dials/signals_dial.py`)
Infrastructure thesis signals with VETO detection.

**Signal categories:**
- Capex signals
- Energy demand signals
- Compute demand signals

**VETO triggers:**
- Major earnings miss
- Guidance cut
- Demand destruction evidence

### Additional Dials

#### ClusterDial (`dials/cluster_dial.py`)
Measures stock correlation clustering (herding behavior).

**Algorithm:**
- 60-day correlation matrix
- Hierarchical clustering
- Count resulting clusters

**Regimes:**
- Healthy: ≥10 clusters (diversified)
- Caution: 7-9 clusters
- Fragile: 4-6 clusters
- Crisis: <4 clusters (extreme herding)

#### LiquidityDial (`dials/liquidity_dial.py`)
Vol-of-Vol based liquidity assessment.

**Metric:** 20-day standard deviation of VIX daily changes

**Interpretation:**
- High vol-of-vol = VIX moving erratically = market makers retreating

#### ETFDial (`dials/etf_dial.py`)
Tracks institutional money flows via ETF volume.

**ETF Universe:**
- SMH, SOXX, XLU (Infrastructure)
- IGV, WCLD (Enterprise)
- XLF (Financial)
- TLT, GLD (Macro)
- BOTZ (Productivity)
- XLY (Demand)
- SPY, QQQ (Benchmarks)

#### LaborDial (`dials/labor_dial.py`)
BLS labor market indicators for recession warning.

**FRED Series:**
- ICSA: Initial jobless claims
- CCSA: Continuing claims
- UNRATE: Unemployment rate
- PAYEMS: Nonfarm payrolls

#### SentimentDial (`dials/sentiment_dial.py`)
Aggregates news sentiment from ARGUS-1 processed articles.

**Metrics:**
- Sentiment score (-100 to +100)
- 5-day trend
- Bull/bear ratio
- Pillar-level sentiment

### Flow Dials

#### StockFlowDial (`dials/stock_flow_dial.py`)
Accumulation/distribution tracking via volume analysis.

**Flow regimes:**
- Strong Accumulation: Volume >1.5x avg + price up
- Accumulation: Volume >1.2x avg + price up
- Distribution: Volume >1.2x avg + price down
- Strong Distribution: Volume >1.5x avg + price down

#### FlowDivergence (`dials/flow_divergence.py`)
Compares ETF flows vs stock flows.

**Divergence signals:**
- ETF ↑ Stock ↓ = Distribution into Strength (bearish)
- ETF ↓ Stock ↑ = Accumulation into Weakness (bullish)
- Both aligned = Confirmation

### Multipliers

#### MacroMultipliers (`dials/macro_multipliers.py`)
Position sizing multiplier from multiple regimes.

**Formula:**
```
Final_Weight = Base × Corr × VIX × Credit × Breadth × Labor
```

**Floor:** 30% minimum multiplier

### Aggregation

#### RegimeArbiter (`portfolio/regime_arbiter.py`)
Unified regime determination from all signals.

**Regimes:**
- NORMAL: Business as usual
- CAUTION: Monitor closely
- ELEVATED: Reduce risk
- FRAGILE: Defensive posture
- BREAK: Maximum defense
- VETO_ALERT: Emergency protocol

**Override conditions:**
- VIX > 35
- Credit spreads blow out
- 3+ fragility conditions
- VETO trigger active

#### PortfolioTracker (`portfolio/portfolio_tracker.py`)
Shadow portfolio management.

**Features:**
- NAV tracking
- Position sizing by regime
- Pillar weight allocation
- Benchmark comparison (SPY)

### Output

#### Dashboard (`dials/dashboard.py`)
Aggregated view of all indicators.

**Sections:**
- Regime summary
- Key metrics (AMRI, Bubble Index, MCI, VIX)
- Signal counts
- Pillar status
- Alerts
- Recommendations

#### MorningBrief (`dials/morning_brief.py`)
Daily market brief for subscribers.

**Sections:**
- Headline
- Regime status
- Key signals
- Risk assessment
- Action items

## Daily Schedule

```
4:30 PM ET  →  run_daily_indicators()  (Bubble Index, Stock Tracker)
4:35 PM ET  →  run_dials()             (All 19 dial modules)
4:45 PM ET  →  run_daily_social_post() (Social media)
```

## API Endpoints

Start the API:
```bash
uvicorn dials_api:app --port 8001
```

**Endpoints:**
- `GET /dashboard` - Current dashboard state
- `GET /brief` - Morning brief
- `GET /regime` - Regime status
- `GET /dials` - All dial readings
- `GET /dials/{name}` - Specific dial
- `GET /history/{name}?days=30` - Historical data
- `GET /pillars` - Pillar status
- `GET /portfolio` - Shadow portfolio
- `GET /alerts` - Active alerts

## Supabase Tables

```sql
-- Core
pillar_index_daily, breadth_daily, mci_daily
vix_dial_daily, vix_history
credit_spread_daily, credit_spread_history
signals_dial_daily, macro_dial_daily

-- Additional
cluster_dial_daily, liquidity_dial_daily
etf_dial_daily, labor_dial_daily
sentiment_dial_daily

-- Flow
stock_flow_dial_daily, flow_divergence_daily
macro_multipliers_daily

-- Aggregation
regime_arbiter_daily, portfolio_nav_daily

-- Output
dashboard_daily, morning_brief_daily
```

## Backfill

```bash
# Today only
python dials_backfill.py --today

# Last 30 days
python dials_backfill.py --days 30

# Specific date range
python dials_backfill.py --start 2024-12-01 --end 2024-12-31

# Specific modules only
python dials_backfill.py --days 7 --module vix --module credit
```

## File Structure

```
y2ai/
├── dials/
│   ├── __init__.py
│   ├── pillar_index.py
│   ├── breadth_dial.py
│   ├── mci.py
│   ├── vix_dial.py
│   ├── credit_spread_dial.py
│   ├── signals_dial.py
│   ├── macro_dial.py
│   ├── cluster_dial.py
│   ├── liquidity_dial.py
│   ├── etf_dial.py
│   ├── labor_dial.py
│   ├── sentiment_dial.py
│   ├── stock_flow_dial.py
│   ├── flow_divergence.py
│   ├── macro_multipliers.py
│   ├── dashboard.py
│   └── morning_brief.py
├── portfolio/
│   ├── __init__.py
│   ├── regime_arbiter.py
│   └── portfolio_tracker.py
├── dials_runner.py
├── dials_backfill.py
├── dials_api.py
└── orchestrator.py
```

## Environment Variables

```bash
# Required
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-anon-key

# Optional but recommended
FRED_API_KEY=your-fred-api-key
ANTHROPIC_API_KEY=your-claude-key  # For newsletter
```

## Regime Decision Tree

```
1. Check VETO triggers → VETO_ALERT
2. Check VIX > 35 → BREAK
3. Check credit blow-out → BREAK
4. Count fragility conditions:
   - VIX > 25
   - Credit widening > 50bps
   - Breadth < 30%
   - MCI < -30
   - 3+ bearish signals
   
   If ≥3 conditions → FRAGILE
   If ≥2 conditions → ELEVATED
   If ≥1 condition → CAUTION
   Else → NORMAL
```

## Signal Interpretation

| MCI Range | Regime | Action |
|-----------|--------|--------|
| > 40 | Melt-Up | Ride momentum, watch for reversal |
| 20-40 | Extension | Stay long leaders |
| -20 to 20 | Knife Edge | No new positions |
| -40 to -20 | Collapse Bias | Reduce exposure |
| < -40 | Break Path | Maximum defense |

## Contributing

1. All calculators follow the same pattern:
   - `__init__()` - Set up Supabase client
   - `calculate()` - Return dataclass
   - `save_to_supabase()` - Persist results

2. Add new dials to `dials/__init__.py`
3. Add to `dials_runner.py` in appropriate phase
4. Create Supabase table
5. Update API if needed

## License

Proprietary - Y2AI Research
