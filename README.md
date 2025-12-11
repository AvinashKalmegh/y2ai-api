# Y2AI Complete System

Replaces Google Sheets and Google Alerts with a self-hosted infrastructure for market regime detection and newsletter automation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Y2AI COMPLETE SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    ARGUS-1: NEWS INTELLIGENCE                        │    │
│  │  (Replaces Google Alerts)                                            │    │
│  │  • NewsAPI, Alpha Vantage, SEC EDGAR, RSS Feeds                      │    │
│  │  • Claude extraction & Y2AI categorization                           │    │
│  │  • FOMO signals, skepticism alerts, spending announcements           │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ↓                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SUPABASE / POSTGRESQL                             │    │
│  │  (Replaces Google Sheets)                                            │    │
│  │  Tables: raw_articles, processed_articles, bubble_index_daily,       │    │
│  │          stock_tracker_daily, newsletter_editions, social_posts      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│         ┌───────────────────────┼───────────────────────┐                    │
│         │                       │                       │                    │
│         ↓                       ↓                       ↓                    │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────────────┐      │
│  │   BUBBLE    │        │   STOCK     │        │   NEWSLETTER        │      │
│  │   INDEX     │        │   TRACKER   │        │   GENERATOR         │      │
│  │   SERVICE   │        │   SERVICE   │        │   SERVICE           │      │
│  │             │        │             │        │                     │      │
│  │ • VIX fetch │        │ • 18 stocks │        │ • Query ARGUS-1     │      │
│  │ • CAPE calc │        │ • 3 pillars │        │ • Query indicators  │      │
│  │ • Credit    │        │ • Perf calc │        │ • Claude generation │      │
│  │ • Score     │        │ • Benchmark │        │ • Format for social │      │
│  └─────────────┘        └─────────────┘        └─────────────────────┘      │
│         │                       │                       │                    │
│         └───────────────────────┼───────────────────────┘                    │
│                                 │                                            │
│                                 ↓                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SOCIAL PUBLISHER                                  │    │
│  │  (Automated posting)                                                 │    │
│  │  • Twitter/X API      • LinkedIn API      • Bluesky AT Protocol      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                                 ↓                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Y2AI WEBSITE (y2ai.us)                            │    │
│  │  • Three dials (live from Supabase, not Google Sheets)               │    │
│  │  • Evidence grid (dynamic based on regime)                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What This Replaces

| Component | BEFORE (Google) | AFTER (Y2AI) |
|-----------|-----------------|--------------|
| News Collection | Google Alerts → Gmail | ARGUS-1 → Supabase |
| Bubble Index | Google Sheets formulas | BubbleIndexService |
| Stock Tracker | GOOGLEFINANCE() | StockTrackerService |
| Data Storage | Google Sheets tabs | PostgreSQL/Supabase |
| Website Data | Google Apps Script API | Supabase REST API |
| Social Posting | Manual copy/paste | SocialPublisher |
| Newsletter | Manual writing | Claude API + Review |

## Modules

### `bubble_index.py`
Calculates the Y2AI Bubble Index using three indicators:
- VIX (volatility regime)
- CAPE (valuation extremeness)
- Credit Spreads (financial stress)

Formula: `Bifurcation = 0.6*BI - 0.2*VI - 0.2*CS`

Regimes: INFRASTRUCTURE, ADOPTION, TRANSITION, BUBBLE_WARNING

### `stock_tracker.py`
Tracks 18 stocks across three pillars:

**Supply Constraint**: TSM, ASML, VRT
**Capital Efficiency**: GOOGL, MSFT, AMZN
**Demand Depth**: NVDA, SNOW, NOW

Calculates Y2AI Index vs SPY, QQQ, Mag7.
Determines daily signal: VALIDATING, NEUTRAL, CONTRADICTING

### `social_publisher.py`
Automated posting to:
- Twitter/X (using Tweepy)
- LinkedIn (REST API)
- Bluesky (AT Protocol)

Supports single posts and threads.

### `storage.py`
Supabase/PostgreSQL integration with SQLite fallback.
Complete schema for all Y2AI data.

### `orchestrator.py`
Main scheduler coordinating all services:
- 4:30 PM ET: Daily indicators update
- 4:45 PM ET: Social media daily post
- 6am, 12pm, 6pm, 10pm ET: News collection (weekdays)
- Sunday 6pm: Newsletter generation
- Monday 8:30am: Newsletter social post

## Integration with ARGUS-1

ARGUS-1 feeds into Y2AI through the shared Supabase database:

```
ARGUS-1 collects → processed_articles table → Y2AI reads for:
  - Newsletter content (weekly)
  - Evidence grid (website)
  - Trend analysis
```

The `y2ai_category` field in processed_articles maps directly to newsletter sections:
- spending: CapEx announcements
- constraints: Supply bottlenecks
- skepticism: Bubble warnings
- smartmoney: Institutional moves
- energy: Power/infrastructure
- adoption: Enterprise AI usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY=sk-ant-...
export SUPABASE_URL=https://xxx.supabase.co
export SUPABASE_KEY=eyJ...

# For Bubble Index (free FRED API key)
export FRED_API_KEY=...

# For Social Publishing
export TWITTER_API_KEY=...
export TWITTER_API_SECRET=...
export TWITTER_ACCESS_TOKEN=...
export TWITTER_ACCESS_SECRET=...
export LINKEDIN_ACCESS_TOKEN=...
export LINKEDIN_PERSON_URN=urn:li:person:...
export BLUESKY_HANDLE=y2ai.bsky.social
export BLUESKY_APP_PASSWORD=...

# Optional (for ARGUS-1 NewsAPI)
export NEWSAPI_KEY=...
export ALPHAVANTAGE_KEY=...
```

### 3. Initialize Database

Run in Supabase SQL editor:
```sql
-- Copy contents of storage.py SCHEMA_SQL
```

Or get schema:
```bash
python -m y2ai.storage --schema
```

### 4. Run Services

**Manual runs:**
```bash
# Run daily indicators
python -m y2ai.orchestrator --indicators

# Run social post
python -m y2ai.orchestrator --social

# Run news collection
python -m y2ai.orchestrator --news

# Run everything
python -m y2ai.orchestrator --all
```

**Background scheduler:**
```bash
python -m y2ai.orchestrator --daemon
```

## Schedule Summary

| Time (ET) | Task | Frequency |
|-----------|------|-----------|
| 6:00 AM | News Collection | Weekdays |
| 10:00 AM | News Collection | Weekends |
| 12:00 PM | News Collection | Weekdays |
| 4:30 PM | Bubble Index + Stock Tracker | Weekdays |
| 4:45 PM | Daily Social Post | Weekdays |
| 6:00 PM | News Collection | Weekdays |
| 6:00 PM | Newsletter Generation | Sunday |
| 8:30 AM | Newsletter Social | Monday |
| 10:00 PM | News Collection | Weekdays |

## Website Integration

Replace your Google Apps Script API endpoint with Supabase:

```javascript
// Old: Google Apps Script
const response = await fetch('https://script.google.com/macros/s/...');

// New: Supabase REST API
const { data } = await supabase
  .from('v_latest_bubble_index')
  .select('*')
  .single();
```

The dashboard data endpoint now returns:
```json
{
  "bubble_index": {
    "value": 23,
    "vix": 19.83,
    "credit_spread": 120,
    "bifurcation_score": 0.77,
    "regime": "INFRASTRUCTURE"
  },
  "stock_tracker": {
    "y2ai_index": 0.65,
    "spy": 0.12,
    "status": "VALIDATING"
  },
  "recent_articles": [...]
}
```

## Files

```
y2ai/
├── __init__.py          # Architecture overview
├── bubble_index.py      # VIX, CAPE, Credit Spread calculations
├── stock_tracker.py     # 18-stock tracker, pillar performance
├── social_publisher.py  # Twitter, LinkedIn, Bluesky automation
├── storage.py           # Supabase schema and operations
├── orchestrator.py      # Main scheduler
└── requirements.txt     # Dependencies

argus1/                  # Sibling package
├── aggregator.py        # Multi-source news collection
├── processor.py         # Claude extraction
├── storage.py           # Shared with Y2AI
├── api.py               # FastAPI endpoints
└── scheduler.py         # News collection schedule
```

## Cost Estimates

| Service | Cost | Notes |
|---------|------|-------|
| Supabase | Free tier | Up to 500MB database |
| Claude API | ~$0.30/day | ~30 articles processed |
| NewsAPI | $449/mo | Optional, RSS is free |
| Twitter API | Free tier | Limited posts |
| LinkedIn API | Free | OAuth required |
| Bluesky | Free | Open protocol |

**Minimum cost: ~$10/month** (Claude API only)
**Full cost: ~$500/month** (with NewsAPI)

## Future Enhancements

1. **Real-time Website Updates**: WebSocket connection for live dials
2. **Mobile App**: React Native dashboard
3. **Email Newsletter**: SendGrid/Mailchimp integration
4. **Historical Charts**: Interactive regime visualization
5. **Alert System**: Slack/Discord notifications for regime changes





<!--Commands -->
python -m y2ai.orchestrator --indicators

python -m y2ai.orchestrator --social

python -m y2ai.orchestrator --news

python -m y2ai.orchestrator --newsletter

python -m y2ai.orchestrator --backfill 2025-11-22 2025-12-05

uvicorn api_processed_articles:app --reload --host 0.0.0.0 --port 8000

<!--Commands -->





✅ What These CLI Arguments Do

These command-line flags allow the orchestrator to manually trigger individual Y2AI automation workflows without waiting for the scheduled times.

1. --indicators

Runs the Daily Market Indicators Pipeline:

Computes Bubble Index (VIX, CAPE, credit spreads)

Generates Daily Stock Tracker Report (18 stocks across 3 pillars)

Stores results in Supabase

🟢 Status: Working
(As confirmed from your logs)

2. --social

Runs the Daily Social Media Posting Job:

Takes the latest stock tracker report

Formats a social-ready summary

Posts automatically to linked platforms (LinkedIn/Twitter)

🟢 Status: Working
(Dependent on stock tracker data — your environment is fine)

3. --news

Runs the ARGUS-1 News Collection Pipeline:

Pulls fresh news from RSS, AlphaVantage, SEC, NewsAPI (if key provided)

Deduplicates + stores raw articles

Sends each article to Claude for automated processing & scoring

Saves structured insights to Supabase

🟢 Status: Working
(You fixed the Supabase schema issues successfully)

4. --newsletter

Runs the Weekly Newsletter Generator:

Loads processed articles from the week

Loads bubble index context

Prepares JSON context

Sends to Claude to draft the weekly Y2AI newsletter

Saves newsletter to file

🔴 Status: Not working yet
Reason: API call to Anthropic returning 404 Not Found due to environment variable not loading in CMD (ANTHROPIC_API_KEY missing).

5. --all

Runs all daily tasks at once, in sequence:

News collection

Indicators

Social posting

Useful for debugging or forcing a full daily update run.

🟢 Status: Working (all its components are functional)