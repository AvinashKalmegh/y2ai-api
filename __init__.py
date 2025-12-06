from dotenv import load_dotenv
load_dotenv()



"""
Y2AI COMPLETE SYSTEM ARCHITECTURE
Replaces Google Sheets + Google Alerts with self-hosted infrastructure

=============================================================================
SYSTEM OVERVIEW
=============================================================================

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
│  │  Tables:                                                             │    │
│  │  • raw_articles, processed_articles (from ARGUS-1)                   │    │
│  │  • bubble_index_daily (VIX, CAPE, Credit Spreads)                    │    │
│  │  • stock_tracker_daily (18 stocks, 3 pillars)                        │    │
│  │  • newsletter_editions (published editions)                          │    │
│  │  • social_posts (scheduled and published posts)                      │    │
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
│  │  • Twitter/X API                                                     │    │
│  │  • LinkedIn API                                                      │    │
│  │  • Bluesky AT Protocol                                               │    │
│  │  • Scheduling & queue management                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                                 ↓                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Y2AI WEBSITE (y2ai.us)                            │    │
│  │  • Three dials (live from Supabase, not Google Sheets)               │    │
│  │  • Evidence grid (dynamic based on regime)                           │    │
│  │  • Newsletter archive                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

=============================================================================
DATA FLOW
=============================================================================

1. ARGUS-1 collects news every 2-8 hours (tiered by source priority)
   → Stores in raw_articles, processed_articles tables

2. BUBBLE INDEX SERVICE runs daily at 4:30 PM ET (after market close)
   → Fetches VIX from CBOE
   → Fetches CAPE from Shiller data
   → Fetches credit spreads from FRED
   → Calculates unified bifurcation score
   → Stores in bubble_index_daily table

3. STOCK TRACKER SERVICE runs daily at 4:30 PM ET
   → Fetches prices for 18 stocks via Yahoo Finance API
   → Calculates pillar performance (Supply, Capital, Demand)
   → Calculates Y2AI Index vs benchmarks (SPY, QQQ, Mag 7)
   → Stores in stock_tracker_daily table

4. NEWSLETTER GENERATOR runs Sunday evening
   → Queries week's processed articles from ARGUS-1
   → Queries latest bubble index and stock performance
   → Claude generates newsletter draft
   → Human reviews and approves
   → Stores in newsletter_editions table

5. SOCIAL PUBLISHER runs on schedule
   → Daily: Stock tracker update (4:45 PM ET)
   → Monday: Newsletter announcement (8:30 AM ET)
   → Posts to Twitter, LinkedIn, Bluesky

6. WEBSITE pulls live data
   → Bubble index dials from Supabase (not Google Sheets)
   → Evidence grid populated from processed_articles
   → Narrative changes based on current regime

=============================================================================
WHAT EACH SERVICE REPLACES
=============================================================================

| Component           | BEFORE (Google)        | AFTER (Y2AI)           |
|---------------------|------------------------|------------------------|
| News Collection     | Google Alerts → Gmail  | ARGUS-1 → Supabase     |
| Bubble Index        | Google Sheets formulas | BubbleIndexService     |
| Stock Tracker       | GOOGLEFINANCE()        | StockTrackerService    |
| Data Storage        | Google Sheets tabs     | PostgreSQL/Supabase    |
| Website Data        | Google Apps Script API | Supabase REST API      |
| Social Posting      | Manual copy/paste      | SocialPublisher        |
| Newsletter          | Manual writing         | Claude API + Review    |

=============================================================================
API KEYS REQUIRED
=============================================================================

# News Sources (ARGUS-1)
NEWSAPI_KEY=...              # Optional, $449/mo for production
ALPHAVANTAGE_KEY=...         # Free, 25 req/day
# SEC EDGAR and RSS are free, no keys needed

# Data Sources (Bubble Index & Stock Tracker)
FRED_API_KEY=...             # Free, for credit spreads
# Yahoo Finance - no API key needed for yfinance library
# CBOE VIX - public data, no key needed

# AI Processing
ANTHROPIC_API_KEY=...        # Claude API for extraction and generation

# Storage
SUPABASE_URL=...             # Your Supabase project URL
SUPABASE_KEY=...             # Supabase anon or service key

# Social Publishing
TWITTER_API_KEY=...          # Twitter/X API v2
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_SECRET=...

LINKEDIN_ACCESS_TOKEN=...    # LinkedIn API (OAuth flow required)

BLUESKY_HANDLE=...           # e.g., y2ai.bsky.social
BLUESKY_APP_PASSWORD=...     # Generate in Bluesky settings

=============================================================================
"""

__version__ = "2.0.0"
__author__ = "Y2AI"
