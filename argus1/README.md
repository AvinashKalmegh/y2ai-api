# ARGUS-1 News Intelligence Layer

Replaces Google Alerts + Google Sheets with direct API access to news sources, storing in PostgreSQL/Supabase.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           ARGUS-1 NEWS INTELLIGENCE LAYER                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NEWS SOURCES (Direct APIs)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ NewsAPI.org │  │ Alpha       │  │ SEC EDGAR   │         │
│  │ (General)   │  │ Vantage     │  │ (Filings)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  RSS FEEDS (Premium Sources)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Reuters     │  │ DataCenter  │  │ Power       │         │
│  │ Technology  │  │ Knowledge   │  │ Magazine    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│           ↓ All feed into unified pipeline ↓                │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │              PROCESSING PIPELINE                   │     │
│  │  • Deduplicate across sources                      │     │
│  │  • Quick relevance filter (keyword matching)       │     │
│  │  • Claude extraction & Y2AI categorization         │     │
│  │  • Store in PostgreSQL/Supabase                    │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│           ↓ Available via API ↓                             │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │              FASTAPI ENDPOINTS                     │     │
│  │  POST /collect      - Run collection               │     │
│  │  GET  /articles     - Query with filters           │     │
│  │  POST /newsletter   - Generate newsletter draft    │     │
│  │  GET  /stats        - Collection statistics        │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Y2AI Categories

Articles are classified into these categories:

| Category | Description | Example |
|----------|-------------|---------|
| spending | CapEx announcements, infrastructure investments | "Microsoft announces $80B AI investment" |
| constraints | Supply shortages, power limitations | "NVIDIA H100 backlog extends to 2025" |
| data | Earnings, revenue, margin analysis | "Google Cloud revenue up 34% YoY" |
| policy | Government regulations, export controls | "US tightens chip export controls" |
| skepticism | Bubble warnings, critical analysis | "Fund managers call AI biggest risk" |
| smartmoney | Institutional moves, hedge fund positions | "Burry takes $1.1B put position" |
| china | China AI developments, US-China competition | "Huawei AI chip production ramps" |
| energy | Power consumption, energy infrastructure | "AI data centers drive 10% demand increase" |
| adoption | Enterprise AI adoption, ROI evidence | "Enterprise AI adoption reaches 65%" |

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Supabase (production storage)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Optional news sources
NEWSAPI_KEY=...          # $449/mo for production
ALPHAVANTAGE_KEY=...     # Free, 25 req/day
```

## Quick Start

### Manual Collection

```python
from argus1 import NewsAggregator, ClaudeProcessor, get_storage

# Collect from all sources
aggregator = NewsAggregator()
articles = aggregator.collect_all(hours_back=24)

# Quick filter for relevance
processor = ClaudeProcessor()
filtered = processor.quick_relevance_filter(articles)

# Process through Claude
processed = processor.process_batch(filtered, max_batch=30)

# Store in database
storage = get_storage()
storage.insert_processed_articles([p.to_dict() for p in processed])
```

### Start API Server

```bash
uvicorn argus1.api:app --reload
```

### Scheduled Collection

```bash
# Start background scheduler
python -m argus1.scheduler

# Or use as cron job
# Weekdays at 6am, 12pm, 6pm, 10pm
0 6,12,18,22 * * 1-5 cd /path/to/argus1 && python -m argus1.scheduler --cron

# Weekends at 10am
0 10 * * 0,6 cd /path/to/argus1 && python -m argus1.scheduler --cron --weekend
```

## API Endpoints

### POST /collect
Run collection from all sources.

```bash
curl -X POST http://localhost:8000/collect \
  -H "Content-Type: application/json" \
  -d '{"hours_back": 24}'
```

### GET /articles
Query processed articles.

```bash
# Get high-impact spending articles
curl "http://localhost:8000/articles?category=spending&min_impact=0.7"

# Get bearish sentiment articles
curl "http://localhost:8000/articles?sentiment=bearish&hours_back=168"
```

### POST /newsletter/generate
Generate newsletter draft from recent articles.

```bash
curl -X POST http://localhost:8000/newsletter/generate \
  -H "Content-Type: application/json" \
  -d '{"days_back": 7}'
```

### GET /stats
Get collection statistics.

```bash
curl "http://localhost:8000/stats?days_back=30"
```

## Database Schema

The system uses two main tables:

**raw_articles**: Unprocessed articles from all sources
- article_hash (unique identifier)
- source_type, source_name
- title, url, content
- collected_at

**processed_articles**: Claude-enriched articles
- All raw fields plus:
- y2ai_category
- extracted_facts (JSON array)
- impact_score (0-1)
- sentiment (bullish/bearish/neutral)
- companies_mentioned, dollar_amounts, key_quotes

## Cost Considerations

| Source | Cost | Requests |
|--------|------|----------|
| NewsAPI | $449/mo | Unlimited |
| Alpha Vantage | Free | 25/day |
| SEC EDGAR | Free | Unlimited |
| RSS Feeds | Free | Unlimited |
| Claude API | ~$0.01/article | Per processing |

Typical daily cost (30 articles processed): ~$0.30

## Comparison to Previous System

| Feature | Google Alerts | ARGUS-1 |
|---------|--------------|---------|
| Source control | Limited | Full control |
| Categorization | Manual | Automated (Claude) |
| Storage | Google Sheets | PostgreSQL |
| Query capability | None | Full SQL + API |
| Scheduling | Google-dependent | Self-hosted |
| Cost | Free | ~$10/mo + API |
| Reliability | Google outages | Self-controlled |

## Files

```
argus1/
├── __init__.py      # Package exports
├── aggregator.py    # Multi-source collection
├── processor.py     # Claude extraction & categorization
├── storage.py       # PostgreSQL/Supabase storage
├── api.py           # FastAPI endpoints
├── scheduler.py     # Automated collection
└── requirements.txt # Dependencies
```
