# NST Integration — NarrativeDensity for FlowOS

Extends the Google Alerts pipeline to feed two FlowOS components:

1. **NarrativeDensity** — 30% weight in Attractor Mass formula
2. **Exit Rule Catalyst Detection** — modifies exit signals based on news density

## Quick Start

### 1. Create the Database Tables

Run `nst_schema.sql` in your Supabase SQL editor:

```sql
-- In Supabase SQL Editor, paste contents of nst_schema.sql
```

This creates:
- `nst_mentions` — individual classified mentions
- `nst_daily_summary` — aggregated daily stats per bubble type

### 2. Process Alerts Through NST

**Option A: Process during ingest (recommended)**

Add this to your collection pipeline after fetching Google Alerts:

```python
from argus1 import process_alert_for_nst, infer_bubble_type_from_alert

# In your collection loop:
for alert in raw_articles:
    # Infer bubble type from content
    bubble_type = infer_bubble_type_from_alert(
        alert_name=alert.source_name,
        headline=alert.title,
        snippet=alert.content
    )
    
    if bubble_type:
        # Classify and store
        mention = process_alert_for_nst(
            alert=alert.to_dict(),
            bubble_type=bubble_type,
            use_claude=True  # Set False to skip sentiment analysis
        )
```

**Option B: Batch process existing articles**

```python
from argus1.nst import NSTStorage, classify_mention, run_daily_aggregation

storage = NSTStorage()

# Classify existing articles
for article in existing_articles:
    mention = classify_mention(
        headline=article['title'],
        snippet=article['content'],
        source=article['source_name'],
        url=article['url'],
        published_at=article['published_at'],
        bubble_type='AI/Compute'  # or infer from content
    )
    storage.save_mention(mention)

# Run daily aggregation
run_daily_aggregation()
```

### 3. Query for FlowOS

**Get NarrativeDensity for Attractor Mass:**

```python
from argus1 import get_narrative_density

result = get_narrative_density('AI/Compute', days=7)

print(f"Status: {result.status}")
print(f"Density: {result.narrative_density}")
print(f"Scaled (0-100): {result.scaled}")  # Use this in Attractor Mass formula
print(f"Total mentions: {result.total_mentions}")
print(f"Attractor mentions: {result.attractor_mentions}")
```

**Get NST Status for Exit Rules:**

```python
from argus1 import get_nst_status

status = get_nst_status('AI/Compute')

print(f"Density Level: {status.density_level}")  # LOW, MODERATE, HIGH, EXTREME
print(f"Catalyst Pending: {status.catalyst_pending}")
print(f"Catalyst Type: {status.catalyst_type}")  # POSITIVE_PENDING or NEGATIVE
```

## Integration with Scheduler

To automatically process NST during collection runs, add this to `scheduler.py` after Phase 3:

```python
# Phase 4b: NST Processing
from .nst import process_alert_for_nst, infer_bubble_type_from_alert, run_daily_aggregation

logger.info("Processing alerts for NST...")
for article in raw_articles:
    bubble_type = infer_bubble_type_from_alert(
        article.source_name, article.title, article.content
    )
    if bubble_type:
        process_alert_for_nst(
            alert=article.to_dict(),
            bubble_type=bubble_type,
            use_claude=False  # Skip Claude for speed; run separately for important ones
        )

# Run daily aggregation
run_daily_aggregation()
result["nst_processed"] = True
```

## Bubble Types

Currently tracking four bubble types:

| Bubble Type | Attractor | Ticker |
|-------------|-----------|--------|
| AI/Compute | NVIDIA | NVDA |
| Energy/Grid | Constellation Energy | CEG |
| Crypto | Coinbase | COIN |
| Clean Energy | Enphase | ENPH |

## Keyword Categories

Each bubble type has four keyword categories:

- **attractor**: Keywords that identify mentions of the attractor stock
- **theme**: General theme keywords (not attractor-specific)
- **stress**: Negative/bearish keywords
- **catalyst**: Event keywords that could move the stock

See `NST_KEYWORDS` in `nst.py` for the full list.

## API Endpoints (if using FastAPI)

Add these to your API:

```python
from fastapi import FastAPI, Query
from argus1 import get_narrative_density, get_nst_status

app = FastAPI()

@app.get("/nst/narrative-density")
def narrative_density_endpoint(
    bubble_type: str = Query(...),
    days: int = Query(default=7)
):
    result = get_narrative_density(bubble_type, days)
    return {
        "status": result.status,
        "narrative_density": result.narrative_density,
        "scaled": result.scaled,
        "total_mentions": result.total_mentions,
        "attractor_mentions": result.attractor_mentions
    }

@app.get("/nst/status")
def nst_status_endpoint(bubble_type: str = Query(...)):
    result = get_nst_status(bubble_type)
    return {
        "density_level": result.density_level,
        "total_mentions": result.total_mentions,
        "catalyst_pending": result.catalyst_pending,
        "catalyst_type": result.catalyst_type
    }
```

## CLI Usage

```bash
# Run daily aggregation
python -m argus1.nst aggregate

# Get status for a bubble
python -m argus1.nst status --bubble "AI/Compute"

# Get narrative density
python -m argus1.nst density --bubble "AI/Compute" --days 7
```

## How NarrativeDensity Works

NarrativeDensity measures what percentage of theme-related news mentions the attractor stock specifically:

```
NarrativeDensity = attractor_mentions / total_mentions
```

For AI/Compute:
- If 10 articles mention "AI infrastructure" but only 3 mention NVIDIA → density = 0.30
- If 10 articles and 8 mention NVIDIA → density = 0.80

High density suggests the attractor is dominating the narrative, which feeds into the Attractor Mass calculation.

## Density Levels

Based on total daily mentions:

| Level | Mentions | Interpretation |
|-------|----------|----------------|
| LOW | 0-2 | Quiet news day |
| MODERATE | 3-5 | Normal activity |
| HIGH | 6-10 | Elevated coverage |
| EXTREME | 10+ | Crisis-level attention |

## Files Created

```
argus1/
├── nst.py           # Main NST module
├── nst_schema.sql   # Database schema
└── NST_README.md    # This file
```

## Next Steps

1. Run `nst_schema.sql` in Supabase
2. Add NST processing to your scheduler
3. Wire up API endpoints for FlowOS
4. Test with `python -m argus1.nst status --bubble "AI/Compute"`
