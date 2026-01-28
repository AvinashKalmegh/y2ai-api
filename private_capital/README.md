# Private Capital Tracker

Tracks AI venture capital and private equity funding as a leading indicator for the ARGUS-1 Infrastructure pillar.

## Overview

When private capital surges into AI while public sentiment is weak, that's institutional money voting differently than retail - a classic Y2AI signal that helps distinguish infrastructure cycles from speculative bubbles.

## Data Sources (All Free)

- **RSS Feeds**: TechCrunch, Crunchbase News, VentureBeat, Reuters
- **Google Alerts**: 7 configured alerts for AI funding news (optional)

## Installation

```bash
pip install feedparser requests
```

## Usage

### Command Line

```bash
# Test RSS collection
python -m private_capital.private_capital --test-rss

# Run full update (requires Supabase)
python -m private_capital.private_capital --update

# Print Supabase schema
python -m private_capital.private_capital --print-schema
```

### Python API

```python
from private_capital import PrivateCapitalTracker, update_private_capital

# Direct use
tracker = PrivateCapitalTracker(supabase_client=client)
result = tracker.update_dial()

print(f"Intensity: {result['intensity'].score} ({result['intensity'].regime})")
print(f"30D Volume: ${result['intensity'].vol_30d_m/1000:.1f}B")
print(f"Megarounds: {result['intensity'].megarounds_30d}")

# Workflow integration (for orchestrator)
result = update_private_capital(supabase_client=client)
```

### Bubble Index Cross-Reference

```python
from private_capital import get_bubble_index_interpretation

interpretation = get_bubble_index_interpretation(
    intensity_score=75,
    intensity_regime='STRONG',
    bubble_index=30
)

print(f"Signal: {interpretation['signal']}")  # EARLY_CYCLE
print(f"Infra Bias: {interpretation['infra_bias']:+.2f}")  # +0.50
```

## Output

### Intensity Score (0-100)

- **SURGE (80+)**: Exceptional capital formation
- **STRONG (60-79)**: Above-average activity
- **NORMAL (40-59)**: Baseline activity
- **WEAK (20-39)**: Below-average
- **FROZEN (<20)**: Capital markets closed

### Calculation Formula

- 40% - 30-day volume vs baseline
- 30% - Megaround concentration (>$500M deals)
- 20% - Category breadth
- 10% - Momentum vs prior 30 days

## Supabase Tables

Run `--print-schema` to get the SQL for creating:

- `private_capital_entries` - Raw funding announcements
- `private_capital_daily` - Aggregated daily metrics

## Integration with Orchestrator

Add to Monday workflow in `orchestrator.py`:

```python
from private_capital import run_monday_private_capital

def run_monday_part2():
    # ... existing code ...
    
    if datetime.now().weekday() == 0:  # Monday
        pc_result = run_monday_private_capital(
            supabase_client=self.supabase,
            bubble_index=current_bubble_index
        )
```

## Google Alerts Setup (Optional)

1. Go to google.com/alerts
2. Create alerts with these queries:
   - `"AI startup" AND ("raised" OR "funding round" OR "Series")`
   - `"artificial intelligence" AND ("Series C" OR "Series D" OR "Series E")`
   - `"AI" AND ("down round" OR "valuation cut")`
3. Click "Show options" → Change delivery to RSS
4. Add RSS URLs to the config

## Categories

- **foundation_model**: OpenAI, Anthropic, Mistral, etc.
- **infrastructure**: Data centers, cloud, CoreWeave
- **chips**: NVIDIA, Groq, Cerebras
- **application**: Cursor, Runway, enterprise AI
- **energy**: Nuclear, grid, power
