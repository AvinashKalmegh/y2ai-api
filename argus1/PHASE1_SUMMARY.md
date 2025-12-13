# ARGUS+Y2AI Phase 1: Error Handling & Resilience

## Summary

Phase 1 introduced a comprehensive resilience layer across all API-calling components of the ARGUS+Y2AI system. The goal was to handle transient failures gracefully, prevent cascade failures, respect rate limits, and provide visibility into system health.

## What Was Built

### 1. Shared Resilience Module (`shared/resilience.py`)

A centralized resilience infrastructure providing:

**Circuit Breakers**
- Prevents repeated calls to failing services
- Three states: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
- Configurable failure threshold and reset timeout
- Per-service circuit breakers with global registry

**Rate Limiters**
- Token bucket implementation for API quota management
- Pre-configured limits for known services (NewsAPI, Alpha Vantage, Anthropic, FRED, etc.)
- Blocking and non-blocking acquisition modes

**Retry with Exponential Backoff**
- Configurable max retries, base delay, max delay
- Jitter to prevent thundering herd
- Retryable exception and status code configuration
- Integration with circuit breakers and rate limiters

**Health Tracking**
- Per-service success/failure counts
- Response time tracking
- Success rate calculation
- Last error tracking

**HTTP Session Management**
- Connection pooling
- Built-in urllib3 retry strategy
- Configurable timeouts

### 2. Enhanced Aggregator (`argus1/aggregator_enhanced.py`)

All news source adapters now use resilience patterns:

| Adapter | Circuit Breaker | Rate Limiter | Retries | Fallback |
|---------|-----------------|--------------|---------|----------|
| NewsAPI | ✓ | ✓ (100/day) | 3 | Skip queries |
| Alpha Vantage | ✓ | ✓ (25/day) | 2 | Skip batches |
| SEC EDGAR | ✓ | — | 3 | Skip terms |
| RSS | Per-feed cooldown | — | 2 | Skip feed |

**RSS Feed Improvements:**
- Individual feed health tracking
- 60-minute cooldown for failing feeds
- HEAD check before full parse
- Graceful degradation (continue with working feeds)

### 3. Enhanced Processor (`argus1/processor_enhanced.py`)

Claude API calls are now resilient:

- 3 retries with exponential backoff
- Circuit breaker (opens after 5 consecutive failures)
- Rate limiter (50/min)
- Consecutive failure tracking (stops batch after 3 in a row)
- JSON parsing with multiple fallback strategies:
  1. Direct parse
  2. Extract from markdown code blocks
  3. Find JSON-like structure
  4. Validate and fix common issues

### 4. Enhanced Bubble Index (`y2ai/bubble_index_enhanced.py`)

Market indicator fetching is now resilient:

| Data Source | Primary | Fallback Strategy |
|-------------|---------|-------------------|
| VIX | yfinance | Historical median (18.0) |
| CAPE | Shiller Excel | SPY P/E estimate → Historical (28.0) |
| Credit Spreads | FRED API | Historical typical values |

**Data Quality Tracking:**
- Each reading tracks which values are "live" vs "fallback"
- `data_quality_score` property (0-1)
- Warning logs when fallbacks are used

### 5. Enhanced Stock Tracker (`y2ai/stock_tracker_enhanced.py`)

Stock data fetching improvements:

- Batch yfinance downloads (efficient)
- Per-ticker failure cooldown (30 min)
- 5-minute result caching
- Failed ticker tracking in reports
- Data quality score in reports

## File Structure

```
argus_y2ai/
├── shared/
│   ├── __init__.py         # Package exports
│   └── resilience.py       # Core resilience infrastructure
├── argus1/
│   ├── aggregator_enhanced.py    # News collection with resilience
│   └── processor_enhanced.py     # Claude processing with resilience
├── y2ai/
│   ├── bubble_index_enhanced.py  # Market indicators with resilience
│   └── stock_tracker_enhanced.py # Stock tracking with resilience
└── requirements.txt
```

## Usage Examples

### Using the Resilient Call Decorator

```python
from shared.resilience import resilient_call

@resilient_call(
    service_name="my_api",
    max_retries=3,
    base_delay=1.0,
    use_circuit_breaker=True,
    use_rate_limiter=True,
)
def fetch_data():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()
```

### Checking System Health

```python
from shared.resilience import get_system_status, get_all_health_status

# Get complete system status
status = get_system_status()
print(status["circuit_breakers"])
print(status["health"])

# Get health for specific service
from shared.resilience import get_health_tracker
tracker = get_health_tracker("newsapi")
print(f"Success rate: {tracker.success_rate}%")
```

### Graceful Degradation

```python
from argus1.aggregator_enhanced import NewsAggregator

aggregator = NewsAggregator()

# This will return results from successful adapters
# even if some fail
articles = aggregator.collect_all(hours_back=24)

# Check what worked
health = aggregator.get_health_status()
for adapter_id, status in health["adapters"].items():
    available = "✓" if status["available"] else "✗"
    print(f"{available} {adapter_id}: {status['success_rate']:.1f}%")
```

## Key Design Decisions

1. **Fail Open for Data Collection**: News collection continues even if some sources fail. The system returns partial results rather than failing completely.

2. **Fail Closed for Processing**: If Claude API is consistently failing, we stop processing to avoid wasting API calls. The circuit breaker prevents runaway costs.

3. **Fallback Values for Indicators**: Bubble Index always returns a result, using historical typical values when live data isn't available. The data quality score lets consumers know how reliable the reading is.

4. **Per-Feed RSS Tracking**: Rather than a single circuit breaker for all RSS, we track each feed independently. One broken feed doesn't take down the whole RSS adapter.

5. **Batch Operations**: Stock data is fetched in batches for efficiency, but failures are tracked per-ticker to avoid re-fetching working tickers.

## Next Steps (Phase 2: Testing)

The resilience module needs unit tests for:
- Circuit breaker state transitions
- Rate limiter token refill
- Retry decorator behavior
- JSON parsing edge cases
- Fallback value application

Integration tests needed for:
- End-to-end collection pipeline
- Supabase storage operations
- Full daily indicator calculation

## Migration Path

To migrate from the original code to the enhanced versions:

1. Keep original files as fallback
2. Import from enhanced modules:
   ```python
   # Instead of:
   from argus1.aggregator import NewsAggregator
   
   # Use:
   from argus1.aggregator_enhanced import NewsAggregator
   ```
3. Add the shared package to Python path
4. Install additional requirements (if any)

The enhanced modules have the same public API as the originals, so the migration should be seamless.
