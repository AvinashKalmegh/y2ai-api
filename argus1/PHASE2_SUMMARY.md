# ARGUS+Y2AI Phase 2: Testing

## Summary

Phase 2 added comprehensive unit and integration tests for all Phase 1 components. The test suite validates resilience patterns, core calculations, data processing, and end-to-end flows.

## Test Files Created

### 1. `tests/test_resilience.py` (~400 lines)

Tests for the shared resilience infrastructure:

**Circuit Breaker Tests**
- Initial state is CLOSED
- Stays closed below failure threshold
- Opens at threshold
- Rejects calls when open
- Transitions to HALF_OPEN after timeout
- Closes on success in HALF_OPEN
- Opens again on failure in HALF_OPEN
- Manual reset works
- Global registry returns same instance

**Rate Limiter Tests**
- Initial tokens at max
- Acquire reduces tokens
- Fails when insufficient (non-blocking)
- Tokens refill over time
- Tokens don't exceed max
- Blocking acquire waits
- Blocking acquire times out
- Wait time calculation accurate

**Retry Decorator Tests**
- Success doesn't retry
- Retries on retryable exception
- Succeeds after retry
- Non-retryable exception fails immediately
- Exponential backoff timing
- Circuit breaker integration

**Health Tracking Tests**
- Initial state
- Record success/failure
- Success rate calculation
- Unhealthy on low success rate

### 2. `tests/test_calculations.py` (~350 lines)

Tests for core Y2AI formulas:

**Bubble Index Calculation**
- CAPE 15 → BI 20 (cheap)
- CAPE 25 → BI 45 (fair value)
- CAPE 35 → BI 70 (expensive)
- CAPE 45 → BI 95 (extreme)
- Clamped at 0 and 100

**Bifurcation Score Formula**
- Neutral inputs → score 0
- High BI → positive contribution
- Low BI → negative contribution
- High VIX reduces score
- High credit spread reduces score
- Combined stress signals
- Infrastructure regime values

**Z-Score Calculation**
- Value at mean → z-score 0
- One std above → z-score 1
- One std below → z-score -1
- Empty history → 0
- Zero std → 0

**Regime Determination**
- Score > 0.5 → INFRASTRUCTURE
- Score 0.2-0.5 → ADOPTION
- Score -0.2 to 0.2 → TRANSITION
- Score < -0.2 → BUBBLE_WARNING
- High VIX forces TRANSITION

**Stock Tracker Calculations**
- Pillar average calculation
- Pillar with missing stocks
- Y2AI Index (9-stock equal weight)
- Status determination (VALIDATING/NEUTRAL/CONTRADICTING)
- Boundary cases

**Data Quality**
- Bubble index tracks live vs fallback
- Stock report tracks fetch success rate

### 3. `tests/test_processor.py` (~400 lines)

Tests for JSON parsing and validation:

**JSON Extraction**
- Clean JSON
- JSON with whitespace
- JSON in markdown code block
- JSON in plain code block
- JSON with surrounding text
- JSON with trailing commas (LLM error)
- Nested objects
- Arrays
- Empty/None text
- No JSON in text
- Malformed JSON
- Multiple JSON objects (takes first)

**Validation and Fixing**
- Valid result unchanged
- Fixes uppercase category
- Fixes category with extra text
- Invalid category defaults to "data"
- Fixes uppercase sentiment
- Invalid sentiment defaults to "neutral"
- Clamps impact score (0-1)
- Converts string impact score
- Converts string to list
- Converts None to empty list

**Category Matching**
- All valid categories
- Partial matches (e.g., "cap spending" → "spending")

**Edge Cases**
- Unicode content
- Escaped quotes
- Newlines in values
- Very long content
- Empty arrays
- Numeric strings
- Boolean values
- Null values

**Integration-Style Tests**
- Typical Claude response format
- Minimal Claude response
- Verbose Claude response with extra fields

### 4. `tests/test_integration.py` (~350 lines)

Integration tests with mocked external APIs:

**News Aggregator**
- NewsAPI success response
- NewsAPI rate limit handling
- Alpha Vantage success response
- Alpha Vantage rate limit in body
- RSS adapter success
- Aggregator deduplication

**Processor Integration**
- Article categorization with mocked Claude
- Batch stops on consecutive failures

**Bubble Index Integration**
- VIX fetcher live data
- VIX fetcher fallback
- Full calculation with mocked fetchers

**Stock Tracker Integration**
- Batch download handling
- Full daily report generation

**End-to-End Flows**
- Article raw → processed flow
- Indicators → social post flow

**Storage Operations**
- Bubble reading to dict
- Stock report to dict

### 5. `tests/conftest.py` (~150 lines)

Pytest configuration and shared fixtures:

**Fixtures**
- `reset_resilience_state` (autouse)
- `sample_raw_article`
- `sample_processed_article`
- `sample_bubble_reading`
- `sample_stock_data`
- `mock_claude_response`

**Markers**
- `slow` - for slow tests
- `integration` - for integration tests
- `unit` - for unit tests

### 6. `pytest.ini`

Pytest configuration:
- Test discovery settings
- Output format
- Marker definitions
- Warning filters

### 7. `run_tests.py`

Test runner script with options:
- `--unit` - unit tests only
- `--integration` - integration tests only
- `--quick` - skip slow tests
- `--coverage` - generate coverage report
- `--verbose` - extra output
- `--file` - specific test file

## Test Coverage

| Component | Tests | Coverage Areas |
|-----------|-------|----------------|
| Circuit Breaker | 11 | State transitions, timeouts, registry |
| Rate Limiter | 8 | Token bucket, refill, blocking |
| Retry Decorator | 6 | Backoff, exceptions, integration |
| Health Tracking | 5 | Metrics, success rate |
| Bubble Index | 15 | Formula, z-scores, regimes |
| Stock Tracker | 8 | Pillars, Y2AI Index, status |
| JSON Parsing | 15 | Extraction, edge cases |
| Validation | 14 | Fixing, defaults |
| Integration | 12 | API mocks, flows |

**Total: ~90 test cases**

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python run_tests.py

# Run unit tests only
python run_tests.py --unit

# Run integration tests only
python run_tests.py --integration

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file tests/test_calculations.py

# Quick run (skip slow tests)
python run_tests.py --quick
```

## Key Test Patterns

### 1. Resilience Reset
Every test resets resilience state via `reset_all()` fixture to ensure isolation.

### 2. API Mocking
External APIs are mocked using `unittest.mock.patch`:
```python
@patch('module.requests.get')
def test_api_call(self, mock_get):
    mock_get.return_value = Mock(status_code=200, json=lambda: {...})
```

### 3. Calculation Verification
Core formulas tested with known inputs/outputs:
```python
def test_bubble_index_fair_value(self):
    # CAPE 25 → BI 45: (25-15)*2.5 + 20 = 45
    assert calc.calculate_bubble_index(25.0) == 45.0
```

### 4. Edge Case Coverage
Tests cover boundary conditions and error cases:
- Empty inputs
- Invalid values
- Rate limits
- API failures

## File Structure

```
argus_y2ai/
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Fixtures and config
│   ├── test_resilience.py    # Circuit breaker, rate limiter, retry
│   ├── test_calculations.py  # Bubble index, stock tracker formulas
│   ├── test_processor.py     # JSON parsing, validation
│   └── test_integration.py   # End-to-end flows
├── pytest.ini                # Pytest configuration
└── run_tests.py              # Test runner script
```

## Next Steps (Phase 3: Data Validation)

Phase 3 will add:
1. Input validation for all external data
2. Schema validation for API responses
3. Data quality checks before storage
4. Alerting for anomalous values
5. Historical data consistency checks

## Dependencies

Add to requirements.txt:
```
pytest>=7.4.0
pytest-cov>=4.1.0
```
