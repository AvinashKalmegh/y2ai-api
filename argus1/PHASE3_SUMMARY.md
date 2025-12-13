# ARGUS+Y2AI Phase 3: Data Validation

## Summary

Phase 3 added a comprehensive data validation layer to catch invalid, malformed, or anomalous data before it enters calculations or storage. The validation module sits between API responses and Supabase, acting as a gatekeeper for data quality.

## What Was Built

### 1. Validation Module (`shared/validation.py`)

A centralized validation infrastructure providing:

**Result Types**
- `ValidationResult`: Container for validation outcomes with issues, metadata, and severity tracking
- `ValidationIssue`: Individual issue with field, message, severity, and actual/expected values
- `ValidationSeverity`: Enum with INFO, WARNING, ERROR, CRITICAL levels

**Range Validators**
- Validates numeric values against defined min/max bounds
- Pre-defined constraints for all Y2AI fields (VIX, CAPE, credit spreads, etc.)
- Anomaly thresholds for flagging unusual but valid values
- NaN and infinity detection
- Automatic type conversion for string numerics
- Clamping utility for correcting out-of-range values

**Schema Validators**
- Validates data structures against schema definitions
- Required/optional field checking
- Type validation with multiple allowed types
- Enum value validation (case-insensitive)
- Pre-built schemas for:
  - NewsAPI article responses
  - Alpha Vantage news feed
  - FRED observation data
  - yfinance quote data
  - Processed articles (Claude output)
  - Bubble index readings
  - Stock reports

**Freshness Validators**
- Validates data timestamps against age thresholds
- Pre-configured thresholds for different data types:
  - Market data: 1 hour
  - News articles: 72 hours
  - Bubble index: 24 hours
  - Credit spreads: 48 hours
  - CAPE: 168 hours (weekly)
- Future timestamp detection
- Multiple timestamp format parsing

**Anomaly Detection**
- Statistical z-score calculation against historical baselines
- Historical min/max extreme detection
- Sudden change detection between consecutive values
- Pre-built baselines for VIX, CAPE, credit spreads, stock changes

**Pre-Storage Validation**
- Validates data before Supabase insertion
- Table-specific schemas with required fields
- Max length enforcement with automatic truncation
- JSON sanitization (NaN → null, datetime → ISO string)
- Batch validation with filtering

### 2. Test Suite (`tests/test_validation.py`)

83 test cases covering:

| Category | Tests | Coverage |
|----------|-------|----------|
| ValidationResult | 6 | State management, severity handling |
| RangeValidator | 16 | Bounds, nulls, NaN, type conversion |
| SchemaValidator | 9 | Required fields, types, enums |
| NewsAPI Validation | 3 | Response structure, article filtering |
| Alpha Vantage | 3 | Rate limits, error messages |
| FRED Validation | 2 | Observation filtering |
| Bubble Reading | 3 | Schema + range integration |
| FreshnessValidator | 7 | Age checking, format parsing |
| AnomalyDetector | 8 | Z-scores, extremes, sudden changes |
| StorageValidator | 7 | Pre-insert validation, sanitization |
| Validation Decorator | 3 | Decorator behavior |
| Convenience Functions | 4 | Batch validation helpers |
| Edge Cases | 6 | Boundary conditions |
| Integration | 2 | Full validation flows |

## Range Constraints Defined

| Field | Min | Max | Description |
|-------|-----|-----|-------------|
| vix | 0 | 150 | VIX volatility index |
| cape | 5 | 70 | Shiller CAPE ratio |
| credit_spread_ig | 20 | 1000 | IG spread (bps) |
| credit_spread_hy | 100 | 3000 | HY spread (bps) |
| bubble_index | 0 | 100 | Y2AI Bubble Index |
| bifurcation_score | -2 | +2 | Bifurcation score |
| impact_score | 0 | 1 | Article impact |
| stock_price | 0.01 | 100,000 | Stock price |
| stock_change_pct | -50 | +100 | Daily change % |

## Anomaly Thresholds

Values outside these ranges trigger warnings (but remain valid):

| Field | Warning Low | Warning High |
|-------|-------------|--------------|
| vix | < 8 | > 60 |
| cape | < 12 | > 50 |
| credit_spread_ig | < 40 | > 400 |
| credit_spread_hy | < 200 | > 1200 |
| stock_change_pct | < -10% | > +15% |

## File Structure

```
argus_y2ai/
├── shared/
│   ├── __init__.py         # Updated with validation exports
│   ├── resilience.py       # Phase 1 resilience (unchanged)
│   └── validation.py       # NEW: Validation infrastructure
├── tests/
│   ├── test_validation.py  # NEW: 83 validation tests
│   └── ... (Phase 2 tests unchanged)
```

## Usage Examples

### Validating Market Indicators

```python
from shared.validation import validate_market_indicators

result = validate_market_indicators(
    vix=18.5,
    cape=32.0,
    credit_ig=100.0,
    credit_hy=400.0,
)

if not result.valid:
    print("Invalid market data!")
    for issue in result.issues:
        print(f"  {issue.field}: {issue.message}")
elif result.has_warnings:
    print("Data valid but flagged:")
    for issue in result.issues:
        if issue.severity == ValidationSeverity.WARNING:
            print(f"  ⚠ {issue.field}: {issue.message}")
```

### Validating API Responses

```python
from shared.validation import SchemaValidator

# NewsAPI response
response = requests.get("https://newsapi.org/v2/everything?q=AI").json()
result = SchemaValidator.validate_newsapi_response(response)

if result.valid:
    valid_articles = result.data["articles"]
    print(f"Got {len(valid_articles)} valid articles")
else:
    print(f"NewsAPI error: {result.issues[0].message}")
```

### Pre-Storage Validation

```python
from shared.validation import StorageValidator

# Validate before Supabase insert
record = {
    "date": "2025-01-15",
    "vix": 22.5,
    "cape": 35.0,
    "bubble_index": 70.0,
    "bifurcation_score": 0.77,
    "regime": "INFRASTRUCTURE",
}

result = StorageValidator.validate_for_storage(record, "bubble_readings")

if result.valid:
    supabase.table("bubble_readings").insert(result.data).execute()
else:
    logger.error(f"Validation failed: {result.issues}")
```

### Anomaly Detection

```python
from shared.validation import AnomalyDetector

# Check if VIX value is anomalous
result = AnomalyDetector.detect_anomaly(45.0, "vix")

if result.metadata.get("is_anomalous"):
    zscore = result.metadata["zscore"]
    print(f"VIX anomaly detected! Z-score: {zscore:.2f}")

# Detect sudden changes
change_result = AnomalyDetector.detect_sudden_change(
    current=45.0,
    previous=20.0,
    field_name="vix",
    max_change_pct=50.0
)

if change_result.has_warnings:
    print(f"Sudden VIX spike: {change_result.metadata['change_pct']:.1f}%")
```

### Using the Decorator

```python
from shared.validation import validate_input, BUBBLE_INDEX_READING_SCHEMA

@validate_input(
    schema=BUBBLE_INDEX_READING_SCHEMA,
    range_fields=["vix", "cape", "bubble_index"],
    raise_on_error=True,
)
def store_bubble_reading(reading):
    """This will validate before execution"""
    supabase.table("bubble_readings").insert(reading).execute()
```

## Key Design Decisions

1. **Warnings vs Errors**: Invalid data fails validation (ERROR), but unusual data that might still be correct triggers warnings. A VIX of 60 is unusual but happened in March 2020—we flag it but don't reject it.

2. **Fail-Safe Storage**: StorageValidator sanitizes data (NaN → null, truncates long strings) to ensure Supabase inserts succeed even with messy data.

3. **Historical Context**: Anomaly detection uses historical baselines (min/max seen, statistical distributions) rather than arbitrary thresholds.

4. **Schema Flexibility**: Schemas allow extra fields and are case-insensitive for enums, matching the reality of messy API responses.

5. **Batch Operations**: `batch_validate` filters invalid records while preserving valid ones, following the Phase 1 "fail open for data collection" philosophy.

## Integration Points

The validation layer integrates with existing Phase 1 components:

- **Before Resilience**: Validate API responses after successful fetch but before processing
- **After Processing**: Validate Claude outputs before storage
- **Health Tracking**: Validation failures can feed into ServiceHealth metrics
- **Graceful Degradation**: Warnings don't block operations; only errors do

## Next Steps (Phase 4: Monitoring & Alerting)

Potential Phase 4 additions:
1. Prometheus metrics for validation failure rates
2. Slack/email alerts for anomalies
3. Dashboard for data quality trends
4. Automatic fallback triggering when quality degrades
5. Historical validation for backfill operations

## Running Tests

```bash
# Run validation tests only
python -m pytest tests/test_validation.py -v

# Run all tests (Phase 2 + Phase 3)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=shared --cov-report=html
```

## Dependencies

No new dependencies required. Uses standard library:
- `datetime` for timestamp handling
- `dataclasses` for result types
- `re` for pattern matching
- `math` for NaN/infinity checks
