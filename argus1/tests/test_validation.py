"""
Tests for the ARGUS+Y2AI Validation Module

Test coverage:
- Range validation (valid values, out of range, null handling)
- Schema validation (required fields, types, enum values)
- API response validation (NewsAPI, Alpha Vantage, FRED)
- Freshness validation (stale data, future timestamps)
- Anomaly detection (z-scores, historical extremes, sudden changes)
- Pre-storage validation (Supabase table schemas)
- Validation decorator
"""

import pytest
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.validation import (
    # Result types
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    
    # Validators
    RangeValidator,
    SchemaValidator,
    FreshnessValidator,
    AnomalyDetector,
    StorageValidator,
    
    # Constants
    RANGE_CONSTRAINTS,
    ANOMALY_THRESHOLDS,
    
    # Schemas
    NEWSAPI_ARTICLE_SCHEMA,
    PROCESSED_ARTICLE_SCHEMA,
    BUBBLE_INDEX_READING_SCHEMA,
    
    # Decorator
    validate_input,
    
    # Convenience
    validate_market_indicators,
    validate_article_batch,
)


# =============================================================================
# VALIDATION RESULT TESTS
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult class"""
    
    def test_initial_state(self):
        """New result should be valid with no issues"""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert len(result.issues) == 0
        assert result.has_warnings is False
        assert result.has_errors is False
    
    def test_add_info_issue(self):
        """Adding INFO issue should not invalidate result"""
        result = ValidationResult(valid=True)
        result.add_issue("field", "Info message", ValidationSeverity.INFO)
        assert result.valid is True
        assert len(result.issues) == 1
    
    def test_add_warning_issue(self):
        """Adding WARNING issue should not invalidate result"""
        result = ValidationResult(valid=True)
        result.add_issue("field", "Warning message", ValidationSeverity.WARNING)
        assert result.valid is True
        assert result.has_warnings is True
        assert result.warning_count == 1
    
    def test_add_error_issue(self):
        """Adding ERROR issue should invalidate result"""
        result = ValidationResult(valid=True)
        result.add_issue("field", "Error message", ValidationSeverity.ERROR)
        assert result.valid is False
        assert result.has_errors is True
        assert result.error_count == 1
    
    def test_add_critical_issue(self):
        """Adding CRITICAL issue should invalidate result"""
        result = ValidationResult(valid=True)
        result.add_issue("field", "Critical message", ValidationSeverity.CRITICAL)
        assert result.valid is False
        assert result.error_count == 1
    
    def test_to_dict(self):
        """Result should serialize to dict correctly"""
        result = ValidationResult(valid=True)
        result.add_issue("test", "message", ValidationSeverity.WARNING)
        
        d = result.to_dict()
        assert d["valid"] is True
        assert d["warning_count"] == 1
        assert d["error_count"] == 0
        assert len(d["issues"]) == 1


# =============================================================================
# RANGE VALIDATION TESTS
# =============================================================================

class TestRangeValidator:
    """Tests for RangeValidator"""
    
    def test_vix_valid_range(self):
        """VIX within normal range should pass"""
        result = RangeValidator.validate_range(18.5, "vix")
        assert result.valid is True
        assert result.data == 18.5
    
    def test_vix_at_minimum(self):
        """VIX at minimum should pass"""
        result = RangeValidator.validate_range(0.0, "vix")
        assert result.valid is True
    
    def test_vix_at_maximum(self):
        """VIX at maximum should pass"""
        result = RangeValidator.validate_range(150.0, "vix")
        assert result.valid is True
    
    def test_vix_below_minimum(self):
        """VIX below minimum should fail"""
        result = RangeValidator.validate_range(-5.0, "vix")
        assert result.valid is False
        assert result.error_count == 1
    
    def test_vix_above_maximum(self):
        """VIX above maximum should fail"""
        result = RangeValidator.validate_range(200.0, "vix")
        assert result.valid is False
    
    def test_vix_anomaly_warning(self):
        """VIX outside normal but within hard limits should warn"""
        result = RangeValidator.validate_range(75.0, "vix")
        assert result.valid is True  # Within hard limits
        assert result.has_warnings is True  # Above anomaly threshold
    
    def test_cape_valid(self):
        """CAPE within range should pass"""
        result = RangeValidator.validate_range(28.0, "cape")
        assert result.valid is True
    
    def test_cape_too_low(self):
        """CAPE below minimum should fail"""
        result = RangeValidator.validate_range(3.0, "cape")
        assert result.valid is False
    
    def test_cape_too_high(self):
        """CAPE above maximum should fail"""
        result = RangeValidator.validate_range(100.0, "cape")
        assert result.valid is False
    
    def test_bubble_index_clamped(self):
        """Bubble index should be clamped to 0-100"""
        result = RangeValidator.validate_range(50.0, "bubble_index")
        assert result.valid is True
        assert result.data == 50.0
    
    def test_impact_score_valid(self):
        """Impact score 0-1 should pass"""
        for val in [0.0, 0.5, 1.0]:
            result = RangeValidator.validate_range(val, "impact_score")
            assert result.valid is True
    
    def test_impact_score_invalid(self):
        """Impact score outside 0-1 should fail"""
        result = RangeValidator.validate_range(1.5, "impact_score")
        assert result.valid is False
    
    def test_null_value_not_allowed(self):
        """Null value should fail when not allowed"""
        result = RangeValidator.validate_range(None, "vix")
        assert result.valid is False
    
    def test_null_value_allowed(self):
        """Null value should pass when allowed"""
        result = RangeValidator.validate_range(None, "pe_ratio")  # allow_null=True
        assert result.valid is True
    
    def test_nan_value(self):
        """NaN should fail validation"""
        result = RangeValidator.validate_range(float('nan'), "vix")
        assert result.valid is False
    
    def test_infinity_value(self):
        """Infinity should fail validation"""
        result = RangeValidator.validate_range(float('inf'), "vix")
        assert result.valid is False
    
    def test_string_convertible(self):
        """Numeric string should be converted"""
        result = RangeValidator.validate_range("25.5", "vix")
        assert result.valid is True
        assert result.data == 25.5
    
    def test_string_not_convertible(self):
        """Non-numeric string should fail"""
        result = RangeValidator.validate_range("not a number", "vix")
        assert result.valid is False
    
    def test_unknown_field(self):
        """Unknown field should pass with INFO"""
        result = RangeValidator.validate_range(42.0, "unknown_field")
        assert result.valid is True
        assert any(i.severity == ValidationSeverity.INFO for i in result.issues)
    
    def test_clamp_to_range(self):
        """Clamping should work correctly"""
        # Within range
        val, clamped = RangeValidator.clamp_to_range(50.0, "bubble_index")
        assert val == 50.0
        assert clamped is False
        
        # Below minimum
        val, clamped = RangeValidator.clamp_to_range(-10.0, "bubble_index")
        assert val == 0.0
        assert clamped is True
        
        # Above maximum
        val, clamped = RangeValidator.clamp_to_range(150.0, "bubble_index")
        assert val == 100.0
        assert clamped is True


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestSchemaValidator:
    """Tests for SchemaValidator"""
    
    def test_valid_processed_article(self):
        """Valid processed article should pass"""
        article = {
            "category": "spending",
            "sentiment": "positive",
            "impact_score": 0.8,
            "key_entities": ["NVIDIA", "Microsoft"],
        }
        result = SchemaValidator.validate_processed_article(article)
        assert result.valid is True
    
    def test_missing_required_field(self):
        """Missing required field should fail"""
        article = {
            "category": "spending",
            "sentiment": "positive",
            # missing impact_score
        }
        result = SchemaValidator.validate_processed_article(article)
        assert result.valid is False
    
    def test_invalid_category(self):
        """Invalid category enum should fail"""
        article = {
            "category": "invalid_category",
            "sentiment": "positive",
            "impact_score": 0.8,
        }
        result = SchemaValidator.validate_processed_article(article)
        assert result.valid is False
    
    def test_invalid_sentiment(self):
        """Invalid sentiment enum should fail"""
        article = {
            "category": "spending",
            "sentiment": "very_happy",  # invalid
            "impact_score": 0.8,
        }
        result = SchemaValidator.validate_processed_article(article)
        assert result.valid is False
    
    def test_case_insensitive_enum(self):
        """Enum values should be case-insensitive"""
        article = {
            "category": "SPENDING",  # uppercase
            "sentiment": "Positive",  # mixed case
            "impact_score": 0.8,
        }
        result = SchemaValidator.validate_processed_article(article)
        # Should pass because of case-insensitive matching
        assert result.valid is True
    
    def test_impact_score_out_of_range(self):
        """Impact score outside 0-1 should fail"""
        article = {
            "category": "spending",
            "sentiment": "positive",
            "impact_score": 2.5,
        }
        result = SchemaValidator.validate_processed_article(article)
        assert result.valid is False
    
    def test_null_data(self):
        """Null data should fail"""
        result = SchemaValidator.validate_schema(None, PROCESSED_ARTICLE_SCHEMA)
        assert result.valid is False
    
    def test_non_dict_data(self):
        """Non-dict data should fail"""
        result = SchemaValidator.validate_schema("not a dict", PROCESSED_ARTICLE_SCHEMA)
        assert result.valid is False
    
    def test_type_mismatch_warning(self):
        """Type mismatch should produce warning"""
        article = {
            "category": "spending",
            "sentiment": "positive",
            "impact_score": "0.8",  # string instead of float
        }
        result = SchemaValidator.validate_schema(
            article, PROCESSED_ARTICLE_SCHEMA, "test"
        )
        assert result.has_warnings is True


class TestNewsAPIValidation:
    """Tests for NewsAPI response validation"""
    
    def test_valid_newsapi_response(self):
        """Valid NewsAPI response should pass"""
        response = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "AI News Article 1",
                    "description": "Description here",
                    "url": "https://example.com/1",
                    "publishedAt": "2025-01-01T12:00:00Z",
                },
                {
                    "title": "AI News Article 2",
                    "url": "https://example.com/2",
                }
            ]
        }
        result = SchemaValidator.validate_newsapi_response(response)
        assert result.valid is True
        assert result.metadata["valid_articles"] == 2
    
    def test_newsapi_error_status(self):
        """NewsAPI error status should fail"""
        response = {
            "status": "error",
            "message": "API key invalid"
        }
        result = SchemaValidator.validate_newsapi_response(response)
        assert result.valid is False
    
    def test_newsapi_filters_invalid_articles(self):
        """Articles without title should be filtered"""
        response = {
            "status": "ok",
            "articles": [
                {"title": "Valid Article"},
                {"description": "No title here"},  # missing title
                {"title": "Another Valid"},
            ]
        }
        result = SchemaValidator.validate_newsapi_response(response)
        assert result.metadata["valid_articles"] == 2
        assert result.metadata["total_articles"] == 3


class TestAlphaVantageValidation:
    """Tests for Alpha Vantage response validation"""
    
    def test_valid_alpha_vantage_response(self):
        """Valid Alpha Vantage response should pass"""
        response = {
            "feed": [
                {
                    "title": "Tech Stock News",
                    "url": "https://example.com/news",
                    "summary": "Summary text",
                    "overall_sentiment_score": 0.5,
                }
            ]
        }
        result = SchemaValidator.validate_alpha_vantage_response(response)
        assert result.valid is True
    
    def test_alpha_vantage_rate_limit(self):
        """Alpha Vantage rate limit message should fail"""
        response = {
            "Information": "Thank you for using Alpha Vantage! Our standard API call frequency is 25 calls per day. Please consider upgrading to a premium plan for higher rate limits."
        }
        result = SchemaValidator.validate_alpha_vantage_response(response)
        assert result.valid is False
    
    def test_alpha_vantage_error_message(self):
        """Alpha Vantage error message should fail"""
        response = {
            "Error Message": "Invalid API call"
        }
        result = SchemaValidator.validate_alpha_vantage_response(response)
        assert result.valid is False


class TestFREDValidation:
    """Tests for FRED response validation"""
    
    def test_valid_fred_response(self):
        """Valid FRED response should pass"""
        response = {
            "observations": [
                {"date": "2025-01-01", "value": "1.5"},
                {"date": "2025-01-02", "value": "1.6"},
                {"date": "2025-01-03", "value": "."},  # placeholder
            ]
        }
        result = SchemaValidator.validate_fred_response(response)
        assert result.valid is True
        assert result.metadata["valid_observations"] == 2
        assert result.metadata["total_observations"] == 3
    
    def test_fred_error_response(self):
        """FRED error response should fail"""
        response = {
            "error_code": 400,
            "error_message": "Bad Request"
        }
        result = SchemaValidator.validate_fred_response(response)
        assert result.valid is False


class TestBubbleReadingValidation:
    """Tests for bubble index reading validation"""
    
    def test_valid_bubble_reading(self):
        """Valid bubble reading should pass"""
        reading = {
            "date": "2025-01-15",
            "vix": 18.5,
            "cape": 32.0,
            "credit_spread_ig": 100.0,
            "credit_spread_hy": 400.0,
            "bubble_index": 62.5,
            "bifurcation_score": 0.45,
            "regime": "ADOPTION",
        }
        result = SchemaValidator.validate_bubble_reading(reading)
        assert result.valid is True
    
    def test_invalid_regime(self):
        """Invalid regime should fail"""
        reading = {
            "date": "2025-01-15",
            "vix": 18.5,
            "cape": 32.0,
            "bubble_index": 62.5,
            "bifurcation_score": 0.45,
            "regime": "INVALID_REGIME",
        }
        result = SchemaValidator.validate_bubble_reading(reading)
        assert result.valid is False
    
    def test_vix_out_of_range_in_reading(self):
        """VIX out of range should fail reading validation"""
        reading = {
            "date": "2025-01-15",
            "vix": 200.0,  # out of range
            "cape": 32.0,
            "bubble_index": 62.5,
            "bifurcation_score": 0.45,
            "regime": "ADOPTION",
        }
        result = SchemaValidator.validate_bubble_reading(reading)
        assert result.valid is False


# =============================================================================
# FRESHNESS VALIDATION TESTS
# =============================================================================

class TestFreshnessValidator:
    """Tests for FreshnessValidator"""
    
    def test_fresh_data(self):
        """Recent timestamp should pass"""
        now = datetime.now(timezone.utc)
        recent = now - timedelta(minutes=30)
        result = FreshnessValidator.validate_timestamp(recent.isoformat(), "market_data")
        assert result.valid is True
    
    def test_stale_data_warning(self):
        """Stale data should produce warning"""
        old = datetime.now(timezone.utc) - timedelta(hours=5)
        result = FreshnessValidator.validate_timestamp(old.isoformat(), "market_data")
        assert result.has_warnings is True
    
    def test_future_timestamp_error(self):
        """Future timestamp should fail"""
        future = datetime.now(timezone.utc) + timedelta(hours=2)
        result = FreshnessValidator.validate_timestamp(future.isoformat(), "market_data")
        assert result.valid is False
    
    def test_various_timestamp_formats(self):
        """Various timestamp formats should be parsed"""
        now = datetime.now(timezone.utc)
        
        formats = [
            now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            now.strftime("%Y-%m-%dT%H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d"),
        ]
        
        for fmt in formats:
            result = FreshnessValidator.validate_timestamp(fmt, "market_data")
            # Should not fail on parsing (may warn on staleness for date-only)
            assert "Cannot parse" not in str([i.message for i in result.issues])
    
    def test_news_article_freshness(self):
        """News article within 72 hours should pass"""
        yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
        result = FreshnessValidator.validate_article_date(yesterday.isoformat())
        assert result.valid is True
        assert not result.has_warnings
    
    def test_old_news_article_warning(self):
        """News article older than 72 hours should warn"""
        old = datetime.now(timezone.utc) - timedelta(days=5)
        result = FreshnessValidator.validate_article_date(old.isoformat())
        assert result.has_warnings is True
    
    def test_custom_max_age(self):
        """Custom max age should be respected"""
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # 30 minute threshold - should warn
        result = FreshnessValidator.validate_timestamp(
            one_hour_ago.isoformat(), "custom", custom_max_age_hours=0.5
        )
        assert result.has_warnings is True
        
        # 2 hour threshold - should pass
        result = FreshnessValidator.validate_timestamp(
            one_hour_ago.isoformat(), "custom", custom_max_age_hours=2.0
        )
        assert not result.has_warnings


# =============================================================================
# ANOMALY DETECTION TESTS
# =============================================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector"""
    
    def test_normal_value_no_anomaly(self):
        """Normal VIX should not be flagged"""
        result = AnomalyDetector.detect_anomaly(18.0, "vix")
        assert not result.has_warnings
        assert result.metadata.get("is_anomalous") is False
    
    def test_high_zscore_anomaly(self):
        """High z-score should be flagged"""
        # VIX of 65 is about 6 standard deviations above mean
        result = AnomalyDetector.detect_anomaly(65.0, "vix")
        assert result.has_warnings
        assert result.metadata.get("is_anomalous") is True
    
    def test_below_historical_minimum(self):
        """Value below historical minimum should warn"""
        result = AnomalyDetector.detect_anomaly(7.0, "vix")  # min_seen is 9.0
        assert result.has_warnings
    
    def test_above_historical_maximum(self):
        """Value above historical maximum should warn"""
        result = AnomalyDetector.detect_anomaly(90.0, "vix")  # max_seen is 82.69
        assert result.has_warnings
    
    def test_zscore_calculation(self):
        """Z-score should be calculated correctly"""
        # VIX baseline: mean=19.5, std=7.5
        zscore, is_anomalous = AnomalyDetector.calculate_zscore(27.0, "vix")
        # z = (27 - 19.5) / 7.5 = 1.0
        assert abs(zscore - 1.0) < 0.01
        assert is_anomalous is False  # z=1 is not anomalous
    
    def test_sudden_change_detection(self):
        """Sudden change should be flagged"""
        result = AnomalyDetector.detect_sudden_change(
            current=40.0, previous=20.0, field_name="vix", max_change_pct=50.0
        )
        assert result.has_warnings is True
        assert result.metadata["change_pct"] == 100.0
    
    def test_normal_change_no_warning(self):
        """Normal change should not be flagged"""
        result = AnomalyDetector.detect_sudden_change(
            current=21.0, previous=20.0, field_name="vix", max_change_pct=50.0
        )
        assert not result.has_warnings
    
    def test_unknown_field_no_anomaly(self):
        """Unknown field should not flag anomaly"""
        result = AnomalyDetector.detect_anomaly(999.0, "unknown_field")
        assert result.valid is True


# =============================================================================
# STORAGE VALIDATION TESTS
# =============================================================================

class TestStorageValidator:
    """Tests for StorageValidator"""
    
    def test_valid_raw_article(self):
        """Valid raw article should pass"""
        article = {
            "title": "Test Article",
            "source_adapter": "newsapi",
            "collected_at": datetime.now().isoformat(),
            "description": "Description text",
        }
        result = StorageValidator.validate_for_storage(article, "articles_raw")
        assert result.valid is True
    
    def test_missing_required_field(self):
        """Missing required field should fail"""
        article = {
            "title": "Test Article",
            # missing source_adapter and collected_at
        }
        result = StorageValidator.validate_for_storage(article, "articles_raw")
        assert result.valid is False
    
    def test_title_truncation(self):
        """Long title should be truncated"""
        long_title = "A" * 600  # longer than 500 char limit
        article = {
            "title": long_title,
            "source_adapter": "newsapi",
            "collected_at": datetime.now().isoformat(),
        }
        result = StorageValidator.validate_for_storage(article, "articles_raw")
        assert result.valid is True
        assert len(result.data["title"]) == 500
        assert result.has_warnings  # truncation warning
    
    def test_nan_sanitization(self):
        """NaN values should be converted to None"""
        record = {
            "date": "2025-01-15",
            "ticker": "NVDA",
            "price": float('nan'),
            "change_pct": 1.5,
        }
        result = StorageValidator.validate_for_storage(record, "stock_snapshots")
        assert result.data["price"] is None
    
    def test_datetime_sanitization(self):
        """Datetime values should be converted to ISO string"""
        now = datetime.now()
        record = {
            "title": "Test",
            "source_adapter": "rss",
            "collected_at": now,
        }
        result = StorageValidator.validate_for_storage(record, "articles_raw")
        assert isinstance(result.data["collected_at"], str)
    
    def test_batch_validate(self):
        """Batch validation should filter invalid records"""
        records = [
            {"title": "Valid 1", "source_adapter": "newsapi", "collected_at": "2025-01-01"},
            {"title": "Valid 2", "source_adapter": "rss", "collected_at": "2025-01-02"},
            {"description": "Invalid - no title"},  # missing required fields
        ]
        valid, failed = StorageValidator.batch_validate(records, "articles_raw")
        assert len(valid) == 2
        assert len(failed) == 1
    
    def test_unknown_table(self):
        """Unknown table should produce warning"""
        result = StorageValidator.validate_for_storage({"data": "test"}, "unknown_table")
        assert result.has_warnings


# =============================================================================
# VALIDATION DECORATOR TESTS
# =============================================================================

class TestValidationDecorator:
    """Tests for @validate_input decorator"""
    
    def test_decorator_with_schema(self):
        """Decorator should validate against schema"""
        @validate_input(schema=PROCESSED_ARTICLE_SCHEMA)
        def process_article(data):
            return data
        
        # Valid data should pass
        valid_data = {
            "category": "spending",
            "sentiment": "positive",
            "impact_score": 0.8,
        }
        result = process_article(valid_data)
        assert result == valid_data
    
    def test_decorator_with_range_fields(self):
        """Decorator should validate range fields"""
        @validate_input(range_fields=["vix", "cape"])
        def process_indicators(data):
            return data
        
        valid_data = {"vix": 20.0, "cape": 30.0}
        result = process_indicators(valid_data)
        assert result == valid_data
    
    def test_decorator_raises_on_error(self):
        """Decorator should raise when configured"""
        @validate_input(
            schema=PROCESSED_ARTICLE_SCHEMA,
            raise_on_error=True
        )
        def process_article(data):
            return data
        
        invalid_data = {"category": "spending"}  # missing required fields
        
        with pytest.raises(ValueError):
            process_article(invalid_data)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience validation functions"""
    
    def test_validate_market_indicators_valid(self):
        """Valid market indicators should pass"""
        result = validate_market_indicators(
            vix=18.0,
            cape=30.0,
            credit_ig=100.0,
            credit_hy=400.0,
        )
        assert result.valid is True
    
    def test_validate_market_indicators_partial(self):
        """Partial indicators should still validate"""
        result = validate_market_indicators(vix=18.0)
        assert result.valid is True
        assert "vix" in result.data
    
    def test_validate_market_indicators_invalid(self):
        """Invalid values should fail"""
        result = validate_market_indicators(vix=200.0)  # out of range
        assert result.valid is False
    
    def test_validate_article_batch(self):
        """Article batch validation should work"""
        articles = [
            {"title": "Article 1", "published_at": datetime.now().isoformat()},
            {"title": "Article 2"},
            {"description": "No title"},  # invalid
        ]
        valid, result = validate_article_batch(articles)
        assert len(valid) == 2
        assert result.metadata["total"] == 3
        assert result.metadata["valid"] == 2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_empty_dict(self):
        """Empty dict should fail schema validation with required fields"""
        result = SchemaValidator.validate_schema({}, PROCESSED_ARTICLE_SCHEMA)
        assert result.valid is False
    
    def test_extra_fields_allowed(self):
        """Extra fields should not cause failure"""
        article = {
            "category": "spending",
            "sentiment": "positive",
            "impact_score": 0.8,
            "extra_field": "ignored",
        }
        result = SchemaValidator.validate_processed_article(article)
        assert result.valid is True
    
    def test_nested_validation(self):
        """Nested structures should be validated"""
        response = {
            "status": "ok",
            "articles": [
                {"title": "Valid"},
                {"source": {"name": "Test", "id": "test"}},  # missing title
            ]
        }
        result = SchemaValidator.validate_newsapi_response(response)
        assert result.metadata["valid_articles"] == 1
    
    def test_bifurcation_score_extremes(self):
        """Bifurcation score at extremes should be valid"""
        for score in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            result = RangeValidator.validate_range(score, "bifurcation_score")
            assert result.valid is True
    
    def test_zscore_extremes(self):
        """Z-score extremes should be valid but flag anomalies"""
        result = RangeValidator.validate_range(8.0, "vix_zscore")
        assert result.valid is True  # within hard limits
    
    def test_credit_spread_ranges(self):
        """Credit spreads should validate correctly"""
        # IG spread
        result = RangeValidator.validate_range(100.0, "credit_spread_ig")
        assert result.valid is True
        
        # HY spread
        result = RangeValidator.validate_range(400.0, "credit_spread_hy")
        assert result.valid is True


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

class TestIntegration:
    """Integration-style tests combining multiple validators"""
    
    def test_full_bubble_reading_flow(self):
        """Full validation flow for bubble reading"""
        reading = {
            "date": "2025-01-15",
            "vix": 22.5,
            "cape": 35.0,
            "credit_spread_ig": 120.0,
            "credit_spread_hy": 450.0,
            "vix_zscore": 0.4,
            "cape_zscore": 1.0,
            "credit_zscore": -0.6,
            "bubble_index": 70.0,
            "bifurcation_score": 0.77,
            "regime": "INFRASTRUCTURE",
            "data_sources": {"vix": "live", "cape": "live", "credit": "live"},
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Schema validation
        schema_result = SchemaValidator.validate_bubble_reading(reading)
        assert schema_result.valid is True
        
        # Storage validation
        storage_result = StorageValidator.validate_for_storage(reading, "bubble_readings")
        assert storage_result.valid is True
        
        # Market indicators validation
        indicators_result = validate_market_indicators(
            vix=reading["vix"],
            cape=reading["cape"],
            credit_ig=reading["credit_spread_ig"],
            credit_hy=reading["credit_spread_hy"],
        )
        assert indicators_result.valid is True
    
    def test_full_article_processing_flow(self):
        """Full validation flow for article processing"""
        raw_article = {
            "title": "NVIDIA Reports Record AI Chip Revenue",
            "description": "NVIDIA announced quarterly results...",
            "url": "https://example.com/nvidia-news",
            "publishedAt": datetime.now(timezone.utc).isoformat(),
            "source": {"name": "TechNews"},
        }
        
        processed_article = {
            "category": "revenue",
            "sentiment": "positive",
            "impact_score": 0.85,
            "key_entities": ["NVIDIA"],
            "key_metrics": ["quarterly revenue"],
        }
        
        # Validate raw article
        raw_valid, raw_result = validate_article_batch([raw_article])
        assert len(raw_valid) == 1
        
        # Validate processed article
        processed_result = SchemaValidator.validate_processed_article(processed_article)
        assert processed_result.valid is True
        
        # Validate for storage
        storage_record = {
            "raw_article_id": "123",
            "category": processed_article["category"],
            "sentiment": processed_article["sentiment"],
            "impact_score": processed_article["impact_score"],
            "processed_at": datetime.now().isoformat(),
        }
        storage_result = StorageValidator.validate_for_storage(
            storage_record, "articles_processed"
        )
        assert storage_result.valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
