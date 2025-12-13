"""
ARGUS+Y2AI DATA VALIDATION MODULE
Comprehensive input validation, schema checking, range validation, and anomaly detection

This module provides validation for:
- API response schemas (NewsAPI, Alpha Vantage, FRED, yfinance)
- Value range constraints (VIX 0-100, CAPE 5-60, impact_score 0-1, etc.)
- Data freshness validation
- Statistical anomaly detection
- Pre-storage validation for Supabase
"""

import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"           # Informational, data is valid
    WARNING = "warning"     # Data is usable but flagged
    ERROR = "error"         # Data is invalid, should not be used
    CRITICAL = "critical"   # Data indicates system failure


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    field: str
    message: str
    severity: ValidationSeverity
    actual_value: Any = None
    expected: str = ""
    
    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "actual_value": str(self.actual_value)[:100] if self.actual_value else None,
            "expected": self.expected,
        }


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    data: Any = None  # The validated (possibly corrected) data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity,
        actual_value: Any = None,
        expected: str = ""
    ):
        self.issues.append(ValidationIssue(
            field=field,
            message=message,
            severity=severity,
            actual_value=actual_value,
            expected=expected,
        ))
        # Mark as invalid for ERROR or CRITICAL severity
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.valid = False
    
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) 
                   for i in self.issues)
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues 
                   if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL))
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
    
    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "issues": [i.to_dict() for i in self.issues],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "metadata": self.metadata,
        }
    
    def log_issues(self, prefix: str = ""):
        """Log all issues at appropriate levels"""
        for issue in self.issues:
            msg = f"{prefix}[{issue.field}] {issue.message}"
            if issue.severity == ValidationSeverity.INFO:
                logger.info(msg)
            elif issue.severity == ValidationSeverity.WARNING:
                logger.warning(msg)
            elif issue.severity == ValidationSeverity.ERROR:
                logger.error(msg)
            elif issue.severity == ValidationSeverity.CRITICAL:
                logger.critical(msg)


# =============================================================================
# VALUE RANGE DEFINITIONS
# =============================================================================

# Range constraints for known fields
# Format: (min, max, allow_null, description)
RANGE_CONSTRAINTS = {
    # Market indicators
    "vix": (0.0, 150.0, False, "VIX volatility index"),
    "cape": (5.0, 70.0, False, "Shiller CAPE ratio"),
    "credit_spread_ig": (20.0, 1000.0, False, "Investment grade spread (bps)"),
    "credit_spread_hy": (100.0, 3000.0, False, "High yield spread (bps)"),
    "bubble_index": (0.0, 100.0, False, "Y2AI Bubble Index"),
    "bifurcation_score": (-2.0, 2.0, False, "Bifurcation score"),
    
    # Z-scores (typically -4 to +4, but can be more extreme)
    "vix_zscore": (-5.0, 10.0, False, "VIX z-score"),
    "cape_zscore": (-4.0, 6.0, False, "CAPE z-score"),
    "credit_zscore": (-4.0, 6.0, False, "Credit spread z-score"),
    
    # Stock data
    "stock_price": (0.01, 100000.0, False, "Stock price"),
    "stock_change_pct": (-50.0, 100.0, False, "Stock daily change %"),
    "pe_ratio": (0.0, 1000.0, True, "Price to earnings ratio"),
    "market_cap": (0.0, 1e13, True, "Market capitalization"),
    
    # Article processing
    "impact_score": (0.0, 1.0, False, "Article impact score"),
    "relevance_score": (0.0, 1.0, True, "Article relevance score"),
    "confidence_score": (0.0, 1.0, True, "Processing confidence"),
}

# Warning thresholds for anomaly detection (z-score-like)
# Values outside these ranges are flagged as warnings
ANOMALY_THRESHOLDS = {
    "vix": (8.0, 60.0),           # Normal VIX range
    "cape": (12.0, 50.0),         # Normal CAPE range
    "credit_spread_ig": (40.0, 400.0),  # Normal IG spread
    "credit_spread_hy": (200.0, 1200.0),  # Normal HY spread
    "stock_change_pct": (-10.0, 15.0),  # Typical daily move
}


# =============================================================================
# RANGE VALIDATORS
# =============================================================================

class RangeValidator:
    """Validates numeric values against defined ranges"""
    
    @staticmethod
    def validate_range(
        value: Any,
        field_name: str,
        custom_range: Optional[Tuple[float, float]] = None,
        allow_null: bool = False,
    ) -> ValidationResult:
        """
        Validate a value against its defined or custom range.
        
        Args:
            value: The value to validate
            field_name: Name of the field (for lookup in RANGE_CONSTRAINTS)
            custom_range: Optional (min, max) override
            allow_null: Whether None/null is acceptable
        """
        result = ValidationResult(valid=True)
        
        # Handle null values
        if value is None:
            constraint = RANGE_CONSTRAINTS.get(field_name)
            null_allowed = constraint[2] if constraint else allow_null
            
            if null_allowed:
                result.add_issue(
                    field_name, "Value is null (allowed)",
                    ValidationSeverity.INFO
                )
            else:
                result.add_issue(
                    field_name, "Value is null but null not allowed",
                    ValidationSeverity.ERROR,
                    actual_value=None,
                    expected="non-null value"
                )
            result.data = value
            return result
        
        # Convert to float for comparison
        try:
            num_value = float(value)
        except (ValueError, TypeError) as e:
            result.add_issue(
                field_name, f"Cannot convert to number: {e}",
                ValidationSeverity.ERROR,
                actual_value=value,
                expected="numeric value"
            )
            return result
        
        # Check for NaN or infinity
        import math
        if math.isnan(num_value):
            result.add_issue(
                field_name, "Value is NaN",
                ValidationSeverity.ERROR,
                actual_value=value
            )
            return result
        
        if math.isinf(num_value):
            result.add_issue(
                field_name, "Value is infinite",
                ValidationSeverity.ERROR,
                actual_value=value
            )
            return result
        
        # Get range constraints
        if custom_range:
            min_val, max_val = custom_range
            description = field_name
        elif field_name in RANGE_CONSTRAINTS:
            min_val, max_val, _, description = RANGE_CONSTRAINTS[field_name]
        else:
            # No constraints defined, pass through with info
            result.add_issue(
                field_name, f"No range constraints defined for {field_name}",
                ValidationSeverity.INFO,
                actual_value=num_value
            )
            result.data = num_value
            return result
        
        # Check hard range constraints
        if num_value < min_val:
            result.add_issue(
                field_name, f"Value {num_value} below minimum {min_val}",
                ValidationSeverity.ERROR,
                actual_value=num_value,
                expected=f"{min_val} to {max_val}"
            )
        elif num_value > max_val:
            result.add_issue(
                field_name, f"Value {num_value} above maximum {max_val}",
                ValidationSeverity.ERROR,
                actual_value=num_value,
                expected=f"{min_val} to {max_val}"
            )
        
        # Check anomaly thresholds (warnings only)
        if field_name in ANOMALY_THRESHOLDS:
            warn_min, warn_max = ANOMALY_THRESHOLDS[field_name]
            if num_value < warn_min:
                result.add_issue(
                    field_name,
                    f"Value {num_value} is unusually low (threshold: {warn_min})",
                    ValidationSeverity.WARNING,
                    actual_value=num_value
                )
            elif num_value > warn_max:
                result.add_issue(
                    field_name,
                    f"Value {num_value} is unusually high (threshold: {warn_max})",
                    ValidationSeverity.WARNING,
                    actual_value=num_value
                )
        
        result.data = num_value
        return result
    
    @staticmethod
    def clamp_to_range(value: float, field_name: str) -> Tuple[float, bool]:
        """
        Clamp a value to its defined range.
        
        Returns:
            Tuple of (clamped_value, was_clamped)
        """
        if field_name not in RANGE_CONSTRAINTS:
            return value, False
        
        min_val, max_val, _, _ = RANGE_CONSTRAINTS[field_name]
        clamped = max(min_val, min(max_val, value))
        was_clamped = clamped != value
        
        if was_clamped:
            logger.warning(f"Clamped {field_name}: {value} -> {clamped}")
        
        return clamped, was_clamped


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Schema for NewsAPI response
NEWSAPI_ARTICLE_SCHEMA = {
    "required": ["title"],
    "optional": ["description", "url", "urlToImage", "publishedAt", "source", "author", "content"],
    "types": {
        "title": str,
        "description": (str, type(None)),
        "url": str,
        "publishedAt": str,
        "source": dict,
    }
}

# Schema for Alpha Vantage news
ALPHA_VANTAGE_ARTICLE_SCHEMA = {
    "required": ["title", "url"],
    "optional": ["summary", "time_published", "source", "banner_image", "topics", "overall_sentiment_score"],
    "types": {
        "title": str,
        "summary": (str, type(None)),
        "url": str,
        "overall_sentiment_score": (float, str, type(None)),
    }
}

# Schema for FRED API response
FRED_OBSERVATION_SCHEMA = {
    "required": ["date", "value"],
    "types": {
        "date": str,
        "value": (str, float, int),
    }
}

# Schema for yfinance ticker data
YFINANCE_QUOTE_SCHEMA = {
    "required": [],
    "optional": ["regularMarketPrice", "previousClose", "open", "dayHigh", "dayLow", 
                 "volume", "marketCap", "trailingPE", "forwardPE"],
    "types": {
        "regularMarketPrice": (float, int, type(None)),
        "previousClose": (float, int, type(None)),
        "volume": (int, float, type(None)),
        "marketCap": (int, float, type(None)),
    }
}

# Schema for processed article (Claude output)
PROCESSED_ARTICLE_SCHEMA = {
    "required": ["category", "sentiment", "impact_score"],
    "optional": ["key_entities", "key_metrics", "summary", "relevance_score"],
    "types": {
        "category": str,
        "sentiment": str,
        "impact_score": (float, int),
        "key_entities": list,
        "key_metrics": list,
    },
    "enum_values": {
        "category": ["spending", "revenue", "pricing", "utilization", "deployment",
                    "competition", "sentiment", "macro", "data"],
        "sentiment": ["positive", "negative", "neutral", "mixed"],
    }
}

# Schema for bubble index reading
BUBBLE_INDEX_READING_SCHEMA = {
    "required": ["date", "vix", "cape", "bubble_index", "bifurcation_score", "regime"],
    "optional": ["credit_spread_ig", "credit_spread_hy", "vix_zscore", "cape_zscore",
                "credit_zscore", "data_sources", "calculated_at"],
    "types": {
        "date": str,
        "vix": (float, int),
        "cape": (float, int),
        "bubble_index": (float, int),
        "bifurcation_score": float,
        "regime": str,
    },
    "enum_values": {
        "regime": ["INFRASTRUCTURE", "ADOPTION", "TRANSITION", "BUBBLE_WARNING"],
    }
}

# Schema for stock report
STOCK_REPORT_SCHEMA = {
    "required": ["date", "stocks"],
    "optional": ["pillar_averages", "y2ai_index", "thesis_status", "data_quality_score",
                "failed_tickers"],
    "types": {
        "date": str,
        "stocks": dict,
        "pillar_averages": dict,
        "y2ai_index": (float, type(None)),
    },
    "enum_values": {
        "thesis_status": ["VALIDATING", "NEUTRAL", "CONTRADICTING"],
    }
}


# =============================================================================
# SCHEMA VALIDATORS
# =============================================================================

class SchemaValidator:
    """Validates data structures against defined schemas"""
    
    @staticmethod
    def validate_schema(
        data: Dict[str, Any],
        schema: Dict[str, Any],
        schema_name: str = "data"
    ) -> ValidationResult:
        """
        Validate a dictionary against a schema definition.
        
        Args:
            data: The data to validate
            schema: Schema definition with "required", "optional", "types", "enum_values"
            schema_name: Name for logging purposes
        """
        result = ValidationResult(valid=True, data=data)
        
        if data is None:
            result.add_issue(
                schema_name, "Data is None",
                ValidationSeverity.ERROR
            )
            return result
        
        if not isinstance(data, dict):
            result.add_issue(
                schema_name, f"Expected dict, got {type(data).__name__}",
                ValidationSeverity.ERROR,
                actual_value=type(data).__name__,
                expected="dict"
            )
            return result
        
        # Check required fields
        for field in schema.get("required", []):
            if field not in data or data[field] is None:
                result.add_issue(
                    f"{schema_name}.{field}",
                    f"Required field '{field}' is missing or null",
                    ValidationSeverity.ERROR
                )
        
        # Check types
        types = schema.get("types", {})
        for field, expected_types in types.items():
            if field in data and data[field] is not None:
                if not isinstance(expected_types, tuple):
                    expected_types = (expected_types,)
                
                if not isinstance(data[field], expected_types):
                    result.add_issue(
                        f"{schema_name}.{field}",
                        f"Type mismatch: expected {expected_types}, got {type(data[field]).__name__}",
                        ValidationSeverity.WARNING,
                        actual_value=type(data[field]).__name__,
                        expected=str(expected_types)
                    )
        
        # Check enum values
        enums = schema.get("enum_values", {})
        for field, valid_values in enums.items():
            if field in data and data[field] is not None:
                value = data[field]
                # Case-insensitive check for strings
                if isinstance(value, str):
                    if value.lower() not in [v.lower() for v in valid_values]:
                        result.add_issue(
                            f"{schema_name}.{field}",
                            f"Invalid enum value: '{value}'",
                            ValidationSeverity.ERROR,
                            actual_value=value,
                            expected=str(valid_values)
                        )
                elif value not in valid_values:
                    result.add_issue(
                        f"{schema_name}.{field}",
                        f"Invalid enum value: '{value}'",
                        ValidationSeverity.ERROR,
                        actual_value=value,
                        expected=str(valid_values)
                    )
        
        return result
    
    @staticmethod
    def validate_newsapi_response(response: Dict[str, Any]) -> ValidationResult:
        """Validate a NewsAPI response"""
        result = ValidationResult(valid=True)
        
        if response.get("status") != "ok":
            result.add_issue(
                "status", f"NewsAPI status is not 'ok': {response.get('status')}",
                ValidationSeverity.ERROR,
                actual_value=response.get("status")
            )
            return result
        
        articles = response.get("articles", [])
        if not isinstance(articles, list):
            result.add_issue(
                "articles", "Articles field is not a list",
                ValidationSeverity.ERROR,
                actual_value=type(articles).__name__
            )
            return result
        
        # Validate each article
        valid_articles = []
        for i, article in enumerate(articles):
            article_result = SchemaValidator.validate_schema(
                article, NEWSAPI_ARTICLE_SCHEMA, f"article[{i}]"
            )
            if article_result.valid:
                valid_articles.append(article)
            else:
                for issue in article_result.issues:
                    if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
                        result.issues.append(issue)
        
        result.data = {"articles": valid_articles, "total_valid": len(valid_articles)}
        result.metadata["total_articles"] = len(articles)
        result.metadata["valid_articles"] = len(valid_articles)
        
        if len(valid_articles) < len(articles):
            result.add_issue(
                "articles",
                f"Filtered {len(articles) - len(valid_articles)} invalid articles",
                ValidationSeverity.WARNING
            )
        
        return result
    
    @staticmethod
    def validate_alpha_vantage_response(response: Dict[str, Any]) -> ValidationResult:
        """Validate an Alpha Vantage news response"""
        result = ValidationResult(valid=True)
        
        # Check for error messages
        if "Error Message" in response or "Note" in response:
            error_msg = response.get("Error Message") or response.get("Note")
            result.add_issue(
                "response", f"Alpha Vantage error: {error_msg}",
                ValidationSeverity.ERROR,
                actual_value=error_msg
            )
            return result
        
        # Check for rate limit message in Information field
        if "Information" in response:
            if "rate limit" in response["Information"].lower():
                result.add_issue(
                    "response", "Alpha Vantage rate limit hit",
                    ValidationSeverity.ERROR,
                    actual_value=response["Information"]
                )
                return result
        
        feed = response.get("feed", [])
        if not isinstance(feed, list):
            result.add_issue(
                "feed", "Feed field is not a list",
                ValidationSeverity.ERROR
            )
            return result
        
        # Validate articles
        valid_articles = []
        for i, article in enumerate(feed):
            article_result = SchemaValidator.validate_schema(
                article, ALPHA_VANTAGE_ARTICLE_SCHEMA, f"article[{i}]"
            )
            if article_result.valid:
                valid_articles.append(article)
        
        result.data = {"feed": valid_articles}
        result.metadata["total_articles"] = len(feed)
        result.metadata["valid_articles"] = len(valid_articles)
        
        return result
    
    @staticmethod
    def validate_fred_response(response: Dict[str, Any]) -> ValidationResult:
        """Validate a FRED API response"""
        result = ValidationResult(valid=True)
        
        if "error_code" in response:
            result.add_issue(
                "response", f"FRED error: {response.get('error_message')}",
                ValidationSeverity.ERROR,
                actual_value=response.get("error_code")
            )
            return result
        
        observations = response.get("observations", [])
        if not isinstance(observations, list):
            result.add_issue(
                "observations", "Observations field is not a list",
                ValidationSeverity.ERROR
            )
            return result
        
        # Filter valid observations (exclude "." placeholder values)
        valid_obs = []
        for obs in observations:
            if obs.get("value") != "." and obs.get("value") is not None:
                valid_obs.append(obs)
        
        result.data = {"observations": valid_obs}
        result.metadata["total_observations"] = len(observations)
        result.metadata["valid_observations"] = len(valid_obs)
        
        return result
    
    @staticmethod
    def validate_processed_article(article: Dict[str, Any]) -> ValidationResult:
        """Validate a Claude-processed article"""
        result = SchemaValidator.validate_schema(
            article, PROCESSED_ARTICLE_SCHEMA, "processed_article"
        )
        
        # Additional range validation
        if "impact_score" in article and article["impact_score"] is not None:
            range_result = RangeValidator.validate_range(
                article["impact_score"], "impact_score"
            )
            result.issues.extend(range_result.issues)
            if not range_result.valid:
                result.valid = False
        
        return result
    
    @staticmethod
    def validate_bubble_reading(reading: Dict[str, Any]) -> ValidationResult:
        """Validate a bubble index reading"""
        result = SchemaValidator.validate_schema(
            reading, BUBBLE_INDEX_READING_SCHEMA, "bubble_reading"
        )
        
        # Validate numeric ranges
        range_fields = ["vix", "cape", "bubble_index", "bifurcation_score",
                       "credit_spread_ig", "credit_spread_hy", "vix_zscore",
                       "cape_zscore", "credit_zscore"]
        
        for field in range_fields:
            if field in reading and reading[field] is not None:
                range_result = RangeValidator.validate_range(reading[field], field)
                result.issues.extend(range_result.issues)
                if not range_result.valid:
                    result.valid = False
        
        return result


# =============================================================================
# FRESHNESS VALIDATION
# =============================================================================

class FreshnessValidator:
    """Validates data freshness based on expected update frequencies"""
    
    # Expected maximum age for different data types (in hours)
    FRESHNESS_THRESHOLDS = {
        "market_data": 1,           # Stock prices, VIX (1 hour during market hours)
        "news_article": 72,         # News articles (3 days)
        "bubble_index": 24,         # Daily calculation
        "credit_spread": 48,        # FRED updates less frequently
        "cape": 168,                # Shiller CAPE updates monthly
        "stock_report": 24,         # Daily reports
    }
    
    @staticmethod
    def validate_timestamp(
        timestamp: Union[str, datetime],
        data_type: str,
        custom_max_age_hours: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate that a timestamp is fresh enough.
        
        Args:
            timestamp: ISO format string or datetime
            data_type: Type of data for threshold lookup
            custom_max_age_hours: Override default threshold
        """
        result = ValidationResult(valid=True)
        
        # Parse timestamp
        ts = None
        try:
            if isinstance(timestamp, str):
                # Try Python's fromisoformat first (handles +00:00 and Z)
                try:
                    # Handle 'Z' suffix by replacing with +00:00
                    ts_str = timestamp.replace('Z', '+00:00')
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    # Fallback to manual format parsing
                    for fmt in [
                        "%Y-%m-%dT%H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%Y%m%dT%H%M%S",
                    ]:
                        try:
                            ts = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                
                if ts is None:
                    result.add_issue(
                        "timestamp", f"Cannot parse timestamp: {timestamp}",
                        ValidationSeverity.WARNING,
                        actual_value=timestamp
                    )
                    return result
                
                # Ensure timezone awareness
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
        except Exception as e:
            result.add_issue(
                "timestamp", f"Error parsing timestamp: {e}",
                ValidationSeverity.WARNING
            )
            return result
        
        # Calculate age
        now = datetime.now(timezone.utc)
        age = now - ts
        age_hours = age.total_seconds() / 3600
        
        # Get threshold
        max_age = custom_max_age_hours or FreshnessValidator.FRESHNESS_THRESHOLDS.get(
            data_type, 24  # Default to 24 hours
        )
        
        result.data = {
            "timestamp": ts.isoformat(),
            "age_hours": round(age_hours, 2),
            "max_age_hours": max_age,
        }
        
        # Check freshness
        if age_hours > max_age:
            result.add_issue(
                "freshness",
                f"Data is stale: {age_hours:.1f} hours old (max: {max_age} hours)",
                ValidationSeverity.WARNING,
                actual_value=f"{age_hours:.1f} hours",
                expected=f"< {max_age} hours"
            )
        
        # Future timestamps are always an error
        if age.total_seconds() < -60:  # Allow 1 minute tolerance
            result.add_issue(
                "freshness",
                f"Timestamp is in the future: {ts}",
                ValidationSeverity.ERROR,
                actual_value=ts.isoformat()
            )
        
        return result
    
    @staticmethod
    def validate_article_date(
        published_at: str,
        max_age_hours: float = 72
    ) -> ValidationResult:
        """Validate article publication date"""
        return FreshnessValidator.validate_timestamp(
            published_at, "news_article", max_age_hours
        )


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

class AnomalyDetector:
    """Detects statistical anomalies in data"""
    
    # Historical baselines for anomaly detection
    HISTORICAL_BASELINES = {
        "vix": {"mean": 19.5, "std": 7.5, "min_seen": 9.0, "max_seen": 82.69},
        "cape": {"mean": 25.0, "std": 8.0, "min_seen": 5.0, "max_seen": 44.0},
        "credit_spread_ig": {"mean": 120.0, "std": 40.0, "min_seen": 40.0, "max_seen": 600.0},
        "credit_spread_hy": {"mean": 450.0, "std": 200.0, "min_seen": 200.0, "max_seen": 2000.0},
        "stock_change_pct": {"mean": 0.0, "std": 2.0, "min_seen": -23.0, "max_seen": 20.0},
    }
    
    @staticmethod
    def calculate_zscore(
        value: float,
        field_name: str,
        custom_baseline: Optional[Dict[str, float]] = None
    ) -> Tuple[float, bool]:
        """
        Calculate z-score and determine if value is anomalous.
        
        Returns:
            Tuple of (zscore, is_anomalous)
        """
        baseline = custom_baseline or AnomalyDetector.HISTORICAL_BASELINES.get(field_name)
        
        if not baseline:
            return 0.0, False
        
        mean = baseline.get("mean", 0)
        std = baseline.get("std", 1)
        
        if std == 0:
            return 0.0, False
        
        zscore = (value - mean) / std
        is_anomalous = abs(zscore) > 3  # 3 sigma threshold
        
        return zscore, is_anomalous
    
    @staticmethod
    def detect_anomaly(
        value: float,
        field_name: str,
        zscore_threshold: float = 3.0
    ) -> ValidationResult:
        """
        Detect if a value is anomalous.
        
        Args:
            value: The value to check
            field_name: Name of the field for baseline lookup
            zscore_threshold: Z-score threshold for flagging anomalies
        """
        result = ValidationResult(valid=True, data=value)
        
        baseline = AnomalyDetector.HISTORICAL_BASELINES.get(field_name)
        if not baseline:
            return result
        
        zscore, is_anomalous = AnomalyDetector.calculate_zscore(value, field_name)
        
        result.metadata["zscore"] = round(zscore, 2)
        result.metadata["is_anomalous"] = is_anomalous
        
        # Check against historical extremes
        min_seen = baseline.get("min_seen")
        max_seen = baseline.get("max_seen")
        
        if min_seen is not None and value < min_seen:
            result.add_issue(
                field_name,
                f"Value {value} below historical minimum {min_seen}",
                ValidationSeverity.WARNING,
                actual_value=value,
                expected=f">= {min_seen}"
            )
        
        if max_seen is not None and value > max_seen:
            result.add_issue(
                field_name,
                f"Value {value} above historical maximum {max_seen}",
                ValidationSeverity.WARNING,
                actual_value=value,
                expected=f"<= {max_seen}"
            )
        
        # Flag as warning if anomalous but within historical range
        if is_anomalous:
            result.add_issue(
                field_name,
                f"Statistical anomaly detected: z-score = {zscore:.2f}",
                ValidationSeverity.WARNING,
                actual_value=value
            )
        
        return result
    
    @staticmethod
    def detect_sudden_change(
        current: float,
        previous: float,
        field_name: str,
        max_change_pct: float = 50.0
    ) -> ValidationResult:
        """
        Detect sudden changes between consecutive values.
        
        Args:
            current: Current value
            previous: Previous value
            field_name: Field name for reporting
            max_change_pct: Maximum allowed percentage change
        """
        result = ValidationResult(valid=True, data=current)
        
        if previous == 0:
            return result
        
        change_pct = abs((current - previous) / previous) * 100
        
        result.metadata["change_pct"] = round(change_pct, 2)
        
        if change_pct > max_change_pct:
            result.add_issue(
                field_name,
                f"Sudden change detected: {change_pct:.1f}% (max: {max_change_pct}%)",
                ValidationSeverity.WARNING,
                actual_value=f"{previous} -> {current}",
                expected=f"change < {max_change_pct}%"
            )
        
        return result


# =============================================================================
# PRE-STORAGE VALIDATION
# =============================================================================

class StorageValidator:
    """Validates data before Supabase insertion"""
    
    # Required fields for each table
    TABLE_SCHEMAS = {
        "articles_raw": {
            "required": ["title", "source_adapter", "collected_at"],
            "optional": ["description", "url", "published_at", "source_name", "source_feed"],
            "max_lengths": {
                "title": 500,
                "description": 5000,
                "url": 2000,
                "source_adapter": 50,
                "source_name": 200,
            }
        },
        "articles_processed": {
            "required": ["raw_article_id", "category", "sentiment", "impact_score", "processed_at"],
            "optional": ["key_entities", "key_metrics", "summary", "relevance_score", "confidence"],
            "max_lengths": {
                "category": 50,
                "sentiment": 20,
                "summary": 5000,
            }
        },
        "bubble_readings": {
            "required": ["date", "vix", "cape", "bubble_index", "bifurcation_score", "regime"],
            "optional": ["credit_spread_ig", "credit_spread_hy", "vix_zscore", "cape_zscore",
                        "credit_zscore", "data_sources", "calculated_at"],
        },
        "stock_snapshots": {
            "required": ["date", "ticker", "price", "change_pct"],
            "optional": ["volume", "market_cap", "pe_ratio", "pillar"],
        }
    }
    
    @staticmethod
    def validate_for_storage(
        data: Dict[str, Any],
        table_name: str,
    ) -> ValidationResult:
        """
        Validate data before Supabase insertion.
        
        Args:
            data: The record to insert
            table_name: Target table name
        """
        result = ValidationResult(valid=True, data=data)
        
        schema = StorageValidator.TABLE_SCHEMAS.get(table_name)
        if not schema:
            result.add_issue(
                "table", f"Unknown table: {table_name}",
                ValidationSeverity.WARNING
            )
            return result
        
        # Check required fields
        for field in schema.get("required", []):
            if field not in data or data[field] is None:
                result.add_issue(
                    field, f"Required field '{field}' missing for {table_name}",
                    ValidationSeverity.ERROR
                )
        
        # Check max lengths for string fields
        max_lengths = schema.get("max_lengths", {})
        for field, max_len in max_lengths.items():
            if field in data and data[field] is not None:
                if isinstance(data[field], str) and len(data[field]) > max_len:
                    # Truncate and warn
                    result.add_issue(
                        field,
                        f"Value truncated from {len(data[field])} to {max_len} chars",
                        ValidationSeverity.WARNING
                    )
                    data[field] = data[field][:max_len]
        
        # Sanitize for JSON storage (handle special types)
        sanitized = StorageValidator._sanitize_for_json(data)
        result.data = sanitized
        
        return result
    
    @staticmethod
    def _sanitize_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for JSON storage"""
        import math
        
        sanitized = {}
        for key, value in data.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, float):
                # Handle NaN and infinity
                if math.isnan(value) or math.isinf(value):
                    sanitized[key] = None
                else:
                    sanitized[key] = value
            elif isinstance(value, datetime):
                sanitized[key] = value.isoformat()
            elif isinstance(value, dict):
                sanitized[key] = StorageValidator._sanitize_for_json(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    StorageValidator._sanitize_for_json(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def batch_validate(
        records: List[Dict[str, Any]],
        table_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[ValidationResult]]:
        """
        Validate a batch of records.
        
        Returns:
            Tuple of (valid_records, failed_results)
        """
        valid_records = []
        failed_results = []
        
        for record in records:
            result = StorageValidator.validate_for_storage(record, table_name)
            if result.valid:
                valid_records.append(result.data)
            else:
                failed_results.append(result)
        
        if failed_results:
            logger.warning(
                f"Batch validation: {len(valid_records)}/{len(records)} valid for {table_name}"
            )
        
        return valid_records, failed_results


# =============================================================================
# VALIDATION DECORATOR
# =============================================================================

def validate_input(
    schema: Optional[Dict[str, Any]] = None,
    range_fields: Optional[List[str]] = None,
    freshness_field: Optional[str] = None,
    freshness_type: str = "market_data",
    raise_on_error: bool = False,
):
    """
    Decorator to validate function inputs.
    
    Args:
        schema: Schema to validate first positional argument against
        range_fields: List of field names to range-validate in first dict arg
        freshness_field: Field containing timestamp to check freshness
        freshness_type: Type for freshness threshold lookup
        raise_on_error: Whether to raise ValidationError on failure
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            validation_results = []
            
            # Get the data to validate (first arg or first kwarg)
            data = args[0] if args else next(iter(kwargs.values()), None)
            
            if data is None:
                if raise_on_error:
                    raise ValueError("No data provided for validation")
                return func(*args, **kwargs)
            
            # Schema validation
            if schema and isinstance(data, dict):
                result = SchemaValidator.validate_schema(data, schema)
                validation_results.append(result)
            
            # Range validation
            if range_fields and isinstance(data, dict):
                for field in range_fields:
                    if field in data:
                        result = RangeValidator.validate_range(data[field], field)
                        validation_results.append(result)
            
            # Freshness validation
            if freshness_field and isinstance(data, dict) and freshness_field in data:
                result = FreshnessValidator.validate_timestamp(
                    data[freshness_field], freshness_type
                )
                validation_results.append(result)
            
            # Check for errors
            has_errors = any(not r.valid for r in validation_results)
            
            if has_errors:
                all_issues = []
                for r in validation_results:
                    all_issues.extend(r.issues)
                
                error_msgs = [
                    f"{i.field}: {i.message}"
                    for i in all_issues
                    if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                ]
                
                if raise_on_error:
                    raise ValueError(f"Validation failed: {'; '.join(error_msgs)}")
                else:
                    for msg in error_msgs:
                        logger.error(f"Validation error: {msg}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_market_indicators(
    vix: Optional[float] = None,
    cape: Optional[float] = None,
    credit_ig: Optional[float] = None,
    credit_hy: Optional[float] = None,
) -> ValidationResult:
    """
    Convenience function to validate all market indicators at once.
    """
    result = ValidationResult(valid=True)
    
    indicators = [
        ("vix", vix),
        ("cape", cape),
        ("credit_spread_ig", credit_ig),
        ("credit_spread_hy", credit_hy),
    ]
    
    validated_data = {}
    
    for field, value in indicators:
        if value is not None:
            field_result = RangeValidator.validate_range(value, field)
            result.issues.extend(field_result.issues)
            if not field_result.valid:
                result.valid = False
            validated_data[field] = field_result.data
            
            # Also check for anomalies
            anomaly_result = AnomalyDetector.detect_anomaly(value, field)
            result.issues.extend(anomaly_result.issues)
    
    result.data = validated_data
    return result


def validate_article_batch(articles: List[Dict[str, Any]]) -> Tuple[List[Dict], ValidationResult]:
    """
    Validate a batch of raw articles.
    
    Returns:
        Tuple of (valid_articles, combined_result)
    """
    result = ValidationResult(valid=True)
    valid_articles = []
    
    for i, article in enumerate(articles):
        # Check required fields
        if not article.get("title"):
            result.add_issue(
                f"article[{i}]", "Missing title",
                ValidationSeverity.ERROR
            )
            continue
        
        # Validate freshness if published_at exists
        if "published_at" in article and article["published_at"]:
            freshness = FreshnessValidator.validate_article_date(article["published_at"])
            result.issues.extend(freshness.issues)
        
        valid_articles.append(article)
    
    result.metadata["total"] = len(articles)
    result.metadata["valid"] = len(valid_articles)
    result.data = valid_articles
    
    return valid_articles, result


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("Validation module loaded")
