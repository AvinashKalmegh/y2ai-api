"""
ARGUS+Y2AI Shared Utilities

This package contains shared infrastructure:
- Circuit breakers for failing services
- Rate limiters for API quotas
- Retry logic with exponential backoff
- Health tracking for all services
- HTTP session management with connection pooling
- Data validation and anomaly detection
"""

from .resilience import (
    # Exceptions
    ResilienceError,
    RetryExhaustedError,
    CircuitOpenError,
    RateLimitError,
    ServiceUnavailableError,
    
    # Circuit Breaker
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker,
    
    # Rate Limiter
    RateLimiter,
    get_rate_limiter,
    RATE_LIMIT_CONFIGS,
    
    # Decorators
    retry_with_backoff,
    resilient_call,
    tracked_call,
    with_fallback,
    
    # HTTP Session
    create_robust_session,
    get_http_session,
    
    # Health Tracking
    ServiceHealth,
    get_health_tracker,
    get_all_health_status,
    
    # Utilities
    aggregate_with_partial_failure,
    reset_all,
    get_system_status,
)

__version__ = "1.1.0"

# Import validation components
from .validation import (
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
    
    # Schemas
    RANGE_CONSTRAINTS,
    ANOMALY_THRESHOLDS,
    NEWSAPI_ARTICLE_SCHEMA,
    ALPHA_VANTAGE_ARTICLE_SCHEMA,
    FRED_OBSERVATION_SCHEMA,
    YFINANCE_QUOTE_SCHEMA,
    PROCESSED_ARTICLE_SCHEMA,
    BUBBLE_INDEX_READING_SCHEMA,
    STOCK_REPORT_SCHEMA,
    
    # Decorator
    validate_input,
    
    # Convenience functions
    validate_market_indicators,
    validate_article_batch,
)

__all__ = [
    # Exceptions
    "ResilienceError",
    "RetryExhaustedError", 
    "CircuitOpenError",
    "RateLimitError",
    "ServiceUnavailableError",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "get_circuit_breaker",
    
    # Rate Limiter
    "RateLimiter",
    "get_rate_limiter",
    "RATE_LIMIT_CONFIGS",
    
    # Decorators
    "retry_with_backoff",
    "resilient_call",
    "tracked_call",
    "with_fallback",
    
    # HTTP Session
    "create_robust_session",
    "get_http_session",
    
    # Health Tracking
    "ServiceHealth",
    "get_health_tracker",
    "get_all_health_status",
    
    # Utilities
    "aggregate_with_partial_failure",
    "reset_all",
    "get_system_status",
    
    # Validation - Result types
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    
    # Validation - Validators
    "RangeValidator",
    "SchemaValidator",
    "FreshnessValidator",
    "AnomalyDetector",
    "StorageValidator",
    
    # Validation - Schemas
    "RANGE_CONSTRAINTS",
    "ANOMALY_THRESHOLDS",
    "NEWSAPI_ARTICLE_SCHEMA",
    "ALPHA_VANTAGE_ARTICLE_SCHEMA",
    "FRED_OBSERVATION_SCHEMA",
    "YFINANCE_QUOTE_SCHEMA",
    "PROCESSED_ARTICLE_SCHEMA",
    "BUBBLE_INDEX_READING_SCHEMA",
    "STOCK_REPORT_SCHEMA",
    
    # Validation - Decorator
    "validate_input",
    
    # Validation - Convenience
    "validate_market_indicators",
    "validate_article_batch",
]
