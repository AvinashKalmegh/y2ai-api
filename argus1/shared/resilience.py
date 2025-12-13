"""
ARGUS+Y2AI RESILIENCE MODULE
Shared error handling, retry logic, circuit breakers, and rate limiting

This module provides robust error handling for all API calls across:
- News API adapters (NewsAPI, Alpha Vantage, SEC EDGAR, RSS)
- Claude API for article processing
- Financial data APIs (yfinance, FRED)
- Social media APIs (Twitter, LinkedIn, Bluesky)
- Supabase storage operations
"""

import time
import logging
import functools
import random
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any, Type, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ResilienceError(Exception):
    """Base exception for resilience-related errors"""
    pass


class RetryExhaustedError(ResilienceError):
    """All retry attempts have been exhausted"""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


class CircuitOpenError(ResilienceError):
    """Circuit breaker is open, calls are being rejected"""
    def __init__(self, service_name: str, reset_time: datetime):
        self.service_name = service_name
        self.reset_time = reset_time
        super().__init__(f"Circuit open for {service_name}, resets at {reset_time}")


class RateLimitError(ResilienceError):
    """Rate limit has been hit"""
    def __init__(self, service_name: str, retry_after: Optional[int] = None):
        self.service_name = service_name
        self.retry_after = retry_after
        msg = f"Rate limit hit for {service_name}"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(msg)


class ServiceUnavailableError(ResilienceError):
    """Service is temporarily unavailable"""
    pass


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, reject all requests immediately
    - HALF_OPEN: Allow one test request to check if service recovered
    
    Transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: After reset_timeout seconds
    - HALF_OPEN -> CLOSED: On successful test request
    - HALF_OPEN -> OPEN: On failed test request
    """
    name: str
    failure_threshold: int = 5
    reset_timeout: int = 60  # seconds
    half_open_max_calls: int = 1
    
    # State tracking
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    half_open_calls: int = field(default=0)
    _lock: Lock = field(default_factory=Lock)
    
    def __post_init__(self):
        self._lock = Lock()
    
    def can_execute(self) -> bool:
        """Check if a call can be executed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if reset timeout has passed
                if self.last_failure_time:
                    elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if elapsed >= self.reset_timeout:
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                        return True
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self):
        """Record a successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (service recovered)")
            else:
                self.success_count += 1
    
    def record_failure(self, exception: Optional[Exception] = None):
        """Record a failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (test request failed)")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit {self.name}: CLOSED -> OPEN "
                        f"(failures: {self.failure_count}, threshold: {self.failure_threshold})"
                    )
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get when the circuit will reset to half-open"""
        if self.state == CircuitState.OPEN and self.last_failure_time:
            return self.last_failure_time + timedelta(seconds=self.reset_timeout)
        return None
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit {self.name}: manually reset to CLOSED")


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_cb_lock = Lock()


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker by name"""
    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return _circuit_breakers[name]


# =============================================================================
# RATE LIMITER
# =============================================================================

@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.
    
    Allows bursts up to max_tokens, refills at rate tokens_per_second.
    """
    name: str
    max_tokens: int = 10
    tokens_per_second: float = 1.0
    
    tokens: float = field(default=None)
    last_update: datetime = field(default=None)
    _lock: Lock = field(default_factory=Lock)
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = float(self.max_tokens)
        if self.last_update is None:
            self.last_update = datetime.utcnow()
        self._lock = Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = datetime.utcnow()
        elapsed = (now - self.last_update).total_seconds()
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.tokens_per_second)
        self.last_update = now
    
    def acquire(self, tokens: int = 1, block: bool = True, timeout: float = 30.0) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            block: If True, wait for tokens; if False, return immediately
            timeout: Max seconds to wait if blocking
        
        Returns:
            True if tokens acquired, False otherwise
        """
        deadline = datetime.utcnow() + timedelta(seconds=timeout) if block else None
        
        while True:
            with self._lock:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                if not block:
                    return False
                
                # Calculate wait time
                needed = tokens - self.tokens
                wait_time = needed / self.tokens_per_second
            
            # Check if we've passed the deadline
            now = datetime.utcnow()
            if deadline and now >= deadline:
                return False
            
            # If wait time would exceed deadline, wait until deadline then return False
            if deadline:
                remaining = (deadline - now).total_seconds()
                if wait_time > remaining:
                    time.sleep(remaining)
                    return False
            
            # Wait and retry
            time.sleep(min(wait_time, 1.0))
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens"""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.tokens_per_second


# Global rate limiter registry
_rate_limiters: Dict[str, RateLimiter] = {}
_rl_lock = Lock()


def get_rate_limiter(name: str, **kwargs) -> RateLimiter:
    """Get or create a rate limiter by name"""
    with _rl_lock:
        if name not in _rate_limiters:
            _rate_limiters[name] = RateLimiter(name=name, **kwargs)
        return _rate_limiters[name]


# Pre-configured rate limiters for known services
RATE_LIMIT_CONFIGS = {
    "newsapi": {"max_tokens": 100, "tokens_per_second": 0.016},  # ~1000/day
    "alphavantage": {"max_tokens": 5, "tokens_per_second": 0.00029},  # 25/day
    "anthropic": {"max_tokens": 50, "tokens_per_second": 0.83},  # ~50/min
    "fred": {"max_tokens": 120, "tokens_per_second": 2.0},  # 120/min
    "twitter": {"max_tokens": 50, "tokens_per_second": 0.033},  # 50/15min
    "bluesky": {"max_tokens": 100, "tokens_per_second": 0.33},  # 100/5min
    "supabase": {"max_tokens": 100, "tokens_per_second": 10.0},  # generous
}


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
    ),
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
    circuit_breaker_name: Optional[str] = None,
    rate_limiter_name: Optional[str] = None,
):
    """
    Decorator that adds retry logic with exponential backoff.
    
    Features:
    - Exponential backoff with optional jitter
    - Integration with circuit breaker
    - Integration with rate limiter
    - Configurable retryable exceptions and status codes
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions that should trigger retry
        retryable_status_codes: HTTP status codes that should trigger retry
        circuit_breaker_name: Name of circuit breaker to use
        rate_limiter_name: Name of rate limiter to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get circuit breaker if configured
            circuit = None
            if circuit_breaker_name:
                circuit = get_circuit_breaker(circuit_breaker_name)
                if not circuit.can_execute():
                    reset_time = circuit.get_reset_time()
                    raise CircuitOpenError(circuit_breaker_name, reset_time)
            
            # Get rate limiter if configured
            limiter = None
            if rate_limiter_name:
                config = RATE_LIMIT_CONFIGS.get(rate_limiter_name, {})
                limiter = get_rate_limiter(rate_limiter_name, **config)
                if not limiter.acquire(block=True, timeout=30.0):
                    raise RateLimitError(rate_limiter_name)
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check for retryable HTTP status codes in response
                    if hasattr(result, 'status_code'):
                        if result.status_code in retryable_status_codes:
                            # Handle rate limit specifically
                            if result.status_code == 429:
                                retry_after = result.headers.get('Retry-After')
                                if retry_after:
                                    wait_time = int(retry_after)
                                    logger.warning(
                                        f"Rate limited, waiting {wait_time}s "
                                        f"(attempt {attempt + 1}/{max_retries + 1})"
                                    )
                                    time.sleep(wait_time)
                                    continue
                            
                            raise ServiceUnavailableError(
                                f"HTTP {result.status_code}: {result.text[:200]}"
                            )
                    
                    # Success - record it
                    if circuit:
                        circuit.record_success()
                    
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        if circuit:
                            circuit.record_failure(e)
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    # Non-retryable exception
                    if circuit:
                        circuit.record_failure(e)
                    raise
            
            # All retries exhausted
            if circuit:
                circuit.record_failure(last_exception)
            
            raise RetryExhaustedError(
                f"All {max_retries + 1} attempts failed",
                last_exception=last_exception
            )
        
        return wrapper
    return decorator


# =============================================================================
# HTTP SESSION FACTORY
# =============================================================================

def create_robust_session(
    total_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (500, 502, 503, 504),
    pool_connections: int = 10,
    pool_maxsize: int = 10,
    timeout: float = 30.0,
) -> requests.Session:
    """
    Create a requests Session with built-in retry logic and connection pooling.
    
    This is useful for HTTP calls that don't need the full decorator approach.
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default timeout
    session.request = functools.partial(session.request, timeout=timeout)
    
    return session


# Global session for reuse
_http_session: Optional[requests.Session] = None
_session_lock = Lock()


def get_http_session() -> requests.Session:
    """Get the global robust HTTP session"""
    global _http_session
    with _session_lock:
        if _http_session is None:
            _http_session = create_robust_session()
        return _http_session


# =============================================================================
# HEALTH TRACKING
# =============================================================================

@dataclass
class ServiceHealth:
    """Track health metrics for a service"""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_error: Optional[str] = None
    avg_response_time_ms: float = 0.0
    _response_times: List[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    
    def __post_init__(self):
        self._lock = Lock()
        self._response_times = []
    
    def record_call(self, success: bool, response_time_ms: float, error: Optional[str] = None):
        """Record a call result"""
        with self._lock:
            self.total_calls += 1
            
            if success:
                self.successful_calls += 1
                self.last_success = datetime.utcnow()
            else:
                self.failed_calls += 1
                self.last_failure = datetime.utcnow()
                self.last_error = error
            
            # Track response times (keep last 100)
            self._response_times.append(response_time_ms)
            if len(self._response_times) > 100:
                self._response_times.pop(0)
            
            self.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_calls == 0:
            return 100.0
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is considered healthy"""
        # Healthy if >80% success rate and last success within 1 hour
        if self.success_rate < 80:
            return False
        if self.last_success:
            if (datetime.utcnow() - self.last_success).total_seconds() > 3600:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(self.success_rate, 2),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_error": self.last_error,
            "is_healthy": self.is_healthy,
        }


# Global health registry
_health_trackers: Dict[str, ServiceHealth] = {}
_health_lock = Lock()


def get_health_tracker(name: str) -> ServiceHealth:
    """Get or create a health tracker by name"""
    with _health_lock:
        if name not in _health_trackers:
            _health_trackers[name] = ServiceHealth(name=name)
        return _health_trackers[name]


def get_all_health_status() -> Dict[str, Dict[str, Any]]:
    """Get health status for all tracked services"""
    with _health_lock:
        return {name: tracker.to_dict() for name, tracker in _health_trackers.items()}


# =============================================================================
# CONVENIENCE DECORATOR FOR TRACKED CALLS
# =============================================================================

def tracked_call(service_name: str):
    """
    Decorator to track call health metrics.
    
    Usage:
        @tracked_call("newsapi")
        def fetch_news():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_health_tracker(service_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                tracker.record_call(success=True, response_time_ms=response_time)
                return result
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                tracker.record_call(
                    success=False, 
                    response_time_ms=response_time,
                    error=str(e)[:200]
                )
                raise
        
        return wrapper
    return decorator


# =============================================================================
# COMBINED RESILIENT CALL DECORATOR
# =============================================================================

def resilient_call(
    service_name: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    use_circuit_breaker: bool = True,
    use_rate_limiter: bool = True,
    circuit_failure_threshold: int = 5,
    circuit_reset_timeout: int = 60,
):
    """
    All-in-one decorator combining retry, circuit breaker, rate limiting, and health tracking.
    
    This is the recommended decorator for most API calls in the system.
    
    Usage:
        @resilient_call("newsapi")
        def fetch_from_newsapi():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get circuit breaker
            circuit = None
            if use_circuit_breaker:
                circuit = get_circuit_breaker(
                    service_name,
                    failure_threshold=circuit_failure_threshold,
                    reset_timeout=circuit_reset_timeout,
                )
                if not circuit.can_execute():
                    reset_time = circuit.get_reset_time()
                    raise CircuitOpenError(service_name, reset_time)
            
            # Get rate limiter
            limiter = None
            if use_rate_limiter and service_name in RATE_LIMIT_CONFIGS:
                config = RATE_LIMIT_CONFIGS[service_name]
                limiter = get_rate_limiter(service_name, **config)
                if not limiter.acquire(block=True, timeout=30.0):
                    raise RateLimitError(service_name)
            
            # Get health tracker
            tracker = get_health_tracker(service_name)
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success
                    response_time = (time.time() - start_time) * 1000
                    tracker.record_call(success=True, response_time_ms=response_time)
                    
                    if circuit:
                        circuit.record_success()
                    
                    return result
                    
                except (requests.exceptions.RequestException, ConnectionError, 
                        TimeoutError, ServiceUnavailableError) as e:
                    last_exception = e
                    response_time = (time.time() - start_time) * 1000
                    
                    if attempt == max_retries:
                        tracker.record_call(
                            success=False,
                            response_time_ms=response_time,
                            error=str(e)[:200]
                        )
                        if circuit:
                            circuit.record_failure(e)
                        break
                    
                    # Calculate backoff delay
                    delay = min(base_delay * (2 ** attempt), 60.0)
                    delay = delay * (0.5 + random.random())  # Add jitter
                    
                    logger.warning(
                        f"[{service_name}] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    tracker.record_call(
                        success=False,
                        response_time_ms=response_time,
                        error=str(e)[:200]
                    )
                    if circuit:
                        circuit.record_failure(e)
                    raise
            
            raise RetryExhaustedError(
                f"[{service_name}] All {max_retries + 1} attempts failed",
                last_exception=last_exception
            )
        
        return wrapper
    return decorator


# =============================================================================
# GRACEFUL DEGRADATION HELPERS
# =============================================================================

def with_fallback(fallback_value: Any):
    """
    Decorator that returns a fallback value on failure instead of raising.
    
    Usage:
        @with_fallback([])
        def get_articles():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__} failed, returning fallback: {e}")
                return fallback_value
        return wrapper
    return decorator


def aggregate_with_partial_failure(
    funcs: List[Callable],
    combine_func: Callable[[List[Any]], Any] = lambda x: x,
) -> Any:
    """
    Execute multiple functions and combine results, tolerating partial failures.
    
    Returns results from successful calls even if some fail.
    
    Usage:
        results = aggregate_with_partial_failure(
            [fetch_newsapi, fetch_rss, fetch_sec],
            combine_func=lambda results: [item for sublist in results for item in sublist]
        )
    """
    results = []
    errors = []
    
    for func in funcs:
        try:
            result = func()
            if result is not None:
                results.append(result)
        except Exception as e:
            errors.append((func.__name__, str(e)))
            logger.warning(f"Partial failure in {func.__name__}: {e}")
    
    if errors:
        logger.info(f"Completed with {len(errors)} failures out of {len(funcs)} calls")
    
    return combine_func(results)


# =============================================================================
# INITIALIZATION AND STATUS
# =============================================================================

def reset_all():
    """Reset all circuit breakers and clear rate limiter tokens (for testing)"""
    global _circuit_breakers, _rate_limiters, _health_trackers
    
    with _cb_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
    
    with _rl_lock:
        _rate_limiters.clear()
    
    with _health_lock:
        _health_trackers.clear()
    
    logger.info("All resilience state reset")


def get_system_status() -> Dict[str, Any]:
    """Get complete system resilience status"""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "circuit_breakers": {},
        "rate_limiters": {},
        "health": {},
    }
    
    with _cb_lock:
        for name, cb in _circuit_breakers.items():
            status["circuit_breakers"][name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "reset_time": cb.get_reset_time().isoformat() if cb.get_reset_time() else None,
            }
    
    with _rl_lock:
        for name, rl in _rate_limiters.items():
            status["rate_limiters"][name] = {
                "tokens_available": round(rl.tokens, 2),
                "max_tokens": rl.max_tokens,
            }
    
    status["health"] = get_all_health_status()
    
    return status


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("Resilience module loaded")
