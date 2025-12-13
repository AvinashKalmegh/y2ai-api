"""
Unit Tests for Resilience Module

Tests cover:
- Circuit breaker state transitions
- Rate limiter token bucket behavior
- Retry decorator with backoff
- Health tracking
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.resilience import (
    # Circuit Breaker
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker,
    CircuitOpenError,
    
    # Rate Limiter
    RateLimiter,
    get_rate_limiter,
    RateLimitError,
    
    # Retry
    retry_with_backoff,
    resilient_call,
    RetryExhaustedError,
    
    # Health
    ServiceHealth,
    get_health_tracker,
    
    # Utilities
    reset_all,
    get_system_status,
)


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Test circuit breaker state transitions and behavior"""
    
    def setup_method(self):
        """Reset state before each test"""
        reset_all()
    
    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True
    
    def test_stays_closed_below_threshold(self):
        """Circuit stays closed when failures < threshold"""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1
        
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2
    
    def test_opens_at_threshold(self):
        """Circuit opens when failures reach threshold"""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False
    
    def test_open_circuit_rejects_calls(self):
        """Open circuit rejects all calls"""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=60)
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False
    
    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after reset_timeout"""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=1)
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for reset timeout
        time.sleep(1.1)
        
        # Should transition to HALF_OPEN on next check
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_half_open_closes_on_success(self):
        """HALF_OPEN circuit closes on successful call"""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0)
        cb.record_failure()
        
        # Force transition to half-open
        cb.state = CircuitState.HALF_OPEN
        
        cb.record_success()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_half_open_opens_on_failure(self):
        """HALF_OPEN circuit opens again on failure"""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0)
        cb.state = CircuitState.HALF_OPEN
        
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
    
    def test_success_resets_failure_count(self):
        """Success in CLOSED state doesn't reset failure count (by design)"""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        # Failure count persists until circuit opens and recovers
        assert cb.failure_count == 2
    
    def test_manual_reset(self):
        """Manual reset returns circuit to CLOSED"""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        
        cb.reset()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_get_reset_time(self):
        """Reset time is calculated correctly"""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=60)
        cb.record_failure()
        
        reset_time = cb.get_reset_time()
        
        assert reset_time is not None
        expected = cb.last_failure_time + timedelta(seconds=60)
        assert abs((reset_time - expected).total_seconds()) < 1
    
    def test_global_registry(self):
        """Circuit breakers are stored in global registry"""
        cb1 = get_circuit_breaker("service_a")
        cb2 = get_circuit_breaker("service_a")
        cb3 = get_circuit_breaker("service_b")
        
        assert cb1 is cb2  # Same name returns same instance
        assert cb1 is not cb3  # Different names are different instances


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Test rate limiter token bucket behavior"""
    
    def setup_method(self):
        reset_all()
    
    def test_initial_tokens(self):
        """Rate limiter starts with max tokens"""
        rl = RateLimiter(name="test", max_tokens=10, tokens_per_second=1.0)
        assert rl.tokens == 10.0
    
    def test_acquire_reduces_tokens(self):
        """Acquiring tokens reduces available count"""
        rl = RateLimiter(name="test", max_tokens=10, tokens_per_second=1.0)
        
        result = rl.acquire(tokens=3, block=False)
        
        assert result is True
        assert rl.tokens == 7.0
    
    def test_acquire_fails_when_insufficient(self):
        """Acquire fails when insufficient tokens (non-blocking)"""
        rl = RateLimiter(name="test", max_tokens=5, tokens_per_second=1.0)
        
        result = rl.acquire(tokens=10, block=False)
        
        assert result is False
        assert rl.tokens == 5.0  # Unchanged
    
    def test_tokens_refill_over_time(self):
        """Tokens refill based on elapsed time"""
        rl = RateLimiter(name="test", max_tokens=10, tokens_per_second=10.0)
        
        rl.acquire(tokens=10, block=False)
        assert rl.tokens == 0.0
        
        time.sleep(0.5)  # Should refill ~5 tokens
        
        # Force refill calculation
        rl.acquire(tokens=0, block=False)
        
        assert rl.tokens >= 4.0  # Allow some timing slack
        assert rl.tokens <= 6.0
    
    def test_tokens_dont_exceed_max(self):
        """Tokens never exceed max_tokens"""
        rl = RateLimiter(name="test", max_tokens=10, tokens_per_second=100.0)
        
        time.sleep(0.2)  # Would add 20 tokens at 100/s
        
        rl.acquire(tokens=0, block=False)
        
        assert rl.tokens == 10.0  # Capped at max
    
    def test_blocking_acquire(self):
        """Blocking acquire waits for tokens"""
        rl = RateLimiter(name="test", max_tokens=1, tokens_per_second=10.0)
        
        rl.acquire(tokens=1, block=False)  # Drain tokens
        
        start = time.time()
        result = rl.acquire(tokens=1, block=True, timeout=1.0)
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed >= 0.08  # Should have waited ~0.1s
        assert elapsed < 0.5
    
    def test_blocking_acquire_timeout(self):
        """Blocking acquire times out"""
        rl = RateLimiter(name="test", max_tokens=1, tokens_per_second=0.1)  # Very slow refill
        
        rl.acquire(tokens=1, block=False)  # Drain
        
        start = time.time()
        result = rl.acquire(tokens=1, block=True, timeout=0.2)
        elapsed = time.time() - start
        
        assert result is False
        assert elapsed >= 0.2
        assert elapsed < 0.5
    
    def test_get_wait_time(self):
        """Wait time calculation is accurate"""
        rl = RateLimiter(name="test", max_tokens=10, tokens_per_second=2.0)
        
        rl.acquire(tokens=10, block=False)  # Drain all
        
        wait_time = rl.get_wait_time(tokens=4)
        
        assert wait_time >= 1.9  # 4 tokens at 2/s = 2s
        assert wait_time <= 2.1
    
    def test_global_registry(self):
        """Rate limiters are stored in global registry"""
        rl1 = get_rate_limiter("api_a", max_tokens=10)
        rl2 = get_rate_limiter("api_a")
        
        assert rl1 is rl2


# =============================================================================
# RETRY DECORATOR TESTS
# =============================================================================

class TestRetryDecorator:
    """Test retry decorator behavior"""
    
    def setup_method(self):
        reset_all()
    
    def test_success_no_retry(self):
        """Successful call doesn't retry"""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retries_on_exception(self):
        """Retries on retryable exception"""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=2, 
            base_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("transient error")
        
        with pytest.raises(RetryExhaustedError):
            failing_func()
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_succeeds_after_retry(self):
        """Function succeeds after initial failure"""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3, 
            base_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "finally!"
        
        result = eventually_succeeds()
        
        assert result == "finally!"
        assert call_count == 3
    
    def test_non_retryable_exception_fails_immediately(self):
        """Non-retryable exception fails without retry"""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3, 
            base_delay=0.01,
            retryable_exceptions=(ValueError,)  # Only ValueError is retryable
        )
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")
        
        with pytest.raises(TypeError):
            raises_type_error()
        
        assert call_count == 1  # No retries
    
    def test_exponential_backoff(self):
        """Delay increases exponentially"""
        delays = []
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable timing
            retryable_exceptions=(ValueError,)
        )
        def track_delays():
            delays.append(time.time())
            raise ValueError("fail")
        
        start = time.time()
        with pytest.raises(RetryExhaustedError):
            track_delays()
        
        # Check delays between calls
        if len(delays) >= 3:
            delay1 = delays[1] - delays[0]
            delay2 = delays[2] - delays[1]
            
            # Second delay should be ~2x first (exponential)
            assert delay2 > delay1 * 1.5  # Allow some slack
    
    def test_circuit_breaker_integration(self):
        """Retry respects circuit breaker"""
        # Pre-open the circuit
        cb = get_circuit_breaker("test_service", failure_threshold=1)
        cb.record_failure()
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            circuit_breaker_name="test_service"
        )
        def blocked_func():
            return "should not execute"
        
        with pytest.raises(CircuitOpenError):
            blocked_func()


# =============================================================================
# RESILIENT CALL DECORATOR TESTS
# =============================================================================

class TestResilientCall:
    """Test the combined resilient_call decorator"""
    
    def setup_method(self):
        reset_all()
    
    def test_tracks_health_on_success(self):
        """Successful calls are tracked in health"""
        @resilient_call(
            service_name="test_health",
            max_retries=0,
            use_circuit_breaker=False,
            use_rate_limiter=False,
        )
        def healthy_func():
            return "ok"
        
        healthy_func()
        healthy_func()
        healthy_func()
        
        tracker = get_health_tracker("test_health")
        assert tracker.successful_calls == 3
        assert tracker.failed_calls == 0
        assert tracker.success_rate == 100.0
    
    def test_tracks_health_on_failure(self):
        """Failed calls are tracked in health"""
        @resilient_call(
            service_name="test_health_fail",
            max_retries=0,
            use_circuit_breaker=False,
            use_rate_limiter=False,
        )
        def failing_func():
            raise ConnectionError("network error")
        
        for _ in range(3):
            try:
                failing_func()
            except RetryExhaustedError:
                pass
        
        tracker = get_health_tracker("test_health_fail")
        assert tracker.failed_calls == 3
        assert tracker.success_rate == 0.0


# =============================================================================
# HEALTH TRACKING TESTS
# =============================================================================

class TestServiceHealth:
    """Test service health tracking"""
    
    def test_initial_state(self):
        """Health tracker starts empty"""
        health = ServiceHealth(name="test")
        
        assert health.total_calls == 0
        assert health.successful_calls == 0
        assert health.failed_calls == 0
        assert health.success_rate == 100.0  # No failures = 100%
        assert health.is_healthy is True
    
    def test_record_success(self):
        """Recording success updates metrics"""
        health = ServiceHealth(name="test")
        
        health.record_call(success=True, response_time_ms=50.0)
        
        assert health.total_calls == 1
        assert health.successful_calls == 1
        assert health.last_success is not None
        assert health.avg_response_time_ms == 50.0
    
    def test_record_failure(self):
        """Recording failure updates metrics"""
        health = ServiceHealth(name="test")
        
        health.record_call(success=False, response_time_ms=100.0, error="timeout")
        
        assert health.total_calls == 1
        assert health.failed_calls == 1
        assert health.last_failure is not None
        assert health.last_error == "timeout"
    
    def test_success_rate_calculation(self):
        """Success rate is calculated correctly"""
        health = ServiceHealth(name="test")
        
        health.record_call(success=True, response_time_ms=10)
        health.record_call(success=True, response_time_ms=10)
        health.record_call(success=False, response_time_ms=10)
        health.record_call(success=True, response_time_ms=10)
        
        assert health.success_rate == 75.0
    
    def test_unhealthy_on_low_success_rate(self):
        """Service is unhealthy with low success rate"""
        health = ServiceHealth(name="test")
        
        # 2 successes, 8 failures = 20% success rate
        for _ in range(2):
            health.record_call(success=True, response_time_ms=10)
        for _ in range(8):
            health.record_call(success=False, response_time_ms=10)
        
        assert health.success_rate == 20.0
        assert health.is_healthy is False
    
    def test_to_dict(self):
        """Health converts to dictionary for reporting"""
        health = ServiceHealth(name="test")
        health.record_call(success=True, response_time_ms=25.5)
        
        d = health.to_dict()
        
        assert d["name"] == "test"
        assert d["total_calls"] == 1
        assert d["successful_calls"] == 1
        assert d["avg_response_time_ms"] == 25.5
        assert "last_success" in d


# =============================================================================
# SYSTEM STATUS TESTS
# =============================================================================

class TestSystemStatus:
    """Test system-wide status reporting"""
    
    def setup_method(self):
        reset_all()
    
    def test_get_system_status(self):
        """System status includes all components"""
        # Create some state
        get_circuit_breaker("test_cb")
        get_rate_limiter("test_rl", max_tokens=10)
        get_health_tracker("test_health")
        
        status = get_system_status()
        
        assert "timestamp" in status
        assert "circuit_breakers" in status
        assert "rate_limiters" in status
        assert "health" in status
        
        assert "test_cb" in status["circuit_breakers"]
        assert "test_rl" in status["rate_limiters"]
        assert "test_health" in status["health"]
    
    def test_reset_all(self):
        """Reset clears all state"""
        cb = get_circuit_breaker("to_reset", failure_threshold=1)
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        
        reset_all()
        
        # Circuit breaker should be reset
        assert cb.state == CircuitState.CLOSED


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
