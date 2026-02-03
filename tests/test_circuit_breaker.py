"""Tests for CircuitBreaker."""
import pytest
import time
from unittest.mock import MagicMock

from tradebot.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker,
    reset_all_breakers,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create a circuit breaker with test settings."""
        return CircuitBreaker(
            name="test",
            failure_threshold=3,
            recovery_timeout=0.1,  # 100ms for faster tests
        )

    def test_initial_state_closed(self, breaker: CircuitBreaker):
        """Test that circuit starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True

    def test_successful_call(self, breaker: CircuitBreaker):
        """Test successful call through circuit."""
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_failed_call_increments_count(self, breaker: CircuitBreaker):
        """Test that failed calls increment failure count."""
        def failing_func():
            raise ValueError("test error")

        # First failure shouldn't open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self, breaker: CircuitBreaker):
        """Test that circuit opens after reaching failure threshold."""
        def failing_func():
            raise ValueError("test error")

        # Fail until threshold
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self, breaker: CircuitBreaker):
        """Test that open circuit rejects calls with CircuitOpenError."""
        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "should not run")

    def test_circuit_uses_fallback_when_open(self, breaker: CircuitBreaker):
        """Test that open circuit uses fallback function."""
        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Call with fallback
        result = breaker.call(
            lambda: "should not run",
            fallback=lambda: "fallback value",
        )
        assert result == "fallback value"

    def test_circuit_transitions_to_half_open(self, breaker: CircuitBreaker):
        """Test that circuit transitions to half-open after timeout."""
        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Wait for recovery timeout
        time.sleep(breaker.recovery_timeout + 0.05)

        # Next call should be allowed (half-open)
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED  # Should close on success

    def test_half_open_failure_reopens_circuit(self, breaker: CircuitBreaker):
        """Test that failure in half-open state reopens circuit."""
        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Wait for recovery timeout
        time.sleep(breaker.recovery_timeout + 0.05)

        # Fail again in half-open state
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Circuit should reopen
        assert breaker.state == CircuitState.OPEN

    def test_manual_reset(self, breaker: CircuitBreaker):
        """Test manual circuit reset."""
        def failing_func():
            raise ValueError("test error")

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        result = breaker.call(lambda: "success")
        assert result == "success"

    def test_get_stats(self, breaker: CircuitBreaker):
        """Test getting circuit breaker statistics."""
        stats = breaker.get_stats()

        assert "name" in stats
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert stats["name"] == "test"
        assert stats["state"] == "closed"


class TestCircuitBreakerGlobalFunctions:
    """Tests for global circuit breaker functions."""

    def test_get_circuit_breaker_creates_new(self):
        """Test that get_circuit_breaker creates new breaker."""
        breaker = get_circuit_breaker("test_service")
        assert breaker is not None
        assert breaker.name == "test_service"

    def test_get_circuit_breaker_returns_same_instance(self):
        """Test that get_circuit_breaker returns same instance."""
        breaker1 = get_circuit_breaker("same_service")
        breaker2 = get_circuit_breaker("same_service")
        assert breaker1 is breaker2

    def test_reset_all_breakers(self):
        """Test resetting all circuit breakers."""
        # Get some breakers and open them
        breaker1 = get_circuit_breaker("service1", failure_threshold=1)
        breaker2 = get_circuit_breaker("service2", failure_threshold=1)

        # Open both
        with pytest.raises(Exception):
            breaker1.call(lambda: 1/0)

        with pytest.raises(Exception):
            breaker2.call(lambda: 1/0)

        # Reset all
        reset_all_breakers()

        # Both should be closed now
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED
