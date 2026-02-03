from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar, Optional

from tradebot.core.logger import get_logger

log = get_logger("circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents repeated calls to a failing service.

    States:
        CLOSED: Normal operation. Failures are counted.
        OPEN: Circuit is tripped. All calls fail immediately.
        HALF_OPEN: Testing if service recovered. Limited calls allowed.

    Usage:
        breaker = CircuitBreaker(name="alpaca")

        try:
            result = breaker.call(lambda: api_client.get_account())
        except CircuitOpenError:
            # Service is unavailable, use fallback
            pass
    """

    name: str
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before half-open
    half_open_max_calls: int = 1  # Calls allowed in half-open

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        return time.monotonic() - self._last_failure_time >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        if old_state != new_state:
            log.info(f"Circuit '{self.name}': {old_state.value} -> {new_state.value}")

    def call(self, func: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            fallback: Optional fallback function if circuit is open

        Returns:
            Result of func() or fallback()

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_try_reset():
                self._transition_to(CircuitState.HALF_OPEN)

            # Reject if OPEN
            if self._state == CircuitState.OPEN:
                if fallback:
                    return fallback()
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open. "
                    f"Retry after {self.recovery_timeout - (time.monotonic() - self._last_failure_time):.1f}s"
                )

            # Limit calls in HALF_OPEN
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    if fallback:
                        return fallback()
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is half-open and at capacity"
                    )
                self._half_open_calls += 1

        # Execute the call outside the lock
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Recovery successful
                self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._success_count += 1
                # Reset failure count after consecutive successes
                if self._success_count >= 2:
                    self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Recovery failed
                self._transition_to(CircuitState.OPEN)
                log.warning(f"Circuit '{self.name}' reopened after failed recovery: {error}")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    log.warning(
                        f"Circuit '{self.name}' opened after {self._failure_count} failures: {error}"
                    )

    def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            log.info(f"Circuit '{self.name}' manually reset")

    def get_stats(self) -> dict:
        """Return current circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
            }


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""
    pass


# Global circuit breakers for different services
_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> CircuitBreaker:
    """Get or create a circuit breaker for a service.

    Args:
        name: Service identifier
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before testing recovery

    Returns:
        CircuitBreaker instance for the service
    """
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _breakers[name]


def reset_all_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    with _breakers_lock:
        for breaker in _breakers.values():
            breaker.reset()
