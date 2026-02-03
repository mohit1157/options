from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from tradebot.core.logger import get_logger

log = get_logger("rate_limiter")


@dataclass
class TokenBucket:
    """Token bucket rate limiter.

    Tokens are added at a fixed rate up to a maximum capacity.
    Each request consumes one token. If no tokens are available,
    the request must wait.
    """

    capacity: float  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire (default: 1)
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if tokens were acquired, False if timeout occurred.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Calculate wait time for enough tokens
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

                # Release lock while waiting
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting.

        Returns:
            True if tokens were acquired, False if not enough tokens.
        """
        return self.acquire(tokens, timeout=0)

    @property
    def available(self) -> float:
        """Return current available tokens."""
        with self._lock:
            self._refill()
            return self.tokens


@dataclass
class RateLimiter:
    """Multi-endpoint rate limiter with separate buckets per endpoint.

    Usage:
        limiter = RateLimiter()
        limiter.add_endpoint("user_lookup", requests_per_window=100, window_seconds=900)
        limiter.add_endpoint("tweets", requests_per_window=10000, window_seconds=2592000)

        if limiter.acquire("user_lookup"):
            # Make request
            pass
    """

    _buckets: Dict[str, TokenBucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_endpoint(
        self,
        name: str,
        requests_per_window: int,
        window_seconds: int,
    ) -> None:
        """Add a rate-limited endpoint.

        Args:
            name: Endpoint identifier
            requests_per_window: Max requests allowed per window
            window_seconds: Time window in seconds
        """
        # Calculate tokens per second
        refill_rate = requests_per_window / window_seconds

        # Use a burst capacity of 10% of the window limit, minimum 1
        burst = max(1.0, requests_per_window * 0.1)

        with self._lock:
            self._buckets[name] = TokenBucket(
                capacity=burst,
                refill_rate=refill_rate,
            )

        log.debug(
            f"Rate limit configured: {name} = {requests_per_window}/{window_seconds}s "
            f"(burst={burst:.1f}, rate={refill_rate:.4f}/s)"
        )

    def acquire(
        self,
        endpoint: str,
        tokens: float = 1.0,
        timeout: Optional[float] = 30.0,
    ) -> bool:
        """Acquire tokens for an endpoint, blocking if necessary.

        Args:
            endpoint: Endpoint name
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds

        Returns:
            True if acquired, False if timeout or unknown endpoint.
        """
        with self._lock:
            bucket = self._buckets.get(endpoint)

        if bucket is None:
            log.warning(f"Unknown endpoint: {endpoint}, allowing request")
            return True

        acquired = bucket.acquire(tokens, timeout)
        if not acquired:
            log.warning(f"Rate limit timeout for endpoint: {endpoint}")
        return acquired

    def try_acquire(self, endpoint: str, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting.

        Returns:
            True if acquired, False if not enough tokens.
        """
        with self._lock:
            bucket = self._buckets.get(endpoint)

        if bucket is None:
            return True

        return bucket.try_acquire(tokens)

    def available(self, endpoint: str) -> float:
        """Return available tokens for an endpoint."""
        with self._lock:
            bucket = self._buckets.get(endpoint)
        return bucket.available if bucket else 0.0


# X API Basic tier rate limits
def create_x_rate_limiter() -> RateLimiter:
    """Create rate limiter configured for X API Basic tier."""
    limiter = RateLimiter()

    # Basic tier limits
    # User lookup: 100 requests per 15 minutes
    limiter.add_endpoint("user_lookup", requests_per_window=100, window_seconds=900)

    # User tweets: 100 requests per 15 minutes
    limiter.add_endpoint("user_tweets", requests_per_window=100, window_seconds=900)

    # Monthly tweet read cap: ~10,000/month = ~0.00386/second
    # We'll track this separately with a larger window
    limiter.add_endpoint("monthly_tweets", requests_per_window=10000, window_seconds=2592000)

    return limiter


# Alpaca API rate limits
def create_alpaca_rate_limiter() -> RateLimiter:
    """Create rate limiter configured for Alpaca API."""
    limiter = RateLimiter()

    # Alpaca allows 200 requests per minute for most endpoints
    limiter.add_endpoint("trading", requests_per_window=200, window_seconds=60)

    # Market data has higher limits
    limiter.add_endpoint("market_data", requests_per_window=200, window_seconds=60)

    return limiter
