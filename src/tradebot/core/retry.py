from __future__ import annotations

import random
import ssl
from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import httpx
from urllib3.exceptions import SSLError as Urllib3SSLError
from requests.exceptions import SSLError as RequestsSSLError

from tradebot.core.logger import get_logger

log = get_logger("retry")

T = TypeVar("T")

# Default retryable exceptions
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    ConnectionError,
    ConnectionResetError,
    TimeoutError,
    ssl.SSLError,
    Urllib3SSLError,
    RequestsSSLError,
    OSError,  # Catches various network-related OS errors
)


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: tuple = RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        min_wait: Minimum wait time between retries in seconds (default: 1.0)
        max_wait: Maximum wait time between retries in seconds (default: 10.0)
        exceptions: Tuple of exception types to retry on

    Example:
        @with_retry(max_attempts=3)
        def fetch_data():
            response = httpx.get(url)
            return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_jitter(initial=min_wait, max=max_wait, jitter=min_wait / 2),
                retry=retry_if_exception_type(exceptions),
                before_sleep=before_sleep_log(log, log_level=20),  # INFO level
                reraise=True,
            )
            def inner() -> T:
                return func(*args, **kwargs)

            return inner()

        return wrapper

    return decorator


def retry_on_rate_limit(
    max_attempts: int = 5,
    base_wait: float = 60.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator specifically for handling rate limit errors (HTTP 429).

    Uses longer backoff times appropriate for rate limit scenarios.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempts = 0
            last_error = None

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        attempts += 1
                        last_error = e

                        # Check for Retry-After header
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except ValueError:
                                wait_time = base_wait * (2 ** (attempts - 1))
                        else:
                            wait_time = base_wait * (2 ** (attempts - 1))

                        # Add jitter
                        wait_time += random.uniform(0, wait_time * 0.1)

                        log.warning(
                            f"Rate limited. Attempt {attempts}/{max_attempts}. "
                            f"Waiting {wait_time:.1f}s"
                        )

                        import time
                        time.sleep(wait_time)
                    else:
                        raise
                except Exception:
                    raise

            if last_error:
                raise last_error
            raise RuntimeError("Retry logic failed unexpectedly")

        return wrapper

    return decorator


class RetryableError(Exception):
    """Marker exception for errors that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """Marker exception for errors that should NOT trigger a retry."""
    pass
