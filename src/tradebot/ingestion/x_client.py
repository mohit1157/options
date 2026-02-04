from __future__ import annotations

import time
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict

import httpx

from tradebot.core.events import BaseEvent
from tradebot.core.logger import get_logger
from tradebot.core.rate_limiter import create_x_rate_limiter, RateLimiter
from tradebot.core.retry import with_retry, RETRYABLE_EXCEPTIONS

log = get_logger("x_client")

# X API v2 base URL
X_API_BASE = "https://api.twitter.com/2"


class XClientError(Exception):
    """Base exception for X client errors."""
    pass


class XRateLimitError(XClientError):
    """Raised when rate limit is exceeded."""
    pass


class XClient:
    """X API v2 client for fetching tweets.

    Implements the X API v2 Basic tier with proper rate limiting
    and caching for user ID lookups.

    Basic tier limits:
    - 10,000 tweets/month read
    - 100 requests/15-min for user lookup
    - 100 requests/15-min for user tweets

    Usage:
        client = XClient(bearer_token="your_token")
        tweets = client.fetch_recent_by_username("deltaone", max_results=5)
        for tweet in tweets:
            print(tweet.text)
        client.close()
    """

    def __init__(
        self,
        bearer_token: str,
        timeout: float = 15.0,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        if not bearer_token:
            raise ValueError("X bearer token is required")

        self.bearer_token = bearer_token
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "TradingBot/1.0",
            },
        )

        # Rate limiter
        self.rate_limiter = rate_limiter or create_x_rate_limiter()

        # Cache for username -> user_id lookups (to minimize API calls)
        self._user_id_cache: Dict[str, str] = {}
        self._cache_ttl = 3600  # Cache user IDs for 1 hour
        self._cache_timestamps: Dict[str, float] = {}

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
        log.debug("X client closed")

    def _is_cache_valid(self, username: str) -> bool:
        """Check if cached user ID is still valid."""
        if username not in self._cache_timestamps:
            return False
        return time.monotonic() - self._cache_timestamps[username] < self._cache_ttl

    @with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
    def resolve_username_to_id(self, username: str) -> Optional[str]:
        """Resolve a username to a user ID.

        GET /2/users/by/username/:username

        Args:
            username: Twitter username (without @)

        Returns:
            User ID string or None if not found
        """
        username = username.lstrip("@").lower()

        # Check cache first
        if self._is_cache_valid(username) and username in self._user_id_cache:
            log.debug(f"Cache hit for @{username}")
            return self._user_id_cache[username]

        # Rate limit check
        if not self.rate_limiter.acquire("user_lookup", timeout=30.0):
            raise XRateLimitError("Rate limit exceeded for user lookup")

        url = f"{X_API_BASE}/users/by/username/{username}"

        try:
            response = self.client.get(url)

            if response.status_code == 404:
                log.warning(f"User @{username} not found")
                return None

            if response.status_code == 429:
                retry_after = response.headers.get("x-rate-limit-reset")
                raise XRateLimitError(
                    f"Rate limit exceeded. Reset at: {retry_after}"
                )

            response.raise_for_status()
            data = response.json()

            if "data" not in data:
                log.warning(f"No data in response for @{username}: {data}")
                return None

            user_id = data["data"]["id"]

            # Cache the result
            self._user_id_cache[username] = user_id
            self._cache_timestamps[username] = time.monotonic()

            log.debug(f"Resolved @{username} to ID {user_id}")
            return user_id

        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error resolving @{username}: {e.response.status_code}")
            raise

    @with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
    def fetch_user_tweets(
        self,
        user_id: str,
        max_results: int = 10,
        since_id: Optional[str] = None,
    ) -> list[dict]:
        """Fetch recent tweets for a user by their ID.

        GET /2/users/:id/tweets

        Args:
            user_id: Twitter user ID
            max_results: Maximum tweets to return (5-100)
            since_id: Only return tweets newer than this ID

        Returns:
            List of tweet dictionaries
        """
        # Rate limit checks
        if not self.rate_limiter.acquire("user_tweets", timeout=30.0):
            raise XRateLimitError("Rate limit exceeded for user tweets")

        if not self.rate_limiter.try_acquire("monthly_tweets", tokens=max_results):
            log.warning("Approaching monthly tweet read limit")

        url = f"{X_API_BASE}/users/{user_id}/tweets"

        params = {
            "max_results": min(max(5, max_results), 100),
            "tweet.fields": "created_at,author_id,text,public_metrics",
            "expansions": "author_id",
            "user.fields": "username,name",
        }

        if since_id:
            params["since_id"] = since_id

        try:
            response = self.client.get(url, params=params)

            if response.status_code == 429:
                retry_after = response.headers.get("x-rate-limit-reset")
                raise XRateLimitError(
                    f"Rate limit exceeded. Reset at: {retry_after}"
                )

            response.raise_for_status()
            data = response.json()

            if "data" not in data:
                log.debug(f"No tweets for user {user_id}")
                return []

            # Extract user info from includes
            users_map = {}
            if "includes" in data and "users" in data["includes"]:
                for user in data["includes"]["users"]:
                    users_map[user["id"]] = user

            # Enrich tweets with user info
            tweets = []
            for tweet in data["data"]:
                author_id = tweet.get("author_id", user_id)
                user_info = users_map.get(author_id, {})
                tweet["_username"] = user_info.get("username", "")
                tweet["_name"] = user_info.get("name", "")
                tweets.append(tweet)

            log.debug(f"Fetched {len(tweets)} tweets for user {user_id}")
            return tweets

        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error fetching tweets for {user_id}: {e.response.status_code}")
            raise

    def fetch_recent_by_username(
        self,
        username: str,
        max_results: int = 10,
        since_id: Optional[str] = None,
    ) -> list[BaseEvent]:
        """Fetch recent tweets for a username and return as BaseEvents.

        This is the main entry point for the ingestion loop.

        Args:
            username: Twitter username (with or without @)
            max_results: Maximum tweets to return

        Returns:
            List of BaseEvent objects
        """
        username = username.lstrip("@")

        # Resolve username to ID
        user_id = self.resolve_username_to_id(username)
        if not user_id:
            log.warning(f"Could not resolve @{username}")
            return []

        # Fetch tweets
        try:
            tweets = self.fetch_user_tweets(
                user_id,
                max_results=max_results,
                since_id=since_id,
            )
        except XRateLimitError:
            log.warning(f"Rate limited when fetching tweets for @{username}")
            return []
        except Exception as e:
            log.error(f"Error fetching tweets for @{username}: {e}")
            return []

        # Convert to BaseEvents
        events: list[BaseEvent] = []
        for tweet in tweets:
            tweet_id = tweet.get("id", "")
            tweet_text = tweet.get("text", "")

            # Parse created_at
            created_at_str = tweet.get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    created_at = datetime.now(timezone.utc)
            else:
                created_at = datetime.now(timezone.utc)

            # Create unique URL for tweet
            tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"

            event = BaseEvent(
                type="social",
                source=f"x/@{username}",
                created_at=created_at,
                text=tweet_text,
                url=tweet_url,
                author=tweet.get("_username") or username,
            )
            events.append(event)

        log.info(f"Fetched {len(events)} tweets from @{username}")
        return events

    def search_recent_tweets(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[BaseEvent]:
        """Search for recent tweets matching a query.

        Note: This requires Basic tier or higher.
        GET /2/tweets/search/recent

        Args:
            query: Search query (e.g., "$SPY" or "stock market")
            max_results: Maximum tweets to return (10-100)

        Returns:
            List of BaseEvent objects
        """
        # Rate limit check
        if not self.rate_limiter.acquire("user_tweets", timeout=30.0):
            raise XRateLimitError("Rate limit exceeded for tweet search")

        url = f"{X_API_BASE}/tweets/search/recent"

        params = {
            "query": query,
            "max_results": min(max(10, max_results), 100),
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username",
        }

        try:
            response = self.client.get(url, params=params)

            if response.status_code == 429:
                raise XRateLimitError("Rate limit exceeded for search")

            response.raise_for_status()
            data = response.json()

            if "data" not in data:
                return []

            # Extract user info
            users_map = {}
            if "includes" in data and "users" in data["includes"]:
                for user in data["includes"]["users"]:
                    users_map[user["id"]] = user.get("username", "")

            # Convert to BaseEvents
            events: list[BaseEvent] = []
            for tweet in data["data"]:
                tweet_id = tweet.get("id", "")
                author_id = tweet.get("author_id", "")
                username = users_map.get(author_id, "unknown")

                created_at_str = tweet.get("created_at", "")
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(
                            created_at_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        created_at = datetime.now(timezone.utc)
                else:
                    created_at = datetime.now(timezone.utc)

                tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"

                event = BaseEvent(
                    type="social",
                    source=f"x/search:{query[:20]}",
                    created_at=created_at,
                    text=tweet.get("text", ""),
                    url=tweet_url,
                    author=username,
                )
                events.append(event)

            log.info(f"Search '{query}' returned {len(events)} tweets")
            return events

        except httpx.HTTPStatusError as e:
            log.error(f"Search error: {e.response.status_code}")
            raise
