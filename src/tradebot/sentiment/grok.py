from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from tradebot.core.events import BaseEvent, SentimentResult
from tradebot.core.logger import get_logger
from tradebot.core.retry import with_retry
from tradebot.sentiment.base import SentimentClient

log = get_logger("grok")

# System prompt for sentiment analysis
SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analyzer. Analyze the sentiment of market-related text and return a JSON response.

Return ONLY valid JSON in this exact format:
{"score": <float between -1 and 1>, "label": "<negative|neutral|positive>"}

Guidelines:
- score: -1.0 (very negative) to 1.0 (very positive), 0.0 for neutral
- label: "negative" if score < -0.15, "positive" if score > 0.15, otherwise "neutral"
- Consider financial impact, market sentiment, and trading implications
- Focus on how this news might affect stock prices

Examples:
- "Company beats earnings by 20%" -> {"score": 0.8, "label": "positive"}
- "CEO resigns amid scandal" -> {"score": -0.9, "label": "negative"}
- "Trading volume remains steady" -> {"score": 0.0, "label": "neutral"}"""


class GrokSentimentError(Exception):
    """Base exception for Grok sentiment errors."""
    pass


class GrokSentimentClient(SentimentClient):
    """Grok sentiment client using xAI API.

    Uses the OpenAI-compatible chat completions endpoint.

    Configuration:
        GROK_API_KEY: Your xAI API key
        GROK_BASE_URL: API base URL (default: https://api.x.ai/v1)
        GROK_MODEL: Model to use (default: grok-3-fast)

    Usage:
        client = GrokSentimentClient(api_key="your_key")
        result = client.analyze(event)
        print(result.score, result.label)
        client.close()
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-3-fast",
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError("GROK_API_KEY is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
        log.debug("Grok client closed")

    def _parse_response(self, content: str) -> SentimentResult:
        """Parse the model response into a SentimentResult.

        Handles various response formats and validates the output.
        """
        content = content.strip()

        # Try to extract JSON from the response
        # Handle cases where model adds extra text
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content)
        if json_match:
            content = json_match.group()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            log.warning(f"Failed to parse JSON response: {content[:100]}")
            # Fall back to neutral
            return SentimentResult(score=0.0, label="neutral")

        # Validate and extract score
        score = data.get("score")
        if score is None:
            log.warning(f"Missing score in response: {data}")
            score = 0.0
        else:
            try:
                score = float(score)
            except (ValueError, TypeError):
                log.warning(f"Invalid score type: {score}")
                score = 0.0

        # Clamp score to valid range
        score = max(-1.0, min(1.0, score))

        # Validate and extract label
        label = data.get("label", "").lower()
        valid_labels = {"negative", "neutral", "positive"}
        if label not in valid_labels:
            # Derive label from score
            if score > 0.15:
                label = "positive"
            elif score < -0.15:
                label = "negative"
            else:
                label = "neutral"

        return SentimentResult(score=score, label=label)

    @with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
    def analyze(self, event: BaseEvent) -> SentimentResult:
        """Analyze sentiment of an event using Grok.

        Args:
            event: BaseEvent to analyze

        Returns:
            SentimentResult with score and label

        Raises:
            GrokSentimentError: If API call fails
        """
        # Prepare the prompt
        user_message = self._format_event(event)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 100,  # Short response expected
        }

        url = f"{self.base_url}/chat/completions"

        try:
            response = self.client.post(url, json=payload)

            if response.status_code == 429:
                raise GrokSentimentError("Rate limit exceeded")

            if response.status_code == 401:
                raise GrokSentimentError("Invalid API key")

            response.raise_for_status()
            data = response.json()

            # Extract content from response
            choices = data.get("choices", [])
            if not choices:
                log.warning("No choices in Grok response")
                return SentimentResult(score=0.0, label="neutral")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                log.warning("Empty content in Grok response")
                return SentimentResult(score=0.0, label="neutral")

            result = self._parse_response(content)
            log.debug(f"Sentiment: {result.score:.2f} ({result.label}) for: {event.text[:50]}...")
            return result

        except httpx.HTTPStatusError as e:
            log.error(f"Grok API error: {e.response.status_code}")
            raise GrokSentimentError(f"API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            log.error(f"Grok request error: {e}")
            raise GrokSentimentError(f"Request error: {e}") from e

    def _format_event(self, event: BaseEvent) -> str:
        """Format an event for sentiment analysis."""
        parts = []

        if event.source:
            parts.append(f"Source: {event.source}")

        if event.author:
            parts.append(f"Author: {event.author}")

        parts.append(f"Text: {event.text[:2000]}")  # Limit text length

        return "\n".join(parts)

    def analyze_batch(
        self,
        events: list[BaseEvent],
        fallback_on_error: bool = True,
    ) -> list[SentimentResult]:
        """Analyze multiple events.

        Args:
            events: List of events to analyze
            fallback_on_error: If True, return neutral on error instead of raising

        Returns:
            List of SentimentResults (same order as input)
        """
        results = []
        for event in events:
            try:
                result = self.analyze(event)
                results.append(result)
            except Exception as e:
                log.warning(f"Batch analysis error: {e}")
                if fallback_on_error:
                    results.append(SentimentResult(score=0.0, label="neutral"))
                else:
                    raise
        return results

    def test_connection(self) -> bool:
        """Test the API connection.

        Returns:
            True if connection is successful
        """
        try:
            test_event = BaseEvent(
                type="news",
                source="test",
                created_at=None,  # Will use current time
                text="Test connection to Grok API",
            )
            self.analyze(test_event)
            return True
        except Exception as e:
            log.error(f"Grok connection test failed: {e}")
            return False
