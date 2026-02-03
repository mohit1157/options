"""Tests for LocalRuleSentiment."""
import pytest
from datetime import datetime, timezone

from tradebot.core.events import BaseEvent
from tradebot.sentiment.local_rule import LocalRuleSentiment


class TestLocalRuleSentiment:
    """Tests for LocalRuleSentiment class."""

    @pytest.fixture
    def sentiment(self) -> LocalRuleSentiment:
        return LocalRuleSentiment()

    def _make_event(self, text: str) -> BaseEvent:
        return BaseEvent(
            type="news",
            source="test",
            created_at=datetime.now(timezone.utc),
            text=text,
        )

    def test_positive_sentiment(self, sentiment: LocalRuleSentiment):
        """Test that positive keywords trigger positive sentiment."""
        event = self._make_event("Company beats earnings expectations with record growth")
        result = sentiment.analyze(event)

        assert result.score > 0
        assert result.label == "positive"

    def test_negative_sentiment(self, sentiment: LocalRuleSentiment):
        """Test that negative keywords trigger negative sentiment."""
        event = self._make_event("Market crash as fraud allegations surface")
        result = sentiment.analyze(event)

        assert result.score < 0
        assert result.label == "negative"

    def test_neutral_sentiment(self, sentiment: LocalRuleSentiment):
        """Test that neutral text returns neutral sentiment."""
        event = self._make_event("The company announced quarterly results today.")
        result = sentiment.analyze(event)

        assert result.label == "neutral"

    def test_mixed_sentiment(self, sentiment: LocalRuleSentiment):
        """Test text with both positive and negative keywords."""
        event = self._make_event("Company beats estimates but faces lawsuit")
        result = sentiment.analyze(event)

        # Score should be close to neutral with mixed signals
        assert -0.5 < result.score < 0.5

    def test_empty_text(self, sentiment: LocalRuleSentiment):
        """Test handling of empty text."""
        event = self._make_event("")
        result = sentiment.analyze(event)

        assert result.score == 0.0
        assert result.label == "neutral"

    def test_case_insensitivity(self, sentiment: LocalRuleSentiment):
        """Test that keyword matching is case insensitive."""
        event = self._make_event("COMPANY BEATS EXPECTATIONS, BULLISH OUTLOOK")
        result = sentiment.analyze(event)

        assert result.score > 0
        assert result.label == "positive"

    def test_multiple_positive_keywords(self, sentiment: LocalRuleSentiment):
        """Test that multiple positive keywords increase score."""
        event_single = self._make_event("Bullish outlook for the market")
        event_multiple = self._make_event("Bullish rally with record growth surge")

        result_single = sentiment.analyze(event_single)
        result_multiple = sentiment.analyze(event_multiple)

        # More positive keywords should result in higher score
        assert result_multiple.score >= result_single.score

    def test_multiple_negative_keywords(self, sentiment: LocalRuleSentiment):
        """Test that multiple negative keywords decrease score."""
        event_single = self._make_event("Company faces lawsuit")
        event_multiple = self._make_event("Market crash as fraud leads to losses")

        result_single = sentiment.analyze(event_single)
        result_multiple = sentiment.analyze(event_multiple)

        # More negative keywords should result in lower (more negative) score
        assert result_multiple.score <= result_single.score

    def test_score_range(self, sentiment: LocalRuleSentiment):
        """Test that scores are within expected range."""
        texts = [
            "Massive bull rally with record growth",
            "Complete crash with fraud and losses",
            "Regular day in the market",
            "",
        ]

        for text in texts:
            event = self._make_event(text)
            result = sentiment.analyze(event)

            # Score should be between -1 and 1
            assert -1.0 <= result.score <= 1.0
            # Label should be valid
            assert result.label in ("positive", "negative", "neutral")

    def test_label_thresholds(self, sentiment: LocalRuleSentiment):
        """Test that labels match expected thresholds."""
        # Strongly positive
        event_pos = self._make_event("Bulls rally surge growth beats record")
        result_pos = sentiment.analyze(event_pos)
        if result_pos.score > 0.15:
            assert result_pos.label == "positive"

        # Strongly negative
        event_neg = self._make_event("Crash fraud lawsuit miss loss bearish")
        result_neg = sentiment.analyze(event_neg)
        if result_neg.score < -0.15:
            assert result_neg.label == "negative"
