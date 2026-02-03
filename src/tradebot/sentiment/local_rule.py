from __future__ import annotations

from tradebot.core.events import BaseEvent, SentimentResult
from tradebot.sentiment.base import SentimentClient

POS = {"beats", "beat", "growth", "surge", "record", "bull", "bullish", "rally", "upgrade", "win"}
NEG = {"miss", "missed", "fraud", "lawsuit", "ban", "bear", "bearish", "crash", "downgrade", "loss"}

class LocalRuleSentiment(SentimentClient):
    """A tiny fallback sentiment scorer. Replace with LLM for real usage."""

    def analyze(self, event: BaseEvent) -> SentimentResult:
        t = (event.text or "").lower()
        pos = sum(1 for w in POS if w in t)
        neg = sum(1 for w in NEG if w in t)
        score = 0.0
        if pos or neg:
            score = (pos - neg) / max(pos + neg, 1)
        label = "neutral"
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        return SentimentResult(score=score, label=label)
