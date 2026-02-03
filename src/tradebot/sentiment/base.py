from __future__ import annotations

from abc import ABC, abstractmethod
from tradebot.core.events import BaseEvent, SentimentResult

class SentimentClient(ABC):
    @abstractmethod
    def analyze(self, event: BaseEvent) -> SentimentResult:
        raise NotImplementedError
