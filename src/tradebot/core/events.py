from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

EventType = Literal["news", "social"]

@dataclass(frozen=True)
class BaseEvent:
    type: EventType
    source: str
    created_at: datetime
    text: str
    url: Optional[str] = None
    author: Optional[str] = None

@dataclass(frozen=True)
class SentimentResult:
    score: float  # -1..+1
    label: str    # negative/neutral/positive

