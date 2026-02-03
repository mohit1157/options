from __future__ import annotations

from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def tick(self) -> None:
        raise NotImplementedError
