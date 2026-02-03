from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str  # buy/sell
    qty: float
    type: str = "market"
    time_in_force: str = "day"
    # Bracket
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
