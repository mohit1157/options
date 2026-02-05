from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from math import floor
from typing import Optional

@dataclass
class RiskManager:
    max_daily_loss_usd: float
    max_position_value_usd: float
    risk_per_trade_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    _daily_start_equity: Optional[float] = field(default=None, init=False, repr=False)
    _daily_start_date: Optional[date] = field(default=None, init=False, repr=False)

    def position_value_ok(self, proposed_value_usd: float) -> bool:
        return proposed_value_usd <= self.max_position_value_usd

    def calc_qty(self, equity_usd: float, price: float) -> float:
        # Risk-based sizing: risk_per_trade_pct of equity / stop distance
        if price <= 0:
            return 0.0
        risk_budget = equity_usd * self.risk_per_trade_pct
        stop_distance = price * self.stop_loss_pct
        if stop_distance <= 0:
            return 0.0
        qty = risk_budget / stop_distance
        # Also cap by max position value
        max_qty_by_value = self.max_position_value_usd / price
        return float(max(0.0, min(qty, max_qty_by_value)))

    def bracket_prices(self, entry_price: float, side: str) -> tuple[float, float]:
        # For buy: SL below, TP above. For sell (short): inverse.
        if side.lower() == "buy":
            sl = entry_price * (1.0 - self.stop_loss_pct)
            tp = entry_price * (1.0 + self.take_profit_pct)
        else:
            sl = entry_price * (1.0 + self.stop_loss_pct)
            tp = entry_price * (1.0 - self.take_profit_pct)
        return (round(sl, 4), round(tp, 4))

    def calc_option_qty(
        self,
        equity_usd: float,
        premium: float,
        *,
        portfolio_pct: Optional[float] = None,
        contract_multiplier: int = 100,
    ) -> int:
        """Calculate options contract quantity based on risk and premium.

        If portfolio_pct is provided, size by that portion of equity.
        Otherwise uses risk_per_trade_pct and max_position_value_usd to cap exposure.
        """
        if premium <= 0 or contract_multiplier <= 0:
            return 0

        if portfolio_pct is not None:
            if portfolio_pct <= 0 or portfolio_pct > 1:
                return 0
            risk_budget = equity_usd * portfolio_pct
        else:
            risk_budget = equity_usd * self.risk_per_trade_pct
        cost_per_contract = premium * contract_multiplier

        if cost_per_contract <= 0:
            return 0

        max_qty_by_risk = risk_budget / cost_per_contract
        if portfolio_pct is not None:
            qty = max_qty_by_risk
        else:
            max_qty_by_value = self.max_position_value_usd / cost_per_contract
            qty = min(max_qty_by_risk, max_qty_by_value)
        return max(0, int(floor(qty)))

    def daily_loss_exceeded(
        self,
        current_equity_usd: float,
        now: Optional[datetime] = None,
    ) -> bool:
        """Check if max daily loss has been exceeded.

        Tracks the starting equity for each UTC day and compares against current equity.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        today = now.date()
        if self._daily_start_date != today or self._daily_start_equity is None:
            self._daily_start_date = today
            self._daily_start_equity = current_equity_usd
            return False

        loss = self._daily_start_equity - current_equity_usd
        return loss >= self.max_daily_loss_usd
