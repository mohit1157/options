from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RiskManager:
    max_daily_loss_usd: float
    max_position_value_usd: float
    risk_per_trade_pct: float
    stop_loss_pct: float
    take_profit_pct: float

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
