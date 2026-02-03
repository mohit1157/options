from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
from alpaca.trading.requests import OrderRequest as AlpacaOrderRequest
from alpaca.common.exceptions import APIError

from tradebot.broker.models import OrderRequest, Position
from tradebot.core.logger import get_logger
from tradebot.core.retry import with_retry

log = get_logger("broker")


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class OrderRejectedError(BrokerError):
    """Raised when an order is rejected by the exchange."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when there are insufficient funds for an order."""
    pass


class InvalidOrderError(BrokerError):
    """Raised when order parameters are invalid."""
    pass


@dataclass
class AlpacaBroker:
    """Alpaca trading broker with error handling and retries.

    Features:
    - Automatic retries on transient failures
    - Proper error classification
    - Position management
    - Bracket order support

    Usage:
        broker = AlpacaBroker(api_key="...", api_secret="...", paper=True)
        account = broker.account()
        broker.place_order(OrderRequest(symbol="AAPL", side="buy", qty=10))
    """

    api_key: str
    api_secret: str
    paper: bool = True

    def __post_init__(self) -> None:
        self.client = TradingClient(
            self.api_key,
            self.api_secret,
            paper=self.paper,
        )
        log.debug(f"Alpaca broker initialized (paper={self.paper})")

    @with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
    def account(self):
        """Get account information.

        Returns:
            Alpaca Account object with equity, buying_power, etc.

        Raises:
            BrokerError: If account info cannot be retrieved
        """
        try:
            return self.client.get_account()
        except APIError as e:
            log.error(f"Failed to get account: {e}")
            raise BrokerError(f"Account error: {e}") from e

    @with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
    def list_positions(self) -> List[Position]:
        """Get all open positions.

        Returns:
            List of Position objects

        Raises:
            BrokerError: If positions cannot be retrieved
        """
        try:
            positions = self.client.get_all_positions()
            out: List[Position] = []
            for p in positions:
                out.append(
                    Position(
                        symbol=p.symbol,
                        qty=float(p.qty),
                        avg_entry_price=float(p.avg_entry_price),
                        market_value=float(p.market_value),
                        unrealized_pl=float(p.unrealized_pl),
                    )
                )
            return out
        except APIError as e:
            log.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Positions error: {e}") from e

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Returns:
            Position object or None if no position
        """
        try:
            p = self.client.get_open_position(symbol)
            return Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            )
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None
            raise BrokerError(f"Position error: {e}") from e

    @with_retry(max_attempts=2, min_wait=0.5, max_wait=5.0)
    def close_position(self, symbol: str) -> Optional[str]:
        """Close a position.

        Returns:
            Order ID if position was closed, None if no position

        Raises:
            BrokerError: If close fails
        """
        try:
            result = self.client.close_position(symbol)
            log.info(f"Closed position for {symbol}")
            return str(result.id) if result else None
        except APIError as e:
            if "position does not exist" in str(e).lower():
                log.debug(f"No position to close for {symbol}")
                return None
            log.error(f"Failed to close position {symbol}: {e}")
            raise BrokerError(f"Close position error: {e}") from e

    def place_order(self, req: OrderRequest) -> str:
        """Place an order.

        Args:
            req: OrderRequest with symbol, side, qty, and optional SL/TP

        Returns:
            Order ID string

        Raises:
            OrderRejectedError: If order is rejected
            InsufficientFundsError: If insufficient buying power
            InvalidOrderError: If order parameters are invalid
            BrokerError: For other errors
        """
        # Validate request
        if req.qty <= 0:
            raise InvalidOrderError(f"Invalid quantity: {req.qty}")

        if req.side.lower() not in ("buy", "sell"):
            raise InvalidOrderError(f"Invalid side: {req.side}")

        side = OrderSide.BUY if req.side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if req.time_in_force.lower() == "day" else TimeInForce.GTC

        # Build bracket components if provided
        take_profit = None
        stop_loss = None
        order_class = None

        if req.take_profit_price or req.stop_loss_price:
            order_class = "bracket"
            if req.take_profit_price:
                take_profit = TakeProfitRequest(limit_price=round(req.take_profit_price, 2))
            if req.stop_loss_price:
                stop_loss = StopLossRequest(stop_price=round(req.stop_loss_price, 2))

        order = AlpacaOrderRequest(
            symbol=req.symbol,
            qty=req.qty,
            side=side,
            type="market",
            time_in_force=tif,
            order_class=order_class,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )

        try:
            res = self.client.submit_order(order_data=order)
            log.info(
                f"Order submitted: {res.id} {req.side.upper()} {req.qty} {req.symbol}"
            )
            return str(res.id)

        except APIError as e:
            error_msg = str(e).lower()

            # Classify the error
            if "insufficient" in error_msg or "buying power" in error_msg:
                log.error(f"Insufficient funds for {req.symbol}: {e}")
                raise InsufficientFundsError(f"Insufficient funds: {e}") from e

            if "rejected" in error_msg:
                log.error(f"Order rejected for {req.symbol}: {e}")
                raise OrderRejectedError(f"Order rejected: {e}") from e

            if "invalid" in error_msg or "not found" in error_msg:
                log.error(f"Invalid order for {req.symbol}: {e}")
                raise InvalidOrderError(f"Invalid order: {e}") from e

            log.error(f"Order failed for {req.symbol}: {e}")
            raise BrokerError(f"Order error: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Returns:
            True if cancelled, False if order not found
        """
        try:
            self.client.cancel_order_by_id(order_id)
            log.info(f"Cancelled order: {order_id}")
            return True
        except APIError as e:
            if "not found" in str(e).lower():
                log.debug(f"Order {order_id} not found to cancel")
                return False
            raise BrokerError(f"Cancel error: {e}") from e

    def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        try:
            result = self.client.cancel_orders()
            count = len(result) if result else 0
            log.info(f"Cancelled {count} orders")
            return count
        except APIError as e:
            raise BrokerError(f"Cancel all error: {e}") from e

    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order status.

        Returns:
            Order dict with status, filled_qty, etc. or None if not found
        """
        try:
            order = self.client.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "status": str(order.status),
                "type": str(order.type),
                "created_at": str(order.created_at),
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise BrokerError(f"Get order error: {e}") from e

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.client.get_clock()
            return clock.is_open
        except APIError as e:
            log.warning(f"Failed to check market status: {e}")
            return False
