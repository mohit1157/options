from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from tradebot.config import Settings
from tradebot.core.logger import get_logger
from tradebot.core.timeutils import utcnow
from tradebot.data.marketdata import fetch_bars, ema
from tradebot.broker.alpaca_broker import AlpacaBroker
from tradebot.broker.models import OrderRequest
from tradebot.risk.risk_manager import RiskManager
from tradebot.data.store import SQLiteStore

log = get_logger("strategy")


@dataclass
class EmaSentimentStrategy:
    """EMA crossover strategy with sentiment gating.

    This strategy:
    1. Monitors EMA crossovers (fast crossing slow)
    2. Requires sentiment confirmation before trading
    3. Uses risk manager for position sizing and bracket orders

    Buy signal:
        - Fast EMA crosses above slow EMA
        - Recent sentiment >= sentiment_threshold

    Sell signal:
        - Fast EMA crosses below slow EMA
        - Recent sentiment <= -sentiment_threshold
    """

    settings: Settings
    broker: AlpacaBroker
    risk: RiskManager
    store: SQLiteStore

    def _sentiment_gate(self) -> float:
        """Aggregate recent sentiment (last 30 minutes) from stored events."""
        since = (utcnow() - timedelta(minutes=30)).isoformat()
        rows = self.store.recent_sentiment(since_iso=since, limit=50)
        if not rows:
            return 0.0
        scores = [r[0] for r in rows]
        return float(sum(scores) / max(len(scores), 1))

    def tick(self) -> None:
        """Execute one tick of the strategy.

        Called periodically from the main loop.
        """
        try:
            acct = self.broker.account()
            equity = float(acct.equity)
        except Exception as e:
            log.error(f"Failed to get account info: {e}")
            return

        sentiment = self._sentiment_gate()
        log.info(f"Recent sentiment avg: {sentiment:.3f}")

        for symbol in self.settings.symbols_list:
            try:
                self._process_symbol(
                    symbol=symbol,
                    equity=equity,
                    sentiment=sentiment,
                )
            except Exception as e:
                log.error(f"Error processing {symbol}: {e}")

    def _process_symbol(
        self,
        symbol: str,
        equity: float,
        sentiment: float,
    ) -> None:
        """Process a single symbol for trading signals."""
        bars = fetch_bars(
            self.settings.alpaca_api_key,
            self.settings.alpaca_api_secret,
            symbol=symbol,
            timeframe=self.settings.timeframe,
            limit=250,
            feed=self.settings.alpaca_data_feed,
        )
        if bars.empty:
            log.warning(f"No bars for {symbol}")
            return

        close = bars["close"]
        fast = ema(close, self.settings.ema_fast)
        slow = ema(close, self.settings.ema_slow)

        # Signal: fast crosses slow on last bar
        if len(fast) < 3 or len(slow) < 3:
            return

        prev_diff = float(fast.iloc[-2] - slow.iloc[-2])
        curr_diff = float(fast.iloc[-1] - slow.iloc[-1])
        price = float(close.iloc[-1])

        # BUY signal
        if (
            prev_diff <= 0
            and curr_diff > 0
            and sentiment >= self.settings.sentiment_threshold
        ):
            self._place_buy_order(symbol, equity, price, sentiment)

        # SELL/Short signal
        elif (
            prev_diff >= 0
            and curr_diff < 0
            and sentiment <= -self.settings.sentiment_threshold
        ):
            self._place_sell_order(symbol, equity, price, sentiment)

    def _place_buy_order(
        self,
        symbol: str,
        equity: float,
        price: float,
        sentiment: float,
    ) -> None:
        """Place a buy order with bracket."""
        qty = self.risk.calc_qty(equity_usd=equity, price=price)
        if qty <= 0:
            log.debug(f"Buy signal for {symbol} but qty=0")
            return

        sl, tp = self.risk.bracket_prices(entry_price=price, side="buy")
        log.info(
            f"Signal BUY {symbol} qty={qty:.4f} price={price:.2f} "
            f"SL={sl:.2f} TP={tp:.2f} sentiment={sentiment:.3f}"
        )

        try:
            order_id = self.broker.place_order(
                OrderRequest(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    take_profit_price=tp,
                    stop_loss_price=sl,
                )
            )

            # Log to audit
            self.store.log_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                order_type="bracket",
                status="submitted",
                order_id=order_id,
                metadata={
                    "price": price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "sentiment": sentiment,
                },
            )
            log.info(f"Buy order submitted: {order_id}")

        except Exception as e:
            log.error(f"Buy order failed for {symbol}: {e}")
            self.store.log_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                order_type="bracket",
                status="error",
                error_message=str(e),
            )

    def _place_sell_order(
        self,
        symbol: str,
        equity: float,
        price: float,
        sentiment: float,
    ) -> None:
        """Place a sell/short order with bracket."""
        qty = self.risk.calc_qty(equity_usd=equity, price=price)
        if qty <= 0:
            log.debug(f"Sell signal for {symbol} but qty=0")
            return

        sl, tp = self.risk.bracket_prices(entry_price=price, side="sell")
        log.info(
            f"Signal SELL {symbol} qty={qty:.4f} price={price:.2f} "
            f"SL={sl:.2f} TP={tp:.2f} sentiment={sentiment:.3f}"
        )

        try:
            order_id = self.broker.place_order(
                OrderRequest(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    take_profit_price=tp,
                    stop_loss_price=sl,
                )
            )

            # Log to audit
            self.store.log_trade(
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="bracket",
                status="submitted",
                order_id=order_id,
                metadata={
                    "price": price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "sentiment": sentiment,
                },
            )
            log.info(f"Sell order submitted: {order_id}")

        except Exception as e:
            log.error(f"Sell order failed for {symbol}: {e}")
            self.store.log_trade(
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="bracket",
                status="error",
                error_message=str(e),
            )
