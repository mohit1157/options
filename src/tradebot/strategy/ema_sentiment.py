from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, datetime, timezone

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
    """EMA crossover strategy with optional sentiment gating.

    This strategy:
    1. Monitors EMA crossovers (fast crossing slow)
    2. Optionally requires sentiment confirmation before trading
    3. Uses risk manager for position sizing and bracket orders

    Buy signal:
        - Fast EMA crosses above slow EMA
        - If enabled: Recent sentiment >= sentiment_threshold

    Sell signal:
        - Fast EMA crosses below slow EMA
        - If enabled: Recent sentiment <= -sentiment_threshold
    """

    settings: Settings
    broker: AlpacaBroker
    risk: RiskManager
    store: SQLiteStore

    def _sentiment_gate(self, symbol: str) -> float:
        """Aggregate recent sentiment (last 30 minutes) for a symbol."""
        since = (utcnow() - timedelta(minutes=30)).isoformat()
        rows = self.store.recent_sentiment(since_iso=since, limit=50, symbol=symbol)
        if not rows:
            return 0.0
        scores = [r[0] for r in rows]
        return float(sum(scores) / max(len(scores), 1))

    def tick(self) -> None:
        """Execute one tick of the strategy.

        Called periodically from the main loop.
        """
        if self.settings.market_open_only and not self.broker.is_market_open():
            log.info("Market closed, skipping strategy tick")
            return

        try:
            acct = self.broker.account()
            equity = float(acct.equity)
        except Exception as e:
            log.error(f"Failed to get account info: {e}")
            return

        if self.risk.daily_loss_exceeded(current_equity_usd=equity):
            log.warning("Max daily loss exceeded, skipping strategy tick")
            return

        for symbol in self.settings.symbols_list:
            try:
                sentiment = 0.0
                if self.settings.use_sentiment:
                    sentiment = self._sentiment_gate(symbol)
                    log.info(f"{symbol} sentiment avg: {sentiment:.3f}")
                self._process_symbol(
                    symbol=symbol,
                    equity=equity,
                    sentiment=sentiment,
                    use_sentiment=self.settings.use_sentiment,
                )
            except Exception as e:
                log.error(f"Error processing {symbol}: {e}")

    def _process_symbol(
        self,
        symbol: str,
        equity: float,
        sentiment: float,
        use_sentiment: bool,
    ) -> None:
        """Process a single symbol for trading signals."""
        # Cooldown check
        if self.settings.trade_cooldown_minutes > 0:
            last_trade = self.store.get_last_trade_time(symbol)
            if last_trade:
                now = datetime.now(timezone.utc)
                delta = now - last_trade.astimezone(timezone.utc)
                if delta < timedelta(minutes=self.settings.trade_cooldown_minutes):
                    log.info(f"{symbol}: cooldown active, skipping")
                    return

        # Skip if already in a position
        try:
            pos = self.broker.get_position(symbol)
            if pos and abs(pos.qty) > 0:
                log.info(f"{symbol}: existing position detected, skipping")
                return
        except Exception as e:
            log.warning(f"{symbol}: position check failed: {e}")

        # Skip if open orders exist
        open_orders = self.broker.list_open_orders(symbol=symbol)
        if open_orders:
            log.info(f"{symbol}: open orders exist, skipping")
            return

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

        bull_signal = prev_diff <= 0 and curr_diff > 0
        bear_signal = prev_diff >= 0 and curr_diff < 0

        if use_sentiment:
            bull_signal = bull_signal and sentiment >= self.settings.sentiment_threshold
            bear_signal = bear_signal and sentiment <= -self.settings.sentiment_threshold

        # BUY signal
        if bull_signal:
            self._place_buy_order(symbol, equity, price, sentiment)

        # SELL/Short signal
        elif bear_signal:
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
        if not self.risk.position_value_ok(qty * price):
            log.warning(f"Buy signal for {symbol} exceeds max position value")
            return

        sl, tp = self.risk.bracket_prices(entry_price=price, side="buy")
        if self.settings.use_sentiment:
            log.info(
                f"Signal BUY {symbol} qty={qty:.4f} price={price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f} sentiment={sentiment:.3f}"
            )
        else:
            log.info(
                f"Signal BUY {symbol} qty={qty:.4f} price={price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f}"
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
            metadata = {
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
            }
            if self.settings.use_sentiment:
                metadata["sentiment"] = sentiment

            self.store.log_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                order_type="bracket",
                status="submitted",
                order_id=order_id,
                metadata=metadata,
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
        if not self.risk.position_value_ok(qty * price):
            log.warning(f"Sell signal for {symbol} exceeds max position value")
            return

        sl, tp = self.risk.bracket_prices(entry_price=price, side="sell")
        if self.settings.use_sentiment:
            log.info(
                f"Signal SELL {symbol} qty={qty:.4f} price={price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f} sentiment={sentiment:.3f}"
            )
        else:
            log.info(
                f"Signal SELL {symbol} qty={qty:.4f} price={price:.2f} "
                f"SL={sl:.2f} TP={tp:.2f}"
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
            metadata = {
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
            }
            if self.settings.use_sentiment:
                metadata["sentiment"] = sentiment

            self.store.log_trade(
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="bracket",
                status="submitted",
                order_id=order_id,
                metadata=metadata,
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
