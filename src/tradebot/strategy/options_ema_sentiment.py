from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta, datetime, timezone
from typing import Optional

from tradebot.config import Settings
from tradebot.core.logger import get_logger
from tradebot.core.timeutils import utcnow
from tradebot.data.marketdata import fetch_bars, ema
from tradebot.broker.alpaca_broker import AlpacaBroker, InvalidOrderError, OrderRejectedError
from tradebot.broker.models import OrderRequest
from tradebot.risk.risk_manager import RiskManager
from tradebot.data.store import SQLiteStore
from tradebot.options.contracts import AlpacaOptionsContractsClient, pick_atm_contract

log = get_logger("strategy.options")


@dataclass
class OptionsEmaSentimentStrategy:
    """Options trading strategy combining EMA crossover with sentiment gating.

    This strategy:
    1. Monitors EMA crossovers (fast crossing slow)
    2. Requires sentiment confirmation before trading
    3. Trades options (calls on bullish, puts on bearish signals)

    Configuration:
        enable_options: Must be True to run
        option_dte_max: Maximum days to expiration
        option_strike_tolerance: Max % distance from ATM
        option_order_qty: Contracts per trade
        option_side_on_bull: "call" for bullish signals
        option_side_on_bear: "put" for bearish signals
    """

    settings: Settings
    broker: AlpacaBroker
    risk: RiskManager
    store: SQLiteStore

    # Client is lazily initialized and reused
    _contracts_client: Optional[AlpacaOptionsContractsClient] = field(
        default=None, init=False, repr=False
    )

    def _get_contracts_client(self) -> AlpacaOptionsContractsClient:
        """Get or create the contracts client (lazy initialization)."""
        if self._contracts_client is None:
            self._contracts_client = AlpacaOptionsContractsClient(
                base_url=self.settings.alpaca_base_url,
                api_key=self.settings.alpaca_api_key,
                api_secret=self.settings.alpaca_api_secret,
            )
            log.debug("Options contracts client initialized")
        return self._contracts_client

    def close(self) -> None:
        """Close the contracts client if open."""
        if self._contracts_client is not None:
            self._contracts_client.close()
            self._contracts_client = None
            log.debug("Options contracts client closed")

    def _sentiment_gate(self, symbol: str) -> float:
        """Get average sentiment from recent events for a symbol."""
        since = (utcnow() - timedelta(minutes=30)).isoformat()
        rows = self.store.recent_sentiment(since_iso=since, limit=50, symbol=symbol)
        if not rows:
            return 0.0
        scores = [r[0] for r in rows]
        return float(sum(scores) / max(len(scores), 1))

    def tick(self) -> None:
        """Execute one tick of the options strategy.

        Called periodically from the main loop when options trading is enabled.
        """
        # Check if options trading is enabled
        if not self.settings.enable_options:
            return
        if self.settings.market_open_only and not self.broker.is_market_open():
            log.info("[OPTIONS] Market closed, skipping options strategy tick")
            return

        try:
            acct = self.broker.account()
            equity = float(acct.equity)
        except Exception as e:
            log.error(f"Failed to get account info: {e}")
            return
        if self.risk.daily_loss_exceeded(current_equity_usd=equity):
            log.warning("[OPTIONS] Max daily loss exceeded, skipping options tick")
            return

        contracts_client = self._get_contracts_client()

        for underlying in self.settings.symbols_list:
            try:
                sentiment = self._sentiment_gate(underlying)
                log.info(f"[OPTIONS] {underlying} sentiment avg: {sentiment:.3f}")
                self._process_symbol(
                    underlying=underlying,
                    equity=equity,
                    sentiment=sentiment,
                    contracts_client=contracts_client,
                )
            except Exception as e:
                log.error(f"[OPTIONS] Error processing {underlying}: {e}")

    def _process_symbol(
        self,
        underlying: str,
        equity: float,
        sentiment: float,
        contracts_client: AlpacaOptionsContractsClient,
    ) -> None:
        """Process a single underlying symbol for options signals."""
        # Cooldown check (by underlying)
        if self.settings.trade_cooldown_minutes > 0:
            last_trade = self.store.get_last_trade_time_for_underlying(underlying)
            if last_trade:
                now = datetime.now(timezone.utc)
                delta = now - last_trade.astimezone(timezone.utc)
                if delta < timedelta(minutes=self.settings.trade_cooldown_minutes):
                    log.info(f"[OPTIONS] {underlying}: cooldown active, skipping")
                    return

        # Fetch price data
        bars = fetch_bars(
            self.settings.alpaca_api_key,
            self.settings.alpaca_api_secret,
            symbol=underlying,
            timeframe=self.settings.timeframe,
            limit=250,
            feed=self.settings.alpaca_data_feed,
        )
        if bars.empty:
            log.debug(f"[OPTIONS] No bars for {underlying}")
            return

        close = bars["close"]
        fast = ema(close, self.settings.ema_fast)
        slow = ema(close, self.settings.ema_slow)

        if len(fast) < 3 or len(slow) < 3:
            log.debug(f"[OPTIONS] Insufficient data for {underlying}")
            return

        prev_diff = float(fast.iloc[-2] - slow.iloc[-2])
        curr_diff = float(fast.iloc[-1] - slow.iloc[-1])
        underlying_price = float(close.iloc[-1])

        # Check for crossover signals with sentiment confirmation
        bull = (
            prev_diff <= 0
            and curr_diff > 0
            and sentiment >= self.settings.sentiment_threshold
        )
        bear = (
            prev_diff >= 0
            and curr_diff < 0
            and sentiment <= -self.settings.sentiment_threshold
        )

        if not (bull or bear):
            return

        # Determine contract type based on signal and config
        if bull:
            preferred = self.settings.option_side_on_bull
            signal_type = "BULL"
        else:
            preferred = self.settings.option_side_on_bear
            signal_type = "BEAR"

        log.info(
            f"[OPTIONS] {signal_type} signal for {underlying} "
            f"(sentiment={sentiment:.3f}, price={underlying_price:.2f})"
        )

        # Find a suitable contract
        contract = self._find_contract(
            underlying=underlying,
            underlying_price=underlying_price,
            preferred_type=preferred,
            contracts_client=contracts_client,
        )

        if not contract:
            log.warning(
                f"[OPTIONS] No suitable {preferred} found for {underlying} "
                f"near {underlying_price:.2f}"
            )
            return

        # Liquidity check (volume)
        if contract.volume < self.settings.option_min_volume:
            log.warning(
                f"[OPTIONS] {contract.symbol} volume {contract.volume} below min "
                f"{self.settings.option_min_volume}"
            )
            return

        # Spread check and premium estimate
        quote = contracts_client.get_latest_option_quote(contract.symbol)
        if quote:
            bid = quote.get("bid")
            ask = quote.get("ask")
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                if spread_pct > self.settings.option_max_spread_pct:
                    log.warning(
                        f"[OPTIONS] {contract.symbol} spread {spread_pct:.2%} "
                        f"exceeds {self.settings.option_max_spread_pct:.2%}"
                    )
                    return
        premium = contracts_client.get_latest_option_mid_price(contract.symbol)

        # Place the order
        if self.settings.option_use_dynamic_qty and premium:
            qty = self.risk.calc_option_qty(equity_usd=equity, premium=premium)
            if qty < 1:
                log.info(f"[OPTIONS] {contract.symbol} dynamic qty=0, skipping")
                return
        else:
            qty = self.settings.option_order_qty
            if premium:
                pos_value = qty * premium * 100
                if not self.risk.position_value_ok(pos_value):
                    log.warning(
                        f"[OPTIONS] {contract.symbol} position value {pos_value:.2f} "
                        f"exceeds max {self.risk.max_position_value_usd:.2f}"
                    )
                    return

        log.info(
            f"[OPTIONS] BUY {contract.symbol} "
            f"(underlying={underlying}, price={underlying_price:.2f}, "
            f"strike={contract.strike_price}, exp={contract.expiration_date}) "
            f"qty={qty}"
        )

        try:
            # Skip if position or open order exists for this contract
            try:
                pos = self.broker.get_position(contract.symbol)
                if pos and abs(pos.qty) > 0:
                    log.info(f"[OPTIONS] {contract.symbol}: existing position, skipping")
                    return
            except Exception as e:
                log.warning(f"[OPTIONS] {contract.symbol}: position check failed: {e}")

            open_orders = self.broker.list_open_orders(symbol=contract.symbol)
            if open_orders:
                log.info(f"[OPTIONS] {contract.symbol}: open orders exist, skipping")
                return

            req = OrderRequest(symbol=contract.symbol, side="buy", qty=qty)

            # Optional bracket for options if premium known
            if self.settings.option_use_bracket and premium:
                sl = premium * (1.0 - self.settings.option_stop_loss_pct)
                tp = premium * (1.0 + self.settings.option_take_profit_pct)
                req.stop_loss_price = round(sl, 4)
                req.take_profit_price = round(tp, 4)

            order_id = self.broker.place_order(req)

            # Log to audit
            self.store.log_trade(
                symbol=contract.symbol,
                side="buy",
                qty=qty,
                order_type="market",
                status="submitted",
                order_id=order_id,
                metadata={
                    "underlying": underlying,
                    "underlying_price": underlying_price,
                    "strike": contract.strike_price,
                    "expiration": contract.expiration_date,
                    "contract_type": contract.type,
                    "signal": signal_type,
                    "sentiment": sentiment,
                    "premium": premium,
                },
            )
            log.info(f"[OPTIONS] Order submitted: {order_id}")

        except (InvalidOrderError, OrderRejectedError) as e:
            if self.settings.option_use_bracket:
                log.warning(f"[OPTIONS] Bracket order failed, retrying market: {e}")
                try:
                    order_id = self.broker.place_order(
                        OrderRequest(symbol=contract.symbol, side="buy", qty=qty)
                    )
                    self.store.log_trade(
                        symbol=contract.symbol,
                        side="buy",
                        qty=qty,
                        order_type="market",
                        status="submitted",
                        order_id=order_id,
                        metadata={
                            "underlying": underlying,
                            "underlying_price": underlying_price,
                            "strike": contract.strike_price,
                            "expiration": contract.expiration_date,
                            "contract_type": contract.type,
                            "signal": signal_type,
                            "sentiment": sentiment,
                            "premium": premium,
                        },
                    )
                    log.info(f"[OPTIONS] Market order submitted: {order_id}")
                    return
                except Exception as ex:
                    log.error(f"[OPTIONS] Market fallback failed: {ex}")

        except Exception as e:
            log.error(f"[OPTIONS] Order failed for {contract.symbol}: {e}")
            self.store.log_trade(
                symbol=contract.symbol,
                side="buy",
                qty=qty,
                order_type="market",
                status="error",
                error_message=str(e),
            )

    def _find_contract(
        self,
        underlying: str,
        underlying_price: float,
        preferred_type: str,
        contracts_client: AlpacaOptionsContractsClient,
    ):
        """Find a suitable options contract."""
        today = datetime.now(timezone.utc).date()
        exp_gte = today.strftime("%Y-%m-%d")
        exp_lte = (today + timedelta(days=self.settings.option_dte_max)).strftime(
            "%Y-%m-%d"
        )

        try:
            contracts = contracts_client.list_contracts(
                underlying=underlying,
                contract_type=preferred_type,
                exp_gte=exp_gte,
                exp_lte=exp_lte,
                limit=200,
            )
        except Exception as e:
            log.error(f"[OPTIONS] Failed to fetch contracts for {underlying}: {e}")
            return None

        return pick_atm_contract(
            contracts,
            underlying_price=underlying_price,
            strike_tolerance=self.settings.option_strike_tolerance,
            dte_max=self.settings.option_dte_max,
            preferred_type=preferred_type,
            client=contracts_client,  # Pass client for volume-based selection
        )
