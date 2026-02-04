from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List

from tradebot.config import Settings
from tradebot.core.logger import get_logger
from tradebot.data.marketdata import fetch_bars, ema
from tradebot.broker.alpaca_broker import AlpacaBroker
from tradebot.broker.models import OrderRequest
from tradebot.risk.risk_manager import RiskManager
from tradebot.data.store import SQLiteStore
from tradebot.options.contracts import AlpacaOptionsContractsClient, pick_atm_contract

log = get_logger("strategy.pop_pullback")


@dataclass
class SetupState:
    phase: str = "idle"  # idle | pullback | hold | entry_wait
    direction: Optional[str] = None  # "call" or "put"
    pop_index: Optional[int] = None
    pullback_index: Optional[int] = None
    pullback_close: Optional[float] = None
    hold_count: int = 0
    hold_closes: List[float] = field(default_factory=list)
    confirmation_index: Optional[int] = None
    confirmation_high: Optional[float] = None
    confirmation_low: Optional[float] = None
    base_low: Optional[float] = None
    base_high: Optional[float] = None
    entry_wait_bars: int = 0


@dataclass
class TradeState:
    underlying: str
    option_symbol: str
    direction: str  # "call" or "put"
    entry_underlying: float
    entry_option: Optional[float]
    stop_price: float
    original_qty: int
    remaining_qty: int
    trimmed: bool = False


@dataclass
class EmaPopPullbackHoldOptionsStrategy:
    """3-Min 9 EMA Pop Pullback Hold Confirmation Strategy for options."""

    settings: Settings
    broker: AlpacaBroker
    risk: RiskManager
    store: SQLiteStore

    _contracts_client: Optional[AlpacaOptionsContractsClient] = field(
        default=None, init=False, repr=False
    )
    _state: Dict[str, SetupState] = field(default_factory=dict, init=False)
    _last_bar_time: Dict[str, datetime] = field(default_factory=dict, init=False)
    _active_trade: Optional[TradeState] = field(default=None, init=False)

    def _get_contracts_client(self) -> AlpacaOptionsContractsClient:
        if self._contracts_client is None:
            self._contracts_client = AlpacaOptionsContractsClient(
                base_url=self.settings.alpaca_base_url,
                api_key=self.settings.alpaca_api_key,
                api_secret=self.settings.alpaca_api_secret,
            )
        return self._contracts_client

    def close(self) -> None:
        if self._contracts_client is not None:
            self._contracts_client.close()
            self._contracts_client = None

    def _get_state(self, symbol: str) -> SetupState:
        if symbol not in self._state:
            self._state[symbol] = SetupState()
        return self._state[symbol]

    def _reset_state(self, symbol: str) -> None:
        self._state[symbol] = SetupState()

    def _stop_buffer_amount(self, base_price: float) -> float:
        if self.settings.pop_pullback_stop_buffer_mode == "percent":
            return base_price * self.settings.pop_pullback_stop_buffer
        return self.settings.pop_pullback_stop_buffer

    def tick(self) -> None:
        if not self.settings.enable_options:
            return

        if self.settings.market_open_only and not self.broker.is_market_open():
            log.info("Market closed, skipping strategy tick")
            return

        try:
            acct = self.broker.account()
            equity = float(acct.equity)
        except Exception as e:
            log.error(f"Failed to get account info: {e}")
            return

        max_loss_hit = self.risk.daily_loss_exceeded(current_equity_usd=equity)
        if max_loss_hit and self._active_trade is None:
            log.warning("Max daily loss exceeded, skipping new entries")
            return

        # Only one trade active at a time
        if self._active_trade:
            self._process_symbol(self._active_trade.underlying, equity)
            return

        for symbol in self.settings.symbols_list:
            self._process_symbol(symbol, equity)

    def _process_symbol(self, symbol: str, equity: float) -> None:
        bars = fetch_bars(
            self.settings.alpaca_api_key,
            self.settings.alpaca_api_secret,
            symbol=symbol,
            timeframe="3Min",
            limit=250,
            feed=self.settings.alpaca_data_feed,
        )
        if bars.empty:
            return

        closes = bars["close"]
        highs = bars["high"]
        lows = bars["low"]
        opens = bars["open"]
        ema_series = ema(closes, self.settings.pop_pullback_ema_length)

        if len(ema_series) < 3:
            return

        # Determine new bars since last tick
        last_time = self._last_bar_time.get(symbol)
        start_idx = len(bars) - 1
        if last_time is not None and last_time in bars.index:
            start_idx = bars.index.get_loc(last_time) + 1
        start_idx = max(1, start_idx)  # ensure we have i-1

        for i in range(start_idx, len(bars)):
            self._process_bar(
                symbol=symbol,
                i=i,
                bars=bars,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                ema_series=ema_series,
                equity=equity,
            )

        self._last_bar_time[symbol] = bars.index[-1]

    def _process_bar(
        self,
        *,
        symbol: str,
        i: int,
        bars,
        opens,
        highs,
        lows,
        closes,
        ema_series,
        equity: float,
    ) -> None:
        ema_now = float(ema_series.iloc[i])
        close_now = float(closes.iloc[i])
        high_now = float(highs.iloc[i])
        low_now = float(lows.iloc[i])

        # Manage active trade first
        if self._active_trade and self._active_trade.underlying == symbol:
            self._manage_trade(
                trade=self._active_trade,
                ema_now=ema_now,
                close_now=close_now,
                high_now=high_now,
                low_now=low_now,
            )
            # Trade may have been closed
            if self._active_trade is None:
                self._reset_state(symbol)
            return

        state = self._get_state(symbol)

        # Phase: idle -> look for pop condition
        if state.phase == "idle":
            if i == 0:
                return
            prev_close = float(closes.iloc[i - 1])
            prev_ema = float(ema_series.iloc[i - 1])

            # Call pop condition
            if close_now > ema_now and prev_close <= prev_ema:
                state.phase = "pullback"
                state.direction = "call"
                state.pop_index = i
                return

            # Put pop condition
            if close_now < ema_now and prev_close >= prev_ema:
                state.phase = "pullback"
                state.direction = "put"
                state.pop_index = i
                return

            return

        # Phase: pullback search
        if state.phase == "pullback":
            if state.direction == "call":
                if low_now <= ema_now and close_now >= ema_now:
                    state.phase = "hold"
                    state.pullback_index = i
                    state.pullback_close = close_now
                    state.hold_count = 0
                    state.hold_closes = []
            else:
                if high_now >= ema_now and close_now <= ema_now:
                    state.phase = "hold"
                    state.pullback_index = i
                    state.pullback_close = close_now
                    state.hold_count = 0
                    state.hold_closes = []
            return

        # Phase: hold confirmation
        if state.phase == "hold":
            hold_required = self.settings.pop_pullback_hold_candles_required
            if state.direction == "call":
                if close_now > ema_now:
                    state.hold_count += 1
                    state.hold_closes.append(close_now)
                else:
                    self._reset_state(symbol)
                    return
            else:
                if close_now < ema_now:
                    state.hold_count += 1
                    state.hold_closes.append(close_now)
                else:
                    self._reset_state(symbol)
                    return

            if state.hold_count < hold_required:
                return

            if self.settings.pop_pullback_strength_filter:
                if state.direction == "call":
                    if hold_required == 1:
                        if not (state.hold_closes[0] > (state.pullback_close or 0.0)):
                            self._reset_state(symbol)
                            return
                    else:
                        if not (
                            state.hold_closes[0] > (state.pullback_close or 0.0)
                            and state.hold_closes[1] > state.hold_closes[0]
                        ):
                            self._reset_state(symbol)
                            return
                else:
                    if hold_required == 1:
                        if not (state.hold_closes[0] < (state.pullback_close or 0.0)):
                            self._reset_state(symbol)
                            return
                    else:
                        if not (
                            state.hold_closes[0] < (state.pullback_close or 0.0)
                            and state.hold_closes[1] < state.hold_closes[0]
                        ):
                            self._reset_state(symbol)
                            return

            # Confirmation achieved
            state.phase = "entry_wait"
            state.confirmation_index = i
            state.confirmation_high = high_now
            state.confirmation_low = low_now
            state.entry_wait_bars = 0

            # Compute base low/high for stop
            if state.pullback_index is not None:
                pb = state.pullback_index
                base_low = float(lows.iloc[pb:i + 1].min())
                base_high = float(highs.iloc[pb:i + 1].max())
                state.base_low = base_low
                state.base_high = base_high
            return

        # Phase: entry wait
        if state.phase == "entry_wait":
            if state.confirmation_high is None or state.confirmation_low is None:
                self._reset_state(symbol)
                return

            entry_triggered = False
            entry_underlying = None

            if state.direction == "call":
                if high_now > state.confirmation_high:
                    entry_triggered = True
                    entry_underlying = state.confirmation_high
            else:
                if low_now < state.confirmation_low:
                    entry_triggered = True
                    entry_underlying = state.confirmation_low

            if entry_triggered and entry_underlying is not None:
                self._enter_trade(
                    symbol=symbol,
                    direction=state.direction or "call",
                    entry_underlying=entry_underlying,
                    base_low=state.base_low,
                    base_high=state.base_high,
                    equity=equity,
                )
                self._reset_state(symbol)
                return

            state.entry_wait_bars += 1
            if state.entry_wait_bars >= self.settings.pop_pullback_entry_timeout_candles:
                self._reset_state(symbol)
            return

    def _enter_trade(
        self,
        *,
        symbol: str,
        direction: str,
        entry_underlying: float,
        base_low: Optional[float],
        base_high: Optional[float],
        equity: float,
    ) -> None:
        if self._active_trade is not None:
            return

        # Guard against existing open positions/orders
        if self.broker.list_positions():
            log.info("Existing positions detected, skipping new entry")
            return

        contracts_client = self._get_contracts_client()
        preferred_type = "call" if direction == "call" else "put"

        try:
            contracts = contracts_client.list_contracts(
                underlying=symbol,
                contract_type=preferred_type,
                exp_gte=None,
                exp_lte=None,
                limit=200,
            )
        except Exception as e:
            log.error(f"Failed to fetch contracts for {symbol}: {e}")
            return

        contract = pick_atm_contract(
            contracts,
            underlying_price=entry_underlying,
            strike_tolerance=self.settings.option_strike_tolerance,
            dte_max=self.settings.option_dte_max,
            preferred_type=preferred_type,
            client=contracts_client,
        )

        if not contract:
            log.warning(f"No suitable {preferred_type} contract for {symbol}")
            return

        premium = contracts_client.get_latest_option_mid_price(contract.symbol)

        if self.settings.option_use_dynamic_qty and premium:
            qty = self.risk.calc_option_qty(equity_usd=equity, premium=premium)
        else:
            qty = self.settings.option_order_qty

        if qty < 1:
            log.info(f"{contract.symbol}: qty=0, skipping entry")
            return

        # Stop price based on underlying
        if direction == "call":
            base_price = base_low if base_low is not None else entry_underlying
            stop_price = base_price - self._stop_buffer_amount(base_price)
        else:
            base_price = base_high if base_high is not None else entry_underlying
            stop_price = base_price + self._stop_buffer_amount(base_price)

        try:
            order_id = self.broker.place_order(
                OrderRequest(symbol=contract.symbol, side="buy", qty=qty)
            )
        except Exception as e:
            log.error(f"Entry order failed for {contract.symbol}: {e}")
            return

        self.store.log_trade(
            symbol=contract.symbol,
            side="buy",
            qty=qty,
            order_type="market",
            status="submitted",
            order_id=order_id,
            metadata={
                "underlying": symbol,
                "entry_underlying": entry_underlying,
                "stop_price": stop_price,
                "contract_type": contract.type,
                "expiration": contract.expiration_date,
                "strike": contract.strike_price,
                "premium": premium,
                "strategy": "pop_pullback_hold",
            },
        )

        self._active_trade = TradeState(
            underlying=symbol,
            option_symbol=contract.symbol,
            direction=direction,
            entry_underlying=entry_underlying,
            entry_option=premium,
            stop_price=stop_price,
            original_qty=int(qty),
            remaining_qty=int(qty),
            trimmed=False,
        )

        log.info(
            f"Entered {direction.upper()} {contract.symbol} qty={qty} "
            f"entry_underlying={entry_underlying:.2f} stop={stop_price:.2f}"
        )

    def _manage_trade(
        self,
        *,
        trade: TradeState,
        ema_now: float,
        close_now: float,
        high_now: float,
        low_now: float,
    ) -> None:
        # Stop loss checks (underlying-based)
        if trade.direction == "call":
            if close_now <= trade.stop_price or low_now <= trade.stop_price:
                self._exit_full(trade, reason="stop")
                return
        else:
            if close_now >= trade.stop_price or high_now >= trade.stop_price:
                self._exit_full(trade, reason="stop")
                return

        # Partial profit target
        if not trade.trimmed:
            profit_on_underlying = self.settings.pop_pullback_profit_calc_on_underlying
            if profit_on_underlying:
                current_price = close_now
                entry_price = trade.entry_underlying
            else:
                current_price = self._get_option_price(trade.option_symbol)
                entry_price = trade.entry_option

            if current_price is not None and entry_price:
                target_pct = self.settings.pop_pullback_target_profit_pct
                if trade.direction == "call":
                    hit = current_price >= entry_price * (1 + target_pct)
                else:
                    if profit_on_underlying:
                        hit = current_price <= entry_price * (1 - target_pct)
                    else:
                        # Option price profits when price increases for long puts
                        hit = current_price >= entry_price * (1 + target_pct)

                if hit:
                    trim_qty = int(trade.original_qty * 0.8)
                    if trim_qty < 1:
                        self._exit_full(trade, reason="target")
                        return
                    if trim_qty >= trade.remaining_qty:
                        self._exit_full(trade, reason="target")
                        return

                    if self._exit_partial(trade, trim_qty, reason="target"):
                        trade.remaining_qty -= trim_qty
                        trade.trimmed = True

        # EMA exit for remaining position
        if trade.remaining_qty <= 0:
            self._exit_full(trade, reason="position_empty")
            return

        ema_exit_mode = self.settings.pop_pullback_ema_exit_mode
        if trade.direction == "call":
            if ema_exit_mode == "close":
                crossed = close_now < ema_now
            else:
                crossed = low_now < ema_now
        else:
            if ema_exit_mode == "close":
                crossed = close_now > ema_now
            else:
                crossed = high_now > ema_now

        if crossed:
            self._exit_full(trade, reason="ema_exit")

    def _get_option_price(self, symbol: str) -> Optional[float]:
        try:
            return self._get_contracts_client().get_latest_option_mid_price(symbol)
        except Exception:
            return None

    def _exit_partial(self, trade: TradeState, qty: int, reason: str) -> bool:
        try:
            order_id = self.broker.place_order(
                OrderRequest(symbol=trade.option_symbol, side="sell", qty=qty)
            )
            self.store.log_trade(
                symbol=trade.option_symbol,
                side="sell",
                qty=qty,
                order_type="market",
                status="submitted",
                order_id=order_id,
                metadata={
                    "underlying": trade.underlying,
                    "reason": reason,
                    "strategy": "pop_pullback_hold",
                    "partial": True,
                },
            )
            log.info(f"Partial exit {trade.option_symbol} qty={qty} reason={reason}")
            return True
        except Exception as e:
            log.error(f"Partial exit failed for {trade.option_symbol}: {e}")
            return False

    def _exit_full(self, trade: TradeState, reason: str) -> None:
        qty = trade.remaining_qty
        if qty <= 0:
            self._active_trade = None
            return

        try:
            order_id = self.broker.place_order(
                OrderRequest(symbol=trade.option_symbol, side="sell", qty=qty)
            )
            self.store.log_trade(
                symbol=trade.option_symbol,
                side="sell",
                qty=qty,
                order_type="market",
                status="submitted",
                order_id=order_id,
                metadata={
                    "underlying": trade.underlying,
                    "reason": reason,
                    "strategy": "pop_pullback_hold",
                    "partial": False,
                },
            )
            log.info(f"Full exit {trade.option_symbol} qty={qty} reason={reason}")
        except Exception as e:
            log.error(f"Full exit failed for {trade.option_symbol}: {e}")
        finally:
            self._active_trade = None
