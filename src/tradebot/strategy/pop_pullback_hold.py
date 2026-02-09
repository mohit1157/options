from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from typing import ClassVar, Optional, Dict, List, Tuple

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
    signal_id: Optional[int] = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    realized_pnl_usd: float = 0.0
    trimmed: bool = False


@dataclass
class EmaPopPullbackHoldOptionsStrategy:
    """3-Min 9 EMA Pop Pullback Hold Confirmation Strategy for options."""

    CALIBRATION_STRATEGY: ClassVar[str] = "pop_pullback_hold"
    CALIBRATION_LOOKBACK_DAYS: ClassVar[int] = 7
    CALIBRATION_MIN_TRADES: ClassVar[int] = 8
    CALIBRATION_CHECK_INTERVAL_HOURS: ClassVar[int] = 12

    settings: Settings
    broker: AlpacaBroker
    risk: RiskManager
    store: SQLiteStore

    _contracts_client: Optional[AlpacaOptionsContractsClient] = field(
        default=None, init=False, repr=False
    )
    _state: Dict[str, SetupState] = field(default_factory=dict, init=False)
    _last_bar_time: Dict[str, datetime] = field(default_factory=dict, init=False)
    _active_trades: Dict[str, TradeState] = field(default_factory=dict, init=False)
    _recovered_symbols: set[str] = field(default_factory=set, init=False)
    _adaptive_target_profit_pct: Optional[float] = field(default=None, init=False)
    _adaptive_runner_target_profit_pct: Optional[float] = field(default=None, init=False)
    _last_calibration_check: Optional[datetime] = field(default=None, init=False)

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

    def _current_target_profit_pct(self) -> float:
        if self._adaptive_target_profit_pct is not None:
            return self._adaptive_target_profit_pct
        return self.settings.pop_pullback_target_profit_pct

    def _current_runner_target_profit_pct(self) -> float:
        if self._adaptive_runner_target_profit_pct is not None:
            return self._adaptive_runner_target_profit_pct
        return self.settings.pop_pullback_runner_target_profit_pct

    def _maybe_recalibrate_thresholds(self) -> None:
        """Recalibrate target thresholds weekly from realized outcomes."""
        now = datetime.now(timezone.utc)

        if self._last_calibration_check is not None:
            elapsed = now - self._last_calibration_check
            if elapsed < timedelta(hours=self.CALIBRATION_CHECK_INTERVAL_HOURS):
                return
        self._last_calibration_check = now

        last = self.store.get_last_calibration(strategy=self.CALIBRATION_STRATEGY)
        if last:
            last_time = last.get("last_calibrated_at")
            if isinstance(last_time, datetime):
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                if now - last_time < timedelta(days=self.CALIBRATION_LOOKBACK_DAYS):
                    params = last.get("params") if isinstance(last.get("params"), dict) else {}
                    if "target_profit_pct" in params:
                        self._adaptive_target_profit_pct = float(params["target_profit_pct"])
                    if "runner_target_profit_pct" in params:
                        self._adaptive_runner_target_profit_pct = float(params["runner_target_profit_pct"])
                    return

        since = now - timedelta(days=self.CALIBRATION_LOOKBACK_DAYS)
        stats = self.store.get_trade_outcome_stats(
            strategy=self.CALIBRATION_STRATEGY,
            since=since,
        )
        total_trades = int(stats.get("total_trades", 0) or 0)

        target = self._current_target_profit_pct()
        runner = self._current_runner_target_profit_pct()
        min_target, max_target = 0.06, 0.08
        min_runner_gap = 0.02
        max_runner = 0.25
        adjusted = False

        if total_trades >= self.CALIBRATION_MIN_TRADES:
            win_rate = float(stats.get("win_rate", 0.0) or 0.0)
            avg_pnl_pct = float(stats.get("avg_pnl_pct", 0.0) or 0.0)

            if win_rate < 0.45 or avg_pnl_pct < 0:
                target = max(min_target, round(target - 0.005, 4))
                runner = max(target + min_runner_gap, round(runner - 0.01, 4))
                adjusted = True
            elif win_rate > 0.58 and avg_pnl_pct > 0.03:
                target = min(max_target, round(target + 0.005, 4))
                runner = min(max_runner, round(max(target + min_runner_gap, runner + 0.01), 4))
                adjusted = True

        runner = round(max(target + min_runner_gap, runner), 4)
        target = round(target, 4)

        self._adaptive_target_profit_pct = target
        self._adaptive_runner_target_profit_pct = runner

        self.store.upsert_calibration(
            strategy=self.CALIBRATION_STRATEGY,
            last_calibrated_at=now,
            params={
                "target_profit_pct": target,
                "runner_target_profit_pct": runner,
            },
            stats={
                **stats,
                "adjusted": adjusted,
                "lookback_days": self.CALIBRATION_LOOKBACK_DAYS,
                "min_trades": self.CALIBRATION_MIN_TRADES,
            },
        )

        if adjusted:
            log.info(
                f"Recalibrated thresholds: target={target:.4f}, runner={runner:.4f}, "
                f"win_rate={float(stats.get('win_rate', 0.0)):.2%}, "
                f"avg_pnl_pct={float(stats.get('avg_pnl_pct', 0.0)):.4f}"
            )

    def _profit_target_hit(
        self,
        *,
        direction: str,
        current_price: Optional[float],
        entry_price: Optional[float],
        target_pct: float,
        qty: int,
        profit_on_underlying: bool,
    ) -> bool:
        if current_price is None or entry_price is None or qty < 1:
            return False

        if profit_on_underlying:
            if direction == "call":
                return current_price >= entry_price * (1 + target_pct)
            return current_price <= entry_price * (1 - target_pct)

        entry_value = entry_price * 100 * qty
        if entry_value <= 0:
            return False
        current_value = current_price * 100 * qty
        profit = current_value - entry_value
        target_profit = entry_value * target_pct
        return profit >= target_profit

    def _parse_option_symbol(self, symbol: str) -> Optional[Tuple[str, str]]:
        """Parse OCC option symbol to extract underlying + type.

        Returns:
            (underlying, direction) where direction is "call" or "put"
        """
        # Example: QQQ260206P00600000
        m = re.match(r"^([A-Za-z]{1,6})\d{6}([CP])\d{8}$", symbol)
        if not m:
            return None
        underlying = m.group(1).upper()
        opt_type = m.group(2)
        direction = "call" if opt_type == "C" else "put"
        return (underlying, direction)

    def _recover_active_trades(self) -> None:
        """Recover open options trades after restart, if any."""
        if self._active_trades:
            return

        try:
            positions = self.broker.list_positions()
        except Exception as e:
            log.debug(f"Recovery: failed to fetch positions: {e}")
            return

        option_positions = []
        for p in positions:
            parsed = self._parse_option_symbol(p.symbol)
            if parsed and abs(p.qty) > 0:
                option_positions.append((p, parsed))

        if not option_positions:
            return

        # Recover all option positions
        for pos, (underlying, direction) in option_positions:
            qty = int(abs(pos.qty))
            entry_option = float(pos.avg_entry_price) if pos.avg_entry_price else None

            # Use latest close as a placeholder for entry_underlying (not used for premium-based exits)
            entry_underlying = 0.0
            try:
                bars = fetch_bars(
                    self.settings.alpaca_api_key,
                    self.settings.alpaca_api_secret,
                    symbol=underlying,
                    timeframe="3Min",
                    limit=1,
                    feed=self.settings.alpaca_data_feed,
                )
                if not bars.empty:
                    entry_underlying = float(bars["close"].iloc[-1])
            except Exception:
                pass

            # Try to recover original qty from audit log (if larger than current)
            original_qty = qty
            recovered_stop_price = 0.0
            try:
                history = self.store.get_trade_history(symbol=pos.symbol, limit=20)
                for rec in history:
                    if rec.get("side") == "buy":
                        try:
                            original_qty = int(float(rec.get("qty") or qty))
                        except Exception:
                            original_qty = qty
                        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                        try:
                            recovered_stop_price = float(metadata.get("stop_price", 0.0))
                        except Exception:
                            recovered_stop_price = 0.0
                        break
            except Exception:
                original_qty = qty

            trimmed = qty < original_qty

            self._active_trades[pos.symbol] = TradeState(
                underlying=underlying,
                option_symbol=pos.symbol,
                direction=direction,
                entry_underlying=entry_underlying,
                entry_option=entry_option,
                stop_price=recovered_stop_price,
                original_qty=original_qty,
                remaining_qty=qty,
                trimmed=trimmed,
            )

            if pos.symbol not in self._recovered_symbols:
                self._recovered_symbols.add(pos.symbol)
                log.info(
                    f"Recovered open options position: {pos.symbol} "
                    f"qty={qty} dir={direction} entry_px={entry_option}"
                )

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

        self._maybe_recalibrate_thresholds()

        # Recover an open trade if bot restarted and an options position exists
        if not self._active_trades:
            self._recover_active_trades()

        max_loss_hit = self.risk.daily_loss_exceeded(current_equity_usd=equity)
        if max_loss_hit and not self._active_trades:
            log.warning("Max daily loss exceeded, skipping new entries")
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
        lookback = self.settings.pop_pullback_zone_lookback_bars
        end = i  # exclude current bar from zone calculation
        start = max(0, end - lookback)
        if end <= start:
            zone_low = None
            zone_high = None
        else:
            zone_low = float(lows.iloc[start:end].min())
            zone_high = float(highs.iloc[start:end].max())

        # Manage active trades for this underlying first
        active_for_symbol = [
            t for t in self._active_trades.values() if t.underlying == symbol
        ]
        if active_for_symbol:
            for trade in list(active_for_symbol):
                self._manage_trade(
                    trade=trade,
                    ema_now=ema_now,
                    close_now=close_now,
                    high_now=high_now,
                    low_now=low_now,
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
            if not any(t.underlying == symbol for t in self._active_trades.values()):
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
        if self._active_trades:
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
            qty = self.risk.calc_option_qty(
                equity_usd=equity,
                premium=premium,
                portfolio_pct=self.settings.option_portfolio_pct,
            )
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

        signal_features = {
            "underlying": symbol,
            "direction": direction,
            "entry_underlying": entry_underlying,
            "base_low": base_low,
            "base_high": base_high,
            "stop_price": stop_price,
            "contract_symbol": contract.symbol,
            "contract_type": contract.type,
            "expiration": contract.expiration_date,
            "strike": contract.strike_price,
            "premium": premium,
            "target_profit_pct": self._current_target_profit_pct(),
            "runner_target_profit_pct": self._current_runner_target_profit_pct(),
            "stop_buffer": self.settings.pop_pullback_stop_buffer,
            "stop_buffer_mode": self.settings.pop_pullback_stop_buffer_mode,
            "equity": equity,
            "qty": qty,
        }
        signal_id = self.store.log_signal_features(
            strategy=self.CALIBRATION_STRATEGY,
            symbol=contract.symbol,
            side="buy",
            timeframe="3Min",
            features=signal_features,
        )

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
                "strategy": self.CALIBRATION_STRATEGY,
                "signal_id": signal_id,
            },
        )

        self._active_trades[contract.symbol] = TradeState(
            underlying=symbol,
            option_symbol=contract.symbol,
            direction=direction,
            entry_underlying=entry_underlying,
            entry_option=premium,
            stop_price=stop_price,
            original_qty=int(qty),
            remaining_qty=int(qty),
            signal_id=signal_id,
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
        zone_low: Optional[float],
        zone_high: Optional[float],
    ) -> None:
        # Snapshot PnL + portfolio status for logging
        entry_option = trade.entry_option
        target_pct = self._current_target_profit_pct()
        runner_pct = self._current_runner_target_profit_pct()
        entry_value = None
        profit = None
        profit_pct = None
        target_profit = None
        runner_target_profit = None
        price_source = "mid"

        active_trades = None
        total_unrealized = None
        current_price = None
        pos_unrealized = None
        try:
            positions = self.broker.list_positions()
            active_trades = len(positions)
            total_unrealized = sum(p.unrealized_pl for p in positions)
            pos = next((p for p in positions if p.symbol == trade.option_symbol), None)
            if pos and abs(pos.qty) > 0:
                mark_px = abs(pos.market_value) / (abs(pos.qty) * 100)
                if mark_px > 0:
                    current_price = mark_px
                    price_source = "mark"
                    pos_unrealized = pos.unrealized_pl
        except Exception as e:
            log.debug(f"Status log: failed to fetch positions: {e}")

        if current_price is None:
            current_price = self._get_option_price(trade.option_symbol)
            price_source = "mid"

        if current_price is not None and entry_option:
            entry_value = entry_option * 100 * trade.original_qty
            profit = (pos_unrealized if pos_unrealized is not None else (current_price * 100 * trade.original_qty - entry_value))
            if entry_value > 0:
                profit_pct = (profit / entry_value) * 100
            target_profit = entry_value * target_pct
            runner_entry_value = entry_option * 100 * max(1, trade.remaining_qty)
            runner_target_profit = runner_entry_value * runner_pct

        parts = [
            f"Trade status {trade.option_symbol}",
            f"dir={trade.direction}",
            f"qty={trade.remaining_qty}/{trade.original_qty}",
        ]
        if current_price is not None and entry_option:
            parts.append(f"opt_px={current_price:.2f}({price_source}) entry_px={entry_option:.2f}")
            if profit is not None and profit_pct is not None:
                parts.append(f"pnl=${profit:.2f} ({profit_pct:.2f}%)")
            if target_profit is not None:
                parts.append(f"target=${target_profit:.2f}")
            if trade.trimmed and runner_target_profit is not None:
                parts.append(f"runner_target=${runner_target_profit:.2f}")
        else:
            parts.append("opt_px=NA")

        if zone_low is not None and zone_high is not None:
            parts.append(f"zone_low={zone_low:.2f} zone_high={zone_high:.2f}")
        if trade.stop_price > 0:
            parts.append(f"stop_price={trade.stop_price:.2f}")
        if active_trades is not None:
            parts.append(f"active_trades={active_trades}")
        if total_unrealized is not None:
            parts.append(f"total_unrealized=${total_unrealized:.2f}")

        log.info(" | ".join(parts))

        # Entry-based hard stop on underlying
        if trade.stop_price > 0:
            if trade.direction == "call":
                stop_hit = low_now <= trade.stop_price or close_now <= trade.stop_price
            else:
                stop_hit = high_now >= trade.stop_price or close_now >= trade.stop_price
            if stop_hit:
                self._exit_full(trade, reason="stop_price", estimated_price=current_price)
                return

        # Zone-based exits (support/resistance)
        zone_stop = False
        zone_target = False
        if zone_low is not None and zone_high is not None:
            if trade.direction == "call":
                if low_now <= zone_low or close_now <= zone_low:
                    zone_stop = True
                if high_now >= zone_high or close_now >= zone_high:
                    zone_target = True
            else:
                if high_now >= zone_high or close_now >= zone_high:
                    zone_stop = True
                if low_now <= zone_low or close_now <= zone_low:
                    zone_target = True

        if zone_stop:
            self._exit_full(trade, reason="zone_stop", estimated_price=current_price)
            return

        # Partial profit target
        if not trade.trimmed:
            profit_on_underlying = self.settings.pop_pullback_profit_calc_on_underlying
            if profit_on_underlying:
                target_price = close_now
                entry_price = trade.entry_underlying
            else:
                target_price = current_price
                entry_price = trade.entry_option

            hit = zone_target
            hit = hit or self._profit_target_hit(
                direction=trade.direction,
                current_price=target_price,
                entry_price=entry_price,
                target_pct=target_pct,
                qty=trade.original_qty,
                profit_on_underlying=profit_on_underlying,
            )

            if hit:
                trim_qty = int(trade.original_qty * 0.9)
                if trim_qty < 1:
                    self._exit_full(trade, reason="target", estimated_price=current_price)
                    return
                if trim_qty >= trade.remaining_qty:
                    self._exit_full(trade, reason="target", estimated_price=current_price)
                    return

                if self._exit_partial(trade, trim_qty, reason="target", estimated_price=current_price):
                    trade.remaining_qty -= trim_qty
                    trade.trimmed = True

        # Runner exit for remaining position: 11% target or EMA fallback.
        if trade.trimmed:
            profit_on_underlying = self.settings.pop_pullback_profit_calc_on_underlying
            if profit_on_underlying:
                runner_price = close_now
                runner_entry_price = trade.entry_underlying
            else:
                runner_price = current_price
                runner_entry_price = trade.entry_option

            runner_hit = self._profit_target_hit(
                direction=trade.direction,
                current_price=runner_price,
                entry_price=runner_entry_price,
                target_pct=runner_pct,
                qty=max(1, trade.remaining_qty),
                profit_on_underlying=profit_on_underlying,
            )
            if runner_hit:
                self._exit_full(trade, reason="runner_target", estimated_price=current_price)
                return

        # EMA exit for remaining position
        if trade.remaining_qty <= 0:
            self._exit_full(trade, reason="position_empty", estimated_price=current_price)
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
            self._exit_full(trade, reason="ema_exit", estimated_price=current_price)

    def _get_option_price(self, symbol: str) -> Optional[float]:
        try:
            return self._get_contracts_client().get_latest_option_mid_price(symbol)
        except Exception:
            return None

    def _estimate_realized_pnl(
        self,
        *,
        entry_price: Optional[float],
        exit_price: Optional[float],
        qty: int,
    ) -> float:
        if qty <= 0:
            return 0.0
        if entry_price is None or exit_price is None:
            return 0.0
        return (exit_price - entry_price) * 100 * qty

    def _exit_partial(
        self,
        trade: TradeState,
        qty: int,
        reason: str,
        estimated_price: Optional[float] = None,
    ) -> bool:
        if qty <= 0:
            return False

        exit_price = estimated_price if estimated_price is not None else self._get_option_price(trade.option_symbol)
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
                    "strategy": self.CALIBRATION_STRATEGY,
                    "partial": True,
                    "signal_id": trade.signal_id,
                    "estimated_exit_price": exit_price,
                },
            )
            trade.realized_pnl_usd += self._estimate_realized_pnl(
                entry_price=trade.entry_option,
                exit_price=exit_price,
                qty=qty,
            )
            log.info(f"Partial exit {trade.option_symbol} qty={qty} reason={reason}")
            return True
        except Exception as e:
            log.error(f"Partial exit failed for {trade.option_symbol}: {e}")
            return False

    def _exit_full(
        self,
        trade: TradeState,
        reason: str,
        estimated_price: Optional[float] = None,
    ) -> None:
        qty = trade.remaining_qty
        if qty <= 0:
            self._active_trades.pop(trade.option_symbol, None)
            return

        exit_price = estimated_price if estimated_price is not None else self._get_option_price(trade.option_symbol)
        closed = False
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
                    "strategy": self.CALIBRATION_STRATEGY,
                    "partial": False,
                    "signal_id": trade.signal_id,
                    "estimated_exit_price": exit_price,
                },
            )
            closed = True
            log.info(f"Full exit {trade.option_symbol} qty={qty} reason={reason}")
        except Exception as e:
            log.error(f"Full exit failed for {trade.option_symbol}: {e}")
        finally:
            if closed:
                realized_final_leg = self._estimate_realized_pnl(
                    entry_price=trade.entry_option,
                    exit_price=exit_price,
                    qty=qty,
                )
                realized_pnl_usd = trade.realized_pnl_usd + realized_final_leg
                entry_notional = (
                    trade.entry_option * 100 * trade.original_qty
                    if trade.entry_option is not None and trade.original_qty > 0
                    else 0.0
                )
                pnl_pct = (realized_pnl_usd / entry_notional) if entry_notional > 0 else None
                self.store.log_trade_outcome(
                    strategy=self.CALIBRATION_STRATEGY,
                    symbol=trade.option_symbol,
                    side="buy",
                    qty=trade.original_qty,
                    pnl_usd=realized_pnl_usd,
                    pnl_pct=pnl_pct,
                    is_win=realized_pnl_usd > 0,
                    signal_id=trade.signal_id,
                    entry_price=trade.entry_option,
                    exit_price=exit_price,
                    closed_at=datetime.now(timezone.utc),
                    metadata={
                        "underlying": trade.underlying,
                        "direction": trade.direction,
                        "reason": reason,
                        "trimmed": trade.trimmed,
                        "opened_at": trade.opened_at.isoformat(),
                    },
                )
            self._active_trades.pop(trade.option_symbol, None)
