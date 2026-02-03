from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta, datetime, timezone
from typing import Optional

from tradebot.config import Settings
from tradebot.core.logger import get_logger
from tradebot.core.timeutils import utcnow
from tradebot.data.marketdata import fetch_bars, ema
from tradebot.broker.alpaca_broker import AlpacaBroker
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

    def _sentiment_gate(self) -> float:
        """Get average sentiment from recent events."""
        since = (utcnow() - timedelta(minutes=30)).isoformat()
        rows = self.store.recent_sentiment(since_iso=since, limit=50)
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

        try:
            acct = self.broker.account()
            equity = float(acct.equity)
        except Exception as e:
            log.error(f"Failed to get account info: {e}")
            return

        sentiment = self._sentiment_gate()
        log.info(f"[OPTIONS] Recent sentiment avg: {sentiment:.3f}")

        contracts_client = self._get_contracts_client()

        for underlying in self.settings.symbols_list:
            try:
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

        # Place the order
        qty = self.settings.option_order_qty

        log.info(
            f"[OPTIONS] BUY {contract.symbol} "
            f"(underlying={underlying}, price={underlying_price:.2f}, "
            f"strike={contract.strike_price}, exp={contract.expiration_date}) "
            f"qty={qty}"
        )

        try:
            order_id = self.broker.place_order(
                OrderRequest(symbol=contract.symbol, side="buy", qty=qty)
            )

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
                },
            )
            log.info(f"[OPTIONS] Order submitted: {order_id}")

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
        )
