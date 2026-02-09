"""Tests for EMA strategy behavior with and without sentiment gating."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from tradebot.config import Settings
from tradebot.strategy.ema_sentiment import EmaSentimentStrategy


def _make_bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 09:30:00", periods=3, freq="min", tz="UTC")
    return pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)


def _fake_ema(series: pd.Series, span: int) -> pd.Series:
    if span == 9:
        # prev_diff <= 0 and curr_diff > 0 against slow EMA below
        return pd.Series([1.0, 1.0, 2.0], index=series.index)
    return pd.Series([1.5, 1.5, 1.5], index=series.index)


def _make_strategy(use_sentiment: bool) -> tuple[EmaSentimentStrategy, MagicMock]:
    settings = Settings(
        symbols="SPY",
        ema_fast=9,
        ema_slow=21,
        sentiment_threshold=0.5,
        use_sentiment=use_sentiment,
        trade_cooldown_minutes=0,
    )

    broker = MagicMock()
    broker.get_position.return_value = None
    broker.list_open_orders.return_value = []
    broker.place_order.return_value = "order-1"

    risk = MagicMock()
    risk.calc_qty.return_value = 10.0
    risk.position_value_ok.return_value = True
    risk.bracket_prices.return_value = (99.5, 101.0)

    store = MagicMock()
    store.get_last_trade_time.return_value = None

    strategy = EmaSentimentStrategy(
        settings=settings,
        broker=broker,
        risk=risk,
        store=store,
    )
    return strategy, broker


@patch("tradebot.strategy.ema_sentiment.ema", side_effect=_fake_ema)
@patch("tradebot.strategy.ema_sentiment.fetch_bars", return_value=_make_bars())
def test_technical_mode_ignores_sentiment_gate(_mock_fetch_bars, _mock_ema):
    """Bull crossover should trade even with low sentiment when sentiment is disabled."""
    strategy, broker = _make_strategy(use_sentiment=False)

    strategy._process_symbol(
        symbol="SPY",
        equity=100_000.0,
        sentiment=0.0,
        use_sentiment=False,
    )

    broker.place_order.assert_called_once()


@patch("tradebot.strategy.ema_sentiment.ema", side_effect=_fake_ema)
@patch("tradebot.strategy.ema_sentiment.fetch_bars", return_value=_make_bars())
def test_sentiment_mode_still_requires_threshold(_mock_fetch_bars, _mock_ema):
    """Bull crossover should be blocked if sentiment threshold is not met."""
    strategy, broker = _make_strategy(use_sentiment=True)

    strategy._process_symbol(
        symbol="SPY",
        equity=100_000.0,
        sentiment=0.1,  # Below configured threshold of 0.5
        use_sentiment=True,
    )

    broker.place_order.assert_not_called()
