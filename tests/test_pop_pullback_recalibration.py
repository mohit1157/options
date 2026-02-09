"""Tests for pop-pullback weekly threshold recalibration."""
from __future__ import annotations

from unittest.mock import MagicMock

from tradebot.config import Settings
from tradebot.strategy.pop_pullback_hold import EmaPopPullbackHoldOptionsStrategy


def _make_strategy() -> tuple[EmaPopPullbackHoldOptionsStrategy, MagicMock]:
    settings = Settings(
        symbols="SPY",
        enable_options=True,
        pop_pullback_target_profit_pct=0.07,
        pop_pullback_runner_target_profit_pct=0.11,
    )
    broker = MagicMock()
    risk = MagicMock()
    store = MagicMock()
    strategy = EmaPopPullbackHoldOptionsStrategy(
        settings=settings,
        broker=broker,
        risk=risk,
        store=store,
    )
    return strategy, store


def test_recalibration_tightens_targets_on_weak_performance():
    """Poor weekly stats should make target thresholds more conservative."""
    strategy, store = _make_strategy()
    store.get_last_calibration.return_value = None
    store.get_trade_outcome_stats.return_value = {
        "total_trades": 12,
        "wins": 4,
        "losses": 8,
        "win_rate": 4 / 12,
        "avg_pnl_usd": -8.0,
        "avg_pnl_pct": -0.02,
        "total_pnl_usd": -96.0,
    }

    strategy._maybe_recalibrate_thresholds()

    assert strategy._current_target_profit_pct() == 0.065
    assert strategy._current_runner_target_profit_pct() == 0.1
    store.upsert_calibration.assert_called_once()


def test_recalibration_expands_targets_on_strong_performance():
    """Strong weekly stats should allow slightly larger profit thresholds."""
    strategy, store = _make_strategy()
    store.get_last_calibration.return_value = None
    store.get_trade_outcome_stats.return_value = {
        "total_trades": 15,
        "wins": 10,
        "losses": 5,
        "win_rate": 10 / 15,
        "avg_pnl_usd": 18.0,
        "avg_pnl_pct": 0.05,
        "total_pnl_usd": 270.0,
    }

    strategy._maybe_recalibrate_thresholds()

    assert strategy._current_target_profit_pct() == 0.075
    assert strategy._current_runner_target_profit_pct() == 0.12
    store.upsert_calibration.assert_called_once()
