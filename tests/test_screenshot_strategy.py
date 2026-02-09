"""Tests for screenshot strategy baseline pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.learning import (
    ScreenshotOutcomeModel,
    ScreenshotTradeExample,
    load_screenshot_examples,
)


def _write_dataset(tmp_path: Path) -> Path:
    dataset = {
        "examples": [
            {
                "trade_id": "t1",
                "screenshot_path": "img1.png",
                "symbol": "SPY",
                "timeframe": "3Min",
                "side": "call",
                "entry_price": 2.0,
                "exit_price": 2.2,
                "pattern": "pop_pullback_hold",
                "tags": ["trend_day", "ema_reclaim"],
                "rationale": "Pop and hold above EMA.",
            },
            {
                "trade_id": "t2",
                "screenshot_path": "img2.png",
                "symbol": "SPY",
                "timeframe": "3Min",
                "side": "call",
                "entry_price": 1.8,
                "exit_price": 1.6,
                "pattern": "pop_pullback_hold",
                "tags": ["chop"],
                "rationale": "Late entry and weak momentum.",
            },
            {
                "trade_id": "t3",
                "screenshot_path": "img3.png",
                "symbol": "QQQ",
                "timeframe": "3Min",
                "side": "put",
                "entry_price": 2.1,
                "exit_price": 1.7,
                "pattern": "rejection_fade",
                "tags": ["resistance_reject"],
                "rationale": "Double top then break down.",
            },
        ]
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


def test_load_examples_and_pnl(tmp_path: Path):
    data_path = _write_dataset(tmp_path)
    examples = load_screenshot_examples(data_path)
    assert len(examples) == 3
    assert isinstance(examples[0], ScreenshotTradeExample)

    # call winner
    assert examples[0].pnl_pct() == pytest.approx(0.1, rel=1e-5)
    # put winner (entry > exit)
    assert examples[2].pnl_pct() == pytest.approx((2.1 - 1.7) / 2.1, rel=1e-5)


def test_model_fit_predict_and_evaluate(tmp_path: Path):
    data_path = _write_dataset(tmp_path)
    examples = load_screenshot_examples(data_path)

    model = ScreenshotOutcomeModel().fit(examples)
    pred = model.predict(examples[0])

    assert -1.0 <= pred.expected_return <= 1.0
    assert 0.0 <= pred.win_probability <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.evidence_count > 0

    metrics = model.evaluate(examples)
    assert metrics["count"] == 3.0
    assert metrics["mae_expected_return"] >= 0
    assert 0 <= metrics["directional_accuracy"] <= 1


def test_model_save_load_roundtrip(tmp_path: Path):
    data_path = _write_dataset(tmp_path)
    examples = load_screenshot_examples(data_path)

    model = ScreenshotOutcomeModel().fit(examples)
    model_path = tmp_path / "model.json"
    model.save(model_path)

    loaded = ScreenshotOutcomeModel.load(model_path)
    pred = loaded.predict(examples[1])
    assert 0.0 <= pred.win_probability <= 1.0
    assert isinstance(pred.contributors, list)

