#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tradebot.learning import ScreenshotOutcomeModel, load_screenshot_examples


def _split_examples(examples, test_ratio: float, seed: int):
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)

    if len(items) < 2:
        return items, []

    test_size = int(len(items) * test_ratio)
    test_size = min(max(test_size, 1), len(items) - 1)
    split_idx = len(items) - test_size
    return items[:split_idx], items[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline screenshot outcome model from labeled trade examples."
    )
    parser.add_argument("--data", required=True, help="Path to JSON/JSONL labeled trade data")
    parser.add_argument(
        "--model-out",
        default="artifacts/screenshot_strategy_model.json",
        help="Output model path",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail if screenshot paths in data do not exist",
    )
    args = parser.parse_args()

    examples = load_screenshot_examples(args.data, strict_paths=args.strict_paths)
    if not examples:
        raise SystemExit("No examples found in dataset.")

    train, test = _split_examples(examples, args.test_ratio, args.seed)
    model = ScreenshotOutcomeModel()
    model.fit(train)

    train_metrics = model.evaluate(train)
    test_metrics = model.evaluate(test) if test else {}

    out_path = Path(args.model_out)
    model.save(out_path)

    print(f"Loaded examples: {len(examples)}")
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    print(f"Model saved to: {out_path.resolve()}")
    print("Train metrics:", train_metrics)
    if test:
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

