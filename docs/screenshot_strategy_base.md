# Screenshot Strategy Base Pipeline

This boilerplate lets you train a baseline outcome model from labeled screenshot trades.

## What is included

- Data schema + loader:
  - `src/tradebot/learning/screenshot_strategy.py`
  - `load_screenshot_examples(...)`
  - `ScreenshotTradeExample`
- Baseline model:
  - `ScreenshotOutcomeModel`
  - Predicts `expected_return`, `win_probability`, `confidence`
  - Supports `should_trade(...)` gating
- Train CLI:
  - `scripts/screenshot_strategy_train.py`
- Data template:
  - `examples/screenshot_trades_template.json`

## Data format

Each row should include:

- `trade_id` (string)
- `screenshot_path` (string; absolute or relative to data file)
- `symbol`, `timeframe`, `side`
- `entry_price`, `exit_price`

Recommended fields:

- `entry_time`, `exit_time`
- `entry_bar_index`, `exit_bar_index`
- `pattern`
- `tags` (list)
- `rationale` (plain-English setup logic)
- `manual_features` (dictionary of numeric features)

## Train baseline model

```bash
PYTHONPATH=src .venv/bin/python scripts/screenshot_strategy_train.py \
  --data examples/screenshot_trades_template.json \
  --model-out artifacts/screenshot_strategy_model.json
```

Optional:

- `--strict-paths` to enforce screenshot files exist.
- `--test-ratio` to control holdout split.

## What to provide next

To improve effectiveness, provide:

- both winning and losing trades
- non-trade examples (setups you skipped)
- consistent rationale tags
- enough samples across sessions/market regimes

The current model is intentionally simple and serves as the first stage before adding vision embeddings and execution rules.

