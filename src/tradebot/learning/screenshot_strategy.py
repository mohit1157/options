from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

Side = Literal["long", "short", "call", "put"]

_SIDE_ALIASES = {
    "buy": "long",
    "sell": "short",
    "bull": "long",
    "bear": "short",
}
_VALID_SIDES = {"long", "short", "call", "put"}


def _normalize_side(value: str) -> Side:
    side = (value or "").strip().lower()
    side = _SIDE_ALIASES.get(side, side)
    if side not in _VALID_SIDES:
        raise ValueError(f"Unsupported side '{value}'. Expected one of {_VALID_SIDES}.")
    return side  # type: ignore[return-value]


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"Invalid datetime '{value}'") from e
    raise ValueError(f"Unsupported datetime value: {value!r}")


def _tokenize(text: str, *, limit: int = 12) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    out: list[str] = []
    for token in tokens:
        if len(token) < 3 or token.isdigit():
            continue
        if token in out:
            continue
        out.append(token)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True)
class ScreenshotTradeExample:
    """Single labeled trade example linked to a chart screenshot."""

    trade_id: str
    screenshot_path: str
    symbol: str
    timeframe: str
    side: Side
    entry_price: float
    exit_price: float
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_bar_index: Optional[int] = None
    exit_bar_index: Optional[int] = None
    rationale: str = ""
    pattern: str = ""
    tags: list[str] = field(default_factory=list)
    manual_features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        *,
        base_dir: Path,
        strict_paths: bool = False,
    ) -> "ScreenshotTradeExample":
        trade_id = str(record.get("trade_id") or record.get("id") or "").strip()
        if not trade_id:
            raise ValueError("trade_id is required")

        screenshot_raw = str(record.get("screenshot_path") or record.get("image_path") or "").strip()
        if not screenshot_raw:
            raise ValueError(f"{trade_id}: screenshot_path is required")

        screenshot_path = Path(screenshot_raw)
        if not screenshot_path.is_absolute():
            screenshot_path = (base_dir / screenshot_path).resolve()

        if strict_paths and not screenshot_path.exists():
            raise FileNotFoundError(f"{trade_id}: screenshot not found: {screenshot_path}")

        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError(f"{trade_id}: symbol is required")

        timeframe = str(record.get("timeframe") or "3Min").strip()
        side = _normalize_side(str(record.get("side") or "long"))

        entry_price = float(record.get("entry_price"))
        exit_price = float(record.get("exit_price"))

        tags_raw = record.get("tags") or []
        if isinstance(tags_raw, str):
            tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
        else:
            tags = [str(t).strip().lower() for t in tags_raw if str(t).strip()]

        features_raw = record.get("manual_features") or {}
        manual_features: dict[str, float] = {}
        if isinstance(features_raw, dict):
            for key, value in features_raw.items():
                try:
                    manual_features[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        return cls(
            trade_id=trade_id,
            screenshot_path=str(screenshot_path),
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=_parse_dt(record.get("entry_time")),
            exit_time=_parse_dt(record.get("exit_time")),
            entry_bar_index=record.get("entry_bar_index"),
            exit_bar_index=record.get("exit_bar_index"),
            rationale=str(record.get("rationale") or record.get("logic") or "").strip(),
            pattern=str(record.get("pattern") or "").strip().lower(),
            tags=tags,
            manual_features=manual_features,
            metadata=dict(record.get("metadata") or {}),
        )

    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side in ("long", "call"):
            return (self.exit_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.exit_price) / self.entry_price

    def is_win(self) -> bool:
        return self.pnl_pct() > 0

    def hold_bars(self) -> Optional[int]:
        if self.entry_bar_index is None or self.exit_bar_index is None:
            return None
        return self.exit_bar_index - self.entry_bar_index

    def to_record(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "screenshot_path": self.screenshot_path,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_bar_index": self.entry_bar_index,
            "exit_bar_index": self.exit_bar_index,
            "rationale": self.rationale,
            "pattern": self.pattern,
            "tags": list(self.tags),
            "manual_features": dict(self.manual_features),
            "metadata": dict(self.metadata),
        }


def load_screenshot_examples(
    path: str | Path,
    *,
    strict_paths: bool = False,
) -> list[ScreenshotTradeExample]:
    """Load trade examples from .json or .jsonl."""
    data_path = Path(path)
    base_dir = data_path.parent.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    rows: list[dict[str, Any]] = []
    suffix = data_path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        for line in data_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    else:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            maybe_rows = payload.get("examples") or payload.get("trades")
            if not isinstance(maybe_rows, list):
                raise ValueError("JSON must be a list or contain 'examples'/'trades' list.")
            rows = maybe_rows
        else:
            raise ValueError("Unsupported dataset structure.")

    examples: list[ScreenshotTradeExample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each row must be an object.")
        examples.append(
            ScreenshotTradeExample.from_record(
                row,
                base_dir=base_dir,
                strict_paths=strict_paths,
            )
        )
    return examples


@dataclass
class _RunningStats:
    count: int = 0
    wins: int = 0
    sum_return: float = 0.0

    @property
    def mean_return(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_return / self.count

    @property
    def win_rate(self) -> float:
        if self.count == 0:
            return 0.5
        return self.wins / self.count

    def update(self, pnl_pct: float) -> None:
        self.count += 1
        self.sum_return += pnl_pct
        if pnl_pct > 0:
            self.wins += 1

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "wins": self.wins, "sum_return": self.sum_return}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_RunningStats":
        return cls(
            count=int(data.get("count", 0)),
            wins=int(data.get("wins", 0)),
            sum_return=float(data.get("sum_return", 0.0)),
        )


@dataclass(frozen=True)
class Prediction:
    expected_return: float
    win_probability: float
    confidence: float
    evidence_count: int
    contributors: list[str] = field(default_factory=list)


@dataclass
class ScreenshotOutcomeModel:
    """Baseline data-driven model for screenshot strategy outcomes.

    This model is intentionally simple:
    - learns empirical return/win-rate statistics by context, pattern, tags, and rationale terms
    - predicts expected return + win probability for new examples
    - can gate trades using confidence and expected return thresholds
    """

    smoothing: float = 5.0
    model_version: str = "0.1"
    trained_at: Optional[str] = None
    global_stats: _RunningStats = field(default_factory=_RunningStats)
    by_context: dict[str, _RunningStats] = field(default_factory=dict)
    by_pattern: dict[str, _RunningStats] = field(default_factory=dict)
    by_tag: dict[str, _RunningStats] = field(default_factory=dict)
    by_logic_term: dict[str, _RunningStats] = field(default_factory=dict)

    def _bucket_key(self, example: ScreenshotTradeExample) -> str:
        return f"{example.symbol}|{example.timeframe}|{example.side}"

    def _update_bucket(self, bucket: dict[str, _RunningStats], key: str, pnl_pct: float) -> None:
        if not key:
            return
        if key not in bucket:
            bucket[key] = _RunningStats()
        bucket[key].update(pnl_pct)

    def fit(self, examples: list[ScreenshotTradeExample]) -> "ScreenshotOutcomeModel":
        self.global_stats = _RunningStats()
        self.by_context = {}
        self.by_pattern = {}
        self.by_tag = {}
        self.by_logic_term = {}

        for ex in examples:
            pnl = ex.pnl_pct()
            self.global_stats.update(pnl)
            self._update_bucket(self.by_context, self._bucket_key(ex), pnl)
            self._update_bucket(self.by_pattern, ex.pattern, pnl)

            for tag in ex.tags:
                self._update_bucket(self.by_tag, tag, pnl)

            for token in _tokenize(ex.rationale):
                self._update_bucket(self.by_logic_term, token, pnl)

        self.trained_at = datetime.now(timezone.utc).isoformat()
        return self

    def _smoothed_mean(self, stats: _RunningStats) -> float:
        global_mean = self.global_stats.mean_return
        denom = stats.count + self.smoothing
        if denom <= 0:
            return global_mean
        return (stats.sum_return + self.smoothing * global_mean) / denom

    def _smoothed_win_rate(self, stats: _RunningStats) -> float:
        global_wr = self.global_stats.win_rate
        denom = stats.count + self.smoothing
        if denom <= 0:
            return global_wr
        return (stats.wins + self.smoothing * global_wr) / denom

    def predict(self, example: ScreenshotTradeExample) -> Prediction:
        global_stat = self.global_stats
        components: list[tuple[str, _RunningStats, float]] = [("global", global_stat, 1.0)]

        context_key = self._bucket_key(example)
        if context_key in self.by_context:
            components.append(("context", self.by_context[context_key], 2.0))

        if example.pattern and example.pattern in self.by_pattern:
            components.append(("pattern", self.by_pattern[example.pattern], 1.5))

        for tag in example.tags:
            stat = self.by_tag.get(tag)
            if stat:
                components.append((f"tag:{tag}", stat, 0.7))

        for token in _tokenize(example.rationale, limit=8):
            stat = self.by_logic_term.get(token)
            if stat:
                components.append((f"logic:{token}", stat, 0.35))

        weighted_return = 0.0
        weighted_win_prob = 0.0
        total_weight = 0.0
        evidence_count = 0
        contributors: list[str] = []

        for name, stat, weight in components:
            mean_ret = self._smoothed_mean(stat)
            win_rate = self._smoothed_win_rate(stat)
            weighted_return += mean_ret * weight
            weighted_win_prob += win_rate * weight
            total_weight += weight
            evidence_count += stat.count
            contributors.append(f"{name}(n={stat.count})")

        if total_weight <= 0:
            return Prediction(
                expected_return=0.0,
                win_probability=0.5,
                confidence=0.0,
                evidence_count=0,
                contributors=[],
            )

        expected_return = weighted_return / total_weight
        win_probability = weighted_win_prob / total_weight
        confidence = min(1.0, evidence_count / 50.0)

        return Prediction(
            expected_return=expected_return,
            win_probability=win_probability,
            confidence=confidence,
            evidence_count=evidence_count,
            contributors=contributors,
        )

    def should_trade(
        self,
        example: ScreenshotTradeExample,
        *,
        min_expected_return: float = 0.002,
        min_win_probability: float = 0.53,
        min_confidence: float = 0.2,
    ) -> bool:
        prediction = self.predict(example)
        return (
            prediction.expected_return >= min_expected_return
            and prediction.win_probability >= min_win_probability
            and prediction.confidence >= min_confidence
        )

    def evaluate(self, examples: list[ScreenshotTradeExample]) -> dict[str, float]:
        if not examples:
            return {
                "count": 0.0,
                "mae_expected_return": 0.0,
                "directional_accuracy": 0.0,
                "brier_win_probability": 0.0,
                "avg_actual_return": 0.0,
                "avg_predicted_return": 0.0,
            }

        abs_error = 0.0
        correct_direction = 0
        brier = 0.0
        sum_actual = 0.0
        sum_predicted = 0.0

        for ex in examples:
            pred = self.predict(ex)
            actual_ret = ex.pnl_pct()
            actual_win = 1.0 if actual_ret > 0 else 0.0

            abs_error += abs(pred.expected_return - actual_ret)
            if (pred.expected_return >= 0) == (actual_ret >= 0):
                correct_direction += 1
            brier += (pred.win_probability - actual_win) ** 2
            sum_actual += actual_ret
            sum_predicted += pred.expected_return

        n = float(len(examples))
        return {
            "count": n,
            "mae_expected_return": abs_error / n,
            "directional_accuracy": correct_direction / n,
            "brier_win_probability": brier / n,
            "avg_actual_return": sum_actual / n,
            "avg_predicted_return": sum_predicted / n,
        }

    def to_dict(self) -> dict[str, Any]:
        def serialize_bucket(bucket: dict[str, _RunningStats]) -> dict[str, Any]:
            return {k: v.to_dict() for k, v in bucket.items()}

        return {
            "model_version": self.model_version,
            "smoothing": self.smoothing,
            "trained_at": self.trained_at,
            "global_stats": self.global_stats.to_dict(),
            "by_context": serialize_bucket(self.by_context),
            "by_pattern": serialize_bucket(self.by_pattern),
            "by_tag": serialize_bucket(self.by_tag),
            "by_logic_term": serialize_bucket(self.by_logic_term),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreenshotOutcomeModel":
        model = cls(
            smoothing=float(data.get("smoothing", 5.0)),
            model_version=str(data.get("model_version", "0.1")),
            trained_at=data.get("trained_at"),
        )
        model.global_stats = _RunningStats.from_dict(dict(data.get("global_stats") or {}))

        for key, value in dict(data.get("by_context") or {}).items():
            model.by_context[key] = _RunningStats.from_dict(dict(value))
        for key, value in dict(data.get("by_pattern") or {}).items():
            model.by_pattern[key] = _RunningStats.from_dict(dict(value))
        for key, value in dict(data.get("by_tag") or {}).items():
            model.by_tag[key] = _RunningStats.from_dict(dict(value))
        for key, value in dict(data.get("by_logic_term") or {}).items():
            model.by_logic_term[key] = _RunningStats.from_dict(dict(value))
        return model

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScreenshotOutcomeModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

