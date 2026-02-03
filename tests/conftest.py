"""Pytest configuration and fixtures for trading bot tests."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variables before importing modules
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_API_SECRET", "test_secret")
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

from tradebot.config import Settings, reload_settings
from tradebot.core.events import BaseEvent, SentimentResult
from tradebot.data.store import SQLiteStore
from tradebot.risk.risk_manager import RiskManager


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def store(temp_db: Path) -> SQLiteStore:
    """Create a SQLiteStore with a temporary database."""
    s = SQLiteStore(path=temp_db)
    s.init()
    return s


@pytest.fixture
def risk_manager() -> RiskManager:
    """Create a RiskManager with default settings."""
    return RiskManager(
        max_daily_loss_usd=250.0,
        max_position_value_usd=2000.0,
        risk_per_trade_pct=0.01,
        stop_loss_pct=0.005,
        take_profit_pct=0.01,
    )


@pytest.fixture
def sample_event() -> BaseEvent:
    """Create a sample BaseEvent for testing."""
    return BaseEvent(
        type="news",
        source="test_source",
        created_at=datetime.now(timezone.utc),
        text="Apple beats earnings expectations with record revenue",
        url="https://example.com/news/1",
        author="Test Author",
    )


@pytest.fixture
def sample_events() -> list[BaseEvent]:
    """Create a list of sample events for testing."""
    now = datetime.now(timezone.utc)
    return [
        BaseEvent(
            type="news",
            source="rss",
            created_at=now,
            text="Stock market rallies on positive economic data",
            url="https://example.com/1",
            author="Reporter A",
        ),
        BaseEvent(
            type="social",
            source="x/@trader",
            created_at=now,
            text="$SPY looking bullish today, expecting breakout",
            url="https://twitter.com/trader/status/123",
            author="trader",
        ),
        BaseEvent(
            type="news",
            source="rss",
            created_at=now,
            text="Fed announces rate hike, markets crash",
            url="https://example.com/2",
            author="Reporter B",
        ),
    ]


@pytest.fixture
def mock_alpaca_client():
    """Create a mock Alpaca trading client."""
    mock = MagicMock()

    # Mock account
    mock_account = MagicMock()
    mock_account.equity = "100000.00"
    mock_account.buying_power = "50000.00"
    mock_account.status = "ACTIVE"
    mock.get_account.return_value = mock_account

    # Mock positions
    mock.get_all_positions.return_value = []

    # Mock clock
    mock_clock = MagicMock()
    mock_clock.is_open = True
    mock.get_clock.return_value = mock_clock

    return mock


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with minimal configuration."""
    # Force reload to pick up test env vars
    return reload_settings()


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for API testing."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {}}
    mock.get.return_value = mock_response
    mock.post.return_value = mock_response
    return mock
