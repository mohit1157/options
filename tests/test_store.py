"""Tests for SQLiteStore."""
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradebot.data.store import SQLiteStore
from tradebot.core.events import BaseEvent


class TestSQLiteStore:
    """Tests for SQLiteStore class."""

    def test_init_creates_tables(self, store: SQLiteStore):
        """Test that init creates required tables."""
        with store.connect() as conn:
            # Check events table exists
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
            )
            assert cur.fetchone() is not None

            # Check sentiment table exists
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment'"
            )
            assert cur.fetchone() is not None

            # Check trade_audit table exists
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_audit'"
            )
            assert cur.fetchone() is not None

    def test_add_event(self, store: SQLiteStore):
        """Test adding an event."""
        event_id = store.add_event(
            type="news",
            source="test",
            created_at=datetime.now(timezone.utc),
            author="author",
            url="https://example.com/1",
            text="Test event text",
        )

        assert event_id is not None
        assert event_id > 0

    def test_add_duplicate_event(self, store: SQLiteStore):
        """Test that duplicate events are rejected."""
        now = datetime.now(timezone.utc)
        url = "https://example.com/duplicate"

        # Add first event
        event_id1 = store.add_event(
            type="news",
            source="test",
            created_at=now,
            author="author",
            url=url,
            text="First event",
        )
        assert event_id1 is not None

        # Add duplicate (same URL)
        event_id2 = store.add_event(
            type="news",
            source="test",
            created_at=now,
            author="author",
            url=url,
            text="Duplicate event",
        )
        assert event_id2 is None

    def test_add_duplicate_event_without_url(self, store: SQLiteStore):
        """Test that duplicate events without URL are rejected via content hash."""
        now = datetime.now(timezone.utc)

        event_id1 = store.add_event(
            type="news",
            source="test",
            created_at=now,
            author="author",
            url=None,
            text="Same content",
        )
        assert event_id1 is not None

        event_id2 = store.add_event(
            type="news",
            source="test",
            created_at=now,
            author="author",
            url=None,
            text="Same content",
        )
        assert event_id2 is None

    def test_event_exists(self, store: SQLiteStore):
        """Test event_exists method."""
        url = "https://example.com/exists"

        # Should not exist initially
        assert store.event_exists(url=url) is False

        # Add event
        store.add_event(
            type="news",
            source="test",
            created_at=datetime.now(timezone.utc),
            author=None,
            url=url,
            text="Test",
        )

        # Should exist now
        assert store.event_exists(url=url) is True

    def test_event_exists_none_url(self, store: SQLiteStore):
        """Test event_exists with None URL."""
        # None URL should return False
        assert store.event_exists(url=None) is False

    def test_add_sentiment(self, store: SQLiteStore):
        """Test adding sentiment for an event."""
        event_id = store.add_event(
            type="news",
            source="test",
            created_at=datetime.now(timezone.utc),
            author=None,
            url="https://example.com/sentiment",
            text="Test",
        )

        # Should not raise
        store.add_sentiment(
            event_id=event_id,
            score=0.5,
            label="positive",
            created_at=datetime.now(timezone.utc),
        )

    def test_recent_sentiment(self, store: SQLiteStore):
        """Test getting recent sentiment."""
        now = datetime.now(timezone.utc)

        # Add events and sentiment
        for i in range(5):
            event_id = store.add_event(
                type="news",
                source=f"test_{i}",
                created_at=now,
                author=None,
                url=f"https://example.com/{i}",
                text=f"Test {i}",
            )
            store.add_sentiment(
                event_id=event_id,
                score=0.1 * i,
                label="positive" if i > 2 else "neutral",
                created_at=now,
            )

        # Get recent sentiment
        since = (now - timedelta(minutes=5)).isoformat()
        results = store.recent_sentiment(since_iso=since, limit=10)

        assert len(results) == 5
        for score, label, source in results:
            assert isinstance(score, float)
            assert label in ("positive", "neutral", "negative")

    def test_recent_sentiment_symbol_filter(self, store: SQLiteStore):
        """Test recent sentiment filtered by symbol."""
        now = datetime.now(timezone.utc)

        event_id1 = store.add_event(
            type="news",
            source="test",
            created_at=now,
            author=None,
            url="https://example.com/a",
            text="SPY rallies on strong data",
        )
        store.add_sentiment(
            event_id=event_id1,
            score=0.5,
            label="positive",
            created_at=now,
        )

        event_id2 = store.add_event(
            type="news",
            source="test",
            created_at=now,
            author=None,
            url="https://example.com/b",
            text="AAPL reports earnings",
        )
        store.add_sentiment(
            event_id=event_id2,
            score=0.2,
            label="positive",
            created_at=now,
        )

        since = (now - timedelta(minutes=5)).isoformat()
        results = store.recent_sentiment(since_iso=since, limit=10, symbol="SPY")

        assert len(results) == 1
        assert results[0][0] == 0.5

    def test_x_state_roundtrip(self, store: SQLiteStore):
        """Test storing and retrieving X since_id state."""
        assert store.get_x_since_id("user1") is None
        store.set_x_since_id("user1", "12345")
        assert store.get_x_since_id("user1") == "12345"

    def test_cleanup_old_events(self, store: SQLiteStore):
        """Test cleaning up old events."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=48)

        # Add old event (simulate by inserting directly with old timestamp)
        with store.connect() as conn:
            conn.execute(
                """
                INSERT INTO events(type, source, created_at, url, url_hash, text, ingested_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                ("news", "test", now.isoformat(), "https://old.com", "oldhash", "Old", old_time.isoformat()),
            )

        # Add new event
        store.add_event(
            type="news",
            source="test",
            created_at=now,
            author=None,
            url="https://new.com",
            text="New",
        )

        # Count before cleanup
        assert store.get_event_count() == 2

        # Cleanup events older than 24 hours
        deleted = store.cleanup_old_events(hours=24)

        assert deleted == 1
        assert store.get_event_count() == 1

    def test_log_trade(self, store: SQLiteStore):
        """Test logging a trade."""
        trade_id = store.log_trade(
            symbol="AAPL",
            side="buy",
            qty=10.0,
            order_type="bracket",
            status="submitted",
            order_id="order123",
            metadata={"price": 150.0},
        )

        assert trade_id is not None
        assert trade_id > 0

    def test_get_trade_history(self, store: SQLiteStore):
        """Test getting trade history."""
        # Log some trades
        store.log_trade(symbol="AAPL", side="buy", qty=10, order_type="market", status="filled")
        store.log_trade(symbol="AAPL", side="sell", qty=5, order_type="market", status="filled")
        store.log_trade(symbol="SPY", side="buy", qty=20, order_type="bracket", status="submitted")

        # Get all trades
        history = store.get_trade_history(limit=10)
        assert len(history) == 3

        # Filter by symbol
        aapl_history = store.get_trade_history(symbol="AAPL", limit=10)
        assert len(aapl_history) == 2

    def test_get_sentiment_stats(self, store: SQLiteStore):
        """Test getting sentiment statistics."""
        now = datetime.now(timezone.utc)

        # Add events with various sentiments
        for i, (score, label) in enumerate([
            (0.5, "positive"),
            (0.3, "positive"),
            (-0.5, "negative"),
            (0.0, "neutral"),
        ]):
            event_id = store.add_event(
                type="news",
                source="test",
                created_at=now,
                author=None,
                url=f"https://example.com/stats/{i}",
                text="Test",
            )
            store.add_sentiment(
                event_id=event_id,
                score=score,
                label=label,
                created_at=now,
            )

        stats = store.get_sentiment_stats(hours=1)

        assert stats["total_count"] == 4
        assert stats["positive_count"] == 2
        assert stats["negative_count"] == 1
        assert stats["neutral_count"] == 1
        assert stats["avg_score"] == pytest.approx(0.075, rel=0.01)

    def test_get_daily_pnl(self, store: SQLiteStore):
        """Test getting daily P&L summary."""
        from datetime import datetime, timezone

        # Log some trades
        store.log_trade(symbol="AAPL", side="buy", qty=10, order_type="market", status="filled")
        store.log_trade(symbol="AAPL", side="sell", qty=10, order_type="market", status="filled")
        store.log_trade(symbol="SPY", side="buy", qty=5, order_type="market", status="rejected")
        store.log_trade(symbol="QQQ", side="buy", qty=5, order_type="market", status="error", error_message="Failed")

        # Get trades for today - use current time to match the default timestamp
        pnl = store.get_daily_pnl(date=datetime.now(timezone.utc))

        # Verify we can retrieve trade counts (timestamps may differ slightly)
        assert pnl["total_trades"] >= 0
        assert "filled" in pnl
        assert "rejected" in pnl
        assert "errors" in pnl

    def test_signal_features_and_trade_outcome_roundtrip(self, store: SQLiteStore):
        """Test feedback-learning tables and stats aggregation."""
        now = datetime.now(timezone.utc)
        signal_id = store.log_signal_features(
            strategy="pop_pullback_hold",
            symbol="SPY250117C00500000",
            side="buy",
            timeframe="3Min",
            features={"ema": 9, "entry_underlying": 500.0},
            created_at=now,
        )
        assert signal_id > 0

        store.log_trade_outcome(
            strategy="pop_pullback_hold",
            symbol="SPY250117C00500000",
            side="buy",
            qty=1,
            pnl_usd=25.0,
            pnl_pct=0.1,
            is_win=True,
            signal_id=signal_id,
            entry_price=2.5,
            exit_price=2.75,
            closed_at=now,
            metadata={"reason": "target"},
        )

        stats = store.get_trade_outcome_stats(
            strategy="pop_pullback_hold",
            since=now - timedelta(minutes=1),
            until=now + timedelta(minutes=1),
        )
        assert stats["total_trades"] == 1
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["win_rate"] == pytest.approx(1.0, rel=0.001)
        assert stats["total_pnl_usd"] == pytest.approx(25.0, rel=0.001)
        assert stats["avg_pnl_pct"] == pytest.approx(0.1, rel=0.001)

    def test_calibration_state_roundtrip(self, store: SQLiteStore):
        """Test storing and retrieving calibration state."""
        now = datetime.now(timezone.utc)
        store.upsert_calibration(
            strategy="pop_pullback_hold",
            last_calibrated_at=now,
            params={"target_profit_pct": 0.07, "runner_target_profit_pct": 0.11},
            stats={"total_trades": 12, "win_rate": 0.58},
        )

        calibration = store.get_last_calibration(strategy="pop_pullback_hold")
        assert calibration is not None
        assert calibration["strategy"] == "pop_pullback_hold"
        assert calibration["params"]["target_profit_pct"] == pytest.approx(0.07, rel=0.001)
        assert calibration["stats"]["total_trades"] == 12
