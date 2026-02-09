from __future__ import annotations

import hashlib
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from tradebot.core.logger import get_logger

log = get_logger("store")


@dataclass
class SQLiteStore:
    """SQLite storage for events and sentiment analysis.

    Features:
    - Event deduplication via URL hash
    - Sentiment tracking per event
    - TTL-based cleanup for old events
    - Audit logging for trades

    Usage:
        store = SQLiteStore()
        store.init()

        # Add event (returns None if duplicate)
        event_id = store.add_event(...)

        # Check for duplicates
        if store.event_exists(url="https://..."):
            print("Already processed")
    """

    path: Path = Path("tradebot.sqlite3")

    def connect(self) -> sqlite3.Connection:
        """Create a database connection."""
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init(self) -> None:
        """Initialize database schema."""
        with self.connect() as conn:
            # Events table with unique constraint on url_hash
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    author TEXT,
                    url TEXT,
                    url_hash TEXT UNIQUE,
                    content_hash TEXT,
                    text TEXT NOT NULL,
                    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Index for faster lookups
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_url_hash ON events(url_hash);
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_content_hash
                ON events(content_hash) WHERE content_hash IS NOT NULL;
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_ingested_at ON events(ingested_at);
                """
            )

            # Sentiment table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    label TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sentiment_created_at ON sentiment(created_at);
                """
            )

            # Trade audit table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    order_id TEXT,
                    fill_price REAL,
                    error_message TEXT,
                    metadata TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_audit_timestamp ON trade_audit(timestamp);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_audit_symbol ON trade_audit(symbol);
                """
            )

            # Signal features for feedback learning
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    timeframe TEXT,
                    features TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_signal_features_strategy_time
                ON signal_features(strategy, created_at);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_signal_features_symbol
                ON signal_features(symbol);
                """
            )

            # Realized trade outcomes linked to originating signals
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    closed_at TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    signal_id INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    qty REAL NOT NULL,
                    pnl_usd REAL NOT NULL,
                    pnl_pct REAL,
                    is_win INTEGER NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(signal_id) REFERENCES signal_features(id) ON DELETE SET NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_strategy_time
                ON trade_outcomes(strategy, closed_at);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_symbol
                ON trade_outcomes(symbol);
                """
            )

            # Last strategy calibration snapshot and stats
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_calibration (
                    strategy TEXT PRIMARY KEY,
                    last_calibrated_at TEXT NOT NULL,
                    params TEXT,
                    stats TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # X ingestion state table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS x_state (
                    username TEXT PRIMARY KEY,
                    since_id TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            log.debug("Database schema initialized")

            # Handle schema upgrades for existing DBs
            self._ensure_column(conn, "events", "content_hash", "TEXT")

    def _compute_url_hash(self, url: Optional[str]) -> Optional[str]:
        """Compute a hash for URL deduplication."""
        if not url:
            return None
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _compute_content_hash(
        self,
        *,
        source: str,
        author: Optional[str],
        text: str,
    ) -> Optional[str]:
        """Compute a content hash for deduplication when URL is missing."""
        if not text:
            return None
        payload = f"{source}|{author or ''}|{text[:1000]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
        """Ensure a column exists; add it if missing (for migrations)."""
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def event_exists(
        self,
        url: Optional[str] = None,
        url_hash: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> bool:
        """Check if an event with the given URL already exists.

        Args:
            url: Event URL to check
            url_hash: Pre-computed URL hash (optional)
            content_hash: Pre-computed content hash (optional)

        Returns:
            True if event exists
        """
        if url is None and url_hash is None and content_hash is None:
            return False

        if url_hash is None:
            url_hash = self._compute_url_hash(url)

        if content_hash is None and url_hash is None:
            return False

        with self.connect() as conn:
            if url_hash is not None:
                cur = conn.execute(
                    "SELECT 1 FROM events WHERE url_hash = ? LIMIT 1",
                    (url_hash,),
                )
                if cur.fetchone() is not None:
                    return True
            if content_hash is not None:
                cur = conn.execute(
                    "SELECT 1 FROM events WHERE content_hash = ? LIMIT 1",
                    (content_hash,),
                )
                return cur.fetchone() is not None
            return False

    def add_event(
        self,
        *,
        type: str,
        source: str,
        created_at: datetime,
        author: Optional[str],
        url: Optional[str],
        text: str,
    ) -> Optional[int]:
        """Add an event to the store.

        Returns:
            Event ID if added, None if duplicate
        """
        url_hash = self._compute_url_hash(url)
        content_hash = self._compute_content_hash(
            source=source,
            author=author,
            text=text,
        )

        # Check for duplicate
        if (url_hash and self.event_exists(url_hash=url_hash)) or (
            content_hash and self.event_exists(content_hash=content_hash)
        ):
            log.debug(f"Duplicate event skipped: {url}")
            return None

        with self.connect() as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO events(type, source, created_at, author, url, url_hash, content_hash, text)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        type,
                        source,
                        created_at.isoformat() if created_at else datetime.now(timezone.utc).isoformat(),
                        author,
                        url,
                        url_hash,
                        content_hash,
                        text,
                    ),
                )
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                # Race condition - event was added by another process
                log.debug(f"Duplicate event (integrity): {url}")
                return None

    def add_sentiment(
        self,
        *,
        event_id: int,
        score: float,
        label: str,
        created_at: datetime,
    ) -> None:
        """Add sentiment analysis for an event."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sentiment(event_id, score, label, created_at) VALUES(?, ?, ?, ?)",
                (event_id, score, label, created_at.isoformat()),
            )

    def recent_sentiment(
        self,
        *,
        since_iso: str,
        limit: int = 50,
        symbol: Optional[str] = None,
    ) -> list[Tuple[float, str, str]]:
        """Get recent sentiment scores.

        Returns:
            List of (score, label, source) tuples
        """
        with self.connect() as conn:
            if symbol:
                sym = symbol.upper()
                cur = conn.execute(
                    """
                    SELECT s.score, s.label, e.source
                    FROM sentiment s
                    JOIN events e ON e.id = s.event_id
                    WHERE s.created_at >= ?
                      AND (
                        UPPER(e.text) LIKE ?
                        OR UPPER(e.text) LIKE ?
                      )
                    ORDER BY s.created_at DESC
                    LIMIT ?
                    """,
                    (since_iso, f"%{sym}%", f"%${sym}%", limit),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT s.score, s.label, e.source
                    FROM sentiment s
                    JOIN events e ON e.id = s.event_id
                    WHERE s.created_at >= ?
                    ORDER BY s.created_at DESC
                    LIMIT ?
                    """,
                    (since_iso, limit),
                )
            return list(cur.fetchall())

    def cleanup_old_events(self, hours: int = 24) -> int:
        """Delete events older than specified hours.

        Args:
            hours: Maximum age in hours

        Returns:
            Number of deleted events
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM events WHERE ingested_at < ?",
                (cutoff,),
            )
            deleted = cur.rowcount
            if deleted > 0:
                log.info(f"Cleaned up {deleted} events older than {hours}h")
            return deleted

    def get_event_count(self) -> int:
        """Get total event count."""
        with self.connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM events")
            return cur.fetchone()[0]

    def get_sentiment_stats(self, hours: int = 24) -> dict:
        """Get sentiment statistics for the past N hours.

        Returns:
            Dict with avg_score, positive_count, negative_count, neutral_count
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    AVG(score) as avg_score,
                    SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN label = 'neutral' THEN 1 ELSE 0 END) as neutral,
                    COUNT(*) as total
                FROM sentiment
                WHERE created_at >= ?
                """,
                (since,),
            )
            row = cur.fetchone()
            return {
                "avg_score": row[0] or 0.0,
                "positive_count": row[1] or 0,
                "negative_count": row[2] or 0,
                "neutral_count": row[3] or 0,
                "total_count": row[4] or 0,
            }

    # Feedback learning methods
    def log_signal_features(
        self,
        *,
        strategy: str,
        symbol: str,
        side: str,
        timeframe: Optional[str],
        features: dict,
        created_at: Optional[datetime] = None,
    ) -> int:
        """Persist input features for a strategy signal."""
        timestamp = (created_at or datetime.now(timezone.utc)).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO signal_features(created_at, strategy, symbol, side, timeframe, features)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    strategy,
                    symbol,
                    side,
                    timeframe,
                    json.dumps(features or {}),
                ),
            )
            return int(cur.lastrowid)

    def log_trade_outcome(
        self,
        *,
        strategy: str,
        symbol: str,
        side: str,
        qty: float,
        pnl_usd: float,
        is_win: bool,
        signal_id: Optional[int] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        metadata: Optional[dict] = None,
        closed_at: Optional[datetime] = None,
    ) -> int:
        """Persist a realized trade outcome for calibration and analytics."""
        timestamp = (closed_at or datetime.now(timezone.utc)).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_outcomes(
                    closed_at,
                    strategy,
                    symbol,
                    side,
                    signal_id,
                    entry_price,
                    exit_price,
                    qty,
                    pnl_usd,
                    pnl_pct,
                    is_win,
                    metadata
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    strategy,
                    symbol,
                    side,
                    signal_id,
                    entry_price,
                    exit_price,
                    qty,
                    pnl_usd,
                    pnl_pct,
                    1 if is_win else 0,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return int(cur.lastrowid)

    def get_trade_outcome_stats(
        self,
        *,
        strategy: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Aggregate win-rate and PnL stats from realized trade outcomes."""
        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_usd) as avg_pnl_usd,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(pnl_usd) as total_pnl_usd
            FROM trade_outcomes
            WHERE 1=1
        """
        params: list[Any] = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if since:
            query += " AND closed_at >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND closed_at < ?"
            params.append(until.isoformat())

        with self.connect() as conn:
            cur = conn.execute(query, params)
            row = cur.fetchone()

        total = int(row[0] or 0)
        wins = int(row[1] or 0)
        losses = max(0, total - wins)
        win_rate = float(wins / total) if total > 0 else 0.0

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_pnl_usd": float(row[2] or 0.0),
            "avg_pnl_pct": float(row[3] or 0.0),
            "total_pnl_usd": float(row[4] or 0.0),
        }

    def get_last_calibration(self, *, strategy: str) -> Optional[dict[str, Any]]:
        """Get the most recent calibration snapshot for a strategy."""
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT last_calibrated_at, params, stats
                FROM strategy_calibration
                WHERE strategy = ?
                """,
                (strategy,),
            )
            row = cur.fetchone()

        if not row:
            return None

        try:
            last_calibrated_at = datetime.fromisoformat(row[0])
        except Exception:
            last_calibrated_at = datetime.now(timezone.utc)

        return {
            "strategy": strategy,
            "last_calibrated_at": last_calibrated_at,
            "params": json.loads(row[1]) if row[1] else {},
            "stats": json.loads(row[2]) if row[2] else {},
        }

    def upsert_calibration(
        self,
        *,
        strategy: str,
        last_calibrated_at: datetime,
        params: Optional[dict] = None,
        stats: Optional[dict] = None,
    ) -> None:
        """Upsert strategy calibration state."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO strategy_calibration(strategy, last_calibrated_at, params, stats, updated_at)
                VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(strategy) DO UPDATE SET
                    last_calibrated_at=excluded.last_calibrated_at,
                    params=excluded.params,
                    stats=excluded.stats,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    strategy,
                    last_calibrated_at.isoformat(),
                    json.dumps(params or {}),
                    json.dumps(stats or {}),
                ),
            )

    # Trade audit methods
    def log_trade(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        status: str,
        order_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Log a trade attempt for auditing.

        Args:
            symbol: Trading symbol
            side: buy/sell
            qty: Order quantity
            order_type: market/limit/bracket
            status: submitted/filled/rejected/error
            order_id: Exchange order ID
            fill_price: Fill price if filled
            error_message: Error message if failed
            metadata: Additional metadata as dict

        Returns:
            Audit log ID
        """
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_audit(symbol, side, qty, order_type, status, order_id, fill_price, error_message, metadata)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    qty,
                    order_type,
                    status,
                    order_id,
                    fill_price,
                    error_message,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return int(cur.lastrowid)

    def get_trade_history(
        self,
        *,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get trade audit history.

        Returns:
            List of trade audit records
        """
        query = "SELECT * FROM trade_audit WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(query, params)
            rows = cur.fetchall()

            result = []
            for row in rows:
                record = dict(row)
                if record.get("metadata"):
                    record["metadata"] = json.loads(record["metadata"])
                result.append(record)
            return result

    def get_daily_pnl(self, date: Optional[datetime] = None) -> dict:
        """Get daily P&L summary from trade audit.

        Returns:
            Dict with trade counts and estimated P&L
        """
        if date is None:
            date = datetime.now(timezone.utc)

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
                FROM trade_audit
                WHERE timestamp >= ? AND timestamp < ?
                """,
                (start.isoformat(), end.isoformat()),
            )
            row = cur.fetchone()
            return {
                "date": start.strftime("%Y-%m-%d"),
                "total_trades": row[0] or 0,
                "filled": row[1] or 0,
                "rejected": row[2] or 0,
                "errors": row[3] or 0,
            }

    def get_last_trade_time(self, symbol: str) -> Optional[datetime]:
        """Get the most recent trade timestamp for a symbol."""
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT timestamp FROM trade_audit
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol,),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                return datetime.fromisoformat(row[0])
            except Exception:
                return None

    def get_last_trade_time_for_underlying(self, underlying: str) -> Optional[datetime]:
        """Get the most recent trade timestamp for an underlying symbol.

        Uses a metadata string match for options trades.
        """
        pattern = f'"underlying": "{underlying}"'
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT timestamp FROM trade_audit
                WHERE symbol = ? OR metadata LIKE ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (underlying, f"%{pattern}%"),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                return datetime.fromisoformat(row[0])
            except Exception:
                return None

    def get_x_since_id(self, username: str) -> Optional[str]:
        """Get last seen X since_id for a username."""
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT since_id FROM x_state WHERE username = ?",
                (username,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set_x_since_id(self, username: str, since_id: str) -> None:
        """Set last seen X since_id for a username."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO x_state(username, since_id, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(username) DO UPDATE SET
                    since_id=excluded.since_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (username, since_id),
            )
