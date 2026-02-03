from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

from rich.logging import RichHandler

# Context variable for correlation ID
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set a correlation ID for request tracing.

    Args:
        cid: Correlation ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    _correlation_id.set(cid)
    return cid


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if set
        cid = get_correlation_id()
        if cid:
            log_data["correlation_id"] = cid

        # Add extra fields from record
        for key in ("symbol", "order_id", "event_id", "error"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Filter that adds context fields to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, use JSON structured format (for production)
        log_file: Optional file path to write logs to

    Examples:
        # Development with rich console output
        setup_logging("DEBUG")

        # Production with JSON output
        setup_logging("INFO", json_output=True)

        # With file output
        setup_logging("INFO", log_file="tradebot.log")
    """
    # Check environment for JSON output
    if os.getenv("LOG_JSON", "").lower() in ("1", "true", "yes"):
        json_output = True

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers = []

    # Add context filter to root logger
    root.addFilter(ContextFilter())

    # Console handler
    if json_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))

    root.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name.

    Args:
        name: Logger name (e.g., "tradebot", "strategy", "broker")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding fields to log records.

    Example:
        with LogContext(symbol="AAPL", order_id="123"):
            log.info("Processing order")  # Will include symbol and order_id
    """

    def __init__(self, **fields: Any):
        self.fields = fields
        self._old_factory = None

    def __enter__(self) -> "LogContext":
        self._old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in self.fields.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args) -> None:
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)


def log_trade_event(
    logger: logging.Logger,
    event_type: str,
    symbol: str,
    **kwargs: Any,
) -> None:
    """Log a trade-related event with standard fields.

    Args:
        logger: Logger instance
        event_type: Type of event (order_submitted, order_filled, etc.)
        symbol: Trading symbol
        **kwargs: Additional fields (side, qty, price, order_id, etc.)
    """
    extra = {"symbol": symbol, "event_type": event_type}
    extra.update(kwargs)

    # Create message
    parts = [f"[{event_type.upper()}]", symbol]
    for key, value in kwargs.items():
        if value is not None:
            parts.append(f"{key}={value}")

    message = " ".join(parts)

    # Use LogContext to add extra fields
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "(trade)",
        0,
        message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)

    logger.handle(record)


def log_error_with_context(
    logger: logging.Logger,
    message: str,
    error: Exception,
    **context: Any,
) -> None:
    """Log an error with additional context.

    Args:
        logger: Logger instance
        message: Error message
        error: Exception that occurred
        **context: Additional context fields
    """
    extra = {"error": str(error), "error_type": type(error).__name__}
    extra.update(context)

    record = logger.makeRecord(
        logger.name,
        logging.ERROR,
        "(error)",
        0,
        f"{message}: {error}",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)

    logger.handle(record)
