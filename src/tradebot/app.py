from __future__ import annotations

import signal
import sys
import time
from datetime import timedelta
from typing import Optional

import typer

from tradebot.config import get_settings, Settings
from tradebot.core.logger import setup_logging, get_logger
from tradebot.core.timeutils import utcnow
from tradebot.core.circuit_breaker import get_circuit_breaker, CircuitOpenError
from tradebot.ingestion.rss import fetch_rss_events
from tradebot.sentiment.local_rule import LocalRuleSentiment
from tradebot.sentiment.grok import GrokSentimentClient
from tradebot.sentiment.base import SentimentClient
from tradebot.data.store import SQLiteStore
from tradebot.broker.alpaca_broker import AlpacaBroker
from tradebot.risk.risk_manager import RiskManager
from tradebot.strategy.ema_sentiment import EmaSentimentStrategy
from tradebot.strategy.options_ema_sentiment import OptionsEmaSentimentStrategy

log = get_logger("tradebot")
cli_app = typer.Typer(help="Autonomous trading bot (Alpaca + X/RSS + sentiment).")

# Global state for graceful shutdown
_shutdown_requested = False
_cleanup_handlers: list = []


def _signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    log.info(f"Received {sig_name}, initiating graceful shutdown...")
    _shutdown_requested = True


def _register_cleanup(handler: callable) -> None:
    """Register a cleanup handler to be called on shutdown."""
    _cleanup_handlers.append(handler)


def _run_cleanup() -> None:
    """Run all registered cleanup handlers."""
    log.info("Running cleanup handlers...")
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            log.warning(f"Cleanup handler error: {e}")
    log.info("Cleanup complete.")


def _make_sentiment(settings: Settings) -> SentimentClient:
    """Create sentiment client based on configuration."""
    if settings.grok_api_key:
        log.info("Using GrokSentimentClient (xAI API).")
        return GrokSentimentClient(
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
            model=settings.grok_model,
        )
    log.info("Using LocalRuleSentiment (fallback).")
    return LocalRuleSentiment()


def _verify_alpaca_connection(broker: AlpacaBroker) -> bool:
    """Verify Alpaca connection on startup."""
    try:
        account = broker.account()
        log.info(f"Alpaca connected. Account status: {account.status}, Equity: ${float(account.equity):,.2f}")
        return True
    except Exception as e:
        log.error(f"Failed to connect to Alpaca: {e}")
        return False


def _verify_grok_connection(client: SentimentClient) -> bool:
    """Verify Grok API connection if configured."""
    if not isinstance(client, GrokSentimentClient):
        return True  # Local client doesn't need verification

    try:
        from tradebot.core.events import BaseEvent
        test_event = BaseEvent(
            type="news",
            source="test",
            created_at=utcnow(),
            text="Test connection",
        )
        client.analyze(test_event)
        log.info("Grok API connection verified.")
        return True
    except Exception as e:
        log.warning(f"Grok API verification failed: {e}. Will use fallback sentiment.")
        return False


@cli_app.command()
def run(paper: bool = typer.Option(True, "--paper/--live", help="Use Alpaca paper or live trading")):
    """Run the trading bot loop."""
    global _shutdown_requested

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    setup_logging("INFO")
    log.info("Starting trading bot...")

    # Load and validate configuration
    try:
        settings = get_settings()
        settings.validate_alpaca_credentials()
        log.info(f"Configuration loaded. Symbols: {settings.symbols_list}")
    except Exception as e:
        log.error(f"Configuration error: {e}")
        raise typer.Exit(code=1)

    # Initialize database
    store = SQLiteStore()
    store.init()

    # Initialize sentiment client
    sentiment_client = _make_sentiment(settings)
    if hasattr(sentiment_client, "close"):
        _register_cleanup(sentiment_client.close)

    # Initialize broker
    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        paper=paper,
    )

    # Verify connections on startup
    if not _verify_alpaca_connection(broker):
        log.error("Cannot start without Alpaca connection")
        _run_cleanup()
        raise typer.Exit(code=1)

    if not _verify_grok_connection(sentiment_client):
        # Fall back to local sentiment if Grok fails
        if isinstance(sentiment_client, GrokSentimentClient):
            sentiment_client.close()
            sentiment_client = LocalRuleSentiment()
            log.warning("Falling back to LocalRuleSentiment")

    # Initialize risk manager
    risk = RiskManager(
        max_daily_loss_usd=settings.max_daily_loss_usd,
        max_position_value_usd=settings.max_position_value_usd,
        risk_per_trade_pct=settings.risk_per_trade_pct,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
    )

    # Initialize strategies
    strategy = EmaSentimentStrategy(settings=settings, broker=broker, risk=risk, store=store)
    opt_strategy = OptionsEmaSentimentStrategy(settings=settings, broker=broker, risk=risk, store=store)

    # Initialize X client if configured
    x_client = None
    if settings.x_bearer_token and settings.x_handles_list:
        try:
            from tradebot.ingestion.x_client import XClient
            x_client = XClient(bearer_token=settings.x_bearer_token)
            _register_cleanup(x_client.close)
            log.info(f"X client initialized. Monitoring handles: {settings.x_handles_list}")
        except Exception as e:
            log.warning(f"Failed to initialize X client: {e}")
    else:
        log.info("X client not configured (no X_BEARER_TOKEN or X_HANDLES)")

    # Get circuit breakers for external services
    alpaca_breaker = get_circuit_breaker("alpaca", failure_threshold=5, recovery_timeout=60.0)
    sentiment_breaker = get_circuit_breaker("sentiment", failure_threshold=3, recovery_timeout=30.0)

    log.info(f"Bot started. Paper={paper}. Symbols={settings.symbols_list}. Timeframe={settings.timeframe}")
    log.info(f"Options trading: {'enabled' if settings.enable_options else 'disabled'}")

    # Main loop
    loop_count = 0
    while not _shutdown_requested:
        loop_count += 1
        cycle_start = time.monotonic()

        try:
            # 1) RSS ingestion
            if settings.rss_feeds_list:
                try:
                    events = fetch_rss_events(settings.rss_feeds_list)
                    new_events = 0
                    for e in events[:50]:
                        # Check for duplicates before adding
                        if store.event_exists(url=e.url):
                            continue

                        event_id = store.add_event(
                            type=e.type,
                            source=e.source,
                            created_at=e.created_at,
                            author=e.author,
                            url=e.url,
                            text=e.text[:5000],
                        )
                        new_events += 1

                        # Analyze sentiment with circuit breaker protection
                        try:
                            s = sentiment_breaker.call(
                                lambda ev=e: sentiment_client.analyze(ev)
                            )
                            store.add_sentiment(
                                event_id=event_id,
                                score=s.score,
                                label=s.label,
                                created_at=utcnow(),
                            )
                        except CircuitOpenError:
                            log.debug("Sentiment circuit open, skipping analysis")
                        except Exception as ex:
                            log.warning(f"Sentiment analysis failed: {ex}")

                    if new_events > 0:
                        log.info(f"Ingested {new_events} new RSS events")
                except Exception as ex:
                    log.warning(f"RSS ingestion error: {ex}")

            # 2) X/Twitter ingestion
            if x_client and settings.x_handles_list:
                try:
                    for handle in settings.x_handles_list:
                        try:
                            tweets = x_client.fetch_recent_by_username(handle, max_results=5)
                            for t in tweets:
                                if store.event_exists(url=t.url):
                                    continue

                                event_id = store.add_event(
                                    type=t.type,
                                    source=t.source,
                                    created_at=t.created_at,
                                    author=t.author,
                                    url=t.url,
                                    text=t.text[:5000],
                                )

                                try:
                                    s = sentiment_breaker.call(
                                        lambda ev=t: sentiment_client.analyze(ev)
                                    )
                                    store.add_sentiment(
                                        event_id=event_id,
                                        score=s.score,
                                        label=s.label,
                                        created_at=utcnow(),
                                    )
                                except CircuitOpenError:
                                    pass
                                except Exception as ex:
                                    log.warning(f"Sentiment failed for tweet: {ex}")
                        except Exception as ex:
                            log.warning(f"Failed to fetch tweets for @{handle}: {ex}")
                except Exception as ex:
                    log.warning(f"X ingestion error: {ex}")

            # 3) Strategy tick with circuit breaker
            try:
                alpaca_breaker.call(lambda: strategy.tick())
            except CircuitOpenError:
                log.warning("Alpaca circuit open, skipping stock strategy tick")
            except Exception as ex:
                log.exception(f"Strategy tick error: {ex}")

            # 4) Options strategy tick (if enabled)
            if settings.enable_options:
                try:
                    alpaca_breaker.call(lambda: opt_strategy.tick())
                except CircuitOpenError:
                    log.warning("Alpaca circuit open, skipping options strategy tick")
                except Exception as ex:
                    log.exception(f"Options strategy tick error: {ex}")

            # 5) Cleanup old events periodically (every 100 cycles)
            if loop_count % 100 == 0:
                try:
                    deleted = store.cleanup_old_events(hours=24)
                    if deleted > 0:
                        log.info(f"Cleaned up {deleted} old events")
                except Exception as ex:
                    log.warning(f"Event cleanup error: {ex}")

        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
            break
        except Exception as ex:
            log.exception(f"Unexpected loop error: {ex}")

        # Wait for next cycle
        cycle_time = time.monotonic() - cycle_start
        sleep_time = max(0, 60 - cycle_time)  # Run every 60 seconds

        # Check for shutdown during sleep
        if sleep_time > 0 and not _shutdown_requested:
            time.sleep(min(sleep_time, 1))  # Wake up every second to check shutdown
            remaining = sleep_time - 1
            while remaining > 0 and not _shutdown_requested:
                time.sleep(min(remaining, 1))
                remaining -= 1

    # Graceful shutdown
    log.info("Shutting down...")
    _run_cleanup()
    log.info("Bot stopped.")


@cli_app.command()
def validate():
    """Validate configuration without starting the bot."""
    setup_logging("INFO")

    try:
        settings = get_settings()
        log.info("Configuration validation passed!")
        log.info(f"  Symbols: {settings.symbols_list}")
        log.info(f"  EMA: fast={settings.ema_fast}, slow={settings.ema_slow}")
        log.info(f"  Risk: max_loss=${settings.max_daily_loss_usd}, max_pos=${settings.max_position_value_usd}")
        log.info(f"  Options: {'enabled' if settings.enable_options else 'disabled'}")

        # Check credentials
        if settings.alpaca_api_key and settings.alpaca_api_secret:
            log.info("  Alpaca: credentials configured")
        else:
            log.warning("  Alpaca: credentials NOT configured")

        if settings.grok_api_key:
            log.info("  Grok: API key configured")
        else:
            log.info("  Grok: using local sentiment (no API key)")

        if settings.x_bearer_token:
            log.info(f"  X API: configured for handles {settings.x_handles_list}")
        else:
            log.info("  X API: not configured")

    except Exception as e:
        log.error(f"Configuration validation failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    cli_app()
