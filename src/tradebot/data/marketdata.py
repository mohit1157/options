from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tradebot.core.logger import get_logger
from tradebot.core.retry import with_retry

log = get_logger("marketdata")


class MarketDataError(Exception):
    """Base exception for market data errors."""
    pass


def _tf(tf: str) -> TimeFrame:
    """Convert timeframe string to Alpaca TimeFrame."""
    t = tf.lower()
    if t in ("1min", "1m", "1"):
        return TimeFrame.Minute
    if t in ("3min", "3m", "3"):
        return TimeFrame(3, TimeFrame.Unit.Minute)
    if t in ("5min", "5m", "5"):
        return TimeFrame(5, TimeFrame.Unit.Minute)
    if t in ("15min", "15m", "15"):
        return TimeFrame(15, TimeFrame.Unit.Minute)
    if t in ("30min", "30m", "30"):
        return TimeFrame(30, TimeFrame.Unit.Minute)
    if t in ("1hour", "1h", "60min", "60m"):
        return TimeFrame.Hour
    if t in ("1day", "1d", "d", "day"):
        return TimeFrame.Day
    return TimeFrame.Minute


@with_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
def fetch_bars(
    api_key: str,
    api_secret: str,
    symbol: str,
    timeframe: str,
    limit: int = 200,
    feed: str = "iex",
) -> pd.DataFrame:
    """Fetch historical bars for a symbol.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        symbol: Stock symbol (e.g., "AAPL")
        timeframe: Bar timeframe (e.g., "1Min", "5Min", "1Day")
        limit: Maximum number of bars to fetch
        feed: Data feed ("iex" or "sip")

    Returns:
        DataFrame with columns: open, high, low, close, volume, etc.
        Returns empty DataFrame on error.

    Example:
        bars = fetch_bars(key, secret, "SPY", "1Min", limit=100)
        if not bars.empty:
            print(f"Latest close: {bars['close'].iloc[-1]}")
    """
    try:
        client = StockHistoricalDataClient(api_key, api_secret)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_tf(timeframe),
            limit=limit,
            feed=feed,
        )
        bars = client.get_stock_bars(req).df

        if bars is None or len(bars) == 0:
            log.debug(f"No bars returned for {symbol}")
            return pd.DataFrame()

        # Handle MultiIndex if present
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0, drop=True)

        log.debug(f"Fetched {len(bars)} bars for {symbol}")
        return bars

    except Exception as e:
        log.error(f"Failed to fetch bars for {symbol}: {e}")
        return pd.DataFrame()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        series: Price series (typically close prices)
        span: EMA period (e.g., 9 for 9-period EMA)

    Returns:
        Series of EMA values
    """
    if len(series) == 0:
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        series: Price series
        window: SMA period

    Returns:
        Series of SMA values
    """
    if len(series) == 0:
        return pd.Series(dtype=float)
    return series.rolling(window=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        series: Price series (typically close prices)
        period: RSI period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    if len(series) < period + 1:
        return pd.Series(dtype=float)

    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Handle edge cases: all gains (no losses) or all losses (no gains)
    rsi_values = pd.Series(index=series.index, dtype=float)

    for i in range(len(series)):
        ag = avg_gain.iloc[i]
        al = avg_loss.iloc[i]

        if al == 0:
            # No losses - RSI is 100 (or near it if there are also no gains)
            rsi_values.iloc[i] = 100.0 if ag > 0 else 50.0
        elif ag == 0:
            # No gains - RSI is 0
            rsi_values.iloc[i] = 0.0
        else:
            rs = ag / al
            rsi_values.iloc[i] = 100 - (100 / (1 + rs))

    return rsi_values


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        series: Price series
        window: Period for moving average
        num_std: Number of standard deviations

    Returns:
        Tuple of (upper_band, middle_band, lower_band) Series
    """
    middle = sma(series, window)
    std = series.rolling(window=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)

    Returns:
        Series of ATR values
    """
    if len(close) < 2:
        return pd.Series(dtype=float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def get_latest_price(
    api_key: str,
    api_secret: str,
    symbol: str,
    feed: str = "iex",
) -> Optional[float]:
    """Get the latest price for a symbol.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        symbol: Stock symbol
        feed: Data feed

    Returns:
        Latest close price or None on error
    """
    bars = fetch_bars(api_key, api_secret, symbol, "1Min", limit=1, feed=feed)
    if bars.empty:
        return None
    return float(bars["close"].iloc[-1])
