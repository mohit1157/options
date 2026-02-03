"""Tests for EMA and other technical indicator calculations."""
import pytest
import pandas as pd
import numpy as np

from tradebot.data.marketdata import ema, sma, rsi, bollinger_bands, atr


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        series = pd.Series([100.0, 102.0, 101.0, 103.0, 102.0])
        result = ema(series, span=3)

        assert len(result) == 5
        # First value should equal first input
        assert result.iloc[0] == 100.0
        # EMA should smooth the series
        assert all(pd.notna(result))

    def test_ema_empty_series(self):
        """Test EMA with empty series."""
        series = pd.Series([], dtype=float)
        result = ema(series, span=9)
        assert len(result) == 0

    def test_ema_single_value(self):
        """Test EMA with single value."""
        series = pd.Series([100.0])
        result = ema(series, span=9)

        assert len(result) == 1
        assert result.iloc[0] == 100.0

    def test_ema_crossover_detection(self):
        """Test that EMA can be used to detect crossovers."""
        # Create an uptrending series
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        fast_ema = ema(prices, span=3)
        slow_ema = ema(prices, span=7)

        # Fast EMA should be above slow EMA in uptrend
        assert fast_ema.iloc[-1] > slow_ema.iloc[-1]

    def test_ema_vs_sma(self):
        """Test that EMA reacts faster to price changes."""
        # Create series with a jump
        prices = pd.Series([100] * 10 + [110] * 5)
        fast_ema = ema(prices, span=5)
        simple_ma = sma(prices, window=5)

        # Right after the jump (index 10), EMA should react faster
        # Compare at the first point after the jump where SMA has caught up
        # EMA at index 10 should be higher than SMA at index 10
        assert fast_ema.iloc[10] > simple_ma.iloc[10]


class TestSMA:
    """Tests for SMA calculation."""

    def test_sma_basic(self):
        """Test basic SMA calculation."""
        series = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])
        result = sma(series, window=3)

        # First 2 values should be NaN (not enough data)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value should be average of first 3
        assert result.iloc[2] == pytest.approx((100 + 102 + 104) / 3)

    def test_sma_empty_series(self):
        """Test SMA with empty series."""
        series = pd.Series([], dtype=float)
        result = sma(series, window=9)
        assert len(result) == 0


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create a series with clear trend
        prices = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
                          45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00])
        result = rsi(prices, period=14)

        assert len(result) == len(prices)
        # RSI should be in [0, 100] range for valid values
        valid_values = result.dropna()
        assert all(0 <= v <= 100 for v in valid_values)

    def test_rsi_overbought(self):
        """Test RSI in overbought condition."""
        # Strong uptrend should result in high RSI
        # Use floats to ensure proper diff calculation
        prices = pd.Series([float(x) for x in range(100, 150)])
        result = rsi(prices, period=14)

        # RSI should be high (overbought > 70)
        # With all gains and no losses, RSI should be very high
        valid_rsi = result.dropna()
        assert len(valid_rsi) > 0
        assert valid_rsi.iloc[-1] > 70

    def test_rsi_oversold(self):
        """Test RSI in oversold condition."""
        # Strong downtrend should result in low RSI
        prices = pd.Series(range(150, 100, -1))  # 50 consecutive losses
        result = rsi(prices, period=14)

        # RSI should be low (oversold < 30)
        assert result.iloc[-1] < 30

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = pd.Series([100, 101, 102])  # Less than period + 1
        result = rsi(prices, period=14)

        assert len(result) == 0


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_basic(self):
        """Test basic Bollinger Bands calculation."""
        prices = pd.Series([100 + i + (i % 5 - 2) for i in range(50)])
        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

        # For valid values, upper > middle > lower
        valid_idx = middle.dropna().index
        for idx in valid_idx:
            assert upper[idx] > middle[idx] > lower[idx]

    def test_bollinger_width(self):
        """Test that Bollinger Band width reflects volatility."""
        # Low volatility series
        low_vol = pd.Series([100 + 0.1 * (i % 2) for i in range(50)])
        upper_lv, middle_lv, lower_lv = bollinger_bands(low_vol, window=20)

        # High volatility series
        high_vol = pd.Series([100 + 5 * (i % 2) for i in range(50)])
        upper_hv, middle_hv, lower_hv = bollinger_bands(high_vol, window=20)

        # High volatility bands should be wider
        width_lv = (upper_lv.iloc[-1] - lower_lv.iloc[-1])
        width_hv = (upper_hv.iloc[-1] - lower_hv.iloc[-1])
        assert width_hv > width_lv


class TestATR:
    """Tests for Average True Range calculation."""

    def test_atr_basic(self):
        """Test basic ATR calculation."""
        high = pd.Series([102, 104, 103, 105, 104, 106, 105, 107] * 5)
        low = pd.Series([98, 100, 99, 101, 100, 102, 101, 103] * 5)
        close = pd.Series([100, 102, 101, 103, 102, 104, 103, 105] * 5)

        result = atr(high, low, close, period=14)

        assert len(result) == len(close)
        # ATR should be positive
        valid_values = result.dropna()
        assert all(v > 0 for v in valid_values)

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        high = pd.Series([102])
        low = pd.Series([98])
        close = pd.Series([100])

        result = atr(high, low, close, period=14)
        assert len(result) == 0
