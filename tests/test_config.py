"""Tests for configuration validation."""
import os
import pytest
from unittest.mock import patch

from tradebot.config import Settings


class TestSettingsValidation:
    """Tests for Settings validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

        assert settings.alpaca_base_url == "https://paper-api.alpaca.markets"
        assert settings.alpaca_data_feed == "iex"
        assert settings.ema_fast == 9
        assert settings.ema_slow == 21
        assert settings.enable_options is False

    def test_ema_validation_fast_greater_than_slow(self):
        """Test that ema_fast < ema_slow is enforced."""
        with pytest.raises(ValueError, match="ema_fast.*must be < ema_slow"):
            Settings(ema_fast=21, ema_slow=9)

    def test_ema_validation_equal_values(self):
        """Test that ema_fast != ema_slow is enforced."""
        with pytest.raises(ValueError, match="ema_fast.*must be < ema_slow"):
            Settings(ema_fast=9, ema_slow=9)

    def test_ema_must_be_positive(self):
        """Test that EMA values must be >= 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            Settings(ema_fast=0, ema_slow=21)

    def test_risk_params_must_be_positive(self):
        """Test that risk parameters must be positive."""
        with pytest.raises(ValueError, match="must be > 0"):
            Settings(max_daily_loss_usd=0)

        with pytest.raises(ValueError, match="must be > 0"):
            Settings(max_position_value_usd=-100)

    def test_percentage_params_range(self):
        """Test that percentage parameters are in valid range."""
        with pytest.raises(ValueError, match="must be in"):
            Settings(risk_per_trade_pct=0)

        with pytest.raises(ValueError, match="must be in"):
            Settings(stop_loss_pct=1.5)

    def test_sentiment_threshold_range(self):
        """Test sentiment threshold is in [-1, 1]."""
        # Valid values
        Settings(sentiment_threshold=0.0)
        Settings(sentiment_threshold=0.5)
        Settings(sentiment_threshold=-0.5)

        # Invalid values
        with pytest.raises(ValueError, match="must be in"):
            Settings(sentiment_threshold=1.5)

    def test_option_dte_max_range(self):
        """Test option_dte_max is reasonable."""
        with pytest.raises(ValueError, match="must be in"):
            Settings(option_dte_max=-1)

        with pytest.raises(ValueError, match="must be in"):
            Settings(option_dte_max=400)

    def test_option_order_qty_positive(self):
        """Test option_order_qty must be >= 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            Settings(option_order_qty=0)

    def test_symbols_list_property(self):
        """Test symbols_list property parses correctly."""
        settings = Settings(symbols="SPY,QQQ,AAPL")
        assert settings.symbols_list == ["SPY", "QQQ", "AAPL"]

    def test_symbols_list_empty(self):
        """Test symbols_list with empty string."""
        settings = Settings(symbols="")
        assert settings.symbols_list == []

    def test_symbols_list_single(self):
        """Test symbols_list with single symbol."""
        settings = Settings(symbols="SPY")
        assert settings.symbols_list == ["SPY"]

    def test_symbols_list_strips_whitespace(self):
        """Test that whitespace is stripped from symbols."""
        settings = Settings(symbols=" SPY , QQQ , AAPL ")
        assert settings.symbols_list == ["SPY", "QQQ", "AAPL"]

    def test_rss_feeds_list_property(self):
        """Test rss_feeds_list property parses correctly."""
        settings = Settings(rss_feeds="http://a.com/feed,http://b.com/feed")
        assert len(settings.rss_feeds_list) == 2

    def test_x_handles_list_property(self):
        """Test x_handles_list property parses correctly."""
        settings = Settings(x_handles="@user1,user2,@user3")
        # Should strip @ prefix
        assert settings.x_handles_list == ["user1", "user2", "user3"]

    def test_symbol_format_validation(self):
        """Test that symbol format is validated."""
        # Valid symbols
        Settings(symbols="SPY,AAPL,BRK.B")

        # Invalid symbols
        with pytest.raises(ValueError, match="Invalid symbol"):
            Settings(symbols="SPY,INVALID SYMBOL,AAPL")

    def test_validate_alpaca_credentials(self):
        """Test Alpaca credentials validation method."""
        settings = Settings(alpaca_api_key="", alpaca_api_secret="")
        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            settings.validate_alpaca_credentials()

        settings = Settings(alpaca_api_key="key", alpaca_api_secret="secret")
        settings.validate_alpaca_credentials()  # Should not raise

    def test_validate_grok_credentials(self):
        """Test Grok credentials validation method."""
        settings = Settings(grok_api_key="")
        with pytest.raises(ValueError, match="GROK_API_KEY"):
            settings.validate_grok_credentials()

        settings = Settings(grok_api_key="key")
        settings.validate_grok_credentials()  # Should not raise

    def test_validate_x_credentials(self):
        """Test X API credentials validation method."""
        settings = Settings(x_bearer_token="")
        with pytest.raises(ValueError, match="X_BEARER_TOKEN"):
            settings.validate_x_credentials()

        settings = Settings(x_bearer_token="token")
        settings.validate_x_credentials()  # Should not raise


class TestSettingsFromEnvironment:
    """Tests for loading settings from environment variables."""

    def test_load_from_env(self):
        """Test that settings are loaded from environment."""
        env = {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_API_SECRET": "test_secret",
            "SYMBOLS": "TSLA,NVDA",
            "EMA_FAST": "5",
            "EMA_SLOW": "15",
            "ENABLE_OPTIONS": "true",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings()

        assert settings.alpaca_api_key == "test_key"
        assert settings.alpaca_api_secret == "test_secret"
        assert settings.symbols == "TSLA,NVDA"
        assert settings.ema_fast == 5
        assert settings.ema_slow == 15
        assert settings.enable_options is True
