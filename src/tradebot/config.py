from __future__ import annotations

import re
from typing import Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation.

    All settings are loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Alpaca credentials
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_feed: str = "iex"

    # Strategy parameters
    symbols: str = "SPY"  # Comma-separated
    timeframe: str = "1Min"
    ema_fast: int = 9
    ema_slow: int = 21
    sentiment_threshold: float = 0.15

    # Risk parameters
    max_daily_loss_usd: float = 250.0
    max_position_value_usd: float = 2000.0
    risk_per_trade_pct: float = 0.01
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.01
    trade_cooldown_minutes: int = 30
    market_open_only: bool = True

    # RSS feeds (comma-separated URLs)
    rss_feeds: str = ""

    # X/Twitter API
    x_bearer_token: str = ""
    x_handles: str = ""  # Comma-separated handles

    # Grok / xAI Sentiment
    grok_api_key: str = ""
    grok_base_url: str = "https://api.x.ai/v1"
    grok_model: str = "grok-3-fast"

    # Options trading
    enable_options: bool = False
    option_dte_max: int = 7
    option_strike_tolerance: float = 0.03
    option_order_qty: int = 1
    option_side_on_bull: str = "call"
    option_side_on_bear: str = "put"
    option_use_dynamic_qty: bool = True
    option_min_volume: int = 100
    option_max_spread_pct: float = 0.15
    option_use_bracket: bool = True
    option_stop_loss_pct: float = 0.25
    option_take_profit_pct: float = 0.5

    @field_validator("ema_fast", "ema_slow")
    @classmethod
    def ema_must_be_positive(cls, v: int, info) -> int:
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def validate_ema_order(self) -> "Settings":
        if self.ema_fast >= self.ema_slow:
            raise ValueError(
                f"ema_fast ({self.ema_fast}) must be < ema_slow ({self.ema_slow})"
            )
        return self

    @field_validator("max_daily_loss_usd", "max_position_value_usd")
    @classmethod
    def must_be_positive_float(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {v}")
        return v

    @field_validator("risk_per_trade_pct", "stop_loss_pct", "take_profit_pct")
    @classmethod
    def pct_must_be_reasonable(cls, v: float, info) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"{info.field_name} must be in (0, 1], got {v}")
        return v

    @field_validator("sentiment_threshold")
    @classmethod
    def sentiment_range(cls, v: float) -> float:
        if not -1 <= v <= 1:
            raise ValueError(f"sentiment_threshold must be in [-1, 1], got {v}")
        return v

    @field_validator("option_dte_max")
    @classmethod
    def dte_reasonable(cls, v: int) -> int:
        if v < 0 or v > 365:
            raise ValueError(f"option_dte_max must be in [0, 365], got {v}")
        return v

    @field_validator("option_strike_tolerance")
    @classmethod
    def strike_tolerance_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"option_strike_tolerance must be in [0, 1], got {v}")
        return v

    @field_validator("option_order_qty")
    @classmethod
    def order_qty_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"option_order_qty must be >= 1, got {v}")
        return v

    @field_validator("trade_cooldown_minutes")
    @classmethod
    def cooldown_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"trade_cooldown_minutes must be >= 0, got {v}")
        return v

    @field_validator("option_min_volume")
    @classmethod
    def option_min_volume_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"option_min_volume must be >= 0, got {v}")
        return v

    @field_validator("option_max_spread_pct", "option_stop_loss_pct", "option_take_profit_pct")
    @classmethod
    def option_pct_range(cls, v: float, info) -> float:
        if v <= 0 or v > 1:
            raise ValueError(f"{info.field_name} must be in (0, 1], got {v}")
        return v

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: str) -> str:
        if not v.strip():
            return v
        symbol_pattern = re.compile(r"^[A-Za-z0-9.]{1,10}$")
        for sym in v.split(","):
            sym = sym.strip()
            if sym and not symbol_pattern.match(sym):
                raise ValueError(f"Invalid symbol format: {sym}")
        return v

    # Property helpers to get parsed lists
    @property
    def symbols_list(self) -> list[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    @property
    def rss_feeds_list(self) -> list[str]:
        return [f.strip() for f in self.rss_feeds.split(",") if f.strip()]

    @property
    def x_handles_list(self) -> list[str]:
        return [h.strip().lstrip("@") for h in self.x_handles.split(",") if h.strip()]

    def validate_alpaca_credentials(self) -> None:
        """Raise if Alpaca credentials are missing."""
        if not self.alpaca_api_key or not self.alpaca_api_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set"
            )

    def validate_grok_credentials(self) -> None:
        """Raise if Grok credentials are missing."""
        if not self.grok_api_key:
            raise ValueError("GROK_API_KEY must be set for Grok sentiment")

    def validate_x_credentials(self) -> None:
        """Raise if X API credentials are missing."""
        if not self.x_bearer_token:
            raise ValueError("X_BEARER_TOKEN must be set for X integration")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
