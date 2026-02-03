"""Tests for RiskManager."""
import pytest
from tradebot.risk.risk_manager import RiskManager


class TestRiskManager:
    """Tests for RiskManager class."""

    def test_position_value_ok_within_limit(self, risk_manager: RiskManager):
        """Test that positions within limit are allowed."""
        assert risk_manager.position_value_ok(1000.0) is True
        assert risk_manager.position_value_ok(2000.0) is True

    def test_position_value_ok_exceeds_limit(self, risk_manager: RiskManager):
        """Test that positions exceeding limit are rejected."""
        assert risk_manager.position_value_ok(2001.0) is False
        assert risk_manager.position_value_ok(5000.0) is False

    def test_calc_qty_basic(self, risk_manager: RiskManager):
        """Test basic position sizing."""
        equity = 100000.0
        price = 100.0

        qty = risk_manager.calc_qty(equity_usd=equity, price=price)

        # Risk budget = 100000 * 0.01 = 1000
        # Stop distance = 100 * 0.005 = 0.5
        # Raw qty = 1000 / 0.5 = 2000
        # Max by value = 2000 / 100 = 20
        # Result should be capped by max_position_value
        assert qty == 20.0

    def test_calc_qty_zero_price(self, risk_manager: RiskManager):
        """Test that zero price returns zero qty."""
        assert risk_manager.calc_qty(equity_usd=100000.0, price=0.0) == 0.0

    def test_calc_qty_negative_price(self, risk_manager: RiskManager):
        """Test that negative price returns zero qty."""
        assert risk_manager.calc_qty(equity_usd=100000.0, price=-100.0) == 0.0

    def test_calc_qty_respects_max_position(self, risk_manager: RiskManager):
        """Test that qty respects max position value."""
        # With price = 50, max_position_value = 2000
        # Max qty by value = 2000 / 50 = 40
        qty = risk_manager.calc_qty(equity_usd=1000000.0, price=50.0)
        assert qty <= 40.0

    def test_bracket_prices_buy(self, risk_manager: RiskManager):
        """Test bracket prices for buy orders."""
        entry = 100.0
        sl, tp = risk_manager.bracket_prices(entry_price=entry, side="buy")

        # For buy: SL below entry, TP above
        assert sl < entry
        assert tp > entry

        # Expected: SL = 100 * (1 - 0.005) = 99.5
        #           TP = 100 * (1 + 0.01) = 101.0
        assert sl == pytest.approx(99.5, rel=0.01)
        assert tp == pytest.approx(101.0, rel=0.01)

    def test_bracket_prices_sell(self, risk_manager: RiskManager):
        """Test bracket prices for sell/short orders."""
        entry = 100.0
        sl, tp = risk_manager.bracket_prices(entry_price=entry, side="sell")

        # For sell: SL above entry, TP below
        assert sl > entry
        assert tp < entry

        # Expected: SL = 100 * (1 + 0.005) = 100.5
        #           TP = 100 * (1 - 0.01) = 99.0
        assert sl == pytest.approx(100.5, rel=0.01)
        assert tp == pytest.approx(99.0, rel=0.01)

    def test_bracket_prices_rounding(self, risk_manager: RiskManager):
        """Test that bracket prices are properly rounded."""
        entry = 123.456
        sl, tp = risk_manager.bracket_prices(entry_price=entry, side="buy")

        # Should be rounded to 4 decimal places
        assert sl == round(sl, 4)
        assert tp == round(tp, 4)


class TestRiskManagerEdgeCases:
    """Edge case tests for RiskManager."""

    def test_very_small_equity(self):
        """Test with very small equity."""
        rm = RiskManager(
            max_daily_loss_usd=10.0,
            max_position_value_usd=50.0,
            risk_per_trade_pct=0.01,
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
        )
        qty = rm.calc_qty(equity_usd=100.0, price=100.0)
        assert qty >= 0

    def test_very_high_price(self):
        """Test with very high price stock."""
        rm = RiskManager(
            max_daily_loss_usd=250.0,
            max_position_value_usd=2000.0,
            risk_per_trade_pct=0.01,
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
        )
        # Price higher than max position value
        qty = rm.calc_qty(equity_usd=100000.0, price=5000.0)
        # Should be limited by max_position_value
        assert qty * 5000 <= 2000.0

    def test_custom_risk_parameters(self):
        """Test with custom risk parameters."""
        rm = RiskManager(
            max_daily_loss_usd=500.0,
            max_position_value_usd=5000.0,
            risk_per_trade_pct=0.02,  # 2% risk per trade
            stop_loss_pct=0.02,  # 2% stop loss
            take_profit_pct=0.04,  # 4% take profit (2:1 R/R)
        )

        sl, tp = rm.bracket_prices(entry_price=100.0, side="buy")
        assert sl == pytest.approx(98.0, rel=0.01)  # 2% below
        assert tp == pytest.approx(104.0, rel=0.01)  # 4% above
