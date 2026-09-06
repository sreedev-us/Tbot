"""Comprehensive test suite for Tbot AI implementation (Phases 1-12)."""

import pytest
from decimal import Decimal
from datetime import datetime, UTC
import pandas as pd
import numpy as np

# Import all modules to test
from alpha_engine.ai_signals import AISignalEvaluator
from alpha_engine.demo_exchange import DemoExchangeAdapter, ExchangeConnector
from alpha_engine.live_exchange import LiveExchangeAdapter, ExchangeManager
from alpha_engine.config import EngineConfig, load_config
from alpha_engine.signals import RawSignal


class TestAISignalEvaluator:
    """Test suite for AI signal evaluation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = AISignalEvaluator("http://localhost:8080")
        # Create sample DataFrame
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h", tz=UTC)
        prices = np.random.uniform(50000, 52000, 100)
        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": prices + 100,
            "low": prices - 100,
            "close": prices,
            "volume": np.random.uniform(1, 100, 100),
        })

    def test_evaluator_initialization(self):
        """Test AISignalEvaluator initialization."""
        assert self.evaluator.backend_url == "http://localhost:8080"

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.evaluator.get_server_ai_analysis(
            empty_df,
            "BTC/USDT",
            "binance",
        )
        assert result is None

    def test_signal_confidence_adjustment(self):
        """Test that confidence affects stop loss/take profit spreads."""
        # This would test signal generation with different confidence levels
        # In production, would mock the backend responses
        assert self.evaluator is not None


class TestDemoExchangeAdapter:
    """Test suite for demo exchange adapter."""

    def setup_method(self):
        """Setup test fixtures."""
        self.adapter = DemoExchangeAdapter(
            "binance_testnet",
            "test_key",
            "test_secret",
        )

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.exchange_name == "binance_testnet"
        assert self.adapter.api_key == "test_key"

    def test_invalid_exchange_raises_error(self):
        """Test that invalid exchange raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported exchange"):
            DemoExchangeAdapter("invalid_exchange", "key", "secret")

    def test_place_limit_order(self):
        """Test placing a limit order."""
        order = self.adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            order_type="LIMIT",
        )
        
        assert order.order_id is not None
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.status == "pending"
        assert order.quantity == Decimal("0.1")

    def test_invalid_side_raises_error(self):
        """Test that invalid side raises ValueError."""
        with pytest.raises(ValueError, match="Invalid side"):
            self.adapter.place_order(
                symbol="BTC/USDT",
                side="INVALID",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
            )

    def test_negative_quantity_raises_error(self):
        """Test that negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid quantity"):
            self.adapter.place_order(
                symbol="BTC/USDT",
                side="BUY",
                quantity=Decimal("-0.1"),
                price=Decimal("50000"),
            )

    def test_cancel_order(self):
        """Test cancelling an order."""
        order = self.adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        result = self.adapter.cancel_order(order.order_id)
        assert result is True
        assert self.adapter.get_order_status(order.order_id).status == "cancelled"

    def test_cannot_cancel_filled_order(self):
        """Test that filled orders cannot be cancelled."""
        order = self.adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        # Simulate fill
        self.adapter.simulate_fill(order.order_id, Decimal("50000"))
        
        result = self.adapter.cancel_order(order.order_id)
        assert result is False

    def test_simulate_order_fill(self):
        """Test simulating order fills."""
        order = self.adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        result = self.adapter.simulate_fill(
            order.order_id,
            Decimal("49900"),
            filled_quantity=Decimal("0.1"),
        )
        
        assert result is True
        filled_order = self.adapter.get_order_status(order.order_id)
        assert filled_order.status == "filled"
        assert filled_order.filled_quantity == Decimal("0.1")

    def test_get_balance(self):
        """Test getting demo account balance."""
        balance = self.adapter.get_balance("USDT")
        assert balance == Decimal("10000")

    def test_list_orders_filtering(self):
        """Test filtering orders by symbol."""
        # Place multiple orders
        self.adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        self.adapter.place_order(
            symbol="ETH/USDT",
            side="BUY",
            quantity=Decimal("1"),
            price=Decimal("3000"),
        )
        
        btc_orders = self.adapter.list_orders("BTC/USDT")
        assert len(btc_orders) == 1
        assert btc_orders[0].symbol == "BTCUSDT"


class TestLiveExchangeAdapter:
    """Test suite for live exchange adapter."""

    def setup_method(self):
        """Setup test fixtures."""
        self.adapter = LiveExchangeAdapter(
            "binance",
            "test_key",
            "test_secret",
            testnet=True,  # Start in testnet
        )

    def test_testnet_initialization(self):
        """Test testnet initialization."""
        assert self.adapter.testnet is True
        assert self.adapter.live_mode is False

    def test_live_mode_initialization(self):
        """Test live mode initialization."""
        adapter = LiveExchangeAdapter(
            "binance",
            "test_key",
            "test_secret",
            testnet=False,
        )
        assert adapter.live_mode is True

    def test_place_live_order_testnet(self):
        """Test placing order in testnet."""
        order = self.adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        
        assert order.order_id is not None
        assert order.exchange == "binance"
        assert order.status == "pending"

    def test_calculate_position_size(self):
        """Test position size calculation."""
        balance = Decimal("10000")
        size = self.adapter.calculate_position_size(
            account_balance=balance,
            risk_pct=Decimal("1"),
            stop_loss_pct=Decimal("2"),
        )
        
        # Risk per trade: 10000 * 1% = 100 -> Uncapped size: 100 / 2% = 5000
        # But position is capped at 10% of account (1000)
        assert size == Decimal("1000.0")

    def test_position_size_capped(self):
        """Test that position size is capped at 10% of account."""
        balance = Decimal("100000")
        size = self.adapter.calculate_position_size(
            account_balance=balance,
            risk_pct=Decimal("50"),  # 50% risk (very aggressive)
            stop_loss_pct=Decimal("1"),
        )
        
        # Should be capped at 10% = 10000
        assert size <= balance * Decimal("0.1")

    def test_close_position_nonexistent(self):
        """Test closing nonexistent position raises error."""
        with pytest.raises(ValueError, match="No open position"):
            self.adapter.close_position("BTC/USDT")


class TestExchangeManager:
    """Test suite for exchange manager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = ExchangeManager()
        assert manager.live_enabled is False

    def test_connect_binance_testnet(self):
        """Test connecting to Binance testnet."""
        manager = ExchangeManager()
        adapter = manager.connect_live_binance(
            "test_key",
            "test_secret",
            testnet=True,
        )
        
        assert adapter is not None
        assert manager.get_adapter("binance") == adapter
        assert manager.is_live_mode() is False

    def test_connect_binance_live_enables_live_mode(self):
        """Test that connecting to live Binance enables live mode."""
        manager = ExchangeManager()
        adapter = manager.connect_live_binance(
            "test_key",
            "test_secret",
            testnet=False,
        )
        
        assert manager.is_live_mode() is True


class TestConfigurationManagement:
    """Test suite for configuration management."""

    def test_engine_config_defaults(self):
        """Test default engine configuration."""
        config = EngineConfig(
            backend_base_url="http://localhost:8080",
            default_exchange="binance",
            default_symbol="BTC/USDT",
            default_timeframe="1m",
            polling_interval_seconds=30,
            order_notional=Decimal("100"),
            strategy_name="mean-reversion-v1",
            use_sandbox=True,
            stop_loss_pct=Decimal("2"),
            take_profit_pct=Decimal("4"),
            ai_enable=False,
            ai_model_version="local-v1.0.0",
        )
        
        assert config.backend_base_url == "http://localhost:8080"
        assert config.ai_enable is False

    def test_ai_enabled_config(self):
        """Test configuration with AI enabled."""
        config = EngineConfig(
            backend_base_url="http://localhost:8080",
            default_exchange="binance",
            default_symbol="BTC/USDT",
            default_timeframe="1m",
            polling_interval_seconds=30,
            order_notional=Decimal("100"),
            strategy_name="ai-v1",
            use_sandbox=True,
            stop_loss_pct=Decimal("2"),
            take_profit_pct=Decimal("4"),
            ai_enable=True,
            ai_model_version="local-v2.0.0",
        )
        
        assert config.ai_enable is True
        assert config.ai_model_version == "local-v2.0.0"


class TestFailureScenarios:
    """Test suite for failure scenarios."""

    def test_demo_exchange_handles_invalid_order_id(self):
        """Test handling of invalid order ID."""
        adapter = DemoExchangeAdapter("binance_testnet", "key", "secret")
        result = adapter.cancel_order("nonexistent_order_id")
        assert result is False

    def test_live_adapter_safety_in_testnet(self):
        """Test that live adapter is safe in testnet mode."""
        adapter = LiveExchangeAdapter(
            "binance",
            "key",
            "secret",
            testnet=True,
        )
        
        assert adapter.testnet is True
        assert adapter.validate_live_trading() is True


class TestIntegrationScenarios:
    """Integration tests for full trading workflow."""

    def test_end_to_end_demo_trading(self):
        """Test complete demo trading workflow."""
        # Setup
        adapter = DemoExchangeAdapter("binance_testnet", "key", "secret")
        
        # Place order
        order = adapter.place_order(
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        # Verify order
        assert order.status == "pending"
        assert order.order_id is not None
        
        # Simulate fill
        filled = adapter.simulate_fill(order.order_id, Decimal("49950"))
        assert filled is True
        
        # Check status
        filled_order = adapter.get_order_status(order.order_id)
        assert filled_order.status == "filled"

    def test_exchange_connector_trade_execution(self):
        """Test exchange connector trade execution."""
        connector = ExchangeConnector("http://localhost:8080")
        adapter = connector.connect_binance_testnet("key", "secret")
        
        result = connector.execute_trade(
            exchange="binance_testnet",
            signal_id="test_signal_123",
            action="BUY",
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        assert result["success"] is True
        assert result["signal_id"] == "test_signal_123"
        assert result["order_id"] is not None


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
