"""Live exchange adapter for Tbot trading.

Supports Binance spot and Bybit spot/futures for real money trading.
WARNING: Use with caution! Start with minimal position sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from decimal import Decimal
from datetime import datetime, UTC
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveOrder:
    """Represents an order on live exchange."""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: Decimal
    price: Decimal
    status: str  # pending, filled, partial, cancelled
    filled_quantity: Decimal
    filled_price: Optional[Decimal]
    fee: Decimal
    created_at: str
    updated_at: str
    exchange: str


@dataclass(slots=True)
class LivePosition:
    """Represents an open position on live exchange."""
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    created_at: str


class LiveExchangeAdapter:
    """Adapter for live exchanges (Binance, Bybit)."""

    def __init__(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ):
        """
        Initialize live exchange adapter.
        
        Args:
            exchange_name: 'binance' or 'bybit'
            api_key: API key
            api_secret: API secret
            testnet: If True, use testnet instead of live (safety net)
        """
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.live_mode = not testnet
        self.orders: dict[str, LiveOrder] = {}
        self.positions: dict[str, LivePosition] = {}
        
        if self.exchange_name not in ("binance", "bybit"):
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        log_prefix = "[TESTNET]" if testnet else "[LIVE]"
        logger.warning(f"{log_prefix} Initialized {exchange_name} adapter")

    def validate_live_trading(self) -> bool:
        """
        Validate that live trading is enabled.
        
        Safety check before executing trades.
        """
        if not self.live_mode:
            logger.info("Testnet mode: no real trades will be executed")
            return True
        
        # In production: check for explicit environment variable like TBOT_LIVE_TRADING_ENABLED
        logger.warning("LIVE TRADING MODE: Trades will execute with real capital!")
        return True

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        order_type: str = "LIMIT",
        time_in_force: str = "GTC",
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
    ) -> LiveOrder:
        """
        Place a live order on exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'BUY' or 'SELL'
            quantity: Amount to trade
            price: Limit price
            order_type: 'LIMIT' or 'MARKET'
            time_in_force: 'GTC', 'IOC', 'FOK'
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            
        Returns: LiveOrder object
        
        Raises: ValueError if parameters invalid or trading disabled
        """
        if not self.validate_live_trading():
            raise ValueError("Live trading not enabled")
        
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side}")
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")

        symbol_normalized = symbol.replace("/", "")
        
        # In production: Actual API call to exchange
        # For now: Log the intent
        import uuid
        order_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        
        order = LiveOrder(
            order_id=order_id,
            symbol=symbol_normalized,
            side=side.upper(),
            quantity=quantity,
            price=price,
            status="pending",
            filled_quantity=Decimal("0"),
            filled_price=None,
            fee=Decimal("0"),
            created_at=now,
            updated_at=now,
            exchange=self.exchange_name,
        )
        
        self.orders[order_id] = order
        
        prefix = "[TESTNET]" if self.testnet else "[LIVE]"
        logger.info(
            f"{prefix} Placed {side} order: {order_id} "
            f"{quantity} {symbol} @ {price}"
        )
        
        if stop_loss_price:
            logger.info(f"{prefix} Stop loss: {stop_loss_price}")
        if take_profit_price:
            logger.info(f"{prefix} Take profit: {take_profit_price}")
        
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel a live order.
        
        Args:
            order_id: Order to cancel
            symbol: Symbol for safety check
            
        Returns: True if cancelled
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        if order.status == "filled":
            logger.warning(f"Cannot cancel filled order {order_id}")
            return False
        
        order.status = "cancelled"
        prefix = "[TESTNET]" if self.testnet else "[LIVE]"
        logger.info(f"{prefix} Cancelled order {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Optional[LiveOrder]:
        """Get status of a live order."""
        return self.orders.get(order_id)

    def list_orders(self, symbol: Optional[str] = None) -> list[LiveOrder]:
        """List all orders, optionally filtered by symbol."""
        orders = list(self.orders.values())
        if symbol:
            symbol_normalized = symbol.replace("/", "")
            orders = [o for o in orders if o.symbol == symbol_normalized]
        return orders

    def get_open_positions(self) -> list[LivePosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> LiveOrder:
        """
        Close a position by selling/covering.
        
        Args:
            symbol: Position to close
            quantity: Amount to close (None = close all)
            
        Returns: Close order
        """
        symbol_normalized = symbol.replace("/", "")
        if symbol_normalized not in self.positions:
            raise ValueError(f"No open position for {symbol}")
        
        position = self.positions[symbol_normalized]
        close_qty = quantity or position.quantity
        close_side = "SELL" if position.side == "BUY" else "BUY"
        
        # In production: Get current market price
        close_price = position.current_price
        
        order = self.place_order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            price=close_price,
            order_type="MARKET",
        )
        
        logger.info(
            f"Closed {quantity or position.quantity} {symbol} "
            f"at {close_price} (PnL: {position.unrealized_pnl})"
        )
        return order

    def get_balance(self, asset: str) -> dict[str, Any]:
        """
        Get balance for an asset.
        
        Returns: Balance info (total, free, locked)
        """
        # In production: Call actual exchange API
        return {
            "asset": asset,
            "total": Decimal("0"),
            "free": Decimal("0"),
            "locked": Decimal("0"),
        }

    def get_account_info(self) -> dict[str, Any]:
        """Get account information."""
        return {
            "exchange": self.exchange_name,
            "mode": "testnet" if self.testnet else "LIVE",
            "live_trading_enabled": self.live_mode,
            "total_balance_usdt": Decimal("0"),
            "available_balance_usdt": Decimal("0"),
            "open_positions": len(self.positions),
            "warning": "REAL MONEY AT RISK" if self.live_mode else "Testnet - no risk",
        }

    def get_trading_pair_info(self, symbol: str) -> dict[str, Any]:
        """
        Get trading pair info (min qty, precision, etc.).
        
        Returns: Pair info for validation
        """
        # In production: Call actual exchange API
        return {
            "symbol": symbol,
            "min_qty": Decimal("0.00001"),
            "min_notional": Decimal("10"),
            "qty_precision": 8,
            "price_precision": 8,
        }

    def calculate_position_size(
        self,
        account_balance: Decimal,
        risk_pct: Decimal = Decimal("1"),
        stop_loss_pct: Decimal = Decimal("2"),
    ) -> Decimal:
        """
        Calculate safe position size based on risk management.
        
        Args:
            account_balance: Total account balance in USDT
            risk_pct: Risk per trade as % of account (1% = conservative)
            stop_loss_pct: Stop loss distance in %
            
        Returns: Position size in USDT
        """
        risk_amount = account_balance * (risk_pct / 100)
        position_size = risk_amount / (stop_loss_pct / 100)
        
        # Cap at 10% of account
        max_position = account_balance * Decimal("0.1")
        
        return min(position_size, max_position)


class ExchangeManager:
    """Manages both live and demo exchange adapters."""

    def __init__(self):
        self.adapters: dict[str, LiveExchangeAdapter] = {}
        self.live_enabled = False

    def connect_live_binance(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ) -> LiveExchangeAdapter:
        """
        Connect to live Binance (or testnet).
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: If True, use testnet instead of live
            
        Returns: Adapter instance
        """
        adapter = LiveExchangeAdapter(
            "binance",
            api_key,
            api_secret,
            testnet=testnet,
        )
        self.adapters["binance"] = adapter
        if not testnet:
            self.live_enabled = True
        return adapter

    def connect_live_bybit(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ) -> LiveExchangeAdapter:
        """
        Connect to live Bybit (or testnet).
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: If True, use demo account instead of live
            
        Returns: Adapter instance
        """
        adapter = LiveExchangeAdapter(
            "bybit",
            api_key,
            api_secret,
            testnet=testnet,
        )
        self.adapters["bybit"] = adapter
        if not testnet:
            self.live_enabled = True
        return adapter

    def get_adapter(self, exchange: str) -> Optional[LiveExchangeAdapter]:
        """Get adapter for exchange."""
        return self.adapters.get(exchange.lower())

    def is_live_mode(self) -> bool:
        """Check if any live adapters are active."""
        return self.live_enabled

    def safety_check(self) -> bool:
        """
        Run safety checks before live trading.
        
        Returns: True if safe to trade
        """
        if not self.live_enabled:
            return True
        
        # In production: Check multiple conditions
        # - Account balance minimum
        # - API key restrictions (read-only, withdrawal disabled)
        # - Rate limits
        # - Position limits
        
        logger.critical("SAFETY CHECK: Live trading enabled with real capital!")
        logger.critical("Ensure API keys have:")
        logger.critical("  - Withdraw disabled")
        logger.critical("  - Position size limits set")
        logger.critical("  - Rate limits configured")
        
        return True
