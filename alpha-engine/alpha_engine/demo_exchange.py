"""Demo/testnet exchange adapter for Tbot trading.

Supports Binance testnet and Bybit demo account for paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from decimal import Decimal


@dataclass(slots=True)
class DemoOrder:
    """Represents an order in demo/testnet environment."""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: Decimal
    price: Decimal
    status: str  # pending, filled, partial, cancelled
    filled_quantity: Decimal
    created_at: str
    updated_at: str


class DemoExchangeAdapter:
    """Adapter for demo/testnet exchanges (Binance testnet, Bybit demo)."""

    def __init__(self, exchange_name: str, api_key: str, api_secret: str):
        """
        Initialize demo exchange adapter.
        
        Args:
            exchange_name: 'binance_testnet' or 'bybit_demo'
            api_key: API key from testnet/demo account
            api_secret: API secret from testnet/demo account
        """
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.orders: dict[str, DemoOrder] = {}
        
        if self.exchange_name not in ("binance_testnet", "bybit_demo"):
            raise ValueError(f"Unsupported exchange: {exchange_name}")

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "LIMIT",
    ) -> DemoOrder:
        """
        Place an order on the demo exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'BUY' or 'SELL'
            quantity: Amount to trade
            price: Limit price (required for LIMIT orders)
            order_type: 'LIMIT' or 'MARKET'
            
        Returns: DemoOrder object
        
        Raises: ValueError if parameters are invalid
        """
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side}")
        
        if order_type == "LIMIT" and price is None:
            raise ValueError("Price is required for LIMIT orders")
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")

        # Convert symbol format if needed
        symbol_normalized = symbol.replace("/", "")
        
        # Create order ID
        import uuid
        from datetime import datetime, UTC
        
        order_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        
        order = DemoOrder(
            order_id=order_id,
            symbol=symbol_normalized,
            side=side.upper(),
            quantity=quantity,
            price=price or Decimal("0"),
            status="pending",
            filled_quantity=Decimal("0"),
            created_at=now,
            updated_at=now,
        )
        
        self.orders[order_id] = order
        print(f"[{self.exchange_name}] Placed {side} order: {order_id}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order to cancel
            
        Returns: True if cancelled, False if not found or already filled
        """
        if order_id not in self.orders:
            print(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        if order.status == "filled":
            print(f"Cannot cancel filled order {order_id}")
            return False
        
        order.status = "cancelled"
        print(f"[{self.exchange_name}] Cancelled order {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Optional[DemoOrder]:
        """Get status of an order."""
        return self.orders.get(order_id)

    def list_orders(self, symbol: Optional[str] = None) -> list[DemoOrder]:
        """List all orders, optionally filtered by symbol."""
        orders = list(self.orders.values())
        if symbol:
            symbol_normalized = symbol.replace("/", "")
            orders = [o for o in orders if o.symbol == symbol_normalized]
        return orders

    def get_balance(self, asset: str) -> Decimal:
        """
        Get balance for an asset in demo account.
        
        For demo purposes, returns simulated balance.
        In production, would call actual exchange API.
        """
        # Simulated balances for demo
        demo_balances = {
            "USDT": Decimal("10000"),
            "BTC": Decimal("1"),
            "ETH": Decimal("10"),
        }
        return demo_balances.get(asset.upper(), Decimal("0"))

    def get_account_info(self) -> dict[str, Any]:
        """Get account information."""
        return {
            "exchange": self.exchange_name,
            "account_type": "testnet" if "testnet" in self.exchange_name else "demo",
            "total_balance_usdt": Decimal("10000"),
            "available_balance_usdt": Decimal("10000"),
            "positions_count": len([o for o in self.orders.values() if o.status == "filled"]),
        }

    def simulate_fill(
        self,
        order_id: str,
        filled_price: Decimal,
        filled_quantity: Optional[Decimal] = None,
        partial: bool = False,
    ) -> bool:
        """
        Simulate order fill (for testing purposes).
        
        Args:
            order_id: Order to fill
            filled_price: Fill price
            filled_quantity: Quantity filled (None = full quantity)
            partial: If True, mark as partial fill instead of full fill
            
        Returns: True if filled
        """
        if order_id not in self.orders:
            print(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        qty = filled_quantity or order.quantity
        
        if qty > order.quantity:
            print(f"Cannot fill {qty} of {order.quantity}")
            return False
        
        order.filled_quantity = qty
        order.price = filled_price
        order.status = "partial" if partial and qty < order.quantity else "filled"
        order.updated_at = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat().replace("+00:00", "Z")
        
        print(f"[{self.exchange_name}] Filled order {order_id}: {qty} @ {filled_price}")
        return True


class ExchangeConnector:
    """High-level connector for demo/testnet exchanges."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.demo_adapters: dict[str, DemoExchangeAdapter] = {}

    def connect_binance_testnet(
        self,
        api_key: str,
        api_secret: str,
    ) -> DemoExchangeAdapter:
        """Connect to Binance testnet."""
        adapter = DemoExchangeAdapter("binance_testnet", api_key, api_secret)
        self.demo_adapters["binance_testnet"] = adapter
        print("Connected to Binance testnet")
        return adapter

    def connect_bybit_demo(
        self,
        api_key: str,
        api_secret: str,
    ) -> DemoExchangeAdapter:
        """Connect to Bybit demo account."""
        adapter = DemoExchangeAdapter("bybit_demo", api_key, api_secret)
        self.demo_adapters["bybit_demo"] = adapter
        print("Connected to Bybit demo")
        return adapter

    def get_adapter(self, exchange: str) -> Optional[DemoExchangeAdapter]:
        """Get adapter for exchange."""
        return self.demo_adapters.get(exchange.lower())

    def execute_trade(
        self,
        exchange: str,
        signal_id: str,
        action: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
    ) -> dict[str, Any]:
        """
        Execute a trade via demo exchange.
        
        Returns: Trade execution result
        """
        adapter = self.get_adapter(exchange)
        if adapter is None:
            return {
                "success": False,
                "error": f"No adapter for {exchange}",
                "signal_id": signal_id,
            }
        
        try:
            order = adapter.place_order(
                symbol=symbol,
                side=action.upper(),
                quantity=quantity,
                price=price,
                order_type="LIMIT",
            )
            
            return {
                "success": True,
                "signal_id": signal_id,
                "order_id": order.order_id,
                "exchange": exchange,
                "symbol": symbol,
                "side": action,
                "quantity": str(quantity),
                "price": str(price),
                "status": order.status,
                "timestamp": order.created_at,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "signal_id": signal_id,
            }

    def validate_credentials(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
    ) -> bool:
        """
        Validate exchange credentials against demo account.
        
        For demo purposes, always returns True.
        In production, would test credentials via API.
        """
        print(f"Validating {exchange} credentials...")
        # In production: make a test API call
        # For now: assume valid if non-empty
        return bool(api_key and api_secret)
