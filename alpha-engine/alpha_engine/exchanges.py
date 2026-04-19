from __future__ import annotations

import os

import ccxt


def build_exchange(exchange_name: str, use_sandbox: bool):
    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls(
        {
            "apiKey": os.getenv(f"{exchange_name.upper()}_API_KEY", ""),
            "secret": os.getenv(f"{exchange_name.upper()}_API_SECRET", ""),
            "enableRateLimit": True,
        }
    )
    if use_sandbox and exchange_name in {"binance", "bybit"}:
        exchange.set_sandbox_mode(True)
    return exchange
