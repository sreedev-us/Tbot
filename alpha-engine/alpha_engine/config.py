from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal

from dotenv import load_dotenv


@dataclass(slots=True)
class EngineConfig:
    backend_base_url: str
    default_exchange: str
    default_symbol: str
    default_timeframe: str
    polling_interval_seconds: int
    order_notional: Decimal
    strategy_name: str
    use_sandbox: bool
    stop_loss_pct: Decimal
    take_profit_pct: Decimal


def load_config() -> EngineConfig:
    load_dotenv()
    return EngineConfig(
        backend_base_url=os.getenv("TBOT_BACKEND_URL", "http://localhost:8080"),
        default_exchange=os.getenv("TBOT_DEFAULT_EXCHANGE", "bybit"),
        default_symbol=os.getenv("TBOT_DEFAULT_SYMBOL", "BTC/USDT"),
        default_timeframe=os.getenv("TBOT_DEFAULT_TIMEFRAME", "1m"),
        polling_interval_seconds=int(os.getenv("TBOT_POLLING_SECONDS", "30")),
        order_notional=Decimal(os.getenv("TBOT_ORDER_NOTIONAL", "100")),
        strategy_name=os.getenv("TBOT_STRATEGY_NAME", "mean-reversion-v1"),
        use_sandbox=os.getenv("TBOT_USE_SANDBOX", "true").lower() == "true",
        stop_loss_pct=Decimal(os.getenv("TBOT_STOP_LOSS_PCT", "0.5")),
        take_profit_pct=Decimal(os.getenv("TBOT_TAKE_PROFIT_PCT", "1.0")),
    )
