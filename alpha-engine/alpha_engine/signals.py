from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pandas as pd


@dataclass(slots=True)
class RawSignal:
    signal_id: str
    correlation_id: str
    asset: str
    exchange: str
    action: str
    confidence: Decimal
    requested_notional: Decimal
    strategy_name: str
    generated_at: datetime

    def to_payload(self) -> dict[str, str]:
        return {
            "signalId": self.signal_id,
            "correlationId": self.correlation_id,
            "asset": self.asset,
            "exchange": self.exchange.upper(),
            "action": self.action,
            "confidence": str(self.confidence),
            "requestedNotional": str(self.requested_notional),
            "strategyName": self.strategy_name,
            "generatedAt": self.generated_at.isoformat().replace("+00:00", "Z"),
        }


def evaluate_mean_reversion(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    order_notional: Decimal,
    strategy_name: str,
) -> RawSignal | None:
    closes = df["close"]
    fast_ma = closes.tail(5).mean()
    slow_ma = closes.tail(20).mean()
    current_price = closes.iloc[-1]
    edge = (slow_ma - current_price) / slow_ma

    if edge <= 0.0025:
        return None

    confidence = min(max(edge * 10, 0.55), 0.99)
    return RawSignal(
        signal_id=f"sig-{uuid4().hex[:16]}",
        correlation_id=uuid4().hex[:16],
        asset=symbol,
        exchange=exchange,
        action="BUY" if current_price < fast_ma else "SELL",
        confidence=Decimal(str(round(confidence, 4))),
        requested_notional=order_notional,
        strategy_name=strategy_name,
        generated_at=datetime.now(UTC),
    )
