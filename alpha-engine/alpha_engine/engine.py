from __future__ import annotations

import signal
import time
from typing import Any

import pandas as pd
import requests
from decimal import Decimal
from datetime import UTC, datetime

from alpha_engine.backend_client import close_trade, get_open_trade, send_telemetry, submit_signal, sync_candles
from alpha_engine.config import load_config
from alpha_engine.exchanges import build_exchange
from alpha_engine.signals import evaluate_mean_reversion


def fetch_ohlcv_frame(exchange: Any, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    frame = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    ).copy()
    return frame.assign(
        timestamp=pd.to_datetime(frame["timestamp"], unit="ms", utc=True),
    )


def _candle_sync_payload(frame: pd.DataFrame, *, symbol: str, exchange: str, timeframe: str) -> dict[str, object]:
    latest = frame.tail(120)
    return {
        "asset": symbol,
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "candles": [
            {
                "timestamp": row.timestamp.isoformat().replace("+00:00", "Z"),
                "open": str(round(float(row.open), 8)),
                "high": str(round(float(row.high), 8)),
                "low": str(round(float(row.low), 8)),
                "close": str(round(float(row.close), 8)),
                "volume": str(round(float(row.volume), 8)),
            }
            for row in latest.itertuples(index=False)
        ],
    }


def _maybe_close_open_trade(config: Any, frame: pd.DataFrame) -> bool:
    open_trade = get_open_trade(
        config.backend_base_url,
        config.default_symbol,
        config.default_exchange,
    )
    if open_trade is None:
        return False

    current_price = Decimal(str(round(float(frame["close"].iloc[-1]), 8)))
    action = open_trade["action"]
    stop_loss_price = Decimal(str(open_trade["stopLossPrice"]))
    take_profit_price = Decimal(str(open_trade["takeProfitPrice"]))

    close_reason = None
    if action == "BUY":
        if current_price <= stop_loss_price:
            close_reason = "STOP_LOSS"
        elif current_price >= take_profit_price:
            close_reason = "TAKE_PROFIT"
    else:
        if current_price >= stop_loss_price:
            close_reason = "STOP_LOSS"
        elif current_price <= take_profit_price:
            close_reason = "TAKE_PROFIT"

    if close_reason is None:
        return True

    response = close_trade(
        config.backend_base_url,
        asset=config.default_symbol,
        exchange=config.default_exchange,
        exit_price=str(current_price),
        closed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        close_reason=close_reason,
    )
    print(f"Closed paper trade {response['tradeId']} with {response['outcome']} via {close_reason}.")
    return True


def run() -> None:
    config = load_config()
    exchange = build_exchange(config.default_exchange, config.use_sandbox)
    should_run = True

    def _handle_shutdown(_signum: int, _frame: object) -> None:
        nonlocal should_run
        should_run = False
        try:
            send_telemetry(
                config.backend_base_url,
                metric_group="engine",
                metric_name="offline",
                metric_value=Decimal("1"),
                metric_unit="count",
                tags=f"exchange={config.default_exchange},asset={config.default_symbol}",
            )
        except Exception as exc:
            print(f"Failed to send shutdown telemetry: {exc}")

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    print(
        f"Alpha engine started for {config.default_exchange}:{config.default_symbol} "
        f"(sandbox={config.use_sandbox})"
    )

    while should_run:
        try:
            control_response = requests.get(
                f"{config.backend_base_url}/api/v1/control/state",
                timeout=5,
            )
            control_response.raise_for_status()
            control_state = control_response.json()
            if control_state.get("engineHalted") is True:
                print("Engine halt is active. Skipping this cycle.")
                time.sleep(config.polling_interval_seconds)
                continue

            frame = fetch_ohlcv_frame(
                exchange,
                config.default_symbol,
                timeframe=config.default_timeframe,
            )
            sync_candles(
                config.backend_base_url,
                _candle_sync_payload(
                    frame,
                    symbol=config.default_symbol,
                    exchange=config.default_exchange,
                    timeframe=config.default_timeframe,
                ),
            )

            if _maybe_close_open_trade(config, frame):
                print("Open demo trade is being tracked. Waiting for exit conditions.")
                time.sleep(config.polling_interval_seconds)
                continue

            signal = evaluate_mean_reversion(
                df=frame,
                exchange=config.default_exchange,
                symbol=config.default_symbol,
                order_notional=config.order_notional,
                strategy_name=config.strategy_name,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
            )
            if signal is None:
                print("No signal generated for this cycle.")
            else:
                response = submit_signal(config.backend_base_url, signal)
                print(f"Submitted {signal.signal_id}: {response}")
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else "<no response>"
            print(f"Backend rejected signal: {body}")
        except Exception as exc:
            print(f"Engine cycle failed: {exc}")

        if should_run:
            time.sleep(config.polling_interval_seconds)

    print("Alpha engine shutdown complete.")
