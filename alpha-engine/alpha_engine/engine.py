from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from alpha_engine.backend_client import submit_signal
from alpha_engine.config import load_config
from alpha_engine.exchanges import build_exchange
from alpha_engine.signals import evaluate_mean_reversion


def fetch_ohlcv_frame(exchange: Any, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    frame = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame


def run() -> None:
    config = load_config()
    exchange = build_exchange(config.default_exchange, config.use_sandbox)
    print(
        f"Alpha engine started for {config.default_exchange}:{config.default_symbol} "
        f"(sandbox={config.use_sandbox})"
    )

    while True:
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

            frame = fetch_ohlcv_frame(exchange, config.default_symbol)
            signal = evaluate_mean_reversion(
                df=frame,
                exchange=config.default_exchange,
                symbol=config.default_symbol,
                order_notional=config.order_notional,
                strategy_name=config.strategy_name,
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

        time.sleep(config.polling_interval_seconds)
