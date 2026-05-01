from __future__ import annotations

from decimal import Decimal

import requests

from alpha_engine.signals import RawSignal


def submit_signal(base_url: str, signal: RawSignal) -> dict:
    response = requests.post(
        f"{base_url}/api/v1/execute",
        json=signal.to_payload(),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def sync_candles(base_url: str, payload: dict) -> None:
    response = requests.post(
        f"{base_url}/api/v1/market-data/candles",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()


def get_open_trade(base_url: str, asset: str, exchange: str) -> dict | None:
    response = requests.get(
        f"{base_url}/api/v1/paper-trades/open",
        params={"asset": asset, "exchange": exchange.upper()},
        timeout=10,
    )
    if response.status_code == 204:
        return None
    response.raise_for_status()
    return response.json()


def close_trade(
    base_url: str,
    *,
    asset: str,
    exchange: str,
    exit_price: str,
    closed_at: str,
    close_reason: str,
) -> dict:
    response = requests.post(
        f"{base_url}/api/v1/paper-trades/close",
        json={
            "asset": asset,
            "exchange": exchange.upper(),
            "exitPrice": exit_price,
            "closedAt": closed_at,
            "closeReason": close_reason,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def send_telemetry(
    base_url: str,
    *,
    metric_group: str,
    metric_name: str,
    metric_value: Decimal,
    metric_unit: str,
    tags: str,
) -> None:
    response = requests.post(
        f"{base_url}/api/v1/telemetry",
        json={
            "metricGroup": metric_group,
            "metricName": metric_name,
            "metricValue": str(metric_value),
            "metricUnit": metric_unit,
            "tags": tags,
        },
        timeout=5,
    )
    response.raise_for_status()
