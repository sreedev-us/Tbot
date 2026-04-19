from __future__ import annotations

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
