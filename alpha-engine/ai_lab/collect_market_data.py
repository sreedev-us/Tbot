from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from ai_lab.data_quality import load_ohlcv_csv, normalize_ohlcv, validate_ohlcv


def _build_exchange(exchange_name: str) -> Any:
    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def _timeframe_ms(exchange: Any, timeframe: str) -> int:
    return int(exchange.parse_timeframe(timeframe) * 1000)


def _safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def fetch_missing_candles(
    *,
    exchange: Any,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int,
) -> pd.DataFrame:
    rows: list[list[float]] = []
    cursor = since_ms
    step_ms = _timeframe_ms(exchange, timeframe)

    while cursor < until_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not candles:
            break

        rows.extend(candles)
        next_cursor = int(candles[-1][0]) + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    closed_before = pd.to_datetime(until_ms, unit="ms", utc=True)
    return normalize_ohlcv(frame.loc[frame["timestamp"] < closed_before].copy())


def collect_once(
    *,
    exchange_name: str,
    symbol: str,
    timeframe: str,
    output: Path,
    bootstrap_days: int,
    limit: int,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    exchange = _build_exchange(exchange_name)
    existing = load_ohlcv_csv(output)

    now_ms = exchange.milliseconds()
    interval_ms = _timeframe_ms(exchange, timeframe)
    until_ms = now_ms - interval_ms

    if existing.empty:
        since_dt = datetime.now(UTC) - timedelta(days=bootstrap_days)
        since_ms = int(since_dt.timestamp() * 1000)
    else:
        since_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000) + interval_ms

    fetched = fetch_missing_candles(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=limit,
    )
    frames = [frame for frame in [existing, fetched] if not frame.empty]
    combined = normalize_ohlcv(pd.concat(frames, ignore_index=True)) if frames else existing
    combined.to_csv(output, index=False)

    report = validate_ohlcv(combined, timeframe)
    report_path = output.with_suffix(".quality.json")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    return {
        "output": str(output),
        "new_rows": len(fetched),
        "total_rows": len(combined),
        "quality_report": str(report_path),
        "quality_passed": report.passed,
        "last_timestamp": report.last_timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect quality OHLCV data for Tbot AI modelling.")
    parser.add_argument("--exchange", default="bybit", help="CCXT exchange id, e.g. bybit, binance, kraken.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--bootstrap-days", type=int, default=180)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--poll", action="store_true", help="Keep collecting new closed candles.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = Path(args.output or f"data/demo/{_safe_symbol_name(args.symbol)}_{args.timeframe}.csv")

    while True:
        result = collect_once(
            exchange_name=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            output=output,
            bootstrap_days=args.bootstrap_days,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        if not args.poll:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
