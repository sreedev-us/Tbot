from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ai_lab.collect_market_data import collect_once


def _safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def train_model(csv_path: Path, version: str, model_type: str, backtest: bool) -> None:
    command = [
        sys.executable,
        "train_ai_model.py",
        "--ohlcv",
        str(csv_path),
        "--version",
        version,
        "--model-type",
        model_type,
    ]
    if backtest:
        command.append("--backtest")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automate data collection and model training for the Tbot demo market."
    )
    parser.add_argument("--exchange", default="bybit")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--bootstrap-days", type=int, default=180)
    parser.add_argument("--version", default="local-v1.0.0")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--min-rows", type=int, default=5000)
    parser.add_argument("--train", action="store_true", help="Train after each successful collection.")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest during training.")
    parser.add_argument("--poll", action="store_true", help="Keep collecting live closed candles.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--train-every-new-rows", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    csv_path = Path(args.output or f"data/demo/{_safe_symbol_name(args.symbol)}_{args.timeframe}.csv")
    rows_at_last_train = 0

    while True:
        result = collect_once(
            exchange_name=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            output=csv_path,
            bootstrap_days=args.bootstrap_days,
            limit=1000,
        )
        print(result)

        total_rows = int(result["total_rows"])
        enough_new_rows = total_rows - rows_at_last_train >= args.train_every_new_rows
        can_train = bool(result["quality_passed"]) and total_rows >= args.min_rows

        if args.train and can_train and (rows_at_last_train == 0 or enough_new_rows):
            train_model(csv_path, args.version, args.model_type, args.backtest)
            rows_at_last_train = total_rows
        elif args.train and not can_train:
            print(f"Skipping training until quality passes and at least {args.min_rows} rows exist.")

        if not args.poll:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
