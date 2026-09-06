#!/usr/bin/env python3
"""
Fetch live historical OHLCV data from Bybit (public, no API key needed)
and save it as a training-ready CSV.

Usage:
  python fetch_live_data.py --symbol BTC/USDT --timeframe 1m --days 180 --output data/live/BTC_USDT_1m_live.csv
"""

import argparse
import time
import os
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd


def fetch_all_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """Paginate through historical candles until we reach `until_ms`."""
    all_candles = []
    current_ms = since_ms
    tf_seconds = exchange.parse_timeframe(timeframe)
    tf_ms = tf_seconds * 1000

    print(f"Fetching {symbol} {timeframe} candles from {exchange.id}...")
    print(f"  From: {datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)}")
    print(f"  To:   {datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)}")

    while current_ms < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_ms, limit=batch_size)
        except Exception as exc:
            print(f"  Error fetching batch at {current_ms}: {exc}. Retrying in 5s...")
            time.sleep(5)
            continue

        if not batch:
            break

        all_candles.extend(batch)
        last_ts = batch[-1][0]
        progress_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"  Fetched {len(all_candles):,} candles so far... up to {progress_dt}")

        if last_ts >= until_ms:
            break

        # If we got fewer candles than expected, we've reached the end
        if len(batch) < max(batch_size // 2, 10):
            break

        current_ms = last_ts + tf_ms
        time.sleep(0.3)  # Be polite to the exchange API

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Filter strictly within range
    since_dt = pd.Timestamp(since_ms, unit="ms", tz="UTC")
    until_dt = pd.Timestamp(until_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= since_dt) & (df["timestamp"] <= until_dt)].reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch live historical OHLCV data for model training")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", default="1m", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=180, help="Number of days of history to fetch")
    parser.add_argument("--output", default="data/live/BTC_USDT_1m_live.csv", help="Output CSV path")
    parser.add_argument("--exchange", default="bybit", help="Exchange to fetch from (default: bybit)")
    args = parser.parse_args()

    # Build exchange (no API keys needed for public OHLCV)
    exchange_class = getattr(ccxt, args.exchange)
    exchange = exchange_class({"enableRateLimit": True})

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    df = fetch_all_candles(exchange, args.symbol, args.timeframe, since_ms, now_ms)

    if df.empty:
        print("No data fetched. Check symbol/exchange and try again.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Done! Saved {len(df):,} candles to: {args.output}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"File size:  {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
