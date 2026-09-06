#!/usr/bin/env python3
"""
Fetch fresh 1h OHLCV for the holdout window.

The training data ends at 2026-09-06 09:00 UTC.
We fetch from that exact point forward (+38 days) to avoid any overlap.

This data MUST NOT be used for any tuning or feature selection.
It is the final holdout for the frozen 1h ATR 2/1 @ 0.85 strategy.
"""
import time
import os
from datetime import datetime, timezone

import ccxt
import pandas as pd

# ── FROZEN PARAMETERS ──────────────────────────────────────────────────────────
# Do not change these. They were locked before the holdout data was fetched.
TRAINING_END_UTC = datetime(2026, 9, 6, 9, 0, 0, tzinfo=timezone.utc)
HOLDOUT_DAYS     = 38
OUTPUT_PATH      = "data/live/BTC_USDT_1h_holdout.csv"
SYMBOL           = "BTC/USDT"
TIMEFRAME        = "1h"
EXCHANGE_ID      = "bybit"
# ───────────────────────────────────────────────────────────────────────────────

def fetch_candles(exchange, since_ms, until_ms, batch_size=1000):
    all_candles = []
    current_ms = since_ms
    tf_ms = exchange.parse_timeframe(TIMEFRAME) * 1000

    print(f"Fetching holdout {SYMBOL} {TIMEFRAME} from {EXCHANGE_ID}...")
    print(f"  From: {datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)}")
    print(f"  To:   {datetime.fromtimestamp(until_ms / 1000, tz=timezone.utc)}")

    while current_ms < until_ms:
        try:
            batch = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME,
                                         since=current_ms, limit=batch_size)
        except Exception as exc:
            print(f"  Error: {exc}. Retrying in 5s...")
            time.sleep(5)
            continue

        if not batch:
            break

        all_candles.extend(batch)
        last_ts = batch[-1][0]
        print(f"  Fetched {len(all_candles):,} candles... "
              f"up to {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)}")

        if last_ts >= until_ms:
            break
        if len(batch) < max(batch_size // 2, 10):
            break

        current_ms = last_ts + tf_ms
        time.sleep(0.3)

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    since_dt = pd.Timestamp(since_ms, unit="ms", tz="UTC")
    until_dt = pd.Timestamp(until_ms, unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= since_dt) & (df["timestamp"] <= until_dt)]
    return df.reset_index(drop=True)


def main():
    # Start exactly 1 bar after training ends (no overlap)
    since_ms = int((TRAINING_END_UTC.timestamp() + 3600) * 1000)
    until_ms = int((TRAINING_END_UTC.timestamp() + HOLDOUT_DAYS * 86400) * 1000)

    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    df = fetch_candles(exchange, since_ms, until_ms)

    if df.empty:
        print("ERROR: No data fetched. Check exchange connectivity.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df):,} holdout candles to: {OUTPUT_PATH}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    actual_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f"Actual duration: {actual_days:.1f} days")


if __name__ == "__main__":
    main()
