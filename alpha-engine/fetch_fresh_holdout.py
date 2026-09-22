#!/usr/bin/env python3
"""
Fetch truly fresh 1h BTC/USDT data: Sep 6 -> Sep 22, 2026.
This data has NEVER been used in any development, training, or diagnostic.
It is the first genuinely independent test period.
"""
import pandas as pd
import ccxt
from pathlib import Path

OUT_PATH = Path("data/live/BTC_USDT_1h_fresh.csv")

def main():
    exchange = ccxt.bybit({"enableRateLimit": True})

    # Sep 6, 2026 09:05 UTC (one bar after training cutoff) -> now
    since_ms = exchange.parse8601("2026-09-06T10:00:00Z")
    print(f"Fetching fresh 1h BTC/USDT from 2026-09-06T10:00Z onwards...")

    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv("BTC/USDT", "1h", since=since_ms, limit=500)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        since_ms = last_ts + 3600_000

        import time
        now_ms = exchange.milliseconds()
        if last_ts >= now_ms - 3600_000:
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Fetched {len(df)} candles")
    print(f"  Start: {df['timestamp'].iloc[0]}")
    print(f"  End:   {df['timestamp'].iloc[-1]}")
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    print(f"  Days:  {days:.1f}")
    print(f"  Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
