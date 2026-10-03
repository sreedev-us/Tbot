"""
phase9_fetch_data.py
====================
Fetches native exchange candles for Phase 9 timeframe research.
Timeframes: 15m, 1h, 4h, 6h, 12h.

Period: 2026-03-01 to 2026-09-27 (to cover the full range of previous experiments,
including Sep 22+).
"""
import time
from pathlib import Path
import pandas as pd
import ccxt

def fetch_timeframe(exchange, symbol, timeframe, since_ms, end_ms):
    print(f"Fetching {timeframe} data...")
    all_candles = []
    current_since = since_ms
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not candles:
                break
                
            # Filter candles that exceed end_ms
            valid_candles = [c for c in candles if c[0] <= end_ms]
            all_candles.extend(valid_candles)
            
            if len(valid_candles) < len(candles):
                break # We reached the end
                
            last_ts = candles[-1][0]
            if last_ts >= end_ms:
                break
                
            # CCXT fetch_ohlcv pagination
            timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
            current_since = last_ts + timeframe_ms
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {timeframe}: {e}")
            time.sleep(1.0)
            
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def main():
    exchange = ccxt.bybit({"enableRateLimit": True})
    symbol = "BTC/USDT"
    timeframes = ["15m", "1h", "4h", "6h", "12h"]
    
    start_date = "2026-03-01T00:00:00Z"
    end_date = "2026-09-27T15:00:00Z"
    
    since_ms = exchange.parse8601(start_date)
    end_ms = exchange.parse8601(end_date)
    
    out_dir = Path("data/phase9")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for tf in timeframes:
        if tf not in exchange.timeframes:
            print(f"Warning: {tf} not natively supported by {exchange.id}. Will skip.")
            continue
            
        df = fetch_timeframe(exchange, symbol, tf, since_ms, end_ms)
        out_path = out_dir / f"BTC_USDT_{tf}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} candles for {tf} to {out_path}")
        print(f"  Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

if __name__ == "__main__":
    main()
