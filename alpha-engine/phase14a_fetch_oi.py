"""
phase14a_fetch_oi.py
====================
Fetches historical Open Interest (OI) data for the Phase 14A diagnostic.
Matches the existing 8-asset universe (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX).
"""
import time
from pathlib import Path
import pandas as pd
import ccxt

def fetch_oi_history(exchange, symbol, since_ms, end_ms):
    print(f"Fetching OI history for {symbol}...")
    all_data = []
    current_since = since_ms
    
    while True:
        try:
            # fetch_open_interest_history typically uses timeframe for bybit
            data = exchange.fetch_open_interest_history(
                symbol, 
                timeframe='1h', 
                since=current_since, 
                limit=200
            )
            
            if not data:
                break
                
            valid_data = [d for d in data if d['timestamp'] < end_ms]
            all_data.extend(valid_data)
            
            if len(valid_data) < len(data):
                break # hit end bound
                
            last_ts = data[-1]['timestamp']
            if last_ts >= end_ms - 1000:
                break
                
            current_since = last_ts + 1000
            time.sleep(0.2)
        except Exception as e:
            # Sometimes specific symbols or timeframes aren't supported
            print(f"Error fetching {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # We need the base volume or openInterest value
        # ccxt returns 'openInterestValue' or 'openInterestAmount'
        if 'openInterestAmount' in df.columns:
            oi_col = 'openInterestAmount'
        elif 'openInterestValue' in df.columns:
            oi_col = 'openInterestValue'
        else:
            # try 'info' dict if ccxt doesn't parse it
            oi_col = df.columns[-1]
            
        df = df[["timestamp", oi_col]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.rename(columns={oi_col: "OI"}, inplace=True)
        df.set_index("timestamp", inplace=True)
    return df

def main():
    exchange = ccxt.bybit({"enableRateLimit": True})
    assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]
    symbols = [f"{a}/USDT:USDT" for a in assets]
    
    start_date = "2026-03-01T00:00:00Z"
    end_date = "2026-09-22T00:00:00Z"
    
    since_ms = exchange.parse8601(start_date)
    end_ms = exchange.parse8601(end_date)
    
    out_dir = Path("data/phase14")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    oi_dfs = {}
    
    for asset, sym in zip(assets, symbols):
        df = fetch_oi_history(exchange, sym, since_ms, end_ms)
        if not df.empty:
            df.columns = [asset]
            oi_dfs[asset] = df
            print(f"  Fetched {len(df)} records for {asset}")
        else:
            print(f"  WARNING: No data for {sym}")
            
    if oi_dfs:
        combined_oi = pd.concat(oi_dfs.values(), axis=1)
        # Resample to ensure perfect 1h alignment
        combined_oi = combined_oi.resample("1h").ffill()
        
        out_csv = out_dir / "oi_1h.csv"
        combined_oi.to_csv(out_csv)
        print(f"\nSaved aligned OI data to {out_csv}")
    else:
        print("Failed to fetch any OI data.")

if __name__ == "__main__":
    main()
