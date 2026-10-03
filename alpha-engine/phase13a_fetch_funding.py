"""
phase13a_fetch_funding.py
=========================
Fetches historical funding rate data for the Phase 13 diagnostic.
Matches the existing 8-asset universe (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX)
against USDT perpetuals on Bybit.
Period: 2026-03-01 to 2026-09-21.
"""
import time
from pathlib import Path
import pandas as pd
import ccxt

def fetch_funding(exchange, symbol, since_ms, end_ms):
    print(f"Fetching funding history for {symbol}...")
    all_data = []
    current_since = since_ms
    
    while True:
        try:
            # Bybit fetch_funding_rate_history typically returns a list of dictionaries
            # Important: the limit is often 200 on bybit
            data = exchange.fetch_funding_rate_history(symbol, since=current_since, limit=200)
            if not data:
                break
                
            valid_data = [d for d in data if d['timestamp'] < end_ms]
            all_data.extend(valid_data)
            
            if len(valid_data) < len(data):
                break # Hit the end boundary
                
            last_ts = data[-1]['timestamp']
            if last_ts >= end_ms - 1000:
                break
                
            # ccxt pagination
            current_since = last_ts + 1000 # add 1 second to avoid duplicates
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            time.sleep(1.0)
            
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # Bybit funding rate usually comes as a float in 'fundingRate'
        df = df[["timestamp", "fundingRate"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
    return df

def main():
    exchange = ccxt.bybit({"enableRateLimit": True})
    assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]
    
    # On Bybit, USDT perps are formatted like "BTC/USDT:USDT" in ccxt 
    # to distinguish from spot "BTC/USDT"
    symbols = [f"{a}/USDT:USDT" for a in assets]
    
    start_date = "2026-03-01T00:00:00Z"
    end_date = "2026-09-22T00:00:00Z" # strict Sep 22 boundary
    
    since_ms = exchange.parse8601(start_date)
    end_ms = exchange.parse8601(end_date)
    
    out_dir = Path("data/phase13")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    funding_dfs = {}
    
    for asset, sym in zip(assets, symbols):
        df = fetch_funding(exchange, sym, since_ms, end_ms)
        if not df.empty:
            df.columns = [asset]
            funding_dfs[asset] = df
            print(f"  Fetched {len(df)} records for {asset}")
        else:
            print(f"  WARNING: No data for {sym}")
            
    # Combine into a single dataframe
    if funding_dfs:
        combined_funding = pd.concat(funding_dfs.values(), axis=1)
        # Forward fill since funding rates are set for 8h periods
        combined_funding = combined_funding.resample("1h").ffill()
        
        out_csv = out_dir / "funding_rates_1h.csv"
        combined_funding.to_csv(out_csv)
        print(f"\nSaved aligned funding data to {out_csv}")
        print(f"Range: {combined_funding.index[0]} to {combined_funding.index[-1]}")
    else:
        print("Failed to fetch any funding data.")

if __name__ == "__main__":
    main()
