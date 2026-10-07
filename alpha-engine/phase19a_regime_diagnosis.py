"""
phase19a_regime_diagnosis.py

Analyzes the structural market state differences between:
- Period 1 (Mar-Jun 2026): The "signal failure" regime.
- Period 2 (Jul-Sep 2026): The "apparent edge" regime.

This script fetches OHLCV data from Binance/Bybit via ccxt, calculates
structural metrics, and performs a statistical comparison between regimes.
"""

import pandas as pd
import numpy as np
import os
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import ccxt
from pathlib import Path

ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT']
DATA_DIR = Path('data/phase19/1h')

def fetch_asset_ohlcv(exchange, symbol, timeframe, since_ms, end_ms):
    print(f"  Fetching {symbol} OHLCV...")
    all_candles = []
    current_since = since_ms
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not candles:
                break
            valid_candles = [c for c in candles if c[0] < end_ms]
            all_candles.extend(valid_candles)
            if len(valid_candles) < len(candles):
                break
            last_ts = candles[-1][0]
            if last_ts >= end_ms - 3600_000:
                break
            current_since = last_ts + 3600_000
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            time.sleep(1.0)
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.bybit({'enableRateLimit': True})
    since_ms = exchange.parse8601("2026-03-01T00:00:00Z")
    end_ms = exchange.parse8601("2026-09-22T00:00:00Z")
    
    df_list = []
    for asset in ASSETS:
        sym = f"{asset[:3]}/{asset[3:]}" if asset.endswith("USDT") else f"{asset[:-4]}/USDT"
        if asset == "DOGEUSDT": sym = "DOGE/USDT"
        elif asset == "AVAXUSDT": sym = "AVAX/USDT"
        
        file_path = DATA_DIR / f"{asset}_1h.csv"
        if file_path.exists():
            print(f"Loading {asset} from cache...")
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df = fetch_asset_ohlcv(exchange, sym, '1h', since_ms, end_ms)
            df.to_csv(file_path, index=False)
            
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df['asset'] = asset
        df['ret_1h'] = df['close'].pct_change()
        
        # 24h Realized Volatility (annualized)
        df['vol_24h'] = df['ret_1h'].rolling(24).std() * np.sqrt(24 * 365)
        
        # Trend strength: 24h EMA spread (fast vs slow EMA of returns or price)
        df['ema_fast'] = df['close'].ewm(span=12).mean()
        df['ema_slow'] = df['close'].ewm(span=24).mean()
        df['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
        
        # ADX Approximation (14-period on 1h)
        high = df['high']
        low = df['low']
        close = df['close']
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=14).mean() / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx_14h'] = dx.ewm(span=14).mean()

        df_list.append(df)
    
    return pd.concat(df_list)

def main():
    print("Loading data and computing individual metrics...")
    df = load_data()
    df.dropna(inplace=True)

    # Pivot returns to calculate cross-sectional metrics
    returns_df = df.pivot(columns='asset', values='ret_1h')
    
    print("Computing cross-sectional metrics...")
    # Cross-sectional dispersion (standard deviation of returns at each timestamp)
    dispersion = returns_df.std(axis=1)
    
    # Pairwise correlation (rolling 24h mean correlation)
    corr_to_btc = returns_df.rolling(24).corr(returns_df['BTCUSDT'])
    mean_corr_btc = corr_to_btc.drop(columns=['BTCUSDT']).mean(axis=1)
    
    # Market-wide metrics dataframe
    market_df = pd.DataFrame({
        'btc_vol_24h': df[df['asset'] == 'BTCUSDT']['vol_24h'],
        'btc_adx_14h': df[df['asset'] == 'BTCUSDT']['adx_14h'],
        'btc_ema_spread': df[df['asset'] == 'BTCUSDT']['ema_spread'],
        'dispersion': dispersion,
        'mean_corr_btc': mean_corr_btc,
        'total_volume': df.groupby('timestamp')['volume'].sum()
    })
    
    market_df.dropna(inplace=True)
    
    # Split into periods
    period1 = market_df[(market_df.index >= '2026-03-01') & (market_df.index < '2026-07-01')]
    period2 = market_df[(market_df.index >= '2026-07-01') & (market_df.index <= '2026-09-21')]
    
    print(f"\nPeriod 1 (Mar-Jun): {len(period1)} hours")
    print(f"Period 2 (Jul-Sep): {len(period2)} hours")
    
    metrics = ['btc_vol_24h', 'btc_adx_14h', 'btc_ema_spread', 'dispersion', 'mean_corr_btc', 'total_volume']
    
    print("\n--- Diagnostic Comparison ---")
    print(f"{'Metric':<20} | {'Period 1 Median':<18} | {'Period 2 Median':<18} | {'Shift':<10} | {'Mann-Whitney p-val'}")
    print("-" * 95)
    
    for m in metrics:
        p1_val = period1[m]
        p2_val = period2[m]
        
        m1 = p1_val.median()
        m2 = p2_val.median()
        
        # Absolute shift for interpretation
        if m in ['btc_ema_spread']:
            shift = (m2 - m1) * 10000 # bps
            unit = "bps"
        elif m in ['total_volume']:
            shift = (m2 - m1) / m1 * 100 # % change
            unit = "%"
        else:
            shift = (m2 - m1)
            unit = ""
            
        # Mann-Whitney U test for distributional difference
        stat, pval = stats.mannwhitneyu(p1_val, p2_val, alternative='two-sided')
        
        print(f"{m:<20} | {m1:<18.5f} | {m2:<18.5f} | {shift:>6.2f} {unit:<3} | {pval:.2e}")

    print("\n--- Absolute Distributions (Interquartile Ranges) ---")
    for m in metrics:
        p1_q25, p1_q75 = period1[m].quantile([0.25, 0.75])
        p2_q25, p2_q75 = period2[m].quantile([0.25, 0.75])
        print(f"{m:<20}")
        print(f"  P1 (Mar-Jun): [{p1_q25:.5f}, {p1_q75:.5f}]")
        print(f"  P2 (Jul-Sep): [{p2_q25:.5f}, {p2_q75:.5f}]")

if __name__ == "__main__":
    main()
