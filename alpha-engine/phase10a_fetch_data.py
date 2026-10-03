"""
phase10a_fetch_data.py
======================
Phase 10A: Pair Universe Discovery

Fetches native 1h exchange candles for 8 major assets.
Performs timestamp alignment and data-quality audits.
Calculates statistical metrics (correlation, ADF, half-life, subperiod stability)
for all 28 unique pairs to identify candidates for Phase 10C.

Period: 2026-03-01 to 2026-09-21 (strict boundary before Sep 22).
"""
import time
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def fetch_asset(exchange, symbol, timeframe, since_ms, end_ms):
    print(f"Fetching {symbol} ({timeframe})...")
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

def calc_halflife(spread):
    """Calculate half-life of mean reversion."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    # align
    spread_lag, spread_diff = spread_lag.align(spread_diff, join='inner')
    if len(spread_lag) < 10:
        return np.nan
        
    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, X).fit()
    lambda_val = model.params.iloc[1]
    
    if lambda_val >= 0:
        return np.inf  # Not mean-reverting
        
    half_life = -np.log(2) / lambda_val
    return half_life

def analyze_pair(df_a, df_b):
    """
    Analyzes cointegration and mean-reversion metrics between two price series.
    Returns beta, full-period ADF p-value, half-life, spread volatility.
    """
    # Align
    df = pd.DataFrame({"A": np.log(df_a), "B": np.log(df_b)}).dropna()
    if len(df) < 100:
        return None
        
    # Estimate static beta via OLS (Engle-Granger step 1)
    # log(P_A) = alpha + beta * log(P_B) + error
    X = sm.add_constant(df["B"])
    model = sm.OLS(df["A"], X).fit()
    beta = model.params["B"]
    
    # Spread = error
    spread = df["A"] - beta * df["B"]
    
    # ADF Test on spread
    adf_result = adfuller(spread, maxlag=1)
    p_value = adf_result[1]
    
    # Half-life
    hl = calc_halflife(spread)
    
    # Spread Volatility
    spread_vol = spread.std()
    
    return {
        "beta": beta,
        "adf_p": p_value,
        "half_life": hl,
        "spread_vol": spread_vol,
        "spread_series": spread
    }

def main():
    exchange = ccxt.bybit({"enableRateLimit": True})
    assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]
    symbols = [f"{a}/USDT" for a in assets]
    timeframe = "1h"
    
    start_date = "2026-03-01T00:00:00Z"
    end_date = "2026-09-22T00:00:00Z" # strict Sep 22 boundary
    
    since_ms = exchange.parse8601(start_date)
    end_ms = exchange.parse8601(end_date)
    
    out_dir = Path("data/phase10")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Data
    print("\n=== 1. FETCHING NATIVE 1H DATA ===")
    asset_dfs = {}
    for sym in symbols:
        asset_name = sym.split("/")[0]
        df = fetch_asset(exchange, sym, timeframe, since_ms, end_ms)
        df.set_index("timestamp", inplace=True)
        asset_dfs[asset_name] = df["close"]
        
    # 2. Timestamp Alignment & Missing Data Audit
    print("\n=== 2. TIMESTAMP ALIGNMENT & DATA AUDIT ===")
    full_idx = pd.date_range(start=start_date, end=end_date, freq="1h", tz="UTC")[:-1] # drop end_date boundary exact match if it exists
    
    aligned_df = pd.DataFrame(index=full_idx)
    for a in assets:
        aligned_df[a] = asset_dfs[a]
        
    print(f"Total expected chronological 1h bars: {len(full_idx)}")
    for a in assets:
        missing = aligned_df[a].isna().sum()
        pct_missing = (missing / len(full_idx)) * 100
        print(f"  {a:<5}: {len(asset_dfs[a]):>4} bars fetched | Missing: {missing:>4} ({pct_missing:.2f}%)")
        
    # Forward fill small gaps
    aligned_df = aligned_df.ffill().dropna()
    print(f"\nAligned valid bars (post-ffill, no NaNs): {len(aligned_df)}")
    
    # 3. Analyze all 28 pairs
    print("\n=== 3. PAIR UNIVERSE DISCOVERY (28 UNIQUE PAIRS) ===")
    pairs = list(itertools.combinations(assets, 2))
    
    # Subperiod boundaries (roughly 2.2 months each)
    total_bars = len(aligned_df)
    sub1 = aligned_df.iloc[:total_bars//3]
    sub2 = aligned_df.iloc[total_bars//3 : 2*total_bars//3]
    sub3 = aligned_df.iloc[2*total_bars//3:]
    
    print(f"Subperiods: P1={len(sub1)} bars, P2={len(sub2)} bars, P3={len(sub3)} bars")
    
    results = []
    
    for a, b in pairs:
        # Full period
        full = analyze_pair(aligned_df[a], aligned_df[b])
        if full is None: continue
        
        # Subperiods
        p1 = analyze_pair(sub1[a], sub1[b])
        p2 = analyze_pair(sub2[a], sub2[b])
        p3 = analyze_pair(sub3[a], sub3[b])
        
        if p1 is None or p2 is None or p3 is None:
            continue
            
        corr = aligned_df[a].pct_change().corr(aligned_df[b].pct_change())
        
        # Stability check: does ADF p-value stay below 0.1 in all subperiods?
        stable = p1['adf_p'] < 0.1 and p2['adf_p'] < 0.1 and p3['adf_p'] < 0.1
        
        results.append({
            "Pair": f"{a}-{b}",
            "Corr": corr,
            "Beta": full["beta"],
            "ADF (Full)": full["adf_p"],
            "HL (Full)": full["half_life"],
            "ADF (P1)": p1["adf_p"],
            "ADF (P2)": p2["adf_p"],
            "ADF (P3)": p3["adf_p"],
            "Stable": stable
        })
        
    # 4. Report
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by="ADF (Full)")
    
    print("\n--- TOP CANDIDATE PAIRS (Sorted by Full-Period ADF p-value) ---")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Save the aligned dataset for Phase 10C
    out_csv = out_dir / "aligned_close_1h.csv"
    aligned_df.to_csv(out_csv)
    print(f"\nSaved aligned data to {out_csv}")


if __name__ == "__main__":
    main()
