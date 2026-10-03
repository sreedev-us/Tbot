"""
phase9_cheap_diagnostic.py
==========================
Cheap forward-return diagnostic for Phase 9 timeframe research.
Evaluates the core mean-reversion hypothesis (EMA dislocation -> reversal)
across native timeframes (15m, 1h, 4h, 6h, 12h) without training XGBoost.

Methodology:
1. Load data for each timeframe (Mar 2026 -> Aug 1, 2026).
2. Calculate `dist_ema20 = (close - EMA20) / EMA20`.
3. Identify extreme dislocation events (e.g., bottom 10% = BUY signal, top 10% = SELL signal).
4. Measure forward returns at 12h, 24h, and 48h clock horizons.
   (Accounting for the differing number of bars per horizon).
5. Report cost-adjusted expectancy if we blindly traded those extremes.
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Horizons in CLOCK HOURS
HORIZONS_HOURS = [12, 24, 48]

def run_diagnostic_for_timeframe(tf_str: str, file_path: Path):
    if not file_path.exists():
        print(f"Skipping {tf_str} (file not found)")
        return
        
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Filter to Mar 12 - Aug 1 (original Phase 3-7 training+holdout period)
    # Actually, we can use the entire period up to Aug 1 to measure the general edge.
    start_t = pd.Timestamp("2026-03-12", tz="UTC")
    end_t = pd.Timestamp("2026-08-01", tz="UTC")
    df = df[(df["timestamp"] >= start_t) & (df["timestamp"] < end_t)].copy()
    
    if len(df) < 100:
        print(f"Skipping {tf_str} (insufficient data)")
        return
        
    # Determine bars per hour based on tf_str
    if tf_str.endswith("m"):
        mins = int(tf_str[:-1])
        bars_per_hour = 60 / mins
    elif tf_str.endswith("h"):
        hrs = int(tf_str[:-1])
        bars_per_hour = 1 / hrs
    else:
        raise ValueError(f"Unknown timeframe format: {tf_str}")
        
    # Calculate EMA20 and Dislocation
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["dist_ema_20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    
    # Calculate Rolling Deciles (using 100-bar rolling window to avoid lookahead bias)
    # rolling.rank(pct=True) gives percentile
    df["dist_pct"] = df["dist_ema_20"].rolling(100).rank(pct=True)
    
    # Generate naive signals
    # Bottom 10% = BUY (price is far below EMA)
    # Top 10% = SELL (price is far above EMA)
    df["signal"] = 0
    df.loc[df["dist_pct"] <= 0.10, "signal"] = 1
    df.loc[df["dist_pct"] >= 0.90, "signal"] = -1
    
    # Calculate forward returns for specific horizons
    results = {}
    for h_hours in HORIZONS_HOURS:
        h_bars = int(h_hours * bars_per_hour)
        if h_bars < 1:
            continue
            
        fwd_col = f"fwd_{h_hours}h_{h_bars}b"
        df[fwd_col] = df["close"].shift(-h_bars) / df["close"] - 1.0
        
        # Expectancy of BUY signals
        buy_mask = df["signal"] == 1
        sell_mask = df["signal"] == -1
        
        buy_ret = df.loc[buy_mask, fwd_col].mean() * 100
        sell_ret = df.loc[sell_mask, fwd_col].mean() * 100 # Raw return
        sell_ret_dir = -sell_ret # Directional return for SELL
        
        # Naive cost adjustment: 0.20% per trade (0.10% entry + 0.10% exit)
        cost_bps = 0.20
        
        buy_cost_adj = buy_ret - cost_bps
        sell_cost_adj = sell_ret_dir - cost_bps
        
        results[h_hours] = {
            "bars": h_bars,
            "buy_ret": buy_ret,
            "sell_ret_dir": sell_ret_dir,
            "buy_cost_adj": buy_cost_adj,
            "sell_cost_adj": sell_cost_adj,
            "n_buy": buy_mask.sum(),
            "n_sell": sell_mask.sum()
        }
        
    print(f"\n=== TIMEFRAME: {tf_str} (Bars/Hour: {bars_per_hour:.2f}) ===")
    print(f"Data points: {len(df)}")
    print(f"Signals: BUY={df['signal'].value_counts().get(1, 0)}, SELL={df['signal'].value_counts().get(-1, 0)}")
    
    print("\n  Horizon (Clock) | Horizon (Bars) | BUY Exp (Raw) | SELL Exp (Raw) | BUY Cost-Adj | SELL Cost-Adj")
    print("  " + "-" * 95)
    for h_hours in HORIZONS_HOURS:
        if h_hours not in results:
            continue
        res = results[h_hours]
        print(f"  {h_hours:>13}h | {res['bars']:>14} | {res['buy_ret']:>12.3f}% | {res['sell_ret_dir']:>13.3f}% | {res['buy_cost_adj']:>11.3f}% | {res['sell_cost_adj']:>12.3f}%")

def main():
    data_dir = Path("data/phase9")
    timeframes = ["15m", "1h", "4h", "6h", "12h"]
    
    print("PHASE 9: CHEAP FORWARD-RETURN DIAGNOSTIC")
    print("Evaluating core mean-reversion hypothesis (rolling EMA dislocation)")
    print("Cost assumption: 0.20% per round-trip trade")
    
    for tf in timeframes:
        file_path = data_dir / f"BTC_USDT_{tf}.csv"
        run_diagnostic_for_timeframe(tf, file_path)

if __name__ == "__main__":
    main()
