import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_engine.regime_detector import RegimeDetector
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.breakout import BreakoutStrategy
from strategies.t3_donchian import DonchianStrategy
from strategies.t4_ema_slope import EMASlopeStrategy
from strategies.t5_multi_tf import MultiTFStrategy

logging.basicConfig(level=logging.WARNING)

def calculate_forward_returns(df: pd.DataFrame, state_df: pd.DataFrame, strategy: Any, horizons: list[int], fee_pct: float = 0.20) -> list[dict[str, Any]]:
    close = df["close"].values
    n = len(df)
    results = []
    
    for i in range(55, n - max(horizons) - 1):
        window_df = df.iloc[: i + 1]
        sig = strategy.signal(window_df)
        
        if sig != 0:
            trend = state_df["trend"].iloc[i]
            entry_price = close[i]
            
            for h in horizons:
                exit_price = close[i + h]
                raw_fwd_ret = sig * (exit_price - entry_price) / entry_price * 100.0
                cost_adj_ret = raw_fwd_ret - fee_pct
                
                results.append({
                    "strategy": strategy.name,
                    "trend": trend,
                    "horizon": f"{h}h",
                    "raw_return": raw_fwd_ret,
                    "cost_adj_return": cost_adj_ret,
                    "signal": sig
                })
                
    return results

def main():
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading data from {data_path}...")
    raw_df = pd.read_csv(data_path)

    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Dataset: {len(full_df)} bars ({full_df['timestamp'].iloc[0].date()} to {full_df['timestamp'].iloc[-1].date()})")

    detector = RegimeDetector()
    state_df = detector.detect_series(full_df)

    strategies = [
        MeanReversionStrategy(),
        TrendFollowingStrategy(), # T1
        BreakoutStrategy(), # T2
        DonchianStrategy(), # T3
        EMASlopeStrategy(), # T4
        MultiTFStrategy(), # T5
    ]

    horizons = [1, 4, 12, 24]
    
    all_results = []
    for strat in strategies:
        print(f"Calculating forward returns for {strat.name}...")
        res = calculate_forward_returns(full_df, state_df, strat, horizons)
        all_results.extend(res)
    
    df_results = pd.DataFrame(all_results)
    if df_results.empty:
        print("No signals generated.")
        return
        
    print("\n" + "=" * 120)
    print("  PHASE 6 MULTI-HORIZON DIAGNOSTIC: STRATEGY SIGNAL EXPECTANCY BY REGIME")
    print("=" * 120)
    
    # Custom sort for horizons
    horizon_order = {f"{h}h": i for i, h in enumerate(horizons)}
    
    for strat in strategies:
        s_name = strat.name
        print(f"\n--- {s_name.upper()} ---")
        strat_df = df_results[df_results["strategy"] == s_name].copy()
        
        if strat_df.empty:
            print("No signals.")
            continue
            
        strat_df["h_order"] = strat_df["horizon"].map(horizon_order)
        
        for trend in ["range", "uptrend", "downtrend", "neutral"]:
            trend_df = strat_df[strat_df["trend"] == trend]
            if trend_df.empty:
                continue
                
            print(f"\n[{trend.upper()}]")
            
            # Print table with horizons
            header = "| Horizon | Signals | Longs | Shorts | Raw Ret | Cost-Adj Ret | Hit Rate | Long Raw Ret | Short Raw Ret |"
            print(header)
            print("|---|---|---|---|---|---|---|---|---|")
            
            for h in horizons:
                hz = f"{h}h"
                h_df = trend_df[trend_df["horizon"] == hz]
                if h_df.empty:
                    continue
                    
                total_sigs = len(h_df)
                longs = len(h_df[h_df["signal"] == 1])
                shorts = len(h_df[h_df["signal"] == -1])
                
                raw_ret = h_df["raw_return"].mean()
                cost_ret = h_df["cost_adj_return"].mean()
                hit_rate = (h_df["raw_return"] > 0).mean() * 100.0
                
                long_ret = h_df[h_df["signal"] == 1]["raw_return"].mean() if longs > 0 else 0.0
                short_ret = h_df[h_df["signal"] == -1]["raw_return"].mean() if shorts > 0 else 0.0
                
                print(f"| {hz:<7} | {total_sigs:>7} | {longs:>5} | {shorts:>6} | {raw_ret:>+6.2f}% | {cost_ret:>+11.2f}% | {hit_rate:>7.1f}% | {long_ret:>+11.2f}% | {short_ret:>+12.2f}% |")

if __name__ == "__main__":
    main()
