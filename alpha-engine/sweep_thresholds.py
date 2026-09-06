#!/usr/bin/env python3
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model, evaluate_window

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    data_path = Path("data/live/BTC_USDT_1m_live.csv")
    model_path = Path("models/local-benchmark-b")

    booster, scaler, feature_columns = load_model(model_path)

    raw_df = pd.read_csv(data_path)
    config = TrainingDataConfig()
    generator = TrainingDataGenerator(config)
    full_df = generator.generate_training_data(raw_df)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    t0 = full_df["timestamp"].iloc[0]
    windows = [
        ("Q1", t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45),  "trending"),
        ("Q2", t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90),  "choppy"),
        ("Q3", t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135), "volatile"),
        ("Q4", t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=179), "recovery"),
    ]

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP REPORT (MODEL B)")
    print("=" * 100)
    header = f"{'Threshold':<12} {'Trades':>6} {'Trades/Day':>11} {'WinRate':>8} {'ProfFact':>9} {'Sharpe':>7} {'MaxDD':>7} {'CumRet%':>8}"
    print(header)
    print("-" * 100)

    for thresh in thresholds:
        all_results = []
        for period_name, start_ts, end_ts, regime in windows:
            window_df = full_df[
                (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
            ].copy()
            
            if len(window_df) < 200: continue
            
            result = evaluate_window(
                booster, scaler, feature_columns,
                window_df, "B", period_name, regime,
                confidence_threshold=thresh,
                fee_pct=0.10
            )
            if result:
                all_results.append(result)
                
        if not all_results:
            print(f"{thresh:<12.2f} {'0':>6} {'0.0':>11} {'-':>8} {'-':>9} {'-':>7} {'-':>7} {'-':>8}")
            continue
            
        total_trades = sum(r.total_trades for r in all_results)
        trades_per_day = total_trades / 180.0
        win_rate = np.mean([r.win_rate for r in all_results])
        profit_factor = np.mean([r.profit_factor for r in all_results])
        sharpe = np.mean([r.sharpe for r in all_results])
        max_dd = np.mean([r.max_drawdown for r in all_results])
        cum_ret = sum(r.cumulative_return for r in all_results)
        
        print(
            f"{thresh:<12.2f} {total_trades:>6} {trades_per_day:>11.1f} "
            f"{win_rate*100:>7.1f}% {profit_factor:>9.2f} {sharpe:>+7.2f} "
            f"{max_dd*100:>6.1f}% {cum_ret:>+8.1f}%"
        )
        
if __name__ == "__main__":
    main()
