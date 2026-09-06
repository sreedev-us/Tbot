#!/usr/bin/env python3
"""
Extended threshold sweep 0.91-0.97 + hold time analysis.
Also tests longer hold windows (60, 120 bars) to see if giving trades more
room improves the fee-to-return ratio at 1-minute resolution.
"""
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model, evaluate_window

logging.basicConfig(level=logging.INFO, format="%(message)s")

def sweep_hold_and_threshold(
    booster, scaler, feature_columns, full_df, windows,
    thresholds, hold_bars, fee_pct=0.10
):
    print("\n" + "=" * 110)
    print(f"SWEEP: max_hold_bars={hold_bars}")
    print("=" * 110)
    header = (
        f"{'Threshold':<12} {'Trades':>6} {'T/Day':>6} "
        f"{'WinRate':>8} {'ProfFact':>9} {'Sharpe':>7} "
        f"{'MaxDD':>7} {'CumRet%':>9}"
    )
    print(header)
    print("-" * 110)

    for thresh in thresholds:
        all_results = []
        for period_name, start_ts, end_ts, regime in windows:
            window_df = full_df[
                (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
            ].copy()
            if len(window_df) < 200:
                continue
            result = evaluate_window(
                booster, scaler, feature_columns,
                window_df, "B", period_name, regime,
                max_hold_bars=hold_bars,
                confidence_threshold=thresh,
                fee_pct=fee_pct,
            )
            if result:
                all_results.append(result)

        if not all_results:
            print(f"{thresh:<12.2f} {'0':>6} {'0.0':>6}")
            continue

        total_trades = sum(r.total_trades for r in all_results)
        tpd = total_trades / 180.0
        win_rate = np.mean([r.win_rate for r in all_results])
        pf = np.mean([r.profit_factor for r in all_results])
        sharpe = np.mean([r.sharpe for r in all_results])
        max_dd = np.mean([r.max_drawdown for r in all_results])
        cum_ret = sum(r.cumulative_return for r in all_results)

        marker = "  *** POSITIVE PF" if cum_ret > 0 else ("  ** PF>1" if pf > 1.0 else "")
        print(
            f"{thresh:<12.2f} {total_trades:>6} {tpd:>6.1f} "
            f"{win_rate*100:>7.1f}% {pf:>9.2f} {sharpe:>+7.2f} "
            f"{max_dd*100:>6.1f}% {cum_ret:>+9.1f}%{marker}"
        )


def main():
    data_path = Path("data/live/BTC_USDT_1m_live.csv")
    model_path = Path("models/local-benchmark-b")

    booster, scaler, feature_columns = load_model(model_path)

    raw_df = pd.read_csv(data_path)
    generator = TrainingDataGenerator(TrainingDataConfig())
    full_df = generator.generate_training_data(raw_df)
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    t0 = full_df["timestamp"].iloc[0]
    windows = [
        ("Q1", t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45),  "trending"),
        ("Q2", t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90),  "choppy"),
        ("Q3", t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135), "volatile"),
        ("Q4", t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=179), "recovery"),
    ]

    # Part 1: Fine-grained sweep 0.90–0.97 with standard 20-bar hold
    fine_thresholds = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
    sweep_hold_and_threshold(
        booster, scaler, feature_columns, full_df, windows,
        fine_thresholds, hold_bars=20
    )

    # Part 2: Longer hold windows — same threshold sweep but give trades 60 and 120 bars
    # Hypothesis: bigger hold window = bigger moves = fee becomes proportionally smaller
    for hold in [60, 120]:
        sweep_hold_and_threshold(
            booster, scaler, feature_columns, full_df, windows,
            thresholds=[0.85, 0.90, 0.92, 0.95],
            hold_bars=hold
        )


if __name__ == "__main__":
    main()
