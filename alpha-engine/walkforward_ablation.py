#!/usr/bin/env python3
"""
Walk-Forward Ablation Comparison (B vs C1 vs C2 vs C3)

Outputs:
  1. H3: Full Threshold Curve (Sharpe/CumRet/PF) for all models
  2. H1: Regime Breakdown at key threshold (Recovery protection)
  3. H2: Extended Execution Costs Evaluation (0.20% to 0.40% total fee)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model, evaluate_window

logging.basicConfig(level=logging.WARNING)


def sweep_model(label, booster, scaler, feature_columns, full_df, windows,
                thresholds, fee_pct=0.10, max_hold_bars=20):
    """Sweep thresholds for a single model. Returns list of (thresh, results_by_regime)."""
    rows = []
    for thresh in thresholds:
        by_regime = {}
        for period_name, start_ts, end_ts, regime in windows:
            window_df = full_df[
                (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
            ].copy()
            if len(window_df) < 200:
                continue
            result = evaluate_window(
                booster, scaler, feature_columns,
                window_df, label, period_name, regime,
                max_hold_bars=max_hold_bars,
                confidence_threshold=thresh,
                fee_pct=fee_pct,
            )
            if result:
                by_regime[regime] = result
        rows.append((thresh, by_regime))
    return rows


def agg_metrics(by_regime):
    results = list(by_regime.values())
    if not results:
        return None
    return {
        "trades": sum(r.total_trades for r in results),
        "tpd":    sum(r.total_trades for r in results) / 180.0,
        "wr":     np.mean([r.win_rate for r in results]),
        "pf":     np.mean([r.profit_factor for r in results]),
        "sharpe": np.mean([r.sharpe for r in results]),
        "maxdd":  np.mean([r.max_drawdown for r in results]),
        "ret":    sum(r.cumulative_return for r in results),
    }


def print_sweep_table(label, sweep_rows):
    print(f"\n{'='*90}")
    print(f"H3: THRESHOLD SWEEP -> {label}")
    print(f"{'='*90}")
    print(f"{'Thresh':>8} {'T/Day':>6} {'WinRate':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'CumRet%':>8}")
    print(f"{'-'*90}")
    for thresh, by_regime in sweep_rows:
        m = agg_metrics(by_regime)
        if not m:
            continue
        mark = " ***" if m["ret"] > 0 else ("  **" if m["pf"] > 1.0 else "    ")
        print(f"{thresh:>8.3f} {m['tpd']:>6.1f} {m['wr']*100:>7.1f}% "
              f"{m['pf']:>6.2f} {m['sharpe']:>+7.2f} {m['maxdd']*100:>6.1f}% "
              f"{m['ret']:>+8.1f}%{mark}")


def print_regime_table(models, thresh):
    """Per-regime breakdown at a specific threshold for all models."""
    regimes = ["trending", "choppy", "volatile", "recovery"]

    print(f"\n{'='*120}")
    print(f"H1: REGIME BREAKDOWN at threshold={thresh:.3f}   (Sharpe / PF / CumRet%  * = profitable)")
    print(f"{'='*120}")
    print(f"{'Model':<15} {'Trending':<24} {'Choppy':<24} {'Volatile':<24} {'Recovery':<24}")
    print(f"{'-'*120}")
    
    def get_by_regime(sweep_rows, t):
        for th, br in sweep_rows:
            if abs(th - t) < 1e-6:
                return br
        return {}

    def fmt(r):
        if r is None:
            return "   --   / -- /    --  "
        mark = "*" if r.cumulative_return > 0 else " "
        return f"{mark}{r.sharpe:+.2f} / {r.profit_factor:.2f} / {r.cumulative_return:+.0f}%"

    for label, sweep in models:
        row = f"{label:<15}"
        br = get_by_regime(sweep, thresh)
        for regime in regimes:
            row += f" {fmt(br.get(regime)):<23}"
        print(row)


def print_execution_cost_table(label, booster, scaler, fc, full_df, windows, thresh):
    costs = [0.20, 0.25, 0.30, 0.35, 0.40]
    
    print(f"\n{'='*80}")
    print(f"H2: EXECUTION COSTS SENSITIVITY -> {label} @ {thresh:.3f}")
    print(f"{'='*80}")
    print(f"{'Total Cost':>10} {'T/Day':>7} {'Sharpe':>7} {'PF':>6} {'CumRet% (Net Exp)':>18}")
    print(f"{'-'*80}")

    for cost in costs:
        by_regime = {}
        for period_name, start_ts, end_ts, regime in windows:
            window_df = full_df[
                (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
            ].copy()
            if len(window_df) < 200:
                continue
            r = evaluate_window(booster, scaler, fc, window_df, label,
                                period_name, regime, confidence_threshold=thresh,
                                fee_pct=cost)
            if r:
                by_regime[regime] = r

        m = agg_metrics(by_regime)
        if not m:
            continue
        mark = " *" if m["ret"] > 0 else "  "
        print(f"{cost:>9.2f}% {m['tpd']:>7.1f} {m['sharpe']:>+7.2f} {m['pf']:>6.2f} "
              f"{m['ret']:>+17.1f}%{mark}")


def main():
    data_path = Path("data/live/BTC_USDT_1m_live.csv")
    fng_path = Path("data/sentiment/fear_greed_historical.csv")
    models_base = Path("models")

    print("Loading models...")
    models_to_test = [
        ("B", "local-benchmark-b"),
        ("C1", "local-model-c1"),
        ("C2", "local-model-c2"),
        ("C3", "local-model-c3"),
    ]
    
    loaded_models = {}
    for label, folder in models_to_test:
        loaded_models[label] = load_model(models_base / folder)

    print("Loading and processing data (C3 ablation level provides superset of features)...")
    raw_df = pd.read_csv(data_path)
    fng_df = pd.read_csv(fng_path)
    config = TrainingDataConfig()

    # Generate the superset of features (C3)
    gen = TrainingDataGenerator(config)
    full_df = gen.generate_training_data(raw_df, fng_df=fng_df, ablation_level="C3")
    full_df = full_df.assign(timestamp=pd.to_datetime(full_df["timestamp"], utc=True))
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    t0 = full_df["timestamp"].iloc[0]
    windows = [
        ("Q1-Trending",  t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45),  "trending"),
        ("Q2-Choppy",    t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90),  "choppy"),
        ("Q3-Volatile",  t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135), "volatile"),
        ("Q4-Recovery",  t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=179), "recovery"),
    ]

    thresholds = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]

    print("\nStarting H3 Sweeps...")
    all_sweeps = []
    for label, folder in models_to_test:
        booster, scaler, fc = loaded_models[label]
        # B vs C relies on 'feature_columns' to slice the dataframe to the correct subset
        sweep = sweep_model(label, booster, scaler, fc, full_df, windows, thresholds)
        all_sweeps.append((label, sweep))
        print_sweep_table(label, sweep)

    print("\nStarting H1 Regime Breakdown...")
    for t in [0.95, 0.96]:
        print_regime_table(all_sweeps, t)

    print("\nStarting H2 Execution Cost Analysis (Threshold = 0.96)...")
    for label, folder in models_to_test:
        booster, scaler, fc = loaded_models[label]
        print_execution_cost_table(label, booster, scaler, fc, full_df, windows, 0.96)


if __name__ == "__main__":
    main()
