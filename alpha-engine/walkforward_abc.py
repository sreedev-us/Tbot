#!/usr/bin/env python3
"""
Walk-Forward B vs C Comparison
Tests Model B and Model C across the same 4 regime windows.

Outputs:
  1. Threshold sweep (both models): first positive Sharpe threshold
  2. Regime robustness table at candidate threshold
  3. Extended slippage sensitivity (0.20% -> 0.40%) at candidate threshold

H1: Does C fix Recovery regime?
H2: Does C survive higher slippage costs?
H3: Does C produce positive Sharpe at a lower threshold?
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
    print(f"THRESHOLD SWEEP: {label}")
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


def print_regime_table(label_b, sweep_b, label_c, sweep_c, thresh):
    """Per-regime breakdown at a specific threshold for both models."""
    regimes = ["trending", "choppy", "volatile", "recovery"]

    def get_by_regime(sweep_rows, t):
        for th, br in sweep_rows:
            if abs(th - t) < 1e-6:
                return br
        return {}

    br_b = get_by_regime(sweep_b, thresh)
    br_c = get_by_regime(sweep_c, thresh)

    def fmt(r):
        if r is None:
            return "   --   / -- /    --  "
        mark = "*" if r.cumulative_return > 0 else " "
        return f"{mark}{r.sharpe:+.2f} / {r.profit_factor:.2f} / {r.cumulative_return:+.0f}%"

    print(f"\n{'='*100}")
    print(f"REGIME BREAKDOWN at threshold={thresh:.3f}   (Sharpe / PF / CumRet%  * = profitable)")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Trending':<24} {'Choppy':<24} {'Volatile':<24} {'Recovery':<24}")
    print(f"{'-'*100}")
    row_b = f"{label_b:<30}"
    row_c = f"{label_c:<30}"
    for regime in regimes:
        row_b += f" {fmt(br_b.get(regime)):<23}"
        row_c += f" {fmt(br_c.get(regime)):<23}"
    print(row_b)
    print(row_c)


def print_slippage_table(label_b, booster_b, scaler_b, fc_b,
                         label_c, booster_c, scaler_c, fc_c,
                         full_df, windows, thresh, slippages):
    print(f"\n{'='*90}")
    print(f"EXTENDED SLIPPAGE SENSITIVITY at threshold={thresh:.3f}")
    print(f"{'='*90}")
    print(f"{'Model':<30} {'Slippage':>9} {'Total':>7} {'Sharpe':>7} {'PF':>6} {'MaxDD':>7} {'CumRet%':>8}")
    print(f"{'-'*90}")

    for booster, scaler, fc, label in [
        (booster_b, scaler_b, fc_b, label_b),
        (booster_c, scaler_c, fc_c, label_c),
    ]:
        first = True
        for slip in slippages:
            total_fee = 0.10 + slip
            by_regime = {}
            for period_name, start_ts, end_ts, regime in windows:
                window_df = full_df[
                    (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
                ].copy()
                if len(window_df) < 200:
                    continue
                r = evaluate_window(booster, scaler, fc, window_df, label,
                                    period_name, regime, confidence_threshold=thresh,
                                    fee_pct=total_fee)
                if r:
                    by_regime[regime] = r

            m = agg_metrics(by_regime)
            if not m:
                continue
            lbl = label if first else " " * len(label)
            mark = " *" if m["ret"] > 0 else "  "
            print(f"{lbl:<30} {slip*100:>8.2f}% {total_fee*100:>6.2f}% "
                  f"{m['sharpe']:>+7.2f} {m['pf']:>6.2f} {m['maxdd']*100:>6.1f}% "
                  f"{m['ret']:>+7.1f}%{mark}")
            first = False
        print()


def main():
    data_path = Path("data/live/BTC_USDT_1m_live.csv")
    fng_path = Path("data/sentiment/fear_greed_historical.csv")
    models_base = Path("models")

    print("Loading models...")
    booster_b, scaler_b, fc_b = load_model(models_base / "local-benchmark-b")
    booster_c, scaler_c, fc_c = load_model(models_base / "local-model-c")

    print("Loading and processing data (with F&G for Model C)...")
    raw_df = pd.read_csv(data_path)
    fng_df = pd.read_csv(fng_path)
    config = TrainingDataConfig()

    # Model B: no F&G features
    gen_b = TrainingDataGenerator(config)
    full_b = gen_b.generate_training_data(raw_df)
    full_b = full_b.assign(timestamp=pd.to_datetime(full_b["timestamp"], utc=True))
    full_b = full_b.sort_values("timestamp").reset_index(drop=True)

    # Model C: with F&G features
    gen_c = TrainingDataGenerator(config)
    full_c = gen_c.generate_training_data(raw_df, fng_df=fng_df)
    full_c = full_c.assign(timestamp=pd.to_datetime(full_c["timestamp"], utc=True))
    full_c = full_c.sort_values("timestamp").reset_index(drop=True)

    t0 = full_b["timestamp"].iloc[0]
    windows = [
        ("Q1-Trending",  t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45),  "trending"),
        ("Q2-Choppy",    t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90),  "choppy"),
        ("Q3-Volatile",  t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135), "volatile"),
        ("Q4-Recovery",  t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=179), "recovery"),
    ]

    thresholds = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]

    print("\nSweeping Model B...")
    sweep_b = sweep_model("B", booster_b, scaler_b, fc_b, full_b, windows, thresholds)
    print_sweep_table("Model B (Server AI + XGBoost, no F&G)", sweep_b)

    print("\nSweeping Model C...")
    sweep_c = sweep_model("C", booster_c, scaler_c, fc_c, full_c, windows, thresholds)
    print_sweep_table("Model C (Server AI + XGBoost + Fear&Greed)", sweep_c)

    # Regime tables at the key thresholds found in Phase 1
    for t in [0.94, 0.96]:
        print_regime_table("B (no F&G)", sweep_b, "C (with F&G)", sweep_c, t)

    # Extended slippage sensitivity at 0.96
    slippages = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print_slippage_table(
        "Model B @ 0.96", booster_b, scaler_b, fc_b,
        "Model C @ 0.96", booster_c, scaler_c, fc_c,
        full_b if True else full_c, windows, 0.96, slippages
    )
    # Also run C-only at its own best threshold
    print_slippage_table(
        "Model C @ 0.96", booster_c, scaler_c, fc_c,
        "Model C @ 0.96", booster_c, scaler_c, fc_c,
        full_c, windows, 0.96, slippages
    )

    print("Done.")


if __name__ == "__main__":
    main()
