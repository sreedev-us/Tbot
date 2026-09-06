#!/usr/bin/env python3
"""
Regime Robustness Test + Slippage Sensitivity for Model B
----------------------------------------------------------
Outputs two tables:
  1. Per-regime breakdown at thresholds [0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97]
     Columns: Trending | Choppy | Volatile | Recovery | AGGREGATE
     Metrics per cell: Sharpe / PF / CumRet%

  2. Slippage sensitivity at threshold 0.96
     Rows: 0.0%, 0.05%, 0.10%, 0.20% slippage added to fee
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model, evaluate_window

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def run_all_windows(booster, scaler, feature_columns, full_df, windows,
                    threshold, max_hold_bars=20, fee_pct=0.10):
    """Returns dict of {regime -> WindowResult} for a given threshold."""
    results = {}
    for period_name, start_ts, end_ts, regime in windows:
        window_df = full_df[
            (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
        ].copy()
        if len(window_df) < 200:
            continue
        result = evaluate_window(
            booster, scaler, feature_columns,
            window_df, "B", period_name, regime,
            max_hold_bars=max_hold_bars,
            confidence_threshold=threshold,
            fee_pct=fee_pct,
        )
        if result:
            results[regime] = result
    return results


def cell(r):
    """Format a WindowResult as Sh/PF/Ret for a table cell."""
    if r is None:
        return "    --   "
    sh = r.sharpe
    pf = r.profit_factor
    ret = r.cumulative_return
    sh_str = f"{sh:+.2f}"
    ret_str = f"{ret:+.0f}%"
    mark = "*" if ret > 0 else " "
    return f"{mark}{sh_str:>5} {pf:.2f} {ret_str:>5}"


def main():
    data_path = Path("data/live/BTC_USDT_1m_live.csv")
    model_path = Path("models/local-benchmark-b")

    print("Loading model B and data...")
    booster, scaler, feature_columns = load_model(model_path)

    raw_df = pd.read_csv(data_path)
    generator = TrainingDataGenerator(TrainingDataConfig())
    full_df = generator.generate_training_data(raw_df)
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    t0 = full_df["timestamp"].iloc[0]
    windows = [
        ("Q1-Trending",  t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45),  "trending"),
        ("Q2-Choppy",    t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90),  "choppy"),
        ("Q3-Volatile",  t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135), "volatile"),
        ("Q4-Recovery",  t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=179), "recovery"),
    ]

    regimes = ["trending", "choppy", "volatile", "recovery"]
    thresholds = [0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970]

    # ------------------------------------------------------------------
    # TABLE 1: Per-regime breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("REGIME ROBUSTNESS TEST  (Model B)    cell format:  Sharpe / PF / CumRet%    * = profitable")
    print("=" * 115)
    regime_labels = ["Trending", "Choppy", "Volatile", "Recovery", "AGGREGATE"]
    header = f"{'Thresh':>7}  {'T/Day':>5}  "
    for r in regime_labels:
        header += f"  {r:<18}"
    print(header)
    print("-" * 115)

    for thresh in thresholds:
        by_regime = run_all_windows(booster, scaler, feature_columns, full_df, windows,
                                    threshold=thresh, fee_pct=0.10)

        all_results = list(by_regime.values())
        if not all_results:
            print(f"{thresh:>7.3f}  {'0':>5}")
            continue

        total_trades = sum(r.total_trades for r in all_results)
        tpd = total_trades / 180.0

        agg_sharpe  = np.mean([r.sharpe for r in all_results])
        agg_pf      = np.mean([r.profit_factor for r in all_results])
        agg_ret     = sum(r.cumulative_return for r in all_results)

        class _Agg:
            sharpe = agg_sharpe
            profit_factor = agg_pf
            cumulative_return = agg_ret

        row = f"{thresh:>7.3f}  {tpd:>5.1f}  "
        for regime in regimes:
            r = by_regime.get(regime)
            row += f"  {cell(r):<18}"
        row += f"  {cell(_Agg()):<18}"
        print(row)

    # ------------------------------------------------------------------
    # TABLE 2: Slippage / cost sensitivity at threshold 0.96
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("SLIPPAGE SENSITIVITY at threshold=0.96  (fee=0.10% base + slippage added)")
    print("=" * 90)
    print(f"{'Slippage':>10}  {'Total fee':>10}  {'T/Day':>6}  {'WinRate':>8}  {'PF':>6}  {'Sharpe':>7}  {'MaxDD':>7}  {'CumRet%':>8}")
    print("-" * 90)

    slippage_values = [0.00, 0.05, 0.10, 0.20]
    base_fee = 0.10

    for slippage in slippage_values:
        total_fee = base_fee + slippage
        by_regime = run_all_windows(booster, scaler, feature_columns, full_df, windows,
                                    threshold=0.96, fee_pct=total_fee)
        all_results = list(by_regime.values())
        if not all_results:
            print(f"{slippage:>9.2f}%  {total_fee:>9.2f}%  {'0':>6}")
            continue

        total_trades = sum(r.total_trades for r in all_results)
        tpd = total_trades / 180.0
        wr = np.mean([r.win_rate for r in all_results]) * 100
        pf = np.mean([r.profit_factor for r in all_results])
        sharpe = np.mean([r.sharpe for r in all_results])
        max_dd = np.mean([r.max_drawdown for r in all_results]) * 100
        cum_ret = sum(r.cumulative_return for r in all_results)
        mark = " *" if cum_ret > 0 else "  "
        print(f"{slippage:>9.2f}%  {total_fee:>9.2f}%  {tpd:>6.1f}  {wr:>7.1f}%  {pf:>6.2f}  {sharpe:>+7.2f}  {max_dd:>6.1f}%  {cum_ret:>+7.1f}%{mark}")

    print("\nDone.")


if __name__ == "__main__":
    main()
