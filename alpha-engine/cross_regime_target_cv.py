#!/usr/bin/env python3
"""
Cross-regime transfer matrix for fixed-horizon target reconstruction.

This follows the Phase 4 gate:
  D2 fails transfer -> reconstruct target -> cross-regime CV.

The matrix is observational. Do not tune future production choices directly
from these regime-pair test results.
"""
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.model_trainer import ModelTrainer
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from target_reconstruction import (
    confidence_calibration,
    generate_fixed_horizon_labels,
    prediction_bias,
)
from validate_htf import simulate
from walkforward_ab import load_model

logging.basicConfig(level=logging.WARNING)

DATA_PATH = Path("data/live/BTC_USDT_1h_live.csv")
ABLATION_LEVEL = "D2c"
ATR_TP = 2.0
ATR_SL = 1.0
CONFIDENCE = 0.85
MAX_HOLD = 24
FEE = 0.20

TARGET_CONFIGS = [
    {"horizon": 12, "threshold": 0.40, "name": "FH-12h-0.4pct"},
    {"horizon": 24, "threshold": 0.60, "name": "FH-24h-0.6pct"},
    {"horizon": 24, "threshold": 1.00, "name": "FH-24h-1.0pct"},
    {"horizon": 48, "threshold": 1.00, "name": "FH-48h-1.0pct"},
]


def train_fixed_horizon_model(train_feats, feature_cols, regime_label, cfg):
    labeled = generate_fixed_horizon_labels(
        train_feats,
        horizon=cfg["horizon"],
        threshold_pct=cfg["threshold"],
    )
    if len(labeled) < 50 or labeled["target"].nunique() < 2:
        return None, None, None, 0

    model_name = (
        f"cross_regime_{cfg['name'].lower()}_"
        f"{regime_label.lower().replace('-', '_')}"
    )
    trainer = ModelTrainer("./models", "xgboost")
    trainer.train(labeled[feature_cols], labeled["target"])
    trainer.save_model(model_name, metadata={
        "feature_columns": feature_cols,
        "ablation_level": ABLATION_LEVEL,
        "target_type": "fixed_horizon_directional_return",
        "target_horizon_bars": cfg["horizon"],
        "target_threshold_pct": cfg["threshold"],
        "description": f"Cross-regime fixed-horizon train: {regime_label}",
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    booster, scaler, fc = load_model(Path("models") / model_name)
    return booster, scaler, fc, len(labeled)


def bucket_summary(calib):
    pieces = []
    for bucket in ["0.85-0.90", "0.90-0.95", "0.95-1.00"]:
        m = calib[bucket]
        pieces.append(f"{bucket}: n={m['count']}, wr={m['win_pct']:.1f}%, avg={m['avg_ret']:+.3f}%")
    return " | ".join(pieces)


def main():
    raw_df = pd.read_csv(DATA_PATH)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    gen = TrainingDataGenerator(TrainingDataConfig())
    full_feats = gen.feature_engineer.engineer_features(raw_df, ablation_level=ABLATION_LEVEL)
    full_feats = full_feats.assign(
        timestamp=pd.to_datetime(full_feats["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)
    feature_cols = gen.get_feature_columns(full_feats)

    t0 = full_feats["timestamp"].iloc[0]
    regimes = [
        ("Q1-Trending", t0 + pd.Timedelta(days=0), t0 + pd.Timedelta(days=45)),
        ("Q2-Choppy", t0 + pd.Timedelta(days=45), t0 + pd.Timedelta(days=90)),
        ("Q3-Volatile", t0 + pd.Timedelta(days=90), t0 + pd.Timedelta(days=135)),
        ("Q4-Recovery", t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=180)),
    ]

    print("=" * 78)
    print("  FIXED-HORIZON TARGET CROSS-REGIME CV")
    print("  Feature set: D2c | ATR 2/1 @ 0.85 | observational only")
    print("=" * 78)

    all_summaries = []
    for cfg in TARGET_CONFIGS:
        print("\n\n" + "=" * 78)
        print(f"  TARGET: {cfg['name']}  horizon={cfg['horizon']} bars  threshold={cfg['threshold']}%")
        print("=" * 78)

        trained = {}
        for label, start, end in regimes:
            train_feats = full_feats[
                (full_feats["timestamp"] >= start) &
                (full_feats["timestamp"] < end)
            ].reset_index(drop=True)
            booster, scaler, fc, n_samples = train_fixed_horizon_model(
                train_feats, feature_cols, label, cfg
            )
            if booster is None:
                print(f"  Training {label}: skipped")
                continue
            trained[label] = (booster, scaler, fc)
            print(f"  Training {label}: {n_samples} samples")

        rows = []
        for train_label, (booster, scaler, fc) in trained.items():
            print(f"\n  Trained on: {train_label}")
            print(f"  {'Test Regime':<16} {'T':>4} {'WR%':>6} {'Exp%':>7} {'PF':>6} "
                  f"{'Sharpe':>7} {'DD%':>6} {'Ret%':>8} {'BUY/UP':>13} "
                  f"{'SELL/DN':>13} {'HOLD%':>7}")
            print("  " + "-" * 112)

            for test_label, start, end in regimes:
                test_df = full_feats[
                    (full_feats["timestamp"] >= start) &
                    (full_feats["timestamp"] < end)
                ].reset_index(drop=True)
                result = simulate(
                    booster, scaler, fc, test_df,
                    confidence_threshold=CONFIDENCE,
                    fee_pct=FEE,
                    max_hold_bars=MAX_HOLD,
                    atr_tp=ATR_TP,
                    atr_sl=ATR_SL,
                )
                bias = prediction_bias(booster, scaler, fc, test_df, CONFIDENCE)
                calib = confidence_calibration(
                    booster, scaler, fc, test_df,
                    fee_pct=FEE,
                    max_hold=MAX_HOLD,
                    atr_tp=ATR_TP,
                    atr_sl=ATR_SL,
                )
                same = "(same)" if train_label == test_label else ""
                if result is None:
                    print(f"  {test_label:<16}    0 trades {same}")
                    continue

                print(f"  {test_label:<16} {result['trades']:>4} {result['win_rate']*100:>5.1f}% "
                      f"{result['avg_pnl_per_trade']:>+6.3f}% {result['profit_factor']:>6.2f} "
                      f"{result['sharpe']:>+7.2f} {result['max_dd_pct']:>5.1f}% "
                      f"{result['net_cum_ret']:>+8.1f}% {bias['buy_pct']:>5.1f}/{bias['up_pct']:<5.1f} "
                      f"{bias['sell_pct']:>5.1f}/{bias['dn_pct']:<5.1f} {bias['hold_pct']:>6.1f}% {same}")
                print("    Confidence: " + bucket_summary(calib))

                rows.append({
                    "target": cfg["name"],
                    "train": train_label,
                    "test": test_label,
                    "ret": result["net_cum_ret"],
                    "expectancy": result["avg_pnl_per_trade"],
                    "buy_gap": bias["buy_pct"] - bias["up_pct"],
                })

        transfer_rows = [r for r in rows if r["train"] != r["test"]]
        profitable = [r for r in transfer_rows if r["ret"] > 0]
        avg_exp = np.mean([r["expectancy"] for r in transfer_rows]) if transfer_rows else 0.0
        avg_ret = np.mean([r["ret"] for r in transfer_rows]) if transfer_rows else 0.0
        avg_buy_gap = np.mean([r["buy_gap"] for r in transfer_rows]) if transfer_rows else 0.0
        print("\n  Transfer summary:")
        print(f"    Profitable off-diagonal windows: {len(profitable)}/{len(transfer_rows)}")
        print(f"    Avg transfer expectancy: {avg_exp:+.3f}%")
        print(f"    Avg transfer net return: {avg_ret:+.1f}%")
        print(f"    Avg BUY minus actual UP gap: {avg_buy_gap:+.1f} pp")
        all_summaries.append((cfg["name"], len(profitable), len(transfer_rows), avg_exp, avg_ret, avg_buy_gap))

    print("\n\n" + "=" * 78)
    print("  FIXED-HORIZON TARGET TRANSFER SUMMARY")
    print("=" * 78)
    print(f"  {'Target':<16} {'Profitable':>12} {'AvgExp%':>9} {'AvgRet%':>9} {'BUYGap':>9}")
    print("  " + "-" * 60)
    for name, prof, total, avg_exp, avg_ret, avg_buy_gap in all_summaries:
        print(f"  {name:<16} {prof:>4}/{total:<7} {avg_exp:>+8.3f}% {avg_ret:>+8.1f}% {avg_buy_gap:>+8.1f}pp")


if __name__ == "__main__":
    main()
