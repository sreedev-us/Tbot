#!/usr/bin/env python3
"""
FROZEN HOLDOUT TEST — Phase 3 Final Gate
==========================================

IMPORTANT: All 180 days of 1h data have been used in the geometry search
(Q4 was one of the walk-forward windows). To create a genuine holdout:

  1. We re-split the 180 days:
       Training:    Day 0  -> Day 142   (first 142 days)
       Holdout:     Day 143 -> Day 180  (final 38 days, never touched by geometry selection)

  2. We retrain 1h Model B on only the training split.

  3. We evaluate the FROZEN geometry (ATR 2/1 @ 0.85) on the holdout split
     exactly once.

FROZEN PARAMETERS (must not change):
  Timeframe:    1h
  Geometry:     ATR 2.0 / 1.0
  Confidence:   0.85
  Fee:          0.20% baseline
"""
import datetime
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from alpha_engine.model_trainer import ModelTrainer
from walkforward_ab import load_model
from validate_htf import simulate, print_metrics

logging.basicConfig(level=logging.WARNING)

# ── FROZEN PARAMETERS ──────────────────────────────────────────────────────────
ATR_TP      = 2.0
ATR_SL      = 1.0
CONFIDENCE  = 0.85
MAX_HOLD    = 24
BASE_FEE    = 0.20
COST_SWEEP  = [0.20, 0.25, 0.30, 0.35, 0.40]

TRAIN_DAYS   = 142   # Train on first 142 days
HOLDOUT_DAYS = 38    # Evaluate on final 38 days

MODEL_NAME  = "local-model-b-1h-holdout"
DATA_PATH   = Path("data/live/BTC_USDT_1h_live.csv")
# ───────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main():
    print("=" * 72)
    print("  PHASE 3 FINAL HOLDOUT -- FROZEN STRATEGY EVALUATION")
    print(f"  Run timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print("=" * 72)
    print("\nFROZEN PARAMETERS:")
    print(f"  Geometry:     ATR {ATR_TP:.1f} / {ATR_SL:.1f}")
    print(f"  Confidence:   {CONFIDENCE}")
    print(f"  Max hold:     {MAX_HOLD} bars")
    print(f"  Base fee:     {BASE_FEE}%")
    print(f"  Train split:  First {TRAIN_DAYS} days")
    print(f"  Holdout:      Final {HOLDOUT_DAYS} days")

    # ── Load full 1h dataset ────────────────────────────────────────────────────
    raw_df = pd.read_csv(DATA_PATH)
    config = TrainingDataConfig()
    gen    = TrainingDataGenerator(config)

    print(f"\nEngineering features on full 1h dataset...")
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    t0  = full_df["timestamp"].iloc[0]
    t_split  = t0 + pd.Timedelta(days=TRAIN_DAYS)
    t_end    = full_df["timestamp"].iloc[-1]

    train_df   = full_df[full_df["timestamp"] <  t_split].reset_index(drop=True)
    holdout_df = full_df[full_df["timestamp"] >= t_split].reset_index(drop=True)

    print(f"\nData split:")
    print(f"  Training:  {train_df['timestamp'].iloc[0]} -> {train_df['timestamp'].iloc[-1]}  ({len(train_df)} bars)")
    print(f"  Holdout:   {holdout_df['timestamp'].iloc[0]} -> {holdout_df['timestamp'].iloc[-1]}  ({len(holdout_df)} bars)")
    holdout_actual_days = (holdout_df['timestamp'].iloc[-1] - holdout_df['timestamp'].iloc[0]).total_seconds() / 86400

    # ── Retrain Model B on training split only ─────────────────────────────────
    section("RETRAINING Model B on training split only")
    feature_cols = gen.get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = gen.label_generator.generate_labels(train_df)["target"] if "target" not in train_df.columns else train_df["target"]

    # Re-generate labels properly for the training split
    train_labeled = gen.generate_training_data(
        raw_df[raw_df.index < len(train_df)],
        ablation_level="B"
    )
    feature_cols = gen.get_feature_columns(train_labeled)
    X_train = train_labeled[feature_cols]
    y_train = train_labeled["target"]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Features:         {len(feature_cols)}")
    trainer = ModelTrainer("./models", "xgboost")
    trainer.train(X_train, y_train)
    model_dir = trainer.save_model(
        MODEL_NAME,
        metadata={
            "training_samples": len(X_train),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "description": "Model B (1h) trained on first 142 days only — holdout split",
            "train_end": str(t_split),
        }
    )
    print(f"  Saved to: {model_dir}")

    # ── Load freshly trained model ─────────────────────────────────────────────
    booster, scaler, fc = load_model(Path("models") / MODEL_NAME)

    # ── Primary holdout evaluation — RUNS EXACTLY ONCE ────────────────────────
    section("HOLDOUT RESULT -- PRIMARY (0.20% cost)")
    geo_kwargs = {"atr_tp": ATR_TP, "atr_sl": ATR_SL}

    result = simulate(
        booster, scaler, fc, holdout_df,
        confidence_threshold=CONFIDENCE,
        fee_pct=BASE_FEE,
        max_hold_bars=MAX_HOLD,
        **geo_kwargs
    )

    if not result:
        print("  No trades generated on the holdout period.")
        return

    print_metrics(result, label=f"1h ATR {ATR_TP}/{ATR_SL} @ {CONFIDENCE} [HOLDOUT]",
                  days=holdout_actual_days)

    # ── ATR sanity check on holdout ─────────────────────────────────────────────
    close = holdout_df["close"].values
    high  = holdout_df["high"].values
    low   = holdout_df["low"].values
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr_pct = atr / close * 100.0
    print(f"\n  ATR check (holdout period):  Median={np.median(atr_pct):.3f}%  "
          f"2xATR TP~={np.median(atr_pct)*2:.3f}%  (vs {BASE_FEE}% cost)")

    # ── Cost sensitivity ───────────────────────────────────────────────────────
    section("COST SENSITIVITY (holdout, frozen geometry)")
    print(f"  {'Cost':>8} {'Trades':>7} {'PF':>6} {'Sharpe':>8} {'MaxDD%':>8} {'NetRet%':>10}")
    print("  " + "-" * 55)
    for cost in COST_SWEEP:
        r = simulate(booster, scaler, fc, holdout_df,
                     confidence_threshold=CONFIDENCE,
                     fee_pct=cost, max_hold_bars=MAX_HOLD, **geo_kwargs)
        if r:
            mark = " *" if r["net_cum_ret"] > 0 else ""
            print(f"  {cost:>7.2f}% {r['trades']:>7} {r['profit_factor']:>6.2f} "
                  f"{r['sharpe']:>+8.2f} {r['max_dd_pct']:>7.1f}% "
                  f"{r['net_cum_ret']:>+10.1f}%{mark}")

    # ── Sub-period breakdown (informational only -- do NOT retune on this) ─────
    section("SUB-PERIOD BREAKDOWN (informational only -- do not retune)")
    h0 = holdout_df["timestamp"].iloc[0]
    h_end = holdout_df["timestamp"].iloc[-1]
    sub_windows = [
        ("W1 (Days 0-12)",  h0,                              h0 + pd.Timedelta(days=12)),
        ("W2 (Days 12-25)", h0 + pd.Timedelta(days=12),     h0 + pd.Timedelta(days=25)),
        ("W3 (Days 25-38)", h0 + pd.Timedelta(days=25),     h_end),
    ]
    print(f"  {'Window':<18} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'MaxDD%':>7} {'NetRet%':>9}")
    print("  " + "-" * 65)
    for label, ws, we in sub_windows:
        w_df = holdout_df[(holdout_df["timestamp"] >= ws) &
                          (holdout_df["timestamp"] < we)].reset_index(drop=True)
        if len(w_df) < 20:
            continue
        r = simulate(booster, scaler, fc, w_df,
                     confidence_threshold=CONFIDENCE,
                     fee_pct=BASE_FEE, max_hold_bars=MAX_HOLD, **geo_kwargs)
        if r:
            print(f"  {label:<18} {r['trades']:>7} {r['win_rate']*100:>5.1f}% "
                  f"{r['profit_factor']:>6.2f} {r['sharpe']:>+7.2f} "
                  f"{r['max_dd_pct']:>6.1f}% {r['net_cum_ret']:>+9.1f}%")

    # ── Verdict ────────────────────────────────────────────────────────────────
    section("HOLDOUT VERDICT")
    pf   = result["profit_factor"]
    ret  = result["net_cum_ret"]
    shar = result["sharpe"]
    dd   = result["max_dd_pct"]

    passed = pf > 1.0 and ret > 0 and shar > 0

    print(f"\n  Profit Factor > 1.0:    {'PASS' if pf > 1.0 else 'FAIL'}  ({pf:.2f})")
    print(f"  Net Return > 0:         {'PASS' if ret > 0 else 'FAIL'}  ({ret:+.1f}%)")
    print(f"  Sharpe > 0:             {'PASS' if shar > 0 else 'FAIL'}  ({shar:+.2f})")
    print(f"  Max DD:                 {dd:+.1f}%")
    print()
    if passed:
        print("  *** HOLDOUT PASSED ***")
        print("  1h ATR 2/1 @ 0.85 is validated as the research baseline.")
        print("  Next: reintroduce F&G on the 1h timeframe (1h B vs 1h C).")
    else:
        print("  *** HOLDOUT FAILED ***")
        print("  Do NOT retune. Analyse failure mode first.")
        print("  Possible causes: regime shift, selection optimism, model instability.")
    print()


if __name__ == "__main__":
    main()
