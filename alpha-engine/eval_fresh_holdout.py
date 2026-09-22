#!/usr/bin/env python3
"""
FRESH HOLDOUT — FH-48h-1.0pct
================================
Evaluates the FH-48h-1.0pct model on Sep 6-22, 2026 data.
This data has NEVER been used in any development step.
Run this script EXACTLY ONCE. Do not retune based on result.

Frozen parameters:
  Model:      FH-48h-1.0pct (D2c features, fixed-horizon target)
  Geometry:   ATR 2.0 / 1.0
  Confidence: 0.85
  Fee:        0.20%
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model
from validate_htf import simulate

logging.basicConfig(level=logging.WARNING)

ATR_TP     = 2.0
ATR_SL     = 1.0
CONFIDENCE = 0.85
MAX_HOLD   = 24
FEE        = 0.20
MODEL_NAME = "FH-48h-1.0pct"
ABLATION   = "D2c"
FRESH_PATH = Path("data/live/BTC_USDT_1h_fresh.csv")


def prediction_bias(booster, scaler, fc, df, confidence=0.85):
    X = df[fc].values
    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        import xgboost as xgb
        probs = booster.predict(xgb.DMatrix(scaled))
    p_down, p_up = probs[:, 0], probs[:, 2]
    total = len(df)
    close = df["close"].values
    up_pct = (close[1:] > close[:-1]).sum() / max(len(close)-1, 1) * 100
    return {
        "buy_pct":  ((p_up > p_down) & (p_up >= confidence)).sum() / total * 100,
        "sell_pct": ((p_down > p_up) & (p_down >= confidence)).sum() / total * 100,
        "up_pct":   up_pct,
        "dn_pct":   100 - up_pct,
    }


def confidence_calibration(booster, scaler, fc, df, fee=0.20, max_hold=24,
                            atr_tp=2.0, atr_sl=1.0):
    df = df.reset_index(drop=True)
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr = np.maximum.reduce([high-low, np.abs(high-prev_c), np.abs(low-prev_c)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    X = df[fc].values
    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        import xgboost as xgb
        probs = booster.predict(xgb.DMatrix(scaled))

    p_down, p_up = probs[:, 0], probs[:, 2]
    results = {}
    for lo, hi in [(0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]:
        pnls = []
        for i in range(len(df) - max_hold):
            if p_up[i] > p_down[i] and lo <= p_up[i] < hi:
                direction = 1
            elif p_down[i] > p_up[i] and lo <= p_down[i] < hi:
                direction = -1
            else:
                continue
            entry = close[i]
            tp_p = entry + direction * atr_tp * atr[i]
            sl_p = entry - direction * atr_sl * atr[i]
            pnl = None
            eb = i + max_hold
            for j in range(i+1, min(i+max_hold+1, len(df))):
                if direction == 1:
                    if high[j] >= tp_p: pnl = (tp_p-entry)/entry*100; eb=j; break
                    if low[j]  <= sl_p: pnl = (sl_p-entry)/entry*100; eb=j; break
                else:
                    if low[j]  <= tp_p: pnl = (entry-tp_p)/entry*100; eb=j; break
                    if high[j] >= sl_p: pnl = (entry-sl_p)/entry*100; eb=j; break
            if pnl is None:
                ep = close[min(eb, len(close)-1)]
                pnl = direction * (ep-entry)/entry*100
            pnls.append(pnl - fee)
        if pnls:
            pnls = np.array(pnls)
            results[f"{lo:.2f}-{hi:.2f}"] = {
                "n": len(pnls), "wr": (pnls > 0).mean() * 100, "avg": pnls.mean()
            }
        else:
            results[f"{lo:.2f}-{hi:.2f}"] = {"n": 0, "wr": 0, "avg": 0}
    return results


def main():
    raw_fresh = pd.read_csv(FRESH_PATH)
    config = TrainingDataConfig()
    gen    = TrainingDataGenerator(config)

    # Engineer D2c features on fresh data
    feats = gen.feature_engineer.engineer_features(raw_fresh, ablation_level=ABLATION)
    feats = feats.assign(
        timestamp=pd.to_datetime(feats["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    close  = feats["close"].values
    btc_ret = (close[-1] - close[0]) / close[0] * 100
    days   = (feats["timestamp"].iloc[-1] - feats["timestamp"].iloc[0]).total_seconds() / 86400

    print("=" * 65)
    print("  GENUINE FRESH HOLDOUT — FH-48h-1.0pct")
    print(f"  {feats['timestamp'].iloc[0].date()} -> {feats['timestamp'].iloc[-1].date()}")
    print(f"  {days:.1f} days | BTC: {close[0]:.0f} -> {close[-1]:.0f} ({btc_ret:+.1f}%)")
    print("=" * 65)
    print("  This data was NEVER used in training, geometry selection,")
    print("  target construction, or any diagnostic step.")
    print()

    booster, scaler, fc = load_model(Path("models") / MODEL_NAME)

    result = simulate(booster, scaler, fc, feats,
                      confidence_threshold=CONFIDENCE,
                      fee_pct=FEE, max_hold_bars=MAX_HOLD,
                      atr_tp=ATR_TP, atr_sl=ATR_SL)

    bias  = prediction_bias(booster, scaler, fc, feats, CONFIDENCE)
    calib = confidence_calibration(booster, scaler, fc, feats, FEE, MAX_HOLD)

    if not result:
        print("  No trades generated.")
        return

    print("RESULT:")
    print(f"  Trades:        {result['trades']} ({result['trades']/days:.1f}/day)")
    print(f"  Win rate:      {result['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {result['profit_factor']:.2f}")
    print(f"  Sharpe:        {result['sharpe']:+.2f}")
    print(f"  Max DD:        {result['max_dd_pct']:+.1f}%")
    print(f"  Net Return:    {result['net_cum_ret']:+.1f}%")
    print(f"  Avg PnL/trade: {result['avg_pnl_per_trade']:+.4f}%")

    print()
    print("Prediction Bias vs Actuals:")
    print(f"  BUY  {bias['buy_pct']:.1f}%  | Actual UP  {bias['up_pct']:.1f}%")
    print(f"  SELL {bias['sell_pct']:.1f}%  | Actual DN  {bias['dn_pct']:.1f}%")

    print()
    print("Confidence Calibration:")
    for bucket, m in calib.items():
        mark = " ***" if m['avg'] > 0 else ""
        print(f"  {bucket}: count={m['n']:>3}, win%={m['wr']:>5.1f}%, avg_ret={m['avg']:>+.3f}%{mark}")

    print()
    print("VERDICT:")
    passed = result["profit_factor"] > 1.0 and result["net_cum_ret"] > 0
    print(f"  PF > 1.0: {'PASS' if result['profit_factor'] > 1.0 else 'FAIL'} ({result['profit_factor']:.2f})")
    print(f"  Net > 0:  {'PASS' if result['net_cum_ret'] > 0 else 'FAIL'} ({result['net_cum_ret']:+.1f}%)")
    print(f"  Sharpe>0: {'PASS' if result['sharpe'] > 0 else 'FAIL'} ({result['sharpe']:+.2f})")
    print()
    if passed:
        print("  *** FRESH HOLDOUT PASSED ***")
    else:
        print("  *** FRESH HOLDOUT FAILED — Do NOT retune. Analyse result. ***")


if __name__ == "__main__":
    main()
