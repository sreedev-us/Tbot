#!/usr/bin/env python3
"""
Train D2a, D2b, D2c ablations on the first 142 days (same training split as holdout test).
Then evaluate each on the holdout (final 38 days) with full diagnostic output.
Model B is the frozen control group.
"""
import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from alpha_engine.model_trainer import ModelTrainer
from walkforward_ab import load_model
from validate_htf import simulate

logging.basicConfig(level=logging.WARNING)

ATR_TP     = 2.0
ATR_SL     = 1.0
CONFIDENCE = 0.85
MAX_HOLD   = 24
FEE        = 0.20
TRAIN_DAYS = 142
DATA_PATH  = Path("data/live/BTC_USDT_1h_live.csv")

ABLATIONS = [
    {"level": "D2a", "model": "local-model-d2a-1h"},
    {"level": "D2b", "model": "local-model-d2b-1h"},
    {"level": "D2c", "model": "local-model-d2c-1h"},
]


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
    buy_mask  = (p_up > p_down) & (p_up >= confidence)
    sell_mask = (p_down > p_up) & (p_down >= confidence)
    close = df["close"].values
    up_pct = (close[1:] > close[:-1]).sum() / (len(close)-1) * 100
    return {
        "buy_pct":  buy_mask.sum() / total * 100,
        "sell_pct": sell_mask.sum() / total * 100,
        "hold_pct": (~(buy_mask | sell_mask)).sum() / total * 100,
        "up_pct":   up_pct,
        "dn_pct":   100 - up_pct,
    }


def confidence_calibration(booster, scaler, fc, df, fee_pct=0.20, max_hold=24, atr_tp=2.0, atr_sl=1.0):
    """Breakdown of win rate and avg return per confidence bucket."""
    df = df.reset_index(drop=True)
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high-low, np.abs(high-prev_close), np.abs(low-prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    X = df[fc].values
    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        import xgboost as xgb
        probs = booster.predict(xgb.DMatrix(scaled))

    p_down, p_up = probs[:, 0], probs[:, 2]
    buckets = [(0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    results = {}
    for lo, hi in buckets:
        pnls = []
        for i in range(len(df) - max_hold):
            if p_up[i] > p_down[i] and lo <= p_up[i] < hi:
                direction = 1
            elif p_down[i] > p_up[i] and lo <= p_down[i] < hi:
                direction = -1
            else:
                continue
            entry = close[i]
            tp_price = entry + direction * atr_tp * atr[i]
            sl_price = entry - direction * atr_sl * atr[i]
            trade_pnl_pct = None
            exit_bar = i + max_hold
            for j in range(i+1, min(i+max_hold+1, len(df))):
                if direction == 1:
                    if high[j] >= tp_price: trade_pnl_pct = (tp_price-entry)/entry*100; exit_bar=j; break
                    if low[j]  <= sl_price: trade_pnl_pct = (sl_price-entry)/entry*100; exit_bar=j; break
                else:
                    if low[j]  <= tp_price: trade_pnl_pct = (entry-tp_price)/entry*100; exit_bar=j; break
                    if high[j] >= sl_price: trade_pnl_pct = (entry-sl_price)/entry*100; exit_bar=j; break
            if trade_pnl_pct is None:
                exit_p = close[min(exit_bar, len(close)-1)]
                trade_pnl_pct = direction * (exit_p-entry)/entry*100
            pnls.append(trade_pnl_pct - fee_pct)
        if pnls:
            pnls = np.array(pnls)
            results[f"{lo:.2f}-{hi:.2f}"] = {
                "count": len(pnls), "win_pct": (pnls>0).mean()*100, "avg_ret": pnls.mean()
            }
        else:
            results[f"{lo:.2f}-{hi:.2f}"] = {"count": 0, "win_pct": 0, "avg_ret": 0}
    return results


def print_model_results(label, result, bias, calib, days):
    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"    Trades:        {result['trades']} ({result['trades']/days:.1f}/day)")
    print(f"    Win rate:      {result['win_rate']*100:.1f}%")
    print(f"    Profit Factor: {result['profit_factor']:.2f}")
    print(f"    Sharpe:        {result['sharpe']:+.2f}")
    print(f"    Max DD:        {result['max_dd_pct']:+.1f}%")
    print(f"    Net Return:    {result['net_cum_ret']:+.1f}%")
    print(f"    Avg PnL/trade: {result['avg_pnl_per_trade']:+.4f}%")
    print(f"\n    Prediction Bias vs Actuals:")
    print(f"    BUY  {bias['buy_pct']:.1f}%   | Actual UP {bias['up_pct']:.1f}%")
    print(f"    SELL {bias['sell_pct']:.1f}%   | Actual DN {bias['dn_pct']:.1f}%")
    print(f"    HOLD {bias['hold_pct']:.1f}%")
    print(f"\n    Confidence Calibration:")
    for bucket, m in calib.items():
        mark = " ***" if m['avg_ret'] > 0 else ""
        print(f"    {bucket}: count={m['count']:>4}, win%={m['win_pct']:>5.1f}%, avg_ret={m['avg_ret']:>+.3f}%{mark}")


def main():
    raw_df = pd.read_csv(DATA_PATH)
    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)

    # Build training split raw data (first 142 days by row count)
    # We use raw_df sliced by timestamp after timestamping
    raw_ts = pd.to_datetime(raw_df["timestamp"], utc=True)
    t0 = raw_ts.iloc[0]
    t_split = t0 + pd.Timedelta(days=TRAIN_DAYS)
    train_raw = raw_df[raw_ts < t_split].reset_index(drop=True)

    print("=" * 65)
    print("  PHASE 4: D2 ABLATION TRAINING")
    print(f"  Training on first {TRAIN_DAYS} days ({len(train_raw)} raw bars)")
    print("=" * 65)

    for abl in ABLATIONS:
        level = abl["level"]
        model_name = abl["model"]
        print(f"\nTraining {level}...")

        train_labeled = gen.generate_training_data(train_raw, ablation_level=level)
        fc = gen.get_feature_columns(train_labeled)
        X_train = train_labeled[fc]
        y_train = train_labeled["target"]

        print(f"  Samples: {len(X_train)}, Features: {len(fc)}")

        trainer = ModelTrainer("./models", "xgboost")
        trainer.train(X_train, y_train)
        trainer.save_model(model_name, metadata={
            "feature_columns": fc,
            "ablation_level": level,
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "description": f"Phase 4 Model {level} — 1h, {TRAIN_DAYS}-day training split",
        })
        print(f"  Saved: {model_name}")

    # Prepare holdout df for all ablations
    print("\n\n" + "=" * 65)
    print("  HOLDOUT EVALUATION (final 38 days) — ALL MODELS")
    print("=" * 65)

    holdout_results = {}
    for abl in ABLATIONS:
        level = abl["level"]
        model_name = abl["model"]

        booster, scaler, fc = load_model(Path("models") / model_name)

        # Feature-engineer holdout with THIS ablation level (so features match)
        full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level=level)
        full_df = full_df.assign(
            timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
        ).sort_values("timestamp").reset_index(drop=True)

        t0 = full_df["timestamp"].iloc[0]
        t_split = t0 + pd.Timedelta(days=TRAIN_DAYS)
        holdout_df = full_df[full_df["timestamp"] >= t_split].reset_index(drop=True)

        actual_days = (holdout_df["timestamp"].iloc[-1] - holdout_df["timestamp"].iloc[0]).total_seconds() / 86400
        geo_kwargs = {"atr_tp": ATR_TP, "atr_sl": ATR_SL}

        result = simulate(booster, scaler, fc, holdout_df,
                          confidence_threshold=CONFIDENCE,
                          fee_pct=FEE, max_hold_bars=MAX_HOLD, **geo_kwargs)
        bias  = prediction_bias(booster, scaler, fc, holdout_df, CONFIDENCE)
        calib = confidence_calibration(booster, scaler, fc, holdout_df,
                                       fee_pct=FEE, max_hold=MAX_HOLD)

        if result:
            print_model_results(f"Model {level} (ATR 2/1 @ 0.85)", result, bias, calib, actual_days)
            holdout_results[level] = result
        else:
            print(f"\n  Model {level}: No trades generated.")

    # Comparison table
    print("\n\n" + "=" * 65)
    print("  SUMMARY COMPARISON vs Model B (frozen control)")
    print("=" * 65)
    print(f"  {'Model':<8} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>8} {'MaxDD%':>7} {'NetRet%':>9}")
    print("  " + "-" * 55)
    print(f"  {'B (ctrl)':<8} {103:>7} {24.3:>5.1f}% {0.39:>6.2f} {-4.23:>+8.2f} {-28.4:>6.1f}% {-28.4:>+9.1f}%")
    for level, r in holdout_results.items():
        print(f"  {level:<8} {r['trades']:>7} {r['win_rate']*100:>5.1f}% "
              f"{r['profit_factor']:>6.2f} {r['sharpe']:>+8.2f} "
              f"{r['max_dd_pct']:>6.1f}% {r['net_cum_ret']:>+9.1f}%")

if __name__ == "__main__":
    main()
