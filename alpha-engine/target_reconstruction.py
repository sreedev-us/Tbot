#!/usr/bin/env python3
"""
Target Reconstruction Experiment — Phase 4 Step 3

Replaces Triple Barrier labeling with a regime-neutral fixed-horizon
directional target.

Triple Barrier target problem:
  - Label = "which barrier gets hit first"
  - In mean-reverting training: extended move → SL hit → SELL
  - In trending holdout: extended move → TP hit → BUY
  - Same feature state → opposite labels = concept drift baked into the target

New target (fixed-horizon directional return):
  - At time T, look forward H bars
  - Compute: future_return = (close[T+H] - close[T]) / close[T]
  - If future_return > +threshold → UP (label 2)
  - If future_return < -threshold → DOWN (label 0)
  - Otherwise → HOLD (label 1)

This is regime-neutral because it measures what actually happened,
not which barrier was hit given a specific geometry.

We test multiple horizons and thresholds to understand sensitivity.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from alpha_engine.model_trainer import ModelTrainer
from walkforward_ab import load_model
from validate_htf import simulate

logging.basicConfig(level=logging.WARNING)

DATA_PATH  = Path("data/live/BTC_USDT_1h_live.csv")
ABLATION   = "D2c"   # Use best D2 feature set
TRAIN_DAYS = 142
ATR_TP     = 2.0
ATR_SL     = 1.0
CONFIDENCE = 0.85
MAX_HOLD   = 24
FEE        = 0.20


def generate_fixed_horizon_labels(df, horizon=24, threshold_pct=0.50):
    """
    Generate fixed-horizon directional labels.

    horizon:       number of bars to look forward
    threshold_pct: minimum return % to qualify as UP/DOWN (not HOLD)

    Returns df with 'target' column: -1=DOWN, 0=HOLD, 1=UP
    (ModelTrainer.train() shifts by +1 internally to get {0, 1, 2})
    """
    df = df.copy()
    close = df["close"].values
    n = len(close)
    labels = np.zeros(n, dtype=int)   # default: HOLD = 0

    for i in range(n - horizon):
        future_ret = (close[i + horizon] - close[i]) / close[i] * 100.0
        if future_ret > threshold_pct:
            labels[i] = 1    # UP
        elif future_ret < -threshold_pct:
            labels[i] = -1   # DOWN

    df.loc[:, "target"] = labels
    df = df.iloc[:-horizon].reset_index(drop=True)
    return df



def train_with_target(labeled_df, feature_cols, model_name):
    X = labeled_df[feature_cols]
    y = labeled_df["target"]
    trainer = ModelTrainer("./models", "xgboost")
    trainer.train(X, y)
    import datetime
    trainer.save_model(model_name, metadata={
        "feature_columns": feature_cols,
        "ablation_level": ABLATION,
        "description": f"Fixed-horizon target {model_name}",
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    return load_model(Path("models") / model_name)


def print_result(label, r, bias, days):
    if r is None:
        print(f"  {label}: No trades")
        return
    print(f"  {label}")
    print(f"    Trades: {r['trades']} ({r['trades']/days:.1f}/day)  "
          f"WR: {r['win_rate']*100:.1f}%  PF: {r['profit_factor']:.2f}  "
          f"Sharpe: {r['sharpe']:+.2f}  MaxDD: {r['max_dd_pct']:+.1f}%  "
          f"NetRet: {r['net_cum_ret']:+.1f}%")
    print(f"    BUY {bias['buy_pct']:.1f}% | SELL {bias['sell_pct']:.1f}% | UP {bias['up_pct']:.1f}%")


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
        "hold_pct": (~(((p_up > p_down) & (p_up >= confidence)) | ((p_down > p_up) & (p_down >= confidence)))).sum() / total * 100,
        "up_pct":   up_pct,
        "dn_pct":   100 - up_pct,
    }


def confidence_calibration(booster, scaler, fc, df, fee_pct=0.20, max_hold=24, atr_tp=2.0, atr_sl=1.0):
    df = df.reset_index(drop=True)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
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
    for lo, hi, label in [(0.85, 0.90, "0.85-0.90"), (0.90, 0.95, "0.90-0.95"), (0.95, 1.01, "0.95-1.00")]:
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
            pnl = None
            exit_bar = i + max_hold
            for j in range(i + 1, min(i + max_hold + 1, len(df))):
                if direction == 1:
                    if high[j] >= tp_price:
                        pnl = (tp_price - entry) / entry * 100
                        exit_bar = j
                        break
                    if low[j] <= sl_price:
                        pnl = (sl_price - entry) / entry * 100
                        exit_bar = j
                        break
                else:
                    if low[j] <= tp_price:
                        pnl = (entry - tp_price) / entry * 100
                        exit_bar = j
                        break
                    if high[j] >= sl_price:
                        pnl = (entry - sl_price) / entry * 100
                        exit_bar = j
                        break
            if pnl is None:
                exit_price = close[min(exit_bar, len(close) - 1)]
                pnl = direction * (exit_price - entry) / entry * 100
            pnls.append(pnl - fee_pct)

        if pnls:
            arr = np.array(pnls)
            results[label] = {
                "count": len(arr),
                "win_pct": (arr > 0).mean() * 100,
                "avg_ret": arr.mean(),
            }
        else:
            results[label] = {"count": 0, "win_pct": 0.0, "avg_ret": 0.0}
    return results


def print_calibration(calib):
    print("    Confidence Calibration:")
    for bucket, m in calib.items():
        print(
            f"      {bucket}: count={m['count']:>4}, "
            f"win%={m['win_pct']:>5.1f}%, avg_ret={m['avg_ret']:>+.3f}%"
        )


def main():
    raw_df = pd.read_csv(DATA_PATH)
    config = TrainingDataConfig()
    gen    = TrainingDataGenerator(config)

    # Engineer features (D2c, no labels yet)
    full_feats = gen.feature_engineer.engineer_features(raw_df, ablation_level=ABLATION)
    full_feats = full_feats.assign(
        timestamp=pd.to_datetime(full_feats["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    t0      = full_feats["timestamp"].iloc[0]
    t_split = t0 + pd.Timedelta(days=TRAIN_DAYS)

    train_feats   = full_feats[full_feats["timestamp"] <  t_split].reset_index(drop=True)
    holdout_feats = full_feats[full_feats["timestamp"] >= t_split].reset_index(drop=True)
    holdout_days  = (holdout_feats["timestamp"].iloc[-1] - holdout_feats["timestamp"].iloc[0]).total_seconds() / 86400

    print("=" * 70)
    print("  TARGET RECONSTRUCTION EXPERIMENT")
    print(f"  Training: {len(train_feats)} bars | Holdout: {len(holdout_feats)} bars ({holdout_days:.0f} days)")
    print("=" * 70)

    # Control: D2c with Triple Barrier target (already trained as local-model-d2c-1h)
    print("\n[CONTROL] D2c + Triple Barrier target (already evaluated: NetRet -18.5%)")
    print("  BUY 8.7% | SELL 52.2% | UP 51.1%")

    # Test grid: horizon × threshold
    configs = [
        {"horizon": 12, "threshold": 0.40, "name": "FH-12h-0.4pct"},
        {"horizon": 24, "threshold": 0.60, "name": "FH-24h-0.6pct"},
        {"horizon": 24, "threshold": 1.00, "name": "FH-24h-1.0pct"},
        {"horizon": 48, "threshold": 1.00, "name": "FH-48h-1.0pct"},
    ]

    geo_kwargs = {"atr_tp": ATR_TP, "atr_sl": ATR_SL}
    feature_cols = gen.get_feature_columns(train_feats)

    print("\n" + "=" * 70)
    print("  FIXED-HORIZON TARGET VARIANTS")
    print("=" * 70)

    for cfg in configs:
        h   = cfg["horizon"]
        thr = cfg["threshold"]
        nm  = cfg["name"]

        # Label training data with fixed-horizon target
        labeled_train = generate_fixed_horizon_labels(train_feats, horizon=h, threshold_pct=thr)
        if len(labeled_train) < 50:
            print(f"\n  {nm}: insufficient samples")
            continue

        n_up   = (labeled_train["target"] == 1).sum()
        n_dn   = (labeled_train["target"] == -1).sum()
        n_hold = (labeled_train["target"] == 0).sum()
        total  = len(labeled_train)

        print(f"\n  [{nm}] horizon={h}h, threshold={thr}%")
        print(f"    Labels: UP={n_up/total*100:.1f}%  DOWN={n_dn/total*100:.1f}%  HOLD={n_hold/total*100:.1f}%")

        X = labeled_train[feature_cols]
        y = labeled_train["target"]
        trainer = ModelTrainer("./models", "xgboost")
        trainer.train(X, y)
        import datetime
        trainer.save_model(nm, metadata={
            "feature_columns": feature_cols,
            "ablation_level": ABLATION,
            "description": f"Fixed-horizon target: horizon={h}, threshold={thr}%",
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        booster, scaler, fc = load_model(Path("models") / nm)

        # Evaluate on holdout using the ATR 2/1 trading geometry
        r    = simulate(booster, scaler, fc, holdout_feats, confidence_threshold=CONFIDENCE,
                        fee_pct=FEE, max_hold_bars=MAX_HOLD, **geo_kwargs)
        bias = prediction_bias(booster, scaler, fc, holdout_feats, CONFIDENCE)
        calib = confidence_calibration(
            booster, scaler, fc, holdout_feats,
            fee_pct=FEE, max_hold=MAX_HOLD,
            atr_tp=ATR_TP, atr_sl=ATR_SL,
        )
        print_result(f"  Holdout", r, bias, holdout_days)
        print(f"    HOLD {bias['hold_pct']:.1f}% | Actual DN {bias['dn_pct']:.1f}%")
        print_calibration(calib)


if __name__ == "__main__":
    main()
