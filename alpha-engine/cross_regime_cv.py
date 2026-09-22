#!/usr/bin/env python3
"""
Cross-Regime Transfer Matrix
=============================
For each of 4 regimes, train a model on that regime alone and evaluate
on all other regimes. Reports directional expectancy, bias ratio,
confidence calibration, PF, Sharpe, and MaxDD.

This tests: "Does predictive signal from one regime transfer to another?"

We are looking for SYSTEMATIC transfer patterns, not perfection.
Even one or two failing transitions is informative.

Regimes (from the 180-day 1h dataset):
  Q1 Trending  (days   0– 45)  2026-03-10 → 2026-04-24
  Q2 Choppy    (days  45– 90)  2026-04-24 → 2026-06-08
  Q3 Volatile  (days  90–135)  2026-06-08 → 2026-07-23
  Q4 Recovery  (days 135–180)  2026-07-23 → 2026-09-06

Note: August-September period is the DIAGNOSTIC holdout, now labeled as such.
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
DATA_PATH  = Path("data/live/BTC_USDT_1h_live.csv")

# We use D2c features (best performing in D2 ablation) for the transfer matrix
ABLATION_LEVEL = "D2c"


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
    up_pct = (close[1:] > close[:-1]).sum() / max(len(close)-1, 1) * 100
    return {
        "buy_pct":  buy_mask.sum() / total * 100,
        "sell_pct": sell_mask.sum() / total * 100,
        "hold_pct": (~(buy_mask | sell_mask)).sum() / total * 100,
        "up_pct":   up_pct,
        "dn_pct":   100 - up_pct,
    }


def confidence_buckets(booster, scaler, fc, df, fee=0.20, max_hold=24,
                        atr_tp=2.0, atr_sl=1.0):
    """Win% and avg return per confidence bucket."""
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
                "n": len(pnls),
                "wr": (pnls > 0).mean() * 100,
                "avg": pnls.mean(),
            }
        else:
            results[f"{lo:.2f}-{hi:.2f}"] = {"n": 0, "wr": 0, "avg": 0}
    return results


def train_on_regime(train_df_raw, regime_label, gen):
    """Train a model on a single regime's raw data. Returns (booster, scaler, fc)."""
    model_name = f"cross_regime_{regime_label.lower().replace(' ', '_')}"
    labeled = gen.generate_training_data(train_df_raw, ablation_level=ABLATION_LEVEL)
    if len(labeled) < 50:
        return None, None, None, 0

    fc = gen.get_feature_columns(labeled)
    X  = labeled[fc]
    y  = labeled["target"]
    trainer = ModelTrainer("./models", "xgboost")
    trainer.train(X, y)
    trainer.save_model(model_name, metadata={
        "feature_columns": fc,
        "ablation_level": ABLATION_LEVEL,
        "description": f"Cross-regime train: {regime_label}",
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    booster, scaler, fc_saved = load_model(Path("models") / model_name)
    return booster, scaler, fc_saved, len(labeled)


def main():
    raw_df = pd.read_csv(DATA_PATH)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    config = TrainingDataConfig()
    gen    = TrainingDataGenerator(config)

    t0 = raw_df["timestamp"].iloc[0]
    regimes = [
        ("Q1-Trending",  t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45)),
        ("Q2-Choppy",    t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90)),
        ("Q3-Volatile",  t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135)),
        ("Q4-Recovery",  t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=180)),
    ]

    # Feature-engineer full dataset once for evaluation windows
    full_feats = gen.feature_engineer.engineer_features(raw_df, ablation_level=ABLATION_LEVEL)
    full_feats = full_feats.assign(
        timestamp=pd.to_datetime(full_feats["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    # Train one model per regime
    trained = {}
    print("=" * 70)
    print("  TRAINING — one model per regime")
    print("=" * 70)
    for label, start, end in regimes:
        raw_slice = raw_df[
            (raw_df["timestamp"] >= start) & (raw_df["timestamp"] < end)
        ].reset_index(drop=True)

        print(f"  Training on {label} ({len(raw_slice)} raw bars)...")
        b, s, fc, n_samples = train_on_regime(raw_slice, label, gen)
        if b is not None:
            trained[label] = (b, s, fc, n_samples)
            print(f"    Samples: {n_samples}, Features: {len(fc)}")
        else:
            print(f"    SKIP: insufficient data for {label}")

    # Evaluation: each trained model tested on every regime window.
    # This matrix is observational only; do not retune D2 from these results.
    print("\n\n" + "=" * 70)
    print("  CROSS-REGIME TRANSFER MATRIX")
    print(f"  Feature set: {ABLATION_LEVEL}  |  ATR 2/1 @ 0.85  |  0.20% cost")
    print("=" * 70)

    # Summary table accumulator
    matrix_rows = []

    for train_label, (b, s, fc, _) in trained.items():
        print(f"\n  Trained on: {train_label}")
        print(f"  {'Test Regime':<16} {'T':>4} {'WR%':>6} {'Exp%':>7} {'PF':>6} "
              f"{'Sharpe':>7} {'DD%':>6} {'Ret%':>8} {'BUY/UP':>13} "
              f"{'SELL/DN':>13} {'HOLD%':>7}")
        print("  " + "-" * 112)

        for test_label, test_start, test_end in regimes:
            # Test on this regime's feature-engineered data
            test_df = full_feats[
                (full_feats["timestamp"] >= test_start) &
                (full_feats["timestamp"] < test_end)
            ].reset_index(drop=True)

            if len(test_df) < 30:
                print(f"  {test_label:<16} SKIP (insufficient bars)")
                continue

            test_days = (test_df["timestamp"].iloc[-1] - test_df["timestamp"].iloc[0]).total_seconds() / 86400

            r = simulate(b, s, fc, test_df,
                         confidence_threshold=CONFIDENCE,
                         fee_pct=FEE, max_hold_bars=MAX_HOLD,
                         atr_tp=ATR_TP, atr_sl=ATR_SL)
            bias = prediction_bias(b, s, fc, test_df, CONFIDENCE)
            calib = confidence_buckets(
                b, s, fc, test_df,
                fee=FEE, max_hold=MAX_HOLD,
                atr_tp=ATR_TP, atr_sl=ATR_SL,
            )

            same = "(same)" if train_label == test_label else ""
            if r:
                mark = " ***" if r["net_cum_ret"] > 0 else ""
                print(f"  {test_label:<16} {r['trades']:>4} {r['win_rate']*100:>5.1f}% "
                      f"{r['avg_pnl_per_trade']:>+6.3f}% {r['profit_factor']:>6.2f} {r['sharpe']:>+7.2f} "
                      f"{r['max_dd_pct']:>5.1f}% {r['net_cum_ret']:>+8.1f}% "
                      f"{bias['buy_pct']:>5.1f}/{bias['up_pct']:<5.1f} "
                      f"{bias['sell_pct']:>5.1f}/{bias['dn_pct']:<5.1f} "
                      f"{bias['hold_pct']:>6.1f}%{mark} {same}")
                bucket_text = []
                for bucket in ["0.85-0.90", "0.90-0.95", "0.95-1.01"]:
                    m = calib[bucket]
                    label = "0.95-1.00" if bucket == "0.95-1.01" else bucket
                    bucket_text.append(
                        f"{label}: n={m['n']}, wr={m['wr']:.1f}%, avg={m['avg']:+.3f}%"
                    )
                print("    Confidence: " + " | ".join(bucket_text))
                matrix_rows.append({
                    "train": train_label, "test": test_label,
                    "trades": r["trades"], "wr": r["win_rate"]*100,
                    "pf": r["profit_factor"], "sharpe": r["sharpe"],
                    "dd": r["max_dd_pct"], "ret": r["net_cum_ret"],
                    "expectancy": r["avg_pnl_per_trade"],
                    "buy_pct": bias["buy_pct"], "sell_pct": bias["sell_pct"],
                    "hold_pct": bias["hold_pct"], "up_pct": bias["up_pct"],
                    "dn_pct": bias["dn_pct"], "confidence": calib,
                })
            else:
                print(f"  {test_label:<16}    0 trades  {same}")

    # Summary: profitable transfer pairs only
    print("\n\n" + "=" * 70)
    print("  PROFITABLE CROSS-REGIME TRANSFERS (Net Return > 0)")
    print("=" * 70)
    profitable = [r for r in matrix_rows if r["ret"] > 0 and r["train"] != r["test"]]
    if profitable:
        print(f"  {'Train':<16} {'Test':<16} {'Exp%':>7} {'PF':>6} {'Sharpe':>7} {'Ret%':>8}")
        print("  " + "-" * 55)
        for r in sorted(profitable, key=lambda x: x["ret"], reverse=True):
            print(f"  {r['train']:<16} {r['test']:<16} "
                  f"{r['expectancy']:>+6.3f}% {r['pf']:>6.2f} {r['sharpe']:>+7.2f} {r['ret']:>+8.1f}%")
    else:
        print("  No profitable cross-regime transfers found.")

    # Directional bias summary
    print("\n\n" + "=" * 70)
    print("  DIRECTIONAL BIAS SUMMARY (BUY% vs Actual UP%)")
    print("  If BUY% is far below UP%, the model is systematically short-biased")
    print("=" * 70)
    print(f"  {'Train':<16} {'Test':<16} {'BUY%':>6} {'SELL%':>7} {'HOLD%':>7} {'UP%':>6} {'DN%':>6} {'Bias'}")
    print("  " + "-" * 65)
    for r in matrix_rows:
        if r["train"] == r["test"]:
            continue
        gap = r["buy_pct"] - r["up_pct"]
        bias_label = "SELL-biased" if gap < -15 else ("BUY-biased" if gap > 15 else "Neutral")
        print(f"  {r['train']:<16} {r['test']:<16} {r['buy_pct']:>5.1f}% "
              f"{r['sell_pct']:>6.1f}% {r['hold_pct']:>6.1f}% "
              f"{r['up_pct']:>5.1f}% {r['dn_pct']:>5.1f}%  {bias_label}")

    transfer_rows = [r for r in matrix_rows if r["train"] != r["test"]]
    profitable_transfers = [r for r in transfer_rows if r["ret"] > 0]
    avg_expectancy = np.mean([r["expectancy"] for r in transfer_rows]) if transfer_rows else 0.0
    avg_return = np.mean([r["ret"] for r in transfer_rows]) if transfer_rows else 0.0

    print("\n\n" + "=" * 70)
    print("  OBSERVATIONAL DECISION GATE")
    print("  Criterion: systematic transfer, not every window profitable")
    print("=" * 70)
    print(f"  Profitable transfer windows: {len(profitable_transfers)}/{len(transfer_rows)}")
    print(f"  Average directional expectancy: {avg_expectancy:+.3f}% per trade")
    print(f"  Average transfer net return: {avg_return:+.1f}%")
    if profitable_transfers and avg_expectancy > 0 and avg_return > 0:
        print("  Read: D2 shows systematic transfer. Reserve a future holdout before F&G/NLP.")
    else:
        print("  Read: D2 does not show systematic transfer. Prioritize target reconstruction before D3.")


if __name__ == "__main__":
    main()
