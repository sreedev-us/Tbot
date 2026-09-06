#!/usr/bin/env python3
"""
Walk-Forward A vs B Comparison
--------------------------------
Tests pre-trained Model A and Model B across:
  - Multiple chronological windows (no leakage)
  - Distinct market regime periods (trending, choppy, volatile, recovery)

Metrics: cumulative return, Sharpe, max drawdown, profit factor, win rate,
         trade count, precision, recall, F1.
"""

import json
import logging
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from alpha_engine.training_pipeline import (
    FeatureEngineer,
    TrainingDataConfig,
    TrainingDataGenerator,
    NON_FEATURE_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path: Path):
    """Load a saved model + scaler + feature list."""
    meta = json.loads((model_path / "metadata.json").read_text())
    feature_columns = meta["feature_columns"]

    scaler_path = model_path / "scaler.pkl"
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    json_model = model_path / "model.json"
    pkl_model = model_path / "model.pkl"
    if json_model.exists():
        booster = xgb.Booster()
        booster.load_model(str(json_model))
    else:
        with pkl_model.open("rb") as f:
            booster = pickle.load(f)

    return booster, scaler, feature_columns


@dataclass
class WindowResult:
    label: str
    period: str
    regime: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    cumulative_return: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate_window(
    booster, scaler, feature_columns,
    test_df: pd.DataFrame,
    label: str,
    period: str,
    regime: str,
    atr_tp_mult: float = 2.0,
    atr_sl_mult: float = 1.0,
    max_hold_bars: int = 20,
    confidence_threshold: float = 0.60,
    fee_pct: float = 0.10,     # 0.1% taker fee, round-trip = 0.2%
) -> Optional[WindowResult]:
    """
    Simulate Triple Barrier trading:
    - Enter on BUY/SELL signal at confidence >= threshold
    - Exit when TP, SL, or max_hold_bars is reached
    - No overlapping positions
    - Fee paid once per round-trip
    """
    test_df = test_df.reset_index(drop=True).copy()

    for col in feature_columns:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X = test_df[feature_columns]
    y = test_df["target"].values
    close = test_df["close"].values
    high  = test_df["high"].values
    low   = test_df["low"].values

    # Pre-compute ATR-14 for barrier sizing
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        probs = booster.predict(xgb.DMatrix(scaled))

    pnls, preds, actuals = [], [], []
    i = 0
    while i < len(test_df) - max_hold_bars:
        p_sell, p_hold, p_buy = probs[i][0], probs[i][1], probs[i][2]

        if p_buy > p_sell and p_buy >= confidence_threshold:
            direction = 1   # LONG
            pred_class = 2
        elif p_sell > p_buy and p_sell >= confidence_threshold:
            direction = -1  # SHORT
            pred_class = 0
        else:
            # Map actual label (-1/0/1) → (0/1/2)
            actual_mapped = {-1: 0, 0: 1, 1: 2}.get(int(y[i]), 1)
            preds.append(1)           # HOLD
            actuals.append(actual_mapped)
            i += 1
            continue

        entry = close[i]
        bar_atr = atr[i]
        tp_price = entry + direction * atr_tp_mult * bar_atr
        sl_price = entry - direction * atr_sl_mult * bar_atr

        trade_pnl_pct = None
        exit_bar = i + max_hold_bars  # default: timeout exit

        for j in range(i + 1, min(i + max_hold_bars + 1, len(test_df))):
            if direction == 1:   # LONG
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j; break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j; break
            else:               # SHORT
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j; break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j; break

        if trade_pnl_pct is None:  # timeout
            exit_p = close[min(exit_bar, len(close) - 1)]
            trade_pnl_pct = direction * (exit_p - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct  # round-trip fee (entry + exit)
        pnls.append(trade_pnl_pct)

        actual_mapped = {-1: 0, 0: 1, 1: 2}.get(int(y[i]), 1)
        preds.append(pred_class)
        actuals.append(actual_mapped)

        i = exit_bar + 1   # skip to after position is closed

    if not pnls:
        return None

    if not pnls:
        return None

    pnls = np.array(pnls)
    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)

    winning = (pnls > 0).sum()
    losing = (pnls <= 0).sum()
    win_rate = winning / len(pnls)

    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls <= 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-8)

    cum_return = pnls.sum()
    returns_frac = pnls / 100.0
    sharpe = float(np.sqrt(252) * np.mean(returns_frac) / (np.std(returns_frac) + 1e-8))

    cum_curve = (1 + returns_frac).cumprod()
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - running_max) / running_max
    max_dd = float(drawdowns.min())

    acc = (preds_arr == actuals_arr).mean()
    buy_pred = preds_arr == 2
    buy_act = actuals_arr == 2
    tp = ((buy_pred) & (buy_act)).sum()
    fp = ((buy_pred) & (~buy_act)).sum()
    fn = ((~buy_pred) & (buy_act)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return WindowResult(
        label=label, period=period, regime=regime,
        total_trades=len(pnls),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        cumulative_return=float(cum_return),
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = Path("data/live/BTC_USDT_1m_live.csv")
    models_base = Path("models")

    model_configs = {
        "A - XGBoost Alone (v3.0.0)":          models_base / "local-v3.0.0",
        "B - Server AI + XGBoost (benchmark-b)": models_base / "local-benchmark-b",
    }

    logger.info("Loading OHLCV data...")
    raw_df = pd.read_csv(data_path)
    logger.info("Loaded %d candles", len(raw_df))

    logger.info("Engineering features for walk-forward test...")
    config = TrainingDataConfig()
    generator = TrainingDataGenerator(config)
    # Use the full pipeline (features + labels) but no output file
    full_df = generator.generate_training_data(raw_df)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Feature set ready: %d rows, %d columns", len(full_df), len(full_df.columns))

    # -----------------------------------------------------------------------
    # Define chronological test windows with regime labels
    # Based on 180 days of 1m BTC data: approx Apr 2026 - Sep 2026
    # We carve out 4 non-overlapping 30-day windows at different times
    # -----------------------------------------------------------------------
    total_days = (full_df["timestamp"].iloc[-1] - full_df["timestamp"].iloc[0]).days
    logger.info("Data spans %d days", total_days)

    # Roughly split into quarters for regime labeling
    t0 = full_df["timestamp"].iloc[0]
    windows = [
        ("Q1 - Early Period",  t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45),  "trending"),
        ("Q2 - Mid Period",    t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90),  "choppy"),
        ("Q3 - Late Period",   t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135), "volatile"),
        ("Q4 - Final Period",  t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=179), "recovery"),
    ]

    all_results: list[WindowResult] = []

    for model_label, model_path in model_configs.items():
        logger.info("\n=== Loading: %s ===", model_label)
        try:
            booster, scaler, feature_columns = load_model(model_path)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path, exc)
            continue

        for period_name, start_ts, end_ts, regime in windows:
            window_df = full_df[
                (full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)
            ].copy()

            if len(window_df) < 200:
                logger.warning("Skipping %s — insufficient data (%d rows)", period_name, len(window_df))
                continue

            result = evaluate_window(
                booster, scaler, feature_columns,
                window_df, model_label, period_name, regime,
            )
            if result:
                all_results.append(result)
                logger.info(
                    "  %-35s | %-20s | Trades=%4d WinRate=%.1f%% Sharpe=%+.2f MaxDD=%.1f%% CumRet=%+.1f%% F1=%.3f",
                    model_label[:35], period_name,
                    result.total_trades, result.win_rate * 100,
                    result.sharpe, result.max_drawdown * 100,
                    result.cumulative_return, result.f1,
                )

    # -----------------------------------------------------------------------
    # Aggregate and print final report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WALK-FORWARD COMPARISON REPORT — A vs B")
    print("=" * 100)
    header = f"{'Model':<42} {'Trades':>6} {'WinRate':>8} {'ProfFact':>9} {'Sharpe':>7} {'MaxDD':>7} {'CumRet%':>8} {'F1':>6}"
    print(header)
    print("-" * 100)

    model_agg = {}
    for label in model_configs.keys():
        rows = [r for r in all_results if r.label == label]
        if not rows:
            continue
        agg = {
            "total_trades": sum(r.total_trades for r in rows),
            "win_rate":          np.mean([r.win_rate for r in rows]),
            "profit_factor":     np.mean([r.profit_factor for r in rows]),
            "sharpe":            np.mean([r.sharpe for r in rows]),
            "max_drawdown":      np.mean([r.max_drawdown for r in rows]),
            "cumulative_return": sum(r.cumulative_return for r in rows),
            "f1":                np.mean([r.f1 for r in rows]),
        }
        model_agg[label] = agg
        print(
            f"{label:<42} {agg['total_trades']:>6} "
            f"{agg['win_rate']*100:>7.1f}% "
            f"{agg['profit_factor']:>9.2f} "
            f"{agg['sharpe']:>+7.2f} "
            f"{agg['max_drawdown']*100:>6.1f}% "
            f"{agg['cumulative_return']:>+8.2f}% "
            f"{agg['f1']:>6.3f}"
        )

        # Per-regime breakdown
        for r in rows:
            print(
                f"  [{r.regime:<10}] {r.period:<20} "
                f"T={r.total_trades:>4} WR={r.win_rate*100:.1f}% "
                f"Sh={r.sharpe:+.2f} DD={r.max_drawdown*100:.1f}% "
                f"Ret={r.cumulative_return:+.1f}%"
            )
        print()

    # Save JSON report
    report = {label: agg for label, agg in model_agg.items()}
    out_path = Path("models/walkforward_ab_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", out_path)

    print("\n" + "=" * 100)
    if len(model_agg) == 2:
        labels = list(model_agg.keys())
        a, b = model_agg[labels[0]], model_agg[labels[1]]
        print("VERDICT:")
        winner = labels[1] if b["sharpe"] > a["sharpe"] and b["cumulative_return"] > a["cumulative_return"] else labels[0]
        print(f"  Sharpe: A={a['sharpe']:+.3f}  B={b['sharpe']:+.3f}")
        print(f"  CumRet: A={a['cumulative_return']:+.2f}%  B={b['cumulative_return']:+.2f}%")
        print(f"  MaxDD:  A={a['max_drawdown']*100:.2f}%  B={b['max_drawdown']*100:.2f}%")
        print(f"  => Leading candidate: {winner}")
    print("=" * 100)


if __name__ == "__main__":
    main()
