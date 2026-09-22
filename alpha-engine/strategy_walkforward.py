"""
strategy_walkforward.py
=======================
Chronological Walk-Forward Simulation of the Integrated Multi-Strategy Engine.

At each bar:
  1. Detect market regime (priority-ordered RegimeDetector)
  2. Query StrategySelector for eligible strategies in that regime
  3. Evaluate signals in serialized order (first non-zero signal acts)
  4. Manage trade through barrier exit (TP / SL / max_hold) with 0.20% fee
  5. Record per-strategy attribution and overall portfolio curve

Benchmarks against:
  - Frozen Model B standalone (control)
  - Buy & Hold BTC
Evaluates both:
  - Full 180-day period
  - Diagnostic Holdout period (Aug 1 - Sep 6, where Model B lost -28.4%)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_engine.regime_detector import RegimeDetector
from alpha_engine.strategy_selector import StrategySelector
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.base_strategy import BaseStrategy
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.registry import StrategyRegistry
from strategies.trend_following import TrendFollowingStrategy

logging.basicConfig(level=logging.WARNING)


def run_multi_strategy_simulation(
    df: pd.DataFrame,
    regimes: pd.Series,
    selector: StrategySelector,
    fee_pct: float = 0.20,
    start_idx: int = 55,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """Run chronological walk-forward with multi-strategy regime selection."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df) if end_idx is None else min(len(df), end_idx)

    # Precompute ATR for each strategy's risk parameters
    atr_series = BaseStrategy._atr(df, period=14).values

    equity = 1.0
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []

    i = start_idx
    while i < n - 36:
        regime = regimes.iloc[i]
        window_df = df.iloc[: i + 1]

        selection = selector.evaluate(window_df, regime)
        if selection is None or selection.signal == 0:
            i += 1
            continue

        sig = selection.signal
        strat_name = selection.strategy_name
        risk_p = selection.risk_params
        max_hold = int(risk_p.get("max_hold_bars", 24))
        atr_tp = float(risk_p.get("atr_tp_mult", 2.0))
        atr_sl = float(risk_p.get("atr_sl_mult", 1.0))

        entry = close[i]
        bar_atr = atr_series[i]
        tp_price = entry + sig * atr_tp * bar_atr
        sl_price = entry - sig * atr_sl * bar_atr

        trade_pnl_pct = None
        exit_bar = min(i + max_hold, n - 1)
        exit_reason = "max_hold"

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if sig == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j
                    exit_reason = "tp"
                    break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j
                    exit_reason = "sl"
                    break
            else:
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j
                    exit_reason = "tp"
                    break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j
                    exit_reason = "sl"
                    break

        if trade_pnl_pct is None:
            exit_p = close[exit_bar]
            trade_pnl_pct = sig * (exit_p - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct
        equity *= 1.0 + trade_pnl_pct / 100.0
        equity_curve.append(equity)

        trades.append({
            "bar_entry": i,
            "bar_exit": exit_bar,
            "strategy": strat_name,
            "regime": regime,
            "signal": "BUY" if sig == 1 else "SELL",
            "pnl_pct": trade_pnl_pct,
            "exit_reason": exit_reason,
        })

        i = exit_bar + 1

    return _compute_metrics(trades, equity_curve)


def run_standalone_model_b(
    df: pd.DataFrame,
    fee_pct: float = 0.20,
    start_idx: int = 55,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """Run standalone frozen Model B (unconditional, no regime filter)."""
    strat = MeanReversionStrategy()
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df) if end_idx is None else min(len(df), end_idx)
    atr_series = BaseStrategy._atr(df, period=14).values

    equity = 1.0
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []

    i = start_idx
    while i < n - 36:
        window_df = df.iloc[: i + 1]
        sig = strat.signal(window_df)
        if sig == 0:
            i += 1
            continue

        entry = close[i]
        bar_atr = atr_series[i]
        tp_price = entry + sig * 2.0 * bar_atr
        sl_price = entry - sig * 1.0 * bar_atr

        trade_pnl_pct = None
        exit_bar = min(i + 24, n - 1)
        exit_reason = "max_hold"

        for j in range(i + 1, min(i + 25, n)):
            if sig == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j
                    exit_reason = "tp"
                    break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j
                    exit_reason = "sl"
                    break
            else:
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j
                    exit_reason = "tp"
                    break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j
                    exit_reason = "sl"
                    break

        if trade_pnl_pct is None:
            exit_p = close[exit_bar]
            trade_pnl_pct = sig * (exit_p - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct
        equity *= 1.0 + trade_pnl_pct / 100.0
        equity_curve.append(equity)

        trades.append({
            "bar_entry": i,
            "bar_exit": exit_bar,
            "strategy": "model_b_standalone",
            "signal": "BUY" if sig == 1 else "SELL",
            "pnl_pct": trade_pnl_pct,
            "exit_reason": exit_reason,
        })

        i = exit_bar + 1

    return _compute_metrics(trades, equity_curve)


def _compute_metrics(trades: list[dict[str, Any]], equity_curve: list[float]) -> dict[str, Any]:
    if not trades:
        return {
            "trades_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "net_ret": 0.0,
            "trades": [],
        }

    pnls = np.array([t["pnl_pct"] for t in trades])
    eq = np.array(equity_curve)
    winning = (pnls > 0).sum()
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100.0

    gross_profit = pnls[pnls > 0].sum() if winning > 0 else 0.0
    gross_loss = abs(pnls[pnls <= 0].sum()) if (len(pnls) - winning) > 0 else 0.0

    # Attribution per strategy
    strat_attr: dict[str, dict[str, Any]] = {}
    for t in trades:
        s = t["strategy"]
        if s not in strat_attr:
            strat_attr[s] = {"trades": 0, "wins": 0, "pnl_sum": 0.0}
        strat_attr[s]["trades"] += 1
        if t["pnl_pct"] > 0:
            strat_attr[s]["wins"] += 1
        strat_attr[s]["pnl_sum"] += t["pnl_pct"]

    return {
        "trades_count": len(trades),
        "win_rate": winning / len(trades) * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "sharpe": (pnls.mean() / pnls.std() * np.sqrt(len(pnls))) if pnls.std() > 1e-8 else 0.0,
        "max_dd_pct": dd.min(),
        "net_ret": (eq[-1] - eq[0]) / eq[0] * 100.0,
        "attribution": strat_attr,
        "trades": trades,
    }


def main():
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    raw_df = pd.read_csv(data_path)

    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    detector = RegimeDetector()
    regimes = detector.detect_series(full_df)

    # Initialize registry & selector
    # Register instances
    registry = StrategyRegistry
    registry.reset()
    selector = StrategySelector(registry)

    # Define holdout boundary (142 days in)
    t0 = full_df["timestamp"].iloc[0]
    t_split = t0 + pd.Timedelta(days=142)
    split_idx = (full_df["timestamp"] < t_split).sum()

    print("=" * 80)
    print("  PHASE 5: MULTI-STRATEGY CHRONOLOGICAL WALK-FORWARD")
    print(f"  Total Data: {len(full_df)} bars ({full_df['timestamp'].iloc[0].date()} to {full_df['timestamp'].iloc[-1].date()})")
    print(f"  Holdout Split: Bar {split_idx} ({t_split.date()})")
    print("=" * 80)

    # 1. Full 180 Days Walk-Forward
    print("\n[A] FULL 180-DAY WALK-FORWARD")
    res_full_multi = run_multi_strategy_simulation(full_df, regimes, selector, start_idx=55)
    res_full_model_b = run_standalone_model_b(full_df, start_idx=55)

    print(f"\n  MULTI-STRATEGY SYSTEM:")
    print(f"    Trades:        {res_full_multi['trades_count']}")
    print(f"    Win Rate:      {res_full_multi['win_rate']:.1f}%")
    print(f"    Profit Factor: {res_full_multi['profit_factor']:.2f}")
    print(f"    Sharpe:        {res_full_multi['sharpe']:+.2f}")
    print(f"    Max DD:        {res_full_multi['max_dd_pct']:+.1f}%")
    print(f"    Net Return:    {res_full_multi['net_ret']:+.1f}%")
    print(f"    Attribution:")
    for s, attr in res_full_multi.get("attribution", {}).items():
        w_pct = attr["wins"] / attr["trades"] * 100 if attr["trades"] > 0 else 0
        print(f"      - {s:<18}: {attr['trades']:>4} trades | Win%: {w_pct:>5.1f}% | PnL sum: {attr['pnl_sum']:>+6.1f}%")

    print(f"\n  MODEL B STANDALONE (Frozen Control):")
    print(f"    Trades:        {res_full_model_b['trades_count']}")
    print(f"    Win Rate:      {res_full_model_b['win_rate']:.1f}%")
    print(f"    Profit Factor: {res_full_model_b['profit_factor']:.2f}")
    print(f"    Sharpe:        {res_full_model_b['sharpe']:+.2f}")
    print(f"    Max DD:        {res_full_model_b['max_dd_pct']:+.1f}%")
    print(f"    Net Return:    {res_full_model_b['net_ret']:+.1f}%")

    # 2. Diagnostic Holdout Period (Aug 1 - Sep 6)
    print("\n" + "-" * 80)
    print("[B] DIAGNOSTIC HOLDOUT PERIOD (Days 142–180: Aug 1 - Sep 6, 2026)")
    print("    *Where Model B previously suffered -28.4% Net Return / 0.39 PF*")
    print("-" * 80)

    res_holdout_multi = run_multi_strategy_simulation(full_df, regimes, selector, start_idx=split_idx)
    res_holdout_model_b = run_standalone_model_b(full_df, start_idx=split_idx)

    btc_h_ret = (full_df["close"].iloc[-1] - full_df["close"].iloc[split_idx]) / full_df["close"].iloc[split_idx] * 100

    print(f"\n  BENCHMARK (BTC Buy & Hold): {btc_h_ret:+.1f}%")
    print(f"\n  MULTI-STRATEGY SYSTEM ON HOLDOUT:")
    print(f"    Trades:        {res_holdout_multi['trades_count']}")
    print(f"    Win Rate:      {res_holdout_multi['win_rate']:.1f}%")
    print(f"    Profit Factor: {res_holdout_multi['profit_factor']:.2f}")
    print(f"    Sharpe:        {res_holdout_multi['sharpe']:+.2f}")
    print(f"    Max DD:        {res_holdout_multi['max_dd_pct']:+.1f}%")
    print(f"    Net Return:    {res_holdout_multi['net_ret']:+.1f}%")
    print(f"    Attribution on Holdout:")
    for s, attr in res_holdout_multi.get("attribution", {}).items():
        w_pct = attr["wins"] / attr["trades"] * 100 if attr["trades"] > 0 else 0
        print(f"      - {s:<18}: {attr['trades']:>4} trades | Win%: {w_pct:>5.1f}% | PnL sum: {attr['pnl_sum']:>+6.1f}%")

    print(f"\n  MODEL B STANDALONE ON HOLDOUT (Control):")
    print(f"    Trades:        {res_holdout_model_b['trades_count']}")
    print(f"    Win Rate:      {res_holdout_model_b['win_rate']:.1f}%")
    print(f"    Profit Factor: {res_holdout_model_b['profit_factor']:.2f}")
    print(f"    Sharpe:        {res_holdout_model_b['sharpe']:+.2f}")
    print(f"    Max DD:        {res_holdout_model_b['max_dd_pct']:+.1f}%")
    print(f"    Net Return:    {res_holdout_model_b['net_ret']:+.1f}%")


if __name__ == "__main__":
    main()
