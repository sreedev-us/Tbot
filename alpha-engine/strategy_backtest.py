"""
strategy_backtest.py
====================
Phase 5 Regime-Strategy Matrix Measurement.

Backtests each candidate strategy independently across each market regime
on the 180-day 1h BTC/USDT dataset.

Produces the observational matrix:
  Regime × Strategy (Mean Reversion, Trend Following, Breakout)

Metrics per cell:
  - Trades (count, trades/day)
  - Win Rate (%)
  - Profit Factor
  - Sharpe Ratio
  - Max Drawdown (%)
  - Net Cumulative Return (%)
  - Directional Bias (BUY% vs SELL%)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_engine.regime_detector import RegimeDetector
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.base_strategy import BaseStrategy
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend_following import TrendFollowingStrategy

logging.basicConfig(level=logging.WARNING)


def simulate_strategy_in_regimes(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    regimes: pd.Series,
    target_regime: str | None = None,
    fee_pct: float = 0.20,
) -> dict[str, Any] | None:
    """
    Simulate a strategy on bars where regime == target_regime (or all if target_regime is None).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # Pre-calculate ATR for risk parameters
    atr = strategy._atr(df, period=14).values

    pnls: list[float] = []
    buy_signals = 0
    sell_signals = 0
    equity = 1.0
    equity_curve = [equity]

    # Pre-determine min lookback
    min_lookback = 55

    i = min_lookback
    while i < n - 36:
        # Check regime filter if specified
        current_regime = regimes.iloc[i]
        if target_regime is not None and current_regime != target_regime:
            i += 1
            continue

        # Evaluate signal on window up to bar i
        window_df = df.iloc[: i + 1]
        sig = strategy.signal(window_df)

        if sig == 0:
            i += 1
            continue

        if sig == 1:
            buy_signals += 1
        elif sig == -1:
            sell_signals += 1

        risk_p = strategy.risk_parameters(window_df)
        max_hold = int(risk_p.get("max_hold_bars", 24))
        atr_tp = float(risk_p.get("atr_tp_mult", 2.0))
        atr_sl = float(risk_p.get("atr_sl_mult", 1.0))

        entry = close[i]
        bar_atr = atr[i]
        tp_price = entry + sig * atr_tp * bar_atr
        sl_price = entry - sig * atr_sl * bar_atr

        trade_pnl_pct = None
        exit_bar = min(i + max_hold, n - 1)

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if sig == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j
                    break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j
                    break
            else:  # sig == -1
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j
                    break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j
                    break

        if trade_pnl_pct is None:
            exit_p = close[exit_bar]
            trade_pnl_pct = sig * (exit_p - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct
        equity *= 1.0 + trade_pnl_pct / 100.0
        equity_curve.append(equity)
        pnls.append(trade_pnl_pct)

        # Move to bar after exit to prevent overlapping position of the same strategy
        i = exit_bar + 1

    if not pnls:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "net_ret": 0.0,
            "buy_pct": 0.0,
            "sell_pct": 0.0,
        }

    pnls_arr = np.array(pnls)
    eq_arr = np.array(equity_curve)
    winning = (pnls_arr > 0).sum()
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100.0

    gross_profit = pnls_arr[pnls_arr > 0].sum() if winning > 0 else 0.0
    gross_loss = abs(pnls_arr[pnls_arr <= 0].sum()) if (len(pnls_arr) - winning) > 0 else 0.0
    total_signals = buy_signals + sell_signals

    return {
        "trades": len(pnls),
        "win_rate": winning / len(pnls) * 100.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "sharpe": (pnls_arr.mean() / pnls_arr.std() * np.sqrt(len(pnls_arr))) if pnls_arr.std() > 1e-8 else 0.0,
        "max_dd_pct": dd.min(),
        "net_ret": (eq_arr[-1] - eq_arr[0]) / eq_arr[0] * 100.0,
        "buy_pct": (buy_signals / total_signals * 100.0) if total_signals > 0 else 0.0,
        "sell_pct": (sell_signals / total_signals * 100.0) if total_signals > 0 else 0.0,
    }


def main():
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading data from {data_path}...")
    raw_df = pd.read_csv(data_path)

    # Feature engineer once so Model B has access to all feature columns
    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Dataset: {len(full_df)} bars ({full_df['timestamp'].iloc[0].date()} to {full_df['timestamp'].iloc[-1].date()})")

    # Detect regimes across all bars
    detector = RegimeDetector()
    regimes = detector.detect_series(full_df)

    print("\nRegime Distribution:")
    regime_counts = regimes.value_counts()
    for reg, count in regime_counts.items():
        print(f"  {reg:<12}: {count:>5} bars ({count / len(regimes) * 100:>5.1f}%)")

    # Instantiate strategies
    strategies: list[BaseStrategy] = [
        MeanReversionStrategy(),
        TrendFollowingStrategy(),
        BreakoutStrategy(),
    ]

    target_regimes = ["range", "uptrend", "downtrend", "high_vol", "low_vol", "unknown", "ALL"]

    matrix: dict[str, dict[str, Any]] = {}

    for strat in strategies:
        print(f"\nEvaluating {strat.name} across regimes...")
        matrix[strat.name] = {}
        for reg in target_regimes:
            target = None if reg == "ALL" else reg
            res = simulate_strategy_in_regimes(strat, full_df, regimes, target_regime=target)
            matrix[strat.name][reg] = res

    # Print markdown table
    print("\n" + "=" * 80)
    print("  PHASE 5: REGIME × STRATEGY OBSERVATIONAL MATRIX (180 DAYS, 1H)")
    print("=" * 80)

    # Table format: Regime | Strategy | Trades | Win% | PF | Sharpe | NetRet | Bias (B/S)
    print("\n| Regime | Strategy | Trades | Win% | Profit Factor | Sharpe | Net Return | Bias (Buy/Sell) |")
    print("|---|---|---|---|---|---|---|---|")
    for reg in target_regimes:
        for strat in strategies:
            res = matrix[strat.name][reg]
            if res["trades"] == 0:
                print(f"| {reg:<10} | {strat.name:<15} | {0:>6} | {'-':>5} | {'-':>13} | {'-':>6} | {'0.0%':>10} | {'-':>15} |")
            else:
                bias_str = f"{res['buy_pct']:.0f}% / {res['sell_pct']:.0f}%"
                print(
                    f"| {reg:<10} | {strat.name:<15} | {res['trades']:>6} | {res['win_rate']:>4.1f}% | "
                    f"{res['profit_factor']:>13.2f} | {res['sharpe']:>+6.2f} | {res['net_ret']:>+9.1f}% | {bias_str:>15} |"
                )

    # Summary table: Net Return Matrix
    print("\n\n### Net Return Matrix (%)\n")
    headers = ["Regime"] + [s.name for s in strategies]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for reg in target_regimes:
        row = [reg]
        for strat in strategies:
            res = matrix[strat.name][reg]
            if res["trades"] == 0:
                row.append("0.0% (0)")
            else:
                row.append(f"{res['net_ret']:+.1f}% ({res['trades']}t, PF {res['profit_factor']:.2f})")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
