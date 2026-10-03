"""
phase7a_unlock_ablation.py
===========================
Model B Unlock Ablation — Phase 7A

Runs three walk-forward variants against the same diagnostic period:

  A — Phase 5 control:   Model B eligible in 'range' only
  B — Full unlock:       Model B eligible in all trend states
  C — LONG-only unlock:  Model B eligible in all states, SHORT signals suppressed

Everything else is frozen:
  - Model B weights
  - feature columns
  - confidence threshold (0.85)
  - TP (2.0 ATR) / SL (1.0 ATR)
  - max hold (24 bars)
  - fee (0.20%)
  - position sizing (fixed)
  - Sep 22+ untouched

Primary question:
  Does unlocking Model B across regimes recover the positive expectancy
  observed in Phase 7 without creating unacceptable drawdown or cost sensitivity?
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from alpha_engine.regime_detector import RegimeDetector, MarketState
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Variant definitions — subclass MeanReversionStrategy, override is_eligible
# ---------------------------------------------------------------------------

class ModelB_RangeOnly(MeanReversionStrategy):
    """Variant A: Phase 5 control — eligible in range only."""
    variant_label = "A — Range Only (Phase 5 control)"

    def is_eligible(self, state: MarketState) -> bool:
        return state.trend == "range" and state.volatility in ["low_vol", "normal_vol"]


class ModelB_FullUnlock(MeanReversionStrategy):
    """Variant B: Eligible in all trend states, both directions."""
    variant_label = "B — Full Unlock (all regimes)"

    def is_eligible(self, state: MarketState) -> bool:
        return True   # Active in every detected state


class ModelB_LongOnly(MeanReversionStrategy):
    """Variant C: Eligible in all trend states, LONG signals only."""
    variant_label = "C — LONG-Only Unlock (all regimes, no shorts)"

    def is_eligible(self, state: MarketState) -> bool:
        return True

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        raw = super().signal(df)
        return raw if raw == 1 else 0


# ---------------------------------------------------------------------------
# Walk-forward engine (mirrors strategy_walkforward.py but variant-aware)
# ---------------------------------------------------------------------------

def run_variant(
    strategy: MeanReversionStrategy,
    df: pd.DataFrame,
    state_df: pd.DataFrame,
    fee_pct: float = 0.20,
    start_idx: int = 55,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """Run a single Model B variant through chronological walk-forward."""
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(df) if end_idx is None else min(len(df), end_idx)

    atr_series = BaseStrategy._atr(df, period=14).values

    equity = 1.0
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []
    long_pnls, short_pnls = [], []

    i = start_idx
    while i < n - 36:
        trend     = state_df["trend"].iloc[i]
        volatility = state_df["volatility"].iloc[i]
        state     = MarketState(trend=trend, volatility=volatility)

        # Eligibility gate
        if not strategy.is_eligible(state):
            i += 1
            continue

        window_df = df.iloc[: i + 1]
        sig = strategy.signal(window_df)

        if sig == 0:
            i += 1
            continue

        entry    = close[i]
        bar_atr  = atr_series[i]
        tp_price = entry + sig * 2.0 * bar_atr
        sl_price = entry - sig * 1.0 * bar_atr
        max_hold = 24

        trade_pnl_pct = None
        exit_bar  = min(i + max_hold, n - 1)
        exit_reason = "max_hold"

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if sig == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j; exit_reason = "tp"; break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j; exit_reason = "sl"; break
            else:
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j; exit_reason = "tp"; break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j; exit_reason = "sl"; break

        if trade_pnl_pct is None:
            trade_pnl_pct = sig * (close[exit_bar] - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct
        equity *= 1.0 + trade_pnl_pct / 100.0
        equity_curve.append(equity)

        trades.append({
            "bar_entry":  i,
            "bar_exit":   exit_bar,
            "trend":      trend,
            "volatility": volatility,
            "signal":     "BUY" if sig == 1 else "SELL",
            "pnl_pct":    trade_pnl_pct,
            "exit_reason": exit_reason,
        })

        if sig == 1:
            long_pnls.append(trade_pnl_pct)
        else:
            short_pnls.append(trade_pnl_pct)

        i = exit_bar + 1

    return _metrics(trades, equity_curve, long_pnls, short_pnls)


def _metrics(trades, equity_curve, long_pnls, short_pnls) -> dict[str, Any]:
    if not trades:
        return {"trades_count": 0, "net_ret": 0.0, "profit_factor": 0.0,
                "sharpe": 0.0, "max_dd_pct": 0.0, "win_rate": 0.0,
                "long_count": 0, "short_count": 0,
                "long_net": 0.0, "short_net": 0.0,
                "long_pf": 0.0, "short_pf": 0.0,
                "by_trend": {}}

    pnls = np.array([t["pnl_pct"] for t in trades])
    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100.0
    wins = (pnls > 0).sum()
    gp   = pnls[pnls > 0].sum() if wins > 0 else 0.0
    gl   = abs(pnls[pnls <= 0].sum()) if (len(pnls) - wins) > 0 else 0.0

    # Per-direction metrics
    def dir_stats(pl_list):
        if not pl_list:
            return 0, 0.0, 0.0, 0.0
        arr = np.array(pl_list)
        gp_ = arr[arr > 0].sum() if (arr > 0).any() else 0.0
        gl_ = abs(arr[arr <= 0].sum()) if (arr <= 0).any() else 0.0
        pf_ = gp_ / gl_ if gl_ > 0 else (999.0 if gp_ > 0 else 0.0)
        return len(arr), arr.sum(), pf_, (arr > 0).mean() * 100

    l_cnt, l_net, l_pf, l_wr = dir_stats(long_pnls)
    s_cnt, s_net, s_pf, s_wr = dir_stats(short_pnls)

    # Attribution by trend
    by_trend: dict[str, dict] = {}
    for t in trades:
        tr = t["trend"]
        if tr not in by_trend:
            by_trend[tr] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_trend[tr]["trades"] += 1
        by_trend[tr]["pnl"]    += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            by_trend[tr]["wins"] += 1

    return {
        "trades_count": len(trades),
        "win_rate":  wins / len(pnls) * 100.0,
        "profit_factor": gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0),
        "sharpe": (pnls.mean() / pnls.std() * np.sqrt(len(pnls))) if pnls.std() > 1e-8 else 0.0,
        "max_dd_pct": dd.min(),
        "net_ret": (eq[-1] - eq[0]) / eq[0] * 100.0,
        "long_count":  l_cnt, "long_net": l_net, "long_pf": l_pf,  "long_wr": l_wr,
        "short_count": s_cnt, "short_net": s_net, "short_pf": s_pf, "short_wr": s_wr,
        "by_trend": by_trend,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(label: str, r: dict, btc_bh: float) -> None:
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  Trades:        {r['trades_count']}  "
          f"(LONG {r['long_count']}, SHORT {r['short_count']})")
    print(f"  Win Rate:      {r['win_rate']:.1f}%")
    print(f"  Profit Factor: {r['profit_factor']:.2f}")
    print(f"  Sharpe:        {r['sharpe']:+.2f}")
    print(f"  Max DD:        {r['max_dd_pct']:+.1f}%")
    print(f"  Net Return:    {r['net_ret']:+.2f}%  (BTC B&H: {btc_bh:+.1f}%)")
    print(f"\n  Direction breakdown:")
    print(f"    LONG  — {r['long_count']} trades | PnL sum {r['long_net']:+.2f}% | "
          f"PF {r['long_pf']:.2f} | WR {r['long_wr']:.1f}%")
    print(f"    SHORT — {r['short_count']} trades | PnL sum {r['short_net']:+.2f}% | "
          f"PF {r['short_pf']:.2f} | WR {r['short_wr']:.1f}%")
    print(f"\n  Attribution by trend state:")
    for trend, attr in sorted(r["by_trend"].items()):
        wr = attr["wins"] / attr["trades"] * 100 if attr["trades"] > 0 else 0
        print(f"    {trend:<12}: {attr['trades']:>4} trades | "
              f"PnL sum {attr['pnl']:>+7.2f}% | WR {wr:.1f}%")


def print_comparison_table(results: list[tuple[str, dict]]) -> None:
    print("\n\n" + "=" * 90)
    print("  VARIANT COMPARISON SUMMARY")
    print("=" * 90)
    print(f"  {'Variant':<35} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8} {'Net Ret':>9}")
    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*9}")
    for label, r in results:
        print(f"  {label:<35} {r['trades_count']:>7} {r['win_rate']:>5.1f}% "
              f"{r['profit_factor']:>6.2f} {r['sharpe']:>+7.2f} "
              f"{r['max_dd_pct']:>+7.1f}% {r['net_ret']:>+8.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading {data_path}...")
    raw_df = pd.read_csv(data_path)

    config = TrainingDataConfig()
    gen    = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Dataset: {len(full_df)} bars "
          f"({full_df['timestamp'].iloc[0].date()} → {full_df['timestamp'].iloc[-1].date()})")

    detector = RegimeDetector()
    state_df = detector.detect_series(full_df)

    # Diagnostic holdout boundary: 142 days in (Aug 1 - Sep 6, same as Phase 5)
    t0      = full_df["timestamp"].iloc[0]
    t_split = t0 + pd.Timedelta(days=142)
    split_idx = int((full_df["timestamp"] < t_split).sum())

    btc_full_bh  = (full_df["close"].iloc[-1] / full_df["close"].iloc[55] - 1) * 100
    btc_hold_bh  = (full_df["close"].iloc[-1] / full_df["close"].iloc[split_idx] - 1) * 100

    print(f"\nHoldout split: bar {split_idx} ({t_split.date()})")
    print(f"BTC Buy & Hold — Full period:     {btc_full_bh:+.1f}%")
    print(f"BTC Buy & Hold — Holdout only:    {btc_hold_bh:+.1f}%")

    variants = [
        ModelB_RangeOnly(),
        ModelB_FullUnlock(),
        ModelB_LongOnly(),
    ]

    print("\n" + "=" * 90)
    print("  PHASE 7A: MODEL B UNLOCK ABLATION")
    print("  Diagnostic period (days 0–180). Sep 22+ untouched.")
    print("=" * 90)

    all_results_full = []
    all_results_hold = []

    for v in variants:
        label = getattr(v, "variant_label", v.name)

        # Full period
        res_full = run_variant(v, full_df, state_df, start_idx=55)
        # Holdout only
        res_hold = run_variant(v, full_df, state_df, start_idx=split_idx)

        print(f"\n\n{'═'*70}")
        print(f"  VARIANT {label}")
        print(f"{'═'*70}")

        print("\n  [FULL 180-DAY PERIOD]")
        print_result(label, res_full, btc_full_bh)

        print("\n  [DIAGNOSTIC HOLDOUT: Days 142–180]")
        print_result(label, res_hold, btc_hold_bh)

        all_results_full.append((f"{label} [Full]",  res_full))
        all_results_hold.append((f"{label} [Hold]", res_hold))

    print_comparison_table(all_results_full + all_results_hold)

    # Cost sensitivity check on full unlock variant
    print("\n\n" + "=" * 70)
    print("  COST SENSITIVITY: Variant B (Full Unlock) — Full Period")
    print("=" * 70)
    print(f"  {'Fee':>8} {'Net Ret':>10} {'PF':>7}")
    for fee in [0.0, 0.10, 0.20, 0.30, 0.40]:
        v_b = ModelB_FullUnlock()
        r   = run_variant(v_b, full_df, state_df, fee_pct=fee, start_idx=55)
        print(f"  {fee:.2f}%    {r['net_ret']:>+9.2f}%  {r['profit_factor']:>7.2f}")


if __name__ == "__main__":
    main()
