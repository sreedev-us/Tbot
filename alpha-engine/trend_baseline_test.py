#!/usr/bin/env python3
"""
Trend Baseline Test on Holdout Period.

Tests a simple MA-crossover trend-following strategy on the SAME holdout
period (Aug 1 - Sep 6, 2026) where Model B lost -28.4%.

If the simple trend strategy profits while Model B fails, it confirms:
  (a) Trending edge existed in the market during the holdout
  (b) Model B's failure is a learned strategy mismatch (mean-reversion bias)
      not an absence of signal

Also tests a simple "buy-and-hold BTC" baseline to quantify the regime.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model
from validate_htf import simulate

logging.basicConfig(level=logging.WARNING)


def print_metrics_block(label, m, days):
    print(f"  {label}")
    print(f"    Trades:        {m['trades']} ({m['trades']/days:.1f}/day)")
    print(f"    Win rate:      {m['win_rate']*100:.1f}%")
    print(f"    Profit Factor: {m['profit_factor']:.2f}")
    print(f"    Sharpe:        {m['sharpe']:+.2f}")
    print(f"    Max DD:        {m['max_dd_pct']:+.1f}%")
    print(f"    Net Return:    {m['net_cum_ret']:+.1f}%")
    # Prediction bias ratio (for trend strategy, all signals are BUY or SELL)
    if 'buy_pct' in m:
        print(f"    BUY signals:   {m['buy_pct']:.1f}%  | Actual UP: {m['actual_up_pct']:.1f}%")
        print(f"    SELL signals:  {m['sell_pct']:.1f}%  | Actual DN: {m['actual_dn_pct']:.1f}%")
    print()


def baseline_ma_crossover(df, fast=20, slow=50, fee_pct=0.20, max_hold=24,
                           atr_tp=2.0, atr_sl=1.0):
    """Simple MA crossover: long when fast MA crosses above slow, short below."""
    df = df.reset_index(drop=True)
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values

    fast_ma = pd.Series(close).rolling(fast, min_periods=1).mean().values
    slow_ma = pd.Series(close).rolling(slow, min_periods=1).mean().values

    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    equity = 1.0
    equity_curve = [equity]
    pnls, buy_signals, sell_signals = [], 0, 0

    i = slow
    while i < len(df) - max_hold:
        # Crossover detection
        if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            direction = 1; buy_signals += 1
        elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            direction = -1; sell_signals += 1
        else:
            i += 1; continue

        entry = close[i]
        tp_price = entry + direction * atr_tp * atr[i]
        sl_price = entry - direction * atr_sl * atr[i]

        trade_pnl_pct = None
        exit_bar = i + max_hold
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            if direction == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0; exit_bar = j; break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0; exit_bar = j; break
            else:
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0; exit_bar = j; break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0; exit_bar = j; break

        if trade_pnl_pct is None:
            exit_p = close[min(exit_bar, len(close)-1)]
            trade_pnl_pct = direction * (exit_p - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct
        equity *= (1.0 + trade_pnl_pct / 100.0)
        equity_curve.append(equity)
        pnls.append(trade_pnl_pct)
        i = exit_bar + 1

    if not pnls:
        return None

    pnls = np.array(pnls)
    eq = np.array(equity_curve)
    winning = (pnls > 0).sum()
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100.0

    gross_profit = pnls[pnls > 0].sum() if winning > 0 else 0
    gross_loss   = abs(pnls[pnls <= 0].sum()) if (len(pnls) - winning) > 0 else 0

    total_bars = len(df) - max_hold
    up_bars = (close[1:] > close[:-1]).sum() / (len(close)-1) * 100

    return {
        "trades": len(pnls),
        "win_rate": winning / len(pnls),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 999.0,
        "sharpe": (pnls.mean() / pnls.std() * np.sqrt(len(pnls))) if pnls.std() > 1e-8 else 0,
        "max_dd_pct": dd.min(),
        "net_cum_ret": (eq[-1] - eq[0]) / eq[0] * 100.0,
        "buy_pct": buy_signals / (buy_signals + sell_signals) * 100 if (buy_signals+sell_signals) > 0 else 0,
        "sell_pct": sell_signals / (buy_signals + sell_signals) * 100 if (buy_signals+sell_signals) > 0 else 0,
        "actual_up_pct": up_bars,
        "actual_dn_pct": 100 - up_bars,
    }


def model_b_directional_bias(booster, scaler, fc, df, confidence=0.85):
    """Report Model B's prediction bias ratio on this window."""
    X = df[fc].values
    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        import xgboost as xgb
        probs = booster.predict(xgb.DMatrix(scaled))

    p_down, p_up = probs[:, 0], probs[:, 2]
    close = df["close"].values
    total = len(df)
    buy_mask = (p_up > p_down) & (p_up >= confidence)
    sell_mask = (p_down > p_up) & (p_down >= confidence)
    up_bars = (close[1:] > close[:-1]).sum() / (len(close)-1) * 100

    return {
        "buy_pct": buy_mask.sum() / total * 100,
        "sell_pct": sell_mask.sum() / total * 100,
        "hold_pct": (~(buy_mask | sell_mask)).sum() / total * 100,
        "actual_up_pct": up_bars,
        "actual_dn_pct": 100 - up_bars,
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

    t0 = full_df["timestamp"].iloc[0]
    t_split = t0 + pd.Timedelta(days=142)
    holdout_df = full_df[full_df["timestamp"] >= t_split].reset_index(drop=True)

    h_start = holdout_df["timestamp"].iloc[0]
    h_end   = holdout_df["timestamp"].iloc[-1]
    actual_days = (h_end - h_start).total_seconds() / 86400
    close = holdout_df["close"].values
    btc_ret = (close[-1] - close[0]) / close[0] * 100

    print("=" * 65)
    print("  TREND BASELINE TEST — HOLDOUT PERIOD")
    print(f"  {h_start.date()} to {h_end.date()}  ({actual_days:.0f} days)")
    print(f"  BTC Return: {btc_ret:+.1f}% ({close[0]:.0f} -> {close[-1]:.0f})")
    print("=" * 65)

    # ── Buy-and-hold benchmark ─────────────────────────────────────────────────
    print("\n[0] BUY-AND-HOLD (no strategy, pure market return)")
    print(f"    BTC total return: {btc_ret:+.1f}%")
    print(f"    This is the baseline the market offered.\n")

    # ── Model B performance (already established in holdout) ──────────────────
    booster_b, scaler_b, fc_b = load_model(Path("models/local-model-b-1h-holdout"))
    bias_b = model_b_directional_bias(booster_b, scaler_b, fc_b, holdout_df)

    print("[1] MODEL B (frozen, ATR 2/1 @ 0.85) — established result")
    print(f"    Trades:        103  (2.9/day)")
    print(f"    Win rate:      24.3%")
    print(f"    Profit Factor: 0.39")
    print(f"    Sharpe:        -4.23")
    print(f"    Net Return:    -28.4%")
    print(f"    BUY signals:   {bias_b['buy_pct']:.1f}%  | Actual UP: {bias_b['actual_up_pct']:.1f}%")
    print(f"    SELL signals:  {bias_b['sell_pct']:.1f}%  | Actual DN: {bias_b['actual_dn_pct']:.1f}%")
    print()

    # ── MA crossover variants ─────────────────────────────────────────────────
    print("[2] SIMPLE TREND BASELINE — MA crossover variants (ATR 2/1 geometry, 0.20% cost)")
    for fast, slow in [(10, 30), (20, 50), (5, 20)]:
        r = baseline_ma_crossover(holdout_df, fast=fast, slow=slow,
                                  fee_pct=0.20, max_hold=24)
        if r:
            print_metrics_block(f"  MA({fast}/{slow}):", r, actual_days)

    # ── Trend-only long bias: always buy signals above MA ─────────────────────
    print("[3] LONG-ONLY TREND: hold long when price > SMA-50 (continuous, no barrier)")
    sma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
    above_ma = (close > sma50)
    # Simple daily P&L sum
    close_ret = np.diff(close) / close[:-1] * 100.0
    long_ret = close_ret * above_ma[:-1].astype(float)
    cum = np.cumsum(long_ret)
    print(f"    Net Return (no-cost):  {cum[-1]:+.1f}%")
    print(f"    % bars above SMA50:   {above_ma.mean()*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
