#!/usr/bin/env python3
"""
Phase 3 Validation Script — Higher Timeframe Strategy

For each candidate geometry, this script produces:

1. Compounded equity curve with correct Max Drawdown
2. Per-regime breakdown (Trending / Choppy / Volatile / Recovery)
3. ATR sanity check (ATR as % of price)
4. Cost sensitivity table (0.20% to 0.40%)
5. Simple MA-crossover baseline comparison (non-AI)

Candidates:
  - 1h  ATR 2.0/1.0 @ threshold 0.85
  - 15m ATR 2.0/1.0 @ threshold 0.95
  - 1h  Fixed 1.5%/0.75% @ threshold 0.95
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Core simulator — compounded equity curve, correct MaxDD, full metrics
# ---------------------------------------------------------------------------
def simulate(booster, scaler, feature_cols, df, confidence_threshold=0.85,
             tp_pct=None, sl_pct=None, atr_tp=None, atr_sl=None,
             fee_pct=0.20, max_hold_bars=20, initial_capital=1.0):
    """
    Simulate discrete-trade strategy.
    Returns dict of metrics including a compounded equity_curve list.
    fee_pct is total round-trip cost as percentage (e.g. 0.20 means 0.20%).
    """
    df = df.reset_index(drop=True)
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values

    # ATR-14 EWM
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    X = df[feature_cols].values
    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        import xgboost as xgb
        probs = booster.predict(xgb.DMatrix(scaled))

    equity = initial_capital
    equity_curve = [equity]
    pnls = []
    hold_durations = []

    i = 0
    while i < len(df) - max_hold_bars:
        p_sell, p_buy = probs[i][0], probs[i][2]
        if p_buy > p_sell and p_buy >= confidence_threshold:
            direction = 1
        elif p_sell > p_buy and p_sell >= confidence_threshold:
            direction = -1
        else:
            i += 1
            continue

        entry = close[i]
        if tp_pct is not None:
            tp_price = entry * (1 + direction * tp_pct / 100.0)
            sl_price = entry * (1 - direction * sl_pct / 100.0)
        else:
            bar_atr = atr[i]
            tp_price = entry + direction * atr_tp * bar_atr
            sl_price = entry - direction * atr_sl * bar_atr

        trade_pnl_pct = None
        exit_bar = i + max_hold_bars

        for j in range(i + 1, min(i + max_hold_bars + 1, len(df))):
            if direction == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j; break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j; break
            else:
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j; break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j; break

        if trade_pnl_pct is None:
            exit_p = close[min(exit_bar, len(close) - 1)]
            trade_pnl_pct = direction * (exit_p - entry) / entry * 100.0

        # Apply fee as a round-trip percentage cost
        trade_pnl_pct -= fee_pct

        # Compound equity
        equity *= (1.0 + trade_pnl_pct / 100.0)
        equity_curve.append(equity)
        pnls.append(trade_pnl_pct)
        hold_durations.append(exit_bar - i)
        i = exit_bar + 1

    if not pnls:
        return None

    pnls = np.array(pnls)
    eq = np.array(equity_curve)

    winning = (pnls > 0).sum()
    total = len(pnls)
    win_rate = winning / total

    gross_profit = pnls[pnls > 0].sum() if winning > 0 else 0
    gross_loss   = abs(pnls[pnls <= 0].sum()) if (total - winning) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Compounded max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100.0   # percentage
    max_dd = dd.min()

    # Net cumulative return (compounded)
    net_cum_ret = (eq[-1] - eq[0]) / eq[0] * 100.0

    # Time-series Sharpe (daily, annualised) — approx via trade-level
    mean_pnl = pnls.mean()
    std_pnl  = pnls.std() if total > 1 else 1e-8
    sharpe_trade = (mean_pnl / std_pnl) * np.sqrt(total) if std_pnl > 1e-8 else 0.0

    avg_hold = np.mean(hold_durations)
    med_hold = np.median(hold_durations)
    avg_pnl_per_trade = pnls.mean()
    med_pnl_per_trade = np.median(pnls)

    return {
        "trades": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe": sharpe_trade,
        "max_dd_pct": max_dd,
        "net_cum_ret": net_cum_ret,
        "avg_hold_bars": avg_hold,
        "median_hold_bars": med_hold,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "median_pnl_per_trade": med_pnl_per_trade,
        "equity_curve": equity_curve,
    }


def agg_results(results: list):
    """Aggregate metrics from multiple windows (compounding chain)."""
    all_trades = sum(r["trades"] for r in results)
    all_pnls = []
    # Reconstruct equity curve chain
    equity = 1.0
    eq_chain = [equity]
    for r in results:
        eq_arr = np.array(r["equity_curve"])
        # Scale to current equity
        scaled = equity * (eq_arr / eq_arr[0])
        eq_chain.extend(scaled[1:].tolist())
        equity = eq_chain[-1]

    eq_arr = np.array(eq_chain)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100.0
    max_dd = dd.min()
    net_cum_ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0] * 100.0

    return {
        "trades": all_trades,
        "win_rate": np.mean([r["win_rate"] for r in results]),
        "profit_factor": np.mean([r["profit_factor"] for r in results]),
        "sharpe": np.mean([r["sharpe"] for r in results]),
        "max_dd_pct": max_dd,
        "net_cum_ret": net_cum_ret,
        "avg_hold_bars": np.mean([r["avg_hold_bars"] for r in results]),
        "avg_pnl_per_trade": np.mean([r["avg_pnl_per_trade"] for r in results]),
        "median_pnl_per_trade": np.mean([r["median_pnl_per_trade"] for r in results]),
    }


# ---------------------------------------------------------------------------
# ATR Sanity Check
# ---------------------------------------------------------------------------
def atr_analysis(df, tf_label):
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr_pct = atr / close * 100.0

    print(f"\nATR ANALYSIS — {tf_label}")
    print(f"  Median ATR%:       {np.median(atr_pct):>8.3f}%")
    print(f"  Mean ATR%:         {np.mean(atr_pct):>8.3f}%")
    print(f"  P25 ATR%:          {np.percentile(atr_pct, 25):>8.3f}%")
    print(f"  P75 ATR%:          {np.percentile(atr_pct, 75):>8.3f}%")
    print(f"  => 2.0×ATR TP est: {np.median(atr_pct)*2:>8.3f}%  (vs 0.20% cost)")
    print(f"  => 4.0×ATR TP est: {np.median(atr_pct)*4:>8.3f}%  (vs 0.20% cost)")
    return np.median(atr_pct)


# ---------------------------------------------------------------------------
# Simple Baseline (MA-crossover)
# ---------------------------------------------------------------------------
def baseline_ma_crossover(df, fast=20, slow=50, fee_pct=0.20, max_hold_bars=20):
    """Simple non-AI strategy: enter long when fast MA crosses above slow MA."""
    df = df.reset_index(drop=True)
    close = df["close"].values
    fast_ma = pd.Series(close).rolling(fast, min_periods=1).mean().values
    slow_ma = pd.Series(close).rolling(slow, min_periods=1).mean().values

    equity = 1.0
    eq_chain = [equity]
    pnls = []
    high  = df["high"].values
    low   = df["low"].values

    # ATR for barriers (same as model)
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    i = slow
    while i < len(df) - max_hold_bars:
        # Crossover signal
        if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            direction = 1
        elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            direction = -1
        else:
            i += 1
            continue

        entry = close[i]
        tp_price = entry + direction * 2.0 * atr[i]  # same ATR 2/1 geometry
        sl_price = entry - direction * 1.0 * atr[i]

        trade_pnl_pct = None
        exit_bar = i + max_hold_bars
        for j in range(i + 1, min(i + max_hold_bars + 1, len(df))):
            if direction == 1:
                if high[j] >= tp_price:
                    trade_pnl_pct = (tp_price - entry) / entry * 100.0
                    exit_bar = j; break
                if low[j] <= sl_price:
                    trade_pnl_pct = (sl_price - entry) / entry * 100.0
                    exit_bar = j; break
            else:
                if low[j] <= tp_price:
                    trade_pnl_pct = (entry - tp_price) / entry * 100.0
                    exit_bar = j; break
                if high[j] >= sl_price:
                    trade_pnl_pct = (entry - sl_price) / entry * 100.0
                    exit_bar = j; break

        if trade_pnl_pct is None:
            exit_p = close[min(exit_bar, len(close) - 1)]
            trade_pnl_pct = direction * (exit_p - entry) / entry * 100.0

        trade_pnl_pct -= fee_pct
        equity *= (1.0 + trade_pnl_pct / 100.0)
        eq_chain.append(equity)
        pnls.append(trade_pnl_pct)
        i = exit_bar + 1

    if not pnls:
        return None
    pnls = np.array(pnls)
    eq = np.array(eq_chain)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100.0

    return {
        "trades": len(pnls),
        "win_rate": (pnls > 0).mean(),
        "profit_factor": pnls[pnls > 0].sum() / abs(pnls[pnls <= 0].sum()) if (pnls <= 0).any() else 999.0,
        "sharpe": (pnls.mean() / pnls.std() * np.sqrt(len(pnls))) if pnls.std() > 1e-8 else 0,
        "max_dd_pct": dd.min(),
        "net_cum_ret": (eq[-1] - eq[0]) / eq[0] * 100.0,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

def print_metrics(m, label="", days=180):
    tpd = m["trades"] / days if days else 0
    print(f"  {'Label:':<20} {label}")
    print(f"  {'Trades:':<20} {m['trades']}")
    print(f"  {'Trades/day:':<20} {tpd:.1f}")
    print(f"  {'Win rate:':<20} {m['win_rate']*100:.1f}%")
    print(f"  {'Profit Factor:':<20} {m['profit_factor']:.2f}")
    print(f"  {'Sharpe (trade):':<20} {m['sharpe']:+.2f}")
    print(f"  {'Max DD (compounded):':<20} {m['max_dd_pct']:+.1f}%")
    print(f"  {'Net Cum Return:':<20} {m['net_cum_ret']:+.1f}%")
    if "avg_pnl_per_trade" in m:
        print(f"  {'Avg PnL/trade:':<20} {m['avg_pnl_per_trade']:+.4f}%")
        print(f"  {'Med PnL/trade:':<20} {m['median_pnl_per_trade']:+.4f}%")
    if "avg_hold_bars" in m:
        print(f"  {'Avg hold (bars):':<20} {m['avg_hold_bars']:.1f}")


def print_regime_table(rows):
    header = f"  {'Geometry':<22} {'Regime':<12} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'NetRet%':>8}"
    print(header)
    print("  " + "-"*70)
    for regime, geo_label, m in rows:
        if m:
            print(f"  {geo_label:<22} {regime:<12} {m['trades']:>7} "
                  f"{m['win_rate']*100:>5.1f}% {m['profit_factor']:>6.2f} "
                  f"{m['sharpe']:>+7.2f} {m['max_dd_pct']:>6.1f}% {m['net_cum_ret']:>+8.1f}%")
        else:
            print(f"  {geo_label:<22} {regime:<12} {'--':>7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def validate_candidate(candidate, booster, scaler, fc, windows_df, tf_days=180,
                        max_hold_bars=20, tf_label=""):
    name = candidate["name"]
    kwargs = candidate["kwargs"]
    conf  = candidate["conf"]
    
    section(f"VALIDATION — {tf_label} | {name} @ conf={conf}")
    
    # --- Full aggregate ---
    all_results = []
    for period_name, regime, df_w in windows_df:
        r = simulate(booster, scaler, fc, df_w, confidence_threshold=conf,
                     fee_pct=0.20, max_hold_bars=max_hold_bars, **kwargs)
        if r:
            r["period"] = period_name
            r["regime"] = regime
            all_results.append(r)
    
    if not all_results:
        print("  No trades generated.")
        return

    agg = agg_results(all_results)
    print_metrics(agg, label=f"{name} (ALL REGIMES)", days=tf_days)
    
    # --- Per-regime breakdown ---
    print(f"\n  REGIME BREAKDOWN:")
    regime_rows = []
    for period_name, regime, df_w in windows_df:
        r = simulate(booster, scaler, fc, df_w, confidence_threshold=conf,
                     fee_pct=0.20, max_hold_bars=max_hold_bars, **kwargs)
        regime_rows.append((regime, name[:22], r))
    print_regime_table(regime_rows)
    
    # --- Cost Sensitivity ---
    print(f"\n  COST SENSITIVITY:")
    print(f"  {'Cost':>8} {'Trades':>7} {'PF':>6} {'Sharpe':>8} {'MaxDD%':>8} {'NetRet%':>10}")
    print("  " + "-"*55)
    for cost in [0.20, 0.25, 0.30, 0.35, 0.40]:
        c_results = []
        for _, _, df_w in windows_df:
            r = simulate(booster, scaler, fc, df_w, confidence_threshold=conf,
                         fee_pct=cost, max_hold_bars=max_hold_bars, **kwargs)
            if r:
                c_results.append(r)
        if c_results:
            m = agg_results(c_results)
            mark = " *" if m["net_cum_ret"] > 0 else ""
            print(f"  {cost:>7.2f}% {m['trades']:>7} {m['profit_factor']:>6.2f} "
                  f"{m['sharpe']:>+8.2f} {m['max_dd_pct']:>7.1f}% {m['net_cum_ret']:>+10.1f}%{mark}")


def main():
    tf_configs = [
        {
            "label": "1h",
            "data":  "data/live/BTC_USDT_1h_live.csv",
            "model": "local-model-b-1h",
            "max_hold": 24,
            "days": 180,
            "candidates": [
                {"name": "ATR 2.0/1.0",       "conf": 0.85, "kwargs": {"atr_tp": 2.0, "atr_sl": 1.0}},
                {"name": "Fixed 1.5%/0.75%",   "conf": 0.95, "kwargs": {"tp_pct": 1.5, "sl_pct": 0.75}},
                {"name": "Fixed 1.0%/0.5%",    "conf": 0.95, "kwargs": {"tp_pct": 1.0, "sl_pct": 0.5}},
            ],
        },
        {
            "label": "15m",
            "data":  "data/live/BTC_USDT_15m_live.csv",
            "model": "local-model-b-15m",
            "max_hold": 40,
            "days": 180,
            "candidates": [
                {"name": "ATR 2.0/1.0",       "conf": 0.95, "kwargs": {"atr_tp": 2.0, "atr_sl": 1.0}},
                {"name": "Fixed 1.0%/0.5%",    "conf": 0.95, "kwargs": {"tp_pct": 1.0, "sl_pct": 0.5}},
            ],
        },
    ]

    for tf in tf_configs:
        tf_label = tf["label"]
        print(f"\n\n{'#'*72}")
        print(f"#  TIMEFRAME: {tf_label}")
        print(f"{'#'*72}")

        booster, scaler, fc = load_model(Path("models") / tf["model"])
        raw_df = pd.read_csv(tf["data"])

        config = TrainingDataConfig()
        gen    = TrainingDataGenerator(config)
        full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
        full_df = full_df.assign(
            timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
        ).sort_values("timestamp").reset_index(drop=True)

        t0 = full_df["timestamp"].iloc[0]
        window_bounds = [
            ("Q1-Trending",  "trending",  t0 + pd.Timedelta(days=0),   t0 + pd.Timedelta(days=45)),
            ("Q2-Choppy",    "choppy",    t0 + pd.Timedelta(days=45),  t0 + pd.Timedelta(days=90)),
            ("Q3-Volatile",  "volatile",  t0 + pd.Timedelta(days=90),  t0 + pd.Timedelta(days=135)),
            ("Q4-Recovery",  "recovery",  t0 + pd.Timedelta(days=135), t0 + pd.Timedelta(days=180)),
        ]

        windows_df = []
        for period_name, regime, start_ts, end_ts in window_bounds:
            w = full_df[(full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)]
            if len(w) >= 50:
                windows_df.append((period_name, regime, w.reset_index(drop=True)))

        # ATR sanity check on full df
        atr_analysis(full_df, tf_label)

        # Validate each candidate
        for cand in tf["candidates"]:
            validate_candidate(cand, booster, scaler, fc, windows_df,
                               tf_days=tf["days"], max_hold_bars=tf["max_hold"],
                               tf_label=tf_label)

        # Simple MA-crossover baseline (using 1h or 15m raw data)
        section(f"SIMPLE BASELINE (MA 20/50 crossover, ATR 2/1 barriers) — {tf_label}")
        b_result = baseline_ma_crossover(full_df, fast=20, slow=50, fee_pct=0.20,
                                         max_hold_bars=tf["max_hold"])
        if b_result:
            print_metrics(b_result, label="MA Crossover baseline", days=tf["days"])
        else:
            print("  No trades generated by baseline.")


if __name__ == "__main__":
    main()
