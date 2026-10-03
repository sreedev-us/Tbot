"""
phase7c_activation_gate.py
===========================
Phase 7C: Prospective Activation-Gate Research

Tests four predefined gate hypotheses against the diagnostic period.
Sep 22+ remains completely untouched.

Rules:
  - Gates are predefined BEFORE examining their performance.
  - No threshold is derived from the holdout data.
  - Model B is frozen. The gate sits outside it.
  - Primary question: Does the gate RESTORE SIGNAL CALIBRATION,
    specifically by removing anti-predictive SELL signals?
  - Secondary question: Does calibration restoration translate to
    improved walk-forward metrics on the holdout period?

Gates:
  G0 — No gate (control): all Model B signals pass through
  G1 — Trend-direction gate: suppress SELL when EMA20 > EMA50
  G2 — Sustained-trend gate: suppress SELL when EMA20>EMA50 AND both MAs rising;
         suppress BUY when EMA20<EMA50 AND both MAs falling
  G3 — Symmetric directional gate: G2 applied symmetrically,
         also suppress BUY in sustained downtrend

Architecture:
  Model B (frozen)
       |
       v
  BUY / SELL signal
       |
       v
  Activation Gate
       |
  allowed / suppressed

Period:
  EARLIER: bars 55–3396  (Mar 14 – Jul 31, 2026) — calibration baseline
  HOLDOUT: bars 3397–end (Aug 1 – Sep 6, 2026)  — where failure occurred
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from alpha_engine.regime_detector import RegimeDetector, MarketState
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy

logging.basicConfig(level=logging.WARNING)

FEE_PCT  = 0.20
MAX_HOLD = 24
HORIZONS = [1, 4, 12, 24]

# ---------------------------------------------------------------------------
# Gate definitions — all pure functions of the window DataFrame at signal time
# ---------------------------------------------------------------------------

GateFn = Callable[[int, pd.DataFrame], bool]   # (signal, window_df) -> allow?


def gate_g0(signal: int, df: pd.DataFrame) -> bool:
    """G0: No gate — all signals pass."""
    return True


def gate_g1(signal: int, df: pd.DataFrame) -> bool:
    """
    G1: Trend-direction gate.
    Suppress SELL when EMA20 > EMA50 (price structure is bullish).
    Allow all BUY signals.
    """
    close = df["close"]
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    if signal == -1 and ema20 > ema50:
        return False   # suppress
    return True


def gate_g2(signal: int, df: pd.DataFrame) -> bool:
    """
    G2: Sustained-trend gate.
    Suppress SELL when EMA20 > EMA50 AND EMA20 rising AND EMA50 rising.
    Suppress BUY  when EMA20 < EMA50 AND EMA20 falling AND EMA50 falling.
    Tests whether *trend persistence* (not just direction) invalidates mean reversion.
    """
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    e20_now, e20_prev = float(ema20.iloc[-1]), float(ema20.iloc[-2])
    e50_now, e50_prev = float(ema50.iloc[-1]), float(ema50.iloc[-2])

    sustained_up   = (e20_now > e50_now) and (e20_now > e20_prev) and (e50_now > e50_prev)
    sustained_down = (e20_now < e50_now) and (e20_now < e20_prev) and (e50_now < e50_prev)

    if signal == -1 and sustained_up:
        return False
    if signal ==  1 and sustained_down:
        return False
    return True


def gate_g3(signal: int, df: pd.DataFrame) -> bool:
    """
    G3: Symmetric directional gate — identical logic to G2.
    Included as an explicit variant to confirm G2 is already symmetric
    (and to serve as a documentation boundary between the two).
    G3 additionally checks 3-bar persistence to require more evidence.
    """
    close = df["close"]
    if len(close) < 4:
        return True

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    e20 = [float(ema20.iloc[j]) for j in [-3, -2, -1]]
    e50 = [float(ema50.iloc[j]) for j in [-3, -2, -1]]

    # EMA20 has been above EMA50 for 3 consecutive bars, both rising for 3 bars
    sustained_up = (
        all(e20[k] > e50[k] for k in range(3))
        and e20[0] < e20[1] < e20[2]
        and e50[0] < e50[1] < e50[2]
    )
    sustained_down = (
        all(e20[k] < e50[k] for k in range(3))
        and e20[0] > e20[1] > e20[2]
        and e50[0] > e50[1] > e50[2]
    )

    if signal == -1 and sustained_up:
        return False
    if signal ==  1 and sustained_down:
        return False
    return True


GATES: list[tuple[str, str, GateFn]] = [
    ("G0", "No gate (control)",        gate_g0),
    ("G1", "Trend-direction",          gate_g1),
    ("G2", "Sustained-trend (1-bar)",  gate_g2),
    ("G3", "Sustained-trend (3-bar)",  gate_g3),
]


# ---------------------------------------------------------------------------
# Signal extractor with gate application
# ---------------------------------------------------------------------------

def extract_gated_signals(
    full_df: pd.DataFrame,
    strategy: MeanReversionStrategy,
    gate_fn: GateFn,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    """Extract all Model B signals, apply gate, record calibration metrics."""
    close = full_df["close"].values
    high  = full_df["high"].values
    low   = full_df["low"].values
    n     = len(full_df)

    records = []

    for i in range(start_idx, min(end_idx, n - max(HORIZONS) - 1)):
        window_df = full_df.iloc[: i + 1]

        # Raw model inference
        if all(col in window_df.columns for col in strategy.feature_columns):
            feat = window_df[strategy.feature_columns].iloc[[-1]]
        else:
            fdf = strategy.feature_engineer.engineer_features(window_df, ablation_level="B")
            if fdf.empty:
                continue
            feat = fdf[strategy.feature_columns].iloc[[-1]]

        scaled = strategy.scaler.transform(feat)
        if hasattr(strategy.booster, "predict_proba"):
            probs = strategy.booster.predict_proba(scaled)[0]
        else:
            probs = strategy.booster.predict(xgb.DMatrix(scaled))[0]

        p_sell, p_buy = float(probs[0]), float(probs[2])
        sig, conf = 0, 0.0
        if p_buy > p_sell and p_buy >= strategy.confidence_threshold:
            sig, conf = 1, p_buy
        elif p_sell > p_buy and p_sell >= strategy.confidence_threshold:
            sig, conf = -1, p_sell

        if sig == 0:
            continue

        allowed = gate_fn(sig, window_df)

        entry = close[i]
        atr14 = float(
            (pd.Series(high[:i+1]) - pd.Series(low[:i+1])).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        )

        record: dict[str, Any] = {
            "bar_idx":  i,
            "signal":   sig,
            "direction": "BUY" if sig == 1 else "SELL",
            "confidence": conf,
            "allowed":  allowed,
        }

        # Forward returns (compute regardless — for suppressed signal analysis)
        for h in HORIZONS:
            fut = close[min(i + h, n - 1)]
            record[f"fwd_{h}h"] = sig * (fut - entry) / entry * 100.0

        # TP/SL resolution
        tp_price = entry + sig * 2.0 * atr14
        sl_price = entry - sig * 1.0 * atr14
        tp_hit = sl_hit = False
        bars_to_res = None

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if sig == 1:
                if high[j] >= tp_price:
                    tp_hit = True; bars_to_res = j - i; break
                if low[j]  <= sl_price:
                    sl_hit = True; bars_to_res = j - i; break
            else:
                if low[j]  <= tp_price:
                    tp_hit = True; bars_to_res = j - i; break
                if high[j] >= sl_price:
                    sl_hit = True; bars_to_res = j - i; break

        record["tp_hit"] = tp_hit
        record["sl_hit"] = sl_hit
        record["bars_to_resolution"] = bars_to_res

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Walk-forward simulation (gated)
# ---------------------------------------------------------------------------

def run_gated_walkforward(
    full_df: pd.DataFrame,
    strategy: MeanReversionStrategy,
    gate_fn: GateFn,
    start_idx: int,
    end_idx: int,
) -> dict[str, Any]:
    close = full_df["close"].values
    high  = full_df["high"].values
    low   = full_df["low"].values
    n     = len(full_df)
    atr_s = BaseStrategy._atr(full_df, period=14).values

    equity = 1.0
    equity_curve = [equity]
    trades: list[dict] = []
    l_pnls: list[float] = []
    s_pnls: list[float] = []

    i = start_idx
    while i < min(end_idx, n - MAX_HOLD - 1):
        window_df = full_df.iloc[: i + 1]
        sig = strategy.signal(window_df)
        if sig == 0 or not gate_fn(sig, window_df):
            i += 1
            continue

        entry   = close[i]
        bar_atr = atr_s[i]
        tp_p    = entry + sig * 2.0 * bar_atr
        sl_p    = entry - sig * 1.0 * bar_atr
        exit_b  = min(i + MAX_HOLD, n - 1)
        pnl     = None
        reason  = "max_hold"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if sig == 1:
                if high[j] >= tp_p: pnl = (tp_p - entry) / entry * 100.0; exit_b = j; reason = "tp"; break
                if low[j]  <= sl_p: pnl = (sl_p - entry) / entry * 100.0; exit_b = j; reason = "sl"; break
            else:
                if low[j]  <= tp_p: pnl = (entry - tp_p) / entry * 100.0; exit_b = j; reason = "tp"; break
                if high[j] >= sl_p: pnl = (entry - sl_p) / entry * 100.0; exit_b = j; reason = "sl"; break

        if pnl is None:
            pnl = sig * (close[exit_b] - entry) / entry * 100.0

        pnl -= FEE_PCT
        equity *= 1.0 + pnl / 100.0
        equity_curve.append(equity)
        trades.append({"signal": sig, "pnl": pnl, "exit_reason": reason})
        (l_pnls if sig == 1 else s_pnls).append(pnl)
        i = exit_b + 1

    return _wf_metrics(trades, equity_curve, l_pnls, s_pnls)


def _wf_metrics(trades, equity_curve, l_pnls, s_pnls) -> dict:
    if not trades:
        return {"n": 0, "net_ret": 0.0, "pf": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "win_rate": 0.0, "long_n": 0, "short_n": 0,
                "long_pf": 0.0, "short_pf": 0.0}
    pnls = np.array([t["pnl"] for t in trades])
    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    wins = (pnls > 0).sum()
    gp   = pnls[pnls > 0].sum() if wins > 0 else 0.0
    gl   = abs(pnls[pnls <= 0].sum()) if (len(pnls) - wins) > 0 else 0.0

    def pf(arr):
        a = np.array(arr)
        g = a[a > 0].sum() if (a > 0).any() else 0.0
        l = abs(a[a <= 0].sum()) if (a <= 0).any() else 0.0
        return g / l if l > 0 else (999.0 if g > 0 else 0.0)

    return {
        "n": len(trades), "win_rate": wins / len(pnls) * 100.0,
        "pf":    gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0),
        "sharpe": (pnls.mean() / pnls.std() * np.sqrt(len(pnls))) if pnls.std() > 1e-8 else 0.0,
        "max_dd": ((eq - peak) / peak * 100.0).min(),
        "net_ret": (eq[-1] - eq[0]) / eq[0] * 100.0,
        "long_n":  len(l_pnls), "short_n": len(s_pnls),
        "long_pf": pf(l_pnls),  "short_pf": pf(s_pnls),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def calibration_table(df: pd.DataFrame, label: str) -> None:
    allowed = df[df["allowed"]]
    suppressed = df[~df["allowed"]]
    print(f"\n  [{label}]  Total Model B signals: {len(df)}")
    print(f"    Allowed:     {len(allowed):>5} ({len(allowed)/len(df)*100:.1f}%)")
    print(f"    Suppressed:  {len(suppressed):>5} ({len(suppressed)/len(df)*100:.1f}%)")
    if suppressed.empty:
        supp_buy_pct = 0.0
        supp_sell_pct = 0.0
    else:
        supp_buy_pct  = (suppressed["signal"] ==  1).mean() * 100
        supp_sell_pct = (suppressed["signal"] == -1).mean() * 100
    print(f"    Suppressed breakdown:  BUY {supp_buy_pct:.1f}%  SELL {supp_sell_pct:.1f}%")

    if allowed.empty:
        print(f"    No allowed signals.")
        return

    buy_frac = (allowed["signal"] == 1).mean() * 100
    sell_frac = 100.0 - buy_frac
    print(f"    Allowed BUY/SELL:  {buy_frac:.1f}% / {sell_frac:.1f}%")

    # Forward return table on ALLOWED signals
    print(f"\n    {'Direction':<6} {'n':>4}  " + "  ".join(f"{h}h" for h in HORIZONS))
    print(f"    {'------':<6} {'--':>4}  " + "  ".join("----" for _ in HORIZONS))
    for direction, grp in allowed.groupby("direction"):
        means = [f"{grp[f'fwd_{h}h'].mean():+.3f}%" for h in HORIZONS]
        print(f"    {direction:<6} {len(grp):>4}  {'  '.join(means)}")
    total_means = [f"{allowed[f'fwd_{h}h'].mean():+.3f}%" for h in HORIZONS]
    print(f"    {'ALL':<6} {len(allowed):>4}  {'  '.join(total_means)}")

    # TP/SL on allowed
    tp_pct = allowed["tp_hit"].mean() * 100
    sl_pct = allowed["sl_hit"].mean() * 100
    print(f"    TP hit: {tp_pct:.1f}%   SL hit: {sl_pct:.1f}%")

    # Confidence calibration monotone check
    try:
        cal = allowed.groupby(
            pd.qcut(allowed["confidence"], q=5,
                    labels=["Low", "Med-Low", "Med", "Med-High", "High"],
                    duplicates="drop"),
            observed=True
        )["fwd_12h"].mean().values.tolist()
        mono = all(cal[k] <= cal[k+1] for k in range(len(cal)-1))
        print(f"    12h confidence monotone: {mono}  ({' → '.join(f'{v:+.3f}%' for v in cal)})")
    except Exception:
        print(f"    12h confidence monotone: (insufficient data for quintile split)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading {data_path}...")
    raw_df = pd.read_csv(data_path)

    config  = TrainingDataConfig()
    gen     = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    detector = RegimeDetector()
    _state_df = detector.detect_series(full_df)   # kept for reference, not used in gates

    t_split   = pd.Timestamp("2026-08-01", tz="UTC")
    split_idx = int((full_df["timestamp"] < t_split).sum())
    n_total   = len(full_df)

    btc_hold_bh = (full_df["close"].iloc[-1] / full_df["close"].iloc[split_idx] - 1) * 100
    btc_full_bh = (full_df["close"].iloc[-1] / full_df["close"].iloc[55] - 1) * 100

    print(f"Dataset: {n_total} bars  ({full_df['timestamp'].iloc[0].date()} -> {full_df['timestamp'].iloc[-1].date()})")
    print(f"Earlier: bars 55–{split_idx-1} | Holdout: bars {split_idx}–{n_total-1} (Aug 1 – Sep 6)")
    print(f"BTC B&H — Full: {btc_full_bh:+.1f}%  |  Holdout: {btc_hold_bh:+.1f}%")

    strategy = MeanReversionStrategy()

    print("\n" + "=" * 90)
    print("  PHASE 7C: ACTIVATION GATE RESEARCH")
    print("  Predefined hypotheses evaluated on diagnostic period. Sep 22+ untouched.")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Primary evaluation: CALIBRATION RESTORATION on HOLDOUT
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  PRIMARY EVALUATION: SIGNAL CALIBRATION ON HOLDOUT (Aug 1 – Sep 6)")
    print("  Question: Does the gate remove the anti-predictive signals?")
    print("=" * 90)

    holdout_calibrations: list[tuple[str, pd.DataFrame]] = []
    for gid, gdesc, gfn in GATES:
        print(f"\n--- {gid}: {gdesc} ---")
        sig_df = extract_gated_signals(full_df, strategy, gfn,
                                       start_idx=split_idx, end_idx=n_total)
        holdout_calibrations.append((f"{gid} [{gdesc}]", sig_df))
        calibration_table(sig_df, f"{gid} Holdout")

    # ------------------------------------------------------------------
    # Calibration baseline on EARLIER period (for reference)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  BASELINE: SIGNAL CALIBRATION ON EARLIER PERIOD (Mar 14 – Jul 31)")
    print("  Reference: What 'working' signals look like.")
    print("=" * 90)

    g0_early = extract_gated_signals(full_df, strategy, gate_g0,
                                     start_idx=55, end_idx=split_idx)
    calibration_table(g0_early, "G0 Earlier (baseline)")

    # ------------------------------------------------------------------
    # Secondary evaluation: WALK-FORWARD on HOLDOUT
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SECONDARY EVALUATION: WALK-FORWARD ON HOLDOUT (Aug 1 – Sep 6)")
    print(f"  BTC B&H on this period: {btc_hold_bh:+.1f}%")
    print("=" * 90)

    wf_results: list[tuple[str, dict]] = []
    for gid, gdesc, gfn in GATES:
        r = run_gated_walkforward(full_df, strategy, gfn,
                                  start_idx=split_idx, end_idx=n_total)
        wf_results.append((f"{gid}: {gdesc}", r))
        print(f"\n  {gid}: {gdesc}")
        print(f"    Trades: {r['n']} (LONG {r['long_n']}, SHORT {r['short_n']})")
        print(f"    Win Rate: {r['win_rate']:.1f}%  |  PF: {r['pf']:.2f}  |  Sharpe: {r['sharpe']:+.2f}")
        print(f"    Max DD: {r['max_dd']:+.1f}%  |  Net Return: {r['net_ret']:+.2f}%")
        print(f"    LONG PF: {r['long_pf']:.2f}  |  SHORT PF: {r['short_pf']:.2f}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  GATE COMPARISON SUMMARY — HOLDOUT PERIOD")
    print("=" * 90)
    print(f"  {'Gate':<35} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8} {'Net Ret':>9}")
    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*9}")
    for label, r in wf_results:
        print(f"  {label:<35} {r['n']:>7} {r['win_rate']:>5.1f}% "
              f"{r['pf']:>6.2f} {r['sharpe']:>+7.2f} "
              f"{r['max_dd']:>+7.1f}% {r['net_ret']:>+8.2f}%")

    # ------------------------------------------------------------------
    # Decision framework
    # ------------------------------------------------------------------
    print("""

================================================================================
  GATE SELECTION CRITERIA
================================================================================

  A gate passes Phase 7C if it satisfies ALL of the following:

  1. CALIBRATION RESTORATION:
       - Allowed signals maintain positive 12h/24h expectancy on holdout
       - 12h confidence monotonicity is preserved (or not broken)

  2. SIGNAL PRESERVATION:
       - Suppressed signals are predominantly the anti-predictive population
         (SELL during sustained uptrend), not the genuine ones
       - Earlier-period signal count is not materially reduced

  3. WALK-FORWARD IMPROVEMENT:
       - Net return on holdout > G0 (improvement over no gate)
       - Max DD is not amplified relative to G0

  4. NO LEAKAGE:
       - Gate logic uses no threshold derived from the Aug-Sep holdout
       - All gate conditions are structural / causal

  The selected gate is then frozen (with Model B and risk geometry)
  and evaluated ONCE on Sep 22+.
""")


if __name__ == "__main__":
    main()
