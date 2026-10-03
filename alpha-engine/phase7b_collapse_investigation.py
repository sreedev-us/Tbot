"""
phase7b_collapse_investigation.py
====================================
Phase 7B: Model B Signal Collapse Investigation

Investigates WHY Model B's edge disappeared in the Aug 1 - Sep 6
diagnostic holdout, following these strict hypotheses in order:

  H1. Distribution shift in signal conditions
  H2. Confidence calibration stability (non-stationarity)
  H3. Signal inversion (did BUY/SELL relationship reverse or just weaken?)
  H4. Geometry (did the reversal still occur but too slowly/deeply?)

Rules:
  - No changes to Model B.
  - No retraining, no feature changes, no threshold changes.
  - Purpose is to EXPLAIN the failure, not rescue the backtest.
  - Sep 22+ untouched.

Split:
  - EARLIER:  Mar 12 – Jul 31, 2026 (~3408 bars)
  - HOLDOUT:  Aug 1  – Sep 6,  2026  (~863 bars)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_engine.regime_detector import RegimeDetector
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy
import xgboost as xgb

logging.basicConfig(level=logging.WARNING)

FEE_PCT = 0.20
MAX_HOLD = 24
HORIZONS = [1, 2, 4, 6, 12, 24]


# ---------------------------------------------------------------------------
# Signal extraction with anatomy and forward returns
# ---------------------------------------------------------------------------

def extract_signals(
    full_df: pd.DataFrame,
    state_df: pd.DataFrame,
    strategy: MeanReversionStrategy,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    """
    Evaluate Model B on every bar in [start_idx, end_idx].
    Record signal, confidence, anatomy, forward returns, MAE/MFE.
    """
    close = full_df["close"].values
    high  = full_df["high"].values
    low   = full_df["low"].values
    n = len(full_df)

    records = []

    for i in range(start_idx, min(end_idx, n - max(HORIZONS) - 1)):
        window_df = full_df.iloc[: i + 1]

        # --- Raw model output ---
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

        p_sell = float(probs[0])
        p_buy  = float(probs[2])

        sig, conf = 0, 0.0
        if p_buy > p_sell and p_buy >= strategy.confidence_threshold:
            sig, conf = 1, p_buy
        elif p_sell > p_buy and p_sell >= strategy.confidence_threshold:
            sig, conf = -1, p_sell

        if sig == 0:
            continue

        entry_price = close[i]

        # --- Anatomy metrics ---
        ema20 = float(window_df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(window_df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        dist_ema20 = (entry_price / ema20 - 1) * 100.0
        ema_spread = (ema20 / ema50 - 1) * 100.0   # positive = short > long = uptrend

        tr = np.maximum(
            window_df["high"] - window_df["low"],
            np.maximum(
                abs(window_df["high"] - window_df["close"].shift(1)),
                abs(window_df["low"]  - window_df["close"].shift(1))
            )
        )
        atr14 = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        atr_pct = atr14 / entry_price * 100.0

        atr_hist = tr.ewm(alpha=1/14, adjust=False).mean().tail(200)
        vol_pct   = float(stats.percentileofscore(atr_hist, atr14))

        adx_val = float(window_df["adx_14"].iloc[-1]) if "adx_14" in window_df.columns else 0.0

        # Realized volatility: std of returns over last 24h
        ret_24 = window_df["close"].pct_change().tail(24)
        realized_vol = float(ret_24.std() * 100.0)

        record: dict[str, Any] = {
            "bar_idx":       i,
            "timestamp":     window_df["timestamp"].iloc[-1],
            "signal":        sig,
            "direction":     "BUY" if sig == 1 else "SELL",
            "confidence":    conf,
            "p_buy":         p_buy,
            "p_sell":        p_sell,
            "trend":         state_df["trend"].iloc[i],
            "volatility":    state_df["volatility"].iloc[i],
            "dist_ema20_pct": dist_ema20,
            "ema_spread_pct": ema_spread,
            "atr_pct":       atr_pct,
            "vol_percentile": vol_pct,
            "adx":           adx_val,
            "realized_vol":  realized_vol,
        }

        # Forward returns
        for h in HORIZONS:
            fut_price = close[i + h]
            record[f"fwd_{h}h"] = sig * (fut_price - entry_price) / entry_price * 100.0
            record[f"raw_fwd_{h}h"] = (fut_price - entry_price) / entry_price * 100.0

        # MAE / MFE over max_hold
        window_h = full_df.iloc[i + 1 : i + 1 + MAX_HOLD]
        if not window_h.empty:
            if sig == 1:
                mfe = float((window_h["high"].max()  - entry_price) / entry_price * 100.0)
                mae = float((window_h["low"].min()   - entry_price) / entry_price * 100.0)
            else:
                mfe = float((entry_price - window_h["low"].min())  / entry_price * 100.0)
                mae = float((entry_price - window_h["high"].max()) / entry_price * 100.0)

            # Time to MFE
            if sig == 1:
                mfe_bar = int(window_h["high"].values.argmax())
                mae_bar = int(window_h["low"].values.argmin())
            else:
                mfe_bar = int(window_h["low"].values.argmin())
                mae_bar = int(window_h["high"].values.argmax())

            # Did actual reversal happen? (price crossed into profit before SL)
            tp_price = entry_price + sig * 2.0 * atr14
            sl_price = entry_price - sig * 1.0 * atr14
            reversal_occurred = False
            sl_occurred = False
            bars_to_reversal = None
            for j_off, row_j in enumerate(window_h.itertuples()):
                if sig == 1:
                    if row_j.high >= tp_price:
                        reversal_occurred = True
                        bars_to_reversal = j_off + 1
                        break
                    if row_j.low <= sl_price:
                        sl_occurred = True
                        bars_to_reversal = j_off + 1
                        break
                else:
                    if row_j.low <= tp_price:
                        reversal_occurred = True
                        bars_to_reversal = j_off + 1
                        break
                    if row_j.high >= sl_price:
                        sl_occurred = True
                        bars_to_reversal = j_off + 1
                        break
        else:
            mfe = mae = 0.0
            mfe_bar = mae_bar = 0
            reversal_occurred = sl_occurred = False
            bars_to_reversal = None

        record["mae"]                = mae
        record["mfe"]                = mfe
        record["bars_to_mfe"]        = mfe_bar
        record["bars_to_mae"]        = mae_bar
        record["tp_hit"]             = reversal_occurred
        record["sl_hit"]             = sl_occurred
        record["bars_to_resolution"] = bars_to_reversal

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def sub(title: str) -> None:
    print(f"\n  --- {title} ---")


def compare_stat(label: str, v_early: float, v_hold: float, fmt: str = "+.3f") -> None:
    delta = v_hold - v_early
    print(f"  {label:<35}  Earlier: {v_early:{fmt}}   Holdout: {v_hold:{fmt}}   Delta: {delta:{fmt}}")


def distribution_summary(label: str, early_s: pd.Series, hold_s: pd.Series) -> None:
    ks_stat, ks_p = stats.ks_2samp(early_s.dropna(), hold_s.dropna())
    print(f"\n  [{label}]")
    print(f"    Earlier  n={len(early_s):>4}  mean={early_s.mean():+.3f}  std={early_s.std():.3f}  p10={early_s.quantile(.1):+.3f}  p90={early_s.quantile(.9):+.3f}")
    print(f"    Holdout  n={len(hold_s):>4}  mean={hold_s.mean():+.3f}  std={hold_s.std():.3f}  p10={hold_s.quantile(.1):+.3f}  p90={hold_s.quantile(.9):+.3f}")
    print(f"    KS test: stat={ks_stat:.3f}, p={ks_p:.4f} {'** SIGNIFICANT **' if ks_p < 0.05 else '(not significant)'}")


def fwd_ret_table(df: pd.DataFrame, label: str) -> None:
    sub(label)
    print(f"    {'Direction':<8} {'n':>5}  " + "  ".join(f"{h}h" for h in HORIZONS))
    print(f"    {'-'*8} {'--':>5}  " + "  ".join("----" for _ in HORIZONS))
    for direction, grp in df.groupby("direction"):
        means = [f"{grp[f'fwd_{h}h'].mean():+.3f}%" for h in HORIZONS]
        print(f"    {direction:<8} {len(grp):>5}  {'  '.join(means)}")
    total = df.groupby(pd.Series(["ALL"] * len(df)))[
        [f"fwd_{h}h" for h in HORIZONS]
    ].mean()
    means = [f"{total[f'fwd_{h}h'].iloc[0]:+.3f}%" for h in HORIZONS]
    print(f"    {'ALL':<8} {len(df):>5}  {'  '.join(means)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading {data_path}...")
    raw_df = pd.read_csv(data_path)

    config   = TrainingDataConfig()
    gen      = TrainingDataGenerator(config)
    full_df  = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df  = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    detector = RegimeDetector()
    state_df = detector.detect_series(full_df)

    # Split boundary
    t0      = full_df["timestamp"].iloc[0]
    t_split = pd.Timestamp("2026-08-01", tz="UTC")
    split_idx = int((full_df["timestamp"] < t_split).sum())

    print(f"Dataset: {len(full_df)} bars  ({full_df['timestamp'].iloc[0].date()} -> {full_df['timestamp'].iloc[-1].date()})")
    print(f"Earlier period: bars 55–{split_idx-1}  ({full_df['timestamp'].iloc[55].date()} to {full_df['timestamp'].iloc[split_idx-1].date()})")
    print(f"Holdout period: bars {split_idx}–end  ({t_split.date()} to {full_df['timestamp'].iloc[-1].date()})")

    strategy = MeanReversionStrategy()

    print("\nExtracting signals from EARLIER period...")
    early_df = extract_signals(full_df, state_df, strategy, start_idx=55, end_idx=split_idx)

    print(f"Extracting signals from HOLDOUT period...")
    hold_df  = extract_signals(full_df, state_df, strategy, start_idx=split_idx, end_idx=len(full_df))

    print(f"\nSignals found — Earlier: {len(early_df)}  |  Holdout: {len(hold_df)}")

    # =====================================================================
    # H1. DISTRIBUTION SHIFT IN SIGNAL CONDITIONS
    # =====================================================================
    section("H1: DISTRIBUTION SHIFT IN SIGNAL CONDITIONS")

    sub("Signal frequency and direction")
    n_bars_early = split_idx - 55
    n_bars_hold  = len(full_df) - split_idx
    e_rate = len(early_df) / n_bars_early * 100
    h_rate = len(hold_df)  / n_bars_hold  * 100
    print(f"  Earlier: {len(early_df)} signals in {n_bars_early} bars  = {e_rate:.2f}% signal rate")
    print(f"  Holdout: {len(hold_df)}  signals in {n_bars_hold}  bars  = {h_rate:.2f}% signal rate")

    e_buy_pct = (early_df["signal"] == 1).mean() * 100
    h_buy_pct = (hold_df["signal"]  == 1).mean() * 100
    compare_stat("BUY signal fraction (%)", e_buy_pct, h_buy_pct, ".1f")

    for feat, lbl in [
        ("confidence",     "Confidence"),
        ("dist_ema20_pct", "Distance from EMA20 (%)"),
        ("ema_spread_pct", "EMA20/EMA50 spread (%)"),
        ("atr_pct",        "ATR % of price"),
        ("vol_percentile", "ATR percentile (vol regime)"),
        ("adx",            "ADX-14"),
        ("realized_vol",   "Realized Vol (24h std %)"),
    ]:
        distribution_summary(lbl, early_df[feat], hold_df[feat])

    sub("Trend-state distribution at time of signal")
    e_trend = early_df["trend"].value_counts(normalize=True) * 100
    h_trend = hold_df["trend"].value_counts(normalize=True) * 100
    print(f"  {'State':<12}  Earlier%   Holdout%")
    for state in ["range", "uptrend", "downtrend", "neutral"]:
        print(f"  {state:<12}  {e_trend.get(state, 0.0):>7.1f}%  {h_trend.get(state, 0.0):>8.1f}%")

    # =====================================================================
    # H2. CONFIDENCE CALIBRATION STABILITY
    # =====================================================================
    section("H2: CONFIDENCE CALIBRATION STABILITY (Non-Stationarity)")

    sub("Forward returns by confidence quintile — EARLIER PERIOD")
    early_df["conf_bin"] = pd.qcut(
        early_df["confidence"], q=5,
        labels=["Low", "Med-Low", "Med", "Med-High", "High"]
    )
    e_cal = early_df.groupby("conf_bin", observed=True)[
        ["confidence", "fwd_4h", "fwd_12h", "fwd_24h"]
    ].agg({"confidence": ["count", "mean"], "fwd_4h": "mean", "fwd_12h": "mean", "fwd_24h": "mean"})
    e_cal.columns = ["n", "avg_conf", "fwd_4h", "fwd_12h", "fwd_24h"]
    print(f"\n  {'Bin':<10} {'n':>5}  {'AvgConf':>8}  {'4h Ret':>8}  {'12h Ret':>8}  {'24h Ret':>8}")
    for bin_name, row in e_cal.iterrows():
        print(f"  {bin_name:<10} {int(row['n']):>5}  {row['avg_conf']:>8.4f}  "
              f"{row['fwd_4h']:>+8.3f}%  {row['fwd_12h']:>+8.3f}%  {row['fwd_24h']:>+8.3f}%")

    sub("Forward returns by confidence quintile — HOLDOUT PERIOD")
    # Use same quintile edges from earlier period
    try:
        bins = e_cal.index.tolist()
        hold_df["conf_bin"] = pd.cut(
            hold_df["confidence"],
            bins=pd.qcut(early_df["confidence"], q=5, retbins=True)[1],
            labels=["Low", "Med-Low", "Med", "Med-High", "High"],
            include_lowest=True
        )
    except Exception:
        hold_df["conf_bin"] = pd.qcut(
            hold_df["confidence"], q=5,
            labels=["Low", "Med-Low", "Med", "Med-High", "High"],
            duplicates="drop"
        )

    h_cal = hold_df.groupby("conf_bin", observed=True)[
        ["confidence", "fwd_4h", "fwd_12h", "fwd_24h"]
    ].agg({"confidence": ["count", "mean"], "fwd_4h": "mean", "fwd_12h": "mean", "fwd_24h": "mean"})
    h_cal.columns = ["n", "avg_conf", "fwd_4h", "fwd_12h", "fwd_24h"]
    print(f"\n  {'Bin':<10} {'n':>5}  {'AvgConf':>8}  {'4h Ret':>8}  {'12h Ret':>8}  {'24h Ret':>8}")
    for bin_name, row in h_cal.iterrows():
        print(f"  {bin_name:<10} {int(row['n']):>5}  {row['avg_conf']:>8.4f}  "
              f"{row['fwd_4h']:>+8.3f}%  {row['fwd_12h']:>+8.3f}%  {row['fwd_24h']:>+8.3f}%")

    sub("Confidence ordering: is monotonicity preserved?")
    e_mono = list(e_cal["fwd_12h"])
    h_mono = list(h_cal["fwd_12h"].fillna(0.0))
    e_ordered = all(e_mono[i] <= e_mono[i+1] for i in range(len(e_mono)-1))
    h_ordered = all(h_mono[i] <= h_mono[i+1] for i in range(len(h_mono)-1))
    print(f"  Earlier 12h calibration monotone:  {e_ordered}")
    print(f"  Holdout 12h calibration monotone:  {h_ordered}")

    # =====================================================================
    # H3. SIGNAL INVERSION TEST
    # =====================================================================
    section("H3: SIGNAL INVERSION TEST (Did BUY/SELL relationship reverse?)")

    fwd_ret_table(early_df, "EARLIER PERIOD — forward returns by direction (signal-adjusted)")
    fwd_ret_table(hold_df,  "HOLDOUT PERIOD — forward returns by direction (signal-adjusted)")

    sub("Raw (unsigned) forward returns — what ACTUALLY happened to price")
    print("  (Positive = price went UP regardless of signal direction)")
    for period_label, df_ in [("Earlier", early_df), ("Holdout", hold_df)]:
        print(f"\n  [{period_label}]")
        for direction, grp in df_.groupby("direction"):
            raw_means = [f"{grp[f'raw_fwd_{h}h'].mean():+.3f}%" for h in HORIZONS]
            print(f"    Model {direction}: price moved {' | '.join(raw_means)}")

    sub("Inversion verdict")
    e_buy_12 = early_df[early_df["direction"] == "BUY"]["fwd_12h"].mean()
    e_sel_12 = early_df[early_df["direction"] == "SELL"]["fwd_12h"].mean()
    h_buy_12 = hold_df[hold_df["direction"]   == "BUY"]["fwd_12h"].mean()
    h_sel_12 = hold_df[hold_df["direction"]   == "SELL"]["fwd_12h"].mean()

    print(f"  BUY  signal 12h return:  Earlier {e_buy_12:+.3f}%  |  Holdout {h_buy_12:+.3f}%")
    print(f"  SELL signal 12h return:  Earlier {e_sel_12:+.3f}%  |  Holdout {h_sel_12:+.3f}%")
    e_spread = e_buy_12 - e_sel_12
    h_spread = h_buy_12 - h_sel_12
    print(f"  BUY - SELL spread:       Earlier {e_spread:+.3f}%  |  Holdout {h_spread:+.3f}%")
    if h_buy_12 < 0 and h_sel_12 > 0:
        print("  VERDICT: *** FULL INVERSION — BUY returns negative, SELL returns positive ***")
    elif (h_buy_12 < e_buy_12 * 0.3) and (h_sel_12 < e_sel_12 * 0.3):
        print("  VERDICT: Both directions degraded — edge weakened across the board (no inversion)")
    elif h_spread < 0:
        print("  VERDICT: *** PARTIAL INVERSION — BUY/SELL spread crossed zero ***")
    else:
        print("  VERDICT: Directional relationship preserved but magnitude weakened")

    # =====================================================================
    # H4. GEOMETRY — Did the reversal still happen?
    # =====================================================================
    section("H4: GEOMETRY — Did the reversal still occur?")

    sub("TP hit rate, SL hit rate, and time to resolution")
    for period_label, df_ in [("Earlier", early_df), ("Holdout", hold_df)]:
        tp_rate = df_["tp_hit"].mean() * 100
        sl_rate = df_["sl_hit"].mean() * 100
        no_touch = 100 - tp_rate - sl_rate
        avg_bars_res = df_["bars_to_resolution"].dropna().mean()
        avg_mfe = df_["mfe"].mean()
        avg_mae = df_["mae"].mean()
        avg_bars_mfe = df_["bars_to_mfe"].mean()
        avg_bars_mae = df_["bars_to_mae"].mean()
        print(f"\n  [{period_label}]  n={len(df_)}")
        print(f"    TP hit:           {tp_rate:.1f}%   (avg {avg_bars_res:.1f} bars to resolution)")
        print(f"    SL hit:           {sl_rate:.1f}%")
        print(f"    Held to max_hold: {no_touch:.1f}%")
        print(f"    MFE: {avg_mfe:+.2f}%  (avg {avg_bars_mfe:.1f} bars to peak)")
        print(f"    MAE: {avg_mae:+.2f}%  (avg {avg_bars_mae:.1f} bars to trough)")

    sub("By direction: TP/SL hit rates")
    for period_label, df_ in [("Earlier", early_df), ("Holdout", hold_df)]:
        print(f"\n  [{period_label}]")
        for direction, grp in df_.groupby("direction"):
            tp = grp["tp_hit"].mean() * 100
            sl = grp["sl_hit"].mean() * 100
            mfe = grp["mfe"].mean()
            mae = grp["mae"].mean()
            print(f"    {direction:<5}: TP {tp:.1f}%  SL {sl:.1f}%  MFE {mfe:+.2f}%  MAE {mae:+.2f}%")

    sub("Key geometry question: Did reversals still happen (TP) even if later?")
    for period_label, df_ in [("Earlier", early_df), ("Holdout", hold_df)]:
        tp_rate = df_["tp_hit"].mean() * 100
        print(f"  [{period_label}]  TP hit rate: {tp_rate:.1f}%   "
              f"=> {'Reversion still occurring' if tp_rate > 40 else '*** REVERSAL DISAPPEARED ***'}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    section("COLLAPSE INVESTIGATION SUMMARY")

    print("""
  The investigation above answers four questions about the holdout failure:

  H1 — Was it distribution shift?
       Compare the KS test results for each anatomy feature.
       If vol_percentile / ATR shifted significantly, the market was moving
       at a different pace than the training distribution.

  H2 — Did confidence stay calibrated?
       If monotonicity breaks in holdout, the model's probability estimates
       became unreliable signals (not just weaker — uncalibrated).
       If monotonicity holds but all bins are negative, the entire signal
       conditional distribution shifted (meaningful but different problem).

  H3 — Did the edge invert?
       Full inversion = BUY returns negative, SELL returns positive.
       That's an adversarial regime. Partial inversion = same but crossed zero.
       Weakening only = evidence of regime change, not adversarial.

  H4 — Did geometry kill it?
       If TP hit rate stayed comparable (>40%) but net return fell,
       the geometry (1x ATR SL) was likely stopping out good trades.
       If TP hit rate collapsed entirely, the reversal itself disappeared.
       These have completely different fixes.
    """)


if __name__ == "__main__":
    main()
