"""
phase8_final_untouched_test.py
==============================
Phase 8: Final Untouched Test (Sep 6 - Sep 27, 2026)

Tests the ORIGINAL FROZEN ARCHITECTURE (Model B + Phase 5 RegimeDetector,
no G1 gate) on the completely unseen data from Sep 6 onwards.

This script combines the old data with the fresh data (to provide sufficient
lookback for moving averages and ATR), then extracts signals and runs a walk-forward
ONLY on the fresh data (bars that occurred after the cutoff).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from alpha_engine.regime_detector import RegimeDetector, MarketState
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy

logging.basicConfig(level=logging.WARNING)

FEE_PCT = 0.20
MAX_HOLD = 24
HORIZONS = [1, 4, 12, 24]


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


def main() -> None:
    # 1. Load and combine data
    live_path = Path("data/live/BTC_USDT_1h_live.csv")
    fresh_path = Path("data/live/BTC_USDT_1h_fresh.csv")
    
    df_live = pd.read_csv(live_path)
    df_fresh = pd.read_csv(fresh_path)
    
    df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], utc=True)
    df_fresh["timestamp"] = pd.to_datetime(df_fresh["timestamp"], utc=True)
    
    # Concat and drop duplicates (in case of overlap)
    raw_df = pd.concat([df_live, df_fresh]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # 2. Engineer features
    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(timestamp=pd.to_datetime(full_df["timestamp"], utc=True)).sort_values("timestamp").reset_index(drop=True)
    
    detector = RegimeDetector()
    state_df = detector.detect_series(full_df)
    
    strategy = MeanReversionStrategy()
    
    # 3. Find the split point for fresh data
    t_fresh = df_fresh["timestamp"].iloc[0]
    split_idx = int((full_df["timestamp"] < t_fresh).sum())
    n_total = len(full_df)
    
    print("\n" + "=" * 90)
    print("  PHASE 8: FINAL UNTOUCHED TEST (SEP 6 - SEP 27, 2026)")
    print("  Testing ORIGINAL FROZEN ARCHITECTURE (Model B + Phase 5 RegimeDetector)")
    print("=" * 90)
    
    print(f"\n  Dataset boundaries:")
    print(f"    Lookback period: {full_df['timestamp'].iloc[0]} to {full_df['timestamp'].iloc[split_idx-1]}")
    print(f"    Untouched test:  {full_df['timestamp'].iloc[split_idx]} to {full_df['timestamp'].iloc[-1]} ({n_total - split_idx} bars)")
    
    btc_bh = (full_df["close"].iloc[-1] / full_df["close"].iloc[split_idx] - 1) * 100
    print(f"    BTC Buy & Hold:  {btc_bh:+.2f}%\n")
    
    # 4. Extract signals and calibration metrics on fresh data
    close = full_df["close"].values
    high  = full_df["high"].values
    low   = full_df["low"].values
    
    records = []
    
    for i in range(split_idx, min(n_total, n_total - MAX_HOLD - 1)):
        window_df = full_df.iloc[: i + 1]
        
        # Original Model B eligibility (Range only)
        trend = state_df["trend"].iloc[i]
        volatility = state_df["volatility"].iloc[i]
        state = MarketState(trend=trend, volatility=volatility)
        
        if not strategy.is_eligible(state):
            continue
            
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
            
        entry = close[i]
        
        record = {
            "bar_idx": i,
            "signal": sig,
            "direction": "BUY" if sig == 1 else "SELL",
            "confidence": conf,
        }
        
        for h in HORIZONS:
            fut = close[min(i + h, n_total - 1)]
            record[f"fwd_{h}h"] = sig * (fut - entry) / entry * 100.0
            record[f"raw_fwd_{h}h"] = (fut - entry) / entry * 100.0
            
        records.append(record)
        
    sig_df = pd.DataFrame(records)
    
    # 5. Run Walk-Forward on fresh data
    atr_s = BaseStrategy._atr(full_df, period=14).values
    equity = 1.0
    equity_curve = [equity]
    trades = []
    l_pnls = []
    s_pnls = []
    
    i = split_idx
    while i < min(n_total, n_total - MAX_HOLD - 1):
        window_df = full_df.iloc[: i + 1]
        
        trend = state_df["trend"].iloc[i]
        volatility = state_df["volatility"].iloc[i]
        state = MarketState(trend=trend, volatility=volatility)
        
        if not strategy.is_eligible(state):
            i += 1
            continue
            
        sig = strategy.signal(window_df)
        if sig == 0:
            i += 1
            continue
            
        entry   = close[i]
        bar_atr = atr_s[i]
        tp_p    = entry + sig * 2.0 * bar_atr
        sl_p    = entry - sig * 1.0 * bar_atr
        exit_b  = min(i + MAX_HOLD, n_total - 1)
        pnl     = None
        reason  = "max_hold"
        
        for j in range(i + 1, min(i + MAX_HOLD + 1, n_total)):
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
        
    wf_res = _wf_metrics(trades, equity_curve, l_pnls, s_pnls)
    
    # 6. Print Results answering the user's explicit questions
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"  Trades:        {wf_res['n']} (LONG {wf_res['long_n']}, SHORT {wf_res['short_n']})")
    print(f"  Win Rate:      {wf_res['win_rate']:.1f}%")
    print(f"  Profit Factor: {wf_res['pf']:.2f}")
    print(f"  Sharpe:        {wf_res['sharpe']:+.2f}")
    print(f"  Max DD:        {wf_res['max_dd']:+.2f}%")
    print(f"  Net Return:    {wf_res['net_ret']:+.2f}%")
    
    if not sig_df.empty:
        print("\n--- 1. Does Model B retain positive 12h/24h expectancy? ---")
        for direction, grp in sig_df.groupby("direction"):
            print(f"  {direction}: 12h = {grp['fwd_12h'].mean():+.3f}% | 24h = {grp['fwd_24h'].mean():+.3f}%")
        print(f"  ALL : 12h = {sig_df['fwd_12h'].mean():+.3f}% | 24h = {sig_df['fwd_24h'].mean():+.3f}%")
        
        print("\n--- 2. Does confidence remain monotonically related to realized return? ---")
        try:
            cal = sig_df.groupby(
                pd.qcut(sig_df["confidence"], q=5, duplicates="drop"), observed=True
            )["fwd_12h"].mean().values.tolist()
            mono = all(cal[k] <= cal[k+1] for k in range(len(cal)-1))
            print(f"  Monotonic: {mono} ({' -> '.join(f'{v:+.3f}%' for v in cal)})")
        except Exception:
            print("  Not enough data for quintile split.")
            
        print("\n--- 3. Does BUY/SELL directional behavior remain valid? ---")
        print("\n--- 4. Does the signal relationship invert again? ---")
        for direction, grp in sig_df.groupby("direction"):
            print(f"  Model {direction} -> Price actually moved 12h: {grp['raw_fwd_12h'].mean():+.3f}%")
    else:
        print("\n--- No signals generated in this period. ---")

    print("\n--- 5 & 6. Cost-adjusted expectancy and drawdown ---")
    print(f"  Cost-Adjusted Expectancy (Net Ret / Trades): {(wf_res['net_ret']/wf_res['n'] if wf_res['n'] > 0 else 0):+.3f}%")
    print(f"  Max Drawdown: {wf_res['max_dd']:+.2f}%")
    
    print("\n--- 7. Does the model behave differently from the Aug-Sep collapse? ---")
    print("  (Compare these results with the Phase 7B/C holdout performance)")


if __name__ == "__main__":
    main()
