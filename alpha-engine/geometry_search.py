#!/usr/bin/env python3
"""
Walk-forward geometry search for 15m and 1h timeframes.
Tests fixed percentage TP/SL and scaled ATR multipliers against a 0.20% baseline fee.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model

logging.basicConfig(level=logging.WARNING)

def evaluate_geometry(booster, scaler, feature_columns, test_df, 
                      confidence_threshold=0.90, tp_pct=None, sl_pct=None, 
                      atr_tp=None, atr_sl=None, fee_pct=0.20, max_hold_bars=20):
    
    df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for inference: {missing}")

    X = df[feature_columns].values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    
    # Pre-compute ATR-14 if needed
    if atr_tp is not None or atr_sl is not None:
        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
        atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        probs = booster.predict(import_xgb().DMatrix(scaled))

    pnls = []
    i = 0
    while i < len(df) - max_hold_bars:
        p_sell, p_hold, p_buy = probs[i][0], probs[i][1], probs[i][2]

        if p_buy > p_sell and p_buy >= confidence_threshold:
            direction = 1
        elif p_sell > p_buy and p_sell >= confidence_threshold:
            direction = -1
        else:
            i += 1
            continue

        entry = close[i]
        
        if tp_pct is not None and sl_pct is not None:
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

        trade_pnl_pct -= fee_pct
        pnls.append(trade_pnl_pct)
        i = exit_bar + 1

    if not pnls:
        return None

    pnls = np.array(pnls)
    winning = (pnls > 0).sum()
    total = len(pnls)
    win_rate = winning / total
    
    gross_profit = pnls[pnls > 0].sum() if winning > 0 else 0
    gross_loss = abs(pnls[pnls <= 0].sum()) if (total - winning) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    
    cum_ret = pnls.sum()
    mean_ret = pnls.mean()
    std_ret = pnls.std() if len(pnls) > 1 else 0
    sharpe = (mean_ret / std_ret * np.sqrt(total)) if std_ret > 1e-6 else 0.0
    
    cum_series = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_series)
    drawdowns = cum_series - running_max
    max_dd = drawdowns.min() if len(drawdowns) > 0 else 0.0

    return {
        "trades": total,
        "win_rate": win_rate,
        "pf": profit_factor,
        "sharpe": sharpe,
        "maxdd": max_dd,
        "cum_ret": cum_ret
    }

def import_xgb():
    import xgboost as xgb
    return xgb

def run_geometry_search(timeframe, filepath, model_name, max_hold_bars):
    print(f"\n{'='*90}")
    print(f"GEOMETRY SEARCH: {timeframe} ({model_name}) | Baseline Cost: 0.20%")
    print(f"{'='*90}")
    
    booster, scaler, fc = load_model(Path("models") / model_name)
    raw_df = pd.read_csv(filepath)
    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    
    # We generate features (no target needed for inference)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(timestamp=pd.to_datetime(full_df["timestamp"], utc=True))
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    
    t0 = full_df["timestamp"].iloc[0]
    # Walk-forward windows (Q1 to Q4 = 180 days)
    windows = [
        t0 + pd.Timedelta(days=0),
        t0 + pd.Timedelta(days=45),
        t0 + pd.Timedelta(days=90),
        t0 + pd.Timedelta(days=135),
        t0 + pd.Timedelta(days=180)
    ]
    
    # We will test geometries at a few confidence thresholds
    thresholds = [0.85, 0.90, 0.95]
    
    geometries = [
        {"name": "Fixed 0.5% / 0.25%", "kwargs": {"tp_pct": 0.5, "sl_pct": 0.25}},
        {"name": "Fixed 1.0% / 0.5%", "kwargs": {"tp_pct": 1.0, "sl_pct": 0.5}},
        {"name": "Fixed 1.5% / 0.75%", "kwargs": {"tp_pct": 1.5, "sl_pct": 0.75}},
        {"name": "Fixed 2.0% / 1.0%", "kwargs": {"tp_pct": 2.0, "sl_pct": 1.0}},
        {"name": "ATR 2.0 / 1.0", "kwargs": {"atr_tp": 2.0, "atr_sl": 1.0}},
        {"name": "ATR 3.0 / 1.5", "kwargs": {"atr_tp": 3.0, "atr_sl": 1.5}},
        {"name": "ATR 4.0 / 2.0", "kwargs": {"atr_tp": 4.0, "atr_sl": 2.0}},
    ]
    
    print(f"{'Geometry':<20} {'Thresh':>6} {'Trades':>6} {'WinRate':>7} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'CumRet%':>8}")
    print("-" * 90)
    
    best_ret = -999
    best_geo = None
    
    for geo in geometries:
        for t in thresholds:
            all_pnls = []
            metrics = []
            
            for i in range(len(windows)-1):
                start_ts = windows[i]
                end_ts = windows[i+1]
                window_df = full_df[(full_df["timestamp"] >= start_ts) & (full_df["timestamp"] < end_ts)]
                
                if len(window_df) < 50:
                    continue
                
                res = evaluate_geometry(
                    booster, scaler, fc, window_df, 
                    confidence_threshold=t, 
                    fee_pct=0.20,
                    max_hold_bars=max_hold_bars,
                    **geo["kwargs"]
                )
                if res:
                    metrics.append(res)
            
            if metrics:
                total_trades = sum(m["trades"] for m in metrics)
                avg_wr = np.mean([m["win_rate"] for m in metrics])
                avg_pf = np.mean([m["pf"] for m in metrics])
                avg_sharpe = np.mean([m["sharpe"] for m in metrics])
                avg_maxdd = np.mean([m["maxdd"] for m in metrics])
                total_ret = sum(m["cum_ret"] for m in metrics)
                
                mark = " ***" if total_ret > 0 and total_trades > 20 else ("  **" if avg_pf > 1.0 and total_trades > 20 else "    ")
                print(f"{geo['name']:<20} {t:>6.2f} {total_trades:>6d} {avg_wr*100:>6.1f}% {avg_pf:>6.2f} {avg_sharpe:>+7.2f} {avg_maxdd*100:>6.1f}% {total_ret:>+8.1f}%{mark}")
                
                if total_ret > best_ret and total_trades > 20:
                    best_ret = total_ret
                    best_geo = f"{geo['name']} at {t:.2f}"
    
    print(f"\nBest Geometry for {timeframe}: {best_geo} with CumRet {best_ret:+.1f}%")

if __name__ == "__main__":
    run_geometry_search("15m", "data/live/BTC_USDT_15m_live.csv", "local-model-b-15m", max_hold_bars=40)
    run_geometry_search("1h", "data/live/BTC_USDT_1h_live.csv", "local-model-b-1h", max_hold_bars=24)
