#!/usr/bin/env python3
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_model(model_path: Path):
    meta = json.loads((model_path / "metadata.json").read_text())
    with (model_path / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    json_model = model_path / "model.json"
    if json_model.exists():
        booster = xgb.Booster()
        booster.load_model(str(json_model))
    else:
        with (model_path / "model.pkl").open("rb") as f:
            booster = pickle.load(f)
    return booster, scaler, meta["feature_columns"]

def main():
    model_path = Path("models/local-benchmark-b")
    data_path = Path("data/live/BTC_USDT_1m_live.csv")

    logger.info("Loading model B...")
    booster, scaler, features = load_model(model_path)

    logger.info("Loading and processing data...")
    raw_df = pd.read_csv(data_path)
    generator = TrainingDataGenerator(TrainingDataConfig())
    full_df = generator.generate_training_data(raw_df)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    for col in features:
        if col not in full_df.columns:
            full_df[col] = 0.0

    X = full_df[features]
    close = full_df["close"].values
    high = full_df["high"].values
    low = full_df["low"].values

    # Pre-compute ATR-14
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    scaled = scaler.transform(X)
    if hasattr(booster, "predict_proba"):
        probs = booster.predict_proba(scaled)
    else:
        probs = booster.predict(xgb.DMatrix(scaled))

    fee_pct = 0.10
    max_hold_bars = 20
    atr_tp = 2.0
    atr_sl = 1.0

    buckets = {
        "0.50-0.60": [],
        "0.60-0.70": [],
        "0.70-0.80": [],
        "0.80-0.90": [],
        "0.90-1.00": []
    }

    logger.info("Simulating EVERY signal for pure expectancy analysis...")
    
    for i in range(len(full_df) - max_hold_bars):
        p_sell, p_hold, p_buy = probs[i]
        
        direction = 0
        if p_buy > p_sell and p_buy >= 0.50:
            direction = 1
            conf = p_buy
        elif p_sell > p_buy and p_sell >= 0.50:
            direction = -1
            conf = p_sell
            
        if direction == 0:
            continue
            
        # Determine bucket
        if 0.50 <= conf < 0.60: b_key = "0.50-0.60"
        elif 0.60 <= conf < 0.70: b_key = "0.60-0.70"
        elif 0.70 <= conf < 0.80: b_key = "0.70-0.80"
        elif 0.80 <= conf < 0.90: b_key = "0.80-0.90"
        else: b_key = "0.90-1.00"
            
        entry = close[i]
        bar_atr = atr[i]
        tp_price = entry + direction * atr_tp * bar_atr
        sl_price = entry - direction * atr_sl * bar_atr
        
        trade_pnl = None
        for j in range(i + 1, i + max_hold_bars + 1):
            if direction == 1:
                if high[j] >= tp_price:
                    trade_pnl = (tp_price - entry) / entry * 100.0
                    break
                if low[j] <= sl_price:
                    trade_pnl = (sl_price - entry) / entry * 100.0
                    break
            else:
                if low[j] <= tp_price:
                    trade_pnl = (entry - tp_price) / entry * 100.0
                    break
                if high[j] >= sl_price:
                    trade_pnl = (entry - sl_price) / entry * 100.0
                    break
                    
        if trade_pnl is None:
            trade_pnl = direction * (close[i + max_hold_bars] - entry) / entry * 100.0
            
        trade_pnl -= fee_pct  # round-trip fee
        buckets[b_key].append(trade_pnl)

    print("\n" + "="*80)
    print("PURE SIGNAL EXPECTANCY BY CONFIDENCE BUCKET (MODEL B)")
    print("="*80)
    print(f"{'Bucket':<15} {'Count':<10} {'WinRate':<10} {'Avg PnL':<10} {'Total PnL':<10}")
    print("-" * 80)
    
    total_trades = 0
    total_pnl = 0.0
    for b_key in sorted(buckets.keys()):
        trades = buckets[b_key]
        count = len(trades)
        total_trades += count
        if count == 0:
            print(f"{b_key:<15} {count:<10} {'-':<10} {'-':<10} {'-':<10}")
            continue
            
        arr = np.array(trades)
        win_rate = (arr > 0).mean() * 100
        avg_pnl = arr.mean()
        sum_pnl = arr.sum()
        total_pnl += sum_pnl
        print(f"{b_key:<15} {count:<10} {win_rate:>6.1f}%    {avg_pnl:>+7.3f}%   {sum_pnl:>+8.1f}%")

if __name__ == "__main__":
    main()
