#!/usr/bin/env python3
"""
Diagnostic script for analyzing the failure of the 1h model on the holdout period.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from walkforward_ab import load_model

logging.basicConfig(level=logging.WARNING)

def analyze_holdout(booster, scaler, feature_cols, df, confidence_threshold=0.85, atr_tp=2.0, atr_sl=1.0, fee_pct=0.20, max_hold_bars=24):
    df = df.reset_index(drop=True)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

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
        
    p_down, p_hold, p_up = probs[:, 0], probs[:, 1], probs[:, 2]
    
    # Base market metrics
    btc_return = (close[-1] - close[0]) / close[0] * 100
    btc_volatility = pd.Series(close).pct_change().std() * np.sqrt(24 * 365) * 100 # Annualized hourly
    
    # Directional movement
    up_bars = (close[1:] > close[:-1]).sum() / (len(close)-1) * 100
    down_bars = (close[1:] < close[:-1]).sum() / (len(close)-1) * 100
    
    # Signal distribution
    buy_mask = (p_up > p_down) & (p_up >= confidence_threshold)
    sell_mask = (p_down > p_up) & (p_down >= confidence_threshold)
    hold_mask = ~(buy_mask | sell_mask)
    
    total_bars = len(df) - max_hold_bars
    if total_bars <= 0:
        return
        
    buy_pct = buy_mask[:total_bars].sum() / total_bars * 100
    sell_pct = sell_mask[:total_bars].sum() / total_bars * 100
    hold_pct = hold_mask[:total_bars].sum() / total_bars * 100
    
    def eval_signal(mask, direction):
        indices = np.where(mask[:total_bars])[0]
        if len(indices) == 0:
            return 0, 0, 0, 0
            
        pnls = []
        for i in indices:
            entry = close[i]
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
            
        pnls = np.array(pnls)
        winning = (pnls > 0).sum()
        win_rate = winning / len(pnls) * 100
        avg_ret = pnls.mean()
        gross_profit = pnls[pnls > 0].sum() if winning > 0 else 0
        gross_loss = abs(pnls[pnls <= 0].sum()) if (len(pnls) - winning) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
        return len(pnls), avg_ret, win_rate, pf

    print("HOLDOUT DIAGNOSTIC")
    print("-" * 28)
    print("\nSignal distribution (Threshold >= 0.85)")
    print(f"BUY:   {buy_pct:.1f}%")
    print(f"SELL:  {sell_pct:.1f}%")
    print(f"HOLD:  {hold_pct:.1f}%")
    
    print("\nActual directional movement")
    print(f"UP:    {up_bars:.1f}%")
    print(f"DOWN:  {down_bars:.1f}%")
    
    print("\nConditional performance")
    print("BUY signals:")
    b_count, b_ret, b_wr, b_pf = eval_signal(buy_mask, 1)
    print(f"    count:                 {b_count}")
    print(f"    average future return: {b_ret:+.3f}%")
    print(f"    win rate:              {b_wr:.1f}%")
    print(f"    profit factor:         {b_pf:.2f}")
    
    print("SELL signals:")
    s_count, s_ret, s_wr, s_pf = eval_signal(sell_mask, -1)
    print(f"    count:                 {s_count}")
    print(f"    average future return: {s_ret:+.3f}%")
    print(f"    win rate:              {s_wr:.1f}%")
    print(f"    profit factor:         {s_pf:.2f}")
    
    print("\nConfidence buckets (Overall Performance)")
    for low_t, high_t in [(0.85, 0.90), (0.90, 0.95), (0.95, 1.00)]:
        b_mask = (p_up > p_down) & (p_up >= low_t) & (p_up < high_t)
        s_mask = (p_down > p_up) & (p_down >= low_t) & (p_down < high_t)
        
        c_b_count, c_b_ret, c_b_wr, c_b_pf = eval_signal(b_mask, 1)
        c_s_count, c_s_ret, c_s_wr, c_s_pf = eval_signal(s_mask, -1)
        
        total_count = c_b_count + c_s_count
        if total_count == 0:
            print(f"{low_t:.2f}–{high_t:.2f}: No signals")
        else:
            w = (c_b_count * c_b_wr + c_s_count * c_s_wr) / total_count
            r = (c_b_count * c_b_ret + c_s_count * c_s_ret) / total_count
            print(f"{low_t:.2f}–{high_t:.2f}: count={total_count}, win%={w:.1f}%, avg_ret={r:+.3f}%")

    print("\nMarket")
    print(f"BTC return over holdout: {btc_return:+.2f}%")
    print(f"BTC volatility (ann.):   {btc_volatility:.2f}%")
    print(f"BTC trend (start/end):   {close[0]:.2f} -> {close[-1]:.2f}")

def main():
    model_path = Path("models/local-model-b-1h-holdout")
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    
    booster, scaler, fc = load_model(model_path)
    
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
    
    print(f"Holdout period: {holdout_df['timestamp'].iloc[0]} to {holdout_df['timestamp'].iloc[-1]}")
    analyze_holdout(booster, scaler, fc, holdout_df)

if __name__ == "__main__":
    main()
