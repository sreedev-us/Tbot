import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_engine.regime_detector import RegimeDetector
from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator
from strategies.mean_reversion import MeanReversionStrategy

logging.basicConfig(level=logging.WARNING)

def calculate_mae_mfe(df: pd.DataFrame, entry_idx: int, max_hold: int, signal: int) -> tuple[float, float]:
    """Calculate Maximum Adverse Excursion and Maximum Favorable Excursion over max_hold bars."""
    entry_price = df["close"].iloc[entry_idx]
    window = df.iloc[entry_idx + 1 : entry_idx + 1 + max_hold]
    if window.empty:
        return 0.0, 0.0
        
    highs = window["high"].values
    lows = window["low"].values
    
    if signal == 1:
        max_high = np.max(highs)
        min_low = np.min(lows)
        mfe = (max_high - entry_price) / entry_price * 100.0
        mae = (min_low - entry_price) / entry_price * 100.0
    else:
        min_low = np.min(lows)
        max_high = np.max(highs)
        mfe = (entry_price - min_low) / entry_price * 100.0
        mae = (entry_price - max_high) / entry_price * 100.0
        
    return mae, mfe

def run_decomposition(df: pd.DataFrame, state_df: pd.DataFrame, strategy: MeanReversionStrategy, horizons: list[int], max_hold: int = 24) -> pd.DataFrame:
    close = df["close"].values
    n = len(df)
    results = []
    
    # We need to evaluate the strategy unconditionally for every bar
    for i in range(55, n - max(horizons) - 1):
        window_df = df.iloc[: i + 1]
        
        # Check if features are already computed in the DataFrame
        if all(col in window_df.columns for col in strategy.feature_columns):
            latest_features = window_df[strategy.feature_columns].iloc[[-1]]
        else:
            features_df = strategy.feature_engineer.engineer_features(window_df, ablation_level="B")
            if features_df.empty:
                continue
            latest_features = features_df[strategy.feature_columns].iloc[[-1]]

        scaled_x = strategy.scaler.transform(latest_features)
        import xgboost as xgb
        if hasattr(strategy.booster, "predict_proba"):
            probs = strategy.booster.predict_proba(scaled_x)[0]
        else:
            probs = strategy.booster.predict(xgb.DMatrix(scaled_x))[0]

        p_sell = float(probs[0])
        p_buy = float(probs[2])
        
        sig = 0
        conf = 0.0
        if p_buy > p_sell and p_buy >= strategy.confidence_threshold:
            sig = 1
            conf = p_buy
        elif p_sell > p_buy and p_sell >= strategy.confidence_threshold:
            sig = -1
            conf = p_sell
            
        if sig != 0:
            trend = state_df["trend"].iloc[i]
            vol_state = state_df["volatility"].iloc[i]
            entry_price = close[i]
            
            # Extract anatomy metrics
            # 1. Distance from mean (using close / EMA20 - 1)
            ema20 = window_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            dist_mean = (entry_price / ema20 - 1) * 100.0
            
            # 2. ATR
            tr = np.maximum(window_df["high"] - window_df["low"], 
                 np.maximum(abs(window_df["high"] - window_df["close"].shift(1)), 
                            abs(window_df["low"] - window_df["close"].shift(1))))
            atr14 = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
            atr_pct = atr14 / entry_price * 100.0
            
            # 3. Volatility percentile (using ATR 14 over last 200 bars)
            atr_history = tr.ewm(alpha=1/14, adjust=False).mean().tail(200)
            vol_percentile = stats.percentileofscore(atr_history, atr14)
            
            # 4. Trend strength (ADX)
            # Using simple ADX calculation approximation from feature eng
            adx = float(window_df["adx_14"].iloc[-1]) if "adx_14" in window_df.columns else 0.0
            
            record = {
                "bar_idx": i,
                "timestamp": window_df["timestamp"].iloc[-1],
                "signal": sig,
                "trend": trend,
                "volatility_state": vol_state,
                "confidence": conf,
                "dist_from_mean_pct": dist_mean,
                "atr_pct": atr_pct,
                "vol_percentile": vol_percentile,
                "adx": adx,
            }
            
            # Forward returns
            for h in horizons:
                exit_price = close[i + h]
                fwd_ret = sig * (exit_price - entry_price) / entry_price * 100.0
                record[f"ret_{h}h"] = fwd_ret
                
            # MAE / MFE
            mae, mfe = calculate_mae_mfe(df, i, max_hold, sig)
            record["mae"] = mae  # Adverse, so generally negative
            record["mfe"] = mfe  # Favorable, generally positive
            
            results.append(record)
            
    return pd.DataFrame(results)

def main():
    data_path = Path("data/live/BTC_USDT_1h_live.csv")
    print(f"Loading data from {data_path}...")
    raw_df = pd.read_csv(data_path)

    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    full_df = gen.feature_engineer.engineer_features(raw_df, ablation_level="B")
    full_df = full_df.assign(
        timestamp=pd.to_datetime(full_df["timestamp"], utc=True)
    ).sort_values("timestamp").reset_index(drop=True)

    print(f"Dataset: {len(full_df)} bars ({full_df['timestamp'].iloc[0].date()} to {full_df['timestamp'].iloc[-1].date()})")

    detector = RegimeDetector()
    state_df = detector.detect_series(full_df)

    strategy = MeanReversionStrategy()
    
    horizons = [1, 2, 4, 6, 12, 24, 48]
    
    print("\nRunning Model B Edge Decomposition...")
    res_df = run_decomposition(full_df, state_df, strategy, horizons, max_hold=24)
    
    if res_df.empty:
        print("No signals found.")
        return
        
    print("\n" + "=" * 100)
    print("  PHASE 7: MODEL B EDGE DECOMPOSITION")
    print("=" * 100)
    
    print(f"\nTotal Signals Evaluated: {len(res_df)} (Longs: {sum(res_df['signal'] == 1)}, Shorts: {sum(res_df['signal'] == -1)})")
    
    # 1. Forward Return Surface
    print("\n--- 1. FORWARD RETURN SURFACE ---")
    ret_cols = [f"ret_{h}h" for h in horizons]
    surf = res_df.groupby("trend")[ret_cols].mean()
    surf_cost = surf - 0.20
    
    print("\nRaw Expectancy by Trend State:")
    print(surf.map(lambda x: f"{x:+.2f}%").to_string())
    print("\nCost-Adjusted Expectancy (-0.20%) by Trend State:")
    print(surf_cost.map(lambda x: f"{x:+.2f}%").to_string())
    
    print("\nCost-Adjusted Expectancy by Direction (All States):")
    dir_surf = res_df.groupby("signal")[ret_cols].mean() - 0.20
    dir_surf.index = dir_surf.index.map({1: "LONG", -1: "SHORT"})
    print(dir_surf.map(lambda x: f"{x:+.2f}%").to_string())
    
    # 2. MAE / MFE Analysis
    print("\n--- 2. MAE / MFE ANALYSIS (24h Hold) ---")
    print("MFE (Maximum Favorable Excursion) measures the peak profit before exit.")
    print("MAE (Maximum Adverse Excursion) measures the peak drawdown before exit.")
    
    mae_mfe = res_df.groupby("trend")[["mae", "mfe"]].mean()
    print("\nAverage Excursions by Trend State:")
    print(mae_mfe.map(lambda x: f"{x:+.2f}%").to_string())
    
    mae_mfe_dir = res_df.groupby("signal")[["mae", "mfe"]].mean()
    mae_mfe_dir.index = mae_mfe_dir.index.map({1: "LONG", -1: "SHORT"})
    print("\nAverage Excursions by Direction:")
    print(mae_mfe_dir.map(lambda x: f"{x:+.2f}%").to_string())
    
    # 3. Confidence Calibration
    print("\n--- 3. CONFIDENCE CALIBRATION ---")
    # Bin confidence into quintiles
    res_df["conf_bin"] = pd.qcut(res_df["confidence"], q=5, labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"])
    
    conf_cal = res_df.groupby("conf_bin", observed=True).agg(
        count=("signal", "count"),
        avg_conf=("confidence", "mean"),
        ret_4h=("ret_4h", "mean"),
        ret_12h=("ret_12h", "mean"),
        ret_24h=("ret_24h", "mean")
    )
    print(conf_cal.to_string(float_format="{:.2f}".format))
    
    # 4. Signal Anatomy (Why was it generated?)
    print("\n--- 4. SIGNAL ANATOMY ---")
    print("Average characteristics of the market at the time of the signal.")
    
    anat = res_df.groupby("signal").agg(
        dist_mean_pct=("dist_from_mean_pct", "mean"),
        atr_pct=("atr_pct", "mean"),
        vol_percentile=("vol_percentile", "mean"),
        adx=("adx", "mean")
    )
    anat.index = anat.index.map({1: "LONG", -1: "SHORT"})
    print(anat.to_string(float_format="{:.2f}".format))
    
    # Anatomy by Trend
    print("\nAnatomy by Trend State:")
    anat_trend = res_df.groupby("trend").agg(
        dist_mean_pct=("dist_from_mean_pct", "mean"),
        atr_pct=("atr_pct", "mean"),
        vol_percentile=("vol_percentile", "mean"),
        adx=("adx", "mean")
    )
    print(anat_trend.to_string(float_format="{:.2f}".format))

if __name__ == "__main__":
    main()
