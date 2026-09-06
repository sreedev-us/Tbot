#!/usr/bin/env python3
"""
Resample 1-minute OHLCV data to 15-minute and 1-hour timeframes.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INPUT_FILE = Path("data/live/BTC_USDT_1m_live.csv")
OUT_15M = Path("data/live/BTC_USDT_15m_live.csv")
OUT_1H = Path("data/live/BTC_USDT_1h_live.csv")

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples OHLCV DataFrame to a given timeframe."""
    logging.info(f"Resampling to {timeframe}...")
    
    # Ensure timestamp is the index and properly typed
    df_res = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_res["timestamp"]):
        df_res["timestamp"] = pd.to_datetime(df_res["timestamp"], utc=True)
        
    df_res.set_index("timestamp", inplace=True)
    
    # Define aggregation rules
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample and drop NaNs (empty periods)
    resampled = df_res.resample(timeframe).agg(agg_rules).dropna()
    
    # Optional columns to carry over (like sentiment) if they exist
    for col in ['sentiment', 'server_trend', 'server_volatility', 'server_regime']:
        if col in df.columns:
            # For these categorical/sentiment values, taking the 'last' value in the window is safest
            res_col = df_res[col].resample(timeframe).last().dropna()
            resampled[col] = res_col
            
    resampled.reset_index(inplace=True)
    return resampled

def main():
    logging.info(f"Loading {INPUT_FILE}...")
    df_1m = pd.read_csv(INPUT_FILE)
    
    # 15 Minute Resampling
    df_15m = resample_data(df_1m, "15min")
    df_15m.to_csv(OUT_15M, index=False)
    logging.info(f"Saved 15m data: {len(df_15m)} rows to {OUT_15M}")
    
    # 1 Hour Resampling
    df_1h = resample_data(df_1m, "1h")
    df_1h.to_csv(OUT_1H, index=False)
    logging.info(f"Saved 1h data: {len(df_1h)} rows to {OUT_1H}")

if __name__ == "__main__":
    main()
