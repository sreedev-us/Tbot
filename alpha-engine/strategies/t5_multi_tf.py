from __future__ import annotations
from typing import Literal
import pandas as pd
from strategies.base_strategy import BaseStrategy

class MultiTFStrategy(BaseStrategy):
    name: str = "t5_multi_tf"

    def is_eligible(self, state: 'MarketState') -> bool:
        return state.trend in ["uptrend", "downtrend"]

    def __init__(self, fast_span: int = 20, slow_span: int = 50) -> None:
        self.fast_span = fast_span
        self.slow_span = slow_span

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        if df.empty or len(df) < max(self.slow_span * 4, 30):
            return 0

        # 1. 1h Timeframe
        close_1h = df["close"]
        ema_fast_1h = self._ema(close_1h, self.fast_span)
        ema_slow_1h = self._ema(close_1h, self.slow_span)
        
        cur_fast_1h = float(ema_fast_1h.iloc[-1])
        cur_slow_1h = float(ema_slow_1h.iloc[-1])

        # 2. 4h Timeframe (strictly completed candles)
        # We set timestamp as index for resampling
        df_dt = df.set_index("timestamp")
        
        # Resample 1h to 4h
        df_4h = df_dt.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        # Check if the last 4h candle is fully completed.
        # A 4h candle is completed if the latest 1h candle in our df is the LAST hour of that 4h block.
        # Since 4h blocks start at 00:00, 04:00, etc., the last hour of a block is 03:00, 07:00, etc.
        # Alternatively, we can just count the number of 1h candles in the last 4h block.
        # If it's less than 4 (and we assume no missing data, which is true for our live CSV), it's incomplete.
        # Safer: just drop the last 4h candle if the current 1h time doesn't close it.
        # A 1h candle at 03:00 (which spans 03:00-04:00) closes exactly at 04:00.
        # We can just use the standard pandas logic: if df_dt.index[-1].hour % 4 != 3: drop last
        
        last_hour = df_dt.index[-1].hour
        if last_hour % 4 != 3:
            df_4h = df_4h.iloc[:-1]
            
        if len(df_4h) < self.slow_span:
            return 0
            
        close_4h = df_4h["close"]
        ema_fast_4h = self._ema(close_4h, self.fast_span)
        ema_slow_4h = self._ema(close_4h, self.slow_span)

        cur_fast_4h = float(ema_fast_4h.iloc[-1])
        cur_slow_4h = float(ema_slow_4h.iloc[-1])

        # Signal Logic
        if cur_fast_1h > cur_slow_1h and cur_fast_4h > cur_slow_4h:
            return 1
        if cur_fast_1h < cur_slow_1h and cur_fast_4h < cur_slow_4h:
            return -1

        return 0

    def risk_parameters(self, df: pd.DataFrame) -> dict:
        atr = float(self._atr(df, period=14).iloc[-1])
        close = float(df["close"].iloc[-1])
        tp_pct = (2.0 * atr / close) * 100.0 if close > 0 else 2.0
        sl_pct = (1.0 * atr / close) * 100.0 if close > 0 else 1.0

        return {
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "max_hold_bars": 24,
            "atr_tp_mult": 2.0,
            "atr_sl_mult": 1.0,
        }
