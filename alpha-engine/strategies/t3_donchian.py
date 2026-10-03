from __future__ import annotations
from typing import Literal
import pandas as pd
from strategies.base_strategy import BaseStrategy

class DonchianStrategy(BaseStrategy):
    name: str = "t3_donchian"

    def is_eligible(self, state: 'MarketState') -> bool:
        return state.trend in ["uptrend", "downtrend"]

    def __init__(self, window: int = 50) -> None:
        self.window = window

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        if df.empty or len(df) < self.window + 2:
            return 0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Prior window highs/lows (excluding current bar)
        prior_highs = high.shift(1).rolling(self.window).max()
        prior_lows = low.shift(1).rolling(self.window).min()

        cur_close = float(close.iloc[-1])
        ref_high = float(prior_highs.iloc[-1])
        ref_low = float(prior_lows.iloc[-1])

        if cur_close > ref_high:
            return 1
        if cur_close < ref_low:
            return -1

        return 0

    def risk_parameters(self, df: pd.DataFrame) -> dict:
        atr = float(self._atr(df, period=14).iloc[-1])
        close = float(df["close"].iloc[-1])
        tp_pct = (3.0 * atr / close) * 100.0 if close > 0 else 3.0
        sl_pct = (1.5 * atr / close) * 100.0 if close > 0 else 1.5

        return {
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "max_hold_bars": 36,
            "atr_tp_mult": 3.0,
            "atr_sl_mult": 1.5,
        }
