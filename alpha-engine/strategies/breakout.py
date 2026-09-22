"""
strategies/breakout.py
======================
Deterministic N-Bar High/Low Breakout Strategy plugin with volume confirmation.

Active in:
  - 'high_vol'
  - 'uptrend'
  - 'downtrend'

Rules:
  - LONG:  close > rolling_high(20 bars) AND volume > vol_ma_20 * 1.5
  - SHORT: close < rolling_low(20 bars)  AND volume > vol_ma_20 * 1.5
  - HOLD:  otherwise

Risk parameters:
  - ATR 3.0 TP / 1.5 SL (wider barrier for breakout momentum), max hold 36 bars
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from strategies.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    name: str = "breakout"
    eligible_regimes: list[str] = ["high_vol", "uptrend", "downtrend"]

    def __init__(self, window: int = 20, vol_multiplier: float = 1.5) -> None:
        self.window = window
        self.vol_multiplier = vol_multiplier

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        if df.empty or len(df) < self.window + 2:
            return 0

        high = df["high"]
        low = df["low"]
        close = df["close"]
        vol = df["volume"]

        # Prior window highs/lows and average volume (excluding current bar)
        prior_highs = high.shift(1).rolling(self.window).max()
        prior_lows = low.shift(1).rolling(self.window).min()
        prior_vol_ma = vol.shift(1).rolling(self.window).mean()

        cur_close = float(close.iloc[-1])
        cur_vol = float(vol.iloc[-1])
        ref_high = float(prior_highs.iloc[-1])
        ref_low = float(prior_lows.iloc[-1])
        avg_vol = float(prior_vol_ma.iloc[-1]) if float(prior_vol_ma.iloc[-1]) > 0 else 1.0

        vol_confirmed = cur_vol >= avg_vol * self.vol_multiplier

        if vol_confirmed:
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
