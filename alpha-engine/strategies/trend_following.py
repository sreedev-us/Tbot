"""
strategies/trend_following.py
=============================
Deterministic Trend-Following Strategy plugin using EMA crossover + ADX filter.

Active only in:
  - 'uptrend'
  - 'downtrend'

Rules:
  - LONG:  EMA20 > EMA50 AND ADX-14 > 25 AND close > EMA20
  - SHORT: EMA20 < EMA50 AND ADX-14 > 25 AND close < EMA20
  - HOLD:  otherwise

Risk parameters:
  - ATR 2.0 TP / 1.0 SL, max hold 24 bars
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from strategies.base_strategy import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    name: str = "trend_following"
    eligible_regimes: list[str] = ["uptrend", "downtrend"]

    def __init__(self, fast_span: int = 20, slow_span: int = 50, adx_threshold: float = 25.0) -> None:
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.adx_threshold = adx_threshold

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        if df.empty or len(df) < max(self.slow_span, 30):
            return 0

        close = df["close"]
        ema_fast = self._ema(close, self.fast_span)
        ema_slow = self._ema(close, self.slow_span)
        adx = self._adx(df, 14)

        cur_close = float(close.iloc[-1])
        cur_fast = float(ema_fast.iloc[-1])
        cur_slow = float(ema_slow.iloc[-1])
        cur_adx = float(adx.iloc[-1])

        if cur_adx > self.adx_threshold:
            if cur_fast > cur_slow and cur_close > cur_fast:
                return 1
            if cur_fast < cur_slow and cur_close < cur_fast:
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
