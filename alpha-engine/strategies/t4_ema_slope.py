from __future__ import annotations
from typing import Literal
import pandas as pd
from strategies.base_strategy import BaseStrategy

class EMASlopeStrategy(BaseStrategy):
    name: str = "t4_ema_slope"

    def is_eligible(self, state: 'MarketState') -> bool:
        return state.trend in ["uptrend", "downtrend"]

    def __init__(self, span: int = 50) -> None:
        self.span = span

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        if df.empty or len(df) < max(self.span, 10):
            return 0

        close = df["close"]
        ema = self._ema(close, self.span)

        # EMA is strictly increasing over the last 3 bars (t, t-1, t-2)
        ema_t = float(ema.iloc[-1])
        ema_t1 = float(ema.iloc[-2])
        ema_t2 = float(ema.iloc[-3])

        if ema_t > ema_t1 > ema_t2:
            return 1
        if ema_t < ema_t1 < ema_t2:
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
