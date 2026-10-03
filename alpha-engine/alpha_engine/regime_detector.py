"""
alpha_engine/regime_detector.py
===============================
Deterministic priority-ordered market regime classifier.

Regime States:
  - 'high_vol'  : 14-bar ATR > 75th percentile of past 50 bars (Priority 1: suppresses trend/breakout)
  - 'uptrend'   : ADX-14 > 25, EMA-20 > EMA-50, close > EMA-20
  - 'downtrend' : ADX-14 > 25, EMA-20 < EMA-50, close < EMA-20
  - 'range'     : ADX-14 < 20
  - 'low_vol'   : 14-bar ATR < 25th percentile of past 50 bars
  - 'unknown'   : Intermediate state not matching above rules
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class MarketState:
    trend: str       # 'uptrend', 'downtrend', 'range', 'neutral'
    volatility: str  # 'high_vol', 'normal_vol', 'low_vol'


class RegimeDetector:
    """Classifies the market into a multi-dimensional MarketState (trend + volatility)."""

    def __init__(
        self,
        atr_period: int = 14,
        adx_period: int = 14,
        vol_window: int = 50,
        trend_fast: int = 20,
        trend_slow: int = 50,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        vol_high_pct: float = 0.75,
        vol_low_pct: float = 0.25,
    ) -> None:
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.vol_window = vol_window
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct

    @staticmethod
    def _calc_indicators(
        df: pd.DataFrame,
        atr_period: int = 14,
        adx_period: int = 14,
        trend_fast: int = 20,
        trend_slow: int = 50,
        vol_window: int = 50,
        vol_high_pct: float = 0.75,
        vol_low_pct: float = 0.25,
    ) -> dict[str, pd.Series]:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        eps = 1e-9

        # 1. True Range & ATR
        tr0 = high - low
        tr1 = (high - close.shift(1)).abs()
        tr2 = (low - close.shift(1)).abs()
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()

        # ATR Percentiles over rolling vol_window
        atr_q75 = atr.rolling(vol_window, min_periods=min(10, vol_window)).quantile(vol_high_pct)
        atr_q25 = atr.rolling(vol_window, min_periods=min(10, vol_window)).quantile(vol_low_pct)

        # 2. EMAs
        ema_fast = close.ewm(span=trend_fast, adjust=False).mean()
        ema_slow = close.ewm(span=trend_slow, adjust=False).mean()

        # 3. ADX
        dm_plus = ((high - high.shift(1)).clip(lower=0)
                   .where(high - high.shift(1) > low.shift(1) - low, 0))
        dm_minus = ((low.shift(1) - low).clip(lower=0)
                    .where(low.shift(1) - low > high - high.shift(1), 0))

        di_p = 100 * dm_plus.ewm(span=adx_period, adjust=False).mean() / (atr + eps)
        di_m = 100 * dm_minus.ewm(span=adx_period, adjust=False).mean() / (atr + eps)
        dx = 100 * (di_p - di_m).abs() / (di_p + di_m + eps)
        adx = dx.ewm(span=adx_period, adjust=False).mean()

        return {
            "atr": atr,
            "atr_q75": atr_q75,
            "atr_q25": atr_q25,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "adx": adx,
            "close": close,
        }

    def detect(self, df: pd.DataFrame) -> MarketState:
        """Detect current regime for the latest bar in df."""
        if df.empty or len(df) < 20:
            return MarketState(trend="neutral", volatility="normal_vol")

        ind = self._calc_indicators(
            df,
            atr_period=self.atr_period,
            adx_period=self.adx_period,
            trend_fast=self.trend_fast,
            trend_slow=self.trend_slow,
            vol_window=self.vol_window,
            vol_high_pct=self.vol_high_pct,
            vol_low_pct=self.vol_low_pct,
        )

        cur_atr = float(ind["atr"].iloc[-1])
        cur_q75 = float(ind["atr_q75"].iloc[-1])
        cur_q25 = float(ind["atr_q25"].iloc[-1])
        cur_adx = float(ind["adx"].iloc[-1])
        cur_close = float(ind["close"].iloc[-1])
        cur_fast = float(ind["ema_fast"].iloc[-1])
        cur_slow = float(ind["ema_slow"].iloc[-1])

        # Trend dimension
        trend = "neutral"
        if cur_adx > self.adx_trend_threshold:
            if cur_fast > cur_slow and cur_close > cur_fast:
                trend = "uptrend"
            elif cur_fast < cur_slow and cur_close < cur_fast:
                trend = "downtrend"
        elif cur_adx < self.adx_range_threshold:
            trend = "range"

        # Volatility dimension
        volatility = "normal_vol"
        if pd.notna(cur_q75) and cur_atr > cur_q75:
            volatility = "high_vol"
        elif pd.notna(cur_q25) and cur_atr < cur_q25:
            volatility = "low_vol"

        return MarketState(trend=trend, volatility=volatility)

    def detect_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized classification across all bars in df. Returns DataFrame with 'trend' and 'volatility'."""
        if df.empty:
            return pd.DataFrame(columns=["trend", "volatility"])

        ind = self._calc_indicators(
            df,
            atr_period=self.atr_period,
            adx_period=self.adx_period,
            trend_fast=self.trend_fast,
            trend_slow=self.trend_slow,
            vol_window=self.vol_window,
            vol_high_pct=self.vol_high_pct,
            vol_low_pct=self.vol_low_pct,
        )

        n = len(df)
        trend = np.full(n, "neutral", dtype=object)
        vol = np.full(n, "normal_vol", dtype=object)

        # Trend masks
        is_uptrend = (ind["adx"] > self.adx_trend_threshold) & (ind["ema_fast"] > ind["ema_slow"]) & (ind["close"] > ind["ema_fast"])
        is_downtrend = (ind["adx"] > self.adx_trend_threshold) & (ind["ema_fast"] < ind["ema_slow"]) & (ind["close"] < ind["ema_fast"])
        is_range = ind["adx"] < self.adx_range_threshold

        trend[is_uptrend] = "uptrend"
        trend[is_downtrend] = "downtrend"
        trend[is_range] = "range"

        # Volatility masks
        is_high_vol = ind["atr"] > ind["atr_q75"]
        is_low_vol = ind["atr"] < ind["atr_q25"]

        vol[is_high_vol] = "high_vol"
        vol[is_low_vol] = "low_vol"

        return pd.DataFrame({"trend": trend, "volatility": vol}, index=df.index)
