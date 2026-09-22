"""
strategies/base_strategy.py
============================
Abstract base class for all trading strategies in the multi-strategy engine.

Every strategy plugin must:
  1. Subclass BaseStrategy
  2. Set a unique `name` class attribute
  3. Declare `eligible_regimes` (which regime states this strategy is active in)
  4. Implement `signal(df)` → -1 / 0 / 1
  5. Implement `risk_parameters(df)` → dict with tp_pct, sl_pct, max_hold_bars

The plugin system in registry.py auto-discovers any .py file in this package
that contains a concrete subclass of BaseStrategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base for all trading strategy plugins."""

    # ── Class-level declarations (must be set in subclasses) ─────────────────
    name: str = ""                    # unique slug, e.g. "trend_following"
    eligible_regimes: list[str] = []  # e.g. ["uptrend", "downtrend"]

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        """
        Analyse the OHLCV window and return a directional signal.

        Args:
            df: OHLCV DataFrame with columns [timestamp, open, high, low,
                close, volume]. The last row is the current bar.

        Returns:
            1  = BUY
            -1 = SELL
            0  = HOLD / no signal
        """
        ...

    @abstractmethod
    def risk_parameters(self, df: pd.DataFrame) -> dict:
        """
        Return position sizing and exit parameters for the current bar.

        Returns a dict with at minimum:
            tp_pct        (float): take-profit distance as % of entry price
            sl_pct        (float): stop-loss distance as % of entry price
            max_hold_bars (int):   maximum bars to hold if neither barrier hit
        """
        ...

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def is_eligible(self, regime: str) -> bool:
        """Return True if this strategy is designed for the given regime."""
        return regime in self.eligible_regimes

    def __repr__(self) -> str:
        return f"<Strategy:{self.name} regimes={self.eligible_regimes}>"

    # ── ATR helper shared by concrete strategies ──────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Exponentially-smoothed ATR (Wilder)."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        prev  = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) over period."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        eps = 1e-9

        tr0 = high - low
        tr1 = (high - close.shift(1)).abs()
        tr2 = (low - close.shift(1)).abs()
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)

        dm_plus = ((high - high.shift(1)).clip(lower=0)
                   .where(high - high.shift(1) > low.shift(1) - low, 0))
        dm_minus = ((low.shift(1) - low).clip(lower=0)
                    .where(low.shift(1) - low > high - high.shift(1), 0))

        atr = tr.ewm(span=period, adjust=False).mean()
        di_p = 100 * dm_plus.ewm(span=period, adjust=False).mean() / (atr + eps)
        di_m = 100 * dm_minus.ewm(span=period, adjust=False).mean() / (atr + eps)
        dx = 100 * (di_p - di_m).abs() / (di_p + di_m + eps)
        return dx.ewm(span=period, adjust=False).mean()
