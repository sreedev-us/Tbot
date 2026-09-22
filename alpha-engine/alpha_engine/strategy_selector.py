"""
alpha_engine/strategy_selector.py
=================================
Deterministic regime-to-strategy selector.

Dispatches market data to eligible strategies based on detected regime.
Implements serialization priority:
  The first eligible strategy that generates a non-zero signal (BUY or SELL)
  is chosen. If no eligible strategy signals, returns None / 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.registry import StrategyRegistry


@dataclass(slots=True)
class SelectionResult:
    strategy_name: str
    signal: Literal[-1, 0, 1]
    risk_params: dict
    regime: str


class StrategySelector:
    """Deterministic strategy selector based on active market regime."""

    def __init__(self, registry: Optional[StrategyRegistry] = None) -> None:
        self.registry = registry or StrategyRegistry

    def select_strategies(self, regime: str) -> list[BaseStrategy]:
        """Return all strategies eligible for the current regime."""
        return self.registry.eligible_for(regime)

    def evaluate(
        self,
        df: pd.DataFrame,
        regime: str,
    ) -> Optional[SelectionResult]:
        """
        Evaluate eligible strategies in order.
        Returns the first strategy that issues a BUY (1) or SELL (-1) signal.
        """
        eligible = self.select_strategies(regime)
        if not eligible:
            return None

        for strategy in eligible:
            sig = strategy.signal(df)
            if sig != 0:
                params = strategy.risk_parameters(df)
                return SelectionResult(
                    strategy_name=strategy.name,
                    signal=sig,
                    risk_params=params,
                    regime=regime,
                )

        return None
