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
import logging

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.registry import StrategyRegistry
from alpha_engine.regime_detector import MarketState

logger = logging.getLogger(__name__)

STRATEGY_PRIORITY = {
    "uptrend": ["trend_following", "breakout"],
    "downtrend": ["trend_following", "breakout"],
    "range": ["mean_reversion"],
    "neutral": [],
}

@dataclass(slots=True)
class SelectionResult:
    strategy_name: str
    signal: Literal[-1, 0, 1]
    risk_params: dict
    state: MarketState


class StrategySelector:
    """Deterministic strategy selector based on explicitly configured priority."""

    def __init__(self, registry: Optional[StrategyRegistry] = None) -> None:
        self.registry = registry or StrategyRegistry

    def get_eligible(self, state: MarketState, registry: StrategyRegistry) -> list[BaseStrategy]:
        """Return explicitly prioritized strategies that are eligible for the state."""
        names = STRATEGY_PRIORITY.get(state.trend, [])
        eligible = []
        for name in names:
            strategy = registry.get(name)
            if strategy and strategy.is_eligible(state):
                eligible.append(strategy)
        return eligible

    def evaluate(
        self,
        df: pd.DataFrame,
        state: MarketState,
    ) -> Optional[SelectionResult]:
        """
        Evaluate eligible strategies in order.
        Returns the first strategy that issues a BUY (1) or SELL (-1) signal.
        """
        eligible = self.get_eligible(state, self.registry)
        
        logger.info("State: %s / %s | Candidates: %s", state.trend, state.volatility, [s.name for s in eligible])
        
        if not eligible:
            return None

        for strategy in eligible:
            sig = strategy.signal(df)
            if sig != 0:
                logger.info("Selected: %s | Signal: %s", strategy.name, "BUY" if sig == 1 else "SELL")
                params = strategy.risk_parameters(df)
                return SelectionResult(
                    strategy_name=strategy.name,
                    signal=sig,
                    risk_params=params,
                    state=state,
                )

        return None
