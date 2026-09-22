"""
strategies/registry.py
======================
Auto-discovery plugin system for trading strategies.

On import, StrategyRegistry scans every .py file in the strategies/ package
and registers any concrete subclass of BaseStrategy it finds.

Usage:
    from strategies.registry import StrategyRegistry

    all_strategies  = StrategyRegistry.load()
    eligible        = StrategyRegistry.eligible_for("uptrend")
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.base_strategy import BaseStrategy

# Resolved lazily to avoid circular imports
_REGISTRY: list["BaseStrategy"] | None = None

_SKIP_MODULES = {"base_strategy", "registry", "__init__"}


def _discover() -> list["BaseStrategy"]:
    from strategies.base_strategy import BaseStrategy as _Base

    import strategies as _pkg

    discovered: list[_Base] = []
    for _, module_name, _ in pkgutil.iter_modules(_pkg.__path__):
        if module_name in _SKIP_MODULES:
            continue
        module = importlib.import_module(f"strategies.{module_name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, _Base)
                and cls is not _Base
                and not inspect.isabstract(cls)
            ):
                discovered.append(cls())
    return discovered


class StrategyRegistry:
    """Lazy-loading registry of all strategy plugins."""

    @classmethod
    def load(cls) -> list["BaseStrategy"]:
        """Return all discovered concrete strategy instances."""
        global _REGISTRY
        if _REGISTRY is None:
            _REGISTRY = _discover()
        return list(_REGISTRY)

    @classmethod
    def eligible_for(cls, regime: str) -> list["BaseStrategy"]:
        """Return strategies eligible for the given regime, in declaration order."""
        return [s for s in cls.load() if s.is_eligible(regime)]

    @classmethod
    def get(cls, name: str) -> "BaseStrategy | None":
        """Return a strategy by its slug name."""
        return next((s for s in cls.load() if s.name == name), None)

    @classmethod
    def reset(cls) -> None:
        """Force re-discovery (useful in tests)."""
        global _REGISTRY
        _REGISTRY = None
