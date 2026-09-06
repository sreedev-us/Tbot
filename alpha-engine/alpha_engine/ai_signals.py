"""AI-based signal generation for paper trading with local in-memory model inference."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
import requests
import xgboost as xgb

from alpha_engine.signals import RawSignal
from alpha_engine.training_pipeline import FeatureEngineer, TrainingDataConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AISignalContext:
    """Context for AI signal generation."""
    server_ai_response: dict[str, Any]
    local_ai_decision: dict[str, Any]
    market_price: Decimal
    timestamp: datetime


class AISignalEvaluator:
    """
    Evaluates trading signals using either direct in-engine model inference
    (XGBoost booster + StandardScaler) or server-side analysis fallback.
    """

    def __init__(
        self,
        backend_url: str,
        model_version: str = "local-gpu-benchmark-v1.0.0",
        models_dir: Optional[str | Path] = None,
        confidence_threshold: float = 0.40,
    ):
        self.backend_url = backend_url
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold

        self.booster: Optional[xgb.Booster] = None
        self.scaler: Optional[Any] = None
        self.feature_columns: Optional[list[str]] = None
        self.feature_engineer: Optional[FeatureEngineer] = None

        base_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parents[1] / "models"
        model_dir = base_dir / model_version
        if model_dir.exists():
            self._load_local_model(model_dir)

    def _load_local_model(self, model_dir: Path) -> None:
        """Load trained booster, scaler, and feature schema from disk."""
        try:
            meta_file = model_dir / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                self.feature_columns = meta.get("feature_columns")

            scaler_file = model_dir / "scaler.pkl"
            if scaler_file.exists():
                with scaler_file.open("rb") as f:
                    self.scaler = pickle.load(f)

            json_model = model_dir / "model.json"
            pkl_model = model_dir / "model.pkl"
            if json_model.exists():
                booster = xgb.Booster()
                booster.load_model(str(json_model))
                self.booster = booster
            elif pkl_model.exists():
                with pkl_model.open("rb") as f:
                    self.booster = pickle.load(f)

            self.feature_engineer = FeatureEngineer(TrainingDataConfig())
            logger.info(
                "Loaded local AI model: %s (%d features)",
                self.model_version,
                len(self.feature_columns or []),
            )
        except Exception as exc:
            logger.warning("Failed to load local AI model %s: %s", self.model_version, exc)

    def get_server_ai_analysis(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
    ) -> Optional[dict[str, Any]]:
        """Get server-side AI analysis from the backend."""
        try:
            if df.empty:
                return None

            latest = df.iloc[-1]
            payload = {
                "asset": symbol,
                "currentPrice": float(latest["close"]),
                "recentPrices": df["close"].tail(50).tolist(),
                "timestamps": [int(pd.Timestamp(t).timestamp() * 1000) for t in df["timestamp"].tail(50)],
                "volume": float(latest["volume"]),
                "sentiment": 0.0,
                "sentimentConfidence": 0.0,
            }

            response = requests.post(
                f"{self.backend_url}/api/v1/analysis",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.debug("Failed to get server AI analysis: %s", exc)
            return None

    def get_local_ai_decision(
        self,
        server_analysis: dict[str, Any],
        symbol: str,
        exchange: str,
        market_price: Decimal,
    ) -> Optional[dict[str, Any]]:
        """Get local AI decision from the backend."""
        try:
            payload = {
                "asset": symbol,
                "exchange": exchange,
                "marketPrice": str(market_price),
                "serverAnalysis": server_analysis,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

            response = requests.post(
                f"{self.backend_url}/api/v1/decision",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.debug("Failed to get local AI decision from backend: %s", exc)
            return None

    def evaluate_ai_signal(
        self,
        df: pd.DataFrame,
        exchange: str,
        symbol: str,
        order_notional: Decimal,
        server_analysis: dict[str, Any],
        strategy_name: str = "ai-benchmark",
        stop_loss_pct: Decimal = Decimal("1.5"),
        take_profit_pct: Decimal = Decimal("3.0"),
    ) -> Optional[RawSignal]:
        """
        Evaluate a signal using in-engine local model inference, 
        incorporating Server AI context.
        """
        if df.empty or len(df) < 20:
            return None

        market_price = Decimal(str(round(float(df["close"].iloc[-1]), 8)))

        # ---------------------------------------------------------------------
        # 1. Primary: Direct In-Memory Local Model Inference
        # ---------------------------------------------------------------------
        if (
            self.booster is not None
            and self.scaler is not None
            and self.feature_columns is not None
            and self.feature_engineer is not None
        ):
            try:
                features_df = self.feature_engineer.engineer_features(df)
                if features_df.empty:
                    return None

                latest_features = features_df[self.feature_columns].iloc[[-1]]
                scaled_x = self.scaler.transform(latest_features)

                if hasattr(self.booster, "predict_proba"):
                    probs = self.booster.predict_proba(scaled_x)[0]
                else:
                    probs = self.booster.predict(xgb.DMatrix(scaled_x))[0]

                # Class mapping: 0: SELL, 1: HOLD, 2: BUY
                p_sell = float(probs[0])
                p_hold = float(probs[1])
                p_buy = float(probs[2])

                if p_buy > p_sell and p_buy >= self.confidence_threshold:
                    decision = "BUY"
                    confidence = Decimal(str(round(p_buy, 4)))
                elif p_sell > p_buy and p_sell >= self.confidence_threshold:
                    decision = "SELL"
                    confidence = Decimal(str(round(p_sell, 4)))
                else:
                    logger.info(
                        "Local AI decision: HOLD (p_buy=%.3f, p_sell=%.3f, p_hold=%.3f)",
                        p_buy, p_sell, p_hold,
                    )
                    return None

                # Adjust spreads based on confidence
                confidence_factor = Decimal("1.0") + (confidence - Decimal("0.5")) * Decimal("0.5")
                stop_loss_spread = stop_loss_pct / confidence_factor
                take_profit_spread = take_profit_pct / confidence_factor

                if decision == "BUY":
                    stop_loss_price = market_price * (Decimal("1") - stop_loss_spread / Decimal("100"))
                    take_profit_price = market_price * (Decimal("1") + take_profit_spread / Decimal("100"))
                else:
                    stop_loss_price = market_price * (Decimal("1") + stop_loss_spread / Decimal("100"))
                    take_profit_price = market_price * (Decimal("1") - take_profit_spread / Decimal("100"))

                signal = RawSignal(
                    signal_id=str(uuid4()),
                    correlation_id=str(uuid4()),
                    asset=symbol,
                    exchange=exchange,
                    action=decision,
                    confidence=confidence,
                    requested_notional=order_notional,
                    strategy_name=strategy_name,
                    generated_at=datetime.now(UTC),
                    market_price=market_price,
                    stop_loss_price=Decimal(str(round(float(stop_loss_price), 8))),
                    take_profit_price=Decimal(str(round(float(take_profit_price), 8))),
                )

                logger.info(
                    "Local AI Signal: %s at %s (conf: %s, SL: %s, TP: %s)",
                    decision, market_price, confidence, stop_loss_price, take_profit_price,
                )
                return signal
            except Exception as exc:
                logger.warning("Error during local AI inference: %s", exc)
                return None

