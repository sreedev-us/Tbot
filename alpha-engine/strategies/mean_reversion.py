"""
strategies/mean_reversion.py
============================
Mean Reversion Strategy plugin wrapping frozen Model B (XGBoost).

Active only in:
  - 'range' (low trend strength, ADX < 20)
  - 'low_vol' (low volatility, quiet consolidation)

Parameters (frozen from Phase 3 walk-forward validation):
  - Model: models/local-model-b-1h-holdout (fallback: models/local-model-b-1h)
  - Confidence threshold: 0.85
  - Risk geometry: ATR 2.0 TP / 1.0 SL, max hold 24 bars
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import xgboost as xgb

from alpha_engine.training_pipeline import FeatureEngineer, TrainingDataConfig
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    name: str = "mean_reversion"
    eligible_regimes: list[str] = ["range", "low_vol"]

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.booster: Optional[xgb.Booster] = None
        self.scaler: Optional[Any] = None
        self.feature_columns: Optional[list[str]] = None
        self.feature_engineer = FeatureEngineer(TrainingDataConfig())

        # Resolve model directory
        if model_path is None:
            base_dir = Path(__file__).resolve().parents[1] / "models"
            cand1 = base_dir / "local-model-b-1h-holdout"
            cand2 = base_dir / "local-model-b-1h"
            model_dir = cand1 if cand1.exists() else cand2
        else:
            model_dir = Path(model_path)

        if model_dir.exists():
            self._load_model(model_dir)
        else:
            logger.warning("MeanReversionStrategy: model directory not found at %s", model_dir)

    def _load_model(self, model_dir: Path) -> None:
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

            logger.info(
                "MeanReversionStrategy loaded model from %s with %d features",
                model_dir.name,
                len(self.feature_columns or []),
            )
        except Exception as exc:
            logger.warning("Failed to load Model B in MeanReversionStrategy: %s", exc)

    def signal(self, df: pd.DataFrame) -> Literal[-1, 0, 1]:
        if (
            df.empty
            or len(df) < 20
            or self.booster is None
            or self.scaler is None
            or not self.feature_columns
        ):
            return 0

        try:
            # Fast path: if features are already computed in the DataFrame
            if all(col in df.columns for col in self.feature_columns):
                latest_features = df[self.feature_columns].iloc[[-1]]
            else:
                features_df = self.feature_engineer.engineer_features(df, ablation_level="B")
                if features_df.empty:
                    return 0
                latest_features = features_df[self.feature_columns].iloc[[-1]]

            scaled_x = self.scaler.transform(latest_features)
            if hasattr(self.booster, "predict_proba"):
                probs = self.booster.predict_proba(scaled_x)[0]
            else:
                probs = self.booster.predict(xgb.DMatrix(scaled_x))[0]

            p_sell = float(probs[0])
            p_buy = float(probs[2])

            if p_buy > p_sell and p_buy >= self.confidence_threshold:
                return 1
            if p_sell > p_buy and p_sell >= self.confidence_threshold:
                return -1
            return 0
        except Exception as exc:
            logger.debug("MeanReversionStrategy inference error: %s", exc)
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
