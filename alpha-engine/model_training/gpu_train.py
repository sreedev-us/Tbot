from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpha_engine.training_pipeline import TrainingDataConfig, TrainingDataGenerator


LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


@dataclass(slots=True)
class TrainingSummary:
    version: str
    preset: str
    rows_raw: int
    rows_training: int
    feature_count: int
    train_rows: int
    validation_rows: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    label_distribution: dict[str, int]
    prediction_distribution: dict[str, int]
    model_hash: str
    created_at: str


PRESETS = {
    "smoke": {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 128,
    },
    "balanced": {
        "max_depth": 6,
        "learning_rate": 0.035,
        "n_estimators": 900,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "max_bin": 256,
    },
    "benchmark": {
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 1000,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "max_bin": 256,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "large": {
        "max_depth": 8,
        "learning_rate": 0.02,
        "n_estimators": 2500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 512,
    },
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_frame(
    csv_path: Path,
    target_return: float,
    stop_loss: float,
    lookahead: int,
    use_atr: bool = True,
    atr_tp: float = 2.0,
    atr_sl: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = pd.read_csv(csv_path)
    generator = TrainingDataGenerator(
        TrainingDataConfig(
            output_dir=str(csv_path.parent),
            lookahead_window=lookahead,
            use_atr_barriers=use_atr,
            atr_multiplier_tp=atr_tp,
            atr_multiplier_sl=atr_sl,
            target_return_pct=target_return,
            stop_loss_pct=stop_loss,
        )
    )
    training_data = generator.generate_training_data(raw)
    feature_cols = generator.get_feature_columns(training_data)
    return raw, training_data, feature_cols


def _chronological_split(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(frame) * (1 - validation_fraction))
    split_index = max(1, min(split_index, len(frame) - 1))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def train(args: argparse.Namespace) -> TrainingSummary:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    raw, training_frame, feature_columns = _load_training_frame(
        csv_path,
        target_return=args.target_return,
        stop_loss=args.stop_loss,
        lookahead=args.lookahead,
        use_atr=args.use_atr,
        atr_tp=args.atr_tp,
        atr_sl=args.atr_sl,
    )
    if len(training_frame) < args.min_rows:
        raise ValueError(
            f"Only {len(training_frame)} training rows are available. "
            f"Need at least {args.min_rows}. Collect more candles first."
        )

    train_frame, validation_frame = _chronological_split(training_frame, args.validation_fraction)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_frame[feature_columns])
    x_validation = scaler.transform(validation_frame[feature_columns])
    y_train = train_frame["target"].astype(int).map(LABEL_TO_CLASS)
    y_validation = validation_frame["target"].astype(int).map(LABEL_TO_CLASS)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": args.seed,
        **PRESETS[args.preset],
    }
    if args.early_stopping_rounds > 0:
        params["early_stopping_rounds"] = args.early_stopping_rounds

    model = xgb.XGBClassifier(**params)
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(x_validation, y_validation)],
        verbose=args.verbose_eval,
    )

    predictions = model.predict(x_validation)
    class_report = classification_report(
        y_validation,
        predictions,
        target_names=["SELL", "HOLD", "BUY"],
        output_dict=True,
        zero_division=0,
    )
    label_distribution = {
        "SELL": int((training_frame["target"] == -1).sum()),
        "HOLD": int((training_frame["target"] == 0).sum()),
        "BUY": int((training_frame["target"] == 1).sum()),
    }
    prediction_distribution = {
        "SELL": int((predictions == LABEL_TO_CLASS[-1]).sum()),
        "HOLD": int((predictions == LABEL_TO_CLASS[0]).sum()),
        "BUY": int((predictions == LABEL_TO_CLASS[1]).sum()),
    }
    model_dir = Path(args.output_dir) / args.version
    model_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = model_dir / "model.pkl"
    json_path = model_dir / "model.json"
    scaler_path = model_dir / "scaler.pkl"
    training_data_path = model_dir / "training_data.csv"

    with pickle_path.open("wb") as handle:
        pickle.dump(model, handle)
    model.save_model(json_path)
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    training_frame.to_csv(training_data_path, index=False)

    model_hash = _hash_file(json_path)
    summary = TrainingSummary(
        version=args.version,
        preset=args.preset,
        rows_raw=len(raw),
        rows_training=len(training_frame),
        feature_count=len(feature_columns),
        train_rows=len(train_frame),
        validation_rows=len(validation_frame),
        accuracy=float(accuracy_score(y_validation, predictions)),
        precision=float(precision_score(y_validation, predictions, average="weighted", zero_division=0)),
        recall=float(recall_score(y_validation, predictions, average="weighted", zero_division=0)),
        f1=float(f1_score(y_validation, predictions, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(y_validation, predictions).tolist(),
        label_distribution=label_distribution,
        prediction_distribution=prediction_distribution,
        model_hash=model_hash,
        created_at=datetime.now(UTC).isoformat(),
    )

    metadata = {
        **asdict(summary),
        "framework": "xgboost",
        "device": "cuda",
        "class_mapping": {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "original_label_mapping": {"-1": "SELL", "0": "HOLD", "1": "BUY"},
        "feature_columns": feature_columns,
        "xgboost_params": params,
        "classification_report": class_report,
        "csv_path": str(csv_path),
        "artifacts": {
            "pickle_model": str(pickle_path),
            "xgboost_json_model": str(json_path),
            "scaler": str(scaler_path),
            "training_data": str(training_data_path),
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a larger XGBoost model using NVIDIA GPU acceleration.")
    parser.add_argument("--csv", required=True, help="OHLCV CSV from ai_lab collection.")
    parser.add_argument("--version", default="local-gpu-v1.0.0")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="benchmark")
    parser.add_argument("--min-rows", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--use-atr", action=argparse.BooleanOptionalAction, default=True, help="Use dynamic ATR-scaled Triple Barriers")
    parser.add_argument("--atr-tp", type=float, default=2.0, help="ATR multiplier for take-profit")
    parser.add_argument("--atr-sl", type=float, default=1.0, help="ATR multiplier for stop-loss")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Rounds of no improvement before early stopping (0 to disable)")
    parser.add_argument("--target-return", type=float, default=1.0)
    parser.add_argument("--stop-loss", type=float, default=0.5)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-eval", type=int, default=100)
    args = parser.parse_args()

    summary = train(args)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
