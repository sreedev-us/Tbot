#!/usr/bin/env python3
"""Train Model C: Technical + Fear & Greed Index sentiment."""
import datetime
import logging
import pandas as pd
import sys

sys.path.insert(0, ".")
from alpha_engine.training_pipeline import TrainingDataGenerator, TrainingDataConfig
from alpha_engine.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

print("Loading OHLCV and F&G data...")
ohlcv = pd.read_csv("data/live/BTC_USDT_1m_live.csv")
fng = pd.read_csv("data/sentiment/fear_greed_historical.csv")
print(f"  OHLCV: {len(ohlcv)} candles | F&G: {len(fng)} days")

print("Engineering features (Model C: OHLCV + F&G)...")
config = TrainingDataConfig()
gen = TrainingDataGenerator(config)
df = gen.generate_training_data(ohlcv, fng_df=fng, output_file="models/training_data_c.csv")

feature_cols = gen.get_feature_columns(df)
fng_features = [c for c in feature_cols if "fng" in c]
print(f"  Feature count: {len(feature_cols)}")
print(f"  F&G features: {fng_features}")

X = df[feature_cols]
y = df["target"]

print("Training Model C (XGBoost)...")
trainer = ModelTrainer("./models", "xgboost")
trainer.train(X, y)

model_dir = trainer.save_model(
    "local-model-c",
    metadata={
        "training_samples": len(df),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "description": "Model C: Technical + Server AI features + Fear&Greed Index",
        "sentiment_source": "alternative.me Fear&Greed Index (daily, causal date-join)",
    },
)
print(f"Model C saved to {model_dir}")
