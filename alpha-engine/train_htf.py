#!/usr/bin/env python3
"""
Train baseline Model B on Higher Timeframes (15m and 1h).
"""
import datetime
import logging
import pandas as pd
import sys

sys.path.insert(0, ".")
from alpha_engine.training_pipeline import TrainingDataGenerator, TrainingDataConfig
from alpha_engine.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

timeframes = {
    "15m": "data/live/BTC_USDT_15m_live.csv",
    "1h": "data/live/BTC_USDT_1h_live.csv"
}

for tf, filepath in timeframes.items():
    print(f"\n{'='*50}\nTraining Baseline B on {tf}\n{'='*50}")
    
    ohlcv = pd.read_csv(filepath)
    print(f"Loaded {len(ohlcv)} candles for {tf}")
    
    # We use ablation_level="B" to freeze F&G sentiment and use only the 64 core features
    config = TrainingDataConfig()
    gen = TrainingDataGenerator(config)
    df = gen.generate_training_data(ohlcv, ablation_level="B")
    
    feature_cols = gen.get_feature_columns(df)
    print(f"Feature count: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df["target"]
    
    print(f"Training XGBoost on {tf} data...")
    trainer = ModelTrainer("./models", "xgboost")
    trainer.train(X, y)
    
    model_name = f"local-model-b-{tf}"
    model_dir = trainer.save_model(
        model_name,
        metadata={
            "training_samples": len(df),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "description": f"Model B (Technical + Server AI) trained on {tf} timeframe",
        }
    )
    print(f"{model_name} saved to {model_dir}")
