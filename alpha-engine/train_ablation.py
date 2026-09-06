#!/usr/bin/env python3
"""Train Ablation Models: C1, C2, C3"""
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

config = TrainingDataConfig()
gen = TrainingDataGenerator(config)

levels = {
    "C1": "Model C1: Technical + Server AI + fng_norm",
    "C2": "Model C2: Technical + Server AI + fng_norm + fng_ma_7",
    "C3": "Model C3: Technical + Server AI + fng_norm + fng_ma_7 + fng_zscore_14",
}

for ablation, desc in levels.items():
    print(f"\n{'='*50}\nTraining {ablation}\n{'='*50}")
    
    # Generate data for this ablation level
    df = gen.generate_training_data(ohlcv, fng_df=fng, ablation_level=ablation)
    feature_cols = gen.get_feature_columns(df)
    fng_features = [c for c in feature_cols if "fng" in c]
    
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  F&G features: {fng_features}")
    
    X = df[feature_cols]
    y = df["target"]
    
    print(f"Training XGBoost for {ablation}...")
    trainer = ModelTrainer("./models", "xgboost")
    trainer.train(X, y)
    
    model_dir = trainer.save_model(
        f"local-model-{ablation.lower()}",
        metadata={
            "training_samples": len(df),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "description": desc,
            "sentiment_source": "alternative.me Fear&Greed Index (strict merge_asof causal join)",
        },
    )
    print(f"{ablation} saved to {model_dir}")
