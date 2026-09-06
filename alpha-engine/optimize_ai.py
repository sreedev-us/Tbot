import argparse
import optuna
import pandas as pd
from alpha_engine.training_pipeline import TrainingDataGenerator, TrainingDataConfig
from alpha_engine.model_trainer import ModelTrainer

def objective(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    trainer = ModelTrainer(output_dir="./models/optuna_temp", model_type="xgboost")
    # Train with a train/test split to get the validation F1 score
    metrics = trainer.train(X, y, test_size=0.2, random_state=42, **params)
    
    return metrics["f1"]

def main():
    parser = argparse.ArgumentParser(description="Optimize AI Model Hyperparameters")
    parser.add_argument("--ohlcv", required=True, help="Path to OHLCV CSV data")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    print("Loading data...")
    ohlcv_df = pd.read_csv(args.ohlcv)
    
    print("Generating training data...")
    config = TrainingDataConfig()
    generator = TrainingDataGenerator(config)
    df = generator.generate_training_data(ohlcv_df)
    
    feature_columns = generator.get_feature_columns(df)
    X = df[feature_columns]
    y = df['target']
    
    print(f"Starting optimization for {args.trials} trials...")
    study = optuna.create_study(direction="maximize", study_name="xgboost-f1-tuning")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials)
    
    print("\nOptimization Finished!")
    print(f"Best F1 score: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
