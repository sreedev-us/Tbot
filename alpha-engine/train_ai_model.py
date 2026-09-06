#!/usr/bin/env python3
"""
End-to-end AI model training pipeline
Usage: python train_ai_model.py --ohlcv data.csv --output models/local-v1.0.0
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
from alpha_engine.training_pipeline import TrainingDataGenerator, TrainingDataConfig
from alpha_engine.model_trainer import ModelTrainer, EnsembleTrainer
from alpha_engine.backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIModelTrainingPipeline:
    """End-to-end training pipeline"""

    def __init__(self, args):
        self.args = args
        self.training_config = TrainingDataConfig(
            output_dir=args.output_dir,
            lookback_window=args.lookback,
            lookahead_window=args.lookahead,
            target_return_pct=args.target_return,
            stop_loss_pct=args.stop_loss
        )

    def run(self):
        """Execute full training pipeline"""
        logger.info("="*60)
        logger.info("AI MODEL TRAINING PIPELINE")
        logger.info("="*60)

        # Step 1: Load OHLCV data
        logger.info("Step 1: Loading OHLCV data...")
        ohlcv_df = pd.read_csv(self.args.ohlcv)
        logger.info(f"Loaded {len(ohlcv_df)} candles")

        # Step 2: Generate training data
        logger.info("Step 2: Generating training features and labels...")
        generator = TrainingDataGenerator(self.training_config)
        training_df = generator.generate_training_data(
            ohlcv_df,
            output_file=os.path.join(self.args.output_dir, "training_data.csv")
        )

        # Step 3: Train model(s)
        logger.info("Step 3: Training models...")
        feature_columns = generator.get_feature_columns(training_df)
        X = training_df[feature_columns]
        y = training_df['target']

        if self.args.ensemble:
            logger.info("Training ensemble models...")
            ensemble = EnsembleTrainer(self.args.output_dir)
            ensemble.train_ensemble(X, y, test_size=0.2)
            models_to_save = list(ensemble.models.values())
        else:
            logger.info(f"Training {self.args.model_type} model...")
            trainer = ModelTrainer(self.args.output_dir, self.args.model_type)
            trainer.train(X, y)
            models_to_save = [trainer]

        # Step 4: Backtesting
        if self.args.backtest:
            logger.info("Step 4: Running walk-forward backtesting...")
            backtester = Backtester(self.training_config)
            backtest_results = backtester.walk_forward_backtest(
                training_df,
                train_period_days=self.args.backtest_train_days,
                validation_period_days=self.args.backtest_val_days,
                test_period_days=self.args.backtest_test_days,
                step_days=self.args.backtest_step_days,
                model_type=self.args.model_type
            )

            aggregated = backtester.aggregate_results(backtest_results)
            logger.info(f"Backtesting Results: {json.dumps(aggregated, indent=2)}")

            # Save backtest results
            backtest_report = os.path.join(self.args.output_dir, "backtest_report.json")
            with open(backtest_report, 'w') as f:
                json.dump(aggregated, f, indent=2)
            logger.info(f"Backtest report saved to {backtest_report}")

        # Step 5: Save models
        logger.info("Step 5: Saving models...")
        for idx, trainer in enumerate(models_to_save):
            version = self.args.version if not self.args.ensemble else f"{self.args.version}-{trainer.model_type}"
            model_dir = trainer.save_model(
                version,
                metadata={
                    "training_samples": len(training_df),
                    "feature_count": len(X.columns),
                    "feature_columns": X.columns.tolist(),
                    "training_date": datetime.utcnow().isoformat(),
                    "config": {
                        "lookback_window": self.training_config.lookback_window,
                        "lookahead_window": self.training_config.lookahead_window,
                        "target_return_pct": self.training_config.target_return_pct,
                        "stop_loss_pct": self.training_config.stop_loss_pct
                    }
                }
            )
            logger.info(f"Model saved to {model_dir}")

        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Train AI decision model for Tbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ai_model.py --ohlcv data/btc_1m.csv --version local-v1.0.0
  python train_ai_model.py --ohlcv data/btc_1m.csv --version local-v1.0.0 --backtest --ensemble
  python train_ai_model.py --ohlcv data/btc_1m.csv --version local-v1.0.0 --model-type lightgbm
        """
    )

    # Data arguments
    parser.add_argument('--ohlcv', required=True, help='Path to OHLCV data CSV')
    parser.add_argument('--output-dir', default='./models', help='Output directory for models')
    parser.add_argument('--version', default='local-v1.0.0', help='Model version string')

    # Feature engineering arguments
    parser.add_argument('--lookback', type=int, default=50, help='Historical lookback window')
    parser.add_argument('--lookahead', type=int, default=20, help='Future lookahead window for labels')
    parser.add_argument('--target-return', type=float, default=1.0, help='Target return %% for buy signal')
    parser.add_argument('--stop-loss', type=float, default=0.5, help='Stop loss %% for sell signal')

    # Model training arguments
    parser.add_argument('--model-type', choices=['xgboost', 'lightgbm'], default='xgboost', 
                       help='Model type to train')
    parser.add_argument('--ensemble', action='store_true', help='Train ensemble of models')

    # Backtesting arguments
    parser.add_argument('--backtest', action='store_true', help='Run walk-forward backtesting')
    parser.add_argument('--backtest-train-days', type=int, default=180, help='Training window (days)')
    parser.add_argument('--backtest-val-days', type=int, default=30, help='Validation window (days)')
    parser.add_argument('--backtest-test-days', type=int, default=30, help='Test window (days)')
    parser.add_argument('--backtest-step-days', type=int, default=30, help='Window step size (days)')

    args = parser.parse_args()

    pipeline = AIModelTrainingPipeline(args)
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
