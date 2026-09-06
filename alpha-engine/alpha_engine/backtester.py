"""
Backtesting framework with walk-forward testing
Prevents data leakage and provides realistic performance estimates
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from alpha_engine.model_trainer import ModelTrainer
from alpha_engine.training_pipeline import TrainingDataGenerator, TrainingDataConfig, NON_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Backtesting performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    cumulative_return: float
    annual_return: float
    avg_trade_pnl: float
    accuracy: float
    precision: float
    recall: float


class BacktestResult:
    """Contains backtesting results"""

    def __init__(self, model_version: str, period: str):
        self.model_version = model_version
        self.period = period
        self.trades = []
        self.predictions = []
        self.actuals = []

    def add_trade(self, entry_price: float, exit_price: float, confidence: float, prediction: int, actual: int):
        """Record a trade"""
        pnl = (exit_price - entry_price) / entry_price
        self.trades.append({
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl * 100,
            "confidence": confidence,
            "prediction": prediction,
            "actual": actual
        })
        self.predictions.append(prediction)
        self.actuals.append(actual)

    def calculate_metrics(self, annual_trading_days: int = 252) -> BacktestMetrics:
        """Calculate performance metrics"""
        if not self.trades:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        trades_df = pd.DataFrame(self.trades)

        # Trade metrics
        winning_trades = (trades_df['pnl_pct'] > 0).sum()
        losing_trades = (trades_df['pnl_pct'] <= 0).sum()
        total_trades = len(trades_df)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0)

        # Returns
        cumulative_return = trades_df['pnl_pct'].sum()
        annual_return = cumulative_return * (annual_trading_days / total_trades)

        # Sharpe ratio
        returns = trades_df['pnl_pct'].values / 100
        sharpe_ratio = np.sqrt(annual_trading_days) * (np.mean(returns) / (np.std(returns) + 1e-8))

        # Max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Prediction accuracy
        predictions_array = np.array(self.predictions)
        actuals_array = np.array(self.actuals)
        accuracy = (predictions_array == actuals_array).mean()

        # Precision (TP / (TP + FP))
        buy_predictions = predictions_array == 1
        buy_actuals = actuals_array == 1
        tp = np.sum((buy_predictions) & (buy_actuals))
        fp = np.sum((buy_predictions) & (~buy_actuals))
        precision = tp / (tp + fp + 1e-8)

        # Recall (TP / (TP + FN))
        fn = np.sum((~buy_predictions) & (buy_actuals))
        recall = tp / (tp + fn + 1e-8)

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            cumulative_return=float(cumulative_return),
            annual_return=float(annual_return),
            avg_trade_pnl=float(trades_df['pnl_pct'].mean()),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall)
        )


class Backtester:
    """Backtesting engine with walk-forward analysis"""

    def __init__(self, config: TrainingDataConfig = None):
        self.config = config or TrainingDataConfig()
        self.results = {}

    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        train_period_days: int = 180,
        validation_period_days: int = 30,
        test_period_days: int = 30,
        step_days: int = 30,
        model_type: str = "xgboost"
    ) -> Dict[str, BacktestResult]:
        """
        Walk-forward backtesting: prevent data leakage
        
        Process:
        1. Split data into rolling windows
        2. For each window: train on train_period, validate on validation_period, test on test_period
        3. Move window forward by step_days
        4. Aggregate results
        
        Args:
            df: Full dataset with features and target
            train_period_days: Training window size
            validation_period_days: Validation window size
            test_period_days: Test window size
            step_days: Days to move window forward
            model_type: "xgboost" or "lightgbm"
            
        Returns:
            Dictionary of BacktestResult by period
        """
        logger.info("Starting walk-forward backtesting...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Convert to days if needed
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['day'] = df['timestamp'].dt.date
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day'] = df['timestamp'].dt.date
        
        unique_days = df['day'].unique()
        results = {}
        window_num = 0
        
        for start_idx in range(0, len(unique_days), step_days):
            window_num += 1
            
            train_end_idx = start_idx + train_period_days
            val_end_idx = train_end_idx + validation_period_days
            test_end_idx = val_end_idx + test_period_days
            
            if test_end_idx > len(unique_days):
                logger.info(f"Reached end of data at window {window_num}")
                break
            
            train_days = unique_days[start_idx:train_end_idx]
            val_days = unique_days[train_end_idx:val_end_idx]
            test_days = unique_days[val_end_idx:test_end_idx]
            
            logger.info(f"Window {window_num}: Train={len(train_days)}d, Val={len(val_days)}d, Test={len(test_days)}d")
            
            # Split data
            train_df = df[df['day'].isin(train_days)]
            val_df = df[df['day'].isin(val_days)]
            test_df = df[df['day'].isin(test_days)]
            
            # Train model
            if len(train_df) < 10:
                logger.warning(f"Insufficient training data for window {window_num}")
                continue
            
            try:
                feature_cols = [col for col in train_df.columns if col not in NON_FEATURE_COLUMNS]
                
                trainer = ModelTrainer(model_type=model_type)
                trainer.train(train_df[feature_cols], train_df['target'])
                
                # Backtest on test set
                if len(test_df) > 0:
                    result = BacktestResult(f"{model_type}-window-{window_num}", f"test_{window_num}")
                    
                    predictions, probabilities = trainer.predict(test_df[feature_cols])
                    
                    for idx, (pred, actual) in enumerate(zip(predictions, test_df['target'].values)):
                        # Use close prices from test set
                        if idx < len(test_df) - 1:
                            entry_price = test_df.iloc[idx]['close']
                            exit_price = test_df.iloc[idx + 1]['close']
                            confidence = float(np.max(probabilities[idx]))
                            result.add_trade(entry_price, exit_price, confidence, pred, actual)
                    
                    metrics = result.calculate_metrics()
                    results[f"window_{window_num}_test"] = result
                    
                    logger.info(f"Window {window_num} Test Metrics: WinRate={metrics.win_rate:.2%}, Accuracy={metrics.accuracy:.2%}, Sharpe={metrics.sharpe_ratio:.2f}")
                
                # Also validate on validation set
                if len(val_df) > 0:
                    val_result = BacktestResult(f"{model_type}-window-{window_num}", f"val_{window_num}")
                    val_predictions, val_probabilities = trainer.predict(val_df[feature_cols])
                    
                    for idx, (pred, actual) in enumerate(zip(val_predictions, val_df['target'].values)):
                        if idx < len(val_df) - 1:
                            entry_price = val_df.iloc[idx]['close']
                            exit_price = val_df.iloc[idx + 1]['close']
                            confidence = float(np.max(val_probabilities[idx]))
                            val_result.add_trade(entry_price, exit_price, confidence, pred, actual)
                    
                    val_metrics = val_result.calculate_metrics()
                    results[f"window_{window_num}_val"] = val_result
                    logger.info(f"Window {window_num} Val Metrics: WinRate={val_metrics.win_rate:.2%}, Accuracy={val_metrics.accuracy:.2%}")
            
            except Exception as e:
                logger.error(f"Error in window {window_num}: {e}")
                continue
        
        logger.info(f"Walk-forward backtesting complete ({window_num} windows)")
        return results

    def aggregate_results(self, results: Dict[str, BacktestResult]) -> Dict:
        """Aggregate metrics across all windows"""
        if not results:
            return {}
        
        all_metrics = [result.calculate_metrics() for result in results.values()]
        
        aggregated = {
            "total_windows": len(results),
            "avg_win_rate": np.mean([m.win_rate for m in all_metrics]),
            "avg_accuracy": np.mean([m.accuracy for m in all_metrics]),
            "avg_sharpe": np.mean([m.sharpe_ratio for m in all_metrics]),
            "avg_profit_factor": np.mean([m.profit_factor for m in all_metrics]),
            "avg_max_drawdown": np.mean([m.max_drawdown for m in all_metrics]),
            "total_trades": sum([m.total_trades for m in all_metrics])
        }
        
        logger.info(f"Aggregated Results: {aggregated}")
        return aggregated


if __name__ == "__main__":
    logger.info("Backtester initialized")
