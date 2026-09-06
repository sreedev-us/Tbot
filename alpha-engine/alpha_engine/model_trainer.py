"""
Model training module for AI decision model
Supports XGBoost, LightGBM, and ensemble methods
"""

import os
import json
import logging
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and validates AI decision models"""

    def __init__(self, output_dir: str = "./models", model_type: str = "xgboost"):
        self.output_dir = output_dir
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metrics = {}
        
        os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params
    ) -> Dict:
        """
        Train model with train/test split
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Test set fraction
            random_state: Random seed
            **model_params: Model hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        self.feature_columns = X.columns.tolist()
        
        # Shift labels from [-1, 0, 1] to [0, 1, 2]
        y_shifted = y + 1
        
        # Prevent data leakage: scale only on training set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_shifted, test_size=test_size, random_state=random_state, stratify=y_shifted
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Compute sample weights to balance classes
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        if self.model_type == "xgboost":
            self.model = self._train_xgboost(X_train_scaled, y_train, sample_weight=sample_weights, **model_params)
        elif self.model_type == "lightgbm":
            self.model = self._train_lightgbm(X_train_scaled, y_train, sample_weight=sample_weights, **model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Model metrics: {self.metrics}")
        return self.metrics

    def _train_xgboost(self, X_train, y_train, sample_weight=None, **params):
        """Train XGBoost model"""
        default_params = {
            "objective": "multi:softmax",
            "num_class": 3,
            "max_depth": 10,
            "learning_rate": 0.1275,
            "n_estimators": 439,
            "subsample": 0.9377,
            "colsample_bytree": 0.7806,
            "min_child_weight": 3,
            "random_state": 42,
            "eval_metric": "mlogloss"
        }
        default_params.update(params)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    def _train_lightgbm(self, X_train, y_train, sample_weight=None, **params):
        """Train LightGBM model"""
        default_params = {
            "objective": "multiclass",
            "num_class": 3,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "random_state": 42,
            "verbose": -1
        }
        default_params.update(params)
        
        model = lgb.LGBMClassifier(**default_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        **model_params
    ) -> Dict:
        """
        Cross-validate model
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of folds
            **model_params: Model hyperparameters
            
        Returns:
            Cross-validation metrics
        """
        logger.info(f"Cross-validating with {cv_folds} folds...")
        
        X_scaled = self.scaler.fit_transform(X)
        y_shifted = y + 1
        
        if self.model_type == "xgboost":
            model = xgb.XGBClassifier(**model_params, random_state=42)
        else:
            model = lgb.LGBMClassifier(**model_params, random_state=42)
        
        cv_scores = cross_val_score(model, X_scaled, y_shifted, cv=cv_folds, scoring='accuracy')
        
        cv_metrics = {
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
            "scores": cv_scores.tolist()
        }
        
        logger.info(f"CV metrics: {cv_metrics}")
        return cv_metrics

    def save_model(self, version: str, metadata: Dict = None) -> str:
        """
        Save model with version
        
        Args:
            version: Model version string (e.g., "local-v1.0.0")
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model directory
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_dir = os.path.join(self.output_dir, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Save metadata
        meta = {
            "version": version,
            "model_type": self.model_type,
            "model_hash": model_hash,
            "created_at": datetime.utcnow().isoformat(),
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "framework": "xgboost" if self.model_type == "xgboost" else "lightgbm"
        }
        
        if metadata:
            meta.update(metadata)
        
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        logger.info(f"Model hash: {model_hash}")
        
        return model_dir

    def load_model(self, version: str):
        """Load a saved model"""
        model_path = os.path.join(self.output_dir, version, "model.pkl")
        scaler_path = os.path.join(self.output_dir, version, "scaler.pkl")
        meta_path = os.path.join(self.output_dir, version, "metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.feature_columns = meta.get('feature_columns', [])
        self.metrics = meta.get('metrics', {})
        
        logger.info(f"Model loaded from {version}")
        return meta

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled) - 1
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities


class EnsembleTrainer:
    """Train ensemble of models"""

    def __init__(self, output_dir: str = "./models"):
        self.output_dir = output_dir
        self.models = {}
        os.makedirs(output_dir, exist_ok=True)

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train multiple models and combine predictions"""
        
        results = {}
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_trainer = ModelTrainer(self.output_dir, "xgboost")
        xgb_metrics = xgb_trainer.train(X, y, test_size=test_size)
        results["xgboost"] = xgb_metrics
        self.models["xgboost"] = xgb_trainer
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_trainer = ModelTrainer(self.output_dir, "lightgbm")
        lgb_metrics = lgb_trainer.train(X, y, test_size=test_size)
        results["lightgbm"] = lgb_metrics
        self.models["lightgbm"] = lgb_trainer
        
        logger.info(f"Ensemble results: {results}")
        return results

    def predict_ensemble(self, X: pd.DataFrame) -> Dict:
        """Make predictions with all models"""
        ensemble_results = {}
        
        for model_name, trainer in self.models.items():
            predictions, probabilities = trainer.predict(X)
            ensemble_results[model_name] = {
                "predictions": predictions,
                "probabilities": probabilities
            }
        
        return ensemble_results


if __name__ == "__main__":
    logger.info("Model Trainer initialized")
