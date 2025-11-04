"""Experiment tracking with MLflow integration."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """
    Light-weight tracker for experiments with MLflow persistence.
    
    Can work with or without MLflow - if MLflow is not configured,
    falls back to in-memory tracking.
    """
    
    def __init__(self, use_mlflow: bool = True, experiment_name: str = "titanic_survival"):
        """
        Initialize tracker.
        
        Args:
            use_mlflow: Whether to use MLflow for persistence
            experiment_name: Name of MLflow experiment
        """
        self.records = []
        self.use_mlflow = use_mlflow and self._check_mlflow()
        
        if self.use_mlflow:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                self.use_mlflow = False
                print("Warning: MLflow not available, using in-memory tracking only")

    def _check_mlflow(self) -> bool:
        """Check if MLflow is properly configured."""
        return MLFLOW_AVAILABLE

    def log(
        self,
        name: str,
        estimator: BaseEstimator,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Log experiment results.
        
        Args:
            name: Model name
            estimator: Trained estimator
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            params: Model parameters
            notes: Additional notes
        """
        record = {
            'model': name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            try:
                record['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                record['roc_auc'] = np.nan
        
        record['params'] = params or getattr(estimator, 'get_params', lambda: {})()
        record['notes'] = notes
        self.records.append(record)
        
        # Log to MLflow if enabled
        if self.use_mlflow and MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_params(record['params'])
                    mlflow.log_metrics({
                        'accuracy': record['accuracy'],
                        'precision': record['precision'],
                        'recall': record['recall'],
                        'f1': record['f1'],
                        'roc_auc': record.get('roc_auc', np.nan)
                    })
                    if notes:
                        mlflow.log_param('notes', notes)
                    mlflow.sklearn.log_model(estimator, "model")
            except Exception as e:
                print(f"Warning: Failed to log to MLflow: {e}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to DataFrame."""
        return pd.DataFrame(self.records)

