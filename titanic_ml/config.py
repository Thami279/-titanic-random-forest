"""Configuration management for titanic_ml package."""

import os
from pathlib import Path
from typing import Dict

# Paths
DATA_DIR = Path(os.getenv("TITANIC_DATA_DIR", "."))
DATA_PATH = DATA_DIR / "titanic.csv"

# Model configuration
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.15"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.15"))

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "titanic_survival")

# Performance budgets
PERFORMANCE_BUDGETS: Dict[str, float] = {
    'min_accuracy': float(os.getenv("MIN_ACCURACY", "0.75")),
    'min_roc_auc': float(os.getenv("MIN_ROC_AUC", "0.80")),
    'max_inference_latency_ms': float(os.getenv("MAX_LATENCY_MS", "100")),
    'max_model_size_mb': float(os.getenv("MAX_MODEL_SIZE_MB", "50"))
}

# Feature lists
NUMERIC_FEATURES = [
    'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'TicketGroupSize', 'FarePerPerson'
]

CATEGORICAL_FEATURES = [
    'Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinKnown'
]

BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES








