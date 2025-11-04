"""End-to-end training utilities for Titanic survival models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from titanic_ml import config
from titanic_ml.models.tracking import ExperimentTracker
from titanic_ml.utils.errors import ModelPerformanceError
from titanic_ml.utils.performance import check_performance_budgets, measure_inference_latency
from titanic_ml.utils.preprocessing import build_preprocessor, engineer_features
from titanic_ml.utils.validation import validate_data


@dataclass
class ModelEvaluation:
    """Container for model metrics and artefacts."""

    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    params: Dict[str, object] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_summary(self) -> Dict[str, object]:
        """Return a JSON-serialisable summary without estimator objects."""
        return {
            "model": self.name,
            "metrics": self.metrics,
            "params": self.params,
            "notes": self.notes,
        }


@dataclass
class TrainingResult:
    """Aggregate outcome of a training run."""

    data_hash: str
    evaluations: List[ModelEvaluation]
    best_model: ModelEvaluation
    tracker_dataframe: pd.DataFrame

    def to_report(self) -> Dict[str, object]:
        """Produce a concise training report."""
        return {
            "data_hash": self.data_hash,
            "best_model": self.best_model.to_summary(),
            "all_models": [evaluation.to_summary() for evaluation in self.evaluations],
        }


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Stable hash for dataset provenance."""
    return pd.util.hash_pandas_object(df.sort_index(axis=1)).values.sum().__int__().__format__("x")


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    test_size: float,
    val_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/validation/test splits honouring config sizes."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Validation size relative to the remaining training data
    relative_val_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _score_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def _log_to_tracker(
    tracker: ExperimentTracker,
    evaluation: ModelEvaluation,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> None:
    """Persist evaluation metrics to experiment tracker."""
    tracker.log(
        name=evaluation.name,
        estimator=evaluation.pipeline.named_steps["model"],
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        params=evaluation.params,
        notes=evaluation.notes,
    )


def _evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    tracker: ExperimentTracker,
    params: Optional[Dict[str, object]] = None,
    notes: Optional[str] = None,
    enforce_budgets: bool = True,
) -> ModelEvaluation:
    """Fit pipeline, compute metrics, optionally enforce budgets, and log results."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_eval)
    y_proba: Optional[np.ndarray]
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_eval)[:, 1]
    else:
        y_proba = None

    metrics = _score_predictions(y_eval, y_pred, y_proba)
    budget_failure: Optional[str] = None
    try:
        check_performance_budgets(metrics, config.PERFORMANCE_BUDGETS)
    except ModelPerformanceError as exc:
        if enforce_budgets:
            raise
        budget_failure = str(exc)

    if budget_failure:
        note_parts = [text for text in [notes, f"Budget check failed: {budget_failure}"] if text]
        notes = " | ".join(note_parts) if note_parts else f"Budget check failed: {budget_failure}"

    evaluation = ModelEvaluation(
        name=name,
        pipeline=pipeline,
        metrics=metrics,
        params=params or pipeline.named_steps["model"].get_params(),
        notes=notes,
    )
    _log_to_tracker(tracker, evaluation, y_eval, y_pred, y_proba)
    return evaluation


def _grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, Iterable[object]],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Pipeline,
    random_state: int,
    scoring: str = "accuracy",
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, object], GridSearchCV]:
    """Perform grid search on the provided estimator."""
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid


def train_models(
    data_path: Path = config.DATA_PATH,
    *,
    use_mlflow: bool = False,
    fast_mode: bool = False,
) -> TrainingResult:
    """Train, evaluate, and compare ensemble models for Titanic survival.

    Args:
        data_path: Location of the Titanic CSV dataset.
        use_mlflow: Whether to persist experiments to MLflow.
        fast_mode: Reduce search space for quicker runs (useful for tests).

    Returns:
        TrainingResult with all model evaluations and metadata.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    raw_df = pd.read_csv(data_path)

    validate_data(
        raw_df,
        required_columns=[
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Ticket",
            "Cabin",
            "Name",
        ],
        numeric_ranges={
            "Age": (0, 100),
            "Fare": (0, 600),
            "SibSp": (0, 10),
            "Parch": (0, 10),
        },
        categorical_values={
            "Sex": ["male", "female"],
            "Embarked": ["S", "C", "Q"],
        },
        allow_missing=["Cabin", "Age", "Embarked"],
    )

    engineered_df = engineer_features(raw_df)
    data_hash = _hash_dataframe(engineered_df)

    X = engineered_df[config.BASELINE_FEATURES]
    y = engineered_df["Survived"]

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X,
        y,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
    )

    tracker = ExperimentTracker(use_mlflow=use_mlflow, experiment_name=config.MLFLOW_EXPERIMENT_NAME)
    preprocessor = build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)

    evaluations: List[ModelEvaluation] = []

    # Baseline random forest (no tuning)
    baseline_estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        oob_score=False,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    baseline_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", baseline_estimator),
        ]
    )
    evaluations.append(
        _evaluate_pipeline(
            "Baseline RandomForest",
            baseline_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            tracker=tracker,
            notes="Baseline configuration using engineered features",
            enforce_budgets=False,
        )
    )

    # Tuned random forest
    rf_param_grid = (
        {
            "model__n_estimators": [150, 300, 450],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_split": [2, 4],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2", 0.8],
            "model__class_weight": [None, "balanced"],
        }
        if not fast_mode
        else {
            "model__n_estimators": [150, 300, 450],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_split": [2, 4],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2", 0.8],
            "model__class_weight": [None, "balanced"],
        }
    )

    tuned_pipeline, tuned_params, tuned_grid = _grid_search(
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        param_grid=rf_param_grid,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        random_state=config.RANDOM_STATE,
        n_jobs=1 if fast_mode else -1,
    )
    evaluations.append(
        _evaluate_pipeline(
            "Tuned RandomForest",
            tuned_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            tracker=tracker,
            params=tuned_params,
            notes=f"GridSearchCV mean accuracy={tuned_grid.best_score_:.4f}",
            enforce_budgets=False,
        )
    )

    # Bagging ensemble
    bagging_param_grid = (
        {
            "model__n_estimators": [50, 100],
            "model__estimator__max_depth": [None, 6],
            "model__estimator__min_samples_split": [2, 4],
        }
        if not fast_mode
        else {"model__n_estimators": [50], "model__estimator__max_depth": [None]}
    )

    bagging_pipeline, bagging_params, bagging_grid = _grid_search(
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            random_state=config.RANDOM_STATE,
        ),
        param_grid=bagging_param_grid,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        random_state=config.RANDOM_STATE,
        n_jobs=1 if fast_mode else -1,
    )
    evaluations.append(
        _evaluate_pipeline(
            "Bagging (DecisionTree)",
            bagging_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            tracker=tracker,
            params=bagging_params,
            notes=f"GridSearchCV mean accuracy={bagging_grid.best_score_:.4f}",
            enforce_budgets=False,
        )
    )

    # Boosting ensemble
    boosting_param_grid = (
        {
            "model__n_estimators": [50, 100],
            "model__estimator__max_depth": [1, 2],
        }
        if not fast_mode
        else {"model__n_estimators": [50], "model__estimator__max_depth": [1]}
    )

    boosting_pipeline, boosting_params, boosting_grid = _grid_search(
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            random_state=config.RANDOM_STATE,
        ),
        param_grid=boosting_param_grid,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        random_state=config.RANDOM_STATE,
        n_jobs=1 if fast_mode else -1,
    )
    evaluations.append(
        _evaluate_pipeline(
            "AdaBoost (DecisionTree)",
            boosting_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            tracker=tracker,
            params=boosting_params,
            notes=f"GridSearchCV mean accuracy={boosting_grid.best_score_:.4f}",
            enforce_budgets=False,
        )
    )

    # Select best based on validation accuracy, then refit on train+val
    evaluations.sort(key=lambda e: e.metrics["accuracy"], reverse=True)
    best_model = evaluations[0]

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    best_model.pipeline.fit(X_train_full, y_train_full)

    y_test_pred = best_model.pipeline.predict(X_test)
    y_test_proba: Optional[np.ndarray]
    if hasattr(best_model.pipeline.named_steps["model"], "predict_proba"):
        y_test_proba = best_model.pipeline.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = None
    test_metrics = _score_predictions(y_test, y_test_pred, y_test_proba)
    check_performance_budgets(test_metrics, config.PERFORMANCE_BUDGETS)

    # Measure inference latency on preprocessed features
    transformed_features = best_model.pipeline.named_steps["prep"].transform(X_test)
    latency_ms = measure_inference_latency(
        best_model.pipeline.named_steps["model"],
        transformed_features,
        n_iterations=50,
    )
    test_metrics["inference_latency_ms"] = latency_ms

    # Persist final test evaluation to tracker
    tracker.log(
        name=f"{best_model.name} (Test Evaluation)",
        estimator=best_model.pipeline.named_steps["model"],
        y_true=y_test,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        params=best_model.params,
        notes=json.dumps({"scope": "test_set", "latency_ms": latency_ms}),
    )

    best_model.metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    tracker_df = tracker.to_dataframe()

    return TrainingResult(
        data_hash=data_hash,
        evaluations=evaluations,
        best_model=best_model,
        tracker_dataframe=tracker_df,
    )
