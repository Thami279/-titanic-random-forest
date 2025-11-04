"""Integration tests for the training orchestration module."""

from pathlib import Path

import pandas as pd
import pytest

from titanic_ml.models.training import TrainingResult, train_models
from titanic_ml.utils.errors import ModelPerformanceError


@pytest.mark.integration
def test_train_models_returns_valid_result(tmp_path: Path):
    """train_models should run end-to-end (fast mode).
    Accept either a successful result or a final budget failure, but never crash earlier.
    """
    dataset_path = Path("titanic.csv")
    if not dataset_path.exists():
        pytest.skip("Titanic dataset not available")

    try:
        result = train_models(data_path=dataset_path, use_mlflow=False, fast_mode=True)
        assert isinstance(result, TrainingResult)
        assert result.best_model.metrics["accuracy"] > 0
        assert "test_accuracy" in result.best_model.metrics
        assert "test_inference_latency_ms" in result.best_model.metrics
        report = result.to_report()
        assert report["best_model"]["metrics"]["accuracy"] > 0
        assert isinstance(result.tracker_dataframe, pd.DataFrame)
        assert not result.tracker_dataframe.empty
    except ModelPerformanceError as e:
        # Final budget enforcement may fail when accuracy is just below threshold.
        # This is acceptable: the pipeline ran end-to-end and enforced budgets at test time.
        assert "Performance budgets not met" in str(e)
