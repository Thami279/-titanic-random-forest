"""Tests for performance monitoring utilities."""

from itertools import count

import numpy as np
import pytest

from titanic_ml.utils.errors import ModelPerformanceError
from titanic_ml.utils.performance import (
    check_performance_budgets,
    measure_inference_latency,
)


class DummyModel:
    """Minimal model for latency testing."""

    def predict(self, X):
        return np.zeros(len(X))


class TestPerformanceBudgets:
    """Tests around budget checking utilities."""

    def test_check_performance_budgets_pass(self):
        metrics = {"accuracy": 0.82, "roc_auc": 0.85, "inference_latency_ms": 80}
        results = check_performance_budgets(metrics)
        assert results["accuracy"] is True
        assert results["roc_auc"] is True
        assert results["inference_latency"] is True

    def test_check_performance_budgets_fail(self):
        metrics = {"accuracy": 0.6, "roc_auc": 0.9}
        with pytest.raises(ModelPerformanceError):
            check_performance_budgets(metrics)

    def test_check_performance_budgets_low_roc_auc_logs_warning(self, caplog):
        metrics = {"accuracy": 0.82, "roc_auc": 0.5}
        with pytest.raises(ModelPerformanceError):
            check_performance_budgets(metrics)
        assert any("ROC-AUC" in message for message in caplog.text.splitlines())

    def test_check_performance_budgets_latency_warning(self, caplog):
        metrics = {"accuracy": 0.82, "roc_auc": 0.85, "inference_latency_ms": 250}
        results = check_performance_budgets(metrics)
        assert results["inference_latency"] is False
        assert any("Latency" in message for message in caplog.text.splitlines())


class TestInferenceLatency:
    """Tests for latency measurement helper."""

    def test_measure_inference_latency(self, monkeypatch):
        # Fake timer: increases by 0.001 seconds each call
        timer = count(start=0.0, step=0.001)

        def fake_time():
            return next(timer)

        monkeypatch.setattr("titanic_ml.utils.performance.time.time", fake_time)

        model = DummyModel()
        X = np.zeros((5, 3))

        latency = measure_inference_latency(model, X, n_iterations=5)

        # Each iteration advances timer by 0.001s twice (start/end) -> 1 ms average
        assert latency == pytest.approx(1.0, rel=0.1)
