"""Tests for titanic_ml.config module."""

import importlib
import os

import pytest


def reload_config():
    """Reload config module to pick up environment changes."""
    from titanic_ml import config

    return importlib.reload(config)


class TestConfigDefaults:
    """Verify default configuration values."""

    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("RANDOM_STATE", raising=False)
        monkeypatch.delenv("TEST_SIZE", raising=False)
        monkeypatch.delenv("VAL_SIZE", raising=False)
        monkeypatch.delenv("MIN_ACCURACY", raising=False)
        monkeypatch.delenv("MIN_ROC_AUC", raising=False)
        monkeypatch.delenv("MAX_LATENCY_MS", raising=False)
        monkeypatch.delenv("MAX_MODEL_SIZE_MB", raising=False)

        config = reload_config()

        assert config.RANDOM_STATE == 42
        assert config.TEST_SIZE == pytest.approx(0.15)
        assert config.VAL_SIZE == pytest.approx(0.15)
        assert config.PERFORMANCE_BUDGETS["min_accuracy"] == pytest.approx(0.75)
        assert config.PERFORMANCE_BUDGETS["min_roc_auc"] == pytest.approx(0.80)
        assert config.PERFORMANCE_BUDGETS["max_inference_latency_ms"] == pytest.approx(100)
        assert config.PERFORMANCE_BUDGETS["max_model_size_mb"] == pytest.approx(50)
        # Baseline features combine numeric and categorical lists
        assert set(config.BASELINE_FEATURES) == set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("RANDOM_STATE", "99")
        monkeypatch.setenv("TEST_SIZE", "0.25")
        monkeypatch.setenv("VAL_SIZE", "0.10")
        monkeypatch.setenv("MIN_ACCURACY", "0.85")
        monkeypatch.setenv("MAX_LATENCY_MS", "50")

        config = reload_config()

        assert config.RANDOM_STATE == 99
        assert config.TEST_SIZE == pytest.approx(0.25)
        assert config.VAL_SIZE == pytest.approx(0.10)
        assert config.PERFORMANCE_BUDGETS["min_accuracy"] == pytest.approx(0.85)
        assert config.PERFORMANCE_BUDGETS["max_inference_latency_ms"] == pytest.approx(50)

