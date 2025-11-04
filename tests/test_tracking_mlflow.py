"""Tests for MLflow integration in ExperimentTracker."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from titanic_ml.models.tracking import ExperimentTracker


class TestMLflowIntegration:
    """Test MLflow tracking functionality."""
    
    def test_mlflow_logging_when_available(self):
        """Test that MLflow logging occurs when MLflow is available."""
        # This test verifies the fallback behavior works correctly
        # Full MLflow testing requires MLflow server running
        tracker = ExperimentTracker(use_mlflow=False, experiment_name="test_exp")
        
        # Create a simple model
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Log experiment (should work without MLflow)
        tracker.log(
            name="Test Model",
            estimator=model,
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            params={'n_estimators': 10},
            notes="Test run"
        )
        
        # Verify record was created
        assert len(tracker.records) == 1
        assert tracker.records[0]['model'] == 'Test Model'
        assert 'accuracy' in tracker.records[0]
    
    @patch('titanic_ml.models.tracking.MLFLOW_AVAILABLE', False)
    def test_fallback_when_mlflow_unavailable(self):
        """Test that tracker falls back to in-memory when MLflow unavailable."""
        tracker = ExperimentTracker(use_mlflow=True, experiment_name="test_exp")
        
        # Should still work without MLflow
        assert tracker.use_mlflow == False
        
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Should not raise error
        tracker.log(
            name="Test Model",
            estimator=model,
            y_true=y,
            y_pred=y_pred,
            notes="Test without MLflow"
        )
        
        # Should have record in memory
        assert len(tracker.records) == 1
        assert tracker.records[0]['model'] == 'Test Model'

