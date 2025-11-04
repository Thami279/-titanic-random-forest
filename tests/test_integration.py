"""Integration tests for preprocessing and model training."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from titanic_ml.models.tracking import ExperimentTracker
from titanic_ml.utils.preprocessing import build_preprocessor, engineer_features


class TestEndToEndWorkflow:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample Titanic-like data."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'Name': [f'Person {i}' for i in range(n_samples)],
            'Pclass': np.random.choice([1, 2, 3], n_samples),
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.randint(1, 80, n_samples),
            'SibSp': np.random.randint(0, 3, n_samples),
            'Parch': np.random.randint(0, 3, n_samples),
            'Fare': np.random.uniform(5, 300, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples),
            'Cabin': np.random.choice(['C85', None], n_samples),
            'Ticket': [f'T{i}' for i in range(n_samples)],
            'Survived': np.random.choice([0, 1], n_samples)
        })
    
    def test_full_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        # Feature engineering
        engineered = engineer_features(sample_data)
        
        # Prepare features
        numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'TicketGroupSize', 'FarePerPerson']
        categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinKnown']
        
        X = engineered[numeric_features + categorical_features]
        y = engineered['Survived']
        
        # Build preprocessor
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Should complete without errors
        assert len(y_pred) == len(y_test)
        assert np.all(np.isin(y_pred, [0, 1]))
    
    def test_experiment_tracker_integration(self, sample_data):
        """Test experiment tracker works with model training."""
        engineered = engineer_features(sample_data)
        
        numeric_features = ['Age', 'Fare']
        categorical_features = ['Sex']
        
        X = engineered[numeric_features + categorical_features]
        y = engineered['Survived']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Test tracker (without MLflow for unit tests)
        tracker = ExperimentTracker(use_mlflow=False)
        tracker.log(
            name='Test Model',
            estimator=pipeline.named_steps['model'],
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            notes='Integration test'
        )
        
        results_df = tracker.to_dataframe()
        assert len(results_df) == 1
        assert 'accuracy' in results_df.columns
        assert results_df.iloc[0]['model'] == 'Test Model'

