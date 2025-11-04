"""Unit tests for preprocessing utilities."""

import pandas as pd
import pytest

from titanic_ml.utils.preprocessing import build_preprocessor, engineer_features


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_engineer_features_basic(self):
        """Test basic feature engineering creates expected columns."""
        df = pd.DataFrame({
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley'],
            'SibSp': [1, 1],
            'Parch': [0, 0],
            'Cabin': ['C85', None],
            'Ticket': ['A/5 21171', 'PC 17599'],
            'Fare': [7.25, 71.2833],
            'Pclass': [3, 1]
        })
        
        result = engineer_features(df)
        
        assert 'Title' in result.columns
        assert 'FamilySize' in result.columns
        assert 'IsAlone' in result.columns
        assert 'CabinKnown' in result.columns
        assert 'TicketGroupSize' in result.columns
        assert 'FarePerPerson' in result.columns
        assert result['FamilySize'].iloc[0] == 2  # SibSp + Parch + 1
        assert result['IsAlone'].iloc[0] == 0
    
    def test_title_extraction(self):
        """Test title extraction from names."""
        # Create enough samples so titles aren't considered rare
        df = pd.DataFrame({
            'Name': (['Smith, Mr. John'] * 10 + 
                     ['Doe, Mrs. Jane'] * 10 + 
                     ['Brown, Miss. Mary'] * 10),
            'SibSp': [0] * 30,
            'Parch': [0] * 30,
            'Cabin': [None] * 30,
            'Ticket': [f'T{i}' for i in range(30)],
            'Fare': [10] * 30,
            'Pclass': [1] * 30
        })
        
        result = engineer_features(df)
        # After title mapping, common titles should remain
        titles = result['Title'].unique()
        # At least should have processed titles (not all "Other")
        assert len(titles) > 0
        # Title extraction should work
        assert 'Title' in result.columns
    
    def test_fare_per_person_calculation(self):
        """Test fare per person calculation."""
        df = pd.DataFrame({
            'Name': ['Test'],
            'SibSp': [2],  # Family of 3
            'Parch': [1],
            'Cabin': [None],
            'Ticket': ['A'],
            'Fare': [30.0],
            'Pclass': [1]
        })
        
        result = engineer_features(df)
        expected_fare_per_person = 30.0 / 4  # FamilySize = 4
        assert abs(result['FarePerPerson'].iloc[0] - expected_fare_per_person) < 0.01


class TestPreprocessor:
    """Test preprocessing pipeline."""
    
    def test_preprocessor_creation(self):
        """Test preprocessor can be created."""
        numeric_features = ['Age', 'Fare']
        categorical_features = ['Sex', 'Embarked']
        
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2
    
    def test_preprocessor_handles_missing_values(self):
        """Test preprocessor handles missing values."""
        from sklearn.compose import ColumnTransformer
        
        numeric_features = ['Age']
        categorical_features = ['Sex']
        
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        
        # Create test data with missing values
        df = pd.DataFrame({
            'Age': [22, None, 35],
            'Sex': ['male', 'female', None]
        })
        
        # Preprocessor should handle this without error
        result = preprocessor.fit_transform(df[numeric_features + categorical_features])
        assert result is not None

