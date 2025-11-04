"""Unit tests for data validation utilities."""

import pandas as pd
import pytest

from titanic_ml.utils.validation import validate_data


class TestDataValidation:
    """Test data validation functions."""
    
    def test_required_columns_check(self):
        """Test validation fails when required columns are missing."""
        from titanic_ml.utils.errors import DataValidationError
        
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_data(df, required_columns=['A', 'B', 'C'])
    
    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        df = pd.DataFrame({
            'Age': [25, 30, 150],  # 150 is out of reasonable range
            'Fare': [10, 20, 30]
        })
        
        numeric_ranges = {'Age': (0, 120)}
        results = validate_data(
            df,
            required_columns=['Age', 'Fare'],
            numeric_ranges=numeric_ranges
        )
        
        assert 'Age_range' in results
        # Should fail because 150 is out of range
        assert results['Age_range'] == False
    
    def test_categorical_validation(self):
        """Test categorical value validation."""
        df = pd.DataFrame({
            'Sex': ['male', 'female', 'unknown'],  # 'unknown' is invalid
            'Embarked': ['S', 'C', 'Q']
        })
        
        categorical_values = {
            'Sex': ['male', 'female'],
            'Embarked': ['S', 'C', 'Q']
        }
        
        results = validate_data(
            df,
            required_columns=['Sex', 'Embarked'],
            categorical_values=categorical_values
        )
        
        assert 'Sex_categorical' in results
        assert results['Sex_categorical'] == False  # Should fail due to 'unknown'
    
    def test_missing_values_validation(self):
        """Test missing values validation."""
        df = pd.DataFrame({
            'Age': [25, None, 30],  # Has missing value
            'Name': ['A', 'B', 'C']
        })
        
        results = validate_data(
            df,
            required_columns=['Age', 'Name'],
            allow_missing=['Name']  # Allow missing in Name
        )
        
        assert 'Age_missing' in results
        assert results['Age_missing'] == False  # Should fail

