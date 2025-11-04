"""Data validation utilities for ensuring data quality."""

import logging
from typing import Dict, List, Optional

import pandas as pd

from titanic_ml.utils.errors import DataValidationError

logger = logging.getLogger(__name__)


def validate_data(
    df: pd.DataFrame,
    required_columns: List[str],
    numeric_ranges: Optional[Dict[str, tuple]] = None,
    categorical_values: Optional[Dict[str, List[str]]] = None,
    allow_missing: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Validate data meets quality requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_ranges: Dict mapping column names to (min, max) tuples
        categorical_values: Dict mapping column names to allowed values
        allow_missing: List of columns where missing values are allowed
        
    Returns:
        Dict with validation results (column -> passed boolean)
    """
    results = {}
    allow_missing = allow_missing or []
    
    # Check required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)
    
    # Validate numeric ranges
    if numeric_ranges:
        for col, (min_val, max_val) in numeric_ranges.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                results[f"{col}_range"] = out_of_range == 0
                if out_of_range > 0:
                    print(f"Warning: {out_of_range} values in {col} out of range [{min_val}, {max_val}]")
    
    # Validate categorical values
    if categorical_values:
        for col, allowed in categorical_values.items():
            if col in df.columns:
                invalid = ~df[col].isin(allowed + [None, pd.NA] if col in allow_missing else allowed)
                invalid_count = invalid.sum()
                results[f"{col}_categorical"] = invalid_count == 0
                if invalid_count > 0:
                    print(f"Warning: {invalid_count} invalid values in {col}")
    
    # Check for unexpected missing values
    for col in required_columns:
        if col not in allow_missing:
            missing_count = df[col].isna().sum()
            results[f"{col}_missing"] = missing_count == 0
            if missing_count > 0:
                print(f"Warning: {missing_count} missing values in {col}")
    
    return results

