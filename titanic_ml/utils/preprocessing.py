"""Data preprocessing and feature engineering utilities."""

import logging
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic_ml.utils.errors import DataValidationError

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features from raw Titanic data.
    
    Args:
        df: Raw DataFrame with Titanic passenger data
        
    Returns:
        DataFrame with engineered features added
    """
    if df is None or df.empty:
        raise DataValidationError("Input DataFrame is empty")
    if 'Name' not in df.columns:
        raise DataValidationError("Missing required column: 'Name'")
    
    logger.info(f"Engineering features for {len(df)} samples")
    out = df.copy()

    # Titles extracted from names
    titles = out['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
    title_map = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Noble', 'Countess': 'Noble', 'Capt': 'Officer', 'Col': 'Officer',
        'Major': 'Officer', 'Dr': 'Officer', 'Rev': 'Officer', 'Sir': 'Noble',
        'Don': 'Noble', 'Jonkheer': 'Noble'
    }
    titles = titles.map(lambda t: title_map.get(t, t))
    counts = titles.value_counts()
    common_titles = counts[counts >= 10].index
    titles = titles.where(titles.isin(common_titles), 'Other')
    out['Title'] = titles.astype('object')

    # Family based features
    out['FamilySize'] = out['SibSp'] + out['Parch'] + 1
    out['IsAlone'] = (out['FamilySize'] == 1).astype(int)

    # Cabin availability
    out['CabinKnown'] = out['Cabin'].notna().astype(int)

    # Ticket group size
    ticket_counts = out['Ticket'].value_counts()
    out['TicketGroupSize'] = out['Ticket'].map(ticket_counts)

    # Fare per person (avoid division by zero)
    out['FarePerPerson'] = out['Fare'] / out['FamilySize'].replace(0, 1)

    # Treat passenger class as categorical
    out['Pclass'] = out['Pclass'].astype(str)

    return out


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer with imputation and encoding pipelines
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor
