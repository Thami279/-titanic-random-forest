"""Custom exceptions for titanic_ml package."""


class TitanicMLError(Exception):
    """Base exception for titanic_ml package."""
    pass


class DataValidationError(TitanicMLError):
    """Raised when data validation fails."""
    pass


class ModelPerformanceError(TitanicMLError):
    """Raised when model doesn't meet performance budgets."""
    pass








