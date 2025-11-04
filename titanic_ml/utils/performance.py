"""Performance monitoring and budget enforcement utilities."""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator

from titanic_ml.utils.errors import ModelPerformanceError

logger = logging.getLogger(__name__)

# Performance budgets
PERFORMANCE_BUDGETS = {
    'min_accuracy': 0.75,
    'min_roc_auc': 0.80,
    'max_inference_latency_ms': 100,
    'max_model_size_mb': 50
}


def check_performance_budgets(
    metrics: Dict[str, float],
    budgets: Optional[Dict[str, float]] = None
) -> Dict[str, bool]:
    """
    Check if metrics meet performance budgets.
    
    Args:
        metrics: Dictionary of metric names to values
        budgets: Optional custom budgets (uses defaults if None)
        
    Returns:
        Dictionary of budget check results (metric -> passed boolean)
        
    Raises:
        ModelPerformanceError if critical budgets not met
    """
    budgets = budgets or PERFORMANCE_BUDGETS
    results = {}
    
    # Check minimum accuracy
    if 'accuracy' in metrics:
        min_acc = budgets.get('min_accuracy', 0.75)
        passed = metrics['accuracy'] >= min_acc
        results['accuracy'] = passed
        if not passed:
            logger.warning(f"Accuracy {metrics['accuracy']:.4f} below budget {min_acc}")
    
    # Check minimum ROC-AUC
    if 'roc_auc' in metrics:
        min_auc = budgets.get('min_roc_auc', 0.80)
        passed = metrics['roc_auc'] >= min_auc
        results['roc_auc'] = passed
        if not passed:
            logger.warning(f"ROC-AUC {metrics['roc_auc']:.4f} below budget {min_auc}")
    
    # Check inference latency (if provided)
    if 'inference_latency_ms' in metrics:
        max_latency = budgets.get('max_inference_latency_ms', 100)
        passed = metrics['inference_latency_ms'] <= max_latency
        results['inference_latency'] = passed
        if not passed:
            logger.warning(f"Latency {metrics['inference_latency_ms']:.2f}ms exceeds budget {max_latency}ms")
    
    # Fail if critical budgets not met
    critical_failed = [k for k, v in results.items() if not v and k in ['accuracy', 'roc_auc']]
    if critical_failed:
        raise ModelPerformanceError(
            f"Performance budgets not met for: {', '.join(critical_failed)}"
        )
    
    return results


def measure_inference_latency(
    model: BaseEstimator,
    X: np.ndarray,
    n_iterations: int = 100
) -> float:
    """
    Measure average inference latency for a model.
    
    Args:
        model: Trained model
        X: Input features
        n_iterations: Number of iterations for averaging
        
    Returns:
        Average latency in milliseconds
    """
    latencies = []
    
    for _ in range(n_iterations):
        start = time.time()
        _ = model.predict(X)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
    
    avg_latency = np.mean(latencies)
    logger.info(f"Average inference latency: {avg_latency:.2f}ms")
    
    return avg_latency








