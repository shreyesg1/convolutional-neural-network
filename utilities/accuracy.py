"""
Accuracy Metric

This module provides utility functions for computing classification accuracy.
"""

import numpy as np
from typing import Union


def accuracy(y_true: Union[np.ndarray, int], y_pred_probs: np.ndarray) -> float:
    """
    Calculate classification accuracy

    Computes the fraction of correctly predicted classes by comparing
    the predicted class (argmax of probabilities) with true labels.

    Args:
        y_true: True labels, either:
               - Class indices: shape (batch_size,) with integer values
               - One-hot encoded: shape (batch_size, num_classes)
        y_pred_probs: Predicted probability distributions, shape (batch_size, num_classes)

    Returns:
        Accuracy as a float in [0, 1], representing the fraction of correct predictions

    Example:
        >>> y_true = np.array([0, 1, 2, 1])
        >>> y_pred = np.array([[0.8, 0.1, 0.1],
        ...                     [0.2, 0.7, 0.1],
        ...                     [0.1, 0.2, 0.7],
        ...                     [0.3, 0.6, 0.1]])
        >>> accuracy(y_true, y_pred)
        1.0
    """
    # Convert one-hot encoded labels to class indices if necessary
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    # Get predicted classes from probability distributions
    preds = np.argmax(y_pred_probs, axis=1)

    # Calculate fraction of correct predictions
    return np.mean(preds == y_true)