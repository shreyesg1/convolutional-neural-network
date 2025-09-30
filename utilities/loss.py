"""
Loss Functions

This module implements common loss functions for neural network training.
"""

import numpy as np
from typing import Union


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for Multi-class Classification

    Measures the dissimilarity between predicted probability distributions
    and true class labels. Lower loss indicates better predictions.

    This is a standalone loss function that expects probability distributions
    as input (after softmax). For numerical stability when combined with softmax,
    consider using the Softmax class which combines both operations.

    Formula: L = -mean(log(p_correct_class))

    Attributes:
        y_pred: Cached predicted probabilities for backward pass
        y_true: Cached true labels for backward pass
    """

    def __init__(self):
        """Initialize the CrossEntropyLoss"""
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: Union[np.ndarray, int]) -> float:
        """
        Compute cross-entropy loss

        Args:
            y_pred: Predicted probability distributions, shape (batch_size, num_classes)
                   Values should be in [0, 1] and sum to 1 along axis=1
            y_true: True class labels, either:
                   - Class indices: shape (batch_size,) with integer values in [0, num_classes)
                   - One-hot encoded: shape (batch_size, num_classes)

        Returns:
            Scalar loss value averaged over the batch

        Note:
            Clipping is applied to prevent log(0) which would result in NaN
        """
        # Clip probabilities to avoid log(0) and log(1) numerical issues
        # Mathematical reason: log(0) = -∞ and log(1) = 0, both cause numerical problems
        # We clip to [ε, 1-ε] where ε is very small (1e-15) to prevent these issues
        # This ensures log(p) is always finite and well-behaved
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Convert one-hot encoded labels to indices if necessary
        # Cross-entropy loss works with class indices, not one-hot vectors
        if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
            self.y_true = np.argmax(y_true, axis=1)
        else:
            self.y_true = y_true

        batch_size = y_pred.shape[0]

        # Extract probabilities for correct classes
        # For each sample, get the probability assigned to its true class
        # This uses advanced indexing: y_pred[i, y_true[i]] for each sample i
        correct_probs = self.y_pred[np.arange(batch_size), self.y_true]

        # Compute negative log-likelihood (cross-entropy)
        # Mathematical formula: L = -mean(log(p_correct_class))
        # This measures how surprised we are by the true labels given our predictions
        # Lower loss means we're less surprised (better predictions)
        loss = -np.mean(np.log(correct_probs))

        return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss with respect to predictions

        The gradient of cross-entropy loss with respect to the predicted
        probabilities is: (y_pred - y_true) / batch_size

        Returns:
            Gradient w.r.t. predictions, shape (batch_size, num_classes)

        Note:
            This assumes y_pred contains probabilities (post-softmax).
            The gradient formula is simplified when softmax + cross-entropy
            are combined.
        """
        batch_size = self.y_pred.shape[0]

        # Initialize gradient as copy of predictions
        # This implements the gradient of cross-entropy loss w.r.t. predicted probabilities
        grad = self.y_pred.copy()

        # Subtract 1 from the correct class positions
        # Mathematical formula: dL/dp_i = p_i - y_i where y_i is one-hot true label
        # For correct class: gradient = p_correct - 1
        # For incorrect classes: gradient = p_incorrect - 0 = p_incorrect
        # This is the same as the softmax + cross-entropy combined gradient
        grad[np.arange(batch_size), self.y_true] -= 1

        # Average over batch
        # This gives the mean gradient across all samples in the batch
        grad /= batch_size

        return grad

    def __call__(self, y_pred: np.ndarray, y_true: Union[np.ndarray, int]) -> float:
        """
        Allow calling the loss as a function

        Args:
            y_pred: Predicted probability distributions
            y_true: True class labels

        Returns:
            Scalar loss value
        """
        return self.forward(y_pred, y_true)