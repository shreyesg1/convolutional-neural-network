"""
Softmax Activation with Cross-Entropy Loss

This module implements the softmax activation function combined with
cross-entropy loss, commonly used as the final layer in classification networks.
"""

import numpy as np
from typing import Union


class Softmax:
    """
    Softmax Activation with Cross-Entropy Loss

    Combines softmax activation and cross-entropy loss for numerical stability
    and computational efficiency. The softmax function converts raw logits into
    probability distributions, and cross-entropy measures the difference between
    predicted and true distributions.

    The combined gradient is particularly simple: (predictions - targets) / batch_size

    Attributes:
        logits: Raw model outputs before softmax
        labels: True class labels (either indices or one-hot encoded)
        probs: Softmax probabilities
    """

    def __init__(self):
        """Initialize the Softmax layer"""
        self.logits = None
        self.labels = None
        self.probs = None

    def forward(self, logits: np.ndarray, labels: Union[np.ndarray, int]) -> float:
        """
        Forward pass: compute softmax probabilities and cross-entropy loss

        Args:
            logits: Raw model outputs (pre-activation), shape (batch_size, num_classes)
            labels: True labels, either:
                   - Class indices: shape (batch_size,) with integer values in [0, num_classes)
                   - One-hot encoded: shape (batch_size, num_classes)

        Returns:
            Scalar cross-entropy loss averaged over the batch

        Note:
            Uses the log-sum-exp trick for numerical stability by subtracting max logits
        """
        self.logits = logits

        # Convert one-hot labels to class indices if necessary
        if labels.ndim == 2:
            self.labels = np.argmax(labels, axis=1)
        else:
            self.labels = labels

        # Numerical stability: subtract max logits before exp
        # This is the "log-sum-exp trick" to prevent overflow in softmax computation
        # Mathematical identity: softmax(x) = softmax(x - c) for any constant c
        # We choose c = max(x) to shift all logits to be ≤ 0, preventing exp() overflow
        # Original formula: p_i = exp(x_i) / sum_j(exp(x_j))
        # Stable formula: p_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        
        # Normalize to get probabilities that sum to 1
        # This ensures sum(p_i) = 1 for each sample, making it a valid probability distribution
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy loss: -log(p_correct_class)
        # Mathematical formula: L = -sum_i y_i * log(p_i) where y_i is one-hot encoded true label
        # For single correct class: L = -log(p_correct_class)
        # This measures the dissimilarity between predicted and true distributions
        # Lower loss means better predictions (higher probability for correct class)
        
        # Add small epsilon to prevent log(0) which would cause numerical issues
        # The epsilon (1e-9) is small enough to not affect the loss significantly
        log_probs = -np.log(self.probs[np.arange(len(logits)), self.labels] + 1e-9)
        loss = np.mean(log_probs)  # Average loss over the batch

        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass: compute gradient of loss with respect to logits

        The combined softmax + cross-entropy gradient simplifies to:
            dL/dlogits = (probs - one_hot(labels)) / batch_size

        Returns:
            Gradient w.r.t. logits, shape (batch_size, num_classes)
        """
        batch_size = self.logits.shape[0]

        # Combined softmax + cross-entropy gradient has a beautiful simplification
        # Mathematical derivation:
        # For softmax: p_i = exp(x_i) / sum_j(exp(x_j))
        # For cross-entropy: L = -log(p_correct)
        # Combined gradient: dL/dx_i = p_i - y_i where y_i is one-hot true label
        # 
        # This means: gradient = predicted_probabilities - true_labels
        # For correct class: gradient = p_correct - 1
        # For incorrect classes: gradient = p_incorrect - 0 = p_incorrect
        
        # Start with softmax probabilities
        dx = self.probs.copy()

        # Subtract 1 from the true class probabilities
        # This implements the "p_i - y_i" formula where y_i = 1 for correct class, 0 for others
        dx[np.arange(batch_size), self.labels] -= 1

        # Average over batch
        # This gives the mean gradient across all samples in the batch
        dx /= batch_size

        return dx

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """
        Get class predictions from logits

        Args:
            logits: Raw model outputs, shape (batch_size, num_classes)

        Returns:
            Predicted class indices, shape (batch_size,)
        """
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Get probability distributions from logits

        Args:
            logits: Raw model outputs, shape (batch_size, num_classes)

        Returns:
            Softmax probabilities, shape (batch_size, num_classes)
        """
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs