"""
Flatten Layer

This module implements a flatten layer that reshapes multi-dimensional inputs
into 2D tensors suitable for fully connected layers.
"""

import numpy as np
from typing import Tuple


class Flatten:
    """
    Flatten Layer

    Reshapes multi-dimensional input tensors into 2D tensors while preserving
    the batch dimension. Commonly used as a bridge between convolutional and
    dense layers.

    Example:
        Input shape: (batch_size, channels, height, width) = (32, 64, 8, 8)
        Output shape: (batch_size, channels * height * width) = (32, 4096)

    Attributes:
        orig_shape: Original input shape, cached for backward pass
    """

    def __init__(self):
        """Initialize the Flatten layer"""
        self.orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the flatten layer

        Reshapes input to (batch_size, -1), flattening all dimensions except batch.

        Args:
            x: Input tensor of shape (batch_size, dim1, dim2, ..., dimN)

        Returns:
            Flattened tensor of shape (batch_size, dim1 * dim2 * ... * dimN)
        """
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the flatten layer

        Reshapes gradient back to original input shape.

        Args:
            d_out: Gradient from next layer, shape (batch_size, flattened_dim)

        Returns:
            Gradient reshaped to original input shape
        """
        return d_out.reshape(self.orig_shape)