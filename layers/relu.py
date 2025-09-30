"""
ReLU Activation Function

This module implements the Rectified Linear Unit (ReLU) activation function,
one of the most commonly used activation functions in neural networks.
"""

import numpy as np


class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation

    Applies the element-wise activation function:
        ReLU(x) = max(0, x)

    This introduces non-linearity while being computationally efficient and
    helping to mitigate the vanishing gradient problem.

    Attributes:
        mask: Binary mask indicating positive values (x > 0), cached for backward pass
    """

    def __init__(self):
        """Initialize the ReLU activation"""
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ReLU activation

        Sets all negative values to zero while keeping positive values unchanged.

        Args:
            x: Input tensor of any shape

        Returns:
            Activated output of same shape as input, with f(x) = max(0, x)
        """
        # ReLU function: f(x) = max(0, x)
        # Mathematical definition: f(x) = x if x > 0, else 0
        # This creates a binary mask indicating which elements are positive
        self.mask = (x > 0)
        
        # Apply ReLU by element-wise multiplication with the mask
        # This is equivalent to: f(x) = x * (x > 0)
        # For positive values: x * 1 = x, for negative values: x * 0 = 0
        return x * self.mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through ReLU activation

        The gradient is 1 for positive inputs and 0 for negative inputs:
            df/dx = 1 if x > 0, else 0

        Args:
            d_out: Gradient from next layer, same shape as forward output

        Returns:
            Gradient w.r.t. input, same shape as d_out
        """
        # ReLU derivative: f'(x) = 1 if x > 0, else 0
        # Mathematical definition: f'(x) = 1 for x > 0, f'(x) = 0 for x ≤ 0
        # The derivative is discontinuous at x = 0, but we define f'(0) = 0
        # 
        # Chain rule: dL/dx = dL/dy * dy/dx = dL/dy * f'(x)
        # Since f'(x) = mask, we simply multiply by the mask
        # This routes gradients only through the positive inputs
        return d_out * self.mask