"""
Dropout Regularization Layer

This module implements dropout, a regularization technique that randomly
sets a fraction of input units to zero during training to prevent overfitting.
"""

import numpy as np
from typing import Optional


class Dropout:
    """
    Dropout Layer

    Randomly sets a fraction of input units to zero during training while scaling
    the remaining units to maintain expected output magnitude. During evaluation,
    no dropout is applied.

    This implements the "inverted dropout" technique where scaling happens during
    training rather than at test time for efficiency.

    Attributes:
        rate: Fraction of inputs to drop (between 0 and 1)
        mask: Binary mask indicating which units to keep (used during training)
        training: Boolean flag indicating whether layer is in training mode
    """

    def __init__(self, dropout_rate: float):
        """
        Initialize the Dropout layer

        Args:
            dropout_rate: Fraction of input units to drop, must be in [0, 1)

        Raises:
            ValueError: If dropout_rate is not in [0, 1)
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {dropout_rate}")

        self.rate = dropout_rate
        self.mask = None
        self.training = True  # Default to training mode

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the dropout layer

        During training: Randomly drops units and scales remaining units by 1/(1-rate)
        During evaluation: Returns input unchanged

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor of same shape as input
        """
        if self.training:
            # Generate binary mask: 1 for kept units, 0 for dropped units
            # Mathematical formula: mask[i] = 1 with probability (1-rate), 0 with probability rate
            # This creates a Bernoulli random variable for each element
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
            
            # Inverted dropout: scale by 1/(1-rate) during training
            # This maintains the expected value: E[output] = E[x * mask] = E[x] * E[mask] = E[x] * 1 = E[x]
            # Without scaling: E[output] = E[x] * (1-rate), which would reduce the signal
            # The scaling factor 1/(1-rate) compensates for the dropped units
            return x * self.mask
        else:
            # No dropout during evaluation
            # During inference, we want the full signal without any random dropping
            # The scaling was already applied during training, so no additional scaling needed
            return x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the dropout layer

        Gradients only flow through units that were kept during forward pass.

        Args:
            d_out: Gradient from next layer, same shape as forward output

        Returns:
            Gradient with respect to input, same shape as d_out
        """
        if self.training:
            # Only propagate gradients through non-dropped units
            # Mathematical formula: dL/dx = dL/dy * dy/dx = dL/dy * mask
            # This implements the gradient of the dropout function
            # Gradients only flow through units that were kept during forward pass
            # The scaling factor 1/(1-rate) is already included in the mask
            return d_out * self.mask
        else:
            # During evaluation, no dropout is applied, so gradients pass through unchanged
            return d_out

    def set_training(self, training: bool) -> None:
        """
        Set the layer mode

        Args:
            training: If True, enable dropout. If False, disable dropout
        """
        self.training = training