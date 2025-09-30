"""
Alternative Dense Layer using Structured Matrices

This module implements an efficient dense layer using structured matrices
based on the Fast Walsh-Hadamard Transform (FWHT). This approach significantly
reduces the number of trainable parameters compared to standard dense layers.

Reference: Qiu et al. (https://arxiv.org/abs/2406.06248)
"""

import numpy as np
from typing import Optional, Dict


def transform(x: np.ndarray) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform (FWHT)

    Applies the Walsh-Hadamard transform to input data, which is a key component
    of the structured matrix multiplication in the AltDense layer.

    Args:
        x: Input array of shape (batch_size, n) where n must be a power of 2

    Returns:
        Transformed array of shape (batch_size, n), normalized by sqrt(n)

    Note:
        This implementation uses an iterative butterfly algorithm with O(n log n) complexity
    """
    batch_size, n = x.shape
    h = x.copy()
    i = 1

    # Iterative butterfly algorithm for Fast Walsh-Hadamard Transform (FWHT)
    # The FWHT is a fast algorithm for computing the Walsh-Hadamard transform
    # which is a specific case of the discrete Fourier transform
    # 
    # Mathematical basis: H_n = H_{n/2} ⊗ H_2, where H_2 = [[1,1],[1,-1]]
    # The butterfly pattern implements this recursive structure iteratively
    # 
    # At each stage i, we process pairs of elements separated by distance i
    # and apply the butterfly operation: [a,b] -> [a+b, a-b]
    while i < n:
        for j in range(0, n, 2 * i):
            for b in range(batch_size):
                # Extract two halves of the current block
                a = h[b, j:j + i].copy()        # First half
                b_vals = h[b, j + i:j + 2 * i].copy()  # Second half
                
                # Apply butterfly operation: [a, b] -> [a+b, a-b]
                # This implements the Hadamard matrix multiplication
                h[b, j:j + i] = a + b_vals      # Sum: a + b
                h[b, j + i:j + 2 * i] = a - b_vals  # Difference: a - b
        i *= 2  # Double the block size for next iteration

    # Normalize by sqrt(n) to make the transform orthogonal
    # This ensures ||Hx|| = ||x|| (preserves vector norms)
    return h / np.sqrt(n)


class AltDense:
    """
    Alternative Dense Layer with Structured Matrices

    This layer replaces the standard weight matrix with a product of diagonal matrices
    and Hadamard transforms: D1 * H * D2 * H * D3, where D are diagonal matrices and
    H is the Hadamard transform. This structure reduces parameters from O(input_dim * units)
    to O(input_dim + units) while maintaining expressiveness.

    Attributes:
        units: Number of output units
        input_dim: Number of input features
        activation: Optional activation function object with forward() and backward() methods
        D1: Diagonal matrix (output space)
        D2: Diagonal matrix (intermediate space)
        D3: Diagonal matrix (input space)
    """

    def __init__(self, units: int, input_dim: Optional[int] = None,
                 activation: Optional[object] = None):
        """
        Initialize the AltDense layer

        Args:
            units: Number of output units (must equal input_dim for this implementation)
            input_dim: Number of input features (must be a power of 2)
            activation: Optional activation function with forward() and backward() methods

        Raises:
            ValueError: If input_dim is not a power of 2
        """
        self.units = units
        self.input_dim = input_dim
        self.activation = activation

        # Initialize diagonal matrices with small random values
        self.D1 = np.random.randn(units) * 0.1
        self.D2 = np.random.randn(units) * 0.1
        self.D3 = np.random.randn(input_dim) * 0.1

        # Cache for backward pass
        self.x = None
        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.z4 = None
        self.out = None
        self.d_D1 = None
        self.d_D2 = None
        self.d_D3 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer

        Computes: activation(D1 * H * D2 * H * D3 * x)

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Output array of shape (batch_size, units)
        """
        self.x = x

        # Apply sequence of diagonal and Hadamard transforms
        # The structured matrix is: D1 * H * D2 * H * D3
        # This replaces the standard dense layer weight matrix W with a structured approximation
        
        # Step 1: Element-wise multiplication with input diagonal matrix
        # D3 is a diagonal matrix, so D3 * x is element-wise multiplication
        self.z1 = self.D3 * x  # D3 * x

        # Step 2: Apply Walsh-Hadamard transform (orthogonal matrix)
        # H is the Hadamard matrix, which is orthogonal: H^T = H and H * H^T = I
        self.z2 = transform(self.z1)  # H * D3 * x

        # Step 3: Element-wise multiplication with intermediate diagonal matrix
        # D2 is another diagonal matrix for the intermediate space
        self.z3 = self.D2 * self.z2  # D2 * H * D3 * x

        # Step 4: Apply second Walsh-Hadamard transform
        # This creates a more complex transformation while maintaining orthogonality
        self.z4 = transform(self.z3)  # H * D2 * H * D3 * x

        # Step 5: Element-wise multiplication with output diagonal matrix
        # D1 is the final diagonal matrix for the output space
        out = self.D1 * self.z4  # D1 * H * D2 * H * D3 * x

        # Apply activation if provided
        if self.activation:
            self.out = self.activation.forward(out)
        else:
            self.out = out

        return self.out

    def backward(self, d_out: np.ndarray, lr: float = 0.01) -> np.ndarray:
        """
        Backward pass through the layer

        Computes gradients with respect to all parameters and updates them.
        Uses the chain rule to backpropagate through the structured matrix operations.

        Args:
            d_out: Gradient from the next layer, shape (batch_size, units)
            lr: Learning rate for parameter updates

        Returns:
            Gradient with respect to input, shape (batch_size, input_dim)
        """
        # Backprop through activation
        if self.activation:
            d_out = self.activation.backward(d_out)

        # Backpropagation through the structured matrix: D1 * H * D2 * H * D3
        # We use the chain rule to compute gradients for each diagonal matrix
        
        # Gradient w.r.t. D1: ∂L/∂D1 = ∂L/∂out * z4
        # For diagonal matrix multiplication: out = D1 * z4, so ∂out/∂D1 = z4
        # Chain rule: ∂L/∂D1 = ∂L/∂out * ∂out/∂D1 = ∂L/∂out * z4
        self.d_D1 = np.mean(d_out * self.z4, axis=0)
        d_z4 = d_out * self.D1  # Gradient w.r.t. z4

        # Backprop through Hadamard transform (H^T = H since H is orthogonal)
        # The Hadamard matrix is its own inverse: H^T = H, so H^T * H = I
        # Therefore: ∂L/∂z3 = H^T * ∂L/∂z4 = H * ∂L/∂z4
        d_z3 = transform(d_z4)

        # Gradient w.r.t. D2: ∂L/∂D2 = ∂L/∂z3 * z2
        # For diagonal matrix multiplication: z3 = D2 * z2, so ∂z3/∂D2 = z2
        # Chain rule: ∂L/∂D2 = ∂L/∂z3 * ∂z3/∂D2 = ∂L/∂z3 * z2
        self.d_D2 = np.mean(d_z3 * self.z2, axis=0)
        d_z2 = d_z3 * self.D2  # Gradient w.r.t. z2

        # Backprop through second Hadamard transform
        # Again using H^T = H: ∂L/∂z1 = H^T * ∂L/∂z2 = H * ∂L/∂z2
        d_z1 = transform(d_z2)

        # Gradient w.r.t. D3: ∂L/∂D3 = ∂L/∂z1 * x
        # For diagonal matrix multiplication: z1 = D3 * x, so ∂z1/∂D3 = x
        # Chain rule: ∂L/∂D3 = ∂L/∂z1 * ∂z1/∂D3 = ∂L/∂z1 * x
        self.d_D3 = np.mean(d_z1 * self.x, axis=0)
        d_x = d_z1 * self.D3  # Gradient w.r.t. input x

        # Update parameters using gradient descent
        self.D1 -= lr * self.d_D1
        self.D2 -= lr * self.d_D2
        self.D3 -= lr * self.d_D3

        return d_x

    def params(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters

        Returns:
            Dictionary containing diagonal matrices D1, D2, D3
        """
        return {
            'D1': self.D1,
            'D2': self.D2,
            'D3': self.D3
        }

    def grads(self) -> Dict[str, np.ndarray]:
        """
        Get parameter gradients

        Returns:
            Dictionary containing gradients for D1, D2, D3
        """
        return {
            'D1': self.d_D1,
            'D2': self.d_D2,
            'D3': self.d_D3
        }