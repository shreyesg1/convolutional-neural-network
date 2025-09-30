"""
Max Pooling Layer

This module implements 2D max pooling, a downsampling operation that selects
the maximum value from each pooling window.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple


class MaxPooling2D:
    """
    2D Max Pooling Layer

    Performs spatial downsampling by taking the maximum value over non-overlapping
    or overlapping windows. This reduces spatial dimensions while preserving the
    most prominent features.

    Uses stride tricks for efficient windowing without copying data.

    Attributes:
        kernel_size: Size of the pooling window
        stride: Step size for sliding the pooling window
        input_shape: Shape of input tensor, cached for backward pass
        cols: Strided view of input windows
        cols_reshaped: Reshaped columns for max operation
        argmax: Indices of maximum values in each window
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        """
        Initialize the MaxPooling2D layer

        Args:
            kernel_size: Size of the pooling window (assumes square window)
            stride: Step size for moving the pooling window
        """
        self.kernel_size = kernel_size
        self.stride = stride

        # Cache for backward pass
        self.input_shape = None
        self.cols = None
        self.cols_reshaped = None
        self.argmax = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the max pooling layer

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Pooled output of shape (batch_size, channels, out_height, out_width)
            where out_height = (height - kernel_size) // stride + 1
            and out_width = (width - kernel_size) // stride + 1
        """
        self.input_shape = x.shape
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride

        # Calculate output dimensions for max pooling
        # Mathematical formula: out_size = (in_size - kernel_size) // stride + 1
        # This computes how many non-overlapping (or overlapping) windows fit in the input
        # For stride=2, kernel=2: (32-2)//2+1 = 16 (exactly half the size)
        # For stride=3, kernel=3: (13-3)//3+1 = 4 (reduces by factor of ~3)
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1

        # Create strided view of input to extract pooling windows efficiently
        # This creates a 6D view where each (out_h, out_w) position contains a (k, k) window
        # The stride parameter controls the spacing between windows
        shape = (B, C, out_h, out_w, k, k)
        strides = (
            x.strides[0],        # Batch stride: move to next sample
            x.strides[1],        # Channel stride: move to next channel
            s * x.strides[2],    # Height stride: move by stride*sample_height_stride
            s * x.strides[3],    # Width stride: move by stride*sample_width_stride
            x.strides[2],        # Kernel height stride: move within window
            x.strides[3],        # Kernel width stride: move within window
        )
        self.cols = as_strided(x, shape=shape, strides=strides)

        # Reshape for efficient max operation
        # Flatten the last two dimensions (kernel_h, kernel_w) into a single dimension
        # This allows us to use np.max() and np.argmax() efficiently
        self.cols_reshaped = self.cols.reshape(B, C, out_h, out_w, -1)

        # Store indices of max values for backward pass
        # argmax[i,j,k,l] gives the index of the maximum value in window (i,j,k)
        # This is crucial for backpropagation - only the max position gets the gradient
        self.argmax = np.argmax(self.cols_reshaped, axis=-1)

        # Take maximum over pooling window
        # Mathematical operation: output[i,j,k] = max(window[i,j,k,:])
        # This reduces spatial dimensions while preserving the most prominent features
        out = np.max(self.cols_reshaped, axis=-1)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the max pooling layer

        Gradients are routed only to the positions that contained the maximum
        values during the forward pass. All other positions receive zero gradient.

        Args:
            d_out: Gradient from next layer, shape (batch_size, channels, out_h, out_w)

        Returns:
            Gradient w.r.t. input, shape (batch_size, channels, height, width)
        """
        B, C, out_h, out_w = d_out.shape
        k = self.kernel_size
        s = self.stride

        # Initialize gradient array for pooling windows
        # Start with zeros - only max positions will receive gradients
        d_cols = np.zeros_like(self.cols_reshaped)

        # Distribute gradients only to max value positions
        # This is the key insight: max pooling is not differentiable everywhere,
        # but we can define the gradient as flowing only to the maximum position
        flat_indices = self.argmax.reshape(-1)  # Flatten all argmax indices
        d_cols_flat = d_cols.reshape(-1, k * k)  # Flatten gradient array
        d_out_flat = d_out.reshape(-1)  # Flatten output gradients

        # Set gradient at max position for each window
        # Mathematical formula: dL/dx[i,j] = dL/dy[pool_pos] if x[i,j] was the max in its window, else 0
        # This implements the subgradient of the max function
        d_cols_flat[np.arange(d_cols_flat.shape[0]), flat_indices] = d_out_flat

        # Reshape back to window structure
        # Convert from flattened form back to (batch, channel, out_h, out_w, kernel_h, kernel_w)
        d_cols = d_cols_flat.reshape(B, C, out_h, out_w, k, k)

        # Accumulate gradients back to input positions
        # This is the reverse of the strided view operation
        # Multiple windows may overlap (if stride < kernel_size), so gradients are summed
        # This implements the "col2im" operation - converting from column format back to image format
        dx = np.zeros(self.input_shape)
        for i in range(out_h):
            for j in range(out_w):
                # Map each output position back to its corresponding input region
                # The input region starts at (i*s, j*s) and has size (k, k)
                dx[:, :, i * s:i * s + k, j * s:j * s + k] += d_cols[:, :, i, j, :, :]

        return dx