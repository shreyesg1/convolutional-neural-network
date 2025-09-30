"""
2D Convolutional Layer

This module implements a 2D convolutional layer with efficient im2col-based
matrix multiplication for the forward and backward passes.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Tuple, Union


class Conv2D:
    """
    2D Convolutional Layer

    Applies 2D convolution over input images using learnable filters.
    Uses the im2col technique to transform the convolution operation into
    efficient matrix multiplication.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels (filters)
        kH: Kernel height
        kW: Kernel width
        weights: Convolutional filters of shape (out_channels, in_channels, kH, kW)
        biases: Bias terms of shape (out_channels, 1)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]]):
        """
        Initialize the Conv2D layer

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (number of filters)
            kernel_size: Size of the convolutional kernel, either an int for square
                        kernels or a tuple (height, width)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        # Initialize weights with small random values
        self.weights = np.random.randn(out_channels, in_channels, self.kH, self.kW) * 0.1
        self.biases = np.zeros((out_channels, 1))

        # Cache for backward pass
        self.input = None
        self.cols = None
        self.out_h = None
        self.out_w = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the convolutional layer

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
            where out_height = height - kH + 1 and out_width = width - kW + 1
        """
        self.input = x
        B, C, H, W = x.shape
        kH, kW = self.kH, self.kW

        # Calculate output dimensions (valid convolution, no padding)
        # Mathematical formula: output_size = input_size - kernel_size + 1
        # This is derived from the convolution operation where we slide the kernel
        # across the input, and the last valid position is at (input_size - kernel_size)
        out_h = H - kH + 1  # Height: 32 - 3 + 1 = 30
        out_w = W - kW + 1  # Width: 32 - 3 + 1 = 30
        self.out_h, self.out_w = out_h, out_w

        # Transform convolution into matrix multiplication using im2col
        # This is a key optimization: instead of nested loops, we reshape the convolution
        # into a matrix multiplication problem. The im2col (image-to-column) transformation
        # extracts all possible kernel-sized patches from the input and arranges them
        # as columns in a matrix.
        
        # Create sliding windows view of input using stride tricks
        # Shape: (batch, channels, out_h, out_w, kernel_h, kernel_w)
        # This creates a 6D view where each (out_h, out_w) position contains a (kH, kW) patch
        shape = (B, C, out_h, out_w, kH, kW)
        strides = (
            x.strides[0],  # Batch stride: move to next sample
            x.strides[1],  # Channel stride: move to next channel
            x.strides[2],  # Height stride: move to next row in output
            x.strides[3],  # Width stride: move to next column in output
            x.strides[2],  # Kernel height stride: move within kernel patch
            x.strides[3],  # Kernel width stride: move within kernel patch
        )
        cols = as_strided(x, shape=shape, strides=strides)
        
        # Reshape to (batch, C*kH*kW, out_h*out_w) for matrix multiplication
        # Each column represents one spatial position, flattened kernel patch
        # This transforms the 6D view into a 2D matrix suitable for multiplication
        cols = cols.reshape(B, C * kH * kW, out_h * out_w)
        self.cols = cols  # Cache for backward pass

        # Flatten filters and perform matrix multiplication
        # Reshape weights from (out_channels, in_channels, kH, kW) to (out_channels, in_channels*kH*kW)
        # This flattens each filter into a row vector for matrix multiplication
        W_flat = self.weights.reshape(self.out_channels, -1)  # (F, C*kH*kW)
        
        # Perform the actual convolution via matrix multiplication
        # W_flat @ cols computes: for each output channel, sum over all input channels and kernel positions
        # Mathematical formula: output[b,f,i,j] = sum(c,k,l) W[f,c,k,l] * input[b,c,i+k,j+l]
        # This is equivalent to: out[b,f,spatial] = sum(channel*kernel) W[f,channel*kernel] * cols[b,channel*kernel,spatial]
        out = W_flat @ cols  # (B, F, OH*OW)
        
        # Add bias with broadcasting: each output channel gets its own bias term
        # Broadcasting: (B, F, OH*OW) + (F, 1) = (B, F, OH*OW)
        out += self.biases  # Add bias with broadcasting

        # Reshape to proper output dimensions
        # Convert from (batch, channels, spatial) back to (batch, channels, height, width)
        out = out.reshape(B, self.out_channels, out_h, out_w)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the convolutional layer

        Computes gradients with respect to weights, biases, and input.

        Args:
            d_out: Gradient from next layer, shape (batch_size, out_channels, out_h, out_w)

        Returns:
            Gradient with respect to input, shape (batch_size, in_channels, height, width)
        """
        B, F, out_h, out_w = d_out.shape
        C, H, W = self.input.shape[1:]
        kH, kW = self.kH, self.kW

        d_out_flat = d_out.reshape(B, F, -1)  # (B, F, OH*OW)

        # Gradient w.r.t. weights: sum over batch and spatial dimensions
        # Mathematical formula: dL/dW[f,c,k,l] = sum(b,i,j) dL/dY[b,f,i,j] * X[b,c,i+k,j+l]
        # Using Einstein summation: 'bfo,bco->fc' means:
        # - b: sum over batch dimension
        # - f: output channel (first dimension of result)
        # - o: sum over spatial positions (output height * width)
        # - c: input channel (second dimension of result)
        # This computes the gradient for each weight by multiplying gradients with corresponding input patches
        d_weights = np.einsum('bfo,bco->fc', d_out_flat, self.cols)
        self.d_weights = d_weights.reshape(self.weights.shape)

        # Gradient w.r.t. biases: sum over batch and spatial dimensions
        # Mathematical formula: dL/db[f] = sum(b,i,j) dL/dY[b,f,i,j]
        # Since bias is added to all spatial positions, we sum the gradient over all positions
        # and all samples in the batch
        self.d_biases = d_out_flat.sum(axis=(0, 2)).reshape(self.biases.shape)

        # Gradient w.r.t. input: backpropagate through the im2col operation
        # This is the most complex part: we need to reverse the im2col transformation
        # and accumulate gradients at overlapping positions in the input
        
        W_flat = self.weights.reshape(F, -1)  # (F, C*kH*kW)
        d_input = np.zeros_like(self.input)  # (B, C, H, W)

        for b in range(B):
            # Compute gradient for each sample in batch
            # Mathematical formula: dL/dX[b,c,i,j] = sum(f,k,l) dL/dY[b,f,i-k,j-l] * W[f,c,k,l]
            # This is implemented as: d_col = W^T @ d_out_flat[b]
            # where W^T transforms gradients from output space back to input patch space
            d_col = W_flat.T @ d_out_flat[b]  # (C*kH*kW, OH*OW)

            # Reshape to match kernel dimensions
            # Convert from flattened patches back to (channels, kernel_h, kernel_w, out_h, out_w)
            d_col_reshaped = d_col.reshape(C, kH, kW, out_h, out_w)

            # Create strided view of d_input to accumulate gradients
            # This creates the same sliding window view as in forward pass, but for gradients
            shape = (C, out_h, out_w, kH, kW)
            strides = (
                d_input.strides[1],  # Channel stride
                d_input.strides[2],  # Height stride (output positions)
                d_input.strides[3],  # Width stride (output positions)
                d_input.strides[2],  # Kernel height stride
                d_input.strides[3],  # Kernel width stride
            )
            d_input_strided = as_strided(d_input[b], shape=shape, strides=strides)

            # Accumulate gradients at each overlapping position
            # Multiple output positions may contribute to the same input position,
            # so we need to sum all contributions (this is why we use +=)
            # Transpose: (C, kH, kW, out_h, out_w) -> (C, out_h, out_w, kH, kW)
            d_input_strided += d_col_reshaped.transpose(0, 3, 4, 1, 2)

        return d_input

    def params(self) -> dict:
        """
        Get layer parameters

        Returns:
            Dictionary containing weights and biases
        """
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def grads(self) -> dict:
        """
        Get parameter gradients

        Returns:
            Dictionary containing gradients for weights and biases
        """
        return {
            'weights': self.d_weights,
            'biases': self.d_biases
        }