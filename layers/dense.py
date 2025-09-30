"""
Fully Connected Dense Layer

This module implements a standard fully connected (dense) layer with optional
activation functions and Xavier weight initialization.
"""

import numpy as np
from typing import Optional, Dict


class Dense:
    """
    Fully Connected Dense Layer

    Implements a standard neural network dense layer that performs the transformation:
    output = activation(x @ W + b)

    Uses Xavier initialization for weights to help maintain gradient magnitudes
    during training.

    Attributes:
        input_dim: Number of input features
        units: Number of output units
        activation: Optional activation function with forward() and backward() methods
        W: Weight matrix of shape (input_dim, units)
        b: Bias vector of shape (units,)
    """

    def __init__(self, input_dim: int, units: int, activation: Optional[object] = None):
        """
        Initialize the Dense layer

        Args:
            input_dim: Number of input features
            units: Number of output units
            activation: Optional activation function object with forward() and backward() methods
        """
        self.input_dim = input_dim
        self.units = units
        self.activation = activation

        # Xavier/Glorot initialization for better gradient flow
        # Mathematical formula: limit = sqrt(6 / (fan_in + fan_out))
        # This ensures the variance of outputs is approximately equal to the variance of inputs
        # For a linear layer: fan_in = input_dim, fan_out = units
        # The factor 6 comes from the uniform distribution range [-limit, limit] having variance = limit²/3
        # We want variance = 1, so limit²/3 = 1, hence limit = sqrt(3) ≈ sqrt(6/2) for symmetric fan_in/fan_out
        limit = np.sqrt(6 / (input_dim + units))
        self.W = np.random.uniform(-limit, limit, (input_dim, units))
        self.b = np.zeros(units)

        # Cache for backward pass
        self.x = None
        self.z = None
        self.out = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the dense layer

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, units)
        """
        self.x = x
        # Linear transformation: z = xW + b
        # Mathematical formula: z[i,j] = sum(k) x[i,k] * W[k,j] + b[j]
        # This computes the weighted sum of inputs for each output unit
        # x: (batch_size, input_dim), W: (input_dim, units), b: (units,)
        # Result: z: (batch_size, units)
        self.z = x @ self.W + self.b  # Linear transformation

        # Apply activation if provided
        # The activation function introduces non-linearity: output = activation(z)
        # Common activations: ReLU(z) = max(0,z), sigmoid(z) = 1/(1+e^(-z)), tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
        if self.activation:
            self.out = self.activation.forward(self.z)
        else:
            self.out = self.z

        return self.out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass through the dense layer

        Computes gradients with respect to weights, biases, and input.
        Note: This method computes but does NOT update weights. Use an optimizer
        to update parameters based on the computed gradients.

        Args:
            d_out: Gradient from next layer, shape (batch_size, units)

        Returns:
            Gradient with respect to input, shape (batch_size, input_dim)
        """
        # Backpropagate through activation if present
        # If there's an activation function, we need to multiply by its derivative
        # Chain rule: dL/dz = dL/dout * dout/dz
        if self.activation:
            d_out = self.activation.backward(d_out)

        batch_size = self.x.shape[0]

        # Compute gradients w.r.t. parameters using chain rule
        # For the linear transformation z = xW + b:
        # dL/dW = dL/dz * dz/dW = dL/dz * x^T
        # dL/db = dL/dz * dz/db = dL/dz (since dz/db = 1)
        
        # Weight gradient: dL/dW = X^T @ dL/dZ
        # Mathematical derivation: dL/dW[i,j] = sum(k) dL/dz[k,j] * x[k,i]
        # This is implemented as matrix multiplication: X^T @ dL/dZ
        # Division by batch_size gives the average gradient over the batch
        self.dW = self.x.T @ d_out / batch_size

        # Bias gradient: dL/db = mean(dL/dZ) across batch
        # Mathematical derivation: dL/db[j] = sum(k) dL/dz[k,j] * 1 = sum(k) dL/dz[k,j]
        # Since bias affects all samples equally, we sum over the batch dimension
        self.db = np.mean(d_out, axis=0)

        # Compute gradient w.r.t. input using chain rule
        # dL/dX = dL/dZ @ W^T
        # Mathematical derivation: dL/dx[i,j] = sum(k) dL/dz[i,k] * W[j,k]
        # This is implemented as matrix multiplication: dL/dZ @ W^T
        d_x = d_out @ self.W.T

        return d_x

    def params(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters

        Returns:
            Dictionary containing weights (W) and biases (b)
        """
        return {'W': self.W, 'b': self.b}

    def grads(self) -> Dict[str, np.ndarray]:
        """
        Get parameter gradients

        Returns:
            Dictionary containing gradients for weights (W) and biases (b)
        """
        return {'W': self.dW, 'b': self.db}