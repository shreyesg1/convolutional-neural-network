"""
CNN Model Container

This module provides a container class for managing neural network layers,
handling forward and backward passes, and tracking parameters and gradients.
"""

import time
from typing import List, Dict, Any
import numpy as np


class CNNModel:
    """
    Convolutional Neural Network Model Container

    A sequential model container that manages a list of layers and coordinates
    forward and backward passes through the network. Supports training/evaluation
    modes and optional performance profiling.

    Attributes:
        layers: List of layer objects, each with forward() and backward() methods
        train_mode: Boolean indicating training (True) or evaluation (False) mode
        runtimes: If True, print timing information for each layer
    """

    def __init__(self, layers: List[Any], runtimes: bool = False):
        """
        Initialize the CNN model

        Args:
            layers: List of layer objects. Each layer should implement:
                   - forward(x): Forward pass method
                   - backward(grad): Backward pass method
                   - params() (optional): Return dict of parameters
                   - grads() (optional): Return dict of gradients
            runtimes: If True, print execution time for each layer's forward/backward pass
        """
        self.layers = layers
        self.train_mode = True
        self.runtimes = runtimes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers

        Passes input through each layer sequentially. Layers with a 'training'
        attribute (e.g., Dropout) are automatically configured based on train_mode.

        Args:
            x: Input tensor, typically shape (batch_size, channels, height, width)

        Returns:
            Output tensor after passing through all layers
        """
        for i, layer in enumerate(self.layers):
            # Configure layer mode for training-dependent layers (e.g., Dropout)
            if hasattr(layer, 'training'):
                layer.training = self.train_mode

            # Time the forward pass if profiling is enabled
            if self.runtimes:
                start = time.time()
                x = layer.forward(x)
                elapsed = time.time() - start
                print(f"Layer {i:2d} ({layer.__class__.__name__:15s}) forward:  {elapsed:.6f}s")
            else:
                x = layer.forward(x)

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers in reverse order

        Propagates gradients backward through the network using backpropagation.
        Each layer computes gradients with respect to its inputs and parameters.

        Args:
            grad: Gradient from the loss function, shape matches forward output

        Returns:
            Gradient with respect to the original input
        """
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i

            # Time the backward pass if profiling is enabled
            if self.runtimes:
                start = time.time()
                grad = layer.backward(grad)
                elapsed = time.time() - start
                print(f"Layer {layer_idx:2d} ({layer.__class__.__name__:15s}) backward: {elapsed:.6f}s")
            else:
                grad = layer.backward(grad)

        return grad

    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Get all trainable parameters from the model

        Collects parameters from all layers that have a params() method.
        Parameters are keyed by "{layer_index}_{param_name}" for uniqueness.

        Returns:
            Dictionary mapping parameter names to parameter arrays

        Example:
            {'0_W': array(...), '0_b': array(...), '3_W': array(...), ...}
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for name, param in layer.params().items():
                    params[f"{i}_{name}"] = param
        return params

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        """
        Get all parameter gradients from the model

        Collects gradients from all layers that have a grads() method.
        Gradients are keyed by "{layer_index}_{param_name}" to match parameters().

        Returns:
            Dictionary mapping parameter names to gradient arrays

        Example:
            {'0_W': array(...), '0_b': array(...), '3_W': array(...), ...}
        """
        grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'grads'):
                for name, grad in layer.grads().items():
                    grads[f"{i}_{name}"] = grad
        return grads

    def train(self) -> None:
        """
        Set model to training mode

        Affects layers like Dropout and BatchNorm that behave differently
        during training vs. evaluation.
        """
        self.train_mode = True

    def eval(self) -> None:
        """
        Set model to evaluation mode

        Disables training-specific behaviors like dropout.
        """
        self.train_mode = False

    def summary(self) -> None:
        """
        Print a summary of the model architecture

        Displays layer types, output shapes (if available), and parameter counts.
        """
        print("=" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':>15}")
        print("=" * 80)

        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = f"{i}: {layer.__class__.__name__}"

            # Count parameters for this layer
            if hasattr(layer, 'params'):
                layer_params = sum(p.size for p in layer.params().values())
                total_params += layer_params
                param_str = f"{layer_params:,}"
            else:
                param_str = "0"

            # Try to infer output shape (not always possible)
            output_shape = "N/A"

            print(f"{layer_name:<30} {output_shape:<20} {param_str:>15}")

        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print("=" * 80)

    def save_weights(self, filepath: str) -> None:
        """
        Save model weights to file

        Args:
            filepath: Path to save the weights
        """
        params = self.parameters()
        np.savez(filepath, **params)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from file

        Args:
            filepath: Path to the saved weights file
        """
        loaded = np.load(filepath)
        params = self.parameters()

        for key in params.keys():
            if key in loaded:
                params[key][:] = loaded[key]

        print(f"Model weights loaded from {filepath}")