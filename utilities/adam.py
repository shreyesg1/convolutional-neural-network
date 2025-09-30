"""
Adam Optimizer

This module implements the Adam (Adaptive Moment Estimation) optimizer,
an algorithm for first-order gradient-based optimization of stochastic
objective functions.

Reference: Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
"""

import numpy as np
from typing import Dict, Optional


class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) Optimizer

    Combines ideas from RMSprop and momentum by computing adaptive learning rates
    for each parameter using estimates of first and second moments of the gradients.

    The algorithm maintains running averages of both the gradients (first moment)
    and the squared gradients (second moment), with bias correction for the early
    stages of training.

    Attributes:
        model: Neural network model with parameters() and grads attributes
        lr: Learning rate (step size)
        beta1: Exponential decay rate for first moment estimates (momentum)
        beta2: Exponential decay rate for second moment estimates (RMSprop)
        epsilon: Small constant for numerical stability
        t: Current time step (number of updates performed)
        m: First moment estimates (momentum) for each parameter
        v: Second moment estimates (uncentered variance) for each parameter
    """

    def __init__(self, model: object, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer

        Args:
            model: Model object with parameters() method returning dict of parameters
                  and grads attribute containing corresponding gradients
            lr: Learning rate (default: 0.001, original paper recommendation)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small constant to prevent division by zero (default: 1e-8)
        """
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step counter

        # Initialize moment estimate dictionaries
        self.m = {}  # First moment (mean of gradients)
        self.v = {}  # Second moment (uncentered variance of gradients)

        # Initialize moments to zero for each parameter
        for name, param in self.model.parameters().items():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def update(self) -> None:
        """
        Perform a single optimization step

        Updates all model parameters using the Adam update rule:
        1. Compute biased moment estimates
        2. Apply bias correction
        3. Update parameters using corrected moments

        Note: This implementation includes a learning rate schedule that reduces
        the learning rate by 10x after 1170 iterations (10 * 117). Consider
        making this configurable for production use.
        """
        self.t += 1

        # Learning rate schedule (hardcoded - consider making configurable)
        if self.t == 10 * 117:
            print("Lowering learning rate by 10x")
            self.lr /= 10

        # Update each parameter
        for name, param in self.model.parameters().items():
            grad = self.model.grads[name]

            # Update biased first moment estimate (momentum)
            # Mathematical formula: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            # This computes an exponentially weighted moving average of gradients
            # β₁ controls the decay rate (typically 0.9): higher values = more memory of past gradients
            # The bias comes from initializing m_0 = 0, making early estimates biased toward zero
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second moment estimate (RMSprop)
            # Mathematical formula: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            # This computes an exponentially weighted moving average of squared gradients
            # β₂ controls the decay rate (typically 0.999): higher values = more memory of past squared gradients
            # The bias comes from initializing v_0 = 0, making early estimates biased toward zero
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moment estimates
            # Corrects for initialization bias towards zero in early iterations
            # Mathematical formulas:
            # m̂_t = m_t / (1 - β₁ᵗ)  # Bias correction for first moment
            # v̂_t = v_t / (1 - β₂ᵗ)  # Bias correction for second moment
            # 
            # Why bias correction is needed:
            # - m_t = (1-β₁) * Σ(i=0 to t) β₁^(t-i) * g_i
            # - E[m_t] = (1-β₁) * Σ(i=0 to t) β₁^(t-i) * E[g_i] ≈ (1-β₁^t) * E[g]
            # - So m̂_t = m_t / (1-β₁^t) gives an unbiased estimate of E[g]
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters using Adam rule
            # Mathematical formula: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
            # This combines momentum (m̂_t) with adaptive learning rates (1/√v̂_t)
            # The ε term prevents division by zero and ensures numerical stability
            # The square root of v̂_t gives the adaptive learning rate per parameter
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self) -> None:
        """
        Reset all gradients to zero

        Should be called before each backward pass to prevent gradient accumulation
        across iterations. Iterates through model layers and zeros out gradient
        dictionaries.
        """
        for layer in self.model.layers:
            if hasattr(layer, 'grads'):
                grads_dict = layer.grads()
                for name in grads_dict:
                    grads_dict[name][:] = 0

    def get_lr(self) -> float:
        """
        Get current learning rate

        Returns:
            Current learning rate value
        """
        return self.lr

    def set_lr(self, lr: float) -> None:
        """
        Set learning rate

        Args:
            lr: New learning rate value
        """
        self.lr = lr

    def state_dict(self) -> Dict:
        """
        Get optimizer state for checkpointing

        Returns:
            Dictionary containing optimizer state (moments, time step, hyperparameters)
        """
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            't': self.t,
            'm': {k: v.copy() for k, v in self.m.items()},
            'v': {k: v.copy() for k, v in self.v.items()}
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load optimizer state from checkpoint

        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.lr = state_dict['lr']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.epsilon = state_dict['epsilon']
        self.t = state_dict['t']
        self.m = {k: v.copy() for k, v in state_dict['m'].items()}
        self.v = {k: v.copy() for k, v in state_dict['v'].items()}