"""
Padding Utilities

This module provides padding functions for preparing data, particularly
for operations that require power-of-2 dimensions (e.g., Walsh-Hadamard Transform).
"""

import numpy as np
from typing import Tuple


def pad(x: np.ndarray) -> np.ndarray:
    """
    Pad 2D array to next power of 2 dimensions

    Pads the spatial dimensions (height and width) of a 2D or 3D array to the
    next power of 2. This is useful for operations like the Fast Walsh-Hadamard
    Transform which require dimensions to be powers of 2.

    Args:
        x: Input array of shape:
           - (height, width) for 2D arrays, or
           - (batch_size, height, width) for 3D arrays

    Returns:
        Padded array with spatial dimensions rounded up to next power of 2.
        If dimensions are already powers of 2, returns the input unchanged.
        Padding uses zero-fill (constant mode).

    Example:
        >>> x = np.ones((32, 28, 28))  # batch of 28x28 images
        >>> padded = pad(x)
        >>> padded.shape
        (32, 32, 32)  # padded to next power of 2

    Note:
        This function pads both height and width dimensions to the same size
        (the next power of 2 after the height dimension).
    """
    n = x.shape[1]  # Get height dimension

    # Calculate next power of 2
    # Mathematical formula: next_pow2 = 2^ceil(log2(n))
    # This finds the smallest power of 2 that is >= n
    # Examples: n=28 -> ceil(log2(28)) = 5 -> 2^5 = 32
    #           n=32 -> ceil(log2(32)) = 5 -> 2^5 = 32 (already power of 2)
    next_pow2 = 2 ** int(np.ceil(np.log2(n)))

    # If already a power of 2, no padding needed
    if n == next_pow2:
        return x

    # Calculate padding width needed
    # This is the amount of zeros to add to reach the next power of 2
    pad_width = next_pow2 - n

    # Pad both height and width dimensions with zeros
    # np.pad format: ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
    # (batch_dim, height_dim, width_dim) -> no padding on batch, pad_width on others
    # This ensures the spatial dimensions are powers of 2 for operations like FFT or Walsh-Hadamard
    return np.pad(x, ((0, 0), (0, pad_width), (0, pad_width)), mode='constant')


def pad_to_size(x: np.ndarray, target_size: int) -> np.ndarray:
    """
    Pad 2D array to specific target size

    Pads spatial dimensions to a specified target size. Useful when you need
    exact dimensions rather than just power-of-2.

    Args:
        x: Input array of shape (batch_size, height, width) or (height, width)
        target_size: Target size for both height and width dimensions

    Returns:
        Padded array with spatial dimensions equal to target_size

    Raises:
        ValueError: If target_size is smaller than current dimensions
    """
    if x.ndim == 2:
        h, w = x.shape
        if h > target_size or w > target_size:
            raise ValueError(f"Target size {target_size} is smaller than current dimensions ({h}, {w})")
        pad_h = target_size - h
        pad_w = target_size - w
        return np.pad(x, ((0, pad_h), (0, pad_w)), mode='constant')
    elif x.ndim == 3:
        _, h, w = x.shape
        if h > target_size or w > target_size:
            raise ValueError(f"Target size {target_size} is smaller than current dimensions ({h}, {w})")
        pad_h = target_size - h
        pad_w = target_size - w
        return np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    else:
        raise ValueError(f"Expected 2D or 3D array, got {x.ndim}D")


def next_power_of_2(n: int) -> int:
    """
    Calculate the next power of 2 greater than or equal to n

    Args:
        n: Input integer

    Returns:
        Next power of 2 >= n

    Example:
        >>> next_power_of_2(28)
        32
        >>> next_power_of_2(32)
        32
    """
    # Mathematical formula: 2^ceil(log2(n))
    # This finds the smallest power of 2 that is >= n
    # The ceiling function ensures we round up to the next integer
    # Examples: n=28 -> log2(28) ≈ 4.807 -> ceil(4.807) = 5 -> 2^5 = 32
    #           n=32 -> log2(32) = 5 -> ceil(5) = 5 -> 2^5 = 32
    return 2 ** int(np.ceil(np.log2(n)))