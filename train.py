"""
Training Loop

This module implements the main training loop for neural networks,
including batching, forward/backward passes, and evaluation.
"""

import numpy as np
from tqdm import tqdm
from typing import Callable, Any


def train(
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        metric: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
) -> dict:
    """
    Train a neural network model

    Implements the standard training loop with mini-batch gradient descent:
    1. Forward pass through model
    2. Compute loss
    3. Backward pass to compute gradients
    4. Update parameters using optimizer
    5. Evaluate on test set after each epoch

    Args:
        model: Model instance with forward() and backward() methods
        optimizer: Optimizer instance with update() and zero_grad() methods
        loss_fn: Loss function with forward() and backward() methods
        metric: Metric function that takes (y_true, y_pred) and returns a score
        X_train: Training data, shape (num_samples, ...)
        y_train: Training labels
        X_test: Test data for evaluation
        y_test: Test labels for evaluation
        epochs: Number of training epochs
        batch_size: Number of samples per batch
        learning_rate: Learning rate (applied if optimizer has 'lr' attribute)
        verbose: If True, print progress bars and metrics

    Returns:
        Dictionary containing training history:
        {
            'train_loss': list of training losses per epoch,
            'train_acc': list of training accuracies per epoch,
            'test_loss': list of test losses per epoch,
            'test_acc': list of test accuracies per epoch
        }
    """
    # Set optimizer learning rate if it has this attribute
    if hasattr(optimizer, 'lr'):
        optimizer.lr = learning_rate

    num_train = X_train.shape[0]
    num_batches = num_train // batch_size

    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(epochs):
        # Training Phase
        model.train_mode = True
        train_loss = 0.0
        train_acc = 0.0

        # Shuffle training data at the start of each epoch
        # This helps prevent the model from learning order-dependent patterns
        indices = np.random.permutation(num_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Progress bar for batches
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}",
                  disable=not verbose, unit="batch") as pbar:

            for batch_idx in range(num_batches):
                # Extract batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward pass: compute model predictions
                logits = model.forward(X_batch)

                # Compute loss using raw logits
                loss = loss_fn.forward(logits, y_batch)
                train_loss += loss

                # Backward pass: compute gradients
                grad_loss = loss_fn.backward()
                model.backward(grad_loss)

                # Update parameters using optimizer
                optimizer.update()

                # Zero out gradients for next iteration
                optimizer.zero_grad()

                # Compute batch accuracy
                # Use stored probabilities from loss function (after softmax)
                if hasattr(loss_fn, 'probs'):
                    probs = loss_fn.probs
                else:
                    # Fallback: compute softmax manually if needed
                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

                batch_acc = metric(y_batch, probs)
                train_acc += batch_acc

                # Update progress bar with current metrics
                if verbose:
                    pbar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'acc': f"{batch_acc * 100:.2f}%"
                    })
                    pbar.update(1)

        # Calculate average training metrics for the epoch
        train_loss /= num_batches
        train_acc = (train_acc / num_batches) * 100

        # Evaluation Phase
        model.train_mode = False

        # Forward pass on test set (no batching for simplicity)
        test_logits = model.forward(X_test)
        test_loss = loss_fn.forward(test_logits, y_test)

        # Compute test accuracy
        if hasattr(loss_fn, 'probs'):
            test_probs = loss_fn.probs
        else:
            test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

        test_acc = metric(y_test, test_probs) * 100

        # Store metrics in history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Print epoch summary
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print()

    return history


def evaluate(
        model: Any,
        loss_fn: Any,
        metric: Callable,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 512
) -> tuple:
    """
    Evaluate model on a dataset

    Args:
        model: Model instance
        loss_fn: Loss function
        metric: Metric function
        X: Input data
        y: True labels
        batch_size: Batch size for evaluation (to handle large datasets)

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.train_mode = False

    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    total_acc = 0.0

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_samples)

        X_batch = X[start:end]
        y_batch = y[start:end]

        # Forward pass
        logits = model.forward(X_batch)
        loss = loss_fn.forward(logits, y_batch)

        # Get probabilities
        if hasattr(loss_fn, 'probs'):
            probs = loss_fn.probs
        else:
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        acc = metric(y_batch, probs)

        total_loss += loss * (end - start)
        total_acc += acc * (end - start)

    avg_loss = total_loss / num_samples
    avg_acc = (total_acc / num_samples) * 100

    return avg_loss, avg_acc