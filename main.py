"""
MNIST CNN Training Script

This script demonstrates training a convolutional neural network on the MNIST
handwritten digit dataset using a custom NumPy-based deep learning framework.
"""

import numpy as np
import tensorflow as tf
import keras

from layers.conv import Conv2D
from layers.relu import ReLU
from layers.maxpool import MaxPooling2D
from layers.flatten import Flatten
from layers.dense import Dense
from layers.dropout import Dropout
from layers.softmax import Softmax
from model import CNNModel
from train import train
from utilities.accuracy import accuracy
from utilities.padding import pad
from utilities.adam import AdamOptimizer


def load_and_preprocess_data():
    """
    Load and preprocess MNIST dataset

    Returns:
        Tuple of (x_train, y_train, x_test, y_test) where:
        - x_train: Training images, shape (60000, 1, 32, 32), normalized to [0, 1]
        - y_train: Training labels, one-hot encoded, shape (60000, 10)
        - x_test: Test images, shape (10000, 1, 32, 32), normalized to [0, 1]
        - y_test: Test labels, one-hot encoded, shape (10000, 10)
    """
    print("Loading MNIST dataset...")

    # Load MNIST data from Keras
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Pad images from 28x28 to 32x32 (next power of 2)
    # Required for certain operations like Walsh-Hadamard Transform
    x_train = pad(x_train)
    x_test = pad(x_test)

    # Reshape to (batch_size, channels, height, width) format
    x_train = x_train.reshape(-1, 1, 32, 32)
    x_test = x_test.reshape(-1, 1, 32, 32)

    # Convert labels to one-hot encoding
    # e.g., 4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    print(f"Image shape: {x_train.shape[1:]}")

    return x_train, y_train, x_test, y_test


def build_model(num_classes=10, enable_timing=False):
    """
    Build CNN model architecture

    Architecture:
        Conv2D(8 filters, 3x3) -> ReLU -> MaxPool(2x2)
        -> Conv2D(16 filters, 3x3) -> ReLU -> MaxPool(3x3, stride=3)
        -> Flatten -> Dense(256) + ReLU -> Dropout(0.25)
        -> Dense(256) -> Dense(num_classes)

    Args:
        num_classes: Number of output classes (default: 10 for MNIST)
        enable_timing: If True, print layer-wise forward/backward timing

    Returns:
        CNNModel instance
    """
    print("\nBuilding model...")

    model = CNNModel([
        # Convolutional Block 1
        Conv2D(in_channels=1, out_channels=8, kernel_size=3),  # (1, 32, 32) -> (8, 30, 30)
        ReLU(),
        MaxPooling2D(kernel_size=2, stride=2),  # (8, 30, 30) -> (8, 15, 15)

        # Convolutional Block 2
        Conv2D(in_channels=8, out_channels=16, kernel_size=3),  # (8, 15, 15) -> (16, 13, 13)
        ReLU(),
        MaxPooling2D(kernel_size=3, stride=3),  # (16, 13, 13) -> (16, 4, 4)

        # Fully Connected Layers
        Flatten(),  # (16, 4, 4) -> 256
        Dense(input_dim=256, units=256, activation=ReLU()),  # Hidden layer with ReLU
        Dropout(dropout_rate=0.25),  # Regularization
        Dense(input_dim=256, units=256),  # Hidden layer (linear)
        Dense(input_dim=256, units=num_classes),  # Output layer
    ], runtimes=enable_timing)

    # Calculate total parameters
    total_params = sum(p.size for p in model.parameters().values())
    print(f"Total parameters: {total_params:,}")

    return model


def main():
    """
    Main training pipeline
    """
    # Configuration
    EPOCHS = 5
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001

    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Build model
    model = build_model(num_classes=10, enable_timing=False)

    # Initialize optimizer and loss function
    optimizer = AdamOptimizer(model, lr=LEARNING_RATE)
    loss_fn = Softmax()

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Optimizer: Adam")
    print("-" * 60)

    # Train the model
    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric=accuracy,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        verbose=True
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()