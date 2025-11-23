"""
Data loading and preprocessing for Fashion MNIST
Handles dataset loading, normalization, and tf.data pipeline creation

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import numpy as np
import tensorflow as tf


def load_fashion_mnist():
    """
    Load and preprocess Fashion MNIST dataset.

    Fashion MNIST contains:
    - 60,000 training images (28x28 grayscale)
    - 10,000 test images
    - 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

    We use Fashion MNIST instead of CIFAR-10 because:
    1. Faster training (28x28 grayscale vs 32x32 RGB)
    2. Still complex enough to demonstrate normalization benefits
    3. Standard benchmark for comparing techniques

    Returns:
        Tuple: (x_train, y_train, x_test, y_test)
            x_train: Training images, shape (60000, 28, 28, 1), normalized to [0, 1]
            y_train: Training labels, shape (60000, 10), one-hot encoded
            x_test: Test images, shape (10000, 28, 28, 1), normalized to [0, 1]
            y_test: Test labels, shape (10000, 10), one-hot encoded
    """
    # Load dataset using Keras (already split into train/test)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1]
    # This prevents saturation in early layers and helps gradient flow
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    # CNN layers expect (batch, height, width, channels)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to one-hot encoding
    # Shape: (N,) -> (N, 10)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    print("✓ Loaded Fashion MNIST:")
    print(f"  Training set: {x_train.shape} images, {y_train.shape} labels")
    print(f"  Test set: {x_test.shape} images, {y_test.shape} labels")

    return x_train, y_train, x_test, y_test


def create_dataloaders(x, y, batch_size, shuffle=True):
    """
    Create tf.data.Dataset pipeline for efficient batching and prefetching.

    The tf.data API provides optimized data loading with:
    - Automatic batching
    - Optional shuffling
    - Prefetching (loads next batch while GPU processes current batch)

    Args:
        x: Images array
        y: Labels array (one-hot encoded)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset

    Returns:
        tf.data.Dataset: Configured data pipeline
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        # Shuffle with buffer size = dataset size for complete shuffling
        # This is important for SGD to see diverse batches
        dataset = dataset.shuffle(buffer_size=len(x))

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch allows the data pipeline to fetch the next batch
    # while the current batch is being processed on GPU
    # AUTOTUNE lets TensorFlow dynamically tune the prefetch buffer
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
