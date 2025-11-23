"""
CNN Model Architecture with configurable normalization schemes.

This module provides a flexible CNN builder that can create models with:
- No normalization (baseline)
- Batch normalization
- Layer normalization
- Weight normalization

The architecture is based on the provided CNN_assignment4.py but modernized
for TensorFlow 2.x and Apple Silicon.

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import tensorflow as tf
from src.normalization import (
    BatchNormalization,
    LayerNormalization,
    WeightNormalization,
)


def create_cnn(
    norm_type=None, use_custom=True, input_shape=(28, 28, 1), num_classes=10
):
    """
    Create a CNN model with specified normalization technique.

    ARCHITECTURE:
        Input (28x28x1)
        ↓
        Conv2D (5x5, 30 filters) + Norm? + ReLU
        ↓
        MaxPool2D (2x2)
        ↓
        Conv2D (5x5, 60 filters) + Norm? + ReLU
        ↓
        MaxPool2D (2x2)
        ↓
        Flatten
        ↓
        Dense (100 units) + Norm? + ReLU
        ↓
        Dense (10 units) + Softmax

    This architecture is similar to LeNet-5 but adapted for Fashion MNIST.

    NORMALIZATION PLACEMENT:
        Normalization is applied AFTER linear/conv operations but BEFORE activation.
        This follows the original papers' recommendations:
            z = W*x + b          # Linear transformation
            z_norm = Norm(z)     # Normalization
            a = activation(z_norm)  # Non-linearity

    Args:
        norm_type: One of None, 'batchnorm', 'layernorm', 'weightnorm'
        use_custom: If True, use custom implementations; else use TF built-ins
        input_shape: Shape of input images (default: 28x28x1 for Fashion MNIST)
        num_classes: Number of output classes (default: 10)

    Returns:
        model: Keras Model
        norm_layers: List of custom normalization layer objects (empty if use_custom=False)
    """

    # Track custom normalization layers for gradient computation
    norm_layers = []

    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = inputs

    # ========== CONV BLOCK 1 ==========
    # Conv2D: 5x5 kernel, 30 filters, same padding
    # Same padding keeps spatial dimensions: 28x28 -> 28x28

    if norm_type == "weightnorm" and use_custom:
        # Weight normalization: normalize the convolutional filters
        wn_conv1 = WeightNormalization(shape=(5, 5, 1, 30), name="conv1_wn")
        norm_layers.append(wn_conv1)

        # Manually create conv layer with weight-normalized filters
        conv1_bias = tf.Variable(tf.zeros([30]), trainable=True, name="conv1_bias")
        w_normalized = wn_conv1()

        # Apply convolution with normalized weights
        x = tf.nn.conv2d(x, w_normalized, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, conv1_bias)
    else:
        # Standard convolution
        x = tf.keras.layers.Conv2D(
            filters=30,
            kernel_size=(5, 5),
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            name="conv1",
        )(x)

    # Apply normalization before activation (if specified)
    if norm_type == "batchnorm":
        if use_custom:
            bn1 = BatchNormalization(num_features=30)
            norm_layers.append(bn1)
            x = bn1(x, training=True)  # Will be updated in training loop
        else:
            x = tf.keras.layers.BatchNormalization(name="bn1")(x)

    elif norm_type == "layernorm":
        if use_custom:
            ln1 = LayerNormalization(normalized_shape=30)
            norm_layers.append(ln1)
            x = ln1(x)
        else:
            x = tf.keras.layers.LayerNormalization(name="ln1")(x)

    # ReLU activation
    x = tf.keras.layers.ReLU(name="relu1")(x)

    # MaxPooling: 2x2, reduces spatial dimensions by half (28x28 -> 14x14)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # ========== CONV BLOCK 2 ==========
    if norm_type == "weightnorm" and use_custom:
        wn_conv2 = WeightNormalization(shape=(5, 5, 30, 60), name="conv2_wn")
        norm_layers.append(wn_conv2)

        conv2_bias = tf.Variable(tf.zeros([60]), trainable=True, name="conv2_bias")
        w_normalized = wn_conv2()

        x = tf.nn.conv2d(x, w_normalized, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, conv2_bias)
    else:
        x = tf.keras.layers.Conv2D(
            filters=60,
            kernel_size=(5, 5),
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            name="conv2",
        )(x)

    if norm_type == "batchnorm":
        if use_custom:
            bn2 = BatchNormalization(num_features=60)
            norm_layers.append(bn2)
            x = bn2(x, training=True)
        else:
            x = tf.keras.layers.BatchNormalization(name="bn2")(x)

    elif norm_type == "layernorm":
        if use_custom:
            ln2 = LayerNormalization(normalized_shape=60)
            norm_layers.append(ln2)
            x = ln2(x)
        else:
            x = tf.keras.layers.LayerNormalization(name="ln2")(x)

    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # After 2 pooling layers: 28 -> 14 -> 7
    # Shape is now: (batch, 7, 7, 60)

    # ========== FLATTEN ==========
    x = tf.keras.layers.Flatten(name="flatten")(x)
    # Shape: (batch, 7*7*60) = (batch, 2940)

    # ========== DENSE BLOCK 1 ==========
    if norm_type == "weightnorm" and use_custom:
        wn_dense1 = WeightNormalization(shape=(2940, 100), name="dense1_wn")
        norm_layers.append(wn_dense1)

        dense1_bias = tf.Variable(tf.zeros([100]), trainable=True, name="dense1_bias")
        w_normalized = wn_dense1()

        x = tf.matmul(x, w_normalized) + dense1_bias
    else:
        x = tf.keras.layers.Dense(
            units=100, use_bias=True, kernel_initializer="he_normal", name="dense1"
        )(x)

    if norm_type == "batchnorm":
        if use_custom:
            bn3 = BatchNormalization(num_features=100)
            norm_layers.append(bn3)
            x = bn3(x, training=True)
        else:
            x = tf.keras.layers.BatchNormalization(name="bn3")(x)

    elif norm_type == "layernorm":
        if use_custom:
            ln3 = LayerNormalization(normalized_shape=100)
            norm_layers.append(ln3)
            x = ln3(x)
        else:
            x = tf.keras.layers.LayerNormalization(name="ln3")(x)

    x = tf.keras.layers.ReLU(name="relu3")(x)

    # ========== OUTPUT LAYER ==========
    # No normalization on output layer (we want logits for softmax)
    if norm_type == "weightnorm" and use_custom:
        wn_dense2 = WeightNormalization(shape=(100, num_classes), name="dense2_wn")
        norm_layers.append(wn_dense2)

        dense2_bias = tf.Variable(
            tf.zeros([num_classes]), trainable=True, name="dense2_bias"
        )
        w_normalized = wn_dense2()

        outputs = tf.matmul(x, w_normalized) + dense2_bias
    else:
        outputs = tf.keras.layers.Dense(
            units=num_classes,
            use_bias=True,
            kernel_initializer="he_normal",
            name="output",
        )(x)

    # Create model
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name=f"cnn_{norm_type or 'baseline'}"
    )

    # Print model summary
    print(f"\n{'=' * 60}")
    print(f"Created model: {norm_type or 'baseline'} (custom={use_custom})")
    print(f"{'=' * 60}")
    model.summary()

    return model, norm_layers
