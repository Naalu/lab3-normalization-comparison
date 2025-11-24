"""
CNN Model Architecture with configurable normalization schemes.
Updated to fix Keras Functional API compatibility issues.

This module provides a flexible CNN builder that can create models with:
- No normalization (baseline)
- Batch normalization
- Layer normalization
- Weight normalization

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import tensorflow as tf
from src.normalization import BatchNormalizationLayer, LayerNormalizationLayer


def create_cnn(
    norm_type=None, use_custom=True, input_shape=(28, 28, 1), num_classes=10
):
    """
    Create a CNN model with specified normalization technique.

    SUPPORTS ALL THREE REQUIRED NORMALIZATIONS:
    - BatchNorm: Normalizes activations across mini-batch
    - LayerNorm: Normalizes activations across features
    - WeightNorm: Reparameterizes weight vectors as w = (g/||v||) * v

    Args:
        norm_type: One of None, 'batchnorm', 'layernorm', 'weightnorm'
        use_custom: If True, use custom implementations; else use TF built-ins
        input_shape: Shape of input images (default: 28x28x1)
        num_classes: Number of output classes (default: 10)

    Returns:
        model: Keras Model
        norm_layers: List of custom normalization objects for verification
    """

    # Track normalization layers for verification
    custom_norm_keras_layers = []

    # Input
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = inputs

    # ========== CONV BLOCK 1 ==========
    if norm_type == "weightnorm" and use_custom:
        # Use weight-normalized conv layer
        from src.normalization import WeightNormConv2D

        conv1 = WeightNormConv2D(
            filters=30,
            kernel_size=(5, 5),
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            name="conv1",
        )
        custom_norm_keras_layers.append(conv1)
        x = conv1(x)
    else:
        # Standard conv layer
        x = tf.keras.layers.Conv2D(
            filters=30,
            kernel_size=(5, 5),
            padding="same",
            use_bias=(norm_type != "batchnorm"),
            kernel_initializer="he_normal",
            name="conv1",
        )(x)

    # Apply activation normalization (if not using weight norm)
    if norm_type == "batchnorm":
        if use_custom:
            bn_layer = BatchNormalizationLayer(num_features=30, name="bn1")
            custom_norm_keras_layers.append(bn_layer)
            x = bn_layer(x)
        else:
            x = tf.keras.layers.BatchNormalization(name="bn1")(x)

    elif norm_type == "layernorm":
        if use_custom:
            ln_layer = LayerNormalizationLayer(normalized_shape=30, name="ln1")
            custom_norm_keras_layers.append(ln_layer)
            x = ln_layer(x)
        else:
            x = tf.keras.layers.LayerNormalization(name="ln1")(x)

    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # ========== CONV BLOCK 2 ==========
    if norm_type == "weightnorm" and use_custom:
        from src.normalization import WeightNormConv2D

        conv2 = WeightNormConv2D(
            filters=60,
            kernel_size=(5, 5),
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            name="conv2",
        )
        custom_norm_keras_layers.append(conv2)
        x = conv2(x)
    else:
        x = tf.keras.layers.Conv2D(
            filters=60,
            kernel_size=(5, 5),
            padding="same",
            use_bias=(norm_type != "batchnorm"),
            kernel_initializer="he_normal",
            name="conv2",
        )(x)

    if norm_type == "batchnorm":
        if use_custom:
            bn_layer = BatchNormalizationLayer(num_features=60, name="bn2")
            custom_norm_keras_layers.append(bn_layer)
            x = bn_layer(x)
        else:
            x = tf.keras.layers.BatchNormalization(name="bn2")(x)

    elif norm_type == "layernorm":
        if use_custom:
            ln_layer = LayerNormalizationLayer(normalized_shape=60, name="ln2")
            custom_norm_keras_layers.append(ln_layer)
            x = ln_layer(x)
        else:
            x = tf.keras.layers.LayerNormalization(name="ln2")(x)

    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # ========== FLATTEN ==========
    x = tf.keras.layers.Flatten(name="flatten")(x)

    # ========== DENSE BLOCK 1 ==========
    if norm_type == "weightnorm" and use_custom:
        from src.normalization import WeightNormDense

        dense1 = WeightNormDense(
            units=100, use_bias=True, kernel_initializer="he_normal", name="dense1"
        )
        custom_norm_keras_layers.append(dense1)
        x = dense1(x)
    else:
        x = tf.keras.layers.Dense(
            units=100,
            use_bias=(norm_type != "batchnorm"),
            kernel_initializer="he_normal",
            name="dense1",
        )(x)

    if norm_type == "batchnorm":
        if use_custom:
            bn_layer = BatchNormalizationLayer(num_features=100, name="bn3")
            custom_norm_keras_layers.append(bn_layer)
            x = bn_layer(x)
        else:
            x = tf.keras.layers.BatchNormalization(name="bn3")(x)

    elif norm_type == "layernorm":
        if use_custom:
            ln_layer = LayerNormalizationLayer(normalized_shape=100, name="ln3")
            custom_norm_keras_layers.append(ln_layer)
            x = ln_layer(x)
        else:
            x = tf.keras.layers.LayerNormalization(name="ln3")(x)

    x = tf.keras.layers.ReLU(name="relu3")(x)

    # ========== OUTPUT LAYER ==========
    if norm_type == "weightnorm" and use_custom:
        from src.normalization import WeightNormDense

        output_layer = WeightNormDense(
            units=num_classes,
            use_bias=True,
            kernel_initializer="he_normal",
            name="output",
        )
        custom_norm_keras_layers.append(output_layer)
        outputs = output_layer(x)
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

    # Extract custom normalization objects for verification
    norm_layers = []
    if use_custom and custom_norm_keras_layers:
        model.build(input_shape=(None,) + input_shape)

        for keras_layer in custom_norm_keras_layers:
            custom_norm = keras_layer.get_custom_norm()
            if custom_norm is not None:
                norm_layers.append(custom_norm)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Created model: {norm_type or 'baseline'} (custom={use_custom})")
    if norm_layers:
        print(f"Custom normalization layers: {len(norm_layers)}")
    print(f"{'=' * 60}")
    model.summary()

    return model, norm_layers
