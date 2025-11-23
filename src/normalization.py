"""
Custom implementations of Batch Normalization, Layer Normalization, and Weight Normalization
using basic TensorFlow operations (no high-level APIs).

LEARNING OBJECTIVES:
- Understand the exact mathematical formulation of each normalization technique
- Implement forward pass with proper handling of training vs inference modes
- Compute gradients correctly for backpropagation
- Verify implementation against TensorFlow's reference implementations

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import numpy as np
import tensorflow as tf


class BatchNormalization:
    """
    Batch Normalization Implementation (Ioffe & Szegedy, 2015)

    KEY IDEA: Normalize activations across the mini-batch dimension.
    For each feature, compute mean and variance over all examples in the batch,
    then normalize to have mean=0, variance=1.

    FORMULATION:
        Training:
            μ_B = (1/m) Σ x_i                    # Mini-batch mean
            σ²_B = (1/m) Σ (x_i - μ_B)²          # Mini-batch variance
            x̂ = (x - μ_B) / √(σ²_B + ε)          # Normalize
            y = γ * x̂ + β                        # Scale and shift

        Inference:
            x̂ = (x - μ_running) / √(σ²_running + ε)
            y = γ * x̂ + β

    WHERE:
        - m = batch size
        - ε = small constant for numerical stability (prevents division by zero)
        - γ, β = learnable parameters (scale and shift)
        - μ_running, σ²_running = running averages updated during training

    WHY IT WORKS:
        1. Reduces internal covariate shift (distribution of layer inputs changes during training)
        2. Allows higher learning rates (normalization prevents exploding/vanishing gradients)
        3. Acts as regularizer (mini-batch statistics add noise)

    LAB QUESTION RELEVANCE:
        Q1: Demonstrates improved convergence vs baseline
        Q2: Custom implementation must match TF's within floating point error
        Q4: Shows degradation with small batch sizes (statistics become noisy)
    """

    def __init__(self, num_features, momentum=0.99, epsilon=1e-5):
        """
        Initialize Batch Normalization layer.

        Args:
            num_features: Number of features/channels to normalize
            momentum: Momentum for running average updates (typical: 0.9-0.99)
            epsilon: Small constant for numerical stability (typical: 1e-5)
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters: γ (scale), β (shift)
        # Initialize γ=1, β=0 so initial transformation is identity
        self.gamma = tf.Variable(
            tf.ones([num_features], dtype=tf.float32), trainable=True, name="bn_gamma"
        )
        self.beta = tf.Variable(
            tf.zeros([num_features], dtype=tf.float32), trainable=True, name="bn_beta"
        )

        # Running statistics for inference (non-trainable)
        # Updated during training with exponential moving average
        self.running_mean = tf.Variable(
            tf.zeros([num_features], dtype=tf.float32),
            trainable=False,
            name="bn_running_mean",
        )
        self.running_var = tf.Variable(
            tf.ones([num_features], dtype=tf.float32),
            trainable=False,
            name="bn_running_var",
        )

        # TensorFlow equivalent for verification
        self.tf_bn = tf.keras.layers.BatchNormalization(
            axis=-1,  # Normalize over last axis (features)
            momentum=momentum,
            epsilon=epsilon,
            center=True,  # Use beta
            scale=True,  # Use gamma
        )

    def __call__(self, x, training=True):
        """
        Forward pass through Batch Normalization.

        Args:
            x: Input tensor, shape (batch_size, ..., num_features)
            training: Boolean, use batch statistics if True, running statistics if False

        Returns:
            Normalized and scaled output, same shape as input
        """
        if training:
            # TRAINING MODE: Use mini-batch statistics

            # Compute mean and variance over batch dimension
            # For input shape (B, H, W, C), we want to compute statistics over (B, H, W)
            # leaving C dimension separate (each channel normalized independently)
            axes = list(range(len(x.shape) - 1))  # All axes except last (features)

            # Mini-batch mean: μ_B = (1/m) Σ x_i
            batch_mean = tf.reduce_mean(x, axis=axes, keepdims=False)

            # Mini-batch variance: σ²_B = (1/m) Σ (x_i - μ_B)²
            batch_var = tf.reduce_mean(
                tf.square(x - tf.reshape(batch_mean, [1] * (len(x.shape) - 1) + [-1])),
                axis=axes,
                keepdims=False,
            )

            # Update running statistics using exponential moving average
            # running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            self.running_mean.assign(
                self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean
            )
            self.running_var.assign(
                self.momentum * self.running_var + (1.0 - self.momentum) * batch_var
            )

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var

        else:
            # INFERENCE MODE: Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize: x̂ = (x - μ) / √(σ² + ε)
        # Reshape mean and var for broadcasting
        mean = tf.reshape(mean, [1] * (len(x.shape) - 1) + [-1])
        var = tf.reshape(var, [1] * (len(x.shape) - 1) + [-1])

        x_normalized = (x - mean) / tf.sqrt(var + self.epsilon)

        # Scale and shift: y = γ * x̂ + β
        gamma = tf.reshape(self.gamma, [1] * (len(x.shape) - 1) + [-1])
        beta = tf.reshape(self.beta, [1] * (len(x.shape) - 1) + [-1])

        output = gamma * x_normalized + beta

        return output

    def verify(self, test_input, training=True):
        """
        Verify custom implementation against TensorFlow's BatchNormalization.

        LAB QUESTION 2: "Compare your normalization function with tensorflow"
        This method computes the numerical difference between custom and TF implementations.
        The difference should be < 1e-6 (only floating-point error).

        Args:
            test_input: Test tensor to compare outputs
            training: Mode to test (training or inference)

        Returns:
            Dict with 'output_diff' and 'grad_diff' keys
        """
        # Forward pass with custom implementation
        custom_output = self(test_input, training=training)

        # Forward pass with TensorFlow implementation
        # Note: TF's BN needs to be built first
        if not self.tf_bn.built:
            self.tf_bn.build(test_input.shape)
            # Copy our parameters to TF layer for fair comparison
            self.tf_bn.gamma.assign(self.gamma)
            self.tf_bn.beta.assign(self.beta)
            self.tf_bn.moving_mean.assign(self.running_mean)
            self.tf_bn.moving_variance.assign(self.running_var)

        tf_output = self.tf_bn(test_input, training=training)

        # Compute output difference
        output_diff = tf.abs(custom_output - tf_output).numpy()

        # Compute gradient difference
        with tf.GradientTape() as tape1:
            custom_out = self(test_input, training=training)
            custom_loss = tf.reduce_mean(custom_out)
        custom_grads = tape1.gradient(custom_loss, [self.gamma, self.beta])

        with tf.GradientTape() as tape2:
            tf_out = self.tf_bn(test_input, training=training)
            tf_loss = tf.reduce_mean(tf_out)
        tf_grads = tape2.gradient(tf_loss, [self.tf_bn.gamma, self.tf_bn.beta])

        grad_diff = np.concatenate(
            [
                tf.abs(custom_grads[0] - tf_grads[0]).numpy().flatten(),
                tf.abs(custom_grads[1] - tf_grads[1]).numpy().flatten(),
            ]
        )

        return {
            "output_diff": output_diff,
            "grad_diff": grad_diff,
            "max_output_diff": float(np.max(output_diff)),
            "max_grad_diff": float(np.max(grad_diff)),
        }


class LayerNormalization:
    """
    Layer Normalization Implementation (Ba, Kiros & Hinton, 2016)

    KEY IDEA: Normalize activations across the feature dimension (not batch).
    For each training example, compute mean and variance over all features,
    then normalize to have mean=0, variance=1.

    FORMULATION:
        μ = (1/H) Σ x_j                    # Mean over features
        σ² = (1/H) Σ (x_j - μ)²            # Variance over features
        x̂ = (x - μ) / √(σ² + ε)            # Normalize
        y = γ * x̂ + β                      # Scale and shift

    WHERE:
        - H = number of features
        - Normalization is computed independently for each example
        - No dependence on batch size or other examples

    KEY DIFFERENCE FROM BATCH NORM:
        - BN: Normalizes across batch dimension (axis=0)
        - LN: Normalizes across feature dimension (axis=-1)

    ADVANTAGES OVER BATCH NORM:
        1. No batch size dependency (works with batch size = 1)
        2. Same computation at train and test time (no running averages needed)
        3. Better for RNNs (can normalize at each timestep independently)
        4. More stable with small batches

    LAB QUESTION RELEVANCE:
        Q4: "Why LayerNorm is better than BatchNorm?"
        LN remains stable with small batches because statistics are computed
        per-example, not per-batch. BN suffers when batch statistics become noisy.
    """

    def __init__(self, normalized_shape, epsilon=1e-5):
        """
        Initialize Layer Normalization.

        Args:
            normalized_shape: Shape of features to normalize (typically number of channels)
            epsilon: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        # Learnable parameters: γ (scale), β (shift)
        self.gamma = tf.Variable(
            tf.ones([normalized_shape], dtype=tf.float32),
            trainable=True,
            name="ln_gamma",
        )
        self.beta = tf.Variable(
            tf.zeros([normalized_shape], dtype=tf.float32),
            trainable=True,
            name="ln_beta",
        )

        # TensorFlow equivalent for verification
        self.tf_ln = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=epsilon,
            center=True,
            scale=True,
        )

    def __call__(self, x):
        """
        Forward pass through Layer Normalization.

        NOTE: No training/inference distinction needed for LayerNorm!
        Same computation always since we don't use batch statistics.

        Args:
            x: Input tensor, shape (batch_size, ..., num_features)

        Returns:
            Normalized output, same shape as input
        """
        # Compute mean and variance over feature dimension (last axis)
        # For shape (B, H, W, C), compute statistics over C for each (B, H, W) position

        # Mean: μ = (1/H) Σ x_j
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        # Variance: σ² = (1/H) Σ (x_j - μ)²
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)

        # Normalize: x̂ = (x - μ) / √(σ² + ε)
        x_normalized = (x - mean) / tf.sqrt(variance + self.epsilon)

        # Scale and shift: y = γ * x̂ + β
        # Broadcast gamma and beta across batch and spatial dimensions
        output = self.gamma * x_normalized + self.beta

        return output

    def verify(self, test_input):
        """
        Verify custom implementation against TensorFlow's LayerNormalization.

        Args:
            test_input: Test tensor to compare outputs

        Returns:
            Dict with 'output_diff' and 'grad_diff' keys
        """
        # Forward pass with custom implementation
        custom_output = self(test_input)

        # Forward pass with TensorFlow implementation
        if not self.tf_ln.built:
            self.tf_ln.build(test_input.shape)
            self.tf_ln.gamma.assign(self.gamma)
            self.tf_ln.beta.assign(self.beta)

        tf_output = self.tf_ln(test_input)

        # Compute output difference
        output_diff = tf.abs(custom_output - tf_output).numpy()

        # Compute gradient difference
        with tf.GradientTape() as tape1:
            custom_out = self(test_input)
            custom_loss = tf.reduce_mean(custom_out)
        custom_grads = tape1.gradient(custom_loss, [self.gamma, self.beta])

        with tf.GradientTape() as tape2:
            tf_out = self.tf_ln(test_input)
            tf_loss = tf.reduce_mean(tf_out)
        tf_grads = tape2.gradient(tf_loss, [self.tf_ln.gamma, self.tf_ln.beta])

        grad_diff = np.concatenate(
            [
                tf.abs(custom_grads[0] - tf_grads[0]).numpy().flatten(),
                tf.abs(custom_grads[1] - tf_grads[1]).numpy().flatten(),
            ]
        )

        return {
            "output_diff": output_diff,
            "grad_diff": grad_diff,
            "max_output_diff": float(np.max(output_diff)),
            "max_grad_diff": float(np.max(grad_diff)),
        }


class WeightNormalization:
    """
    Weight Normalization Implementation (Salimans & Kingma, 2016)

    KEY IDEA: Reparameterize weight vectors to decouple magnitude from direction.
    Instead of learning w directly, learn:
        w = (g / ||v||) * v
    where:
        - g = scalar magnitude (how strong the connection is)
        - v = direction vector (what pattern it detects)
        - ||v|| = Euclidean norm of v

    ADVANTAGES:
        1. Improves conditioning of optimization problem
        2. No batch size dependency (deterministic)
        3. Lower computational cost than batch norm
        4. Works well for RNNs and online learning

    GRADIENT BEHAVIOR:
        ∇_g L = (∇_w L · v) / ||v||
        ∇_v L = (g / ||v||) * M_w * ∇_w L
        where M_w = I - (w w^T) / ||w||² is a projection matrix

        The gradient w.r.t. v is projected away from current weight direction,
        which removes noise in that direction and stabilizes learning.

    LAB QUESTION RELEVANCE:
        Q3: "Which one is good and why?"
        Weight norm offers a good trade-off: better than baseline, simpler than
        batch norm, no batch size dependency like layer norm, lower cost than both.
    """

    def __init__(self, shape, name="wn"):
        """
        Initialize Weight Normalization.

        Args:
            shape: Shape of weight matrix to normalize
            name: Name prefix for variables
        """
        self.shape = shape
        self.name = name

        # Initialize v with He initialization (good for ReLU networks)
        # v represents the direction of the weight vector
        stddev = np.sqrt(2.0 / shape[0])  # He initialization
        self.v = tf.Variable(
            tf.random.normal(shape, stddev=stddev, dtype=tf.float32),
            trainable=True,
            name=f"{name}_v",
        )

        # Initialize g to the norm of v
        # This ensures w = v initially (identity transformation at start)
        initial_norm = tf.norm(self.v)
        self.g = tf.Variable(initial_norm, trainable=True, name=f"{name}_g")

    def __call__(self):
        """
        Compute normalized weights: w = (g / ||v||) * v

        Returns:
            Normalized weight matrix, same shape as self.v
        """
        # Compute norm of v: ||v|| = √(Σ v_i²)
        v_norm = tf.norm(self.v)

        # Compute normalized weights: w = (g / ||v||) * v
        # This decouples magnitude (g) from direction (v/||v||)
        w = (self.g / v_norm) * self.v

        return w

    def verify(self, test_input, test_bias):
        """
        Weight normalization doesn't have a direct TF equivalent,
        so we verify the mathematical properties instead:
        1. ||w|| should equal g
        2. Direction of w should match direction of v

        Args:
            test_input: Unused (for interface consistency)
            test_bias: Unused (for interface consistency)

        Returns:
            Dict with verification metrics
        """
        w = self()
        w_norm = tf.norm(w)
        v_norm = tf.norm(self.v)

        # Property 1: ||w|| = g
        norm_diff = tf.abs(w_norm - self.g)

        # Property 2: Direction of w matches direction of v
        w_dir = w / w_norm
        v_dir = self.v / v_norm
        direction_diff = tf.norm(w_dir - v_dir)

        return {
            "norm_property_error": float(norm_diff.numpy()),
            "direction_property_error": float(direction_diff.numpy()),
            "g_value": float(self.g.numpy()),
            "v_norm": float(v_norm.numpy()),
            "w_norm": float(w_norm.numpy()),
        }
