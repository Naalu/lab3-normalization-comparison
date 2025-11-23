"""
Training and evaluation functions using tf.GradientTape for custom backpropagation.

This module implements:
- Training loop with custom gradient computation
- Evaluation on validation/test sets
- Integration of custom normalization layers with Keras models

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import tensorflow as tf
from tqdm import tqdm


def train_model(model, norm_layers, train_ds, val_ds, epochs, learning_rate):
    """
    Train the model using tf.GradientTape for explicit gradient computation.

    WHY GRADIENTTAPE?
    The assignment requires using GradientTape to understand the backward pass.
    GradientTape records operations for automatic differentiation, allowing us to:
    1. Compute gradients of loss w.r.t. all trainable variables
    2. Include custom normalization layer gradients
    3. Apply gradients manually (rather than using model.fit())

    TRAINING LOOP STRUCTURE:
        For each epoch:
            For each batch:
                1. Forward pass (compute predictions)
                2. Compute loss
                3. Compute gradients using GradientTape
                4. Apply gradients with optimizer
                5. Track metrics

    Args:
        model: Keras model
        norm_layers: List of custom normalization layers (may be empty)
        train_ds: Training dataset (tf.data.Dataset)
        val_ds: Validation dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        history: Dict with keys 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'
    """

    # Initialize optimizer (Adam is default for this lab)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss function: categorical crossentropy (for one-hot encoded labels)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Metrics
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    # History tracking
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Collect all trainable variables
    # This includes model weights AND custom normalization parameters
    trainable_vars = model.trainable_variables
    if norm_layers:
        for norm_layer in norm_layers:
            if hasattr(norm_layer, "gamma"):
                trainable_vars.append(norm_layer.gamma)
            if hasattr(norm_layer, "beta"):
                trainable_vars.append(norm_layer.beta)
            if hasattr(norm_layer, "g"):  # Weight normalization
                trainable_vars.append(norm_layer.g)
            if hasattr(norm_layer, "v"):
                trainable_vars.append(norm_layer.v)

    print(f"\n{'=' * 60}")
    print(f"Starting training for {epochs} epochs")
    print(f"Learning rate: {learning_rate}")
    print(f"Trainable variables: {len(trainable_vars)}")
    print(f"{'=' * 60}\n")

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reset metrics at start of each epoch
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()

        # Iterate over training batches with progress bar
        pbar = tqdm(train_ds, desc="Training", leave=False)
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            # ========== FORWARD PASS ==========
            # Use GradientTape to record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Forward pass through model
                # For custom normalization, we need to manually call them with training=True
                if norm_layers and any(
                    hasattr(nl, "running_mean") for nl in norm_layers
                ):
                    # This is batch norm - need to set training mode
                    # Note: This is a simplified approach; in production, you'd use
                    # Keras's training argument propagation
                    logits = model(x_batch, training=True)
                else:
                    logits = model(x_batch, training=True)

                # Compute loss
                loss = loss_fn(y_batch, logits)

            # ========== BACKWARD PASS ==========
            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tape.gradient(loss, trainable_vars)

            # Apply gradients to update weights
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update metrics
            train_loss_metric.update_state(loss)
            train_acc_metric.update_state(y_batch, logits)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{train_loss_metric.result():.4f}",
                    "acc": f"{train_acc_metric.result():.4f}",
                }
            )

        # Get epoch metrics
        epoch_train_loss = train_loss_metric.result().numpy()
        epoch_train_acc = train_acc_metric.result().numpy()

        # ========== VALIDATION ==========
        val_loss, val_acc = evaluate_model(model, val_ds, loss_fn)

        # Store history
        history["train_loss"].append(float(epoch_train_loss))
        history["train_accuracy"].append(float(epoch_train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        # Print epoch summary
        print(
            f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}"
        )
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}\n")

    return history


def evaluate_model(model, dataset, loss_fn):
    """
    Evaluate model on a dataset.

    This function runs the model in inference mode (no gradient computation).
    For batch normalization, this uses the running mean/variance instead of
    batch statistics.

    Args:
        model: Keras model
        dataset: tf.data.Dataset to evaluate on
        loss_fn: Loss function

    Returns:
        Tuple of (average_loss, accuracy)
    """
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for x_batch, y_batch in dataset:
        # Forward pass in inference mode
        logits = model(x_batch, training=False)

        # Compute loss
        loss = loss_fn(y_batch, logits)

        # Update metrics
        loss_metric.update_state(loss)
        acc_metric.update_state(y_batch, logits)

    avg_loss = loss_metric.result().numpy()
    accuracy = acc_metric.result().numpy()

    return avg_loss, accuracy


def evaluate_final_test(model, test_ds):
    """
    Evaluate model on test set and return detailed metrics.

    Args:
        model: Trained Keras model
        test_ds: Test dataset

    Returns:
        Dict with test metrics
    """
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    test_loss, test_acc = evaluate_model(model, test_ds, loss_fn)

    print(f"\n{'=' * 60}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"{'=' * 60}\n")

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
