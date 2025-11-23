"""
Utilities for Lab 3: Normalization Techniques
Handles: seed setting, device configuration, I/O operations, plotting

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def derive_seed_from_string(s):
    """
    Derive a deterministic seed from a string.
    For "Karl", this computes: K(75) + a(97) + r(114) + l(108) = 394

    This ensures reproducibility across all experiments while using a
    meaningful, memorable seed value derived from the instructor's name.

    Args:
        s: String to convert to seed

    Returns:
        Integer seed value
    """
    seed = sum([ord(c) for c in s])
    print(f"Derived seed from '{s}': {seed}")
    return seed


def setup(seed_string="Karl"):
    """
    Initialize the experimental environment with reproducible random seeds
    and configure TensorFlow for Apple Silicon GPU.

    LAB QUESTION RELEVANCE: Proper setup ensures all experiments are
    reproducible and comparable. This is critical for scientific validity.

    Args:
        seed_string: String to derive seed from (default: "Karl")
    """
    # Derive seed deterministically
    seed = derive_seed_from_string(seed_string)

    # Set all random seeds for reproducibility
    # TensorFlow uses multiple random number generators that must all be seeded
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set TensorFlow to deterministic operations where possible
    # Note: Some GPU operations may still have non-deterministic behavior
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Configure GPU memory growth to prevent TensorFlow from allocating
    # all GPU memory at once (important for Apple Silicon)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured {len(gpus)} GPU(s) with memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected - training will use CPU")

    print(f"✓ Environment setup complete (seed={seed})")


def save_results(results, filepath):
    """
    Save experiment results to JSON file.
    Converts numpy/TensorFlow types to native Python types for JSON serialization.

    Args:
        results: Dictionary of results (can contain numpy arrays, TF tensors)
        filepath: Path to save JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy/TensorFlow types to native Python types
    def convert(obj):
        if isinstance(obj, (np.ndarray, tf.Tensor)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.numpy().tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        else:
            return obj

    serializable_results = convert(results)

    with open(filepath, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"✓ Results saved to {filepath}")


def load_results(filepath):
    """
    Load experiment results from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary of results
    """
    with open(filepath, "r") as f:
        results = json.load(f)
    return results


def plot_training_curves(all_results, save_path):
    """
    Plot training and validation curves for all experiments.
    Creates a 2x2 grid showing: train loss, train acc, val loss, val acc

    LAB QUESTION 1: "Compare Results with and without Normalization"
    This plot directly addresses this question by overlaying all methods.

    Args:
        all_results: Dict mapping experiment_id -> results dict
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Define colors for each normalization type
    colors = {
        "baseline": "#2E86AB",  # Blue
        "batchnorm": "#A23B72",  # Purple
        "layernorm": "#F18F01",  # Orange
        "weightnorm": "#06A77D",  # Green
    }

    metrics = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("train_accuracy", "Training Accuracy", axes[0, 1]),
        ("val_loss", "Validation Loss", axes[1, 0]),
        ("val_accuracy", "Validation Accuracy", axes[1, 1]),
    ]

    for exp_id, results in all_results.items():
        # Extract normalization type and batch size from exp_id
        # Format: "normtype_bsXXX"
        parts = exp_id.split("_")
        norm_type = parts[0]
        batch_size = parts[1] if len(parts) > 1 else "bs128"

        # Only plot batch_size=128 for main comparison
        if "bs128" not in batch_size:
            continue

        history = results["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        color = colors.get(norm_type, "#000000")

        for metric_key, title, ax in metrics:
            if metric_key in history:
                ax.plot(
                    epochs,
                    history[metric_key],
                    color=color,
                    linewidth=2,
                    label=norm_type,
                    alpha=0.8,
                )
                ax.set_xlabel("Epoch", fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(title, fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Training curves saved to {save_path}")
    plt.close()


def plot_verification(verification_results, save_path, norm_type):
    """
    Plot verification results comparing custom vs TensorFlow implementations.
    Shows both output differences and gradient differences.

    LAB QUESTION 2: "Compare your normalization function with tensorflow"
    This plot shows numerical differences should be < 1e-6.

    Args:
        verification_results: Dict with 'output_diff' and 'grad_diff' keys
        save_path: Where to save plot
        norm_type: 'batchnorm' or 'layernorm' for title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot output differences
    if "output_diff" in verification_results:
        output_diff = verification_results["output_diff"]
        mean_diff = np.mean(output_diff)
        max_diff = np.max(output_diff)

        axes[0].hist(
            output_diff.flatten(),
            bins=50,
            color="#2E86AB",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0].axvline(
            mean_diff,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_diff:.2e}",
        )
        axes[0].set_xlabel("Absolute Difference", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title(
            f"{norm_type.upper()} Output: Custom vs TF\nMax Diff: {max_diff:.2e}",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Plot gradient differences
    if "grad_diff" in verification_results:
        grad_diff = verification_results["grad_diff"]
        mean_diff = np.mean(grad_diff)
        max_diff = np.max(grad_diff)

        axes[1].hist(
            grad_diff.flatten(), bins=50, color="#A23B72", alpha=0.7, edgecolor="black"
        )
        axes[1].axvline(
            mean_diff,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_diff:.2e}",
        )
        axes[1].set_xlabel("Absolute Difference", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title(
            f"{norm_type.upper()} Gradients: Custom vs TF\nMax Diff: {max_diff:.2e}",
            fontsize=14,
            fontweight="bold",
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Verification plot saved to {save_path}")
    plt.close()


def plot_small_batch_comparison(bn_results, ln_results, save_path):
    """
    Compare BatchNorm vs LayerNorm performance on small batches.

    LAB QUESTION 4: "Why LayerNorm is better than BatchNorm?"
    This plot demonstrates LN's robustness to small batch sizes.

    Args:
        bn_results: Results dict for batchnorm with small batch
        ln_results: Results dict for layernorm with small batch
        save_path: Where to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extract histories
    bn_history = bn_results["history"]
    ln_history = ln_results["history"]

    epochs_bn = range(1, len(bn_history["val_accuracy"]) + 1)
    epochs_ln = range(1, len(ln_history["val_accuracy"]) + 1)

    # Plot validation accuracy
    axes[0].plot(
        epochs_bn,
        bn_history["val_accuracy"],
        color="#A23B72",
        linewidth=2.5,
        label="BatchNorm",
        marker="o",
        markersize=4,
    )
    axes[0].plot(
        epochs_ln,
        ln_history["val_accuracy"],
        color="#F18F01",
        linewidth=2.5,
        label="LayerNorm",
        marker="s",
        markersize=4,
    )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Validation Accuracy", fontsize=12)
    axes[0].set_title(
        "Small Batch (size=4): Validation Accuracy", fontsize=14, fontweight="bold"
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot validation loss
    axes[1].plot(
        epochs_bn,
        bn_history["val_loss"],
        color="#A23B72",
        linewidth=2.5,
        label="BatchNorm",
        marker="o",
        markersize=4,
    )
    axes[1].plot(
        epochs_ln,
        ln_history["val_loss"],
        color="#F18F01",
        linewidth=2.5,
        label="LayerNorm",
        marker="s",
        markersize=4,
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation Loss", fontsize=12)
    axes[1].set_title(
        "Small Batch (size=4): Validation Loss", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Add text box with final metrics
    bn_final_acc = bn_history["val_accuracy"][-1]
    ln_final_acc = ln_history["val_accuracy"][-1]

    textstr = f"Final Validation Accuracy:\nBatchNorm: {bn_final_acc:.4f}\nLayerNorm: {ln_final_acc:.4f}\nDifference: {ln_final_acc - bn_final_acc:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axes[0].text(
        0.05,
        0.05,
        textstr,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Small batch comparison saved to {save_path}")
    plt.close()
