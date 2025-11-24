"""
Experiment orchestrator for Lab 3: Normalization Techniques

This module runs all experiments defined in config.py and generates
all required plots and analysis.

Course: CS 599 Deep Learning
Author: Karl Reger
Date: Novemeber 2025
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
from experiments.config import EXPERIMENTS, SHARED_CONFIG
from src.data import create_dataloaders, load_fashion_mnist
from src.models import create_cnn
from src.train import evaluate_final_test, train_model
from src.utils import (
    load_results,
    plot_small_batch_comparison,
    plot_training_curves,
    plot_verification,
    save_results,
    setup,
)


def run_all():
    """
    Main experiment orchestrator.

    Executes all experiments defined in config.py with automatic result caching.
    If an experiment has already been run, it loads the cached results instead
    of re-running (saves significant time during development).

    WORKFLOW:
        1. Setup environment (seeds, GPU config)
        2. Load data once
        3. For each experiment configuration:
            a. Check if results exist (skip if yes)
            b. Create model with specified normalization
            c. Train model
            d. Evaluate on test set
            e. Verify implementation (if applicable)
            f. Save results
        4. Generate all plots
        5. Print summary
    """

    print("\n" + "=" * 70)
    print(" " * 15 + "LAB 3: NORMALIZATION TECHNIQUES")
    print(" " * 20 + "Experiment Runner")
    print("=" * 70 + "\n")

    # ========== SETUP ==========
    setup(seed_string="Karl")

    # Create directories
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # ========== LOAD DATA ==========
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    # Split test set into validation and test (50/50 split)
    # This gives us: 60k train, 5k val, 5k test
    val_size = len(x_test) // 2
    x_val, y_val = x_test[:val_size], y_test[:val_size]
    x_test_final, y_test_final = x_test[val_size:], y_test[val_size:]

    print(
        f"  Final split: {len(x_train)} train, {len(x_val)} val, {len(x_test_final)} test"
    )

    # ========== RUN EXPERIMENTS ==========
    all_results = {}
    total_experiments = sum(len(exp["batch_sizes"]) for exp in EXPERIMENTS)
    current_exp = 0

    for exp_config in EXPERIMENTS:
        for batch_size in exp_config["batch_sizes"]:
            current_exp += 1

            # Generate experiment ID
            exp_id = f"{exp_config['name']}_bs{batch_size}"
            results_file = f"results/metrics/{exp_id}.json"

            print("\n" + "=" * 70)
            print(f"EXPERIMENT {current_exp}/{total_experiments}: {exp_id}")
            print(f"Description: {exp_config['description']}")
            print("=" * 70)

            # Check if results already exist
            if os.path.exists(results_file):
                print(f"\n✓ Loading cached results from {results_file}")
                all_results[exp_id] = load_results(results_file)
                continue

            # Create dataloaders with specified batch size
            print(f"\nCreating dataloaders (batch_size={batch_size})...")
            train_ds = create_dataloaders(x_train, y_train, batch_size, shuffle=True)
            val_ds = create_dataloaders(x_val, y_val, batch_size, shuffle=False)
            test_ds = create_dataloaders(
                x_test_final, y_test_final, batch_size, shuffle=False
            )

            # Create model
            print(
                f"\nCreating model with {exp_config['norm_type'] or 'no'} normalization..."
            )
            model, norm_layers = create_cnn(
                norm_type=exp_config["norm_type"],
                use_custom=True,  # Always use custom for this lab
                input_shape=(28, 28, 1),
                num_classes=10,
            )

            # Train model
            print("\nTraining model...")
            start_time = time.time()

            history = train_model(
                model=model,
                norm_layers=norm_layers,
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=SHARED_CONFIG["epochs"],
                learning_rate=SHARED_CONFIG["learning_rate"],
            )

            training_time = time.time() - start_time
            print(f"\nTraining completed in {training_time:.2f} seconds")

            # Evaluate on test set
            test_metrics = evaluate_final_test(model, test_ds)

            # Verify implementation if requested
            verification = None
            if exp_config.get("verify", False) and norm_layers:
                print("\n" + "-" * 70)
                print("VERIFYING IMPLEMENTATION")
                print("-" * 70)

                verification = {}

                if exp_config["norm_type"] == "weightnorm":
                    # WeightNorm verification checks mathematical properties
                    for i, norm_layer in enumerate(norm_layers):
                        print(f"  Verifying WeightNorm layer {i}...")
                        if hasattr(norm_layer, "verify"):
                            layer_verification = norm_layer.verify()
                            verification[f"layer_{i}"] = layer_verification

                            print(
                                f"    Norm property error: {layer_verification['norm_property_error']:.2e}"
                            )
                            print(
                                f"    Direction property error: {layer_verification['direction_property_error']:.2e}"
                            )

                            if layer_verification["norm_property_error"] < 1e-5:
                                print("    ✓ Verification PASSED (error < 1e-5)")
                            else:
                                print(
                                    f"    ⚠ Warning: error = {layer_verification['norm_property_error']:.2e}"
                                )
                else:
                    # BatchNorm/LayerNorm verification compares against TF
                    # Create test inputs with the correct shapes for each normalization layer
                    print("  Getting intermediate layer outputs for verification...")

                    test_input = tf.random.normal([batch_size, 28, 28, 1])

                    # Define expected input shapes for each normalization layer
                    # Based on CNN architecture: Conv→BN→ReLU→Pool → Conv→BN→ReLU→Pool → Flatten→Dense→BN→ReLU
                    if exp_config["norm_type"] == "batchnorm":
                        # bn1: after conv1 (28x28x30)
                        # bn2: after conv2 (14x14x60)
                        # bn3: after dense1 (100 units)
                        test_shapes = [
                            (batch_size, 28, 28, 30),
                            (batch_size, 14, 14, 60),
                            (
                                batch_size,
                                100,
                            ),  # ← FIXED: after Dense(100), not after Flatten
                        ]
                    elif exp_config["norm_type"] == "layernorm":
                        # Same shapes as BatchNorm
                        test_shapes = [
                            (batch_size, 28, 28, 30),
                            (batch_size, 14, 14, 60),
                            (
                                batch_size,
                                100,
                            ),  # ← FIXED: after Dense(100), not after Flatten
                        ]
                    else:
                        test_shapes = []

                    for i, norm_layer in enumerate(norm_layers):
                        if i < len(test_shapes):
                            print(
                                f"  Verifying layer {i} with shape {test_shapes[i]}..."
                            )

                            # Create test input with correct shape
                            layer_test_input = tf.random.normal(test_shapes[i])

                            if hasattr(norm_layer, "verify"):
                                try:
                                    layer_verification = norm_layer.verify(
                                        layer_test_input
                                    )
                                    verification[f"layer_{i}"] = layer_verification

                                    if "max_output_diff" in layer_verification:
                                        print(
                                            f"    Max output diff: {layer_verification['max_output_diff']:.2e}"
                                        )
                                        print(
                                            f"    Max gradient diff: {layer_verification['max_grad_diff']:.2e}"
                                        )

                                        if layer_verification["max_output_diff"] < 1e-6:
                                            print(
                                                "    ✓ Verification PASSED (output diff < 1e-6)"
                                            )
                                        else:
                                            print(
                                                f"    ⚠ Verification issue (output diff = {layer_verification['max_output_diff']:.2e})"
                                            )
                                except Exception as e:
                                    print(f"    ⚠ Verification error: {e}")
                                    import traceback

                                    traceback.print_exc()
                                    verification[f"layer_{i}"] = {"error": str(e)}
                        else:
                            print(
                                f"  Skipping verification for layer {i} (no shape definition)"
                            )

            # Store results
            results = {
                "config": exp_config,
                "batch_size": batch_size,
                "history": history,
                "test_metrics": test_metrics,
                "training_time": training_time,
                "verification": verification,
            }

            save_results(results, results_file)
            all_results[exp_id] = results

            print(f"\n✓ Experiment {exp_id} complete!")

    # ========== GENERATE PLOTS ==========
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Plot 1: Training curves for all methods (bs=128)
    print("\n1. Plotting training curves (all methods, batch_size=128)...")
    plot_training_curves(all_results, "plots/convergence_all.png")

    # Plot 2: Verification plots
    for exp_id, results in all_results.items():
        if results.get("verification"):
            norm_type = results["config"]["norm_type"]
            print(f"\n2. Plotting verification for {exp_id}...")

            if norm_type == "weightnorm":
                from src.utils import plot_weightnorm_verification

                plot_weightnorm_verification(
                    results["verification"], f"plots/verification_{exp_id}.png"
                )
            else:
                plot_verification(
                    results["verification"],
                    f"plots/verification_{exp_id}.png",
                    norm_type,
                )

    # Plot 3: Small batch comparison (if both exist)
    bn_small = all_results.get("batchnorm_bs4")
    ln_small = all_results.get("layernorm_bs4")
    if bn_small and ln_small:
        print("\n3. Plotting small batch comparison...")
        plot_small_batch_comparison(
            bn_small, ln_small, "plots/small_batch_comparison.png"
        )

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Experiment':<25} {'Batch Size':<12} {'Final Test Acc':<15} {'Training Time'}"
    )
    print("-" * 70)

    for exp_id, results in sorted(all_results.items()):
        test_acc = results["test_metrics"]["test_accuracy"]
        train_time = results["training_time"]
        batch_size = results["batch_size"]

        print(
            f"{exp_id:<25} {batch_size:<12} {test_acc:>6.4f} ({test_acc * 100:>5.2f}%)  {train_time:>7.1f}s"
        )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\n✓ Results saved in: results/metrics/")
    print("✓ Plots saved in: plots/")
    print("\nYou can now write your LaTeX report using these results.\n")

    # ========== LAB QUESTION GUIDANCE ==========
    print("\n" + "=" * 70)
    print("GUIDANCE FOR ANSWERING LAB QUESTIONS")
    print("=" * 70)

    print("\nQ1: Compare Results with and without Normalization")
    print("    → See: plots/convergence_all.png")
    print(
        "    → Compare: baseline_bs128 vs batchnorm_bs128, layernorm_bs128, weightnorm_bs128"
    )
    print("    → Note: Convergence speed, final accuracy, training stability")

    print("\nQ2: Compare custom vs TensorFlow implementations")
    print("    → See: plots/verification_*.png")
    print("    → Check: results/metrics/*_bs128.json (verification section)")
    print("    → Expected: Output/gradient differences < 1e-6")

    print("\nQ3: Which normalization is best and why?")
    print("    → Analyze: All experiments across multiple dimensions")
    print("    → Consider: Accuracy, speed, stability, batch size sensitivity")
    print("    → Trade-offs: BatchNorm (best with large batches) vs LayerNorm (robust)")

    print("\nQ4: Why is LayerNorm better than BatchNorm?")
    print("    → See: plots/small_batch_comparison.png")
    print("    → Compare: batchnorm_bs4 vs layernorm_bs4")
    print("    → Key insight: LN stable with small batches, BN degrades\n")


if __name__ == "__main__":
    run_all()
