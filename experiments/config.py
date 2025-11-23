"""
Experiment configurations for Lab 3: Normalization Techniques

This file defines all experiments to be run, including:
- Baseline (no normalization)
- Batch normalization (custom and TF, multiple batch sizes)
- Layer normalization (custom and TF, multiple batch sizes)
- Weight normalization (custom only, multiple batch sizes)

Course: CS 599 Deep Learning
Author: Karl Reger
Date: November 2025
"""

# Seed derived from "Karl": K(75) + a(97) + r(114) + l(108) = 394
SEED = 394

# Shared configuration across all experiments
SHARED_CONFIG = {
    "epochs": 15,
    "learning_rate": 0.001,
    "optimizer": "adam",
}

# Define all experiments
# Each experiment will be run for all specified batch sizes
EXPERIMENTS = [
    {
        "name": "baseline",
        "norm_type": None,
        "batch_sizes": [128],  # Baseline only needs one batch size
        "verify": False,
        "description": "CNN without any normalization (baseline for comparison)",
    },
    {
        "name": "batchnorm",
        "norm_type": "batchnorm",
        "batch_sizes": [128, 4],  # Test normal and small batch
        "verify": True,
        "description": "Batch Normalization - normalizes across mini-batch",
    },
    {
        "name": "layernorm",
        "norm_type": "layernorm",
        "batch_sizes": [128, 4],  # Test normal and small batch
        "verify": True,
        "description": "Layer Normalization - normalizes across features",
    },
    {
        "name": "weightnorm",
        "norm_type": "weightnorm",
        "batch_sizes": [128, 4],  # Test normal and small batch
        "verify": False,  # No direct TF equivalent
        "description": "Weight Normalization - reparameterizes weight vectors",
    },
]

# LAB QUESTION MAPPING:
# Q1: "Compare with/without normalization"
#     → Compare baseline vs all normalization methods at bs=128
#
# Q2: "Compare custom vs TF implementations"
#     → Run verification for batchnorm and layernorm
#
# Q3: "Which is best and why?"
#     → Analyze all results across different dimensions:
#        - Convergence speed (epochs to reach target accuracy)
#        - Final accuracy
#        - Training stability (variance in loss curves)
#        - Computational cost (time per epoch)
#
# Q4: "Why LayerNorm better than BatchNorm?"
#     → Compare batchnorm_bs4 vs layernorm_bs4
#        LayerNorm should show much better stability with small batches
