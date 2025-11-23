"""
Verification script for TensorFlow Metal GPU support on Apple Silicon
Run this before starting the lab to ensure proper setup
"""

import sys

import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU Verification for Apple Silicon")
print("=" * 60)

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Check for GPU devices
gpus = tf.config.list_physical_devices("GPU")
print(f"\nNumber of GPUs detected: {len(gpus)}")

if gpus:
    print("\n✓ GPU detected successfully!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # Try a simple computation on GPU
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print("\n✓ GPU computation test passed!")
        print(f"  Result shape: {c.shape}")

    except Exception as e:
        print(f"\n✗ GPU computation test failed: {e}")
        sys.exit(1)
else:
    print("\n✗ No GPU detected!")
    print("  Please ensure tensorflow-metal is installed correctly")
    sys.exit(1)

print("\n" + "=" * 60)
print("Setup verified successfully! Ready to proceed.")
print("=" * 60)
