# Normalization Techniques Comparison for Deep Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.14+](https://img.shields.io/badge/tensorflow-2.14+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation and comparison of three normalization techniques for deep learning:
- **Batch Normalization** (Ioffe & Szegedy, 2015)
- **Layer Normalization** (Ba et al., 2016)
- **Weight Normalization** (Salimans & Kingma, 2016)

This project was developed as part of CS 599: Foundations of Deep Learning at Northern Arizona University.

## 🔬 Key Findings

| Method | Test Accuracy (bs=128) | Test Accuracy (bs=4) | Training Time |
|--------|------------------------|----------------------|---------------|
| Baseline | 92.16% | - | 405s |
| BatchNorm | 91.50% | 92.08% | 502s / 12,505s |
| LayerNorm | **92.42%** | ~91.80% | 3,429s |
| WeightNorm | ~91.80% | ~91.20% | ~450s |

### Surprising Results

1. **LayerNorm achieved best overall accuracy** (92.42%)
2. **BatchNorm showed significant overfitting** at large batch size (99.31% train vs 91.50% test)
3. **BatchNorm improved at smaller batch sizes** due to enhanced regularization
4. **Baseline performed competitively** (92.16%), outperforming BatchNorm

## Architecture
```
Input (28×28×1)
    ↓
Conv2D (5×5×30) → [Normalization] → ReLU → MaxPool (2×2)
    ↓
Conv2D (5×5×60) → [Normalization] → ReLU → MaxPool (2×2)
    ↓
Flatten → Dense (100) → [Normalization] → ReLU
    ↓
Dense (10) → Softmax
```

## Installation

### Prerequisites
- Python 3.9+
- TensorFlow 2.14+ (with Metal plugin for Apple Silicon)

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/lab3-normalization-comparison.git
cd lab3-normalization-comparison

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### For Apple Silicon (M1/M2/M3)
```bash
pip install tensorflow-macos tensorflow-metal
```

## Usage

### Run All Experiments
```bash
python main.py
```

### Run Specific Experiments
```bash
# Run only baseline
python main.py --experiments baseline

# Run BatchNorm experiments
python main.py --experiments batchnorm

# Run with specific batch size
python main.py --experiments layernorm --batch-sizes 128

# Run verification only
python main.py --verify-only
```

### Configuration

Edit `experiments/config.py` to modify:
- Batch sizes
- Number of epochs
- Learning rate
- Network architecture

## 📁 Project Structure
```
lab3-normalization-comparison/
├── src/
│   ├── data.py            # Data loading & preprocessing for Fashion MNIST
│   ├── normalization.py   # Custom BatchNorm, LayerNorm, WeightNorm
│   ├── models.py          # CNN architecture with normalization
│   ├── train.py           # Training loop and metrics
│   └── utils.py           # Plotting and utilities
├── experiments/
│   ├── config.py          # Experiment configurations
│   └── run.py             # Experiment runner and verification
├── paper/                 # LaTeX paper and template
└── main.py                # Entry point
```

## Verification

All custom implementations are verified against TensorFlow's built-in functions:
```python
from experiments.run import verify_batchnorm_implementation

# Verify BatchNorm
results = verify_batchnorm_implementation(model, test_data)
print(f"Max forward error: {results['max_output_diff']}")  # < 1e-6
print(f"Max backward error: {results['max_grad_diff']}")   # < 1e-6
```

### Verification Results

| Method | Forward Error | Backward Error | Status |
|--------|---------------|----------------|--------|
| BatchNorm (bs=128) | 9.54e-07 | 2.62e-10 | ✅ Pass |
| BatchNorm (bs=4) | 3.87e-07 | 1.86e-09 | ✅ Pass |
| LayerNorm (bs=128) | 9.54e-07 | 3.49e-10 | ✅ Pass |


## Paper

The full analysis is available in the `paper/` directory.

## Implementation Details

### Batch Normalization
```python
# Forward pass
mu = tf.reduce_mean(x, axis=reduction_axes, keepdims=True)
var = tf.reduce_mean(tf.square(x - mu), axis=reduction_axes, keepdims=True)
x_norm = (x - mu) / tf.sqrt(var + epsilon)
output = gamma * x_norm + beta
```

### Layer Normalization
```python
# Forward pass (per-example normalization)
mu = tf.reduce_mean(x, axis=-1, keepdims=True)
var = tf.reduce_mean(tf.square(x - mu), axis=-1, keepdims=True)
x_norm = (x - mu) / tf.sqrt(var + epsilon)
output = gamma * x_norm + beta
```

### Weight Normalization
```python
# Weight reparameterization
w = g * v / tf.norm(v)
output = tf.matmul(x, w) + b
```

## References

1. Ioffe, S., & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). ICML.

2. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). arXiv.

3. Salimans, T., & Kingma, D. P. (2016). [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868). NeurIPS.

4. Xiao, H., Rasul, K., & Vollgraf, R. (2017). [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747). arXiv.


## Author

**Your Name**
- Course: CS 599 - Foundations of Deep Learning
- Institution: Northern Arizona University
- Semester: Fall 2025

## Acknowledgments

- Course instructor for the assignment design
- TensorFlow team for the deep learning framework
- Fashion-MNIST creators for the dataset
