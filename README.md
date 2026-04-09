# CS-DIP: Curvature-Steered Deep Image Prior

<p align="center">
  <strong>A Geometric Architecture for Self-Supervised Inverse Problems</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**CS-DIP** introduces a novel **Curvature-Modulated Convolution (CM-Conv)** layer that integrates differential geometry directly into the Deep Image Prior framework. Unlike standard DIP, which lacks mechanisms to distinguish structural details from noise, CS-DIP dynamically modulates convolutional filters based on the local **Gaussian** and **Mean** curvatures of the image manifold.

### Key Contributions

1. **CM-Conv Layer** — The first convolution layer to explicitly modulate filter responses based on differential geometric curvature of intermediate feature maps
2. **Structural Regularization** — Shifts from loss-based to architecture-based regularization, where the network structure itself inhibits noise artifacts
3. **No Early Stopping** — Eliminates the need for early stopping heuristics, providing stable convergence superior to standard DIP

> **Paper:** "Curvature-Steered Deep Image Prior (CS-DIP): A Geometric Architecture for Self-Supervised Inverse Problems"
> *IEEE Signal Processing Letters*

---

## Architecture

```
                        CS-DIP U-Net Architecture
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   Input z ──► [CM-Conv Block] ──────────────────────► [CM-Conv Block] ──► Output
    │   (noise)         │                  ▲                    │        (image)
    │                   ▼                  │ skip (feat+κ)      │
    │              [DownBlock]  ──────────────────────►  [UpBlock]
    │                   │                  ▲                    │
    │                   ▼                  │ skip (feat+κ)      │
    │              [DownBlock]  ──────────────────────►  [UpBlock]
    │                   │                  ▲                    │
    │                   ▼                  │ skip (feat+κ)      │
    │              [DownBlock]  ──────────────────────►  [UpBlock]
    │                   │                  ▲                    │
    │                   ▼                  │ skip (feat+κ)      │
    │              [DownBlock]  ──────────────────────►  [UpBlock]
    │                   │                                       │
    │                   └──► [Bottleneck CM-Conv Block] ────────┘
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

    CM-Conv Layer Detail:
    ┌─────────────────────────────────────────┐
    │  Input X ──► Curvature Map (K, H, κ)    │
    │         │                               │
    │         ├──► W_s * X  (structure path)  │
    │         │                               │
    │         └──► W_c * X  (curvature path)  │
    │                                         │
    │  Y = σ(κ)·(W_c*X) + (1-σ(κ))·(W_s*X)  │
    └─────────────────────────────────────────┘
```

**Skip connections** transfer both feature maps **and** curvature maps (κ) from encoder to decoder, preserving geometric consistency across scales.

---

## Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/cs-dip/cs-dip.git
cd cs-dip

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Quick Start

### Single Image Denoising

```bash
python scripts/train.py \
    --config configs/denoise_sigma25.yaml \
    --image path/to/clean_image.png \
    --output_dir results/
```

### Single Image Super-Resolution

```bash
python scripts/train.py \
    --config configs/sr_x2.yaml \
    --image path/to/hr_image.png \
    --output_dir results/
```

### Interactive Demo with Visualization

```bash
python scripts/demo.py \
    --config configs/denoise_sigma25.yaml \
    --image path/to/image.png \
    --show
```

This produces a side-by-side visualization with degraded input, restored output, curvature maps, and convergence curves.

### Benchmark Evaluation

```bash
# Download datasets first (see data/README.md)

python scripts/evaluate.py \
    --config configs/denoise_sigma25.yaml \
    --data_dir data/ \
    --dataset Set5
```

---

## Configurations

| Config File              | Task            | Key Parameters                  |
|--------------------------|-----------------|----------------------------------|
| `denoise_sigma25.yaml`   | Denoising σ=25  | 3000 iters, λ=0.01              |
| `denoise_sigma50.yaml`   | Denoising σ=50  | 3000 iters, λ=0.02              |
| `sr_x2.yaml`             | SR ×2           | 5000 iters, λ=0.005             |
| `sr_x4.yaml`             | SR ×4           | 5000 iters, λ=0.005             |

All configurations use Adam optimizer with lr=0.001 and a 5-scale U-Net (channels: 64→128→256→512→512).

---

## Project Structure

```
CS-DIP/
├── cs_dip/                     # Core package
│   ├── models/
│   │   ├── curvature.py        # Differential geometry (Sobel, K, H)
│   │   ├── cm_conv.py          # CM-Conv layer & block
│   │   └── cs_dip_net.py       # Full U-Net architecture
│   ├── losses/
│   │   └── losses.py           # Data fidelity + curvature consistency
│   ├── data/
│   │   └── datasets.py         # Benchmark dataset loaders
│   └── utils/
│       ├── metrics.py          # PSNR, SSIM
│       ├── degradation.py      # Noise, downsampling operators
│       └── io_utils.py         # Image I/O, seed management
├── scripts/
│   ├── train.py                # Single-image CS-DIP optimization
│   ├── evaluate.py             # Batch benchmark evaluation
│   └── demo.py                 # Visualization demo
├── configs/                    # YAML experiment configurations
├── tests/                      # Unit tests
├── data/                       # Datasets (download separately)
└── results/                    # Output directory
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_curvature.py -v
python -m pytest tests/test_cm_conv.py -v
python -m pytest tests/test_losses.py -v
python -m pytest tests/test_metrics.py -v
```

---

## Implementation Details

- **Framework:** PyTorch
- **Network Depth:** 5 encoder/decoder scales
- **Channels:** 64 → 128 → 256 → 512 → 512
- **Optimizer:** Adam (lr = 0.001)
- **Iterations:** 3,000 (denoising), 5,000 (super-resolution)
- **Curvature Computation:** Non-trainable Sobel filters for I_x, I_y, I_xx, I_xy, I_yy
- **Learnable Parameters:** α, β (curvature mixing weights), dual convolution kernels (W_s, W_c)

---

## References

1. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep image prior. *CVPR*.
2. Cheng, X., Xu, S., & Tao, W. (2024). An Effective Yet Fast Early Stopping Metric for Deep Image Prior in Image Denoising. *IEEE Signal Processing Letters*.
3. Elad, M. (2010). Sparse and redundant representations. *Springer*.
4. Cheng, K., Prasad, S., et al. (2024). Local Curvature Optimization for Self-Supervised Image Restoration. *IEEE Signal Processing Letters*, Vol. 31.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
