# A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias

This repository implements both **theoretical and empirical analyses** of optimization stability for paper **A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias**.  
It contains tools for small-scale synthetic experiments, large-scale image training (e.g., CIFAR-10), and spectral analysis of gradient and Hessian coherence.

---

## Structure

```
.
├── bypass_bn.py          # Utilities for freezing/unfreezing BatchNorm statistics (used with SAM)
├── coherence_local.py    # Synthetic (C, r)-generalization experiments; computes coherence, Hessian spectra, PCA rank
├── large_scale.py        # Large-scale CIFAR-10 experiments (ResNet18 + SAM/SGD); logs to Weights & Biases
├── nn_training.py        # Stability and unlearning analysis on synthetic datasets with analytic Hessians
├── sam.py                # SAM optimizer implementation (adapted from Davda et al. 2021)
└── README.md
```

---

## Installation

### Requirements
You can install the dependencies with:

```bash
pip install torch torchvision wandb numpy scipy scikit-learn tqdm pytorch-metric-learning torch-optimizer
```

---

## Usage Overview

### 1. Synthetic Coherence in linear stability setting (`coherence_local.py`)

This script explores **coherence** and **dynamics** in Synthetic setting (PSD matrices as data) as stated in theory part of the paper. To reproduce the figure 1 (a,b,c) in the paper, one can run the following command:

#### Example:
```bash
python coherence_local.py --mode SAM --rho 0.1 --experiment coherence --epochs 50
```

**Purpose:**
- Analyze divergence/stability boundary as a function of batch size and data coherence (`sigma`)
- Log transition between stable and divergent regimes

**Key arguments:**
| Argument | Default | Description |
|-----------|----------|-------------|
| `--mode` | `SGD` | Optimization method: `SGD` or `SAM` or `random`|
| `--rho` | `0.5` | Radius for SAM | 
| `--use_metric` | `False` | Enables additional contrastive metric learning term |
| `--lambda_metric` | `0.1` | Weight of contrastive regularization |

**Key wandb metrics:**
- Success indicator (stability)
- Divergence step
- Weight norm growth
- Heatmap table over `(batch_size, sigma)`

---

### 2. Stability in 2 layer neural network (`nn_training.py`)

The script experiments with the local and global optimization for different optimization (SGD and SAM) under 2 layer neural network with synthetic data. For local behavior, we initialize the model around the generalizing solution with perturbation and begin optimization to understand the converging behavior for SGD and SAM. For global dynamics, we randomly initialize the network and perform optimization directly and track different quantities during the process. Several quantities can be controlled test different combination of complexity of data and models and can be found in the script. Here, we provide example line ready to be used directly.

#### Example:
```bash
python nn_training.py --mode SAM --rho 0.5
```

**Key arguments:**
| Argument | Default | Description |
|-----------|----------|-------------|
| `--mode` | `SGD` | Optimization method: `SGD` or `SAM` or `random`|
| `--rho` | `0.5` | Radius for SAM |
| `--experiment` | `coherence` | Experiments method: `coherence` or `purturb` |

**Purpose:**
- Compute top eigenvalue of coherence operator analytically
- Log transition between stable and divergent regimes
- Check the coupling of the coherence and other optimization quantities.

**Key wandb metrics:**
- Coherence across steps
- Test / Training loss and accuracy across steps
- Largest eigenvlaue of samplewise Hessian

---

### 3. Large-Scale CIFAR-10 Experiments (`large_scale.py`)

Trains a **ResNet-18** with either **SGD** or **SAM**, measuring:
- Test accuracy  
- Feature rank  
- Largest eigenvalue of gradient similarity matrix  (Note: due to the computation limit, we restrict the calculation to analyze the gradient similarity instead of exact calculation of Hessian under subset of CIFAR10 dataset)

#### Example:
```bash
python large_scale.py --optimizer sam --rho 0.05 --epochs 200 --batch_size 128
```

**Logs include:**
- Training/validation accuracy  
- Gradient coherence eigenvalues  
- Feature PCA ranks (90%, 95%, 99%, 99.9% variance thresholds)

**Internals:**
- Uses `bypass_bn.py` to control BatchNorm momentum during SAM steps  
- Uses `sam.py` for sharpness-aware optimization  
- Logs experiments to Weights & Biases (`wandb`)

---


## Logging and Visualization

All scripts log results to **Weights & Biases** (https://wandb.ai/).  
You can visualize stability maps, eigenvalue trends, and coherence evolution across epochs.

To disable logging:
```bash
export WANDB_MODE=disabled
```