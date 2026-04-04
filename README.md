# Hybrid Multi-Objective GA–PSO Neural Architecture Search

> Automated neural architecture search using Genetic Algorithms, Particle Swarm Optimization, surrogate-assisted evaluation, and multi-objective Pareto optimization — runnable on a free Google Colab T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/nas-ga-pso/blob/main/Phase1_NAS_Setup.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Results at a Glance

| Dataset | Test Accuracy | Parameters | Search Evaluations | GPU Hours |
|---------|--------------|------------|-------------------|-----------|
| MNIST | **99.38%** | 239,898 (0.24M) | 65 | ~1.5 |
| CIFAR-10 | **80.85%** | 2,501,642 (2.50M) | 42 | ~3.1 |

**90% reduction** in architecture evaluations via surrogate active learning (650 → 65 on MNIST).  
**14 Pareto-optimal architectures** discovered on MNIST spanning 6,330 to 1.11M parameters.

---

## What This Project Does

Designing a neural network architecture manually is slow, expensive, and relies on expert intuition. This project automates the process using four components working together:

1. **Genetic Algorithm** — searches the discrete architecture space (layer counts, filter sizes, activations, pooling types) using tournament selection, uniform crossover, and integer mutation.
2. **Particle Swarm Optimization** — tunes continuous hyperparameters (learning rate, dropout, batch size) for each promising architecture, implemented from scratch in ~80 lines of NumPy.
3. **Surrogate Model** — a Random Forest regressor that predicts validation accuracy from architecture features, eliminating 90% of expensive training runs via an active learning loop.
4. **Multi-Objective Pareto Analysis** — uses NSGA-II to find architectures that balance accuracy, model size, and training speed simultaneously, giving a full trade-off curve instead of a single answer.

---

## System Architecture

```
Random Population
       │
       ▼
┌─────────────────┐
│  Genetic        │  ← Tournament Selection
│  Algorithm      │  ← Uniform Crossover
│  (Architecture) │  ← Integer Mutation
└────────┬────────┘
         │ candidate chromosomes
         ▼
┌─────────────────┐
│  Surrogate      │  ← Random Forest Regressor
│  Model          │  ← Active Learning Loop
│  (Filter)       │  ← Scores 200, trains top-5
└────────┬────────┘
         │ top-K candidates only
         ▼
┌─────────────────┐
│  PSO            │  ← 15 particles, 8 iterations
│  Hyperparameter │  ← Optimises: LR, dropout, batch size
│  Optimizer      │
└────────┬────────┘
         │ trained architectures
         ▼
┌─────────────────┐
│  NSGA-II        │  ← 3 objectives: accuracy, params, time
│  Pareto Front   │  ← 14 non-dominated solutions on MNIST
└─────────────────┘
```

---

## Repository Structure

```text
Hybrid-GA-PSO-Neural-Architecture-Search/
├── config/                     # Global configuration (paths, hyperparameters, experiment settings)
│   ├── __init__.py
│   └── config.py
│
├── data/                       # Datasets and dataset loaders
│   ├── MNIST/                  # MNIST data (or link/instructions)
│   └── cifar-10-batches-py/    # CIFAR‑10 data (or link/instructions)
│
├── evaluation/                 # Multi‑objective evaluation and Pareto utilities
│   ├── __init__.py
│   ├── multi_objective.py      # Objective definitions (accuracy, params, FLOPs, etc.)
│   └── pareto.py               # Pareto‑front construction and dominance checks
│
├── ga/                         # Genetic Algorithm components
│   ├── __init__.py
│   ├── genetic_algorithm.py    # GA loop (selection, crossover, mutation, replacement)
│   ├── operators.py            # Crossover/mutation operators
│   └── population.py           # Population representation and utilities
│
├── pso/                        # Particle Swarm Optimization components
│   ├── __init__.py
│   ├── pso_fitness.py          # PSO fitness evaluation
│   └── pso_optimizer.py        # PSO update rules and main loop
│
├── search_space/               # NAS search‑space definition
│   ├── __init__.py
│   ├── architecture_validator.py  # Validity checks for sampled architectures
│   ├── chromosome.py              # Chromosome / architecture encoding
│   └── search_space_utils.py      # Helper utilities for search‑space operations
│
├── surrogate/                  # Surrogate model and active learning
│   ├── __init__.py
│   ├── active_learning.py      # Active‑learning loop for querying new architectures
│   └── surrogate_model.py      # Surrogate model definition and training
│
├── training/                   # Proxy / final training pipeline
│   ├── __init__.py
│   └── proxy_trainer.py        # Training of candidate architectures with proxy budget
│
├── utils/                      # General utilities and helpers
│   └── __init__.py             # (Extend with logging, seeding, CLI helpers, etc.)
│
├── results/                    # Logs, metrics, and generated plots
│   ├── logs/                   # CSV/JSON logs from experiments
│   │   ├── ablation_table.csv
│   │   ├── best_config.json
│   │   ├── best_config_phase5.json
│   │   ├── cifar10_final.json
│   │   ├── cifar10_ga_results.json
│   │   ├── final_results.json
│   │   ├── ga_history.json
│   │   ├── ga_results.csv
│   │   ├── pareto_front.csv
│   │   ├── phase5_results.json
│   │   ├── pso_results.json
│   │   ├── random_search_results.json
│   │   └── unified_results_table.csv
│   │
│   └── plots/                  # Figures used in the paper/report
│       ├── ablation_study.png
│       ├── active_learning_results.png
│       ├── chromosome_schema.png
│       ├── convergence_analysis.png
│       ├── dataset_distribution.png
│       ├── dataset_samples.png
│       ├── final_comparison.png
│       ├── final_training_curve.png
│       ├── ga_convergence.png
│       ├── pareto_front.png
│       ├── pso_comparison.png
│       ├── pso_convergence.png
│       ├── search_space_distribution.png
│       ├── surrogate_final_quality.png
│       └── surrogate_seed_quality.png
│
├── README.md                   # Project description and usage
└── .gitignore                  # Git ignore rules
```

---

## Quickstart

### Option 1 — Google Colab (Recommended)

Click the badge at the top of this README. No installation needed — runs on free T4 GPU.

Run notebooks in order: Phase1 → Phase2 → Phase3 → Phase4 → Phase5 → Phase6 → Phase7.

### Option 2 — Local Setup

```bash
git clone https://github.com/yourusername/nas-ga-pso.git
cd nas-ga-pso
pip install -r requirements.txt
jupyter notebook Phase1_NAS_Setup.ipynb
```

### Requirements

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

## Phase-by-Phase Guide

| Phase | Notebook | What it does | Runtime |
|-------|----------|-------------|---------|
| 1 | `Phase1_NAS_Setup` | Installs dependencies, verifies GPU | 2 min |
| 2 | `Phase2_Chromosome_Encoding` | Defines 12-gene chromosome, `decode_chromosome()` | 5 min |
| 3 | `Phase3_Genetic_Algorithm` | Runs GA on MNIST (5 generations × 10 population) | 45 min |
| 4 | `Phase4_PSO_Optimizer` | PSO hyperparameter search (15 particles × 8 iterations) | 15 min |
| 5 | `Phase5_Surrogate_Model` | Trains Random Forest surrogate, 3 active learning rounds | 20 min |
| 6 | `Phase6_Pareto_Analysis` | NSGA-II multi-objective Pareto front extraction | 5 min |
| 7 | `Phase7_Experiments` | Ablation studies + CIFAR-10 search (`RUN_CIFAR=True` for overnight) | 30 min + 3 hr |

> **Tip:** Set `RUN_CIFAR = False` in Phase 7 to skip CIFAR-10 during the day and run it overnight.

---

## Architecture Encoding

Each candidate network is represented as a 12-gene integer chromosome:

| Gene | Name | Range | Options |
|------|------|-------|---------|
| 0 | `num_conv_layers` | 1–4 | Number of conv blocks |
| 1–4 | `filters_1..4_idx` | 0–5 | {16, 32, 64, 128, 256, 512} filters |
| 5 | `kernel_idx` | 0–2 | {3×3, 5×5, 7×7} kernels |
| 6 | `pool_idx` | 0–2 | {MaxPool, AvgPool, None} |
| 7 | `num_dense` | 1–3 | Dense layers |
| 8 | `dense_units_idx` | 0–4 | {64, 128, 256, 512, 1024} units |
| 9 | `activation_idx` | 0–3 | {ReLU, ELU, LeakyReLU, SELU} |
| 10 | `use_batchnorm` | 0–1 | Batch normalisation on/off |
| 11 | `use_skip` | 0–1 | Skip connections on/off |

Total search space: **~6.7 million** unique architecture configurations.

**Best MNIST architecture** `[4, 0, 1, 3, 0, 1, 1, 2, 2, 3, 1, 0]`:
- 4 conv blocks with 16→32→128→16 filters, 5×5 kernel, average pooling
- 2 dense layers × 256 units, SELU activation, batch normalisation
- 239,898 parameters → **99.38% test accuracy**

**Best CIFAR-10 architecture** `[3, 5, 2, 4, 0, 2, 1, 1, 0, 1, 0, 1]`:
- 3 conv blocks with 512→64→256 filters, 7×7 kernel, average pooling
- 1 dense layer × 64 units, ELU activation, skip connections
- 2,501,642 parameters → **80.85% test accuracy**

---

## Key Results

### Ablation Study (MNIST)

| Variant | Best Val Acc | Mean Val Acc | Compute Saving |
|---------|-------------|-------------|----------------|
| Random Search | 99.20% | 88.91% | 0% |
| GA Only | 99.30% | 96.05% | 0% |
| GA + PSO | 99.40%* | — | 0% |
| **Full System (Ours)** | **99.38%** | 99.38% | **90%** |

*20-epoch full training with PSO-tuned hyperparameters.

### Surrogate Savings

| Metric | Value |
|--------|-------|
| Candidates scored by surrogate | 600 |
| Real evaluations performed | 65 |
| Evaluations skipped | 585 |
| Compute saving | **90%** |
| Approx. GPU time saved | ~7.3 hours |

### CIFAR-10 GA Convergence

| Generation | Best Val Acc | Avg Val Acc |
|-----------|-------------|------------|
| 1 | 62.18% | 47.88% |
| 2 | 62.18% | 54.81% |
| 3 | 63.54% | 59.41% |
| 4 | 65.92% | 61.29% |
| 5 | 65.92% | 61.73% |

GA proxy best (65.92%) → Full 20-epoch training → **80.85% test accuracy**

---

## PSO Hyperparameters Found

| Parameter | Default | PSO-Tuned | Improvement |
|-----------|---------|-----------|-------------|
| Learning rate | 1×10⁻³ | 8.117×10⁻³ | +1.10% proxy acc |
| Dropout | 0.30 | 0.285 | — |
| Batch size | 64 | 32 | — |

---

## Pareto Front (MNIST — Top 5)

| Accuracy | Parameters | Train Time | Notes |
|----------|-----------|------------|-------|
| 99.30% | 506,794 | 48.0s | Best accuracy |
| 99.17% | 691,370 | 47.9s | — |
| 99.10% | 137,930 | 49.4s | Best compact high-accuracy |
| 98.68% | 239,898 | 46.2s | Best balanced (selected for full training) |
| 94.50% | 114,602 | 47.4s | Smallest competitive model |

Full 14-architecture Pareto front available in `results/pareto_front.csv`.

---

## Limitations

- CIFAR-10 accuracy (80.85%) is below state-of-the-art (>95%) due to the search space not including depthwise separable convolutions or attention mechanisms — the contribution of this work is the framework efficiency, not raw accuracy.
- Surrogate model is trained on MNIST data only and was not transferred to CIFAR-10 search.
- Population size of 10 over 5 generations is sufficient for demonstrating the approach but larger populations would improve coverage on harder tasks.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{yourname2026hybrid,
  title     = {Hybrid Multi-Objective GA--PSO Neural Architecture Search
               with Surrogate Fitness Prediction},
  author    = {Your Name},
  booktitle = {Proceedings of the International Joint Conference on
               Neural Networks (IJCNN)},
  year      = {2026}
}
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built and tested on Google Colab T4 GPU. Total search time: ~1.5 GPU-hours (MNIST) + ~3.1 GPU-hours (CIFAR-10).*
