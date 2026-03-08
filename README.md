# Diffusion From Scratch

A comprehensive mini-course on diffusion models for image generation — implemented from scratch in Python, NumPy, and PyTorch.

## What Is This?

11 Jupyter notebook modules covering everything from NumPy fundamentals to latent diffusion and flow matching, with timed interview exercises. Built for hands-on learning: every concept is taught, then implemented, then exercised.

## Modules

| # | Module | Focus |
|---|--------|-------|
| 0 | NumPy Foundations | Broadcasting, einsum, vectorization |
| 1 | PyTorch Fundamentals | Autograd, nn.Module, custom layers |
| 2 | Convolutions | Conv layers, normalization, residual blocks |
| 3 | Attention | Self-attention, multi-head, spatial attention |
| 4 | U-Net Architecture | Encoder-decoder, skip connections, time conditioning |
| 5 | Diffusion Math | Forward process, reverse process, ELBO, noise schedules |
| 6 | Training | Full training loop, EMA, monitoring |
| 7 | Sampling | DDPM, DDIM, accelerated sampling |
| 8 | Conditioning & Guidance | Class conditioning, CFG, guidance scale |
| 9 | Advanced Topics | Latent diffusion, DiT, flow matching |
| 10 | Interview Simulation | Timed exercises, verbal prompts, code review |

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

## Structure

```
├── README.md
├── requirements.txt
├── module_00_numpy_foundations.ipynb
├── module_01_pytorch_fundamentals.ipynb
├── ...
├── utils/                  # Shared utilities
│   ├── visualization.py
│   ├── data.py
│   └── schedule.py
├── checkpoints/            # Saved models (gitignored)
├── data/                   # Datasets (gitignored)
├── assets/                 # Diagrams
└── docs/                   # Build specs & paper references
```

## Papers

See [docs/papers.md](docs/papers.md) for the complete reading list with arxiv links and a recommended reading order.

## Target Hardware

MacBook with Apple Silicon (MPS) or CPU. No CUDA required.
