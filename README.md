<p align="center">
  <img src="assets/equation.png" alt="x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε" width="520">
</p>

<h1 align="center">Diffusion From Scratch</h1>

<p align="center">
  <em>Learn to reverse noise into images. One notebook at a time.</em>
</p>

<p align="center">
  <a href="#modules"><img alt="Modules" src="https://img.shields.io/badge/modules-11-E8A87C?style=flat-square"></a>
  <a href="#setup"><img alt="Hardware" src="https://img.shields.io/badge/runs_on-MPS_%7C_CPU-85CDCA?style=flat-square"></a>
  <a href="docs/papers.md"><img alt="Papers" src="https://img.shields.io/badge/papers-linked-6C63FF?style=flat-square"></a>
</p>

<p align="center">
  <img src="assets/hero.png" alt="Forward diffusion process: clean image → noise" width="720">
</p>

---

Diffusion models turn static into pictures. Add noise until an image is unrecognizable, then train a neural network to undo it — one step at a time. That's the engine behind Stable Diffusion, DALL·E 3, and Imagen.

This repo builds it all from scratch across 11 notebooks: forward process, U-Net, training loop, DDPM/DDIM sampling, classifier-free guidance, latent diffusion, DiT, and flow matching. Every line of code is yours to write and understand.

## Modules

| # | Module | What you'll build |
|---|--------|-------------------|
| 0 | NumPy Foundations | Broadcasting, einsum, vectorized operations |
| 1 | PyTorch Fundamentals | Autograd, nn.Module, custom layers |
| 2 | Convolutions | Conv2d from scratch, GroupNorm, residual blocks |
| 3 | Attention | Self-attention, multi-head, spatial attention |
| 4 | U-Net Architecture | Encoder-decoder with skip connections and time conditioning |
| 5 | Diffusion Math | Forward process, reverse process, ELBO, noise schedules |
| 6 | Training | Full training loop with EMA and loss monitoring |
| 7 | Sampling | DDPM, DDIM, and accelerated sampling |
| 8 | Conditioning & Guidance | Class conditioning and classifier-free guidance |
| 9 | Advanced Topics | Latent diffusion, DiT, flow matching |
| 10 | Interview Simulation | Timed exercises, verbal prompts, bug hunts |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Runs on Apple Silicon (MPS) or CPU. No CUDA required.

## Papers

Every concept links back to its source paper. See [docs/papers.md](docs/papers.md) for the full reading list with arxiv links.

## Structure

```
├── module_00 … module_10.ipynb   # The notebooks
├── utils/                        # Shared code (UNet, schedules, viz)
├── docs/                         # Build specs & paper references
├── checkpoints/                  # Saved models (gitignored)
└── data/                         # Datasets (gitignored)
```

---

<p align="center">
  <em>"All you need is noise and patience."</em>
</p>
