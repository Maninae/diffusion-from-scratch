# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Jupyter notebook curriculum (11 modules) teaching diffusion models from scratch using PyTorch.

## Audience & Tone

**Target audience:** Strong undergrad — Stanford freshman/sophomore level. They know statistical ML concepts, can code in Python and NumPy, but are learning diffusion models for the first time.

**Writing style — this is critical:**
- Should read like a learner-friendly, elegantly presented undergrad course — not a PhD research paper
- Simple and incredibly educational, without sacrificing nuance, detail, or technical depth
- Math is important: equations should be explained and mentioned, but always balanced with practical, tangible examples
- Give tensor shapes and array shapes throughout (convs, attention, diffusion) — shapes ground the abstract in the concrete
- Instead of just statistical formulas, explain what's happening to the data: "pixel values slowly get erased, noise increases until it dominates"
- No massive text blocks — break into separate paragraphs with visual breathing room
- Use bullet points for series, sequences, lists, or sets of things
- Use **bold** sparingly to emphasize key points
- Every equation should be followed by intuition or a concrete example

## Efficiency

When working on this codebase, preserve tokens. Use subagents for parallel work across modules. Be thorough, make no mistakes, verify everything.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks via VS Code (select `.venv` kernel) or `jupyter notebook`.

**Hardware:** Apple Silicon (MPS) primary, CUDA supported, CPU fallback. Use `get_device()` from utils for auto-detection. No hardcoded CUDA assumptions.

## Architecture

**Notebooks** (`module_00` through `module_10`): Self-contained modules, ordered by dependency. Each follows: concept intro → worked example → exercise → solution → capstone. Later modules import from `utils/`.

**`utils/`**: Shared library imported by notebooks (modules don't import each other):
- `unet.py` — Full U-Net: `SinusoidalTimestepEmbedding`, `ResBlock`, `AttentionBlock`, `DownBlock`, `UpBlock`, `UNet`. Forward: `(x, t, class_label?) → predicted_noise`
- `diffusion.py` — Core diffusion ops: `q_sample()`, `train_step()`, `ddpm_sample()`, `prepare_schedule()`
- `schedule.py` — Noise schedules (`linear_schedule`, `cosine_schedule`, `sigmoid_schedule`). All return dicts of precomputed tensors (`betas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, etc.)
- `data.py` — `get_mnist_dataloader()`, `get_cifar10_dataloader()`, `get_device()`
- `visualization.py` — `show_images()`, `show_denoising_trajectory()`, `plot_loss_curve()`, `plot_schedule()`, `denormalize()`

**`docs/`**: Authoritative specs. `build-instructions.md` is the master build spec. Each `module-XX-*.md` defines that module's content. `papers.md` has all arxiv references.

## Code Conventions

- Type hints on function signatures, docstrings on all functions
- Shape comments after tensor operations: `# (B, C, H, W)`
- Meaningful variable names: `alpha_bar_t` not `ab`, `noise_pred` not `np`
- Device-agnostic: always use `device` variable, never hardcode device
- Reproducibility: `torch.manual_seed(42)` at notebook tops
- Images normalized to `[-1, 1]`; use `denormalize()` for display
- Visualization: `plt.style.use('seaborn-v0_8-whitegrid')`, always label axes

## Code Decomposition Principle

Notebook cells should focus on learning-critical code only. All boilerplate and repeated utility code belongs in `utils/`:
- **Visualization**: Use `show_images()`, `plot_loss_curve()`, etc. from `utils.visualization` — don't repeat matplotlib boilerplate in cells
- **Forward diffusion**: Use `q_sample()` from `utils.diffusion` — don't redefine it in every notebook
- **Training**: Use `train_step()` from `utils.diffusion` when the training loop itself isn't the learning goal
- **Sampling**: Use `ddpm_sample()` from `utils.diffusion` for standard DDPM generation
- **Schedule setup**: Use `prepare_schedule()` instead of inline `{k: v.to(device) for k, v in schedule.items()}`

The rule: if code is taught in one module, it should be imported from utils in all subsequent modules. Students implement it once to learn; after that, it's abstracted away so they can focus on new concepts.

## Module Build Order (dependency chain)

`utils/` → 0 → 1 → 2,3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

Module 4's U-Net is importable for Modules 6–10. Module 5's schedules live in `utils/schedule.py`. Module 6 saves checkpoints for Modules 7–8.

## No Test Suite

Validation is interactive — run notebook cells top-to-bottom. No pytest, no CI/CD.
