# Module 7: Sampling & Inference

## Purpose
Implement the reverse process — going from pure noise to clean images. Understand both stochastic (DDPM) and deterministic (DDIM) sampling, and the tradeoffs between quality and speed.

## 📄 Key Papers
- **DDPM** — Ho et al. 2020. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
  - *Read for:* Algorithm 2 (sampling procedure). The full T-step reverse process.
- **Denoising Diffusion Implicit Models (DDIM)** — Song et al. 2020. [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
  - *Read for:* The deterministic sampling alternative. Section 4 for the generalized forward process, Equation 12 for the DDIM update rule. The η parameter that interpolates between DDPM and DDIM.
- **Progressive Distillation for Fast Sampling of Diffusion Models** — Salimans & Ho 2022. [arxiv.org/abs/2202.00512](https://arxiv.org/abs/2202.00512)
  - *Read for:* Reducing sampling steps through distillation. Shows you can get good samples in 4-8 steps.

## Sections

### 7.1 — DDPM Sampling (Algorithm 2)

**Concepts to teach:**
- Start from `x_T ~ N(0, I)` — pure random noise
- For t = T, T-1, ..., 1:
  1. Predict noise: `ε_θ = model(x_t, t)`
  2. Compute predicted mean: `μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) ε_θ)`
  3. Sample: `x_{t-1} = μ_θ + σ_t z` where `z ~ N(0, I)` (except at t=1, where z=0)
- σ_t: the variance of the reverse step. Two common choices:
  - `σ_t² = β_t` (DDPM default)
  - `σ_t² = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t` (posterior variance, often better)
- The full loop takes T=1000 forward passes — slow!

**Worked example:**
- Implement `ddpm_sample(model, shape, noise_schedule, T)` — the complete sampling loop
- Generate samples from a trained model, visualize the denoising trajectory (show x_t at t=1000, 750, 500, 250, 100, 50, 10, 0)

**Exercise:**
- Implement DDPM sampling with both variance choices
- Generate a grid of samples, verify they look like training data

---

### 7.2 — Why We Add Noise During Sampling

**Concepts to teach:**
- Counterintuitive: during sampling we ADD noise at each step (the σ_t z term)
- Stochastic sampling: the added noise provides exploration — helps the model correct errors
- Without noise (σ_t = 0): deterministic mapping from x_T → x_0 (this is DDIM)
- Stochastic sampling produces more diverse samples
- Deterministic sampling allows: interpolation in latent space, exact reconstruction

**Worked example:**
- Generate samples with and without the noise term — compare diversity
- Show that same x_T with noise → different x_0 each time; without noise → same x_0

---

### 7.3 — DDIM: The Deterministic ODE Formulation

**Concepts to teach:**
- DDIM generalizes DDPM: introduces parameter η ∈ [0, 1]
  - η = 1: equivalent to DDPM (full stochastic)
  - η = 0: fully deterministic (the DDIM update)
- DDIM update rule:
  ```
  predicted_x0 = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t
  direction = √(1-ᾱ_{t-1} - σ_t²) · ε_θ
  noise = σ_t · z
  x_{t-1} = √ᾱ_{t-1} · predicted_x0 + direction + noise
  ```
  where `σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})`
- Key insight: DDIM defines a non-Markovian forward process that has the SAME marginals q(x_t | x_0) as DDPM
- The ODE perspective: when η=0, sampling follows a probability flow ODE — a smooth trajectory from noise to data

**Worked example:**
- Implement `ddim_sample(model, shape, noise_schedule, T, eta)`
- Show that eta=1 matches DDPM, eta=0 is deterministic
- Demonstrate: same noise seed with eta=0 always produces the same image

**Exercise:**
- Implement DDIM sampling, sweep η from 0 to 1 and visualize the effect on sample diversity

---

### 7.4 — Fewer Steps: Accelerated Sampling

**Concepts to teach:**
- DDPM needs 1000 steps → slow (1000 model forward passes per image)
- DDIM allows using a **subset of timesteps** — e.g., [0, 50, 100, ..., 950, 1000]
- With 50 steps: ~20× faster, minimal quality loss
- With 10 steps: still recognizable but noticeable quality drop
- How to select the subset: uniform spacing (most common), quadratic spacing, custom
- The quality-speed tradeoff: more steps = better quality, fewer steps = faster generation

**Worked example:**
- Generate the same image with 1000, 200, 50, 20, 10 steps — compare quality side by side
- Measure wall-clock time for each
- Plot quality (visual or simple metric) vs number of steps

**Exercise:**
- Implement timestep sub-selection for DDIM
- Find the minimum number of steps for "acceptable" quality on your trained model

---

### 7.5 — Noise Schedules at Inference

**Concepts to teach:**
- The noise schedule at inference doesn't HAVE to match training — but usually does
- For DDIM with reduced steps: the schedule values at the selected timesteps matter
- Rescaling: when using fewer steps, some papers rescale the schedule
- Practical advice: just use the same schedule as training with uniformly spaced timestep subsets

---

### 7.6 — Implementation: Full Sampling Pipeline

**Concepts to teach:**
- Complete sampling function: noise generation → iterative denoising → clip to [-1, 1] → denormalize to [0, 1]
- Clipping: `x_0.clamp(-1, 1)` — sometimes done at each step or only at the end
- Dynamic thresholding (from Imagen): clip x_0 prediction at each step to a percentile, then rescale — improves high-guidance samples
- Batch sampling: generate multiple images at once for efficiency
- Visualization: denoising trajectory as an animation or grid

**Worked example:**
- Full end-to-end sampling pipeline
- Generate a 4×4 grid of samples
- Visualize the denoising process for one sample as a row of images

**Exercise:**
- Implement the full pipeline with both DDPM and DDIM options
- Generate samples, save to disk

---

### 7.7 — Ancestral Sampling vs Probability Flow ODE

**Concepts to teach:**
- Two equivalent perspectives on the same process:
  1. **Ancestral sampling (SDE):** stochastic, follows a Stochastic Differential Equation
  2. **Probability flow ODE:** deterministic, follows an Ordinary Differential Equation
- Both produce samples from the same distribution p_θ(x_0)
- The ODE formulation: `dx = [f(x,t) - ½g(t)² ∇_x log p_t(x)] dt`
  - f and g are the drift and diffusion coefficients of the forward SDE
  - ∇_x log p_t(x) ≈ -ε_θ(x_t, t) / √(1-ᾱ_t) — the score function
- DDIM (η=0) is the Euler discretization of this ODE
- Advanced ODE solvers (Heun, DPM-Solver) can sample in even fewer steps

**Worked example:**
- Implement the probability flow ODE using `torchdiffeq` or manual Euler method
- Compare to DDIM — they should produce very similar results

**Exercise:**
- Implement Euler ODE sampling, compare quality and speed to DDIM

---

## Module 7 Capstone Exercise

**Implement both DDPM and DDIM samplers, comprehensive comparison:**
- Generate 64 samples with each sampler
- DDIM: test with 1000, 100, 50, 20, 10 steps
- Compare: visual quality, diversity, wall-clock time
- Show denoising trajectories for both
- Demonstrate deterministic property of DDIM: same seed → same image
- Demonstrate stochastic property of DDPM: same seed → different images (due to noise injection)
