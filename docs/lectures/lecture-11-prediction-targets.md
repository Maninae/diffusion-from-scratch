# Lecture 11: Prediction Targets & Advanced Schedules
**Module:** 5 — Scaling Diffusion
**Estimated reading time:** 20 minutes
**Dependencies:** Lectures 3, 10

## Learning Objectives
- Compare three prediction parameterizations (ε, x_0, v) and their numerical stability properties
- Derive the conversion formulas between all three parameterizations
- Explain advanced noise schedule improvements: cosine, offset noise, zero terminal SNR, continuous-time
- Know when to use which prediction target and schedule combination in practice

## Narrative Arc
**The problem:** We've been predicting noise (ε-prediction) throughout this course — it's the DDPM default and it works well. But it has a flaw: at very low noise levels (t near 0), the noise is tiny and the loss signal is weak. At very high noise levels (t near T), predicting noise is numerically stable but the model has almost no signal to work with. Can we do better?

**The attempt:** An alternative: predict the clean image x_0 directly. This makes low-noise timesteps easy (x_t ≈ x_0, just predict what you see). But at high noise, it's asking the model to hallucinate a clean image from near-pure noise — the gradients are huge and unstable.

**The solution:** v-prediction (Salimans & Ho, 2022) defines velocity as v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0 — a blend of noise and signal that's numerically stable across ALL timesteps. Similarly, advanced noise schedules (offset noise, zero terminal SNR) fix subtle training-inference mismatches that cause artifacts at extreme pixel values. These refinements are what separate a good model from a production model.

## Section Outline

### Section 11.1: Three Prediction Parameterizations
All three are mathematically equivalent — you can convert between them given (x_t, t). They differ in numerical stability and gradient behavior.

**ε-prediction (DDPM standard):**
- Model predicts: the noise ε that was added to x_0
- Loss: `||ε - ε_θ(x_t, t)||²`
- Strengths: stable at high noise (t near T), well-studied, default choice
- Weakness: at low noise (t near 0), ε is tiny → loss signal is weak, training signal per step is small

**x_0-prediction:**
- Model predicts: the clean image x_0 directly
- Loss: `||x_0 - x̂_0(x_t, t)||²`
- Strengths: intuitive, good at low noise (x_t ≈ x_0)
- Weakness: at high noise, predicting x_0 from near-pure noise → large errors, unstable gradients

**v-prediction (Salimans & Ho, 2022):**
- Model predicts: velocity v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0
- Loss: `||v - v_θ(x_t, t)||²`
- Strengths: balanced across all timesteps, numerically stable at both extremes
- Used in: Imagen, SD 2.x, modern architectures
- Intuition: v interpolates between "predict noise" (high t) and "predict image" (low t) — adapting to whatever's easier at each timestep

### Section 11.2: Conversion Formulas

| From | To x_0 | To ε | To v |
|------|--------|------|------|
| ε | `x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t` | — | `v = √ᾱ_t·ε - √(1-ᾱ_t)·x_0` |
| x_0 | — | `ε = (x_t - √ᾱ_t·x_0) / √(1-ᾱ_t)` | `v = √ᾱ_t·ε - √(1-ᾱ_t)·x_0` |
| v | `x_0 = √ᾱ_t·x_t - √(1-ᾱ_t)·v` | `ε = √(1-ᾱ_t)·x_t + √ᾱ_t·v` | — |

Key identity: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` — this is always true (it's the forward process).

### Section 11.3: When to Use Which
- **ε-prediction:** default choice. Use when starting a new project or following DDPM recipes.
- **x_0-prediction:** useful when you want to inspect the model's "current best guess" at x_0 during sampling (e.g., for dynamic thresholding). Sometimes better with cosine schedule.
- **v-prediction:** use when training with schedules that approach zero terminal SNR, or when you need stability across all timesteps. The modern default for production systems.
- **Practical note:** the choice matters less than getting the rest of the pipeline right. Any parameterization works well with proper training.

### Section 11.4: Noise Schedule Improvements

**Cosine schedule (Nichol & Dhariwal 2021) — recap from Lecture 2:**
- Smoother SNR decay than linear
- ᾱ_t follows a cosine curve — information preserved longer
- Generally preferred over linear for all applications

**Offset noise (Lin et al., 2023):**
- Problem: standard Gaussian noise has zero mean per channel → the model can never generate images that are entirely dark or entirely bright (the mean is always near zero)
- Solution: add a per-channel bias to the noise:
  - `noise = randn_like(x) + 0.1 * randn(B, C, 1, 1)`
  - The `randn(B, C, 1, 1)` term shifts the channel mean
- Result: model can generate very dark or very bright images

**Zero terminal SNR (Lin et al., 2023):**
- Problem: some schedules don't actually reach SNR=0 at t=T — there's residual signal
- This means training uses x_T with trace signal, but inference starts from pure N(0, I) — a mismatch
- Solution: enforce that ᾱ_T = 0 exactly (SNR(T) = 0)
- Requires v-prediction (ε-prediction is undefined when ᾱ_T = 0)

**Log-SNR linear schedule:**
- Linear in log-SNR space → equal difficulty across timesteps
- Produces more uniform training signal

**Continuous-time schedules:**
- Instead of discrete t ∈ {0, 1, ..., T-1}, use continuous t ∈ [0, 1]
- Enables arbitrary step counts at inference without schedule modification
- Natural formulation for flow matching (Lecture 13)

### Section 11.5: Practical Guidance Summary
| Scenario | Prediction | Schedule | Notes |
|----------|-----------|----------|-------|
| Starting out (MNIST) | ε | Linear or cosine | Simplest, well-documented |
| Production model | v | Cosine + zero SNR | Best stability |
| Dark/bright images | v + offset noise | Cosine + zero SNR | Fixes mean-shift issue |
| Few-step sampling | v | Cosine + zero SNR | v-prediction + DDIM works best |

## Key Equations
- v-prediction target: `v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0`
- v-prediction loss: `L = E_{t,x_0,ε}[ ||v - v_θ(x_t, t)||² ]`
- Offset noise: `ε = ε_pixel + δ · ε_channel` where ε_channel ~ N(0, I) with shape (B, C, 1, 1)
- Zero terminal SNR: enforce ᾱ_T = 0
- SNR: `SNR(t) = ᾱ_t / (1 - ᾱ_t)`

## Code Examples
- Conversion functions between ε, x_0, v (pseudocode, then PyTorch reference)
- Training step with v-prediction — minimal change from ε-prediction
- Offset noise implementation (2 lines of code change)
- Schedule comparison: plot SNR curves for linear, cosine, cosine+zero-SNR

## Diagrams & Visuals
- **Stability comparison chart:** loss magnitude vs timestep for ε, x_0, and v prediction — show v is balanced
- **Conversion diagram:** triangle with ε, x_0, v at vertices, arrows with formulas
- **Offset noise comparison:** generated dark/bright images with and without offset noise
- **Schedule comparison plots:** ᾱ_t and log-SNR for linear, cosine, cosine+zero-SNR

## Source Material
- Module 09 §9.5 (v-Prediction vs ε-Prediction vs x_0 Prediction)
- Module 09 §9.6 (Noise Schedule Improvements)
- Module 05 §5.8-5.9 (SNR and schedules — foundational material from Lecture 2)
