# Lecture 7: Sampling — DDPM, DDIM & Beyond
**Module:** 3 — Training & Sampling
**Estimated reading time:** 30 minutes
**Dependencies:** Lectures 3 (μ_θ formula from §3.5), 6 (trained model checkpoint)

## Learning Objectives
- Implement DDPM sampling (Algorithm 2): iterative denoising from x_T to x_0
- Explain why noise is added during stochastic sampling and what happens without it
- Implement DDIM sampling with the η parameter that interpolates between stochastic and deterministic
- Use timestep sub-selection to accelerate DDIM from 1000 steps to 50 or fewer
- Understand the Probability Flow ODE perspective and its connection to advanced solvers (optional/advanced)

## Narrative Arc
**The problem:** We've trained a model that can predict noise at any timestep (Lecture 6). Now we need to generate images — start from pure noise x_T and iteratively denoise to x_0. DDPM's Algorithm 2 does this, but it requires 1000 sequential forward passes. Generating a single image takes seconds on a GPU, minutes on CPU. For practical applications, this is too slow.

**The attempt:** DDPM sampling is stochastic — it adds fresh noise at each step. This produces diverse samples but prevents step-skipping. You can't just run 50 steps instead of 1000 because the noise injection at each step is calibrated for the full 1000-step schedule.

**The solution:** DDIM (Song et al., 2020) reformulates sampling as a non-Markovian process with a tunable parameter η. When η=0, sampling becomes fully deterministic — a smooth trajectory from noise to data. This deterministic path can be traversed in arbitrary step counts. Using 50 DDIM steps gives nearly identical quality to 1000 DDPM steps, at 20× the speed.

## Section Outline

### Section 7.0: Loading the Trained Model
- Load the checkpoint saved in Lecture 6: model weights, EMA weights, schedule, step number
- Use EMA weights for all sampling (smoother, higher-quality samples)
- Verify the model generates reasonable outputs before proceeding

**Pseudocode:**
```
checkpoint = torch.load('diffusion_mnist.pt', map_location=device)
model.load_state_dict(checkpoint['ema'])   # use EMA weights for sampling
schedule = checkpoint['schedule']
model.eval()                                # disable dropout
```

### Section 7.1: DDPM Sampling — Algorithm 2
- Start from pure noise: `x_T ~ N(0, I)`
- For t = T, T-1, ..., 1:
  1. Predict noise: `ε_θ = model(x_t, t)`
  2. Compute predicted mean: `μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) · ε_θ)`
  3. Sample: `x_{t-1} = μ_θ + σ_t · z` where z ~ N(0, I)
  4. At t=1: no noise added (z = 0) — final step is deterministic
- Two variance choices:
  - `σ_t² = β_t` (DDPM default — simpler)
  - `σ_t² = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t` (posterior variance — often better)

**Pseudocode:**
```
def ddpm_sample(model, shape, schedule, T=1000):
    x = randn(shape)                        # x_T ~ N(0, I)

    for t in reversed(range(1, T+1)):
        noise_pred = model(x, t)             # predict ε_θ

        # Compute mean
        coeff1 = 1 / sqrt(schedule.alphas[t])
        coeff2 = schedule.betas[t] / sqrt(1 - schedule.alphas_cumprod[t])
        mean = coeff1 * (x - coeff2 * noise_pred)

        # Add noise (except at final step)
        if t > 1:
            noise = randn_like(x)
            sigma = sqrt(schedule.betas[t])
            x = mean + sigma * noise
        else:
            x = mean

    return x                                 # x_0 prediction
```

### Section 7.2: Why We Add Noise During Sampling
- Counterintuitive: we're trying to remove noise, but we add MORE noise at each step
- **Stochastic sampling provides exploration:** the noise lets the model correct errors from previous steps. Without it, any early mistake compounds.
- **Diversity:** same starting x_T → different samples each time (different noise injections)
- **Without noise (σ_t = 0):** deterministic mapping from x_T → x_0. Same x_T always gives same x_0. This is DDIM (η=0).
- Stochastic → more diverse samples. Deterministic → reproducible samples, enables interpolation.

### Section 7.3: DDIM — Deterministic Sampling
- DDIM generalizes DDPM with parameter η ∈ [0, 1]:
  - η = 1: equivalent to DDPM (full stochastic)
  - η = 0: fully deterministic
- The DDIM update rule:
  1. Predict x_0: `x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t`
  2. Compute direction: `dir = √(1-ᾱ_{t-1} - σ_t²) · ε_θ`
  3. Compute noise: `σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1})`
  4. Step: `x_{t-1} = √ᾱ_{t-1} · x̂_0 + dir + σ_t · z`
- Key insight: DDIM defines a non-Markovian forward process that has the SAME marginals q(x_t | x_0) as DDPM — so the same trained model works for both!
- No retraining needed — DDIM is a drop-in replacement for the DDPM sampler

**Pseudocode:**
```
def ddim_sample(model, shape, schedule, num_steps=50, eta=0.0):
    # Select timestep subsequence
    timesteps = linspace(0, T-1, num_steps, dtype=int)

    x = randn(shape)                         # x_T ~ N(0, I)

    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_prev = timesteps[i-1] if i > 0 else 0

        noise_pred = model(x, t)

        # Predict x_0
        alpha_bar_t = schedule.alphas_cumprod[t]
        alpha_bar_prev = schedule.alphas_cumprod[t_prev]
        x0_pred = (x - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

        # Compute sigma
        sigma = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                      sqrt(1 - alpha_bar_t / alpha_bar_prev)

        # Direction pointing to x_t
        dir = sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred

        # DDIM step
        noise = randn_like(x) if t > 0 else 0
        x = sqrt(alpha_bar_prev) * x0_pred + dir + sigma * noise

    return x
```

### Section 7.4: Fewer Steps — Accelerated Sampling
- DDPM needs all T=1000 steps — each step depends on the previous via noise injection
- DDIM with η=0 follows a smooth trajectory — you can sample it at any granularity
- **Timestep sub-selection:** instead of [999, 998, ..., 0], use [999, 979, 959, ..., 19, 0]
  - Uniform spacing: `timesteps = linspace(0, T-1, num_steps)`
  - Quadratic spacing: more steps near t=0 (where detail emerges)
- Quality vs speed tradeoff:
  - 1000 steps: best quality (baseline)
  - 200 steps: nearly identical quality
  - 50 steps: minimal quality loss — the sweet spot
  - 20 steps: visible degradation but still recognizable
  - 10 steps: significant quality loss
- This is why DDIM was a breakthrough — 20× faster generation from the same trained model

### Section 7.5: Dynamic Thresholding (Advanced)
- At high guidance scales (Lecture 9), predicted x_0 can overshoot [-1, 1]
- Standard clipping: `x0_pred.clamp(-1, 1)` — crude, can cause artifacts
- **Dynamic thresholding (Imagen):** at each step, compute the p-th percentile of |x0_pred|, clip to that, then rescale to [-1, 1]
- Enables higher guidance scales without color saturation artifacts

### Section 7.6: The Full Sampling Pipeline
- Complete end-to-end:
  1. Generate x_T ~ N(0, I) with shape (B, C, H, W)
  2. Iterative denoising (DDPM or DDIM)
  3. Clamp to [-1, 1]: `x_0 = x_0.clamp(-1, 1)`
  4. Denormalize: `x_display = (x_0 + 1) / 2` — now in [0, 1]
  5. Display as image grid

### Section 7.7: The Probability Flow ODE (Optional/Advanced)
- Two equivalent views of the same sampling process:
  1. **SDE (stochastic):** DDPM — random noise at each step
  2. **ODE (deterministic):** smooth trajectory from x_T to x_0
- The ODE: `dx = [f(x,t) - ½g(t)² ∇_x log p_t(x)] dt`
  - The score function: `∇_x log p_t(x) ≈ -ε_θ(x_t, t) / √(1-ᾱ_t)`
- DDIM (η=0) is the Euler discretization of this ODE
- Advanced ODE solvers (DPM-Solver, Heun's method) can achieve good quality in even fewer steps (10-20)
- This perspective unifies diffusion with score-based models and flow matching (Lecture 13)

## Key Equations
- DDPM mean: `μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) · ε_θ)`
- DDPM sampling: `x_{t-1} = μ_θ + σ_t · z`
- DDIM x_0 prediction: `x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t`
- DDIM step: `x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}-σ²) · ε_θ + σ · z`
- Score function: `∇_x log p_t(x) ≈ -ε_θ / √(1-ᾱ_t)`

## Code Examples
- `ddpm_sample()` — complete DDPM sampling loop (pseudocode first, PyTorch reference)
- `ddim_sample()` — DDIM with eta parameter and timestep sub-selection
- Denoising trajectory visualization: show x_t at selected timesteps during sampling
- Speed comparison: wall-clock time for 1000 vs 50 vs 20 steps

## Diagrams & Visuals
- **Denoising trajectory strip:** x_T → x_750 → x_500 → x_250 → x_100 → x_50 → x_0 for one sample
- **DDPM vs DDIM comparison grid:** same starting noise, samples at 1000 DDPM steps vs 50 DDIM steps
- **Steps vs quality plot:** sample quality (visual or FID-proxy) as a function of step count
- **Stochastic vs deterministic:** same x_T → multiple runs with DDPM (different each time) vs DDIM (identical)
- **ODE trajectory visualization (optional):** smooth curve in a 2D latent space from noise to data

## Source Material
- Module 07 §7.1 (DDPM Sampling / Algorithm 2)
- Module 07 §7.2 (Why We Add Noise)
- Module 07 §7.3 (DDIM)
- Module 07 §7.4 (Fewer Steps / Accelerated Sampling)
- Module 07 §7.5 (Noise Schedules at Inference)
- Module 07 §7.6 (Full Sampling Pipeline)
- Module 07 §7.7 (Probability Flow ODE)
