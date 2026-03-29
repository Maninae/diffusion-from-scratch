# Lecture 13: Flow Matching & Rectified Flows
**Module:** 6 — Modern Architectures
**Estimated reading time:** 25 minutes
**Dependencies:** Lecture 7 (sampling / ODE perspective)

## Learning Objectives
- Articulate DDPM's complexity baggage: schedules, cumulative products, ELBO chain, 1000 steps
- Explain velocity fields as the conceptual bridge from DDPM to flow matching, and how v-prediction was a partial step in this direction
- Derive the flow matching training objective and show it's dramatically simpler than DDPM
- Describe rectified flows and how iterative straightening reduces sampling cost
- Connect flow matching to DDPM as a continuous generalization

## Narrative Arc
**The problem:** DDPM works, but it carries baggage. You need a carefully designed noise schedule (linear? cosine? sigmoid?), cumulative products of alphas, a loss derived through a long chain of ELBO math, and 50-1000 sampling steps. The framework is powerful but complex. Is there a simpler way to get from noise to data?

**The attempt:** Velocity fields offer a different perspective. Instead of "add noise step by step, then learn to reverse it," think of generation as transporting particles from a noise distribution to a data distribution. Define a velocity field v(x, t) that tells each particle which direction to move. v-prediction (Lecture 11) was a step in this direction — it balanced the numerical instabilities of ε-prediction at boundary timesteps. But v-prediction was still a patch on top of the DDPM framework: you still needed noise schedules, cumulative alpha products, and all the forward/reverse process machinery. The velocity idea was right, but the framework around it was holding it back.

**The solution:** Flow matching (Lipman et al., 2023) takes the velocity field idea to its logical conclusion and throws away the rest. Define straight-line interpolation paths from data to noise: x_t = (1-t) · x_0 + t · x_1. The velocity along each path is simply (x_1 - x_0) — constant, trivial to compute. The training loss: predict this velocity. No noise schedule, no cumulative alpha products, no ELBO derivation. Just interpolate, predict velocity, minimize MSE. Rectified flows further straighten the learned paths, enabling high-quality generation in very few steps. This is the framework behind Stable Diffusion 3 and Flux.

## Section Outline

### Section 13.1: DDPM's Baggage — What We're Trying to Escape
Before introducing flow matching, take stock of what DDPM requires:
- **A noise schedule:** choose β_t values (linear? cosine? sigmoid?), compute cumulative products ᾱ_t
- **Schedule-derived quantities:** √ᾱ_t, √(1-ᾱ_t), posterior variance β̃_t — a whole zoo of precomputed tensors
- **A long derivation chain:** forward process → Bayes' rule → ELBO → KL between Gaussians → ε-reparameterization → simplified loss
- **1000 sampling steps** (or DDIM tricks to reduce it)
- **Schedule sensitivity:** linear vs cosine gives different results; zero terminal SNR matters; the schedule IS a hyperparameter

All of this works — we've proven that through this entire course. But it's a lot of machinery for what should be a simple goal: transport samples from N(0, I) to the data distribution.

### Section 13.2: Velocity Fields — The Bridge from DDPM to Flow Matching

**The problem with DDPM's prediction targets:**
- Recall from Lecture 11: ε-prediction is unstable at boundary timesteps. At t≈0, the noise is tiny and the loss signal is weak. At t≈T, the model has almost no signal to work with.
- v-prediction (Salimans & Ho, 2022) partially fixed this: define v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0, which balances the prediction difficulty across all timesteps
- This was a real improvement — Imagen and SD 2.x adopted v-prediction for exactly this reason

**Where v-prediction falls short:**
- v-prediction fixed the *prediction target*, but it didn't simplify the *framework*
- You still need: a noise schedule (β_t values), cumulative products (ᾱ_t), the full ELBO-derived loss structure, and the forward/reverse process machinery
- You still sample in 50-1000 discrete steps through the schedule
- v-prediction was the right *concept* (think in terms of velocities) trapped in the wrong *framework* (DDPM's schedule-driven noising process)
- The question became: what if we took the velocity idea seriously and built an entire framework around it?

**Reframing generation as particle transport:**
- Instead of "add noise step by step, learn to reverse it," think of generation as **moving particles from a noise distribution to a data distribution**
- Each particle follows a path from its starting position (noise) to its destination (data)
- A velocity field v(x, t) tells each particle which direction to move at each moment
- The model learns this velocity field: `dx/dt = v_θ(x_t, t)`
- To generate: start at noise (t=1), follow the velocity field to data (t=0)

**What this framework needs:**
- A path connecting each noise sample to each data sample (the interpolation)
- A target velocity along that path (what the model learns to predict)
- An ODE solver to follow the learned velocity field at inference
- That's it. No schedule. No ELBO. No reverse process derivation.
- The question is: what paths should we use?

### Section 13.3: Straight-Line Interpolation
- Define the interpolation path between data x_0 and noise x_1:
  - `x_t = (1-t) · x_0 + t · x_1` where t ∈ [0, 1]
  - At t=0: x_t = x_0 (data)
  - At t=1: x_t = x_1 (noise)
- The velocity along this straight path:
  - `dx/dt = x_1 - x_0` — constant velocity! Just "go from data toward noise"
- For generation, we reverse: start at t=1 (noise), integrate backward to t=0 (data)
  - Equivalently: learn v_θ(x_t, t) to predict (x_1 - x_0) and integrate the ODE backward

**Important convention note:** Flow matching literature uses different t conventions. In this lecture:
- t=0 is data, t=1 is noise
- Some papers use the opposite convention — always check!

### Section 13.4: The Flow Matching Training Objective
- Beautifully simple:
  1. Sample data: x_0 ~ p_data
  2. Sample noise: x_1 ~ N(0, I)
  3. Sample time: t ~ Uniform(0, 1)
  4. Interpolate: x_t = (1-t) · x_0 + t · x_1
  5. Target velocity: u = x_1 - x_0
  6. Loss: `L = ||v_θ(x_t, t) - u||²`

**Pseudocode:**
```
def flow_matching_train_step(model, x_0):
    x_1 = randn_like(x_0)                           # noise
    t = rand(batch_size)                              # t ∈ [0, 1]
    x_t = (1 - t) * x_0 + t * x_1                   # interpolate
    target_velocity = x_1 - x_0                       # straight-line velocity
    velocity_pred = model(x_t, t)                     # predict velocity
    loss = mse_loss(velocity_pred, target_velocity)
    return loss
```

Compare to DDPM training:
```
# DDPM requires:
alphas_cumprod = cumprod(1 - betas)                  # schedule computation
sqrt_alpha_bar = sqrt(alphas_cumprod[t])              # schedule indexing
sqrt_one_minus = sqrt(1 - alphas_cumprod[t])          # more schedule indexing
x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise  # weighted combination
# Flow matching: x_t = (1-t) * x_0 + t * x_1         # just linear interpolation
```

### Section 13.5: Sampling via ODE Integration
- Start from x_1 ~ N(0, I) (pure noise)
- Integrate backward: `x_{t-dt} = x_t - dt · v_θ(x_t, t)`
- Simplest solver: Euler method with fixed step size
- Better solvers: Heun's method (2nd-order), RK4, adaptive step size

**Pseudocode:**
```
def flow_matching_sample(model, shape, num_steps=50):
    x = randn(shape)                                  # x_1 ~ N(0, I)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = 1.0 - i * dt                              # t goes from 1 → 0
        velocity = model(x, t)
        x = x - dt * velocity                         # Euler step

    return x                                           # x_0 (generated data)
```

### Section 13.6: Why Flow Matching Works Better
- **No noise schedule to design:** no β_t, no α_t, no ᾱ_t — the interpolation path IS the "schedule"
- **Simpler math:** no ELBO derivation, no KL divergence, no reparameterization trick
- **Straight paths → fewer steps:** because the learned flow approximately follows straight lines, ODE solvers can take larger steps without losing accuracy
- **More stable training:** the target velocity (x_1 - x_0) has constant magnitude regardless of t, unlike ε-prediction where the signal varies with noise level
- **Connection to optimal transport:** straight-line interpolation with independent coupling is a reasonable approximation to optimal transport

### Section 13.7: Rectified Flows
- **The insight:** the initial learned flow isn't perfectly straight — it curves because different data-noise pairs share regions of space and their flows interact
- **Rectification:** iteratively straighten the flows:
  1. Train an initial flow matching model
  2. Generate (x_0, x_1) pairs by running the model forward and backward
  3. Retrain on these paired samples — the new flow is straighter because the pairs are coupled
  4. Repeat (typically 1-2 rounds is enough)
- **Result:** straighter trajectories → fewer ODE steps for same quality
- Rectified flows can generate good samples in as few as 1-4 steps
- Used in Stable Diffusion 3 and InstaFlow

### Section 13.8: Connection to Diffusion
- Flow matching is a **continuous generalization** of DDPM
- The connection:
  - DDPM with ε-prediction → specific ODE (the probability flow ODE from Lecture 7)
  - Flow matching → general ODE with straight-line paths
  - DDPM can be viewed as flow matching with a specific (non-straight) interpolation path:
    - `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` (DDPM path — curved, schedule-dependent, discrete t ∈ {0,...,T-1})
    - vs `x_t = (1-t) · x_0 + t · x_1` (flow matching path — straight, schedule-free, continuous t ∈ [0,1])
  - The time mapping: DDPM's discrete timestep t_ddpm corresponds roughly to flow matching's continuous t_fm = t_ddpm / T. But the interpolation coefficients are different — DDPM uses √ᾱ_t and √(1-ᾱ_t) (schedule-dependent, non-linear), while flow matching uses (1-t) and t (linear).
- **Score matching ↔ flow matching duality:**
  - Score: `∇_x log p_t(x)` — gradient of log density
  - Velocity: `v_θ(x_t, t)` — transport velocity
  - Related by: `v = f(x,t) - ½g(t)² ∇_x log p_t(x)` (from the SDE framework)
- **The unified picture:** diffusion, score matching, and flow matching are all ways to learn a mapping from noise to data. They differ in:
  - The interpolation path (curved vs straight)
  - The training target (ε vs score vs velocity)
  - The sampling procedure (SDE vs ODE)

### Section 13.9: Practical Impact
- **Stable Diffusion 3:** uses flow matching with rectified flows (not DDPM)
- **Flux:** flow matching + DiT architecture
- **Training recipe:** simpler than DDPM — fewer hyperparameters, more robust
- **Sampling:** 20-50 Euler steps for good quality; with rectification, as few as 4 steps
- **The trend:** new frontier models increasingly use flow matching over DDPM

## Key Equations
- Interpolation path: `x_t = (1-t) · x_0 + t · x_1`
- Target velocity: `u = x_1 - x_0`
- Training loss: `L = E_{t,x_0,x_1}[ ||v_θ(x_t, t) - (x_1 - x_0)||² ]`
- Euler sampling step: `x_{t-dt} = x_t - dt · v_θ(x_t, t)`
- Connection to DDPM: `x_t^{ddpm} = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` vs `x_t^{fm} = (1-t) · x_0 + t · x_1`

## Code Examples
- Flow matching training step — highlight simplicity vs DDPM (pseudocode first)
- Euler ODE sampling — the complete sampling loop
- Side-by-side comparison: DDPM training step vs flow matching training step (same model, different objective)
- 2D toy example: learn a velocity field on point clouds, visualize with quiver plot

## Diagrams & Visuals
- **Straight-line paths diagram:** show data points (blue) and noise points (red) connected by straight arrows — the interpolation paths
- **Velocity field quiver plot:** on a 2D toy problem, show the learned velocity field as arrows
- **DDPM vs flow matching paths:** side-by-side — DDPM's curved paths (controlled by schedule) vs flow matching's straight lines
- **Rectification diagram:** show curved initial flows → straighter flows after 1-2 rounds of rectification
- **Sampling steps comparison:** quality at 10, 20, 50 steps for DDPM vs flow matching — flow matching degrades more gracefully

## Source Material
- Module 09 §9.8 (Rectified Flows / Flow Matching)
- Module 07 §7.7 (Probability Flow ODE — foundational connection)
