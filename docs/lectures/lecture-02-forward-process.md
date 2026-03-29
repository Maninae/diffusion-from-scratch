# Lecture 2: The Forward Process & Noise Schedules
**Module:** 1 — The Diffusion Framework
**Estimated reading time:** 25 minutes
**Dependencies:** Lecture 1

## Learning Objectives
- Define the forward noising process mathematically: q(x_t | x_{t-1})
- Derive the closed-form shortcut for jumping directly to any timestep t
- Explain the reparameterization trick and why it enables gradient-based training
- Implement and compare linear, cosine, and sigmoid noise schedules
- Interpret the signal-to-noise ratio (SNR) as a unified lens on noise schedules

## Narrative Arc
**The problem:** We claimed diffusion models "add noise step by step" — but iterating through 1000 steps to noise a single training image would be absurdly slow. Training requires millions of (image, timestep, noise) triples. We need a way to jump directly to any noise level in one shot.

**The attempt:** The forward process is a Markov chain of Gaussian steps: q(x_t | x_{t-1}). Each step multiplies the signal by √(1-β_t) and adds √β_t noise. You could iterate this chain — but for training, you need x_t given x_0 for arbitrary t.

**The solution:** Because Gaussians compose beautifully, we can derive a closed-form: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε. One multiplication, one addition — we can noise any image to any timestep instantly. The schedule (how β_t varies with t) controls how quickly information is destroyed, and this choice matters more than you'd think.

## Section Outline

### Section 2.1: The Forward Markov Chain
- Each step adds a small amount of Gaussian noise:
  - `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)`
- β_t is the noise variance at step t — small (e.g., 0.0001 to 0.02)
- The √(1-β_t) scaling slightly shrinks the signal each step — prevents the variance from blowing up
- After T=1000 steps: x_T ≈ N(0, I) — the image has been completely destroyed
- This is a fixed process — no learnable parameters. The network only learns the reverse.

**Pseudocode:**
```
def single_forward_step(x_prev, beta_t):
    noise = sample_gaussian(shape=x_prev.shape)
    x_t = sqrt(1 - beta_t) * x_prev + sqrt(beta_t) * noise
    return x_t
```

### Section 2.2: The Cumulative Products — α_t and ᾱ_t
- Define: α_t = 1 - β_t (the signal retention at step t)
- Define: ᾱ_t = ∏_{s=1}^{t} α_s (cumulative signal retention from step 0 to step t)
- ᾱ_t decays from ~1 (t=0, mostly signal) to ~0 (t=T, mostly noise)
- These are precomputed once and stored — they're the "schedule" that the model uses

**Pseudocode:**
```
betas = linear_schedule(T=1000)          # (T,)
alphas = 1 - betas                       # (T,)
alphas_cumprod = cumulative_product(alphas)  # (T,) — this is ᾱ_t
```

### Section 2.3: The Closed-Form Shortcut
- The key derivation: because each step is Gaussian, and Gaussians compose:
  - `q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)`
- In sampling form: **x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε**, where ε ~ N(0, I)
- √ᾱ_t scales how much of the original image survives
- √(1-ᾱ_t) scales how much noise is added
- At t=0: √ᾱ_0 ≈ 1, √(1-ᾱ_0) ≈ 0 → x_0 is nearly untouched
- At t=T: √ᾱ_T ≈ 0, √(1-ᾱ_T) ≈ 1 → x_T is pure noise

**Pseudocode:**
```
def q_sample(x_0, t, alphas_cumprod):
    """Noise x_0 to timestep t in one shot."""
    noise = sample_gaussian(shape=x_0.shape)      # ε ~ N(0, I)
    sqrt_alpha_bar = sqrt(alphas_cumprod[t])       # signal coefficient
    sqrt_one_minus = sqrt(1 - alphas_cumprod[t])   # noise coefficient
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
    return x_t, noise  # return noise too — it's the training target!
```

### Section 2.4: The Reparameterization Trick
- We need x_t to be differentiable with respect to model parameters (for backprop)
- Sampling from N(μ, σ²) directly is not differentiable — the randomness blocks gradients
- The trick: sample ε ~ N(0, I) first (independent of parameters), then compute x_t = μ + σ · ε
- The randomness (ε) is external; the computation (μ + σ · ε) is a deterministic, differentiable function
- Same trick used in VAEs (we'll see it again in Lecture 10 when building the VAE for latent diffusion) — it's the standard way to make stochastic nodes differentiable
- In diffusion: ε is not just a computational trick — it becomes the training target (predict the noise)

### Section 2.5: Signal-to-Noise Ratio
- SNR(t) = ᾱ_t / (1 - ᾱ_t) — ratio of signal power to noise power
- At t=0: SNR is large (mostly signal)
- At t=T: SNR ≈ 0 (pure noise)
- SNR provides a unified way to compare different schedules
- log-SNR is often more useful (linear in the meaningful range)
- The model's difficulty varies with SNR: mid-range SNR is hardest (partial signal, partial noise)

### Section 2.6: Noise Schedules
**Linear schedule (DDPM, Ho et al. 2020):**
- β_t increases linearly from β_1 = 0.0001 to β_T = 0.02
- Simple and effective, but destroys information too quickly in early steps
- ᾱ_t drops rapidly at first, then slowly — uneven difficulty distribution

**Cosine schedule (Nichol & Dhariwal 2021):**
- ᾱ_t follows a cosine curve: `ᾱ_t = cos((t/T + s) / (1+s) · π/2)²` with s=0.008
- Smoother SNR decay — information is preserved longer
- More gradual destruction in early steps → the model spends more time at medium noise levels
- Generally preferred over linear

**Sigmoid schedule:**
- β_t follows a sigmoid curve — fast ramp in the middle, gentle at extremes
- Compromise between linear and cosine

**Pseudocode:**
```
def linear_schedule(T, beta_start=0.0001, beta_end=0.02):
    return linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    steps = linspace(0, T, T+1)
    f_t = cos((steps/T + s) / (1+s) * pi/2) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return clip(betas, max=0.999)
```

### Section 2.7: Visualizing the Forward Process
- Show the same image at t = 0, 100, 250, 500, 750, 1000 — watch it dissolve into static
- Plot ᾱ_t curves for linear vs cosine vs sigmoid side by side
- Plot log-SNR curves — the cosine schedule has a much more gradual decline
- Key visual: "what the model sees" at different noise levels — at mid-t, you can barely see structure; at high-t, it's pure noise

## Key Equations
- Forward step: `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)`
- Cumulative product: `ᾱ_t = ∏_{s=1}^{t} (1 - β_s)`
- Closed-form: `q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)`
- Sampling form: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`
- SNR: `SNR(t) = ᾱ_t / (1 - ᾱ_t)`
- Cosine schedule: `ᾱ_t = cos²((t/T + s) / (1+s) · π/2)`

## Code Examples
- `q_sample(x_0, t, schedule)` — noise an image to timestep t (pseudocode first, PyTorch reference at end)
- `linear_schedule(T)` and `cosine_schedule(T)` — compute β, α, ᾱ for the full schedule
- Verification: sequential noising for 500 steps matches closed-form `q_sample(x_0, 500, ...)` (same distribution)

## Diagrams & Visuals
- **Forward process strip:** single MNIST digit at t = {0, 100, 250, 500, 750, 1000}
- **Schedule comparison plots:** ᾱ_t curves (linear vs cosine vs sigmoid) and log-SNR curves
- **Signal vs noise decomposition:** at a given t, show the signal component (√ᾱ_t · x_0) and noise component (√(1-ᾱ_t) · ε) separately, then their sum

## Source Material
- Module 05 §5.2 (Forward Process)
- Module 05 §5.3 (Noise Schedule: β_t, α_t, ᾱ_t)
- Module 05 §5.4 (Reparameterization Trick)
- Module 05 §5.8 (Signal-to-Noise Ratio)
- Module 05 §5.9 (Variance Schedules)
