# Module 5: The Math of Diffusion

## Purpose
Understand the probabilistic framework behind diffusion models — forward process, reverse process, the loss derivation. This is the theoretical heart. You should be able to explain why predicting noise works and derive the simplified loss.

## 📄 Key Papers
- **Denoising Diffusion Probabilistic Models (DDPM)** — Ho et al. 2020. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
  - *Read for:* THE foundational paper. Section 2 for forward process, Section 3 for reverse process and training. Algorithm 1 (training) and Algorithm 2 (sampling) are the key reference.
- **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** — Sohl-Dickstein et al. 2015. [arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585)
  - *Read for:* The original diffusion idea. More theoretical/physics-oriented. Read if you want the deep foundations.
- **Improved Denoising Diffusion Probabilistic Models** — Nichol & Dhariwal 2021. [arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)
  - *Read for:* Cosine noise schedule (Section 3.2), learned variance, importance of the noise schedule. Practical improvements.
- **Score-Based Generative Modeling through Stochastic Differential Equations** — Song et al. 2021. [arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)
  - *Read for:* The unified framework that connects DDPM and score matching through SDEs. Section 3 for the forward/reverse SDE pair. Elegant but more mathematical.
- **Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation** — Kingma et al. 2023. [arxiv.org/abs/2303.00848](https://arxiv.org/abs/2303.00848)
  - *Read for:* Modern understanding of the diffusion objective. SNR perspective.

## Sections

### 5.1 — Generative Models Landscape

**Concepts to teach:**
- The goal: learn p(x) from data, then sample new x
- **GANs:** generator vs discriminator game. No explicit p(x). Mode collapse issues.
- **VAEs:** encoder-decoder with latent space. Explicit ELBO. Often blurry.
- **Normalizing Flows:** invertible transforms. Exact log-likelihood. Architecture constraints.
- **Autoregressive:** model p(x) as product of conditionals. Exact likelihood. Slow sampling.
- **Diffusion:** gradually denoise from noise to data. Implicit p(x) through the reverse process. Currently dominant for image generation.
- Why diffusion won: training stability (no adversarial game), sample quality (beat GANs), flexible architecture

**Content:** Conceptual comparison table, key tradeoffs. Brief — this sets context, not deep dives into each.

---

### 5.2 — Forward Process: Adding Noise Step by Step

**Concepts to teach:**
- Start with clean data x_0
- At each step, add a small amount of Gaussian noise: `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)`
- β_t is small (e.g., 0.0001 to 0.02), so each step only adds a tiny bit of noise
- After many steps (T=1000), x_T ≈ pure Gaussian noise N(0, I)
- This is a fixed process — no learned parameters
- Markov chain: each step only depends on the previous step

**Worked example:**
- Implement single-step noising: given x_{t-1}, produce x_t
- Run the full chain for T=1000 steps on an image, visualize degradation at t=0, 250, 500, 750, 1000

---

### 5.3 — Noise Schedule: β_t, α_t, ᾱ_t

**Concepts to teach:**
- Define: `α_t = 1 - β_t`
- Define: `ᾱ_t = ∏_{s=1}^{t} α_s` (cumulative product of alphas)
- The **key property**: we can skip to any timestep directly:
  - `q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)`
  - `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε` where `ε ~ N(0, I)`
- This means: no need to iterate through all steps during training!
- √ᾱ_t controls how much signal remains, √(1-ᾱ_t) controls how much noise
- Derivation: use the Markov property and properties of Gaussians (sum of Gaussians is Gaussian)

**Worked example:**
- Compute β_t, α_t, ᾱ_t for a linear schedule
- Plot all three curves
- Demonstrate the closed-form: noise an image directly to t=500 in one step, compare to doing 500 sequential steps (they match!)

**Exercise:**
- Implement the full schedule computation: β → α → ᾱ → √ᾱ → √(1-ᾱ)
- Precompute all these as tensors for T=1000 (this is what the model stores as buffers)
- Verify: sequential noising matches the closed-form

---

### 5.4 — The Reparameterization Trick (Revisited)

**Concepts to teach:**
- We need `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε` to be differentiable with respect to parameters
- The trick: sample `ε ~ N(0, I)` first, then compute x_t deterministically
- ε is the noise we want the model to predict — it's the training target
- Same trick used in VAEs — separates randomness from the computation graph

**Worked example:**
- Show two ways to sample x_t: (1) sample from N(√ᾱ_t x_0, (1-ᾱ_t) I) directly, (2) reparameterize as √ᾱ_t x_0 + √(1-ᾱ_t) ε
- Demonstrate they produce identical distributions but (2) allows gradient flow

---

### 5.5 — Reverse Process: What the Model Learns

**Concepts to teach:**
- The reverse process undoes the forward process: start from x_T ~ N(0, I), denoise step by step to get x_0
- True reverse: `q(x_{t-1} | x_t)` is intractable (depends on the entire data distribution)
- Learned reverse: `p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))`
- The model learns to predict the mean μ_θ (and optionally the variance Σ_θ)
- Key insight: when β_t is small, the reverse step is also approximately Gaussian (Feller 1949)
- The posterior `q(x_{t-1} | x_t, x_0)` IS tractable: this is used to derive the training objective

**Worked example:**
- Write out `q(x_{t-1} | x_t, x_0)` — it's Gaussian with known mean and variance
- Show the formula: `μ̃_t = (√ᾱ_{t-1} β_t x_0 + √α_t (1-ᾱ_{t-1}) x_t) / (1-ᾱ_t)`

---

### 5.6 — ELBO Derivation: Why Predicting Noise Works

**Concepts to teach:**
- The ELBO (Evidence Lower Bound): decompose the log-likelihood into manageable KL divergence terms
- Each KL term compares learned reverse `p_θ(x_{t-1} | x_t)` to true posterior `q(x_{t-1} | x_t, x_0)`
- Since both are Gaussian, KL divergence has a closed-form
- The KL simplifies to: matching means → predict the mean of the reverse step
- **The ε-prediction reparameterization:**
  - Instead of predicting μ_θ directly, reparameterize: model predicts ε_θ (the noise)
  - Then μ_θ is computed from ε_θ using the closed-form relationship
  - Loss simplifies to: `L = E[||ε - ε_θ(x_t, t)||²]`
- **Why this works so well:** it's simpler, more stable, and empirically produces better samples
- **Alternative parameterizations:**
  - x_0 prediction: model predicts x̂_0 instead of ε. Mathematically equivalent. Sometimes better for certain schedules.
  - v-prediction: `v = √ᾱ_t ε - √(1-ᾱ_t) x_0`. Numerically better near t=0 and t=T.

**Worked example:**
- Walk through the ELBO decomposition step by step (with LaTeX)
- Show the simplification from ELBO → KL between Gaussians → MSE on noise
- Implement the conversion: ε_θ → μ_θ → x_{t-1}

**Exercise:**
- Given a predicted ε_θ and timestep t, compute the predicted x_0 and μ_θ
- Implement all three prediction modes: ε, x_0, v — show they're equivalent

---

### 5.7 — The Simplified Loss

**Concepts to teach:**
- The full ELBO loss has timestep-dependent weights
- DDPM simplified loss: drop the weights, just use `L_simple = E_t,x_0,ε[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]`
- In words: sample a training image, sample a random timestep, add noise, predict the noise, MSE loss
- This is Algorithm 1 from the DDPM paper — the entire training procedure
- Why dropping weights works: empirically produces better sample quality (focuses on perceptually important timesteps)

**Worked example:**
- Implement the simplified loss in 10 lines of PyTorch
- Show it's literally: `loss = F.mse_loss(model(noisy_x, t), noise)`

**Exercise:**
- Implement `diffusion_loss(model, x_0, noise_schedule)` — the complete training loss function

---

### 5.8 — Signal-to-Noise Ratio (SNR) Perspective

**Concepts to teach:**
- `SNR(t) = ᾱ_t / (1 - ᾱ_t)` — the ratio of signal power to noise power at timestep t
- At t=0: SNR is high (mostly signal, little noise)
- At t=T: SNR ≈ 0 (almost pure noise)
- The model's job is harder at low SNR (hard to find the signal in the noise)
- Different noise schedules create different SNR curves
- The ELBO loss weights terms by 1/SNR — simplified loss ignores this

**Worked example:**
- Plot SNR curve for linear and cosine schedules
- Visualize what the model "sees" at different SNR levels

**Exercise:**
- Implement SNR computation, plot for different schedules
- Implement SNR-weighted loss

---

### 5.9 — Variance Schedules: Linear, Cosine, Learned

**Concepts to teach:**
- **Linear schedule:** β_t linearly increases from β_1=0.0001 to β_T=0.02. Original DDPM.
- **Cosine schedule:** ᾱ_t follows a cosine curve. Proposed in Improved DDPM.
  - Avoids the problem where linear schedule destroys too much info in early steps
  - Formula: `ᾱ_t = cos((t/T + s) / (1+s) · π/2)²` where s=0.008
- **Learned schedule:** let the model learn β_t (or log variance). More flexible but harder to train.
- **Practical guidance:** cosine schedule is generally preferred over linear

**Worked example:**
- Implement both linear and cosine schedules
- Plot ᾱ_t and SNR curves side by side
- Noise an image with both schedules, compare visual quality at same timestep

**Exercise:**
- Implement linear, cosine, and sigmoid schedules
- Compare training dynamics on a simple diffusion model with each schedule

---

## Module 5 Exercises — Math Reinforcement

1. **Derivation exercise:** Starting from `q(x_t | x_0)`, derive that `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε` (pen-and-paper, verify numerically)
2. **Verify the posterior:** Compute `q(x_{t-1} | x_t, x_0)` numerically by sampling, compare to the closed-form Gaussian
3. **SNR analysis:** For a given schedule, at which timestep does SNR = 1? What does this mean?
4. **Implement forward noising** on a batch of CIFAR-10 images, visualize at 10 evenly spaced timesteps
5. **Schedule comparison:** Train the same model with linear vs cosine schedule for 10 epochs, compare loss curves
