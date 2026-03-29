# Lecture 3: The Reverse Process & the Simplified Loss
**Module:** 1 — The Diffusion Framework
**Estimated reading time:** 30 minutes
**Dependencies:** Lecture 2

## Learning Objectives
- Explain why the true reverse distribution q(x_{t-1} | x_t) is intractable
- Derive the tractable posterior q(x_{t-1} | x_t, x_0) and show it is Gaussian
- Walk through the ELBO decomposition that connects log-likelihood to per-timestep KL terms
- Derive the ε-prediction reparameterization that simplifies the loss to MSE on noise
- State DDPM Algorithm 1 (Training) and explain every line
- Preview alternative parameterizations (x_0, v-prediction) — detailed treatment in Lecture 11

## Narrative Arc
**The problem:** We've defined the forward process (Lecture 2) — we can noise any image to any level. But generation requires the reverse: starting from pure noise x_T, iteratively produce x_{T-1}, x_{T-2}, ..., x_0. The true reverse q(x_{t-1} | x_t) requires knowing the entire data distribution — intractable.

**The attempt:** There's a saving grace: if we additionally condition on x_0 (the clean image), the posterior q(x_{t-1} | x_t, x_0) IS tractable — it's a Gaussian with a known mean and variance. We can derive this with Bayes' rule. But we don't have x_0 at test time (that's what we're trying to generate).

**The solution:** Train a neural network to approximate the reverse step. The ELBO derivation shows that optimizing log-likelihood reduces to matching the means of two Gaussians at each timestep. A clever reparameterization shows that predicting the mean is equivalent to predicting the noise ε that was added. The entire training procedure collapses to: add noise, predict the noise, minimize MSE. Six lines of pseudocode.

## Section Outline

### Section 3.1: The Reverse Process — What We Need
- To generate images: start from x_T ~ N(0, I), apply p_θ(x_{t-1} | x_t) for t = T, ..., 1
- Each reverse step undoes one forward noising step
- The learned reverse: `p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))`
- The model must output a mean (and optionally variance) for the reverse Gaussian
- Key fact (Feller 1949): when β_t is small, the reverse of a Gaussian forward step is also approximately Gaussian

### Section 3.2: Why the True Reverse Is Intractable
- The true reverse: q(x_{t-1} | x_t) = ∫ q(x_{t-1} | x_t, x_0) q(x_0 | x_t) dx_0
- This requires marginalizing over ALL possible clean images x_0 that could have produced x_t
- q(x_0 | x_t) depends on the data distribution — we don't have this in closed form
- This is why we need a neural network: learn to approximate q(x_{t-1} | x_t) from data

### Section 3.3: The Tractable Posterior — q(x_{t-1} | x_t, x_0)
- If we know both x_t AND x_0, the posterior IS tractable via Bayes' rule
- Apply Bayes: q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)
- Both terms on the right are Gaussian (from the forward process), so the product is Gaussian
- The posterior mean: `μ̃_t = (√ᾱ_{t-1} · β_t · x_0 + √α_t · (1-ᾱ_{t-1}) · x_t) / (1-ᾱ_t)`
- The posterior variance: `β̃_t = (1-ᾱ_{t-1}) / (1-ᾱ_t) · β_t`
- This is a weighted average of x_0 and x_t — the model's "best guess" for x_{t-1}

### Section 3.4: The ELBO Derivation

<details><summary><strong>Expand full ELBO derivation</strong> (collapsible)</summary>

**Step 1: Start with log-likelihood**
- We want to maximize log p_θ(x_0) — the probability the model assigns to real data

**Step 2: Introduce the forward process as a variational bound**
- log p_θ(x_0) ≥ E_q[log p_θ(x_0:T) - log q(x_1:T | x_0)] — the ELBO

**Step 3: Decompose into per-timestep terms**
- L = L_0 + L_1 + ... + L_{T-1} + L_T
- L_t = D_KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t)) for t > 1
- Each term measures: how well does the learned reverse match the true posterior?

**Step 4: KL between two Gaussians has a closed form**
- Both q(x_{t-1} | x_t, x_0) and p_θ(x_{t-1} | x_t) are Gaussian
- KL divergence between Gaussians: depends on means and variances
- If we fix the variance (use β̃_t for both): KL reduces to mean-squared difference of means

**Step 5: The loss is about matching means**
- Minimize ||μ̃_t - μ_θ(x_t, t)||² at each timestep

</details>

**Key takeaway:** the ELBO tells us to make the learned reverse step match the true posterior at every timestep. When both are Gaussian with fixed variance, this reduces to matching means.

### Section 3.5: The ε-Prediction Reparameterization
- The posterior mean μ̃_t depends on x_0 — but we can rewrite x_0 in terms of x_t and ε:
  - From the forward process: x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t
- Substituting into μ̃_t: the mean becomes a function of x_t, t, and the noise ε
- Instead of predicting μ_θ directly, predict ε_θ(x_t, t) — the noise that was added
- Then compute μ_θ from ε_θ using the closed-form relationship
- The loss simplifies to: **L = E_{t,x_0,ε}[ ||ε - ε_θ(x_t, t)||² ]**
- In words: sample noise, add it to an image, ask the model to predict that noise, minimize MSE

### Section 3.6: The Simplified Loss (L_simple)
- The full ELBO loss has timestep-dependent weights (1/SNR scaling)
- DDPM finding: dropping these weights and using uniform weighting works better empirically
- L_simple = E_{t,x_0,ε}[ ||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||² ]
- Why it works: uniform weighting emphasizes perceptually important mid-noise timesteps
- This is the loss used in practice by nearly all diffusion models

### Section 3.7: DDPM Algorithm 1 — The Training Procedure
The entire training algorithm:
```
repeat until converged:
    x_0 ~ dataset                           # sample a training image
    t ~ Uniform({1, ..., T})                # random timestep
    ε ~ N(0, I)                             # sample noise
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε     # noise the image
    loss = ||ε - ε_θ(x_t, t)||²            # predict noise, MSE loss
    gradient step on loss                    # update the network
```
Six lines. That's the entire training procedure for the model that powers Stable Diffusion 1.5 and DALL-E 2. The same core idea, with variations (v-prediction, flow matching), underlies everything since — including Stable Diffusion 3 and Sora. The simplicity is remarkable — and it all follows from the math.

### Section 3.8: Alternative Parameterizations (Preview)
- **ε-prediction:** predict the noise. Standard (DDPM). This lecture's focus.
- **x_0-prediction:** predict the clean image directly. Sometimes better for low-noise steps.
- **v-prediction:** predict the velocity v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0. Numerically stable across all timesteps.
- All three are mathematically equivalent — you can convert between them given (x_t, t).
- Detailed treatment and comparison in Lecture 11.

## Key Equations
- Learned reverse: `p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)`
- Tractable posterior mean: `μ̃_t = (√ᾱ_{t-1} · β_t · x_0 + √α_t · (1-ᾱ_{t-1}) · x_t) / (1-ᾱ_t)`
- Posterior variance: `β̃_t = (1-ᾱ_{t-1}) / (1-ᾱ_t) · β_t`
- x_0 from ε: `x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t`
- μ_θ from ε_θ: `μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t)) · ε_θ)`
- Simplified loss: `L_simple = E_{t,x_0,ε}[ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||² ]`

## Code Examples
- **Posterior mean computation** (pseudocode): given x_t, x_0, t → compute μ̃_t
- **ε → μ_θ conversion** (pseudocode): given ε_θ, x_t, t → compute predicted mean
- **The training step** (pseudocode first, then PyTorch reference):
  ```
  def train_step(model, x_0, schedule):
      t = random_integers(0, T, batch_size)
      noise = sample_gaussian(x_0.shape)
      x_t = sqrt(schedule.alpha_bar[t]) * x_0 + sqrt(1 - schedule.alpha_bar[t]) * noise
      noise_pred = model(x_t, t)
      loss = mean_squared_error(noise, noise_pred)
      return loss
  ```

## Diagrams & Visuals
- **Reverse process diagram:** x_T → p_θ → x_{T-1} → p_θ → ... → x_0, with the neural network at each step
- **Posterior visualization:** at a specific timestep, show q(x_{t-1} | x_t, x_0) as a distribution — it's a narrow Gaussian centered between x_t and x_0
- **ELBO decomposition diagram:** log p(x_0) → sum of KL terms → sum of MSE terms → simplified loss
- **Algorithm 1 as a flow diagram:** x_0 → sample t → sample ε → noise → predict → loss → update

## Source Material
- Module 05 §5.5 (Reverse Process)
- Module 05 §5.6 (ELBO Derivation)
- Module 05 §5.7 (Simplified Loss)
