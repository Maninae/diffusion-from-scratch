# Assessment Track
**Section:** Assessment (separate from numbered modules)
**Estimated time:** 3-4 hours total
**Dependencies:** All core lectures (1-9); Lectures 10-13 for advanced questions

## Purpose
Self-evaluation exercises that test deep understanding, not surface-level recall. Four formats: timed coding challenges, verbal explanation prompts, a debugging exercise, and a system design discussion. Each format targets a different skill — implementation speed, conceptual clarity, code reading, and architectural reasoning.

---

## Part 1: Coding Challenges

### Challenge 1: Minimal Diffusion on 2D Point Clouds (45 minutes)

**The prompt:**
> "Implement a diffusion model that learns to generate samples from a 2D spiral distribution. Implement: the noise schedule, a simple MLP denoiser, the training loop, and the sampling loop."

**What this tests:**
- Core diffusion understanding stripped of image complexity
- Can you implement the algorithm from memory without architecture distractions?

**What a strong solution includes:**
- Correct noise schedule computation (β, α, ᾱ — precomputed as tensors)
- Correct forward process: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`
- MLP denoiser: input is (x_t concat t_emb), output is predicted ε, with shape (B, 2)
- Sinusoidal timestep embedding
- Training loop: sample x_0 → sample t → sample ε → noise → predict → MSE loss → gradient step
- DDPM sampling loop: x_T → iterative denoising → x_0
- Visualization: generated samples overlaid on training data

**Starter code provided:**
```python
import torch, torch.nn as nn, matplotlib.pyplot as plt

def make_spiral(n_points=1000):
    t = torch.linspace(0, 4*torch.pi, n_points)
    x = t * torch.cos(t) / (4*torch.pi)
    y = t * torch.sin(t) / (4*torch.pi)
    return torch.stack([x, y], dim=1) + 0.05 * torch.randn(n_points, 2)

data = make_spiral(2000)
# Your implementation below...
```

**Expected deliverables:**
1. Noise schedule (linear or cosine)
2. MLP model with timestep conditioning
3. Training loop (should converge in ~5000 steps)
4. Sampling function
5. Plot: generated samples overlaid on training data

**Evaluation criteria:**
- Correct implementation of core diffusion mechanics
- Clean code structure with meaningful variable names
- Working end-to-end: training converges, samples match distribution

---

### Challenge 2: DDIM Sampling with CFG (45 minutes)

**The prompt:**
> "Given a pre-trained class-conditional diffusion model, implement DDIM sampling with classifier-free guidance."

**What this tests:**
- Sampling algorithm implementation (the inference side, not training)
- CFG formula and batched forward passes
- Timestep sub-selection for accelerated sampling

**What a strong solution includes:**
- Correct DDIM update formula with x_0 prediction step
- Timestep sub-selection (uniform spacing over T steps)
- CFG: two forward passes (batched), correct combination formula
- Guidance scale parameter with proper s=1 baseline
- No noise at final step, proper clamping and denormalization

**Starter code provided:**
```python
model = load_pretrained_model()    # model(x_t, t, class_label) -> noise_pred
schedule = load_schedule()          # Contains alphas_cumprod, etc.
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES

def ddim_sample_cfg(model, schedule, shape, class_label,
                     guidance_scale=4.0, num_steps=50):
    # Your implementation here...
    pass
```

**Expected deliverables:**
1. Timestep sub-selection (uniform spacing)
2. DDIM sampling loop with x_0 prediction
3. CFG: batched conditional + unconditional forward passes
4. Working generation of class-conditional samples
5. Grid visualization at different guidance scales

**Evaluation criteria:**
- Correct DDIM formula (not just DDPM with fewer steps)
- CFG combination is correct: `ε_uncond + s · (ε_cond - ε_uncond)`
- Handles edge cases: no noise at final step, proper clamping

---

### Challenge 3: Training Step from Scratch (45 minutes)

**The prompt:**
> "Implement the complete training step for a DDPM-style diffusion model. Given a batch of images and a model, implement: noise schedule setup, the noising process, loss computation, and a training loop."

**What this tests:**
- Schedule computation (precomputing derived quantities)
- Forward process implementation (correct indexing)
- Training mechanics (optimizer, gradient clipping)

**What a strong solution includes:**
- Schedule computation: β → α → ᾱ → √ᾱ → √(1-ᾱ), all precomputed as tensors
- Correct indexing: `schedule_values[t]` where t has shape (B,), properly broadcasted
- Training step: sample t, sample ε, noise image, predict noise, MSE loss
- Gradient clipping
- Loss decreases over first 100 steps (verification)

**Starter code provided:**
```python
model = UNet(...)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
T = 1000

# Implement: setup_schedule(), train_step(), run training loop
```

**Expected deliverables:**
1. Schedule computation function
2. `train_step(model, x_0, optimizer, schedule)` function
3. Training loop with loss logging
4. Verify loss decreases over first 100 steps

---

## Part 2: Conceptual Questions

*Practice explaining these clearly and concisely. For each: try to answer before reading the model answer.*

### Core Understanding
1. **"Walk me through the forward diffusion process. What happens to an image at each step?"**
   - Key points: Gaussian noise added, signal scaled down, after T steps → pure noise, closed-form shortcut
   - Common mistake: forgetting the signal scaling (√(1-β_t) factor)

2. **"Why do we predict noise instead of predicting the clean image directly?"**
   - Key points: ELBO derivation leads to mean matching, ε-reparameterization simplifies the loss, more uniform difficulty across timesteps
   - Common mistake: saying "it's just a design choice" — there's mathematical justification

3. **"What is the reparameterization trick and why do we need it?"**
   - Key points: separate randomness from computation, enables gradient flow through stochastic nodes
   - Common mistake: confusing it with the noise schedule computation

4. **"Explain the simplified DDPM loss. Why does MSE on noise work?"**
   - Key points: ELBO → KL between Gaussians → mean matching → ε-prediction, dropping weights improves quality
   - Common mistake: not connecting the simplified loss back to the ELBO

5. **"What's the difference between DDPM and DDIM sampling?"**
   - Key points: DDPM is stochastic (adds noise each step), DDIM is deterministic (η=0), same trained model, DDIM enables step-skipping
   - Common mistake: saying DDIM requires retraining

### Architecture
6. **"Why does the diffusion U-Net use GroupNorm instead of BatchNorm?"**
   - Key points: BatchNorm statistics depend on batch size, inconsistent between training (large batch) and sampling (single image). GroupNorm computes per-sample → consistent behavior.

7. **"Where do you put attention layers in the U-Net and why?"**
   - Key points: O(n²) cost, only at lower resolutions (16×16, 8×8), captures global structure that convolutions miss

8. **"How does timestep conditioning work?"**
   - Key points: sinusoidal embedding → MLP → inject into ResBlocks (addition or AdaGN)

9. **"What are skip connections and why are they important?"**
   - Key points: carry fine spatial detail from encoder to decoder, gradient highway, without them → blurry outputs

### Guidance & Conditioning
10. **"Explain classifier-free guidance. How does training differ?"**
    - Key points: random label dropout during training, same model does conditional + unconditional, extrapolate at inference
    - Common mistake: confusing with classifier guidance (separate classifier)

11. **"What does the guidance scale control?"**
    - Key points: diversity-fidelity tradeoff, s>1 amplifies conditioning, acts like inverse temperature

12. **"How do negative prompts work mechanically?"**
    - Key points: replace ε_uncond with ε_neg in CFG formula, steers away from negative concept

### Advanced
13. **"What's the difference between pixel-space and latent diffusion?"**
    - Key points: VAE compresses to latent space, diffusion runs on smaller representation, decode at end

14. **"What are the tradeoffs between ε-prediction, x_0-prediction, and v-prediction?"**
    - Key points: ε stable at high noise, x_0 at low noise, v balanced across all t

15. **"What is flow matching and how does it differ from diffusion?"**
    - Key points: learn velocity field along straight paths, simpler objective, no schedule, fewer steps

16. **"How would you scale this model to generate 512×512 images?"**
    - Key points: latent diffusion (VAE), DiT architecture, gradient checkpointing, mixed precision

---

## Part 3: Debug Challenge

**The task:** find all 10 bugs in a complete diffusion pipeline. The bugs are NOT labeled — you must find them by reading the code carefully.

**Bugs embedded in the code (students must discover these):**

1. **Wrong scaling in forward process:** uses `alpha_t` instead of `alpha_bar_t` (cumulative product) — the noising is wrong at every timestep

2. **Missing sqrt:** `x_t = alpha_bar * x_0 + (1 - alpha_bar) * noise` instead of `sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise` — means and variances are wrong

3. **Off-by-one in sampling loop:** loop goes from T to 0 instead of T to 1 — adds noise at t=0 (final step should be noiseless)

4. **Sampling with training weights instead of EMA:** uses `model` directly for sampling instead of the EMA weights — produces noisier, lower-quality samples (the whole point of EMA from Lecture 6 is to use the smoothed weights for generation)

5. **Gradient accumulation:** missing `optimizer.zero_grad()` — gradients accumulate across steps, training explodes

6. **Wrong normalization:** data loaded in [0, 1] but model expects [-1, 1] — color shift in all outputs

7. **Timestep embedding bug:** passing raw integer t directly as float instead of sinusoidal embedding — the model can't distinguish timesteps properly

8. **CFG formula wrong:** `eps_uncond + scale * eps_cond` instead of `eps_uncond + scale * (eps_cond - eps_uncond)` — guidance doesn't extrapolate correctly

9. **Device mismatch:** schedule tensors remain on CPU while model tensors are on GPU/MPS — silent errors or crashes

10. **Variance schedule not clipped:** extreme β values cause numerical instability — NaN loss after a few hundred steps

**Format:**
- Present the buggy code (looks plausible, compiles, runs partially)
- Students identify and fix all bugs
- Solution includes: what each bug causes, why it's wrong, and the fix

---

## Part 4: System Design Discussion

**The prompt:**
> "You've built a diffusion model that works on 28×28 MNIST. Your manager asks you to scale it to generate 512×512 photorealistic images. Walk through your approach."

**Expected discussion structure:**

### The Resolution Problem
- 28×28 → 512×512 = 335× more pixels
- Can't just make the U-Net bigger — attention is O(n²)
- Need a fundamentally different approach

### Architecture Decisions
- **Latent diffusion:** train a VAE (512→64, 8× spatial compression), do diffusion in latent space
- **DiT vs U-Net:** at this scale, consider DiT for better scaling properties
- **Attention placement:** only at 32×32 and below (even in latent space)
- **Gradient checkpointing:** trade compute for memory
- **Mixed precision:** fp16 or bf16 training for 2× memory savings

### Training Decisions
- **Dataset:** MNIST is too small. Need 100K+ diverse images minimum
- **Training time:** 100K-1M steps (days to weeks on multiple GPUs)
- **Batch size:** 256+ with distributed training
- **EMA:** critical at this scale (decay 0.9999)
- **Schedule:** cosine schedule with zero terminal SNR

### Inference Decisions
- **Sampler:** DDIM with 50 steps (not 1000 DDPM steps)
- **CFG:** guidance scale 7.5 for text-to-image
- **Dynamic thresholding:** for high guidance scales
- **Batched generation:** generate multiple images in parallel

### Follow-up Questions to Consider
- "What's the memory bottleneck and how would you address it?"
- "How would you add text conditioning?"
- "What metrics would you use to evaluate quality?"
- "How would you reduce inference latency for a production API?"

---

## General Tips

These challenges help you identify gaps in your understanding. Approach them as self-evaluation, not performance.

1. **Think out loud.** Narrate your reasoning as you code — it helps you catch your own mistakes and clarifies your mental model.
2. **Start simple.** Get the basic version working first, then add complexity. Don't over-engineer from the start.
3. **Write clean code from the start.** Meaningful variable names (`sqrt_alpha_bar_t`, not `a`), logical structure, brief comments on key lines. This reflects real understanding.
4. **Know your shapes.** At every step, know the tensor shapes. Print them if unsure. Shape mismatches are the #1 bug.
5. **If you get stuck, that's the point.** Identify exactly where your understanding breaks down — that's what to review. It's normal to need to derive formulas from first principles.
6. **Have a plan.** Spend 2-3 minutes outlining before coding: "First schedule, then model, then training, then sampling."
7. **Know the math cold.** `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` — you should be able to write this without thinking.

## Source Material
- Module 10 §10.1 (2D Point Cloud Diffusion)
- Module 10 §10.2 (DDIM + CFG Sampling)
- Module 10 §10.3 (Training Step)
- Module 10 §10.4 (Verbal Explanation Prompts)
- Module 10 §10.5 (Debug Challenge)
- Module 10 §10.6 (System Design Discussion)
