# Content Architecture: Diffusion From Scratch — Course Website

Prepared by Content Architect agent, 2026-03-28.

---

## Executive Summary

The current curriculum is 11 Jupyter notebooks (modules 00–10) designed as linear, self-contained units. Reorganizing for a website means rethinking structure around **learning flow**, not file boundaries.

**Key decisions:**

- **Modules 00–02 are cut.** CS231N-level prerequisite assumed — students know CNNs, backprop, optimization, normalization, residual connections, and basic attention.
- **Module 03 is partially absorbed.** Generic attention is prerequisite knowledge. Diffusion-specific attention (spatial self-attention in U-Net, cross-attention for conditioning, sinusoidal embeddings as timestep conditioning) is folded into relevant lectures.
- **Math comes before architecture.** The original ordering (U-Net in module 04, math in module 05) is reversed. Students should understand *what* diffusion is before building the network that implements it. You need to know about noise prediction and timestep conditioning to understand why U-Net is designed the way it is.
- **Module 09 is split into 4 lectures** across 2 modules. It covers too much (VAEs, latent diffusion, Stable Diffusion, prediction targets, schedule improvements, DiT, flow matching) for a single page.
- **Module 10 becomes a standalone assessment track**, not a numbered module. It serves a different purpose (self-evaluation) than the lecture content.

**Result:** 6 modules, 13 lectures, plus an assessment track. Each lecture is a single HTML page with equations, diagrams, code snippets, and interactive elements (Phase 2).

---

## Proposed Site Structure

Following the CS234 pattern: a sidebar with numbered modules, each containing 2–3 lectures. Clicking a module expands to show its lectures. Each lecture is a standalone page.

```
Module 1: The Diffusion Framework
  Lecture 1:  Why Diffusion? The Generative Models Landscape
  Lecture 2:  The Forward Process & Noise Schedules
  Lecture 3:  The Reverse Process & the Simplified Loss

Module 2: The Denoising Network
  Lecture 4:  U-Net — Encoder, Decoder, Skip Connections
  Lecture 5:  Time Conditioning & Spatial Attention

Module 3: Training & Sampling
  Lecture 6:  Training a Diffusion Model
  Lecture 7:  Sampling — DDPM, DDIM & Beyond

Module 4: Controllable Generation
  Lecture 8:  Conditioning & Classifier Guidance
  Lecture 9:  Classifier-Free Guidance

Module 5: Scaling Diffusion
  Lecture 10: Latent Diffusion & Stable Diffusion
  Lecture 11: Prediction Targets & Advanced Schedules

Module 6: Modern Architectures
  Lecture 12: Diffusion Transformers (DiT)
  Lecture 13: Flow Matching & Rectified Flows

Assessment Track (separate section)
  Coding Challenges
  Conceptual Questions
  Debug Challenge
  System Design Discussion
```

---

## Detailed Lecture Breakdown

### Module 1: The Diffusion Framework

*What diffusion models are, the math that makes them work, and why they've taken over generative modeling.*

**Prerequisite knowledge:** Probability distributions, Bayes' rule, KL divergence, gradient descent, neural networks as function approximators.

---

#### Lecture 1: Why Diffusion? The Generative Models Landscape

| | |
|---|---|
| **Source material** | Module 05 §5.1 |
| **Estimated depth** | Conceptual (light math) |
| **Dependencies** | None — this is the entry point |

**Content:**
- The generative modeling problem: learn p(x) from samples
- Survey of approaches and their tradeoffs:
  - GANs: adversarial training, mode collapse, no explicit density
  - VAEs: ELBO, blurry outputs, amortized inference
  - Normalizing flows: exact likelihood, architectural constraints
  - Autoregressive: tractable likelihood, slow sequential generation
- **Diffusion models:** stable training, state-of-the-art quality, flexible architecture
- The core intuition: destruction is easy (add noise), generation is learned (remove noise)
- Historical timeline: Sohl-Dickstein 2015 → DDPM 2020 → explosion of progress

**Why this comes first:** Sets the stage. Students need to understand what problem diffusion solves and why it's worth a full course before diving into equations.

---

#### Lecture 2: The Forward Process & Noise Schedules

| | |
|---|---|
| **Source material** | Module 05 §5.2–5.4, §5.8–5.9 |
| **Estimated depth** | Math-heavy (derivations + implementation) |
| **Dependencies** | Lecture 1 |

**Content:**
- The forward (noising) process: `q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I)`
- Building the schedule: β_t, α_t = 1-β_t, ᾱ_t = ∏α_s
- **The closed-form shortcut:** `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε` — skip directly to any timestep
- The reparameterization trick: deterministic function + noise → differentiable sampling
- Signal-to-noise ratio: `SNR(t) = ᾱ_t / (1-ᾱ_t)` — the unified lens
- Noise schedules and their impact:
  - **Linear** (DDPM): β_t from 0.0001 to 0.02
  - **Cosine** (Improved DDPM): smoother SNR decay, less early destruction
  - **Sigmoid**: compromise between linear and cosine
- Visualizing: what images look like at different noise levels, ᾱ_t curves, log-SNR curves

**Why combined:** Forward process and schedules are inseparable — the schedule *defines* the forward process. Seeing them together builds intuition for what noise levels mean.

---

#### Lecture 3: The Reverse Process & the Simplified Loss

| | |
|---|---|
| **Source material** | Module 05 §5.5–5.7 |
| **Estimated depth** | Math-heavy (the deepest derivation in the course) |
| **Dependencies** | Lecture 2 |

**Content:**
- The reverse process: learn `p_θ(x_{t-1} | x_t)` to undo noising
- Why the true reverse is intractable (requires marginalizing over all x_0)
- The tractable posterior: `q(x_{t-1} | x_t, x_0)` is Gaussian (derivation)
- The ELBO decomposition: log-likelihood → sum of per-timestep KL terms
- KL between two Gaussians → mean matching
- **The ε-reparameterization:** predicting noise instead of the mean
  - `L = ||ε - ε_θ(x_t, t)||²` — why this works
- The simplified loss: drop per-timestep weights → `L_simple = E[||ε - ε_θ(...)||²]`
- **DDPM Algorithm 1 (Training):** the 6-line algorithm that changed everything
- Alternative parameterizations preview: ε-prediction vs x_0 vs v-prediction (detailed treatment in Lecture 11)

**Why this is its own lecture:** This is the mathematical heart of diffusion. It deserves focused attention without being diluted by architecture or implementation details.

---

### Module 2: The Denoising Network

*The neural network that learns to predict noise. How U-Net's encoder-decoder-skip architecture, time conditioning, and attention placement create the ideal denoiser.*

**Prerequisite knowledge:** Convolutional layers, residual connections (ResNets), group normalization, self-attention mechanism.

---

#### Lecture 4: U-Net — Encoder, Decoder, Skip Connections

| | |
|---|---|
| **Source material** | Module 04 §4.1–4.4 |
| **Estimated depth** | Architecture (implementation-focused) |
| **Dependencies** | Lecture 3 (need to know what the network predicts) |

**Content:**
- U-Net history: from medical segmentation (2015) to diffusion denoiser (2020)
- What changed for diffusion: time conditioning, GroupNorm, attention, SiLU activation
- **Encoder path:** progressive downsampling
  - Channel progression: 64 → 128 → 256 → 512
  - Spatial resolution halving at each level
  - ResBlocks: Conv → GroupNorm → SiLU → Conv → residual
- **Decoder path:** upsampling + skip concatenation
  - Mirror of encoder, channels decrease
  - Channel math: skip_ch + decoder_ch → conv to reduce
- **Skip connections — why they're critical:**
  - Fine-grained spatial info flows encoder → decoder
  - Gradient highway for training
  - Concatenation (DDPM) vs addition
  - Ablation: denoising quality with vs without skips
- Tensor shapes at every stage (the grounding principle)

**Content gap to fill:** Skip connection ablation was specified in the Module 04 spec but missing from the notebook. This is valuable pedagogy — implement for the website.

**Content gap to fill:** U-Net architecture diagram (original vs diffusion-adapted) was specified but missing. For a website, this is critical — create an SVG or interactive diagram.

---

#### Lecture 5: Time Conditioning & Spatial Attention

| | |
|---|---|
| **Source material** | Module 04 §4.5–4.9, Module 03 §3.3 (sinusoidal encoding), §3.6 (spatial attention) |
| **Estimated depth** | Architecture (implementation-focused) |
| **Dependencies** | Lecture 4 |

**Content:**
- **Time conditioning — how the network knows what noise level it's denoising:**
  - Sinusoidal timestep embedding (same formula as transformer positional encoding, different purpose)
  - MLP projection to match feature dimensions
  - Injection methods:
    - Addition: project time_emb, add to features after GroupNorm
    - Scale-shift (AdaGN): predict γ, β from time_emb → `y = γ·GroupNorm(x) + β`
- **Where attention goes in U-Net:**
  - O(n²) cost means attention only at lower resolutions (16×16, 8×8)
  - Spatial self-attention: reshape (B,C,H,W) → (B,H×W,C), attend, reshape back
  - DDPM: attention at 16×16 only. Improved DDPM: 32×32, 16×16, 8×8
- **The full ResBlock with time embedding:** complete data flow and tensor shapes
- **Assembling the full U-Net:**
  - Initial conv → time MLP → down blocks → middle block → up blocks → final conv
  - Middle block: ResBlock → Attention → ResBlock
  - Parameter count analysis and where compute lives
- **Cross-attention preview:** Q from features, K/V from conditioning signal (detailed in Module 4)

**Why attention content from Module 03 is folded here:** Students know attention from CS231N. What they need to learn is *how diffusion models use it* — spatial reshaping, resolution-dependent placement, cross-attention for conditioning. That's architectural knowledge, not attention fundamentals.

**Content gap to fill:** AdaGN (scale-shift) time injection was specified but only additive was implemented.

**Critical fix from review:** The U-Net is MNIST-hardcoded (28×28). Must use (B, 3, 32, 32) CIFAR-10 input as the spec requires. The exported `utils/unet.py` must be THE canonical U-Net used in all subsequent lectures.

---

### Module 3: Training & Sampling

*From theory to practice. How to train a diffusion model (DDPM Algorithm 1) and generate images from pure noise (Algorithm 2, DDIM, and beyond).*

---

#### Lecture 6: Training a Diffusion Model

| | |
|---|---|
| **Source material** | Module 06 §6.1–6.9 |
| **Estimated depth** | Implementation-heavy |
| **Dependencies** | Lectures 3, 5 (need the loss and the architecture) |

**Content:**
- **Data loading & preprocessing:**
  - Normalize to [-1, 1] (matches zero-mean Gaussian noise)
  - Data augmentation (random horizontal flip)
- **DDPM Algorithm 1 — the training loop:**
  1. Sample x_0 from data
  2. Sample random t ~ Uniform(1, T)
  3. Sample ε ~ N(0, I)
  4. Compute x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
  5. Predict ε_θ(x_t, t)
  6. Loss = MSE(ε, ε_θ)
- **Timestep sampling strategies:**
  - Uniform (standard)
  - Importance-weighted (P2 weighting by SNR)
- **Training stability:**
  - Gradient clipping (prevent explosion)
  - Learning rate warmup + cosine decay
  - AdamW optimizer (standard for diffusion)
- **EMA (Exponential Moving Average):**
  - `θ_ema = 0.9999·θ_ema + 0.0001·θ` — use for sampling, not gradients
  - Why: smoother weights → better FID
- **Common pitfalls:** loss spikes, plateaus, NaN, color shift, slow convergence (100K+ steps is normal)
- **Monitoring:** visual inspection > loss curve > FID

**Critical fix from review:** Import U-Net from `utils/unet.py` instead of redefining. Fix the UpBlock skip connection bug (only uses last skip per level). Three incompatible U-Nets across Modules 4/6/7 is the review's #1 finding.

---

#### Lecture 7: Sampling — DDPM, DDIM & Beyond

| | |
|---|---|
| **Source material** | Module 07 §7.1–7.7 |
| **Estimated depth** | Math + implementation |
| **Dependencies** | Lecture 6 |

**Content:**
- **DDPM sampling (Algorithm 2):**
  - Start x_T ~ N(0, I)
  - For t = T, T-1, ..., 1: predict ε_θ → compute μ_θ → sample x_{t-1} = μ_θ + σ_t z
  - No noise at final step (t=1)
  - Variance choices: β_t (simple) vs β̃_t (posterior)
  - **Problem:** 1000 forward passes = slow
- **DDIM — deterministic sampling:**
  - Generalize with η ∈ [0,1]: η=1 is DDPM, η=0 is fully deterministic
  - Same trained model, different sampler
  - Non-Markovian forward process with identical marginals
  - **Key benefit:** timestep sub-selection (use 50 steps instead of 1000, 20× faster)
- **Accelerated sampling:**
  - Uniform spacing, quadratic spacing
  - Quality vs speed tradeoff: 50 steps ≈ minimal loss, 10 steps = visible degradation
- **Probability Flow ODE perspective** (optional/advanced):
  - DDIM(η=0) = Euler discretization of an ODE
  - Score function: ∇log p_t ≈ -ε_θ / √(1-ᾱ_t)
  - Advanced solvers (DPM-Solver, Heun) for even fewer steps
- **Dynamic thresholding** (Imagen): clip x_0 prediction to percentile for high-guidance stability
- **The full sampling pipeline:** noise → iterative denoising → clamp → denormalize → display

**Critical fix from review:** Must import from `utils/unet.py` and load Module 6 checkpoint successfully. The review found checkpoint incompatibility due to architecture mismatch.

---

### Module 4: Controllable Generation

*How to tell a diffusion model what to generate. From class labels to text prompts, from classifier gradients to classifier-free guidance.*

---

#### Lecture 8: Conditioning & Classifier Guidance

| | |
|---|---|
| **Source material** | Module 08 §8.1–8.2 |
| **Estimated depth** | Conceptual + implementation |
| **Dependencies** | Lecture 7 |

**Content:**
- **The shift:** unconditional ε_θ(x_t, t) → conditional ε_θ(x_t, t, c)
- **Class-conditional generation:**
  - Embedding the class label (nn.Embedding → learned vector)
  - Injection methods: addition to time embedding, concatenation, AdaGN
  - Architecture modification: add embedding layer, modify ResBlocks
- **Classifier guidance** (historical context — superseded by CFG):
  - Use external classifier p_φ(y | x_t) trained on noisy images
  - During sampling: shift noise prediction by classifier gradient
  - `ε̂ = ε_θ - s·√(1-ᾱ_t)·∇log p_φ(y|x_t)`
  - Guidance scale s: higher → sharper but less diverse
  - **Problem:** requires training a separate classifier on all noise levels

---

#### Lecture 9: Classifier-Free Guidance

| | |
|---|---|
| **Source material** | Module 08 §8.3–8.7, Module 03 §3.5 (cross-attention) |
| **Estimated depth** | Core concept (likely interview topic) |
| **Dependencies** | Lecture 8 |

**Content:**
- **The brilliant trick:** random label dropout during training (10–20%)
  - Same model learns conditional AND unconditional prediction
  - No separate classifier needed
- **The guidance formula:**
  - `ε̂ = ε_uncond + s·(ε_cond - ε_uncond)`
  - s=1 (standard conditional), s>1 (amplified), s=0 (unconditional)
  - Requires 2 forward passes (or batch them)
  - Typical scales: 2–4 (class labels), 7.5–15 (text-to-image)
- **Why CFG works — the diversity-fidelity tradeoff:**
  - Higher guidance = sharper, more "on-topic" outputs
  - Mathematically: sampling from p(x|c)^s — sharpened conditional
  - Analogous to inverse temperature
- **Text conditioning via cross-attention:**
  - Text → encoder (CLIP/T5) → embedding sequence
  - Injected via cross-attention: Q from image features, K/V from text
  - CFG with text: drop text conditioning randomly during training
- **Negative prompts:**
  - Replace ε_uncond with ε_neg in the guidance formula
  - `ε̂ = ε_neg + s·(ε_cond - ε_neg)` — steer *away* from negative concept

**Fix from review:** CFG formula presented inconsistently between LaTeX and code. Show explicit algebraic equivalence. Device mismatch — schedule tensors on CPU mixed with model tensors on GPU/MPS.

---

### Module 5: Scaling Diffusion

*How to go from 32×32 MNIST to 512×512 photorealistic images. The innovations that make modern diffusion practical.*

---

#### Lecture 10: Latent Diffusion & Stable Diffusion

| | |
|---|---|
| **Source material** | Module 09 §9.1–9.4 |
| **Estimated depth** | Architecture + conceptual |
| **Dependencies** | Lectures 5, 9 |

**Content:**
- **The resolution bottleneck:**
  - 256×256 = 196K values, 512×512 = 786K values
  - Attention is O(n²) in spatial dimension — infeasible at high res
- **VAE primer** (compact — CS231N students know the idea):
  - Encoder: image → latent (256×256×3 → 32×32×4, 64× compression)
  - Decoder: latent → reconstructed image
  - ELBO: reconstruction loss + KL regularization
  - Reparameterization trick (same as diffusion!)
- **Latent diffusion — the key insight:**
  - Train VAE separately → freeze it
  - Encode all training images: z_0 = E(x)
  - Run diffusion entirely in latent space (much smaller U-Net)
  - Inference: sample z_0 from diffusion → decode x = D(z_0)
  - Benefits: faster, perceptually meaningful compression, separates perception from generation
- **The Stable Diffusion architecture:**
  - Three components: VAE (8× spatial compression), U-Net (64×64 latent, cross-attention), text encoder (CLIP/T5)
  - Full inference pipeline walkthrough
  - SD 1.5 → 2.x → SDXL → SD3 evolution

**Fix from review:** VAE uses BatchNorm2d (contradicts GroupNorm recommendation). LatentDiffusionMLP uses linear time embedding instead of sinusoidal.

---

#### Lecture 11: Prediction Targets & Advanced Schedules

| | |
|---|---|
| **Source material** | Module 09 §9.5–9.6 |
| **Estimated depth** | Math + practical |
| **Dependencies** | Lectures 3, 10 |

**Content:**
- **Three prediction parameterizations** (mathematically equivalent, different properties):
  - **ε-prediction:** predict noise. Standard (DDPM). Stable at high noise, struggles near t=0.
  - **x_0-prediction:** predict clean image. Good at low noise, unstable at high noise.
  - **v-prediction:** predict velocity `v = √ᾱ_t·ε - √(1-ᾱ_t)·x_0`. Balanced, stable across all t. Used in Imagen, SD 2.x.
- Conversion formulas between all three (table)
- When to use which (practical guidance)
- **Noise schedule improvements:**
  - Cosine schedule (Improved DDPM): smoother SNR decay
  - Offset noise: per-channel bias for extreme values (very dark/bright images)
  - Zero terminal SNR: ensure SNR→0 at t=T (fixes train-inference mismatch)
  - Log-SNR linear: equal difficulty across timesteps
  - Continuous-time schedules: t ∈ [0,1], arbitrary step counts

---

### Module 6: Modern Architectures

*Beyond U-Net. The architectures and training frameworks driving the current frontier.*

---

#### Lecture 12: Diffusion Transformers (DiT)

| | |
|---|---|
| **Source material** | Module 09 §9.7 |
| **Estimated depth** | Architecture (conceptual + implementation) |
| **Dependencies** | Lectures 5, 10 |

**Content:**
- **The trend:** replace U-Net with a vision transformer
- **DiT architecture:**
  - Patchify: image → non-overlapping patches → flatten to sequence
  - Positional encoding (2D)
  - Transformer blocks (standard: LayerNorm → Attention → FFN)
  - Unpatchify: sequence → image
- **Conditioning via adaLN-Zero:**
  - Adaptive layer norm: predict scale, shift, gate from time+class embedding
  - Gate initialized to zero → identity function at init → stable training
- **Why transformers for diffusion:**
  - Better scaling properties (transformer scaling laws apply)
  - Simpler architecture (no encoder-decoder, no skip connections)
  - More flexible conditioning
  - DiT scaling results: DiT-XL/2 achieves state-of-the-art FID on ImageNet
- **U-Net vs DiT comparison:** when each is preferred

**Content gap:** The review noted no exercise for DiT efficiency comparison. Add "Train DiT on MNIST" exercise.

**Fix from review:** DiT forward uses float t but integer-based schedule. Align time representations.

---

#### Lecture 13: Flow Matching & Rectified Flows

| | |
|---|---|
| **Source material** | Module 09 §9.8 |
| **Estimated depth** | Math + conceptual |
| **Dependencies** | Lecture 7 (sampling/ODE perspective) |

**Content:**
- **A different framework:** learn a velocity field transporting noise → data
- **The formulation:**
  - Straight-line interpolation: x_t = (1-t)·x_0 + t·x_1
  - Training loss: `||v_θ(x_t, t) - (x_1 - x_0)||²`
  - Simpler than DDPM: no schedule tuning, no cumulative products
- **Why flow matching works:**
  - Straight paths → fewer ODE steps needed
  - No noise schedule to design
  - Simpler math, easier to understand
  - Connection to optimal transport
- **Rectified flows:** iteratively straighten trajectories for even fewer steps
- **Connection to diffusion:**
  - Flow matching is a continuous generalization
  - DDPM can be viewed as a special case
  - Score matching ↔ flow matching duality
- **Practical impact:** used in Stable Diffusion 3, modern architectures

**Content gap from review:** Flow matching convention (t=0 data vs t=1 data) is not clarified. Add explicit note.

---

### Assessment Track

*Self-evaluation exercises. Separate section of the site — not a numbered module.*

| | |
|---|---|
| **Source material** | Module 10 (all sections) |
| **Dependencies** | All lectures |

**Pages:**

1. **Coding Challenges** (from 10.1–10.3)
   - 2D point cloud diffusion (45 min)
   - DDIM + CFG sampling (45 min)
   - Training step from scratch (45 min)
   - Each with: timer, starter code, evaluation criteria, solution, debrief

2. **Conceptual Questions** (from 10.4)
   - 16 verbal explanation prompts with model answers
   - Core concepts, architecture, guidance, advanced topics

3. **Debug Challenge** (from 10.5)
   - Find 10 bugs in a complete diffusion pipeline
   - Tests code reading and understanding, not implementation
   - **Critical fix:** Remove `# BUG:` labels — they defeat the purpose

4. **System Design** (from 10.6)
   - Scale 32×32 → 512×512: architecture, training, inference decisions
   - Open-ended discussion format

---

## Content Mapping: Source → Destination

| Original Module | Destination | Treatment |
|---|---|---|
| **00 — NumPy** | Cut | Prerequisite (CS231N) |
| **01 — PyTorch** | Cut | Prerequisite (CS231N) |
| **02 — ConvNets** | Cut | GroupNorm, residual connections, transposed convolutions assumed known. Referenced in U-Net lectures where needed. |
| **03 — Attention** | Partially absorbed | §3.3 (sinusoidal encoding) → Lecture 5. §3.5 (cross-attention) → Lecture 9. §3.6 (spatial attention) → Lecture 5. Generic attention mechanics cut. |
| **04 — U-Net** | Lectures 4–5 | Reordered after math (Module 1). |
| **05 — Diffusion Math** | Lectures 1–3 | Split into 3 focused lectures. Moved to front of course. |
| **06 — Training** | Lecture 6 | Condensed into single lecture. |
| **07 — Sampling** | Lecture 7 | Condensed. DDPM + DDIM + ODE in one lecture. |
| **08 — Conditioning** | Lectures 8–9 | Split: conditioning/classifier guidance vs CFG. |
| **09 — Advanced** | Lectures 10–13 | Split into 4 lectures across 2 modules. |
| **10 — Interview** | Assessment track | Separate section, not a numbered module. |

---

## Content Redundancies to Resolve

| Redundancy | Where | Resolution |
|---|---|---|
| Three incompatible U-Net implementations | Notebooks 04, 06, 07 | Single canonical U-Net in `utils/unet.py`. All lectures import it. |
| Cosine schedule reimplemented 3 times | Notebooks 05, 06, `utils/schedule.py` | Single implementation in `utils/schedule.py`. All lectures import it. |
| Reparameterization trick explained 3 times | Notebooks 00, 05, 09 | Teach once in Lecture 2, reference in Lectures 3 and 10. |
| CFG formula in two forms without equivalence | Notebook 08 §8.3–8.4 | Show both forms and explicit algebraic equivalence in Lecture 9. |

---

## Content Gaps to Fill

| Gap | Source | Priority |
|---|---|---|
| U-Net architecture diagram (SVG/interactive) | Module 04 spec §4.1 | **High** — a course website needs visual diagrams |
| Skip connection ablation (with/without comparison) | Module 04 spec §4.4 | **High** — great pedagogy, specified but missing |
| End-to-end diffusion pipeline diagram | New for website | **High** — single visual: forward → training → sampling |
| AdaGN scale-shift time injection | Module 04 spec §4.5 | Medium — only additive was implemented |
| CIFAR-10 training | Module 06 spec capstone | Medium — MNIST-only currently |
| DDPM vs DDIM vs DPM-Solver comparison table | New for website | Medium — side-by-side sampling methods |
| DiT exercise (parameter efficiency comparison) | Module 09 spec §9.7 | Medium |
| Stable Diffusion inference walkthrough with tensor shapes | New for website | Medium — currently only conceptual |
| Memory profiling of U-Net | Module 04 spec §4.9 | Low |
| P2 loss-aware weighting discussion | Module 06 spec §6.3 | Low |
| Offset noise comparison (with/without) | Module 09 spec §9.6 | Low |

---

## Content to Cut or Make Optional

| Content | Recommendation |
|---|---|
| All of Modules 00–01 (NumPy, PyTorch) | **Cut.** Prerequisite. Offer optional appendix pages linked from prerequisites card. |
| All of Module 02 (ConvNets) | **Cut.** Prerequisite. Depthwise separable, basic conv, padding/stride are CS231N. |
| Generic attention from Module 03 (§3.1–3.2, §3.4, §3.7–3.8) | **Cut.** Known from CS231N. Only diffusion-specific usage retained. |
| Classifier guidance (Lecture 8, §8.2) | **Keep but mark "historical context."** Superseded by CFG. |
| Probability Flow ODE (Lecture 7, §7.7) | **Keep but mark optional/advanced.** Important for understanding, not implementation-critical. |
| Stable Diffusion architecture overview | **Keep as conceptual.** No implementation, just architecture understanding. |
| ViT capstone from Module 03 | **Cut.** Tangent from diffusion track. |

---

## Dependency Graph

```
Lecture 1 (Why Diffusion?)
    │
    ▼
Lecture 2 (Forward Process & Schedules)
    │
    ▼
Lecture 3 (Reverse Process & Loss)
    │
    ├───────────────────────────┐
    ▼                           ▼
Lecture 4 (U-Net Enc/Dec)  Lecture 7 ────────────────┐
    │                      (Sampling)                 │
    ▼                           │                     │
Lecture 5 (Time & Attention)    │                     │
    │                           ▼                     │
    ├──────► Lecture 6 ───► Lecture 8                  │
    │       (Training)     (Conditioning)              │
    │                           │                     │
    │                           ▼                     │
    │                      Lecture 9 ◄────────────────┘
    │                      (CFG)
    │                           │
    ├───────────────────────────┤
    │                           ▼
    │                      Lecture 10
    │                      (Latent Diffusion)
    │                           │
    │                      ┌────┴────┐
    │                      ▼         ▼
    │                 Lecture 11  Lecture 12
    │                 (Pred Targets) (DiT)
    │                                │
    │                           Lecture 13
    │                           (Flow Matching)
    │
    └──────────────► Assessment Track
                     (requires all core)
```

**Core path (9 lectures):** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

**Full path (13 lectures):** Core path + 10, 11, 12, 13

**Assessment:** After completing core path (ideally full path for advanced questions).

---

## Appendix Pages (Optional, from cut content)

These wouldn't appear in the main sidebar nav but would be linked from a "Prerequisites" card at the top of the site:

1. **PyTorch Refresher** — condensed from Module 01. Tensor creation, autograd, nn.Module, custom layers, optimizers.
2. **Convolution Refresher** — condensed from Module 02. 2D conv, padding/stride, transposed conv, GroupNorm, residual connections.
3. **Attention Refresher** — condensed from Module 03. Dot-product attention, multi-head, positional encoding, transformer block.

Content exists — low effort to create, high value for students who need a brush-up.

---

## Open Questions for User

1. **Math depth in Lecture 3 (ELBO derivation):** The full derivation is 2+ pages of dense math. Options:
   - (a) Full derivation with step-by-step walkthrough
   - (b) Key steps with collapsible "expand for full derivation" sections
   - (c) State the result, link to appendix
   - **Recommendation:** (b) — the derivation IS the learning, but web format lets us collapse it.

2. **Code in lectures:** How much runnable code should lectures contain?
   - (a) Pseudocode / annotated snippets only (exercises separate in Phase 2)
   - (b) Complete implementations inline (current notebook style)
   - (c) Key implementations inline, utilities referenced
   - **Recommendation:** (c) — show learning-critical code, link to utils for boilerplate.

3. **VAE depth in Lecture 10:** CS231N students may know VAEs. Options:
   - (a) Full VAE lecture (encoder, decoder, ELBO, implementation)
   - (b) Quick refresher (1–2 paragraphs) then straight to latent diffusion
   - **Recommendation:** (b) — VAEs are prerequisite-adjacent. Quick refresher with link to deeper resource.

4. **Assessment track placement:** Should it appear in the sidebar alongside content modules, or as a separate top-level section?
   - **Recommendation:** Separate top-level section. It serves a different purpose than lectures.

5. **CIFAR-10 training:** The spec calls for it but current notebooks only do MNIST (feasible on CPU/MPS). Should we:
   - (a) Add as "extended" section with compute requirements note
   - (b) Keep MNIST-only for accessibility
   - **Recommendation:** (a) — show both, with MNIST as the default and CIFAR-10 as "if you have a GPU."

6. **Appendix pages from cut content:** Worth creating, or overkill? Content already exists in notebooks 00–03, so the effort is mostly reformatting.
   - **Recommendation:** Yes — low effort, high value as safety net for students below the prereq bar.
