# Module 8: Conditioning & Guidance

## Purpose
Understand how to control what a diffusion model generates. Class conditioning, classifier guidance, and classifier-free guidance are essential for practical diffusion models.

## 📄 Key Papers
- **Classifier-Free Diffusion Guidance** — Ho & Salimans 2022. [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
  - *Read for:* THE key paper for CFG. Section 2 for the formulation. The random label dropout training trick. The guidance scale formula. Short and essential.
- **Diffusion Models Beat GANs on Image Synthesis** — Dhariwal & Nichol 2021. [arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
  - *Read for:* Classifier guidance (using a separate classifier's gradients). Section 4 for the guidance mechanism. Also: the ADM architecture improvements.
- **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)** — Saharia et al. 2022. [arxiv.org/abs/2205.11487](https://arxiv.org/abs/2205.11487)
  - *Read for:* Text-to-image with CFG. Dynamic thresholding. Shows how guidance scale affects quality.

## Sections

### 8.1 — Class-Conditional Generation

**Concepts to teach:**
- Unconditional model: `ε_θ(x_t, t)` — generates any image
- Conditional model: `ε_θ(x_t, t, c)` — generates images of class c
- How to inject the class label:
  1. **Embedding + addition:** `c_emb = nn.Embedding(num_classes, d); time_emb = time_emb + c_emb(c)`
  2. **Concatenation:** add class embedding to the time embedding before MLP
  3. **AdaGN:** predict different scale/shift per class (most expressive)
- Class label as integer → learned embedding → injected same way as timestep
- Training: each sample has a class label, model learns to generate class-specific images

**Worked example:**
- Modify the U-Net from Module 4 to accept a class label
- Implement class embedding + addition to time embedding
- Forward pass: `model(x_t, t, class_label)` → noise prediction

**Exercise:**
- Add class conditioning to your MNIST U-Net
- Train class-conditional model: generate specific digits on command

---

### 8.2 — Classifier Guidance

**Concepts to teach:**
- Idea: use a separately trained classifier `p_φ(y | x_t)` to steer generation
- During sampling, at each step:
  1. Predict noise: `ε_θ(x_t, t)`
  2. Compute classifier gradient: `∇_{x_t} log p_φ(y | x_t)`
  3. Shift the noise prediction: `ε̂ = ε_θ - s · √(1-ᾱ_t) · ∇_{x_t} log p_φ(y | x_t)`
  4. Use ε̂ for the denoising step
- s is the guidance scale: higher s = more influence from classifier = sharper but less diverse
- Problem: requires training a separate classifier on noisy images (at all noise levels)
- This is why classifier-free guidance was invented — avoids the separate classifier

**Worked example:**
- Train a simple classifier on noisy MNIST images
- Implement classifier-guided sampling
- Show the effect of guidance scale s on sample quality and diversity

**Exercise:**
- Implement classifier guidance, sweep s from 0 to 10

---

### 8.3 — Classifier-Free Guidance (CFG): The Key Insight

**Concepts to teach:**
- **The brilliant trick:** during training, randomly drop the class label (replace with a null/unconditional token) with some probability (typically 10-20%)
- This means the SAME model learns both:
  - Conditional prediction: `ε_θ(x_t, t, c)` — when class label is given
  - Unconditional prediction: `ε_θ(x_t, t, ∅)` — when class label is dropped (null token)
- No separate classifier needed!
- Implementation during training:
  ```python
  # Randomly drop labels with probability p_uncond (e.g., 0.1)
  mask = torch.rand(batch_size) < p_uncond
  labels[mask] = null_label  # e.g., num_classes (a special "no class" index)
  ```

**Worked example:**
- Modify the training loop to randomly drop class labels
- Show both conditional and unconditional predictions from the same model

---

### 8.4 — The Guidance Scale Formula

**Concepts to teach:**
- During sampling, combine conditional and unconditional predictions:
  ```
  ε̂ = ε_uncond + s · (ε_cond - ε_uncond)
  ```
  - s = 1: standard conditional generation (no guidance boost)
  - s > 1: amplified conditioning — sharper, more "on-topic" samples
  - s = 0: unconditional generation
  - s < 0: "negative" guidance — move AWAY from the condition
- Requires **two forward passes** per sampling step: one with label, one without
  - Or batch them: `model(cat([x_t, x_t]), cat([t, t]), cat([c, null]))` and split output
- Typical guidance scales: 2-4 for class-conditional, 7.5-15 for text-to-image
- The math: this is implicitly modeling `p(x|c) ∝ p(x) · p(c|x)^s` — amplifying the classifier-like signal

**Worked example:**
- Implement CFG in the sampling loop
- Batch the conditional/unconditional forward passes for efficiency
- Generate samples at s=1, 2, 4, 8 — show increasing fidelity but decreasing diversity

**Exercise:**
- Implement CFG-guided sampling with configurable guidance scale
- Sweep guidance scales, create a grid showing the quality/diversity tradeoff
- Show that s=1 matches pure conditional sampling

---

### 8.5 — Why CFG Works: Trading Diversity for Fidelity

**Concepts to teach:**
- The diversity-fidelity tradeoff: higher guidance = more "on-topic" but less varied
- Visually: s=1 generates diverse digits but some are ambiguous; s=7 generates perfect digits but they look similar
- Mathematically: CFG sharpens the conditional distribution, concentrating mass on high-likelihood samples
- The "temperature" analogy: guidance scale acts like inverse temperature on the conditional distribution
- In practice: almost all state-of-the-art results use CFG with s > 1

**Worked example:**
- Generate 100 samples at different guidance scales, compute diversity metrics (e.g., average pairwise distance)
- Plot fidelity (how recognizable) vs diversity as guidance scale increases

---

### 8.6 — Text Conditioning Preview

**Concepts to teach:**
- How text-to-image works (Stable Diffusion, Imagen, DALL-E):
  1. Text → text encoder (CLIP or T5) → text embeddings
  2. Text embeddings injected via **cross-attention** in the U-Net (Module 3.5)
  3. Q from image features, K/V from text embeddings
  4. CFG with text: drop text conditioning randomly during training
- This is the same conditioning mechanism as class labels, just with richer embeddings
- We won't implement this fully (requires pretrained text encoders) but understanding the architecture is important

**Content:** Conceptual with architecture diagrams. Show where cross-attention fits in the U-Net.

---

### 8.7 — Negative Prompts: How They Work Mechanically

**Concepts to teach:**
- In the CFG formula: `ε̂ = ε_uncond + s · (ε_cond - ε_uncond)`
- "Negative prompt" replaces ε_uncond with ε_neg:
  `ε̂ = ε_neg + s · (ε_cond - ε_neg)`
- This steers generation AWAY from the negative prompt
- Example: positive="cat", negative="blurry" → generates sharp cats
- Mechanical understanding: the model amplifies the difference between what you want and what you don't want

**Content:** Conceptual. Brief exercise to implement negative prompt support in the sampler.

---

## Module 8 Capstone Exercise

**Add class conditioning to your MNIST diffusion model, implement CFG:**

1. Modify U-Net to accept class labels (with null token for unconditional)
2. Modify training loop to randomly drop labels (10% of the time)
3. Train for 20K-50K steps
4. Implement CFG-guided sampling
5. Generate a grid: rows = digits 0-9, columns = guidance scales 1, 2, 4, 8
6. Demonstrate: "generate the digit 7" with different guidance scales
7. Show unconditional generation (s=0) vs guided generation
