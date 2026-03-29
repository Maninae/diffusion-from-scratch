# Lecture 9: Classifier-Free Guidance
**Module:** 4 — Controllable Generation
**Estimated reading time:** 25 minutes
**Dependencies:** Lecture 8, Lecture 7 (for DDIM connection)

## Learning Objectives
- Explain the classifier-free guidance training trick: random label dropout creates a single model that does both conditional and unconditional prediction
- Derive and implement the CFG formula: ε̂ = ε_uncond + s · (ε_cond - ε_uncond)
- Interpret guidance scale s as inverse temperature on the conditional distribution
- Understand text conditioning via cross-attention (conceptual — builds on spatial attention from Lecture 5)
- Explain how negative prompts work mechanically within the CFG framework

## Narrative Arc
**The problem:** Classifier guidance (Lecture 8) works, but it requires training a separate classifier on noisy images at all noise levels. Want to condition on text instead of class labels? Train a new classifier. Want to change guidance strength? Still need that separate model. The approach is expensive, inflexible, and inelegant.

**The attempt:** What if we could get guidance without any classifier at all? The trick would need to come from the diffusion model itself — somehow combining conditional and unconditional predictions to amplify the conditioning signal.

**The solution:** Train a SINGLE model that can do both conditional and unconditional prediction. During training, randomly drop the conditioning (replace class label with a null token) 10-20% of the time. At inference, run the model twice — once with and once without the condition — and extrapolate in the direction of the conditioning signal. The guidance scale controls how aggressively you amplify. No separate classifier, no extra training, no gradient computation. This is classifier-free guidance (CFG), and it's used in virtually every modern diffusion system.

## Section Outline

### Section 9.1: The Training Trick — Random Label Dropout
- During training, with probability p_uncond (typically 10-20%):
  - Replace the class label with a special null token (e.g., index = num_classes)
  - The model learns to predict noise without class information
- With probability 1 - p_uncond:
  - Train normally with the real class label
- Result: the SAME model learns both:
  - `ε_θ(x_t, t, c)` — conditional noise prediction (when given a label)
  - `ε_θ(x_t, t, ∅)` — unconditional noise prediction (when given null token)

**Pseudocode:**
```
def train_step_cfg(model, x_0, class_label, schedule, p_uncond=0.1):
    # Randomly drop labels
    drop_mask = random(batch_size) < p_uncond
    class_label[drop_mask] = NULL_CLASS     # null token

    # Standard training with modified labels
    t = random_integers(0, T-1, batch_size)
    noise = randn_like(x_0)
    x_t = q_sample(x_0, t, noise, schedule)
    noise_pred = model(x_t, t, class_label)
    loss = mse_loss(noise_pred, noise)
    return loss
```

### Section 9.2: The Guidance Formula
- At inference, for each denoising step:
  1. Run the model with the condition: `ε_cond = ε_θ(x_t, t, c)`
  2. Run the model without condition: `ε_uncond = ε_θ(x_t, t, ∅)`
  3. Combine: **`ε̂ = ε_uncond + s · (ε_cond - ε_uncond)`**
- The term `(ε_cond - ε_uncond)` is the "conditioning signal" — it captures what's different about the conditional prediction
- s is the guidance scale:
  - s = 0: pure unconditional generation
  - s = 1: standard conditional generation (no amplification)
  - s > 1: amplified conditioning — sharper, more "on-topic" outputs
  - Typical: s = 2-4 for class labels, s = 7.5-15 for text-to-image

**Algebraic equivalence:** the formula can also be written as:
- `ε̂ = (1-s) · ε_uncond + s · ε_cond`
- To see they're equivalent: expand the first form: `ε_uncond + s·ε_cond - s·ε_uncond = (1-s)·ε_uncond + s·ε_cond`
- The first form makes the "amplification" interpretation clearer (we're adding s times the conditioning signal). The second form shows it's a weighted interpolation (and extrapolation when s > 1).

**Efficiency — batched forward pass:**
```
# Instead of two separate forward passes:
x_combined = concat([x_t, x_t], dim=0)         # (2B, C, H, W)
t_combined = concat([t, t], dim=0)              # (2B,)
c_combined = concat([class_label, null_labels]) # (2B,)

noise_pred = model(x_combined, t_combined, c_combined)
eps_cond, eps_uncond = noise_pred.chunk(2, dim=0)
eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

### Section 9.3: Why CFG Works — The Diversity-Fidelity Tradeoff
- **Mathematically:** CFG samples from a sharpened conditional distribution:
  - `p_cfg(x | c) ∝ p(x) · (p(c | x) / p(c))^s`
  - By Bayes' rule, p(c|x)/p(c) = p(x|c)/p(x), so this becomes:
    `p(x) · (p(x|c)/p(x))^s = p(x|c)^s / p(x)^{s-1}`
  - Higher s → sharper distribution → samples concentrate near the mode
- **The temperature analogy:** s acts like inverse temperature
  - s = 1 → sampling at "temperature 1" (standard conditional)
  - s > 1 → "lower temperature" — distribution peaks sharpen, tails shrink
  - s → ∞ → always generate the single most likely image for that class
- **Visual effect:**
  - s = 1: diverse but some samples are ambiguous/low-quality
  - s = 4: most samples are sharp, clear, on-topic — some diversity lost
  - s = 10: very sharp but all samples look similar — diversity severely reduced
- **Why this matters practically:** nearly all SOTA results use CFG with s > 1. It's the single most impactful trick for improving generation quality.

### Section 9.4: Text Conditioning via Cross-Attention
- CFG generalizes beyond class labels to any conditioning signal
- **Text-to-image pipeline (Stable Diffusion, Imagen, DALL-E):**
  1. Text prompt → text encoder (CLIP or T5) → embedding sequence of shape (seq_len, d_text)
  2. Embedding injected via **cross-attention** in the U-Net:
     - Q from image features (spatial tokens)
     - K, V from text embeddings
     - Each spatial position in the image attends to all text tokens
  3. CFG with text: during training, randomly drop the text conditioning (replace with empty string embedding)
- Cross-attention is the same mechanism as self-attention (Lecture 5), but Q comes from one source and K/V from another
- We won't implement text conditioning (requires pretrained encoders), but the architecture is important to understand

### Section 9.5: Negative Prompts
- In the standard CFG formula: `ε̂ = ε_uncond + s · (ε_cond - ε_uncond)`
- **Negative prompts** replace ε_uncond with ε_neg:
  - `ε̂ = ε_neg + s · (ε_cond - ε_neg)`
- The model steers AWAY from the negative prompt and TOWARD the positive prompt
- Example: positive = "sharp photo of a cat", negative = "blurry, low quality"
  - The model amplifies the difference between "sharp cat" and "blurry low quality"
- Mechanically: requires a third forward pass (or batched as 3× the batch)
  - Unconditional (null), conditional (positive), conditional (negative)
- This is how users control Stable Diffusion quality in practice

### Section 9.6: The Complete CFG Sampling Pipeline

**Pseudocode:**
```
def cfg_sample(model, shape, schedule, class_label, guidance_scale=4.0, num_steps=50):
    x = randn(shape)
    timesteps = linspace(0, T-1, num_steps, dtype=int)
    null_label = full(batch_size, NULL_CLASS)

    for t in reversed(timesteps):
        # Batched forward pass
        x_double = concat([x, x], dim=0)
        t_double = concat([t, t], dim=0)
        c_double = concat([class_label, null_label], dim=0)

        eps_both = model(x_double, t_double, c_double)
        eps_cond, eps_uncond = eps_both.chunk(2, dim=0)

        # CFG combination
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM step with guided noise prediction
        x = ddim_step(x, eps_guided, t, schedule)

    return denormalize(x.clamp(-1, 1))
```

## Key Equations
- CFG formula: `ε̂ = ε_uncond + s · (ε_cond - ε_uncond)`
- Equivalent form: `ε̂ = (1-s) · ε_uncond + s · ε_cond`
- Implicit distribution: `p_cfg(x | c) ∝ p(x | c)^s / p(x)^{s-1}`
- Negative prompt: `ε̂ = ε_neg + s · (ε_cond - ε_neg)`

## Code Examples
- Training with random label dropout (pseudocode first, PyTorch reference)
- CFG sampling with batched forward pass
- Guidance scale sweep: generate same digit at s = {0, 1, 2, 4, 8}
- Negative prompt implementation (bonus)

## Diagrams & Visuals
- **CFG formula diagram:** visual showing ε_uncond, ε_cond, and the extrapolated ε̂ as vectors
- **Guidance scale sweep grid:** rows = digit classes (0-9), columns = s = {1, 2, 4, 8} — show increasing sharpness
- **Diversity-fidelity tradeoff curve:** plot showing fidelity (quality) vs diversity as s increases
- **Cross-attention diagram:** image features (Q) attending to text embeddings (K, V) — show the attention pattern
- **Training diagram:** show random label dropout during training — some samples have labels, some have null

## Source Material
- Module 08 §8.3 (CFG: The Key Insight)
- Module 08 §8.4 (The Guidance Scale Formula)
- Module 08 §8.5 (Why CFG Works)
- Module 08 §8.6 (Text Conditioning Preview)
- Module 08 §8.7 (Negative Prompts)
- Module 03 §3.5 (Cross-Attention)
