# Module 9: Latent Diffusion & Advanced Topics

## Purpose
Understand how diffusion scales to high-resolution images and the modern architectural directions. This is the "how does Stable Diffusion actually work" module, plus emerging approaches.

## 📄 Key Papers
- **High-Resolution Image Synthesis with Latent Diffusion Models** — Rombach et al. 2022. [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
  - *Read for:* THE latent diffusion paper (basis for Stable Diffusion). Section 3: the VAE encoder/decoder, doing diffusion in latent space. Figure 1 for the architecture. Why it's 10-100× more efficient than pixel-space diffusion.
- **Auto-Encoding Variational Bayes (VAE)** — Kingma & Welling 2013. [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
  - *Read for:* The original VAE paper. Section 2-3 for the ELBO, encoder-decoder framework, reparameterization trick. Foundation for understanding latent space.
- **Scalable Diffusion Models with Transformers (DiT)** — Peebles & Xie 2023. [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
  - *Read for:* Replacing U-Net with a transformer. Patch embedding + transformer blocks + adaLN conditioning. Table 1 for scaling results. The direction the field is moving.
- **Flow Matching for Generative Modeling** — Lipman et al. 2023. [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
  - *Read for:* The flow matching alternative to diffusion. Simpler training objective, straight paths from noise to data. Section 3 for the key formulation.
- **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** — Liu et al. 2023. [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
  - *Read for:* Rectified flows — straighter trajectories mean fewer sampling steps. Used in Stable Diffusion 3. Section 3 for rectification procedure.
- **Elucidating the Design Space of Diffusion-Based Generative Models** — Karras et al. 2022. [arxiv.org/abs/2206.00364](https://arxiv.org/abs/2206.00364)
  - *Read for:* Careful analysis of noise schedules, network preconditioning, and sampling. Excellent practical guidance. The "EDM" framework.

## Sections

### 9.1 — The Resolution Problem

**Concepts to teach:**
- Pixel-space diffusion at 256×256 with 3 channels: the U-Net processes 196,608 values per image
- At 512×512: 786,432 values — attention becomes quadratically expensive
- At 1024×1024: infeasible for most GPUs
- Memory and compute scale with spatial resolution squared
- The bottleneck: not the model size, but the feature map sizes (especially with attention)

**Worked example:**
- Profile memory usage for a U-Net at 32×32, 64×64, 128×128, 256×256
- Show the explosion point where you OOM

---

### 9.2 — VAE Primer

**Concepts to teach:**
- **Encoder:** maps image x → latent z (lower dimensional)
  - Typically: 256×256×3 → 32×32×4 (64× compression!)
  - Architecture: conv layers with downsampling, outputs mean μ and log-variance log σ²
- **Decoder:** maps latent z → reconstructed image x̂
  - Architecture: conv layers with upsampling
- **The ELBO loss:** reconstruction loss + KL divergence
  - Reconstruction: how well does the decoder reconstruct the input?
  - KL: how close is the learned latent distribution to N(0, I)?
- **Reparameterization trick** (again!): z = μ + σ · ε, ε ~ N(0, I)
- The latent space: a compressed, semantically meaningful representation of images

**Worked example:**
- Implement a simple VAE (convolutional encoder/decoder)
- Train on MNIST or CIFAR-10
- Visualize: reconstructions, latent space interpolation

**Exercise:**
- Build and train a VAE
- Verify the latent space is smooth (interpolation produces meaningful images)

---

### 9.3 — Latent Diffusion: Compress → Diffuse → Decode

**Concepts to teach:**
- The key idea: do diffusion in the LATENT space of a pretrained VAE, not pixel space
- Pipeline:
  1. **Train VAE** on images: learn encoder E and decoder D
  2. **Encode training data:** z_0 = E(x) for all training images
  3. **Train diffusion model** on z_0 instead of x — the U-Net is much smaller because z is much smaller
  4. **At inference:** sample z_0 from the diffusion model, then x = D(z_0)
- Benefits:
  - Latent space is lower-dimensional → faster training and sampling
  - Perceptual compression: VAE discards imperceptible detail
  - The diffusion model focuses on semantics, not pixel-level detail
- Stable Diffusion: VAE compresses 512×512×3 → 64×64×4 (48× compression)

**Worked example:**
- Use the trained VAE from 9.2
- Encode MNIST to latent space
- Train a diffusion model on the latent codes
- Sample: generate latent → decode → image

**Exercise:**
- Implement the full latent diffusion pipeline
- Compare: quality and speed vs pixel-space diffusion

---

### 9.4 — Stable Diffusion Architecture Overview

**Concepts to teach:**
- Three main components:
  1. **VAE:** encoder (512×512→64×64) and decoder (64×64→512×512)
  2. **U-Net:** operates on 64×64 latent, with cross-attention for text conditioning
  3. **Text encoder:** CLIP (OpenAI) or T5 — converts text prompt to embeddings
- The inference pipeline:
  1. Encode text prompt → text embeddings
  2. Sample random latent z_T ~ N(0, I)
  3. Denoise: z_{T} → z_{T-1} → ... → z_0 (U-Net, conditioned on text, with CFG)
  4. Decode: x = Decoder(z_0)
- SD 1.5 vs SD 2.x vs SDXL: progressively larger, better text encoders
- This is conceptual — we won't implement the full thing, but you should be able to explain the architecture

**Content:** Architecture diagram description, component breakdown. No implementation (too complex for prep), but clear understanding.

---

### 9.5 — v-Prediction vs ε-Prediction vs x_0 Prediction

**Concepts to teach:**
- Three ways to parameterize what the model predicts:
  1. **ε-prediction:** predict the noise. Standard in DDPM. `loss = ||ε - ε_θ(x_t, t)||²`
  2. **x_0-prediction:** predict the clean image directly. `loss = ||x_0 - x_0_θ(x_t, t)||²`
  3. **v-prediction:** predict the "velocity" `v = √ᾱ_t · ε - √(1-ᾱ_t) · x_0`. Used in Imagen, SD 2.x.
- All three are mathematically equivalent — you can convert between them given (x_t, t)
- **Why it matters:**
  - ε-prediction: numerically stable at high noise, struggles near t=0
  - x_0-prediction: good at low noise, unstable at high noise
  - v-prediction: balanced — numerically stable across all timesteps
- Conversion formulas:
  - From ε: `x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t`
  - From x_0: `ε = (x_t - √ᾱ_t · x_0) / √(1-ᾱ_t)`
  - From v: `x_0 = √ᾱ_t · x_t - √(1-ᾱ_t) · v`, `ε = √(1-ᾱ_t) · x_t + √ᾱ_t · v`

**Worked example:**
- Implement all three prediction modes
- Train the same model with each, compare loss curves

**Exercise:**
- Implement the conversion functions between all three parameterizations
- Verify they're equivalent on a trained model

---

### 9.6 — Noise Schedule Improvements

**Concepts to teach:**
- **Cosine schedule (Improved DDPM):** smoother SNR transition, less information destruction in early steps
- **Offset noise:** add a small offset to the noise (e.g., bias toward uniform color) — helps generate very dark or very bright images that pure Gaussian noise can't reach
- **Zero terminal SNR:** ensure SNR actually reaches 0 at t=T (some schedules don't, causing artifacts)
- **Continuous-time schedules:** instead of discrete t ∈ {1,...,T}, use continuous t ∈ [0, 1]. Enables arbitrary step counts at inference.
- **Log-SNR linear schedule:** linear in log-SNR space — equal difficulty across timesteps

**Worked example:**
- Implement offset noise: `noise = torch.randn_like(x) + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, 1)`
- Compare samples with and without offset noise

---

### 9.7 — DiT: Diffusion Transformer

**Concepts to teach:**
- The trend: replacing the U-Net with a vision transformer
- Architecture: patchify image (or latent) → flatten to sequence → transformer blocks → unpatchify
- Conditioning: **Adaptive Layer Norm (adaLN-Zero)** — predict scale, shift, and gate from time+class embedding
  - Similar in spirit to AdaGN in U-Net, but for transformer blocks
- No explicit encoder-decoder or skip connections — the transformer handles everything
- Scaling properties: DiT scales better with compute than U-Net (follows the transformer scaling trend)
- Used in: SORA (OpenAI), Stable Diffusion 3

**Worked example:**
- Implement a minimal DiT block: patch embedding → transformer with adaLN → unpatchify
- Compare parameter efficiency to U-Net

---

### 9.8 — Rectified Flows / Flow Matching

**Concepts to teach:**
- A different (simpler!) framework that achieves similar results to diffusion
- **Key idea:** learn a vector field that transports samples from noise distribution to data distribution
- Training objective: `loss = ||v_θ(x_t, t) - (x_1 - x_0)||²`
  - Where x_t = (1-t) · x_0 + t · x_1 (linear interpolation between data and noise)
  - Much simpler than the DDPM objective!
- **Straight paths:** the learned flow follows approximately straight lines from noise to data
- **Fewer sampling steps:** because paths are straighter, you need fewer ODE solver steps
- **Rectification:** iteratively straighten the paths for even fewer steps
- Used in: Stable Diffusion 3, modern architectures
- Connection to diffusion: flow matching is a continuous-time generalization; DDPM is a special case with a specific noise schedule

**Content:** Conceptual + simplified implementation. Show the training loop is actually simpler than DDPM.

**Exercise:**
- Implement flow matching training on 2D point clouds (Gaussian mixture → target distribution)
- Show the learned vector field with a quiver plot
- Compare: flow matching vs DDPM on the same toy problem

---

## Module 9 Capstone Exercise

**Implement a simple VAE, then do latent diffusion on the encoded space:**
1. Train a convolutional VAE on MNIST (latent dim: 32 or 64)
2. Encode all training images to latent space
3. Train a small diffusion model on the latent codes
4. Sample: generate random latent → decode → image
5. Compare: pixel-space diffusion (Module 6) vs latent diffusion — quality, speed, training time
