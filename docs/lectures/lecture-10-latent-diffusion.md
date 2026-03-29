# Lecture 10: Latent Diffusion & Stable Diffusion
**Module:** 5 — Scaling Diffusion
**Estimated reading time:** 25 minutes
**Dependencies:** Lectures 5, 9

## Learning Objectives
- Explain the resolution bottleneck: why pixel-space diffusion fails at high resolution
- Describe VAE architecture and the ELBO loss (compact treatment — students know the idea from CS231N)
- Explain the latent diffusion key insight: train VAE separately, then do diffusion in latent space
- Walk through the Stable Diffusion architecture: VAE + U-Net + text encoder
- Trace the full inference pipeline from text prompt to generated image

## Narrative Arc
**The problem:** Our diffusion model works on 28×28 MNIST. But real-world applications need 512×512 or 1024×1024. At 512×512, the U-Net processes 786K pixel values per step. Attention is O(n²) in spatial dimension — at this resolution, a single attention layer would need 4 billion entries in its attention matrix. Pixel-space diffusion at high resolution is computationally infeasible.

**The attempt:** You could avoid attention at high resolutions (only use convolutions) — but this loses the global context that makes diffusion outputs coherent. You could use progressive generation (low-res → upscale) — but this requires training multiple models and loses information.

**The solution:** Latent diffusion (Rombach et al., 2022). Train a VAE to compress images into a much smaller latent space (512×512×3 → 64×64×4, a 48× compression). Then run the entire diffusion process — forward noising, training, sampling — in this compressed latent space. The U-Net is smaller, attention is affordable, and training is 10-100× cheaper. At inference, decode the generated latent back to pixel space. This is the architecture behind Stable Diffusion.

## Section Outline

### Section 10.1: The Resolution Bottleneck
- Pixel counts at different resolutions:
  - 28×28×1 (MNIST): 784 values
  - 32×32×3 (CIFAR): 3,072 values
  - 256×256×3: 196,608 values
  - 512×512×3: 786,432 values
- Attention cost scales as O((H×W)²):
  - At 32×32: ~1M operations per attention layer
  - At 256×256: ~4 billion operations — impractical
- Memory for skip connections also explodes with resolution
- The bottleneck isn't the model size — it's the feature map sizes

### Section 10.2: VAE Primer
*(Compact treatment — CS231N students know the core idea. More than a refresher, less than a full lecture.)*

**Encoder:** maps image x → latent z (lower dimensional)
- Architecture: convolutional layers with downsampling
- Outputs two things: mean μ and log-variance log σ² of the latent distribution
- Typical compression: 256×256×3 → 32×32×4 (64× compression)

**Decoder:** maps latent z → reconstructed image x̂
- Architecture: convolutional layers with upsampling
- Trained to reconstruct the original image from the latent code

**The ELBO loss:**
- Reconstruction loss: `E[||x - D(z)||²]` — how well does the decoder reconstruct?
- KL regularization: `D_KL(q(z|x) || N(0,I))` — keep the latent distribution close to a standard Gaussian
- Total: `L_VAE = L_recon + β · L_KL`
- The β weight controls the tradeoff: higher β → smoother latent space, lower reconstruction quality

**Reparameterization trick (again!):**
- z = μ + σ · ε, where ε ~ N(0, I)
- Same trick as in diffusion (Lecture 2) — separate randomness from the computation graph

**Normalization note:** VAE encoders/decoders typically use BatchNorm2d — one of the few places in the diffusion pipeline where BatchNorm appears. The diffusion U-Net uses GroupNorm (consistent behavior at any batch size during sampling). Don't confuse the two: the VAE is trained separately and always runs in eval mode during diffusion training/sampling, so BatchNorm's batch-size dependence isn't an issue.

**The latent space:**
- A compressed, semantically meaningful representation of images
- Nearby points in latent space → similar images
- Interpolation between two latent codes → smooth visual transition

### Section 10.3: Latent Diffusion — The Key Insight
- **The three-step recipe:**
  1. Train a VAE on images — learn encoder E and decoder D. Then freeze it.
  2. Encode all training images: z_0 = E(x) — now you have a dataset of latent codes
  3. Train a diffusion model on z_0 instead of x — much smaller U-Net
- **At inference:**
  1. Sample z_T ~ N(0, I) in latent space
  2. Denoise: z_T → z_{T-1} → ... → z_0 using the diffusion model
  3. Decode: x = D(z_0) — back to pixel space
- **Why this works:**
  - The VAE handles perceptual compression — it strips away imperceptible detail
  - The diffusion model only needs to model the semantically important structure
  - Latent space is much smaller → smaller U-Net, cheaper attention, faster training
  - The two stages specialize: VAE handles perception, diffusion handles generation

**Pseudocode:**
```
# Stage 1: Train VAE (done once)
vae = train_vae(image_dataset)

# Stage 2: Encode training data
latent_dataset = [vae.encode(x) for x in image_dataset]

# Stage 3: Train diffusion on latents
diffusion_model = train_diffusion(latent_dataset)  # same as Lecture 6, but on latents

# Inference
z_T = randn(batch_size, latent_channels, latent_h, latent_w)
z_0 = ddim_sample(diffusion_model, z_T, schedule, num_steps=50)
images = vae.decode(z_0)  # back to pixel space
```

### Section 10.4: The Stable Diffusion Architecture
**Three main components:**

1. **VAE (KL-regularized autoencoder)**
   - Encoder: 512×512×3 → 64×64×4 (8× spatial compression)
   - Decoder: 64×64×4 → 512×512×3
   - Trained separately on large image datasets

2. **U-Net (operates in 64×64 latent space)**
   - Much more affordable than pixel-space at 512×512
   - Has cross-attention layers for text conditioning
   - Time-conditioned as in Lectures 4-5

3. **Text Encoder (CLIP or T5)**
   - Converts text prompt → embedding sequence
   - SD 1.5: CLIP ViT-L/14 (77 tokens, 768-dim)
   - SDXL: CLIP + OpenCLIP (dual text encoders)
   - SD 3: T5-XXL (more powerful text understanding)

**The full inference pipeline:**
1. Text prompt → text encoder → text embeddings (seq_len, d_text)
2. Sample z_T ~ N(0, I) with shape (1, 4, 64, 64)
3. Iterative denoising with CFG:
   - For each step: model predicts noise, conditioned on text (cross-attention)
   - CFG: combine conditional + unconditional predictions
4. Decode: x = VAE.decode(z_0) → (1, 3, 512, 512)
5. Denormalize and display

### Section 10.5: SD Evolution
- **SD 1.5:** CLIP text encoder, 64×64 latent, ~860M U-Net params
- **SD 2.x:** OpenCLIP text encoder, v-prediction, larger model
- **SDXL:** dual text encoders, 128×128 base latent, refiner model
- **SD 3:** DiT architecture (Lecture 12), T5 text encoder, flow matching (Lecture 13)
- The trend: better text encoders, larger models, more advanced architectures

## Key Equations
- VAE ELBO: `L = E[||x - D(z)||²] + β · D_KL(q(z|x) || N(0,I))`
- VAE reparameterization: `z = μ + σ · ε`
- Latent diffusion training: same as pixel-space but on z_0 = E(x)
- Spatial compression: 512×512×3 → 64×64×4 (48× fewer values)

## Code Examples
- VAE encode/decode pseudocode
- Latent diffusion training loop — highlight what changes from pixel-space (almost nothing)
- Full inference pipeline pseudocode: text → encode → denoise → decode → display
- No full implementation (too complex) — conceptual understanding + pseudocode

## Diagrams & Visuals
- **Resolution bottleneck chart:** bar chart of pixel count and attention cost at different resolutions
- **VAE compression diagram:** image → encoder → latent (small) → decoder → reconstructed image
- **Latent diffusion pipeline:** full diagram showing VAE encoder, diffusion in latent space, VAE decoder
- **Stable Diffusion architecture:** three-component diagram (text encoder, U-Net, VAE) with data flow arrows
- **SD evolution timeline:** visual showing progression from SD 1.5 to SD 3

## Source Material
- Module 09 §9.1 (The Resolution Problem)
- Module 09 §9.2 (VAE Primer)
- Module 09 §9.3 (Latent Diffusion)
- Module 09 §9.4 (Stable Diffusion Architecture Overview)
