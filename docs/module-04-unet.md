# Module 4: The U-Net Architecture

## Purpose
Build the denoising network used in diffusion models. The U-Net is THE architecture — understand every piece: encoder, decoder, skip connections, time conditioning, attention placement.

## 📄 Key Papers
- **U-Net: Convolutional Networks for Biomedical Image Segmentation** — Ronneberger et al. 2015. [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
  - *Read for:* The original U-Net for segmentation. The encoder-decoder + skip connection architecture. Figure 1 is iconic.
- **Denoising Diffusion Probabilistic Models** — Ho et al. 2020. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
  - *Read for:* Section 4 and Appendix B — the specific U-Net architecture used for diffusion. How timestep conditioning is injected.
- **Diffusion Models Beat GANs on Image Synthesis** — Dhariwal & Nichol 2021. [arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
  - *Read for:* Architecture improvements to the diffusion U-Net. Adaptive Group Normalization (AdaGN), attention at multiple resolutions, BigGAN-style residual blocks.

## Sections

### 4.1 — U-Net History: From Segmentation to Generation

**Concepts to teach:**
- Original purpose: biomedical image segmentation (2015)
- Key insight: combine high-level semantics (deep layers) with fine spatial detail (shallow layers)
- Why it was adopted for diffusion: the model needs to output a full image (or noise map) at the same resolution as input — encoder-decoder is natural
- The adaptation: add time conditioning, add attention, use GroupNorm instead of BatchNorm

**Content:** Mostly conceptual/historical. Include the original U-Net figure and the diffusion-adapted version side by side (describe for diagram generation).

---

### 4.2 — Encoder Path: Progressive Downsampling

**Concepts to teach:**
- The encoder reduces spatial resolution while increasing channel depth
- Typical progression: 64 → 128 → 256 → 512 channels, with 2× downsampling at each level
- Downsampling methods: stride-2 convolution (most common) or average pooling + conv
- Each level: 2 ResBlocks (optionally with attention at certain resolutions)
- Feature hierarchy: low-level features (edges, textures) → high-level features (objects, structure)

**Worked example:**
- Implement a `DownBlock(nn.Module)`: 2 ResBlocks + optional attention + downsample
- Print feature map shapes at each level

**Exercise:**
- Build the full encoder: chain 4 DownBlocks with increasing channels
- Verify the spatial dimensions halve and channels double at each level

---

### 4.3 — Decoder Path: Upsampling + Skip Connections

**Concepts to teach:**
- Mirror of the encoder: increase spatial resolution, decrease channels
- Upsampling methods: `F.interpolate(nearest)` + conv (preferred) or transposed convolution
- At each level: concatenate with the skip connection from the corresponding encoder level
- Channel count after concatenation: encoder_channels + decoder_channels → use conv to reduce
- Each level: 2 ResBlocks (optionally with attention)

**Worked example:**
- Implement an `UpBlock(nn.Module)`: upsample + concat skip + 2 ResBlocks + optional attention
- Show the channel math: how concatenation doubles channels, then conv reduces them

**Exercise:**
- Build the full decoder: chain 4 UpBlocks with decreasing channels
- Verify final output matches input spatial dimensions

---

### 4.4 — Skip Connections: Why They Matter

**Concepts to teach:**
- Without skip connections: the decoder must reconstruct spatial detail from a tiny bottleneck — lossy
- With skip connections: fine-grained spatial info flows directly from encoder to decoder
- Concatenation vs addition: DDPM uses concatenation (more expressive), some architectures use addition
- The gradient highway: skip connections also help gradient flow during training (same principle as residual connections)
- Practical detail: must store encoder feature maps during forward pass — memory cost

**Worked example:**
- Train a tiny U-Net with and without skip connections on a simple denoising task
- Visualize the quality difference in reconstructed images

---

### 4.5 — Time Conditioning: Sinusoidal Timestep Embeddings

**Concepts to teach:**
- The model needs to know what timestep it's denoising at (t=1 is very different from t=999)
- Sinusoidal embedding: same formula as positional encoding in transformers
  - `emb[2i] = sin(t / 10000^(2i/d))`, `emb[2i+1] = cos(t / 10000^(2i/d))`
- MLP projection: sinusoidal embedding → Linear → SiLU → Linear → time embedding vector
- The time embedding is a single vector per sample in the batch, not spatial

**Injection methods:**
- **Addition:** project time_emb to match channels, add to feature map (broadcast over spatial dims)
- **Scale-shift (AdaGN):** predict scale γ and shift β from time_emb, apply after GroupNorm: `y = γ * GroupNorm(x) + β`
  - This is more expressive — used in improved diffusion architectures

**Worked example:**
- Implement `SinusoidalTimestepEmbedding(nn.Module)`
- Implement the MLP that projects the embedding
- Show both injection methods (add vs scale-shift)

**Exercise:**
- Implement full timestep embedding pipeline: integer t → sinusoidal → MLP → injection into ResBlock

---

### 4.6 — Where Attention Goes in the U-Net

**Concepts to teach:**
- Attention is expensive: O(n²) where n = H × W
- Typically placed at lower resolutions: 16×16 and 8×8 (not at 64×64 or 128×128)
- DDPM: attention at 16×16 resolution only
- Improved DDPM / ADM: attention at 32×32, 16×16, 8×8
- The attention here is spatial self-attention (Module 3.6)
- Each attention-enabled ResBlock: ResBlock → Attention → output
- Optional cross-attention for conditioning (text, class)

**Worked example:**
- Show the compute cost difference: attention at 64×64 vs 16×16
- Demonstrate that attention at lower resolutions still captures global structure

---

### 4.7 — ResBlock with Time Embedding: Full Implementation

**Concepts to teach:**
- The core building block of the diffusion U-Net
- Structure: `Conv → GroupNorm → SiLU → time_proj → Conv → GroupNorm → SiLU → residual`
- Where time gets injected: after first conv+norm+activation, before second conv
- Residual connection: project input with 1×1 conv if channel dimensions change
- Dropout (optional, often 0.1 or 0.0)

**Worked example:**
- Full `ResBlock(nn.Module)` implementation with time conditioning
- Test with: matching channels (identity shortcut) and mismatched channels (1×1 projection)
- Print parameter count

**Exercise:**
- Implement the ResBlock, including both add and scale-shift time injection modes
- Verify forward/backward pass works, check for shape mismatches

---

### 4.8 — Assembling the Full U-Net

**Concepts to teach:**
- Putting it all together: the complete architecture
- **Structure:**
  1. Initial conv: image channels → base_channels
  2. Time embedding: MLP to produce time vector
  3. Down blocks: [ResBlock, ResBlock, (optional Attention), Downsample] × N_levels
  4. Middle block: ResBlock → Attention → ResBlock
  5. Up blocks: [ResBlock, ResBlock, (optional Attention), Upsample] × N_levels
  6. Final norm + activation + conv: base_channels → image channels
- Skip connections: stored in a list during encoder, popped during decoder
- The model predicts noise (ε) — output has same shape as input image

**Worked example:**
- Full `UNet(nn.Module)` implementation — every piece wired together
- Forward pass walkthrough: trace the tensor shapes at every stage
- Configuration: `image_channels=3, base_channels=64, channel_mults=(1,2,4,8), attention_resolutions=[16,8]`

**Exercise:**
- Assemble the complete U-Net
- Verify: `model(noisy_image, timestep)` returns a tensor with the same shape as `noisy_image`
- Count parameters, compare to published model sizes

---

### 4.9 — Parameter Count Analysis

**Concepts to teach:**
- Where the parameters live: most are in the middle and lower-resolution blocks (more channels)
- Attention layers can be a significant fraction of total parameters
- Practical model sizes: DDPM used ~36M params for 32×32 CIFAR, ~114M for 256×256 faces
- Memory considerations: activations for skip connections dominate memory during training
- Gradient checkpointing: trade compute for memory

**Worked example:**
- Break down parameter count by module: encoder, middle, decoder, time embedding
- Profile peak memory during forward pass

**Exercise:**
- Given a target parameter budget (e.g., 20M), design a U-Net configuration that fits

---

## Module 4 Capstone Exercise

**Build a complete toy U-Net and verify every tensor shape:**
- Input: (B, 3, 32, 32) for CIFAR-10 sized images
- Timestep: (B,) integer tensor
- Output: (B, 3, 32, 32) predicted noise
- Channel progression: 64 → 128 → 256 → 512
- Attention at 8×8 resolution
- Print shapes at every stage of forward pass
- This U-Net will be used for training in Module 6
