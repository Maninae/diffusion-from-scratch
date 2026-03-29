# Lecture 4: U-Net — Encoder, Decoder, Skip Connections
**Module:** 2 — The Denoising Network
**Estimated reading time:** 25 minutes
**Dependencies:** Lecture 3 (need to know what the network predicts)

## Learning Objectives
- Trace the U-Net's evolution from medical segmentation (2015) to diffusion denoiser (2020)
- Describe the encoder path: progressive downsampling with increasing channel depth
- Describe the decoder path: upsampling with skip connections from encoder
- Explain why skip connections are critical for denoising quality (with ablation evidence)
- Track tensor shapes through every stage of the encoder-decoder pipeline

## Narrative Arc
**The problem:** Lecture 3 gave us the training objective — predict the noise ε added to an image. But what neural network architecture should we use? The network takes a noisy image (B, 1, 28, 28) and must output a noise prediction of the same shape. A plain CNN could do this, but it would lose fine spatial detail through successive convolutions.

**The attempt:** An encoder-only CNN compresses spatial information into a bottleneck — good for classification but destructive for pixel-level prediction. We need both high-level understanding (what objects are present) AND fine spatial detail (exact edge positions, textures). How do you get both?

**The solution:** U-Net. An encoder progressively compresses (capturing high-level features), a decoder progressively expands (recovering spatial resolution), and skip connections carry fine-grained detail directly from encoder to decoder at each resolution level. The result: a network that reasons at multiple scales simultaneously.

## Section Outline

### Section 4.1: U-Net History — From Segmentation to Generation
- Original U-Net (Ronneberger et al., 2015): designed for biomedical image segmentation
- The key insight: combine high-level semantics from deep layers with fine spatial detail from shallow layers
- Why adopted for diffusion: the model must output a full image (noise prediction) at the same resolution as input — encoder-decoder is a natural fit
- What changed for diffusion: GroupNorm instead of BatchNorm, SiLU activation, time conditioning, attention layers

### Section 4.2: The Encoder Path — Progressive Downsampling
- The encoder reduces spatial resolution while increasing channel depth
- Typical progression for MNIST: 1 → 64 → 128 → 256 channels, spatial: 28 → 14 → 7
- Each encoder level contains:
  - 2 ResBlocks (Conv → GroupNorm → SiLU → Conv + residual connection) — channels stay constant within a level
  - Optional attention at lower resolutions
  - Downsample: stride-2 convolution (halves spatial dimensions) — channel increase happens at the transition between levels (the first conv of the next level's ResBlock projects to the new channel count)
- Feature hierarchy emerges:
  - Early layers: edges, textures, local patterns
  - Deep layers: objects, global structure, semantic content
- Tensor shape tracking through encoder:
  ```
  Input:          (B, 1, 28, 28)
  Initial conv:   (B, 64, 28, 28)
  Level 1:        (B, 64, 28, 28)  → downsample → (B, 64, 14, 14)
  Level 2:        (B, 128, 14, 14) → downsample → (B, 128, 7, 7)
  Level 3:        (B, 256, 7, 7)   [bottleneck — no downsample]
  ```

**Pseudocode:**
```
class DownBlock:
    def forward(x, time_emb):
        h = resblock_1(x, time_emb)    # same channels
        h = resblock_2(h, time_emb)    # same channels
        skip = h                        # save for decoder
        h = downsample(h)               # halve spatial dims
        return h, skip
```

### Section 4.3: The Decoder Path — Upsampling + Skip Concatenation
- Mirror of the encoder: increase spatial resolution, decrease channels
- Upsampling method: `F.interpolate(nearest)` + conv (preferred over transposed convolution — avoids checkerboard artifacts)
- At each level: concatenate with the corresponding encoder skip connection
- Channel math after concatenation:
  - Decoder has C_dec channels, skip has C_enc channels
  - After concat: C_dec + C_enc channels
  - Conv reduces back to target channel count
- Tensor shape tracking through decoder:
  ```
  Bottleneck:     (B, 256, 7, 7)
  Upsample:       (B, 256, 14, 14)
  + Skip concat:  (B, 256+128, 14, 14) = (B, 384, 14, 14)
  ResBlocks:      (B, 128, 14, 14)
  Upsample:       (B, 128, 28, 28)
  + Skip concat:  (B, 128+64, 28, 28) = (B, 192, 28, 28)
  ResBlocks:      (B, 64, 28, 28)
  Final conv:     (B, 1, 28, 28)       — same shape as input!
  ```

**Pseudocode:**
```
class UpBlock:
    def forward(x, skip, time_emb):
        x = upsample(x)                # double spatial dims
        x = concat(x, skip, dim=1)     # concat along channel dim
        x = resblock_1(x, time_emb)    # reduce channels
        x = resblock_2(x, time_emb)
        return x
```

### Section 4.4: Skip Connections — Why They're Critical
- **Without skips:** the decoder must reconstruct all spatial detail from a tiny bottleneck (7×7). Fine details (edges, textures) are lost. The network produces blurry, smooth noise predictions.
- **With skips:** fine-grained spatial information flows directly from encoder to decoder at every resolution. The decoder combines high-level reasoning (from the bottleneck) with low-level detail (from skips).
- **Concatenation vs addition:**
  - DDPM uses concatenation (more expressive — the network can learn how to combine)
  - Some architectures use addition (simpler, fewer parameters)
- **The gradient highway:** skip connections also help gradient flow during training — same principle as residual connections in ResNets
- **Memory cost:** encoder feature maps must be stored during the forward pass for the decoder to use — this is the main memory bottleneck during training

**Ablation evidence:** (content gap — implement for website)
- Train a small U-Net with and without skip connections on a simple denoising task
- Without skips: denoising quality is significantly worse — blurry, missing detail
- With skips: sharp, accurate noise predictions

### Section 4.5: The Middle Block (Bottleneck)
- Between encoder and decoder: ResBlock → Attention → ResBlock
- This is where global reasoning happens — at the lowest spatial resolution
- Attention here is cheap (small spatial dimension) but powerful (global receptive field)

### Section 4.6: The Full Architecture — Assembly
- Putting it all together:
  1. Initial conv: image_channels → base_channels
  2. Encoder: [DownBlock × N_levels] — store skip connections
  3. Middle: ResBlock → Attention → ResBlock
  4. Decoder: [UpBlock × N_levels] — consume skip connections in LIFO order (last encoder skip used first, because the decoder mirrors the encoder — the deepest encoder level connects to the first decoder level)
  5. Final: GroupNorm → SiLU → Conv → image_channels
- Output has same shape as input — it's the predicted noise ε_θ

**Pseudocode:**
```
class UNet:
    def forward(x, t):
        # Time embedding (detailed in Lecture 5)
        t_emb = time_mlp(sinusoidal_embedding(t))

        # Initial projection
        h = init_conv(x)                    # (B, 1, 28, 28) → (B, 64, 28, 28)

        # Encoder (save skips)
        skips = []
        for down_block in encoder:
            h, skip = down_block(h, t_emb)
            skips.append(skip)

        # Middle
        h = mid_block(h, t_emb)

        # Decoder (use skips in reverse)
        for up_block in decoder:
            skip = skips.pop()
            h = up_block(h, skip, t_emb)

        # Output
        return final_conv(h)                # (B, 64, 28, 28) → (B, 1, 28, 28)
```

## Key Equations
- ResBlock: `output = Conv(SiLU(GroupNorm(Conv(SiLU(GroupNorm(x)))))) + shortcut(x)`
- Downsampling: stride-2 Conv2d — `(B, C, H, W) → (B, C_out, H/2, W/2)`
- Upsampling: `F.interpolate(scale_factor=2) + Conv2d` — `(B, C, H, W) → (B, C_out, 2H, 2W)`
- Skip concatenation: `(B, C_dec, H, W) cat (B, C_enc, H, W) → (B, C_dec+C_enc, H, W)`

## Code Examples
- `DownBlock` pseudocode: 2 ResBlocks + downsample, return skip
- `UpBlock` pseudocode: upsample + skip concat + 2 ResBlocks
- Full `UNet.forward()` pseudocode with shape annotations
- PyTorch reference implementation at end (imports from `utils/unet.py`)

## Diagrams & Visuals
- **U-Net architecture diagram:** the classic U shape showing encoder (left), bottleneck (bottom), decoder (right), skip connections (horizontal arrows), with tensor shapes at every stage
- **Original U-Net vs diffusion U-Net:** side-by-side showing what was added (time conditioning, GroupNorm, attention)
- **Skip connection ablation:** denoising results with and without skip connections
- **Channel progression diagram:** visual showing how channels grow (encoder) and shrink (decoder)

## Source Material
- Module 04 §4.1 (U-Net History)
- Module 04 §4.2 (Encoder Path)
- Module 04 §4.3 (Decoder Path)
- Module 04 §4.4 (Skip Connections)
- Module 04 §4.8 (Assembling the Full U-Net)
