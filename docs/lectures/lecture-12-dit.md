# Lecture 12: Diffusion Transformers (DiT)
**Module:** 6 — Modern Architectures
**Estimated reading time:** 20 minutes
**Dependencies:** Lectures 5, 10

## Learning Objectives
- Explain why the field is moving from U-Net to transformer-based denoisers
- Describe the DiT architecture: patchify → transformer blocks → unpatchify
- Explain adaLN-Zero conditioning: adaptive layer norm with zero-initialized gates
- Compare U-Net vs DiT on architecture complexity, scaling properties, and flexibility
- Understand DiT's role in modern systems (Sora, SD 3)

## Narrative Arc
**The problem:** The U-Net has been the workhorse of diffusion models since DDPM (2020). It works well, but it's architecturally complex — encoder, decoder, skip connections, attention at specific resolutions, careful channel progression. As models scale to billions of parameters, we want something simpler with better scaling properties.

**The attempt:** Vision Transformers (ViT) showed that transformers can match or beat CNNs on image classification by treating image patches as tokens. Could the same idea work for diffusion? The challenge: diffusion needs conditioning on timestep and class/text, and the output must be a full image, not a class label.

**The solution:** DiT (Peebles & Xie, 2023) replaces the U-Net entirely. Patchify the image (or latent) into a flat sequence, process with standard transformer blocks, and unpatchify back. Conditioning uses adaLN-Zero — adaptive layer norm where scale, shift, and gate parameters are predicted from the timestep+class embedding, with gates initialized to zero for stable training. DiT scales better than U-Net, is architecturally simpler, and achieves state-of-the-art FID on ImageNet. It's the architecture behind Sora and Stable Diffusion 3.

## Section Outline

### Section 12.1: The Trend — Transformers for Everything
- Transformers have dominated NLP (2017), vision classification (ViT, 2020), and now diffusion
- Why: transformers scale more predictably with compute (transformer scaling laws)
- U-Net limitations at scale:
  - Complex architecture with many bespoke components (skip connections, multi-resolution attention)
  - Scaling requires careful architectural tuning (channel multipliers, attention resolutions)
  - Hard to benefit from the transformer scaling recipe (just add layers and data)

### Section 12.2: DiT Architecture

**Patchify: image → sequence**
- Split the image (or latent) into non-overlapping patches
  - Patch size p: typically 2, 4, or 8
  - Example: 32×32 image with p=4 → 8×8 = 64 patches
  - Each patch is flattened and linearly projected to the model dimension
  - Shape: (B, 3, 32, 32) → (B, 64, d_model) — 64 tokens of dimension d_model
  - **MNIST note:** 28×28 is not evenly divisible by common patch sizes. Pad to 32×32 first, or use p=2 (→ 14×14 = 196 patches) or p=7 (→ 4×4 = 16 patches)

**Positional encoding:**
- Add 2D positional encoding (learnable or sinusoidal) to distinguish patch positions
- Each patch needs to know where in the image it came from

**Transformer blocks:**
- Standard transformer structure: LayerNorm → Self-Attention → Residual → LayerNorm → FFN → Residual
- But with adaLN-Zero conditioning (Section 12.3) instead of standard LayerNorm

**Unpatchify: sequence → image**
- Linear projection from d_model to patch_size² × channels
- Reshape flat sequence back to spatial grid
- Shape: (B, 64, d_model) → (B, 3, 32, 32)

**Pseudocode:**
```
class DiT:
    def forward(x, t, class_label=None):
        # Condition embedding
        c = time_mlp(sinusoidal(t))
        if class_label is not None:
            c = c + class_emb(class_label)

        # Patchify
        tokens = patchify(x)                    # (B, N, d_model)
        tokens = tokens + pos_encoding          # add position info

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)           # adaLN-Zero conditioning

        # Unpatchify
        noise_pred = unpatchify(tokens)         # (B, C, H, W)
        return noise_pred
```

### Section 12.3: Conditioning via adaLN-Zero
- Standard LayerNorm: `y = (x - μ) / σ · γ + β` with learned γ, β
- **Adaptive LayerNorm (adaLN):** predict γ and β from the conditioning signal c instead of learning them
  - `γ, β = Linear(c)` — condition-dependent normalization
  - Same idea as AdaGN from Lecture 5, but for transformer blocks
- **adaLN-Zero:** additionally predict a gate parameter α, initialized to zero
  - `γ, β, α = Linear(c)`
  - After attention/FFN: `output = x + α · sublayer(adaLN(x, γ, β))`
  - α starts at 0 → at initialization, each transformer block is an identity function
  - This means the model starts as "do nothing" and gradually learns to process — very stable training

**Pseudocode:**
```
class DiTBlock:
    def forward(x, c):
        # Predict adaLN-Zero parameters from conditioning
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln_mlp(c).chunk(6, dim=-1)

        # Self-attention with adaLN
        h = layer_norm(x) * (1 + gamma1) + beta1
        h = self_attention(h)
        x = x + alpha1 * h                      # gated residual

        # FFN with adaLN
        h = layer_norm(x) * (1 + gamma2) + beta2
        h = ffn(h)
        x = x + alpha2 * h                      # gated residual

        return x
```

### Section 12.4: Why DiT Scales Better
- **Simpler architecture:** no encoder-decoder, no skip connections, no multi-resolution design
  - Just stack more transformer blocks to make it bigger
- **Transformer scaling laws apply:** DiT-XL/2 with more compute → lower FID, predictably
- **DiT scaling results (Peebles & Xie, ImageNet 256×256 — not MNIST):**
  - DiT-S/2 (33M params): FID 68.4
  - DiT-B/2 (130M params): FID 43.5
  - DiT-L/2 (458M params): FID 23.3
  - DiT-XL/2 (675M params): FID 9.62 → with CFG: FID 2.27 (state-of-the-art on ImageNet)
- **Flexible conditioning:** adaLN-Zero handles any conditioning; cross-attention for sequences
- **More parallelizable:** transformers are better optimized on modern hardware (GPUs, TPUs)

### Section 12.5: U-Net vs DiT — When to Use Each
| Property | U-Net | DiT |
|----------|-------|-----|
| Architecture complexity | High (many components) | Low (repeated blocks) |
| Scaling behavior | Diminishing returns | Follows scaling laws |
| Skip connections | Yes (critical for fine detail) | No (relies on deep attention) |
| Multi-resolution processing | Built-in (encoder-decoder) | Flat sequence (patch-based) |
| Training stability | Good (proven) | Good (adaLN-Zero helps) |
| Best for | Small/medium models, established recipes | Large-scale, frontier models |
| Used in | DDPM, SD 1.5, SDXL | Sora, SD 3, Flux |

### Section 12.6: DiT in Practice
- **SORA (OpenAI):** DiT for video generation — treats video as spatiotemporal patches
- **Stable Diffusion 3:** replaces U-Net with a "MM-DiT" (multimodal DiT) with joint text-image attention
- **Flux:** DiT-based architecture with improved efficiency
- The trend is clear: new frontier models are DiT-based, not U-Net-based

## Key Equations
- Patchify: `tokens = Linear(flatten(patches))` with shape (B, H/p × W/p, d_model)
- adaLN: `y = (1 + γ) · LayerNorm(x) + β` where `[γ, β, α] = Linear(c)`
- Gated residual: `x = x + α · sublayer(adaLN(x, γ, β))`
- Unpatchify: inverse of patchify — reshape tokens back to spatial grid

## Code Examples
- `DiTBlock` with adaLN-Zero — the core building block (pseudocode first, PyTorch reference)
- Patchify and unpatchify operations
- Minimal DiT forward pass: patchify → transformer blocks → unpatchify
- Parameter count comparison: U-Net vs DiT at equivalent capacity

## Diagrams & Visuals
- **DiT architecture diagram:** input image → patchify → transformer blocks → unpatchify → noise prediction
- **adaLN-Zero block diagram:** show conditioning signal flowing into normalization and gating
- **U-Net vs DiT side-by-side:** architectural comparison highlighting structural differences
- **DiT scaling plot:** FID vs model size/compute showing predictable scaling
- **Patch visualization:** show how an image is split into patches and each becomes a token

## Source Material
- Module 09 §9.7 (DiT: Diffusion Transformer)
- Module 04 concepts (for U-Net comparison)
