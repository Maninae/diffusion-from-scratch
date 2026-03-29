# Lecture 5: Time Conditioning & Spatial Attention
**Module:** 2 — The Denoising Network
**Estimated reading time:** 25 minutes
**Dependencies:** Lecture 4

## Learning Objectives
- Implement sinusoidal timestep embeddings and explain their connection to transformer positional encoding
- Compare two time injection methods: additive and scale-shift (AdaGN)
- Explain where attention goes in the U-Net and why placement is resolution-dependent
- Describe spatial self-attention: reshape (B,C,H,W) → (B,H×W,C), attend, reshape back
- Assemble the complete U-Net with time conditioning and attention, ready for training

## Narrative Arc
**The problem:** The U-Net from Lecture 4 takes a noisy image and outputs a noise prediction — but it has no idea what noise level it's working with. Denoising at t=50 (slight haze) is a completely different task from denoising at t=950 (pure static). A network that can't distinguish these will fail at both.

**The attempt:** We could pass t as a scalar input — concatenate it to the image or add it as a feature. But a single number doesn't give the network enough information to modulate its behavior across hundreds of distinct noise levels. The network needs a rich, high-dimensional representation of time.

**The solution:** Sinusoidal timestep embeddings (borrowed from transformer positional encoding) map each integer t to a high-dimensional vector with unique, smooth structure. An MLP projects this into the right dimension, and it's injected into every ResBlock — the network knows exactly what noise level it's denoising at every layer. Additionally, spatial self-attention at lower resolutions lets the network capture long-range dependencies that convolutions alone miss.

## Section Outline

### Section 5.1: Why the Network Needs to Know t
- Denoising at t=10: image is nearly clean, network should make tiny corrections
- Denoising at t=500: half signal, half noise — network must separate them
- Denoising at t=990: almost pure noise — network must hallucinate plausible structure
- Without time conditioning: the network would produce the same output regardless of noise level — catastrophic
- The solution: inject a representation of t into every ResBlock

### Section 5.2: Sinusoidal Timestep Embeddings
- Same formula as transformer positional encoding, different purpose:
  - `emb[2i] = sin(t / 10000^(2i/d))`
  - `emb[2i+1] = cos(t / 10000^(2i/d))`
- Each dimension oscillates at a different frequency → unique fingerprint per timestep
- Properties: smooth (nearby timesteps have similar embeddings), unbounded (works for any T)
- Output shape: integer t → vector of dimension d (typically 128 or 256)

**Pseudocode:**
```
def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    frequencies = 1 / (10000 ** (range(half_dim) / half_dim))
    angles = t * frequencies             # (B, half_dim)
    embedding = concat(sin(angles), cos(angles), dim=-1)  # (B, dim)
    return embedding
```

### Section 5.3: MLP Projection
- The sinusoidal embedding is projected through a small MLP to match feature dimensions:
  - `sinusoidal(t) → Linear(dim, 4*dim) → SiLU → Linear(4*dim, 4*dim) → time_emb`
- The MLP learns to transform the raw sinusoidal features into the representation most useful for the network
- The time embedding is a single vector per sample: shape (B, emb_dim) — no spatial dimensions

**Pseudocode:**
```
class TimestepMLP:
    def forward(t):
        emb = sinusoidal_embedding(t, base_dim)    # (B, base_dim)
        emb = linear_1(emb)                         # (B, 4 * base_dim)
        emb = silu(emb)
        emb = linear_2(emb)                         # (B, 4 * base_dim)
        return emb
```

### Section 5.4: Time Injection Methods

**Method 1: Addition**
- Project time_emb to match the channel count of the feature map
- Add to features after GroupNorm (broadcast over spatial dims)
- Simple, effective, used in original DDPM

```
# Inside ResBlock
h = conv1(silu(group_norm1(x)))        # (B, C, H, W)
time_proj = linear(time_emb)            # (B, C)
h = h + time_proj[:, :, None, None]     # broadcast over H, W
h = conv2(silu(group_norm2(h)))
```

**Method 2: Scale-Shift (AdaGN)**
- Predict both scale γ and shift β from time_emb
- Apply after GroupNorm: `y = γ · GroupNorm(x) + β`
- More expressive: the network can scale and shift features differently per timestep
- Used in improved diffusion architectures (Dhariwal & Nichol 2021)

```
# Inside ResBlock with AdaGN
h = conv1(silu(group_norm1(x)))
scale, shift = linear(time_emb).chunk(2, dim=1)  # each (B, C)
h = group_norm2(h) * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
h = conv2(silu(h))
```

### Section 5.5: The Full ResBlock with Time Embedding
- Complete data flow:
  1. GroupNorm → SiLU → Conv (first half)
  2. Inject time embedding (add or scale-shift)
  3. GroupNorm → SiLU → Dropout → Conv (second half)
  4. Residual connection (1×1 conv if channels change)
- This is the core building block — the U-Net is built by stacking these

**Pseudocode:**
```
class ResBlock:
    def forward(x, time_emb):
        h = conv1(silu(group_norm1(x)))         # (B, C_out, H, W)
        h = h + time_proj(silu(time_emb))[:,:,None,None]  # inject time
        h = conv2(dropout(silu(group_norm2(h)))) # (B, C_out, H, W)
        return h + shortcut(x)                   # residual connection
```

### Section 5.6: Where Attention Goes in the U-Net
- Attention is O(n²) where n = H × W — the spatial token count
  - At 28×28: n = 784 tokens, attention matrix = 784² ≈ 600K entries — feasible
  - At 14×14: n = 196 tokens — cheap
  - At 7×7: n = 49 tokens — very cheap
  - At 256×256: n = 65,536 — attention matrix = 4 billion entries — infeasible
- DDPM: attention only at 16×16 resolution
- Improved DDPM / ADM: attention at 32×32, 16×16, 8×8
- For our MNIST model: attention at 7×7 (bottleneck) — cheap and effective

### Section 5.7: Spatial Self-Attention
- Images aren't sequences — but we can reshape them:
  1. Apply GroupNorm to the feature map (normalize before attention, not after)
  2. Reshape: (B, C, H, W) → (B, H×W, C) — each spatial position becomes a token
  3. Apply multi-head self-attention on the sequence
  3. Reshape back: (B, H×W, C) → (B, C, H, W)
- This lets the network capture global structure: "this region of the image should be consistent with that distant region"
- Convolutions are local (3×3 receptive field); attention is global (every position attends to every other)

**Pseudocode:**
```
class SpatialSelfAttention:
    def forward(x):
        B, C, H, W = x.shape
        h = group_norm(x)                                # normalize before reshape
        h = h.reshape(B, C, H*W).transpose(1, 2)        # (B, H*W, C)
        attn_out = multi_head_attention(h, h, h)         # (B, H*W, C)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        return x + attn_out  # residual connection
```

### Section 5.8: Cross-Attention Preview
- Self-attention: Q, K, V all come from the image features
- Cross-attention: Q from image features, K and V from an external conditioning signal (text, class)
- This is how text-to-image models (Stable Diffusion) inject text information into the U-Net
- Detailed treatment in Lecture 9 — for now, just know it's the same mechanism with different K/V sources

### Section 5.9: The Complete U-Net — Putting It All Together
- Architecture summary:
  1. Initial conv: image_channels → base_channels
  2. Time embedding: sinusoidal → MLP
  3. Down blocks: [ResBlock(+time), ResBlock(+time), (optional Attention), Downsample] × N
  4. Middle: ResBlock(+time) → Attention → ResBlock(+time)
  5. Up blocks: [Upsample, concat skip, ResBlock(+time), ResBlock(+time), (optional Attention)] × N
  6. Final: GroupNorm → SiLU → Conv → image_channels
- Forward signature: `model(x_t, t) → ε_θ` — optionally `model(x_t, t, class_label)`
- Parameter count: where the parameters live (most in middle/lower-resolution blocks)
- This U-Net is exported to `utils/unet.py` and used in all subsequent lectures

## Key Equations
- Sinusoidal embedding: `emb[2i] = sin(t / 10000^{2i/d})`, `emb[2i+1] = cos(t / 10000^{2i/d})`
- Additive injection: `h = h + Linear(time_emb)[:, :, None, None]`
- AdaGN: `y = (1 + γ) · GroupNorm(x) + β` where `[γ, β] = Linear(time_emb)`
- Self-attention: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

## Code Examples
- `SinusoidalTimestepEmbedding` — sinusoidal → MLP (pseudocode, then PyTorch reference)
- `ResBlock` with time injection — both additive and AdaGN variants
- `SpatialSelfAttention` — reshape, attend, reshape back
- Complete `UNet.forward()` with all pieces connected
- Verification: `model(noisy_image, t).shape == noisy_image.shape`

## Diagrams & Visuals
- **Sinusoidal embedding heatmap:** rows = timesteps, columns = embedding dimensions, color = value
- **Time injection diagram:** show how the (B, emb_dim) time vector gets projected and added to (B, C, H, W) feature maps
- **Attention placement diagram:** U-Net with colored blocks showing which levels have attention
- **Complete U-Net data flow:** full forward pass with tensor shapes annotated at every stage

## Source Material
- Module 04 §4.5 (Time Conditioning: Sinusoidal Embeddings)
- Module 04 §4.6 (Where Attention Goes)
- Module 04 §4.7 (ResBlock with Time Embedding)
- Module 04 §4.8 (Assembling the Full U-Net)
- Module 04 §4.9 (Parameter Count Analysis)
- Module 03 §3.3 (Positional Encodings — sinusoidal connection)
- Module 03 §3.6 (Spatial Self-Attention)
