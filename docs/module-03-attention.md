# Module 3: Attention & Transformers

## Purpose
Understand self-attention and multi-head attention deeply enough to implement them from scratch. Diffusion U-Nets use spatial self-attention at select resolutions — you need to know how QKV projections, scaling, and multi-head mechanics work.

## 📄 Key Papers
- **Attention Is All You Need** — Vaswani et al. 2017. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
  - *Read for:* The original transformer. Section 3.2 for scaled dot-product attention, Section 3.2.2 for multi-head attention. The 1/√d_k scaling explanation.
- **An Image is Worth 16x16 Words** — Dosovitskiy et al. 2020 (ViT). [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
  - *Read for:* How attention applies to images — patch embedding, positional encoding for 2D. Section 3.

## Sections

### 3.1 — Dot-Product Attention from Scratch

**Concepts to teach:**
- Attention as a soft lookup: Query asks a question, Keys are addresses, Values are the content
- Dot-product similarity: `scores = Q @ K^T`
- The scaling factor `1/√d_k`: prevents softmax from saturating when d_k is large
- Softmax: converts scores to attention weights (probabilities)
- Weighted sum: `output = weights @ V`
- The full formula: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

**Worked example:**
- Implement single-head attention from scratch using only matrix multiplies and softmax
- Visualize attention weights as a heatmap
- Show what happens without scaling (softmax saturates, gradients vanish)

**Exercise:**
- Implement `scaled_dot_product_attention(Q, K, V)` with proper scaling
- Add an optional attention mask (for causal or padding masking)
- Verify output matches `F.scaled_dot_product_attention`

---

### 3.2 — Multi-Head Attention

**Concepts to teach:**
- Why multiple heads: each head can attend to different aspects (position, content, etc.)
- Split d_model into h heads, each with d_k = d_model / h
- Parallel attention computations — can be batched efficiently
- Concatenate outputs, project through `W_O`
- The mechanics: `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O`
- Reshaping: (batch, seq, d_model) → (batch, heads, seq, d_k) → back

**Worked example:**
- Implement `MultiHeadAttention(nn.Module)` from scratch
- Show the reshape operations clearly: split heads, compute attention, merge heads
- Compare to `nn.MultiheadAttention` output

**Exercise:**
- Implement multi-head attention using einsum for the score computation
- Profile memory usage: how does it scale with sequence length?

---

### 3.3 — Positional Encodings

**Concepts to teach:**
- Why we need them: attention is permutation-invariant, position information is lost
- Sinusoidal encodings: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`
- Intuition: each dimension oscillates at a different frequency, creating a unique "fingerprint" for each position
- Learned positional embeddings: `nn.Embedding(max_len, d_model)`
- **2D positional encodings for images:** separate row and column encodings, or learned 2D grid
- Connection to timestep embeddings in diffusion: same sinusoidal formula, different purpose (encode time, not position)

**Worked example:**
- Implement sinusoidal positional encoding, visualize the patterns
- Show how it generalizes to unseen positions (vs learned embeddings)

**Exercise:**
- Implement both sinusoidal and learned positional encodings
- Implement 2D positional encoding for image patches
- Implement sinusoidal timestep embedding (this is used directly in diffusion!)

---

### 3.4 — Layer Norm vs Batch Norm in Attention Contexts

**Concepts to teach:**
- LayerNorm: normalize across features (not across batch)
- Why attention uses LayerNorm over BatchNorm: no batch dependence, works with variable-length sequences
- Pre-norm vs post-norm: modern transformers prefer pre-norm (normalize before attention/FFN)
- Implementation: `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`

**Worked example:**
- Implement LayerNorm from scratch
- Show pre-norm transformer block structure: `x + Attention(LayerNorm(x))`

**Exercise:**
- Implement LayerNorm, compare to `nn.LayerNorm`

---

### 3.5 — Cross-Attention: Conditioning on External Info

**Concepts to teach:**
- Self-attention: Q, K, V all come from the same source
- Cross-attention: Q comes from one source, K and V from another
- The mechanism for injecting conditioning information (text, class labels, timesteps)
- In diffusion: Q from the image features, K/V from text encoder output (e.g., CLIP)
- Same attention formula, just different sources for Q vs K/V

**Worked example:**
- Implement cross-attention: image features attend to text embeddings
- Show how it differs from self-attention in code (minimal change)

**Exercise:**
- Build a cross-attention module that conditions image features on class embeddings
- This previews how text-to-image diffusion works

---

### 3.6 — Spatial Self-Attention for Images

**Concepts to teach:**
- Images aren't sequences — but we can reshape them to be: (B, C, H, W) → (B, H*W, C)
- Each spatial position becomes a "token"
- Attention can capture long-range dependencies that convolutions can't
- The cost: O(n²) where n = H × W — that's why diffusion models only use attention at lower resolutions (16×16 or 8×8, not 256×256)
- Reshape back to spatial after attention: (B, H*W, C) → (B, C, H, W)

**Worked example:**
- Take a feature map, reshape to sequence, apply self-attention, reshape back
- Measure compute time for different spatial resolutions — show the quadratic scaling

**Exercise:**
- Implement a `SpatialSelfAttention(nn.Module)` that handles the reshaping internally
- This is the exact module used inside diffusion U-Nets

---

### 3.7 — Efficient Attention and Memory Considerations

**Concepts to teach:**
- Standard attention memory: O(n²) for the attention matrix
- Flash Attention: fused kernel that avoids materializing the full attention matrix — O(n) memory
- `torch.nn.functional.scaled_dot_product_attention` — uses Flash Attention automatically
- When to use attention in a U-Net: only at bottleneck / lower resolutions
- Chunked attention: process in chunks when Flash Attention isn't available

**Worked example:**
- Compare memory usage: naive attention vs `F.scaled_dot_product_attention`
- Show the OOM that happens if you try attention at 256×256 resolution naively

**Exercise:**
- Implement chunked attention that processes the sequence in blocks

---

### 3.8 — Full Transformer Block

**Concepts to teach:**
- The standard transformer block: LayerNorm → MultiHeadAttention → Residual → LayerNorm → FFN → Residual
- FFN: two linear layers with activation in between, typically `d_model → 4*d_model → d_model`
- SiLU/GELU activation (not ReLU) in modern transformers
- Stacking blocks: depth of the transformer

**Worked example:**
- Implement a full `TransformerBlock(nn.Module)` with pre-norm
- Stack 4 blocks, verify gradient flow through the full stack

**Exercise:**
- Build a minimal Vision Transformer: patch embedding → positional encoding → N transformer blocks → classification head
- Train on CIFAR-10 to verify it works

---

## Module 3 Capstone Exercise

**Implement multi-head self-attention from scratch, apply to image patches:**
- Patch an image into 4×4 or 8×8 patches
- Flatten patches into a sequence
- Add 2D positional encoding
- Apply multi-head self-attention
- Verify attention patterns make spatial sense (nearby patches attend to each other)
- This is a warm-up for how attention is used inside the diffusion U-Net
