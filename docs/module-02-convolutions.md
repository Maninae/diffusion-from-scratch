# Module 2: Convolutions & Image Processing

## Purpose
Understand the building blocks of image-based neural networks. The U-Net in diffusion is built from conv layers, normalization, and residual connections — you need to know how each works internally.

## 📄 Key Papers
- **Deep Residual Learning for Image Recognition** — He et al. 2015. [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
  - *Read for:* Residual connections and why they enable training very deep networks. Section 3 for the key insight about identity mappings.
- **Group Normalization** — Wu & He 2018. [arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
  - *Read for:* Why GroupNorm works better than BatchNorm for small batches (diffusion models often use small batches). Table 1 for the comparison.

## Sections

### 2.1 — 1D and 2D Convolution from Scratch

**Concepts to teach:**
- What convolution does: sliding a kernel over input, computing dot products
- 1D convolution first (simpler to visualize), then extend to 2D
- The relationship between convolution and correlation (flip the kernel)
- Multi-channel convolution: input channels × output channels × kernel_h × kernel_w
- Why convolutions work for images: translation equivariance, parameter sharing, local receptive fields

**Worked example:**
- Implement 1D convolution with nested loops, then vectorized
- Implement 2D convolution with `np.lib.stride_tricks` or unfolding
- Compare output to `F.conv2d`

**Exercise:**
- Implement multi-channel 2D convolution from scratch
- Apply edge detection kernels (Sobel, Laplacian) to an image using your implementation

---

### 2.2 — Padding, Stride, Dilation — The Output Size Formula

**Concepts to teach:**
- The output size formula: `floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)`
- Padding modes: zero, reflect, replicate — when to use which
- Stride > 1: downsampling via convolution (alternative to pooling)
- Dilation: expanding the receptive field without more parameters (atrous convolution)
- "Same" padding: choose padding to keep spatial dimensions unchanged

**Worked example:**
- Calculate output sizes for various configs
- Demonstrate how stride=2 halves spatial dimensions (used in U-Net encoder)

**Exercise:**
- Given input size and desired output size, compute the required padding/stride
- Implement a function that automatically computes "same" padding

---

### 2.3 — Transposed Convolutions (Upsampling)

**Concepts to teach:**
- Why we need upsampling: the decoder path of U-Net needs to increase spatial resolution
- Transposed convolution: NOT the inverse of convolution — it's the gradient of convolution
- How it works: insert zeros between input elements, then convolve
- The checkerboard artifact problem and why it happens
- Alternatives: `F.interpolate` (nearest/bilinear) + regular conv — often preferred in practice
- Output size formula for transposed conv

**Worked example:**
- Implement transposed convolution, show the zero-insertion mechanism
- Demonstrate checkerboard artifacts
- Compare with interpolate + conv approach

**Exercise:**
- Build an upsampling block using both approaches, compare output quality

---

### 2.4 — Depthwise Separable Convolutions

**Concepts to teach:**
- Standard conv: O(C_in × C_out × K² × H × W) operations
- Depthwise conv: convolve each channel independently
- Pointwise conv: 1×1 conv to mix channels
- Separable = depthwise + pointwise: same expressive power, far fewer parameters
- Where they show up: efficient architectures, some diffusion U-Net variants

**Worked example:**
- Implement depthwise separable conv, count parameters vs standard conv
- Show they produce similar results with ~9x fewer parameters (for 3×3 kernels)

**Exercise:**
- Replace standard convs in a small network with separable convs, compare accuracy and speed

---

### 2.5 — Batch Normalization

**Concepts to teach:**
- The problem: internal covariate shift (debated, but the original motivation)
- What it does: normalize activations to zero mean, unit variance per channel across the batch
- Learnable affine parameters: γ (scale) and β (shift)
- Training vs eval: batch statistics vs running statistics
- Running mean/var: exponential moving average updated during training
- Why it helps: smoother loss landscape, enables higher learning rates

**Worked example:**
- Implement BatchNorm from scratch for 2D inputs (B, C, H, W)
- Track running statistics, switch between train and eval modes
- Verify against `nn.BatchNorm2d`

**Exercise:**
- Train a ConvNet with and without BatchNorm, compare convergence speed

---

### 2.6 — Group Normalization (Why Diffusion Models Prefer It)

**Concepts to teach:**
- BatchNorm's weakness: depends on batch size, breaks with batch_size=1
- GroupNorm: divide channels into groups, normalize within each group
- No dependence on batch size — works the same whether batch is 1 or 1024
- LayerNorm and InstanceNorm as special cases of GroupNorm
- **Why diffusion models use GroupNorm:** training often uses small batches (GPU memory), and inference is often batch_size=1
- Typical group count: 32 groups, or group_size=32 channels per group

**Worked example:**
- Implement GroupNorm from scratch
- Show it produces the same output regardless of batch size (unlike BatchNorm)
- Verify against `nn.GroupNorm`

**Exercise:**
- Implement all four: BatchNorm, LayerNorm, InstanceNorm, GroupNorm — show they're all the same operation with different grouping

---

### 2.7 — Residual Connections: Gradient Flow Intuition

**Concepts to teach:**
- The degradation problem: deeper ≠ better (without residual connections)
- The residual connection: `y = F(x) + x` — learn the residual, not the mapping
- Why it helps: gradient flows through the identity shortcut, avoids vanishing gradients
- Identity mapping: the network can always learn F(x) = 0 (i.e., do nothing)
- When dimensions don't match: 1×1 conv projection shortcut
- Pre-activation vs post-activation residual blocks

**Worked example:**
- Implement a ResBlock: Conv → GroupNorm → SiLU → Conv → GroupNorm + shortcut
- Show gradient magnitudes with and without residual connections in a deep network

**Exercise:**
- Build a ResBlock with optional dimension change (1×1 projection)
- This is the exact block used in diffusion U-Nets — emphasize this connection

---

## Module 2 Capstone Exercise

**Build a small ConvNet classifier from custom layers, train on CIFAR-10:**
- Custom Conv2d layers (or use `nn.Conv2d` but with manual forward logic)
- GroupNorm (from scratch or `nn.GroupNorm`)
- Residual connections
- SiLU activation (not ReLU — diffusion models use SiLU)
- Downsampling via stride-2 conv
- Train to >70% accuracy on CIFAR-10
- This architecture is a simplified version of the U-Net encoder
