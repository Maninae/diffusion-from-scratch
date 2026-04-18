# Module 0: Python & NumPy Foundations

## Purpose
Warm up vectorization skills and ensure NumPy fluency before touching PyTorch. Clean, vectorized code is essential — sloppy, loop-heavy code signals inexperience.

## 📄 Key Papers
None for this module — this is foundational tooling.

## Sections

### 0.1 — Array Internals: dtype, strides, contiguous vs non-contiguous

**Concepts to teach:**
- What a NumPy array actually is in memory (contiguous block + metadata)
- `dtype`: why it matters (float32 vs float64 performance, memory)
- `strides`: how NumPy navigates memory without copying
- C-contiguous vs Fortran-contiguous: what `.T` does to strides
- Views vs copies: when does an operation create new memory?
- `.reshape()` vs `.resize()` vs `.view()` (preview for PyTorch)

**Worked example:**
- Create an array, inspect `.strides`, `.flags`, `.dtype`
- Show how transpose changes strides but not data
- Demonstrate when slicing creates a view vs a copy

**Exercise:**
- Given a 2D array, predict whether operations return views or copies
- Optimize a function that's accidentally creating copies in a hot loop

---

### 0.2 — Broadcasting: The Rules, Common Pitfalls, Mental Model

**Concepts to teach:**
- The 3 broadcasting rules (trailing dimensions, size-1 expansion, alignment from right)
- Mental model: "stretch" the smaller array without actually copying
- Common pitfalls: accidentally broadcasting when you wanted an error
- Shape debugging: reading error messages, using `np.broadcast_shapes()`

**Worked example:**
- Add a (3,4) matrix and a (4,) vector — explain alignment
- Add a (3,1) column vector and a (1,4) row vector → (3,4) outer sum
- Show a subtle bug where broadcasting silently gives wrong results

**Exercises (5+):**
- Predict output shapes for various input combinations
- Normalize a batch of images: subtract per-channel mean (batch, H, W, C) — do it in one line
- Compute pairwise distances between two point sets without loops
- Implement batched outer product using broadcasting only

---

### 0.3 — Advanced Indexing: Fancy Indexing, Boolean Masks, np.where

**Concepts to teach:**
- Integer array indexing (fancy indexing) — always returns a copy
- Boolean mask indexing — filtering elements
- `np.where(condition, x, y)` — vectorized conditional
- `np.take`, `np.put`, `np.take_along_axis` for gathering
- Multi-dimensional indexing: `arr[rows, cols]` pattern

**Worked example:**
- Select specific elements from a 2D array using index arrays
- Mask all values below a threshold, replace with 0
- Gather operation: given indices, select from a batch (preview of how timestep selection works in diffusion)

**Exercises:**
- Implement top-k selection without sorting the full array
- Given a batch of images and a batch of crop coordinates, extract crops vectorized
- Implement `np.where` from scratch using boolean multiplication

---

### 0.4 — Einsum Mastery

**Concepts to teach:**
- Einsum notation: subscripts map to axes
- Common patterns and their einsum equivalents:
  - Matrix multiply: `ij,jk->ik`
  - Batch matrix multiply: `bij,bjk->bik`
  - Dot product: `i,i->`
  - Outer product: `i,j->ij`
  - Trace: `ii->`
  - Transpose: `ij->ji`
  - Sum over axis: `ij->i`
  - Element-wise multiply + sum (attention scores): `bhqd,bhkd->bhqk`
- When einsum is clearer than alternatives, when it's not

**Worked example:**
- Compute attention scores using einsum (preview of Module 3)
- Implement batched bilinear form: `x^T A x` for a batch

**Exercises:**
- Translate 10 NumPy expressions to einsum and vice versa
- Implement multi-head attention dot products using only einsum
- Compute Frobenius norm of a batch of matrices using einsum

---

### 0.5 — Vectorization Patterns: Replacing Loops

**Concepts to teach:**
- The performance gap: vectorized vs loop (demonstrate with timing)
- Cumulative operations: `np.cumsum`, `np.cumprod` (used in α_bar calculation!)
- Sliding window operations: `np.lib.stride_tricks.sliding_window_view`
- Reduction operations: `np.sum`, `np.mean`, `np.max` with axis parameter
- `np.apply_along_axis` — when you truly can't vectorize
- Vectorized operations that are non-obvious: `np.searchsorted`, `np.digitize`, `np.bincount`

**Worked example:**
- Compute cumulative product of alphas (directly relevant to diffusion!)
- Implement moving average without loops
- Vectorized histogram computation

**Exercises:**
- Given a loop-based implementation, rewrite as vectorized (5 examples, increasing difficulty)
- Implement `np.cumprod` from scratch using only `np.cumsum` and `np.exp`/`np.log`
- Timed challenge: vectorize a nested loop that computes pairwise cosine similarities

---

### 0.6 — Random Number Generation

**Concepts to teach:**
- `np.random.Generator` vs legacy `np.random.*`
- Seeds and reproducibility: why it matters for debugging
- Key distributions: uniform, normal (Gaussian), categorical
- **The reparameterization trick** — sample z = μ + σ * ε where ε ~ N(0,1)
  - Why: makes sampling differentiable (critical for diffusion and VAEs)
  - Preview of how this is used in the forward diffusion process
- Sampling from non-standard distributions using inverse CDF

**Worked example:**
- Generate reproducible random samples
- Implement reparameterized Gaussian sampling
- Visualize different noise schedules as distributions

**Exercises:**
- Implement the reparameterization trick for a diagonal Gaussian
- Sample from a mixture of Gaussians using only uniform samples
- Generate Gaussian noise matching specific spatial dimensions (preview of diffusion noising)

---

## Module 0 Exercises — Comprehensive Challenge Set

After all sections, include 10-15 standalone exercises that combine multiple concepts:

1. **Batch normalization from scratch** — compute mean/var per channel across batch, normalize, apply affine (broadcasting + reduction)
2. **Softmax implementation** — numerically stable, vectorized over batch dimension
3. **K-nearest neighbors** — pairwise distance matrix, top-k selection, all vectorized
4. **Image patch extraction** — extract non-overlapping patches from a batch of images using reshape/stride tricks
5. **Convolution via matrix multiply** — im2col approach, pure NumPy
6. **Cumulative alpha schedule** — given β_1...β_T, compute α_t, ᾱ_t, √ᾱ_t, √(1-ᾱ_t) — all vectorized
7. **Multinomial sampling** — sample from categorical distributions without `np.random.choice` loop
8. **Einsum gauntlet** — 10 rapid-fire einsum translations
9. **Memory profiling** — given code, predict peak memory usage based on array sizes and operations
10. **Timed vectorization** (10 min) — given a slow double loop, vectorize it to run 100x faster
