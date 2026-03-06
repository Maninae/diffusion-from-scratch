# Module 1: PyTorch Fundamentals

## Purpose
Build PyTorch fluency from the ground up. The interview is PyTorch-based — you need to write modules, training loops, and custom layers without hesitation.

## 📄 Key Papers
- **PyTorch: An Imperative Style, High-Performance Deep Learning Library** — Paszke et al. 2019. [arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)
  - *Read for:* Design philosophy of eager execution, autograd internals.

## Sections

### 1.1 — Tensor Creation, dtypes, Device Management

**Concepts to teach:**
- Tensor creation: `torch.tensor`, `torch.zeros`, `torch.randn`, `torch.arange`, `torch.linspace`
- dtypes: `float32` (default), `float64`, `int64`, `bool` — when to use which
- Device management: `.to(device)`, `torch.device('cuda')`, `torch.device('mps')`
- NumPy interop: `.numpy()`, `torch.from_numpy()` — shared memory!
- Factory functions: `torch.zeros_like`, `torch.ones_like`, `torch.randn_like` (used heavily in diffusion)

**Worked example:**
- Create tensors on different devices, demonstrate `.to()` semantics
- Show the shared memory gotcha with NumPy conversion

**Exercise:**
- Write a utility function that auto-detects the best available device (CUDA > MPS > CPU)
- Create a batch of random noise tensors matching a given image tensor's shape and device (this is literally what diffusion does)

---

### 1.2 — Autograd Deep Dive

**Concepts to teach:**
- `requires_grad=True`: what it means, which tensors need it
- Computation graph: dynamic construction during forward pass
- `.backward()`: backpropagation through the graph
- `.grad`: where gradients accumulate (and the zero-grad gotcha!)
- `torch.no_grad()`: disabling autograd for inference (saves memory)
- `detach()`: removing a tensor from the graph
- `grad_fn`: inspecting the chain of operations

**Worked example:**
- Build a simple computation, call `.backward()`, inspect `.grad`
- Show the gradient accumulation bug (forgetting `optimizer.zero_grad()`)
- Demonstrate memory savings with `torch.no_grad()` during inference

**Exercise:**
- Manually compute gradients for a small network, verify against autograd
- Implement gradient clipping from scratch using `.grad` access

---

### 1.3 — In-Place Operations and Why They Break Autograd

**Concepts to teach:**
- In-place operations: `add_()`, `mul_()`, `relu_()`, `[:]` assignment
- Why they're dangerous: they overwrite data needed for backward pass
- When they're safe: leaf tensors before any operation, inference mode
- The error message: "one of the variables needed for gradient computation has been modified by an inplace operation"
- Best practice: avoid in-place ops in training code unless you know exactly what you're doing

**Worked example:**
- Show an in-place op that silently produces wrong gradients
- Show the RuntimeError when autograd catches it
- Demonstrate the safe alternative

**Exercise:**
- Given code with in-place ops, identify which ones are safe and which break autograd

---

### 1.4 — nn.Module Anatomy

**Concepts to teach:**
- `__init__`: registering submodules and parameters
- `forward()`: the computation — called via `model(x)`, not `model.forward(x)`
- Parameter registration: `nn.Parameter` vs regular tensors
- `register_buffer`: persistent state that isn't a parameter (e.g., running mean in BatchNorm, noise schedule constants in diffusion)
- `model.parameters()` and `model.named_parameters()` — what the optimizer sees
- `model.train()` vs `model.eval()` — what changes (dropout, batchnorm)
- Nested modules: `nn.ModuleList`, `nn.ModuleDict`, `nn.Sequential`

**Worked example:**
- Build a module with parameters, buffers, and submodules
- Show that a plain `self.tensor = torch.randn(...)` is NOT registered
- Iterate parameters, count them, check devices

**Exercise:**
- Build a `TimestepEmbedding` module (sinusoidal embeddings — preview of diffusion)
- Verify all parameters are registered and on the correct device

---

### 1.5 — Custom Layers from Scratch

**Concepts to teach:**
- `nn.Linear` internals: `y = xW^T + b` — implement manually
- `nn.Conv2d` internals: implement as matrix multiply via unfolding
- Weight initialization: Xavier, Kaiming, why it matters
- Comparing your implementation to PyTorch's: numerical equivalence

**Worked example:**
- Implement `MyLinear(nn.Module)` with proper weight init
- Forward pass: `x @ self.weight.T + self.bias`
- Verify output matches `nn.Linear` for the same weights

**Exercise:**
- Implement `MyConv2d` using `F.unfold` and matrix multiply
- Implement `MyGroupNorm` from scratch (important — diffusion models use GroupNorm)

---

### 1.6 — Loss Functions from Raw Tensors

**Concepts to teach:**
- MSE loss: `mean((pred - target)^2)` — this is THE diffusion loss
- Cross-entropy loss: softmax + negative log likelihood — implement step by step
- Numerical stability: log-sum-exp trick
- Reduction modes: `mean` vs `sum` vs `none` — when each matters
- Why `F.mse_loss(noise_pred, noise)` is the entire diffusion training objective

**Worked example:**
- Implement MSE loss, verify against `F.mse_loss`
- Implement numerically stable cross-entropy, verify against `F.cross_entropy`

**Exercise:**
- Implement Huber loss (smooth L1) from scratch
- Implement weighted MSE loss where different timesteps get different weights (relevant to diffusion training)

---

### 1.7 — Optimizers: What They Actually Do

**Concepts to teach:**
- SGD: `θ = θ - lr * grad` — the simplest update
- SGD with momentum: exponential moving average of gradients
- Adam: adaptive learning rates per parameter, first/second moment estimates, bias correction
- AdamW: weight decay done right (decoupled from gradient)
- Learning rate: the most important hyperparameter
- Why Adam is the default for diffusion models

**Worked example:**
- Implement SGD update manually using `.grad`
- Implement Adam update manually, verify against `torch.optim.Adam`
- Show the difference in training curves between SGD and Adam

**Exercise:**
- Implement a minimal Adam optimizer class from scratch
- Train a small network with your optimizer, compare to PyTorch's

---

## Module 1 Capstone Exercise

**Build a complete MLP from scratch, train on MNIST:**
- Custom `Linear` layer (no `nn.Linear`)
- Custom `ReLU` activation
- Custom MSE or CrossEntropy loss
- Manual training loop with `torch.optim.Adam`
- Must achieve >95% accuracy
- No `nn.Sequential` allowed — wire everything manually in `forward()`

This exercise integrates everything from the module and builds the muscle memory for writing PyTorch from scratch.
