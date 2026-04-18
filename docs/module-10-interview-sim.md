# Module 10: Assessment & Coding Challenges

## Purpose
Apply everything under time pressure. These exercises simulate a technical coding assessment: live coding with image diffusion, practical Python/PyTorch, structured code, verbal explanation of thought process.

## 📄 Key Papers
All papers from previous modules apply here. No new papers — this is about execution.

## Format Notes for the Coding Agent Building This

Each exercise should include:
- **Timer cell:** A markdown cell stating the time limit and what's expected
- **Starter code:** Minimal imports and any provided scaffolding (like a pre-trained model checkpoint path)
- **Evaluation criteria:** What "good" looks like (code structure, correctness, completeness)
- **Solution:** Complete, clean, well-commented implementation
- **Debrief:** Common mistakes, key evaluation criteria, talking points

---

## Exercises

### 10.1 — Timed Exercise (45 min): Minimal Diffusion on 2D Point Clouds

**Setup:** This is the simplest possible diffusion model — no images, no convolutions, just MLPs on 2D points. Tests core diffusion understanding without architecture complexity.

**Prompt:**
> "Implement a diffusion model that learns to generate samples from a 2D distribution (e.g., a spiral or Swiss roll). You should implement: the forward noising process, a simple MLP denoiser, the training loop, and the sampling loop."

**Evaluation criteria:**
- Correct noise schedule computation (β, α, ᾱ)
- Correct forward process: `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε`
- Simple MLP that takes (x_t, t) and predicts ε
- Timestep embedding (sinusoidal)
- Training loop: sample x_0, sample t, sample ε, compute loss
- Sampling loop: iterative denoising from x_T → x_0
- Clean code structure, meaningful variable names

**Starter code provided:**
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate spiral dataset
def make_spiral(n_points=1000):
    t = torch.linspace(0, 4*np.pi, n_points)
    x = t * torch.cos(t) / (4*np.pi)
    y = t * torch.sin(t) / (4*np.pi)
    return torch.stack([x, y], dim=1) + 0.05 * torch.randn(n_points, 2)

data = make_spiral(2000)
# Your implementation below...
```

**Expected deliverables in 45 minutes:**
1. Noise schedule (linear or cosine)
2. MLP model with timestep conditioning
3. Training loop (should converge in ~5000 steps)
4. Sampling function
5. Visualization: generated samples overlaid on training data

---

### 10.2 — Timed Exercise (45 min): Implement Sampling with CFG

**Setup:** Given a pre-trained class-conditional U-Net (from Module 8), implement the full DDIM sampling loop with classifier-free guidance.

**Prompt:**
> "You have a trained class-conditional diffusion model. Implement DDIM sampling with classifier-free guidance. The model accepts (x_t, t, class_label) where class_label can be a null token for unconditional prediction."

**Evaluation criteria:**
- Correct DDIM update formula
- CFG: two forward passes (conditional + unconditional), correct combination
- Guidance scale parameter
- Timestep sub-selection for fewer steps
- Proper handling of the final step (no noise added)
- Clipping/denormalization of output

**Starter code provided:**
```python
# Pre-trained model (treat as black box)
model = load_pretrained_model()  # model(x_t, t, class_label) -> noise_pred
noise_schedule = load_schedule()  # Contains alphas_cumprod, etc.
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES  # Index for unconditional

def ddim_sample_cfg(model, noise_schedule, shape, class_label, 
                     guidance_scale=7.5, num_steps=50):
    """
    Generate samples using DDIM with classifier-free guidance.
    
    Args:
        model: trained conditional diffusion model
        noise_schedule: precomputed schedule values
        shape: (B, C, H, W) output shape
        class_label: integer class to generate
        guidance_scale: CFG scale (1.0 = no guidance)
        num_steps: number of DDIM steps
    
    Returns:
        Generated images in [0, 1] range
    """
    # Your implementation here...
```

**Expected deliverables in 45 minutes:**
1. Timestep sub-selection (uniform spacing over T steps)
2. DDIM sampling loop
3. CFG implementation (batched conditional + unconditional)
4. Working generation of class-conditional samples
5. Grid visualization at different guidance scales

---

### 10.3 — Timed Exercise (45 min): Training Step from Scratch

**Setup:** Implement a single training step for an image diffusion model, including all the pieces: noise schedule, forward process, loss computation.

**Prompt:**
> "Implement the complete training step for a DDPM-style diffusion model. Given a batch of images and a model, implement: noise schedule setup, the noising process, loss computation, and a training loop that runs for N steps."

**Evaluation criteria:**
- Precomputing schedule values as tensors
- Correct indexing of schedule values by timestep
- Proper random timestep sampling (one per batch element)
- Correct noise addition using the closed-form
- MSE loss between predicted and actual noise
- Gradient clipping
- EMA (bonus)

**Starter code provided:**
```python
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Assume model is provided
model = UNet(...)  # Takes (x_t, t) -> noise_pred
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# CIFAR-10 data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

T = 1000  # Total timesteps

# Your implementation: setup schedule, implement train_step, run training loop
```

**Expected deliverables in 45 minutes:**
1. Noise schedule computation (all derived quantities)
2. `train_step(model, x_0, optimizer)` function
3. Training loop with loss logging
4. Verify loss decreases over first 100 steps

---

### 10.4 — Verbal Explanation Prompts

Practice explaining these concepts clearly and concisely — the kind of questions that test deep understanding.

**Core Understanding:**
1. "Walk me through the forward diffusion process. What happens to an image at each step?"
2. "Why do we predict noise instead of predicting the clean image directly?"
3. "What is the reparameterization trick and why do we need it?"
4. "Explain the simplified DDPM loss. Why does MSE on noise work?"
5. "What's the difference between DDPM and DDIM sampling?"

**Architecture:**
6. "Why does the diffusion U-Net use GroupNorm instead of BatchNorm?"
7. "Where do you put attention layers in the U-Net and why?"
8. "How does timestep conditioning work? Walk me through the sinusoidal embedding."
9. "What are skip connections and why are they important in the U-Net?"

**Guidance & Conditioning:**
10. "Explain classifier-free guidance. How does training differ from unconditional diffusion?"
11. "What does the guidance scale control? What happens as you increase it?"
12. "How do negative prompts work mechanically?"

**Advanced:**
13. "What's the difference between pixel-space and latent diffusion? Why is latent better for high-res?"
14. "What are the tradeoffs between ε-prediction, x_0-prediction, and v-prediction?"
15. "What is flow matching and how does it differ from diffusion?"
16. "How would you scale this model to generate 512×512 images?"

**For each question, the notebook should include:**
- A "try to answer before reading" prompt
- A concise, well-structured model answer (2-4 paragraphs)
- Key points to hit (bullet list)
- Common mistakes to avoid

---

### 10.5 — Code Review Exercise: Find and Fix the Bugs

**Setup:** Present a buggy diffusion model implementation. The student must identify and fix all bugs.

**Bugs to include (hidden in otherwise reasonable code):**
1. **Wrong scaling in forward process:** Using `alpha_t` instead of `alpha_bar_t` (cumulative product)
2. **Missing sqrt:** `x_t = alpha_bar * x_0 + (1 - alpha_bar) * noise` instead of `sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise`
3. **Off-by-one in sampling:** Loop goes from T to 0 instead of T to 1 (adding noise at t=0)
4. **Forgot to set model to eval mode** during sampling
5. **Gradient accumulation bug:** Missing `optimizer.zero_grad()`
6. **Wrong normalization:** Data in [0, 1] but model expects [-1, 1]
7. **Timestep embedding bug:** Using `t` directly as float instead of sinusoidal embedding
8. **CFG formula wrong:** `eps_uncond + scale * eps_cond` instead of `eps_uncond + scale * (eps_cond - eps_uncond)`
9. **Device mismatch:** Schedule tensors on CPU, model on GPU
10. **Variance schedule computed wrong:** `beta` values not clipped, causing numerical issues

**Format:**
- Present the buggy code (looks plausible at first glance)
- "Find all bugs" exercise
- Solutions with explanations of what each bug causes

---

### 10.6 — Design Discussion: Scaling to High Resolution

**Prompt:**
> "You've built a diffusion model that works on 32×32 CIFAR images. Your manager asks you to scale it to 512×512. Walk me through your approach."

**Expected discussion points:**
1. **The problem:** Pixel-space U-Net at 512×512 is too expensive (memory, compute, attention)
2. **Solution 1: Latent diffusion** — Train a VAE, do diffusion in latent space
   - Compression ratio: 512×512×3 → 64×64×4
   - U-Net operates on much smaller feature maps
   - Trade-off: VAE reconstruction quality limits output quality
3. **Solution 2: Cascaded models** — Low-res diffusion → super-resolution diffusion
   - Train a 64×64 base model, then a 64→512 upsampler
   - Each model is simpler, but you need two models
4. **Architecture considerations:**
   - Attention only at 32×32 and below (too expensive at 64×64+)
   - Consider DiT (transformer) for better scaling properties
   - Gradient checkpointing to reduce memory
   - Mixed precision (fp16/bf16) training
5. **Training considerations:**
   - Much larger dataset needed (CIFAR-10 is too small)
   - Longer training (100K+ steps minimum)
   - Larger batch sizes (distributed training)
   - EMA becomes more important
6. **Inference considerations:**
   - DDIM with 50 steps instead of 1000
   - Classifier-free guidance for quality
   - Batched sampling for throughput

**Format:**
- Present the discussion prompt
- Structured model answer with the points above
- Follow-up questions to deepen understanding

---

## General Tips (Include as Final Section)

1. **Think out loud.** Narrate your reasoning as you code — not just the what, but the why.
2. **Start with the simplest version.** Get something working first, then optimize. Don't try to implement everything at once.
3. **Write clean code from the start.** Good variable names, logical structure, brief comments on key lines.
4. **Know your shapes.** At every step, know the tensor shapes. Print them if unsure. Shape mismatches are the #1 bug.
5. **Have a plan before coding.** Spend 2-3 minutes outlining your approach before writing code. "First I'll set up the schedule, then the model, then training, then sampling."
6. **Know the math cold.** Be able to write `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε` without hesitation. This is the foundation of everything.
