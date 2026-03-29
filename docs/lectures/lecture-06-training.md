# Lecture 6: Training a Diffusion Model
**Module:** 3 — Training & Sampling
**Estimated reading time:** 25 minutes
**Dependencies:** Lectures 3, 5 (need the loss function and the U-Net architecture)

## Learning Objectives
- Implement the complete DDPM training loop from data loading to loss computation
- Explain EMA (Exponential Moving Average) and why it improves sample quality
- Configure training stability tools: gradient clipping, learning rate warmup, cosine decay
- Diagnose common training pitfalls: loss spikes, color shift, NaN, slow convergence
- Train a working diffusion model on MNIST that generates recognizable digits

## Narrative Arc
**The problem:** We have the math (Lecture 3) and the architecture (Lectures 4-5). But turning Algorithm 1 into a working training pipeline involves dozens of practical decisions — data normalization, optimizer choice, learning rate schedule, gradient clipping, weight averaging — and getting any of them wrong can silently destroy training.

**The attempt:** A naive implementation (just the 6-line algorithm) will train, but slowly and with poor sample quality. The loss might spike, the model might produce color-shifted images, or convergence might take 10× longer than necessary.

**The solution:** A carefully engineered training loop with the standard recipe: normalize to [-1, 1], AdamW optimizer, gradient clipping at 1.0, EMA with decay 0.9999, and visual monitoring every few thousand steps. These aren't optional extras — they're the difference between a model that generates recognizable images in 5K steps and one that's still producing noise at 50K.

## Section Outline

### Section 6.1: Data Loading & Preprocessing
- **Normalize to [-1, 1]**, not [0, 1]:
  - The model predicts noise ε ~ N(0, I) — mean-zero, symmetric
  - Images in [-1, 1] match this: symmetric range, zero-centered
  - Transform: `ToTensor()` → `Normalize([0.5], [0.5])` for MNIST
- **Data augmentation:** none needed for MNIST. Horizontal flips would destroy digit identity (3, 4, 5, 7 become unrecognizable; 6/9 swap meaning). Diffusion's random noise at each training step provides strong implicit augmentation — each image is seen at thousands of different noise levels, effectively multiplying the dataset.
- **DataLoader setup:** batch_size=64, shuffle=True, num_workers for speed
- **Denormalization for display:** `x_display = (x + 1) / 2` to map back to [0, 1]

**Pseudocode:**
```
transform = Compose([
    ToTensor(),                    # [0, 255] → [0, 1]
    Normalize([0.5], [0.5]),       # [0, 1] → [-1, 1]
])
dataset = MNIST(root='./data', train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### Section 6.2: The Training Loop — DDPM Algorithm 1 in Code
- The complete training step, expanded from the 6-line algorithm:

**Pseudocode:**
```
def train_step(model, x_0, optimizer, schedule, device):
    # 1. Sample random timesteps — one per image in the batch
    t = random_integers(0, T-1, size=batch_size).to(device)

    # 2. Sample noise
    noise = randn_like(x_0)                                    # (B, 1, 28, 28)

    # 3. Create noisy images using closed-form
    sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[t]           # (B,)
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t] # (B,)
    # Reshape for broadcasting: (B,) → (B, 1, 1, 1)
    x_t = sqrt_alpha_bar[:, None, None, None] * x_0 + \
          sqrt_one_minus[:, None, None, None] * noise          # (B, 1, 28, 28)

    # 4. Predict noise
    noise_pred = model(x_t, t)                                 # (B, 1, 28, 28)

    # 5. Compute loss
    loss = mse_loss(noise_pred, noise)

    # 6. Update weights
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()
```

- Key implementation detail: schedule values must be on the same device as the model
- Use `prepare_schedule()` from `utils/schedule.py` to move all tensors to device

### Section 6.3: Timestep Sampling Strategies
- **Uniform sampling** (standard): `t = torch.randint(0, T, (batch_size,))`
  - Each sample gets a random t, independent of others
  - Simple, works well, used in DDPM
- **Why uniform is suboptimal:**
  - Low-t (little noise): easy for the model, small loss
  - High-t (lots of noise): also relatively easy (near-zero signal)
  - Mid-t: hardest, most informative gradients
- **P2 weighting (Perception Prioritized):**
  - Weight loss by 1/(1 + SNR(t)) — upweights mid-noise timesteps
  - Improves FID, especially on complex datasets
- **Practical advice:** start with uniform. Switch to importance-weighted if needed.

### Section 6.4: Gradient Clipping
- Diffusion loss can spike — especially in early training when the model is poor
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- This rescales gradients if their total norm exceeds 1.0
- Prevents catastrophic parameter updates from outlier batches
- Standard practice in all modern diffusion training

### Section 6.5: EMA — Exponential Moving Average
- Maintain a smoothed copy of model weights:
  - `θ_ema = decay · θ_ema + (1 - decay) · θ`
- Typical decay: 0.9999 (the EMA model changes very slowly)
- **Why EMA helps:** training weights oscillate around the optimum; EMA averages out these oscillations, producing smoother, better-quality samples
- **Critical:** use EMA weights for sampling/evaluation, NOT for computing gradients
- Implementation: maintain a shadow copy of parameters, update after each training step

**Pseudocode:**
```
class EMA:
    def __init__(model, decay=0.9999):
        self.shadow = copy(model.parameters())
        self.decay = decay

    def update(model):
        for ema_param, model_param in zip(self.shadow, model.parameters()):
            ema_param.data = decay * ema_param + (1 - decay) * model_param

    def apply_to(model):
        # Temporarily replace model weights with EMA weights for sampling
        ...
```

### Section 6.6: Learning Rate Schedule
- **Warmup:** gradually increase LR from 0 to target over first 1000 steps
  - Prevents early instability when the model hasn't learned anything yet
- **Cosine decay:** LR follows a cosine curve from target down to 0
- **Constant LR:** the original DDPM used a constant lr=2e-4 with Adam (not AdamW) — also works and is simpler
- **Practical recipe:** AdamW with lr=2e-4, linear warmup for 1000 steps, cosine decay to 0. AdamW adds weight decay regularization (decoupled from the learning rate), which is modern best practice. For MNIST, plain Adam with constant LR is perfectly fine.

### Section 6.7: Common Training Pitfalls
- **Loss spikes:** gradient explosion → use gradient clipping (max_norm=1.0)
- **Loss plateau at start:** normal for first ~100-1000 steps — the model is learning
- **Color shift in samples:** normalization mismatch — data is [0, 1] but model expects [-1, 1], or denormalization is wrong
- **NaN loss:** usually overflow in sinusoidal embeddings or divide-by-zero in schedule — check dtypes (use float32)
- **Slow convergence:** diffusion models need many steps. 5K steps for MNIST, 50K+ for CIFAR. Don't panic.
- **All samples look the same:** mode collapse is rare in diffusion, but check for bugs in noise sampling

### Section 6.8: Monitoring Training
- **Visual inspection is king:** generate a grid of samples every 1K-5K steps
  - "If you see recognizable digits, training is working"
- **Loss curve:** should decrease but slowly and noisily. Not perfectly correlated with sample quality.
  - Use a moving average (window=100) for readability
- **FID (Fréchet Inception Distance):** the standard metric for generative quality
  - Compares real vs generated image statistics through InceptionV3 features
  - Lower is better. DDPM on CIFAR-10: FID ≈ 3.17
  - Too expensive to compute during training — use for final evaluation only
- **For this course:** visual inspection + loss curve is sufficient

### Section 6.9: Saving Checkpoints
- Save periodically (every 5K-10K steps) so training can resume and Lecture 7 can load the model
- What to save:
  - Model state dict
  - EMA state dict
  - Optimizer state dict
  - Schedule dict
  - Current step number
- This checkpoint is loaded in Lecture 7 for sampling and in Lectures 8-9 for conditioning

**Pseudocode:**
```
def save_checkpoint(model, ema, optimizer, schedule, step, path='diffusion_mnist.pt'):
    torch.save({
        'model': model.state_dict(),       # for continued training
        'ema': ema.shadow_state_dict(),     # for sampling (Lecture 7)
        'optimizer': optimizer.state_dict(),
        'schedule': schedule,
        'step': step,
    }, path)
    # Save both model AND EMA: model weights for resuming training,
    # EMA weights for high-quality sampling. Lecture 7 loads 'ema'.
```

- **Why save both?** The model weights are what the optimizer updates — needed to resume training. The EMA weights are the smoothed copy — they produce better samples and are what Lecture 7 loads for generation.
```

### Section 6.10: Putting It All Together — The Full Training Script

**Pseudocode:**
```
# Setup
model = UNet(image_channels=1, base_channels=64, ...)  # from utils
schedule = prepare_schedule(cosine_schedule(T=1000), device)
optimizer = AdamW(model.parameters(), lr=2e-4)
ema = EMA(model, decay=0.9999)

# Training loop
for step in range(num_steps):
    x_0, _ = next(dataloader)           # (B, 1, 28, 28) in [-1, 1]
    x_0 = x_0.to(device)

    loss = train_step(model, x_0, optimizer, schedule, device)
    ema.update(model)

    if step % 1000 == 0:
        # Generate samples with EMA model for visualization
        samples = ddpm_sample(ema_model, shape=(16, 1, 28, 28), schedule)
        display_grid(samples)
        log(f"Step {step}, Loss: {loss:.4f}")
```

## Key Equations
- Training loss: `L = E_{t,x_0,ε}[ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||² ]`
- EMA update: `θ_ema = 0.9999 · θ_ema + 0.0001 · θ`
- Gradient clipping: rescale if `||∇L||₂ > max_norm`

## Code Examples
- `train_step()` — complete training step (pseudocode first, PyTorch reference)
- `EMA` class — init, update, apply
- Full training script — data loading through sample generation
- Uses `q_sample()` from `utils/diffusion`, U-Net from `utils/unet`

## Diagrams & Visuals
- **Training progress strip:** sample grids at step 0, 1K, 5K, 10K, 20K — watch digits emerge from noise
- **Loss curve plot:** typical diffusion training loss over 20K steps (noisy, gradually decreasing)
- **EMA vs non-EMA comparison:** same checkpoint, samples from both — EMA is smoother
- **Training pipeline flowchart:** data → normalize → sample t,ε → noise → predict → loss → backprop → EMA update

## Source Material
- Module 06 §6.1 (Data Loading & Preprocessing)
- Module 06 §6.2 (Training Algorithm)
- Module 06 §6.3 (Random Timestep Sampling)
- Module 06 §6.4 (Noise Prediction)
- Module 06 §6.5 (Loss Computation)
- Module 06 §6.6 (EMA)
- Module 06 §6.7 (Learning Rate Schedules)
- Module 06 §6.8 (Common Pitfalls)
- Module 06 §6.9 (Monitoring)
