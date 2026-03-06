# Module 6: Training a Diffusion Model

## Purpose
Implement the full training loop from scratch. Every detail: data loading, the training algorithm, EMA, monitoring. By the end you'll have a trained model that generates images.

## 📄 Key Papers
- **DDPM** — Ho et al. 2020. [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
  - *Read for:* Algorithm 1 (training procedure). Section 4 for training details (lr, EMA, batch size).
- **Improved DDPM** — Nichol & Dhariwal 2021. [arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)
  - *Read for:* Practical training improvements. Learning the variance. Importance sampling of timesteps.

## Sections

### 6.1 — Data Loading & Preprocessing

**Concepts to teach:**
- Normalize images to [-1, 1] (not [0, 1]) — the model outputs noise which is mean-zero, and the final x_0 prediction should be in [-1, 1]
- Why [-1, 1]: symmetric range, matches Gaussian noise distribution
- Data augmentation: random horizontal flip (standard for CIFAR/ImageNet), no heavy augmentation needed
- `torch.utils.data.DataLoader`: batching, shuffling, num_workers, pin_memory
- Datasets: MNIST (28×28, grayscale — fastest for debugging), CIFAR-10 (32×32, color — more realistic)

**Worked example:**
- Set up CIFAR-10 dataset with proper transforms: `ToTensor()` → `Normalize([0.5]*3, [0.5]*3)` → RandomHorizontalFlip
- Create DataLoader with appropriate batch size

**Exercise:**
- Write a `denormalize()` function that maps [-1, 1] back to [0, 1] for visualization
- Verify normalization is correct by checking data range

---

### 6.2 — The Training Algorithm (DDPM Algorithm 1)

**Concepts to teach:**
- The algorithm is beautifully simple. Literally:
  ```
  repeat:
    x_0 ~ q(x_0)                    # sample a training image
    t ~ Uniform({1,...,T})            # sample a random timestep
    ε ~ N(0, I)                       # sample noise
    x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε   # noise the image
    loss = ||ε - ε_θ(x_t, t)||²      # predict the noise, compute MSE
    gradient step on loss
  ```
- That's the entire algorithm. Everything from Module 5 reduces to these 6 lines.
- Implementation details: precompute √ᾱ_t and √(1-ᾱ_t) as buffers, index with t

**Worked example:**
- Implement the full training step as a function: `train_step(model, x_0, optimizer, noise_schedule)`
- Walk through each line with comments explaining the math

**Exercise:**
- Implement the training step, run one batch, verify loss is reasonable (~1.0 at initialization for ε-prediction)

---

### 6.3 — Random Timestep Sampling

**Concepts to teach:**
- Uniform sampling: `t = torch.randint(0, T, (batch_size,))` — the default in DDPM
- Each sample in the batch gets a different random timestep
- **Importance sampling:** some timesteps contribute more to the loss/gradient
  - Low-t (little noise): easy for the model, small gradients
  - High-t (lots of noise): also relatively easy (predict roughly zero signal)
  - Mid-t: hardest, most informative gradients
  - Can sample t proportional to loss magnitude — but uniform works well enough in practice
- **Loss-aware sampling** (P2 weighting): weight the loss by SNR to focus on hard timesteps

**Worked example:**
- Compare uniform vs importance-sampled t distributions
- Show loss magnitude vs timestep (typically highest in the middle)

**Exercise:**
- Implement uniform timestep sampling
- Implement loss-aware sampling: track running loss per timestep, sample proportionally

---

### 6.4 — Noise Prediction: The Forward Pass

**Concepts to teach:**
- The model takes: (x_t, t) and outputs ε_θ — predicted noise
- x_t has shape (B, C, H, W) — same as the input image but noisier
- t has shape (B,) — integer timesteps, one per sample
- ε_θ has shape (B, C, H, W) — same as x_t
- Inside the model: t → timestep embedding → injected into ResBlocks (Module 4.5)

**Worked example:**
- Full forward pass walkthrough: prepare inputs, call model, get output
- Verify output shape matches input shape

---

### 6.5 — Loss Computation and Backprop

**Concepts to teach:**
- Loss: `F.mse_loss(noise_pred, noise)` — that's it
- Mean reduction: average over batch, channels, height, width
- Backpropagation: `loss.backward()`
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — prevents training instability
- Optimizer step: `optimizer.step()`
- Zero gradients: `optimizer.zero_grad()` — at the START of each step (or end, but be consistent)

**Worked example:**
- Complete training step with loss computation, backward, clip, step, zero_grad

---

### 6.6 — EMA (Exponential Moving Average) of Model Weights

**Concepts to teach:**
- EMA maintains a smoothed copy of model weights: `θ_ema = decay * θ_ema + (1-decay) * θ`
- Typical decay: 0.9999 (very slow-moving average)
- Why: reduces noise in the generated samples, produces better FID scores
- The EMA model is used for evaluation/sampling, NOT for computing gradients
- Implementation: maintain a second copy of the model, update after each training step

**Worked example:**
- Implement `EMA` class: `__init__`, `update()`, `apply()`, `restore()`
- Show the difference: samples from regular model vs EMA model

**Exercise:**
- Implement EMA from scratch
- Integrate it into the training loop

---

### 6.7 — Learning Rate Schedules

**Concepts to teach:**
- Warmup: gradually increase LR from 0 to target over first N steps — prevents early training instability
- Cosine decay: LR follows a cosine curve from target down to 0 (or min_lr)
- Constant LR: sometimes fine for diffusion models (DDPM used 2e-4 constant with Adam)
- Linear warmup + cosine decay: the standard modern recipe
- Implementation: `torch.optim.lr_scheduler` or manual

**Worked example:**
- Implement warmup + cosine decay schedule, plot the LR curve

**Exercise:**
- Add LR scheduling to the training loop

---

### 6.8 — Common Training Pitfalls

**Concepts to teach:**
- **Loss spikes:** gradient explosion — use gradient clipping
- **Loss plateau at start:** model hasn't learned anything yet — this is normal for ~100-1000 steps
- **Mode collapse:** generating the same image repeatedly — usually a bug, not a fundamental issue with diffusion
- **Color shift:** generated images have wrong color distribution — check normalization ([-1,1] vs [0,1] mismatch)
- **NaN loss:** usually overflow in sinusoidal embeddings or divide-by-zero in schedule computation — check dtypes
- **Slow convergence:** diffusion models need many steps (100K-1M). Don't panic if loss is high after 1000 steps.

**Content:** Troubleshooting guide with symptoms, causes, and fixes.

---

### 6.9 — Monitoring Training: FID Intuition, Visual Inspection

**Concepts to teach:**
- **Visual inspection:** generate samples every N steps, look at them. Most important metric.
- **Training loss curve:** should decrease but slowly, and is not perfectly correlated with sample quality
- **FID (Fréchet Inception Distance):** measures quality + diversity of generated samples
  - Lower is better. Compares statistics (mean + covariance) of real vs generated features from InceptionV3
  - DDPM on CIFAR-10: FID ≈ 3.17
  - Don't compute FID during training (expensive) — use it for final evaluation
- **Inception Score (IS):** measures quality + diversity. Less used now. Higher is better.
- For a prep notebook: visual inspection is sufficient. FID is good to understand conceptually.

**Worked example:**
- Set up a callback that generates and saves sample grids every 1000 training steps
- Log training loss with a simple moving average

---

## Module 6 Capstone Exercise

**Full training loop on MNIST (fast) then CIFAR-10 (realistic):**

**MNIST (debugging):**
- Train the toy U-Net from Module 4 on 28×28 grayscale MNIST
- 10K steps, batch_size=64
- Should see recognizable digits in ~5K steps
- Use for verifying the pipeline works

**CIFAR-10 (the real test):**
- Train on 32×32 color CIFAR-10
- 50K-100K steps for decent results
- EMA with decay=0.9999
- Cosine noise schedule
- Gradient clipping at 1.0
- Adam with lr=2e-4
- Generate sample grids every 5000 steps
- Target: recognizable (if blurry) objects after 50K steps
