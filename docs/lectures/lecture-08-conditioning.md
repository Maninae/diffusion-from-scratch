# Lecture 8: Conditioning & Classifier Guidance
**Module:** 4 — Controllable Generation
**Estimated reading time:** 20 minutes
**Dependencies:** Lecture 7

## Learning Objectives
- Explain the shift from unconditional to conditional generation: ε_θ(x_t, t) → ε_θ(x_t, t, c)
- Implement class-conditional generation by embedding class labels and injecting them into the U-Net
- Describe classifier guidance: using a separately trained classifier's gradients to steer generation
- Explain why classifier guidance was superseded by classifier-free guidance (Lecture 9)

## Narrative Arc
**The problem:** Our diffusion model generates images — but we can't control what it generates. We get random digits, random objects. For any practical application (text-to-image, conditional generation), we need to tell the model what to produce: "generate a 7" or "generate a cat."

**The attempt:** The first approach (Dhariwal & Nichol, 2021): train a separate classifier on noisy images, then during sampling, use its gradients to nudge the denoising process toward the desired class. This works — it produces sharp, class-specific images. But it requires training an entirely separate model on all noise levels, which is expensive and inflexible.

**The solution:** First, we need class-conditional architecture — a network that accepts a class label alongside the noisy image. This lecture covers how to embed and inject class labels. Then we cover classifier guidance as the historical approach to controllable generation, setting up the "why classifier-free guidance is better" argument in Lecture 9.

## Section Outline

### Section 8.1: From Unconditional to Conditional
- Unconditional model: `ε_θ(x_t, t)` — generates any image from the training distribution
- Conditional model: `ε_θ(x_t, t, c)` — generates images matching condition c
- The condition c can be: a class label, a text prompt, another image, an audio clip
- For this course: c is a digit class (0-9) for MNIST
- The architecture change is minimal — we're adding one more input to the network

### Section 8.2: Class Embedding & Injection
- Class label (integer) → learned embedding vector:
  - `class_emb = nn.Embedding(num_classes, emb_dim)` — lookup table, one vector per class
- **Injection method 1 — Addition to time embedding:**
  - Simply add class embedding to the time embedding before the MLP
  - `combined_emb = time_emb + class_emb(c)`
  - The network receives a single conditioning vector that encodes both time and class
  - Simplest approach, works well for class-conditional

- **Injection method 2 — Separate injection:**
  - Process class embedding through its own MLP
  - Add to features at each ResBlock independently from time
  - More expressive but more parameters

- **Injection method 3 — AdaGN:**
  - Predict separate scale/shift for class conditioning
  - Most expressive — used in advanced architectures

**Pseudocode:**
```
class ConditionalUNet(UNet):
    def __init__(self, num_classes, ...):
        # +1 for a null class token used in CFG (Lecture 9) — index num_classes
        # means "no class," enabling unconditional prediction from the same model
        self.class_emb = Embedding(num_classes + 1, emb_dim)

    def forward(self, x_t, t, class_label=None):
        t_emb = self.time_mlp(sinusoidal_embedding(t))

        if class_label is not None:
            c_emb = self.class_emb(class_label)   # (B, emb_dim)
            t_emb = t_emb + c_emb                  # combine

        # Rest of U-Net forward pass uses t_emb as before
        ...
```

### Section 8.3: Training the Conditional Model
- Training is nearly identical to unconditional (Lecture 6)
- Each training sample now includes a label: `(x_0, class_label)`
- The label is embedded and injected alongside the time embedding
- Loss is unchanged: `L = ||ε - ε_θ(x_t, t, c)||²`
- The model learns to make class-specific noise predictions

**Pseudocode:**
```
def train_step(model, x_0, class_label, optimizer, schedule):
    t = random_integers(0, T-1, batch_size)
    noise = randn_like(x_0)
    x_t = q_sample(x_0, t, noise, schedule)
    noise_pred = model(x_t, t, class_label)      # now with class!
    loss = mse_loss(noise_pred, noise)
    ...
```

### Section 8.4: Classifier Guidance (Historical Context)
- **The idea:** use a separately trained classifier p_φ(y | x_t) to steer generation
- Train a classifier on noisy images at all noise levels (expensive!)
- During sampling, at each step:
  1. Predict noise: `ε_θ(x_t, t)`
  2. Compute classifier gradient: `∇_{x_t} log p_φ(y | x_t)` — "which direction makes x_t look more like class y?"
  3. Shift the noise prediction: `ε̂ = ε_θ - s · √(1-ᾱ_t) · ∇_{x_t} log p_φ(y | x_t)`
  4. Use ε̂ for the denoising step
- s is the **guidance scale**: controls how strongly the classifier steers
  - s = 0: no guidance (unconditional)
  - s = 1: standard classifier guidance
  - s > 1: amplified guidance — sharper but less diverse

**Pseudocode:**
```
def classifier_guided_sample(model, classifier, class_label, schedule, guidance_scale):
    x = randn(shape)
    for t in reversed(range(T)):
        # Predict noise
        noise_pred = model(x, t)

        # Classifier gradient
        x.requires_grad_(True)
        log_prob = classifier(x, t).log_softmax(dim=-1)[:, class_label]
        grad = autograd.grad(log_prob.sum(), x)[0]
        x.requires_grad_(False)

        # Shift noise prediction
        noise_pred = noise_pred - guidance_scale * sqrt(1 - schedule.alphas_cumprod[t]) * grad

        # Standard DDPM step with modified noise prediction
        x = ddpm_step(x, noise_pred, t, schedule)
    return x
```

### Section 8.5: Why Classifier Guidance Falls Short
- **Extra model needed:** must train a classifier on noisy images at all noise levels — expensive
- **Inflexible:** changing what you condition on requires training a new classifier
- **Gradient computation:** requires backprop through the classifier at every sampling step — slow
- **Limited to classification:** doesn't naturally extend to rich conditioning (text prompts)
- These limitations motivated classifier-free guidance (Lecture 9)

## Key Equations
- Conditional model: `ε_θ(x_t, t, c)` where c is the class label
- Classifier guidance shift: `ε̂ = ε_θ - s · √(1-ᾱ_t) · ∇_{x_t} log p_φ(y | x_t)`
- Guided reverse mean: use ε̂ in place of ε_θ in the standard DDPM mean formula
- Implicit distribution: sampling from `p(x | y) ∝ p(x) · p(y | x)^s`

## Code Examples
- `ConditionalUNet` — modified U-Net with class embedding (pseudocode first, PyTorch reference)
- Conditional training step — minimal change from Lecture 6
- `classifier_guided_sample()` — sampling with classifier gradients
- Guidance scale sweep: generate digits at s = {0, 1, 3, 5, 10}

## Diagrams & Visuals
- **Conditional U-Net diagram:** show where class embedding enters (added to time embedding before MLP)
- **Classifier guidance diagram:** denoising step + classifier gradient arrow pushing toward desired class
- **Guidance scale sweep grid:** rows = digit classes, columns = guidance scales — show sharpening with higher s
- **Quality-diversity tradeoff curve:** as guidance increases, fidelity up but diversity down

## Source Material
- Module 08 §8.1 (Class-Conditional Generation)
- Module 08 §8.2 (Classifier Guidance)
