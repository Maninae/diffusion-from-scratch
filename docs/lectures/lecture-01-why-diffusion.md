# Lecture 1: Why Diffusion? The Generative Models Landscape
**Module:** 1 — The Diffusion Framework
**Estimated reading time:** 15 minutes
**Dependencies:** None — this is the entry point

## Learning Objectives
- Articulate the generative modeling problem: learn p(x) from samples, then generate new x
- Compare GANs, VAEs, normalizing flows, autoregressive models, and diffusion models on key tradeoffs
- Explain the core diffusion intuition: destruction is easy, generation is learned
- Trace the historical timeline from Sohl-Dickstein 2015 to the modern diffusion explosion

## Narrative Arc
**The problem:** We want to generate new, realistic images from a learned distribution — but every existing approach has a painful tradeoff. GANs are unstable to train and collapse modes. VAEs produce blurry samples. Flows are constrained architecturally. Autoregressive models are painfully slow.

**The attempt:** Each framework tries a different trick to model p(x): adversarial games, variational bounds, invertible transforms, sequential factorization. Each makes a different sacrifice — training stability, sample quality, architectural freedom, or generation speed.

**The solution:** Diffusion models take a completely different angle. Instead of learning to generate in one shot, they learn to gradually reverse a simple destruction process. Add noise step by step until data becomes pure static, then train a neural network to reverse each step. The result: stable training (no adversarial dynamics), state-of-the-art sample quality, and flexible architecture choices.

## Section Outline

### Section 1.1: The Generative Modeling Problem
- What "generative model" means: given samples from p(x), learn to generate new samples
- Why this is hard: high-dimensional distributions, no closed-form density
- The two things we want: (1) sample quality, (2) training stability
- Brief note on evaluation: FID, visual inspection, likelihood

### Section 1.2: The Generative Models Zoo
Cover each with: core idea, one-sentence mechanism, key strength, key weakness.

**GANs (Goodfellow et al., 2014)**
- Generator vs discriminator adversarial game
- Strength: sharp, high-quality samples
- Weakness: mode collapse, training instability, no explicit density

**VAEs (Kingma & Welling, 2013)**
- Encoder-decoder with latent space, trained via ELBO
- Strength: explicit probabilistic framework, smooth latent space
- Weakness: blurry outputs (MSE reconstruction penalty)

**Normalizing Flows (Rezende & Mohamed, 2015)**
- Chain of invertible transforms; exact log-likelihood via change of variables
- Strength: exact density evaluation
- Weakness: invertibility constraint limits architecture expressiveness

**Autoregressive Models (van den Oord et al., 2016)**
- Factor p(x) as product of conditionals, predict one pixel/token at a time
- Strength: tractable exact likelihood, powerful density estimation
- Weakness: sequential generation is slow; can't parallelize

### Section 1.3: Enter Diffusion Models
- The core intuition in plain language: "destroying an image is easy — just add noise. What if we could learn to undo it?"
- The two processes:
  - **Forward:** gradually add Gaussian noise over T steps until the image is pure static
  - **Reverse:** learn a neural network to predict and remove the noise, step by step
- Why this works: each denoising step is a small, learnable problem. The network only has to remove a tiny bit of noise at each step — not generate an entire image from scratch.
- Key properties: stable training (MSE loss, no adversarial game), state-of-the-art FID scores, flexible architecture

### Section 1.4: Historical Timeline
- **2015:** Sohl-Dickstein et al. — "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." The original idea, framed through statistical physics. Promising but not competitive.
- **2020:** Ho et al. — "Denoising Diffusion Probabilistic Models (DDPM)." The breakthrough paper. Simple training procedure, competitive image quality. Algorithm 1 and Algorithm 2 — the foundation of everything in this course.
- **2021:** Dhariwal & Nichol — "Diffusion Models Beat GANs." Classifier guidance, architecture improvements. Diffusion officially surpasses GANs on ImageNet FID.
- **2021:** Song et al. — Score-based SDE framework. Unifies diffusion and score matching.
- **2022:** Rombach et al. — Latent Diffusion / Stable Diffusion. Diffusion in compressed latent space → practical high-resolution generation.
- **2022–present:** DALL-E 2, Imagen, SDXL, Sora, Flux. Diffusion becomes the backbone of generative AI.

### Section 1.5: What's Ahead in This Course
- Roadmap preview: math → architecture → training → sampling → conditioning → scaling → modern directions
- What you'll build: a working MNIST diffusion model with classifier-free guidance
- The payoff: by the end, you can read and understand any diffusion paper

## Key Equations
- None in this lecture — it's conceptual. The equations come in Lecture 2.

## Code Examples
- No code in this lecture. It sets the conceptual stage.
- Optional: a teaser visualization — "here's what our model will generate by the end of Module 3" showing a grid of MNIST digits from a trained model.

## Diagrams & Visuals
- **Generative models comparison table:** rows = {GANs, VAEs, Flows, Autoregressive, Diffusion}, columns = {Training stability, Sample quality, Likelihood, Speed, Architecture flexibility}
- **The diffusion intuition diagram:** clean image → progressively noisier → pure noise (forward), then reverse arrow with "learned denoiser" label
- **Historical timeline:** visual timeline from 2015 to present with key papers marked

## Source Material
- Module 05 §5.1 (Generative Models Landscape)
- Historical context from paper citations across all module specs
