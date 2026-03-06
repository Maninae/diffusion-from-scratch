# Papers Reference — Complete Arxiv List

Every paper referenced across the notebook, organized by topic. Each entry includes the arxiv link, what to read it for, and which module references it.

---

## 🔵 Core Diffusion Papers

### Denoising Diffusion Probabilistic Models (DDPM)
- **Authors:** Ho, Jain, Abbeel (2020)
- **Link:** [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- **Read for:** THE foundational paper. Algorithm 1 (training) and Algorithm 2 (sampling) are the key references. Section 2 for forward process, Section 3 for reverse process and loss derivation.
- **Referenced in:** Modules 4, 5, 6, 7

### Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- **Authors:** Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015)
- **Link:** [arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585)
- **Read for:** The original diffusion idea. More theoretical/physics-oriented. Historical context for where DDPM came from.
- **Referenced in:** Module 5

### Improved Denoising Diffusion Probabilistic Models
- **Authors:** Nichol & Dhariwal (2021)
- **Link:** [arxiv.org/abs/2102.09672](https://arxiv.org/abs/2102.09672)
- **Read for:** Cosine noise schedule (Section 3.2), learned variance, importance sampling of timesteps. Practical improvements that matter.
- **Referenced in:** Modules 5, 6

### Denoising Diffusion Implicit Models (DDIM)
- **Authors:** Song, Meng, Ermon (2020)
- **Link:** [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
- **Read for:** Deterministic sampling, the η parameter, DDIM update rule (Equation 12). Enables fewer sampling steps and latent space interpolation.
- **Referenced in:** Module 7

### Score-Based Generative Modeling through Stochastic Differential Equations
- **Authors:** Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021)
- **Link:** [arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)
- **Read for:** Unified framework connecting DDPM and score matching through SDEs. Section 3 for forward/reverse SDE pair. Elegant but more mathematical.
- **Referenced in:** Module 5

### Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation
- **Authors:** Kingma, Gao (2023)
- **Link:** [arxiv.org/abs/2303.00848](https://arxiv.org/abs/2303.00848)
- **Read for:** Modern understanding of the diffusion objective. SNR perspective on the loss.
- **Referenced in:** Module 5

### Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
- **Authors:** Karras, Aittala, Aila, Laine (2022)
- **Link:** [arxiv.org/abs/2206.00364](https://arxiv.org/abs/2206.00364)
- **Read for:** Careful analysis of noise schedules, network preconditioning, and sampling. Excellent practical guidance for building diffusion models.
- **Referenced in:** Module 9

---

## 🟢 Guidance & Conditioning Papers

### Classifier-Free Diffusion Guidance
- **Authors:** Ho & Salimans (2022)
- **Link:** [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
- **Read for:** THE key paper for CFG. Section 2 for the formulation. The random label dropout training trick. The guidance scale formula. Short and essential.
- **Referenced in:** Module 8

### Diffusion Models Beat GANs on Image Synthesis (ADM)
- **Authors:** Dhariwal & Nichol (2021)
- **Link:** [arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
- **Read for:** Classifier guidance (Section 4), architecture improvements (Adaptive GroupNorm, attention at multiple resolutions, BigGAN-style residual blocks). Also proves diffusion > GANs.
- **Referenced in:** Modules 4, 8

### Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)
- **Authors:** Saharia et al. (2022)
- **Link:** [arxiv.org/abs/2205.11487](https://arxiv.org/abs/2205.11487)
- **Read for:** Text-to-image with CFG. Dynamic thresholding. Shows how guidance scale affects quality. Uses T5 text encoder.
- **Referenced in:** Module 8

---

## 🟡 Architecture Papers

### U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Authors:** Ronneberger, Fischer, Brox (2015)
- **Link:** [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
- **Read for:** The original U-Net. Encoder-decoder + skip connections. Figure 1 is iconic. The architecture that diffusion models adapted.
- **Referenced in:** Module 4

### Attention Is All You Need
- **Authors:** Vaswani et al. (2017)
- **Link:** [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Read for:** The original transformer. Section 3.2 for scaled dot-product attention, Section 3.2.2 for multi-head attention. The 1/√d_k scaling explanation.
- **Referenced in:** Module 3

### An Image is Worth 16x16 Words (ViT)
- **Authors:** Dosovitskiy et al. (2020)
- **Link:** [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- **Read for:** How attention applies to images — patch embedding, positional encoding for 2D. Section 3.
- **Referenced in:** Module 3

### Deep Residual Learning for Image Recognition (ResNet)
- **Authors:** He, Zhang, Ren, Sun (2015)
- **Link:** [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- **Read for:** Residual connections and why they enable training very deep networks. Section 3 for the key insight about identity mappings.
- **Referenced in:** Module 2

### Group Normalization
- **Authors:** Wu & He (2018)
- **Link:** [arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
- **Read for:** Why GroupNorm works better than BatchNorm for small batches. Table 1 for the comparison. Directly relevant — diffusion models use GroupNorm.
- **Referenced in:** Module 2

### Scalable Diffusion Models with Transformers (DiT)
- **Authors:** Peebles & Xie (2023)
- **Link:** [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
- **Read for:** Replacing U-Net with a transformer. Patch embedding + transformer blocks + adaLN conditioning. Table 1 for scaling results. The direction the field is moving.
- **Referenced in:** Module 9

---

## 🟣 Latent Diffusion & VAE Papers

### High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)
- **Authors:** Rombach, Blattmann, Lorenz, Esser, Ommer (2022)
- **Link:** [arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- **Read for:** THE latent diffusion paper. Section 3 for VAE + diffusion in latent space. Figure 1 for the architecture. Why it's 10-100× more efficient.
- **Referenced in:** Module 9

### Auto-Encoding Variational Bayes (VAE)
- **Authors:** Kingma & Welling (2013)
- **Link:** [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- **Read for:** The original VAE paper. Sections 2-3 for ELBO, encoder-decoder, reparameterization trick. Foundation for latent diffusion.
- **Referenced in:** Module 9

---

## 🔴 Modern Directions Papers

### Flow Matching for Generative Modeling
- **Authors:** Lipman, Chen, Ben-Hamu, Nickel (2023)
- **Link:** [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- **Read for:** The flow matching alternative to diffusion. Simpler training objective, straight paths from noise to data. Section 3 for the key formulation.
- **Referenced in:** Module 9

### Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
- **Authors:** Liu, Gong, Liu (2023)
- **Link:** [arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
- **Read for:** Rectified flows — straighter trajectories mean fewer sampling steps. Used in Stable Diffusion 3. Section 3 for rectification.
- **Referenced in:** Module 9

### Progressive Distillation for Fast Sampling of Diffusion Models
- **Authors:** Salimans & Ho (2022)
- **Link:** [arxiv.org/abs/2202.00512](https://arxiv.org/abs/2202.00512)
- **Read for:** Reducing sampling steps through distillation. Shows 4-8 step generation is possible.
- **Referenced in:** Module 7

---

## 🟠 Foundational ML Papers (Background)

### PyTorch: An Imperative Style, High-Performance Deep Learning Library
- **Authors:** Paszke et al. (2019)
- **Link:** [arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)
- **Read for:** PyTorch design philosophy, autograd internals. Good background reading.
- **Referenced in:** Module 1

---

## Reading Order Recommendation

If time is limited, read in this order:

1. **DDPM** (Ho et al. 2020) — the foundation, read fully
2. **Classifier-Free Guidance** (Ho & Salimans 2022) — short and essential
3. **DDIM** (Song et al. 2020) — for sampling
4. **Improved DDPM** (Nichol & Dhariwal 2021) — practical improvements
5. **Latent Diffusion** (Rombach et al. 2022) — for Stable Diffusion understanding
6. **DiT** (Peebles & Xie 2023) — modern direction
7. **Flow Matching** (Lipman et al. 2023) — emerging alternative

Everything else is supplementary / deepening understanding.
