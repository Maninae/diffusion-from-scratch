# Diffusion From Scratch — Diffusion Models from Scratch

## Context

**Role:** Senior MLE  
**Referral:** [removed]  
**Recruiter:** the recruiter ([removed])  
**Coordinator:** [removed] / [removed]  
**HQ:** [removed]

### Interview Timeline
- Feb 27 — Recruiter screen with the recruiter
- Mar 4 — Hiring Manager screen with [removed] (Senior EM) ✅ Went well
- Mar 5 — Michael confirmed next step: **45-min Technical Phone Screen (live coding)**
- [removed] emailed to schedule the tech screen

### What the Interview Is
From Michael's email (Mar 5, 2026):
> - **The Topic:** This will be an image diffusion model-based coding question.
> - **The Format:** This is NOT a LeetCode-style algorithmic puzzle. We are interested in your practical application of Python and PyTorch fundamentals.
> - **The Goal:** The interviewer wants to see how you structure your code and hear your thought process as you build.

### What We're Building
A comprehensive Jupyter notebook curriculum — essentially a mini-course — covering diffusion models from absolute fundamentals through advanced topics, with emphasis on:
1. **NumPy vectorization** (no sloppy loops)
2. **PyTorch fluency** (custom modules, autograd, training loops)
3. **Diffusion model implementation from scratch** (forward process, U-Net, training, sampling)
4. **Practical interview readiness** (timed exercises, verbal explanation practice)

**Location:** `~/Developer/diffusion-prep/`  
**Estimated study time:** 15-20 hours if done thoroughly  
**Build method:** Delegate to a coding agent (Claude Code / Codex) using this spec

---

## Report Files

| File | Contents |
|------|----------|
| `README.md` | This file — overview, context, index |
| `module-00-numpy.md` | Module 0: Python & NumPy Foundations |
| `module-01-pytorch.md` | Module 1: PyTorch Fundamentals |
| `module-02-convolutions.md` | Module 2: Convolutions & Image Processing |
| `module-03-attention.md` | Module 3: Attention & Transformers |
| `module-04-unet.md` | Module 4: The U-Net Architecture |
| `module-05-math.md` | Module 5: The Math of Diffusion |
| `module-06-training.md` | Module 6: Training a Diffusion Model |
| `module-07-sampling.md` | Module 7: Sampling & Inference |
| `module-08-conditioning.md` | Module 8: Conditioning & Guidance |
| `module-09-advanced.md` | Module 9: Latent Diffusion & Advanced Topics |
| `module-10-interview-sim.md` | Module 10: Interview Simulation |
| `papers.md` | Complete paper reference list with arxiv links |
| `build-instructions.md` | Instructions for the coding agent building the notebooks |

---

## Pedagogical Structure (Every Module)

Each module follows this pattern:

1. **📖 Concept Introduction** — Markdown cells with math (LaTeX), diagrams (described for generation), and intuition. Focus on *why* before *how*.
2. **📄 Key Papers** — Callout box with arxiv links and a one-liner on what to look for when reading.
3. **🔍 Worked Example** — Code walkthrough with detailed commentary. The student reads and runs, not writes.
4. **✏️ Exercises** — Increasing difficulty. Some are timed for interview simulation. Clear problem statement, expected output described.
5. **✅ Solutions** — In collapsed/hidden cells. Complete, clean, well-commented.

---

## Key Design Principles

- **Pedagogy first.** Every new concept gets a proper introduction before any code. Don't assume knowledge — teach it.
- **From scratch.** Implement everything manually before using library shortcuts. The student should understand what `nn.Conv2d` does because they built it.
- **Vectorized always.** No Python loops where NumPy/PyTorch broadcasting works. Call out anti-patterns explicitly.
- **Interview-realistic.** The timed exercises in Module 10 mirror the actual 45-minute format. Practical, not theoretical.
- **Paper-linked.** Every major concept links back to its foundational paper on arxiv with a note on which section to read.
- **Progressive complexity.** Module 0 assumes solid Python but rusty NumPy. By Module 10, you're implementing diffusion models under time pressure.
