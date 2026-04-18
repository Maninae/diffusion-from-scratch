# Docs — Diffusion From Scratch

## Overview

Build specs and content architecture for the course website. These files define the content, structure, and pedagogical approach for each module and lecture.

---

## File Index

| File | Contents |
|------|----------|
| `README.md` | This file — docs overview |
| `content-architecture.md` | Site structure: 6 modules, 13 lectures, assessment track |
| `build-instructions.md` | Notebook coding standards and build order |
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
| `module-10-interview-sim.md` | Module 10: Assessment & Coding Challenges |
| `papers.md` | Complete paper reference list with arxiv links |
| `lectures/` | Per-lecture content specs for the course website |

---

## Pedagogical Structure (Every Module)

Each module follows this pattern:

1. **Concept Introduction** — Markdown cells with math (LaTeX), diagrams, and intuition. Focus on *why* before *how*.
2. **Key Papers** — Callout box with arxiv links and a one-liner on what to look for when reading.
3. **Worked Example** — Code walkthrough with detailed commentary. The student reads and runs, not writes.
4. **Exercises** — Increasing difficulty. Clear problem statement, expected output described.
5. **Solutions** — In collapsed/hidden cells. Complete, clean, well-commented.

---

## Key Design Principles

- **Pedagogy first.** Every new concept gets a proper introduction before any code. Don't assume knowledge — teach it.
- **From scratch.** Implement everything manually before using library shortcuts. The student should understand what `nn.Conv2d` does because they built it.
- **Vectorized always.** No Python loops where NumPy/PyTorch broadcasting works. Call out anti-patterns explicitly.
- **Paper-linked.** Every major concept links back to its foundational paper on arxiv with a note on which section to read.
- **Progressive complexity.** Module 0 assumes solid Python but rusty NumPy. By the assessment track, you're implementing diffusion models under time pressure.
