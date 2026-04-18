# Build Instructions

Spec for building the Jupyter notebook curriculum.

---

## Context

**Goal:** A comprehensive, pedagogical Jupyter notebook curriculum covering diffusion models from scratch.

---

## Project Structure

```
├── requirements.txt
├── README.md
├── module_00_numpy_foundations.ipynb
├── module_01_pytorch_fundamentals.ipynb
├── module_02_convolutions.ipynb
├── module_03_attention.ipynb
├── module_04_unet.ipynb
├── module_05_diffusion_math.ipynb
├── module_06_training.ipynb
├── module_07_sampling.ipynb
├── module_08_conditioning.ipynb
├── module_09_advanced.ipynb
├── module_10_interview_sim.ipynb
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Shared plotting helpers
│   ├── data.py               # Dataset loading utilities
│   └── schedule.py           # Noise schedule implementations
├── checkpoints/              # Saved model checkpoints (gitignored)
├── data/                     # Downloaded datasets (gitignored)
└── assets/                   # Any images/diagrams
```

---

## Requirements

```
torch>=2.0
torchvision
numpy
matplotlib
jupyter
tqdm
einops
```

**Target hardware:** Apple Silicon (MPS) or CPU. No CUDA required. Training should be feasible on CPU/MPS for toy models.

---

## Notebook Standards

### Cell Structure

Each module follows this pattern:

1. **Title cell (markdown):** Module name, learning objectives, estimated time
2. **Key Papers cell (markdown):** Callout box with arxiv links and one-liner descriptions
3. **For each section:**
   a. **Concept introduction (markdown):** Explain the concept with LaTeX math, diagrams (use matplotlib to generate inline), intuition. Focus on WHY before HOW.
   b. **Worked example (code):** Complete, runnable implementation with detailed inline comments. The student reads and runs, doesn't write.
   c. **Exercise (markdown):** Clear problem statement, expected output described, time estimate if applicable.
   d. **Solution (code):** In a cell marked with `# SOLUTION — try the exercise above before running this`. Use cell metadata or a comment-based collapse hint.
4. **Capstone exercise** at the end of each module

### Code Style

- **Clean, readable Python.** No clever one-liners that sacrifice readability.
- **Meaningful variable names:** `alpha_bar_t` not `ab`, `noise_pred` not `np` (also conflicts with numpy).
- **Type hints** on function signatures.
- **Docstrings** on all functions.
- **Shape comments** after tensor operations: `# (B, C, H, W)`
- **Device-agnostic:** Use a `device` variable set at the top of each notebook: `device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')`
- **Reproducibility:** Set seeds at the top of each notebook: `torch.manual_seed(42)`

### Math Rendering

- Use LaTeX in markdown cells: `$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$`
- For important equations, use display math: `$$`
- Always define notation before using it

### Visualization

- Use matplotlib for all plots
- Consistent style: `plt.style.use('seaborn-v0_8-whitegrid')` or similar clean style
- Always label axes
- For image grids: use `torchvision.utils.make_grid` or a custom grid function
- For denoising trajectories: show progression as a row of images at key timesteps
- Color-blind friendly palettes where possible

---

## Module-by-Module Build Instructions

**For each module, read the corresponding `module-XX-*.md` file in this directory.** That file contains:
- Every section and subsection
- What concepts to teach in each section
- What the worked example should demonstrate
- What exercises to include
- What the capstone exercise should be

**The markdown files are the authoritative spec.** Follow them closely. Don't skip sections. Don't reduce the depth.

### Build Order

Build in order (later modules depend on earlier ones):
1. `utils/` — shared utilities first
2. Module 0 — no dependencies
3. Module 1 — no dependencies
4. Module 2 — uses PyTorch from Module 1
5. Module 3 — uses PyTorch from Module 1
6. Module 4 — uses Modules 2 + 3
7. Module 5 — uses Module 0 (NumPy) + basic PyTorch
8. Module 6 — uses Modules 4 + 5
9. Module 7 — uses Module 6 (trained model)
10. Module 8 — uses Modules 6 + 7
11. Module 9 — uses all previous
12. Module 10 — synthesis of everything

### Cross-Module Dependencies

- **Module 4 U-Net** should be importable as a class for use in Modules 6-10
- **Module 5 noise schedule** should be in `utils/schedule.py` for reuse
- **Module 6 trained model checkpoints** should be saved for use in Modules 7-8
- Put reusable code in `utils/` to avoid massive code duplication

---

## Quality Checklist

Before considering a module complete:

- [ ] All cells run top-to-bottom without errors
- [ ] All exercises have solutions
- [ ] All concepts from the spec are covered (check against module-XX-*.md)
- [ ] LaTeX renders correctly
- [ ] Visualizations are clear and labeled
- [ ] Code follows the style guide above
- [ ] Shape comments on tensor operations
- [ ] Paper references include working arxiv links
- [ ] Training loops are feasible on CPU/MPS (adjust dataset size / epochs if needed)
- [ ] No hardcoded CUDA assumptions (use `device` variable)

---

## Paper References

See `papers.md` in this directory for the complete list with arxiv links. Each module spec indicates which papers to reference and where.
