"""
CFG Guidance Scale Comparison Grid

Generates images at different classifier-free guidance scales using SD-Turbo,
stitches them into a labeled horizontal grid.

Requirements:
    pip install diffusers torch accelerate transformers pillow

Usage:
    python demos/cfg_comparison.py
"""

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "stabilityai/sd-turbo"
PROMPT = "a photograph of a cat sitting in sunlight"
GUIDANCE_SCALES = [0.1, 0.5, 1.0, 2.0, 4.0]
NUM_STEPS = 25
SEED = 42
DEVICE = "mps"
DTYPE = torch.float16
OUTPUT_DIR = "demos"

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
print(f"Loading {MODEL_ID} ...")
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    variant="fp16",
)
pipe = pipe.to(DEVICE)
print("Pipeline ready.\n")

# ---------------------------------------------------------------------------
# Generate images at each guidance scale
# ---------------------------------------------------------------------------
images: list[Image.Image] = []

for scale in GUIDANCE_SCALES:
    print(f"Generating with guidance_scale={scale} ...")
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        guidance_scale=scale,
        num_inference_steps=NUM_STEPS,
        generator=generator,
    )
    img = result.images[0]

    # Save individual image
    fname = f"{OUTPUT_DIR}/cfg_s{scale}.png"
    img.save(fname)
    print(f"  Saved {fname}")
    images.append(img)

# ---------------------------------------------------------------------------
# Stitch into a labeled horizontal grid
# ---------------------------------------------------------------------------
print("\nStitching grid ...")

LABEL_HEIGHT = 40
PADDING = 8
img_w, img_h = images[0].size
n = len(images)

grid_w = n * img_w + (n - 1) * PADDING
grid_h = img_h + LABEL_HEIGHT

grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
draw = ImageDraw.Draw(grid)

# Try to load a reasonable font; fall back to default
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except OSError:
    font = ImageFont.load_default()

for i, (img, scale) in enumerate(zip(images, GUIDANCE_SCALES)):
    x_offset = i * (img_w + PADDING)
    grid.paste(img, (x_offset, 0))

    label = f"w = {scale}"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = x_offset + (img_w - text_w) // 2
    text_y = img_h + (LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

grid_path = f"{OUTPUT_DIR}/cfg_comparison.png"
grid.save(grid_path)
print(f"Grid saved to {grid_path}")
print("Done.")
