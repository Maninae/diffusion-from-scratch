"""Shared visualization helpers for diffusion-from-scratch notebooks."""

from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def set_style() -> None:
    """Apply a clean, consistent matplotlib style across all notebooks."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] back to [0, 1] for display.

    Args:
        images: Tensor with values in [-1, 1].

    Returns:
        Tensor clamped to [0, 1].
    """
    return (images * 0.5 + 0.5).clamp(0, 1)


def show_images(
    images: torch.Tensor,
    nrow: int = 8,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """Display a batch of images as a grid.

    Args:
        images: (B, C, H, W) tensor in [-1, 1] or [0, 1].
        nrow: Number of images per row in the grid.
        title: Optional title for the plot.
        figsize: Optional (width, height) for the figure.
    """
    # Auto-denormalize if values look like [-1, 1]
    if images.min() < -0.1:
        images = denormalize(images)
    images = images.detach().cpu()

    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    if figsize is None:
        h = max(2, images.shape[0] // nrow + 1)
        figsize = (min(nrow, images.shape[0]) * 1.5, h * 1.5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid_np, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_denoising_trajectory(
    trajectory: List[torch.Tensor],
    timesteps: Optional[List[int]] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """Show a row of images from a denoising trajectory.

    Args:
        trajectory: List of image tensors (C, H, W) or (1, C, H, W) at selected steps.
        timesteps: Optional list of timestep labels for each image.
        figsize: Optional figure size.
    """
    n = len(trajectory)
    if figsize is None:
        figsize = (n * 2, 2.5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, img in enumerate(trajectory):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img[0]
        if img.min() < -0.1:
            img = (img * 0.5 + 0.5).clamp(0, 1)
        axes[i].imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
        axes[i].axis("off")
        if timesteps is not None:
            axes[i].set_title(f"t={timesteps[i]}", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_schedule(
    values_dict: dict,
    title: str = "Noise Schedule",
    xlabel: str = "Timestep t",
    ylabel: str = "Value",
    figsize: tuple = (8, 4),
) -> None:
    """Plot one or more schedule curves.

    Args:
        values_dict: Mapping of label → 1-D array/tensor of values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for label, vals in values_dict.items():
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        ax.plot(vals, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_curve(
    losses: list,
    window: int = 50,
    title: str = "Training Loss",
    figsize: tuple = (8, 4),
) -> None:
    """Plot a training loss curve with optional smoothing.

    Args:
        losses: List of per-step loss values.
        window: Moving-average window size for smoothing.
        title: Plot title.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(losses, alpha=0.3, color="steelblue", label="Raw")
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(losses)),
            smoothed,
            color="steelblue",
            label=f"Smoothed (w={window})",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
