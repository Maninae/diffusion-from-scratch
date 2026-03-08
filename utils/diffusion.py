"""Core diffusion operations shared across notebooks.

These functions are first implemented from scratch in Modules 5-6 as learning
exercises, then imported here for reuse in Modules 7-10.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: sample x_t given x_0 and timestep t.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Args:
        x_0: Clean images, shape (B, C, H, W).
        t: Timesteps, shape (B,).
        schedule: Dict of precomputed schedule tensors (must be on same device as x_0).
        noise: Optional pre-generated noise; sampled if None.

    Returns:
        (x_t, noise) — noisy images and the noise that was added, both (B, C, H, W).
    """
    if noise is None:
        noise = torch.randn_like(x_0)  # (B, C, H, W)

    sqrt_alpha_bar = schedule["sqrt_alphas_cumprod"][t]  # (B,)
    sqrt_one_minus = schedule["sqrt_one_minus_alphas_cumprod"][t]  # (B,)

    # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
    sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None]
    sqrt_one_minus = sqrt_one_minus[:, None, None, None]

    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise  # (B, C, H, W)
    return x_t, noise


def prepare_schedule(
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Move all schedule tensors to the specified device.

    Args:
        schedule: Dict of 1-D schedule tensors (on CPU or any device).
        device: Target device.

    Returns:
        New dict with all tensors on the target device.
    """
    return {k: v.to(device) for k, v in schedule.items()}


def train_step(
    model: nn.Module,
    x_0: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    schedule: Dict[str, torch.Tensor],
    num_timesteps: int = 1000,
    max_grad_norm: float = 1.0,
) -> float:
    """One training step of DDPM (Algorithm 1 from Ho et al. 2020).

    Args:
        model: Noise prediction network. Forward signature: model(x_t, t) -> noise_pred.
        x_0: Clean images, shape (B, C, H, W), already on the correct device.
        optimizer: Optimizer for model parameters.
        schedule: Dict of precomputed schedule tensors (already on correct device).
        num_timesteps: Total number of diffusion timesteps T.
        max_grad_norm: Max norm for gradient clipping.

    Returns:
        Scalar loss value.
    """
    model.train()
    device = x_0.device
    B = x_0.shape[0]

    # 1. Sample random timesteps
    t = torch.randint(0, num_timesteps, (B,), device=device)  # (B,)

    # 2. Forward diffusion
    x_t, noise = q_sample(x_0, t, schedule)  # (B, C, H, W) each

    # 3. Predict noise
    noise_pred = model(x_t, t)  # (B, C, H, W)

    # 4. Compute loss
    loss = F.mse_loss(noise_pred, noise)

    # 5. Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: Dict[str, torch.Tensor],
    shape: Tuple[int, ...],
    device: torch.device,
    clip: bool = True,
) -> torch.Tensor:
    """Generate images using DDPM reverse process (Algorithm 2 from Ho et al. 2020).

    Args:
        model: Trained noise prediction network.
        schedule: Dict of precomputed schedule tensors (on device).
        shape: Shape of images to generate, e.g. (16, 1, 28, 28).
        device: Device to generate on.
        clip: Whether to clamp final output to [-1, 1].

    Returns:
        Generated images, shape = `shape`.
    """
    model.eval()
    T = len(schedule["betas"])

    x_t = torch.randn(shape, device=device)  # Start from pure noise

    for t_val in reversed(range(T)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

        beta_t = schedule["betas"][t_val]
        alpha_t = schedule["alphas"][t_val]
        alpha_bar_t = schedule["alphas_cumprod"][t_val]
        sqrt_recip_alpha = schedule["sqrt_recip_alphas"][t_val]

        noise_pred = model(x_t, t)

        # Predicted mean
        mean = sqrt_recip_alpha * (
            x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred
        )

        if t_val > 0:
            posterior_var = schedule["posterior_variance"][t_val]
            z = torch.randn_like(x_t)
            x_t = mean + posterior_var.sqrt() * z
        else:
            x_t = mean

    if clip:
        x_t = x_t.clamp(-1, 1)
    return x_t
