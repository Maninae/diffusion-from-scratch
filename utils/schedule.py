"""Noise schedule implementations for diffusion models.

All schedules return a dictionary of pre-computed tensors:
    betas          — (T,) noise variances
    alphas         — (T,) = 1 - betas
    alphas_cumprod — (T,) = cumulative product of alphas  (ᾱ_t)
    sqrt_alphas_cumprod          — √ᾱ_t
    sqrt_one_minus_alphas_cumprod — √(1 - ᾱ_t)
    sqrt_recip_alphas            — 1/√α_t
    posterior_variance           — β̃_t for DDPM sampling
"""

from typing import Dict

import torch


def _build_schedule_dict(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Given a beta schedule, derive and return all related quantities.

    Args:
        betas: (T,) tensor of noise variances in (0, 1).

    Returns:
        Dictionary mapping name → pre-computed tensor.
    """
    betas = betas.clamp(min=1e-5, max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": 1.0 / torch.sqrt(alphas),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "snr": alphas_cumprod / (1.0 - alphas_cumprod),
    }


def linear_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Linear noise schedule (original DDPM).

    Args:
        T: Number of diffusion timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        Schedule dictionary.
    """
    betas = torch.linspace(beta_start, beta_end, T)
    return _build_schedule_dict(betas)


def cosine_schedule(
    T: int = 1000,
    s: float = 0.008,
) -> Dict[str, torch.Tensor]:
    """Cosine noise schedule (Improved DDPM, Nichol & Dhariwal 2021).

    Reference: https://arxiv.org/abs/2102.09672 Section 3.2

    Args:
        T: Number of diffusion timesteps.
        s: Small offset to prevent β_t from being too small near t=0.

    Returns:
        Schedule dictionary.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / T) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(max=0.999).float()
    return _build_schedule_dict(betas)


def sigmoid_schedule(
    T: int = 1000,
    start: float = -3.0,
    end: float = 3.0,
) -> Dict[str, torch.Tensor]:
    """Sigmoid noise schedule — smooth transition between extremes.

    Args:
        T: Number of diffusion timesteps.
        start: Sigmoid input at t=0 (more negative → smaller beta).
        end: Sigmoid input at t=T (more positive → larger beta).

    Returns:
        Schedule dictionary.
    """
    t = torch.linspace(start, end, T)
    betas = torch.sigmoid(t)
    # Scale to a reasonable range
    betas = betas * (0.02 - 1e-4) + 1e-4
    return _build_schedule_dict(betas)


def get_schedule(
    name: str = "cosine",
    T: int = 1000,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Convenience function to get a schedule by name.

    Args:
        name: One of 'linear', 'cosine', 'sigmoid'.
        T: Number of timesteps.
        **kwargs: Extra arguments forwarded to the schedule function.

    Returns:
        Schedule dictionary.
    """
    schedules = {
        "linear": linear_schedule,
        "cosine": cosine_schedule,
        "sigmoid": sigmoid_schedule,
    }
    if name not in schedules:
        raise ValueError(f"Unknown schedule '{name}'. Choose from {list(schedules.keys())}")
    return schedules[name](T=T, **kwargs)
