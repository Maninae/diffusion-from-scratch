"""Dataset loading utilities for diffusion-from-scratch notebooks."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(
    batch_size: int = 64,
    image_size: int = 28,
    root: str = "./data",
    num_workers: int = 0,
    normalize_to_minus1_plus1: bool = True,
) -> DataLoader:
    """Load MNIST with appropriate transforms for diffusion training.

    Images are returned as single-channel tensors in [-1, 1].

    Args:
        batch_size: Batch size.
        image_size: Resize images to this size.
        root: Data directory.
        num_workers: Number of dataloader workers.
        normalize_to_minus1_plus1: If True, normalize to [-1, 1]; else [0, 1].

    Returns:
        DataLoader yielding (images, labels).
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),  # [0, 1]
    ]
    if normalize_to_minus1_plus1:
        transform_list.append(transforms.Normalize([0.5], [0.5]))  # [-1, 1]

    dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose(transform_list),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_cifar10_dataloader(
    batch_size: int = 64,
    image_size: int = 32,
    root: str = "./data",
    num_workers: int = 0,
    normalize_to_minus1_plus1: bool = True,
    flip: bool = True,
) -> DataLoader:
    """Load CIFAR-10 with appropriate transforms for diffusion training.

    Images are returned as 3-channel tensors in [-1, 1].

    Args:
        batch_size: Batch size.
        image_size: Resize images to this size.
        root: Data directory.
        num_workers: Number of dataloader workers.
        normalize_to_minus1_plus1: If True, normalize to [-1, 1]; else [0, 1].
        flip: Apply random horizontal flip augmentation.

    Returns:
        DataLoader yielding (images, labels).
    """
    transform_list = []
    if image_size != 32:
        transform_list.append(transforms.Resize(image_size))
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())  # [0, 1]
    if normalize_to_minus1_plus1:
        transform_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))  # [-1, 1]

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose(transform_list),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_device() -> torch.device:
    """Auto-detect the best available device (MPS > CUDA > CPU).

    Returns:
        torch.device for the best available hardware accelerator.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
