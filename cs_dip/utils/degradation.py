"""
Image Degradation Operators
===========================

Provides degradation functions for simulating inverse problem settings:
- Gaussian noise addition (denoising)
- Bicubic downsampling (super-resolution)
- Factory function for task-based operator selection
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Add Gaussian noise to an image tensor.

    Args:
        image: Clean image tensor in ``[0, 1]``, shape ``(B, C, H, W)``
            or ``(C, H, W)``.
        sigma: Noise standard deviation on the ``[0, 255]`` scale. The
            noise is normalized to ``[0, 1]`` before addition.
        seed: Optional random seed for reproducibility.

    Returns:
        Noisy image tensor clamped to ``[0, 1]``.
    """
    if seed is not None:
        rng = torch.Generator(device=image.device)
        rng.manual_seed(seed)
        noise = torch.randn(image.shape, generator=rng, device=image.device)
    else:
        noise = torch.randn_like(image)

    sigma_normalized = sigma / 255.0
    noisy = image + sigma_normalized * noise
    return noisy.clamp(0.0, 1.0)


def bicubic_downsample(
    image: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """Downsample an image using bicubic interpolation.

    Args:
        image: Input image tensor, shape ``(B, C, H, W)``.
        scale_factor: Integer downsampling factor (e.g., 2 or 4).

    Returns:
        Downsampled image tensor.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    h, w = image.shape[2], image.shape[3]
    new_h, new_w = h // scale_factor, w // scale_factor
    return F.interpolate(
        image,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)


def bicubic_upsample(
    image: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """Upsample an image using bicubic interpolation.

    Args:
        image: Input image tensor, shape ``(B, C, H, W)``.
        scale_factor: Integer upsampling factor (e.g., 2 or 4).

    Returns:
        Upsampled image tensor.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return F.interpolate(
        image,
        scale_factor=float(scale_factor),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)


def get_degradation_operator(
    task: str,
    **params,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Factory function to create a forward degradation operator.

    For denoising, the degradation is identity (noise is added separately).
    For super-resolution, the degradation is bicubic downsampling.

    Args:
        task: Task type, one of ``{'denoise', 'sr'}``.
        **params: Task-specific parameters:
            - ``scale_factor`` (int): For ``'sr'`` task.

    Returns:
        Callable degradation function, or ``None`` for identity degradation.

    Raises:
        ValueError: If task is not recognized.
    """
    if task == "denoise":
        return None  # Identity — noise is pre-applied to target
    elif task == "sr":
        scale = params.get("scale_factor", 2)
        return lambda img: bicubic_downsample(img, scale)
    else:
        raise ValueError(f"Unknown task '{task}'. Supported: 'denoise', 'sr'.")
