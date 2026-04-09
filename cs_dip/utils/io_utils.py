"""
I/O Utilities
=============

Helper functions for image loading/saving, random seed management,
and noise input generation for the CS-DIP pipeline.
"""

import os
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_image(path: str, as_gray: bool = False) -> torch.Tensor:
    """Load an image file as a float32 tensor in [0, 1].

    Args:
        path: Path to the image file.
        as_gray: If ``True``, convert to single-channel grayscale.

    Returns:
        Image tensor of shape ``(1, C, H, W)`` where C is 1 or 3.
    """
    img = Image.open(path)
    if as_gray:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img_np = np.array(img).astype(np.float32) / 255.0

    if img_np.ndim == 2:
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)  # (1, C, H, W)

    return tensor


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor as a PNG image.

    Args:
        tensor: Image tensor of shape ``(B, C, H, W)`` or ``(C, H, W)``,
            values in ``[0, 1]``.
        path: Output file path.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch element
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)

    if tensor.shape[0] == 1:
        img_np = (tensor[0].numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(img_np, mode="L")
    else:
        img_np = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img = Image.fromarray(img_np, mode="RGB")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    img.save(path)


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Sets seeds for ``random``, ``numpy``, ``torch`` (CPU and CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_noise_input(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a fixed random noise input tensor ``z``.

    This noise tensor is kept fixed throughout optimization — only the
    network weights are updated.

    Args:
        batch_size: Batch size (typically 1).
        channels: Number of noise channels.
        height: Spatial height matching the output image.
        width: Spatial width matching the output image.
        seed: Random seed for noise generation. If ``None``, uses
            the current random state.
        device: Target device (``'cpu'`` or ``'cuda'``).

    Returns:
        Noise tensor of shape ``(B, C, H, W)``.
    """
    if seed is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        z = torch.randn(batch_size, channels, height, width, generator=rng)
    else:
        z = torch.randn(batch_size, channels, height, width)
    return z.to(device)
