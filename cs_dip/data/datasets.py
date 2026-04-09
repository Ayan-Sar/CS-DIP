"""
Benchmark Dataset Loaders
=========================

Provides a unified ``BenchmarkDataset`` class for loading standard image
restoration benchmarks: Set5, Set14, BSD68, and Urban100.

Images are loaded as PyTorch tensors in [0, 1] range. The loader
supports both RGB and luminance-only (Y-channel) evaluation modes.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image (H, W, 3) in [0, 255] to YCbCr.

    Returns:
        YCbCr image array of shape ``(H, W, 3)`` in float64.
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 16.0 + 65.481 * r / 255.0 + 128.553 * g / 255.0 + 24.966 * b / 255.0
    cb = 128.0 - 37.797 * r / 255.0 - 74.203 * g / 255.0 + 112.0 * b / 255.0
    cr = 128.0 + 112.0 * r / 255.0 - 93.786 * g / 255.0 - 18.214 * b / 255.0
    return np.stack([y, cb, cr], axis=-1)


class BenchmarkDataset(Dataset):
    """Dataset loader for standard image restoration benchmarks.

    Supports Set5, Set14, BSD68, and Urban100. Images are loaded from
    the file system and returned as float32 tensors normalized to [0, 1].

    Args:
        root_dir: Root directory containing dataset folders.
        dataset_name: Name of the benchmark dataset. One of
            ``{'Set5', 'Set14', 'BSD68', 'Urban100'}``.
        y_channel_only: If ``True``, convert to YCbCr and return only the
            luminance (Y) channel. Default: ``False`` (returns RGB).
        crop_size: If provided, center-crop images to this size.
            Useful for ensuring consistent tensor shapes.
    """

    SUPPORTED_DATASETS = {"Set5", "Set14", "BSD68", "Urban100"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        y_channel_only: bool = False,
        crop_size: Optional[int] = None,
    ):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )

        self.dataset_name = dataset_name
        self.y_channel_only = y_channel_only
        self.crop_size = crop_size

        dataset_dir = Path(root_dir) / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}. "
                f"Please download the dataset first. See data/README.md."
            )

        # Collect image paths
        self.image_paths = sorted([
            str(p)
            for p in dataset_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        ])

        if not self.image_paths:
            # Check subdirectories (e.g., HR/ folder)
            for subdir in dataset_dir.iterdir():
                if subdir.is_dir():
                    self.image_paths.extend(sorted([
                        str(p)
                        for p in subdir.iterdir()
                        if p.suffix.lower() in self.IMAGE_EXTENSIONS
                    ]))

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in {dataset_dir}. Check the directory structure."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load and return an image.

        Returns:
            Dictionary with keys:

            - ``'image'`` — Image tensor of shape ``(C, H, W)``
            - ``'filename'`` — Basename of the image file
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float64)

        if self.y_channel_only:
            ycbcr = _rgb_to_ycbcr(img_np)
            y = ycbcr[:, :, 0:1]  # (H, W, 1)
            y = y / 255.0  # Normalize to [0, 1]
            tensor = torch.from_numpy(y.transpose(2, 0, 1)).float()
        else:
            img_np = img_np / 255.0
            tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        if self.crop_size is not None:
            _, h, w = tensor.shape
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2
            tensor = tensor[
                :, top : top + self.crop_size, left : left + self.crop_size
            ]

        return {
            "image": tensor,
            "filename": os.path.basename(img_path),
        }
