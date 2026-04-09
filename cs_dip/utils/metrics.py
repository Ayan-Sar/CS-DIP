"""
Image Quality Metrics
=====================

Differentiable-friendly implementations of PSNR and SSIM for evaluating
image restoration quality. Operates on PyTorch tensors.
"""

import torch
import torch.nn.functional as F


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        pred: Predicted image tensor of shape ``(B, C, H, W)`` or
            ``(C, H, W)``, values in ``[0, max_val]``.
        target: Ground-truth image tensor of the same shape.
        max_val: Maximum possible pixel value. Default: ``1.0``.

    Returns:
        PSNR value in dB (float). Returns ``float('inf')`` for
        identical images.
    """
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
    size_average: bool = True,
) -> float:
    """Compute Structural Similarity Index Measure (SSIM).

    Uses a Gaussian sliding window to compute local SSIM statistics.
    Implementation follows Wang et al. (2004).

    Args:
        pred: Predicted image tensor, shape ``(B, C, H, W)``.
        target: Ground-truth image tensor, same shape.
        window_size: Size of the Gaussian window. Default: ``11``.
        max_val: Maximum possible pixel value. Default: ``1.0``.
        size_average: If ``True``, return mean SSIM over the batch.

    Returns:
        SSIM value in ``[0, 1]`` (float).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.shape[1]

    # Create 1D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Create 2D separable Gaussian window
    window_2d = g.unsqueeze(1) * g.unsqueeze(0)  # (ws, ws)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_target

    ssim_map = (
        (2.0 * mu_pred_target + C1)
        * (2.0 * sigma_pred_target + C2)
    ) / (
        (mu_pred_sq + mu_target_sq + C1)
        * (sigma_pred_sq + sigma_target_sq + C2)
    )

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3]).tolist()
