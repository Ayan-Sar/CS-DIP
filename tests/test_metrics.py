"""
Tests for Image Quality Metrics
================================

Validates PSNR and SSIM implementations against known analytical values.
"""

import pytest
import torch

from cs_dip.utils.metrics import compute_psnr, compute_ssim


class TestPSNR:
    """Test PSNR computation."""

    def test_identical_images_infinite_psnr(self):
        """Identical images should yield infinite (or very high) PSNR."""
        x = torch.rand(1, 3, 32, 32)
        psnr = compute_psnr(x, x)
        assert psnr == float("inf"), f"Expected inf PSNR for identical images, got {psnr}"

    def test_known_psnr_value(self):
        """Test PSNR for known MSE.

        MSE = 0.01 → PSNR = 10*log10(1/0.01) = 20 dB
        """
        target = torch.ones(1, 1, 100, 100) * 0.5
        # Add known noise: MSE = sigma^2, sigma = 0.1 → MSE = 0.01
        pred = target + 0.1
        psnr = compute_psnr(pred, target)
        expected_psnr = 20.0  # 10*log10(1/0.01)
        assert abs(psnr - expected_psnr) < 0.5, f"Expected ~{expected_psnr}, got {psnr:.2f}"

    def test_3d_input(self):
        """Test with 3D input tensors (C, H, W)."""
        x = torch.rand(3, 32, 32)
        y = torch.rand(3, 32, 32)
        psnr = compute_psnr(x, y)
        assert isinstance(psnr, float)
        assert psnr > 0

    def test_higher_noise_lower_psnr(self):
        """More noise should result in lower PSNR."""
        target = torch.rand(1, 1, 64, 64)
        low_noise = target + torch.randn_like(target) * 0.01
        high_noise = target + torch.randn_like(target) * 0.1
        psnr_low = compute_psnr(low_noise, target)
        psnr_high = compute_psnr(high_noise, target)
        assert psnr_low > psnr_high, "Lower noise should give higher PSNR"


class TestSSIM:
    """Test SSIM computation."""

    def test_identical_images_ssim_one(self):
        """Identical images should have SSIM ≈ 1.0."""
        x = torch.rand(1, 3, 64, 64)
        ssim = compute_ssim(x, x)
        assert abs(ssim - 1.0) < 0.01, f"Expected SSIM≈1.0 for identical images, got {ssim:.4f}"

    def test_different_images_ssim_below_one(self):
        """Different images should have SSIM < 1.0."""
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        ssim = compute_ssim(x, y)
        assert ssim < 1.0, f"Expected SSIM < 1.0 for different images, got {ssim:.4f}"

    def test_ssim_range(self):
        """SSIM should be in [-1, 1]."""
        x = torch.rand(1, 1, 64, 64)
        y = torch.rand(1, 1, 64, 64)
        ssim = compute_ssim(x, y)
        assert -1.0 <= ssim <= 1.0, f"SSIM out of range: {ssim}"

    def test_3d_input(self):
        """Test with 3D input tensors."""
        x = torch.rand(3, 64, 64)
        y = torch.rand(3, 64, 64)
        ssim = compute_ssim(x, y)
        assert isinstance(ssim, float)

    def test_single_channel(self):
        """Test SSIM with single-channel (grayscale) images."""
        x = torch.rand(1, 1, 64, 64)
        ssim = compute_ssim(x, x)
        assert abs(ssim - 1.0) < 0.01

    def test_correlated_images_higher_ssim(self):
        """Slightly noisy version should have higher SSIM than random."""
        target = torch.rand(1, 1, 64, 64)
        noisy = target + torch.randn_like(target) * 0.05
        random_img = torch.rand(1, 1, 64, 64)
        ssim_noisy = compute_ssim(noisy.clamp(0, 1), target)
        ssim_random = compute_ssim(random_img, target)
        assert ssim_noisy > ssim_random, "Correlated images should have higher SSIM"
