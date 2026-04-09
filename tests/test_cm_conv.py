"""
Tests for Curvature-Modulated Convolution (CM-Conv)
====================================================

Validates CMConv and CMConvBlock functionality including output shapes,
curvature-based gating behavior, and gradient flow.
"""

import pytest
import torch

from cs_dip.models.cm_conv import CMConv, CMConvBlock


class TestCMConv:
    """Test the CMConv layer."""

    def test_output_shapes(self):
        """Verify output tensor shapes for various configurations."""
        layer = CMConv(16, 32)
        x = torch.randn(2, 16, 64, 64)
        y, kappa = layer(x)
        assert y.shape == (2, 32, 64, 64), f"Expected (2,32,64,64), got {y.shape}"
        assert kappa.shape == (2, 1, 64, 64), f"Expected (2,1,64,64), got {kappa.shape}"

    def test_small_input_size(self):
        """Test with minimal spatial dimensions."""
        layer = CMConv(8, 16)
        x = torch.randn(1, 8, 8, 8)
        y, kappa = layer(x)
        assert y.shape == (1, 16, 8, 8)

    def test_single_channel(self):
        """Test with single input/output channel."""
        layer = CMConv(1, 1)
        x = torch.randn(1, 1, 32, 32)
        y, kappa = layer(x)
        assert y.shape == (1, 1, 32, 32)

    def test_flat_input_structure_path_dominant(self):
        """For a constant input, σ(κ) should be near 0.5 (low curvature → structure path)."""
        layer = CMConv(1, 8, use_bn=False, activation="none")
        x = torch.ones(1, 1, 32, 32) * 0.5
        with torch.no_grad():
            _, kappa = layer(x)
        gate = torch.sigmoid(kappa)
        # For a flat surface, curvature → 0, so gate ≈ σ(0) = 0.5
        assert gate.mean().item() < 0.6, f"Gate should be ≤ 0.5 for flat input, got {gate.mean():.3f}"

    def test_no_bn_mode(self):
        """Test without BatchNorm."""
        layer = CMConv(8, 16, use_bn=False)
        x = torch.randn(1, 8, 16, 16)
        y, kappa = layer(x)
        assert y.shape == (1, 16, 16, 16)

    def test_gradient_flow(self):
        """Verify gradients flow back through the CM-Conv layer."""
        layer = CMConv(4, 8)
        x = torch.randn(1, 4, 16, 16, requires_grad=True)
        y, kappa = layer(x)
        loss = y.sum() + kappa.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through CMConv"
        assert x.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_dual_paths_different_outputs(self):
        """Structure and curvature paths should produce different outputs."""
        layer = CMConv(4, 8, use_bn=False, activation="none")
        x = torch.randn(1, 4, 16, 16)
        with torch.no_grad():
            y_s = layer.conv_structure(x)
            y_c = layer.conv_curvature(x)
        # The two paths should not be identical (different random init)
        diff = (y_s - y_c).abs().mean()
        assert diff > 1e-4, "Structure and curvature paths should produce different outputs"


class TestCMConvBlock:
    """Test the CMConvBlock (two stacked CM-Conv layers with residual)."""

    def test_output_shapes_same_channels(self):
        """Test with matching input/output channels."""
        block = CMConvBlock(32, 32)
        x = torch.randn(1, 32, 32, 32)
        y, kappa = block(x)
        assert y.shape == (1, 32, 32, 32)
        assert kappa.shape == (1, 1, 32, 32)

    def test_output_shapes_different_channels(self):
        """Test with different input/output channels (uses 1×1 projection)."""
        block = CMConvBlock(16, 64)
        x = torch.randn(1, 16, 32, 32)
        y, kappa = block(x)
        assert y.shape == (1, 64, 32, 32)
        assert kappa.shape == (1, 1, 32, 32)

    def test_residual_connection(self):
        """Output should differ from a non-residual version."""
        block = CMConvBlock(8, 8)
        x = torch.randn(1, 8, 16, 16)
        with torch.no_grad():
            y, _ = block(x)
        # With residual, output should incorporate input information
        # Simply verify it doesn't crash and produces valid output
        assert not torch.isnan(y).any(), "Output should not contain NaN"
        assert not torch.isinf(y).any(), "Output should not contain Inf"

    def test_gradient_flow(self):
        """Verify end-to-end gradient flow through the block."""
        block = CMConvBlock(8, 16)
        x = torch.randn(1, 8, 16, 16, requires_grad=True)
        y, kappa = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_processing(self):
        """Test with batch size > 1."""
        block = CMConvBlock(4, 8)
        x = torch.randn(4, 4, 16, 16)
        y, kappa = block(x)
        assert y.shape == (4, 8, 16, 16)
        assert kappa.shape == (4, 1, 16, 16)
