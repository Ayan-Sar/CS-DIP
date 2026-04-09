"""
Tests for Differential Geometry Engine
=======================================

Validates Sobel derivative computation, Gaussian curvature, Mean curvature,
and CurvatureMap module against known analytical surfaces.
"""

import pytest
import torch

from cs_dip.models.curvature import (
    CurvatureMap,
    SobelDerivatives,
    compute_gaussian_curvature,
    compute_mean_curvature,
)


class TestSobelDerivatives:
    """Test the SobelDerivatives module."""

    def setup_method(self):
        """Initialize the Sobel module."""
        self.sobel = SobelDerivatives()

    def test_output_shapes(self):
        """Verify output tensor shapes match input spatial dimensions."""
        x = torch.randn(2, 3, 32, 32)
        I_x, I_y, I_xx, I_xy, I_yy = self.sobel(x)
        for deriv in [I_x, I_y, I_xx, I_xy, I_yy]:
            assert deriv.shape == (2, 1, 32, 32), f"Expected (2,1,32,32), got {deriv.shape}"

    def test_constant_image_zero_derivatives(self):
        """A constant image should have near-zero derivatives."""
        x = torch.ones(1, 1, 16, 16) * 0.5
        I_x, I_y, I_xx, I_xy, I_yy = self.sobel(x)
        for name, deriv in [("I_x", I_x), ("I_y", I_y), ("I_xx", I_xx), ("I_xy", I_xy), ("I_yy", I_yy)]:
            # Interior pixels (avoid boundary effects from padding)
            interior = deriv[:, :, 2:-2, 2:-2]
            assert interior.abs().max() < 1e-5, f"{name} should be ~0 for constant image"

    def test_linear_gradient_x(self):
        """A linear ramp in x should have constant I_x and zero I_xx."""
        x = torch.linspace(0, 1, 32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = x.expand(1, 1, 16, 32)  # (1, 1, 16, 32) — ramp along W
        I_x, I_y, I_xx, I_xy, I_yy = self.sobel(x)
        # I_x should be approximately constant in the interior
        interior_Ix = I_x[:, :, 2:-2, 4:-4]
        std = interior_Ix.std()
        assert std < 0.05, f"I_x should be roughly constant for linear gradient, std={std:.4f}"

    def test_gradient_flow(self):
        """Verify gradients flow through the Sobel computation."""
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        I_x, I_y, I_xx, I_xy, I_yy = self.sobel(x)
        loss = (I_x ** 2 + I_y ** 2).sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through Sobel derivatives"
        assert x.grad.abs().sum() > 0, "Gradients should be non-zero"


class TestGaussianCurvature:
    """Test Gaussian curvature computation."""

    def test_flat_surface_zero_curvature(self):
        """A flat surface (constant image) should have K ≈ 0."""
        # Constant image → all derivatives zero
        zeros = torch.zeros(1, 1, 16, 16)
        K = compute_gaussian_curvature(zeros, zeros, zeros, zeros, zeros)
        assert K.abs().max() < 1e-6, "Flat surface should have zero Gaussian curvature"

    def test_known_curvature_values(self):
        """Test curvature computation with known derivative values."""
        # For a sphere-like surface I(x,y) = x² + y² at center:
        # I_x = 2x, I_y = 2y, I_xx = 2, I_yy = 2, I_xy = 0
        # At (0,0): K = (2*2 - 0) / (1 + 0 + 0)^2 = 4.0
        I_x = torch.zeros(1, 1, 1, 1)
        I_y = torch.zeros(1, 1, 1, 1)
        I_xx = torch.full((1, 1, 1, 1), 2.0)
        I_yy = torch.full((1, 1, 1, 1), 2.0)
        I_xy = torch.zeros(1, 1, 1, 1)
        K = compute_gaussian_curvature(I_x, I_y, I_xx, I_xy, I_yy)
        expected = 4.0  # I_xx * I_yy / 1^2
        assert abs(K.item() - expected) < 0.01, f"Expected K≈{expected}, got {K.item():.4f}"


class TestMeanCurvature:
    """Test Mean curvature computation."""

    def test_flat_surface_zero_curvature(self):
        """A flat surface should have H ≈ 0."""
        zeros = torch.zeros(1, 1, 16, 16)
        H = compute_mean_curvature(zeros, zeros, zeros, zeros, zeros)
        assert H.abs().max() < 1e-6, "Flat surface should have zero Mean curvature"

    def test_known_curvature_values(self):
        """Test with known derivative values at origin of paraboloid."""
        # At (0,0) of z = x² + y²:
        # H = ((1+0)*2 - 0 + (1+0)*2) / (2*(1)^1.5) = 4/2 = 2.0
        I_x = torch.zeros(1, 1, 1, 1)
        I_y = torch.zeros(1, 1, 1, 1)
        I_xx = torch.full((1, 1, 1, 1), 2.0)
        I_yy = torch.full((1, 1, 1, 1), 2.0)
        I_xy = torch.zeros(1, 1, 1, 1)
        H = compute_mean_curvature(I_x, I_y, I_xx, I_xy, I_yy)
        expected = 2.0
        assert abs(H.item() - expected) < 0.01, f"Expected H≈{expected}, got {H.item():.4f}"


class TestCurvatureMap:
    """Test the CurvatureMap module."""

    def test_output_shapes(self):
        """Verify output shapes from CurvatureMap."""
        cm = CurvatureMap()
        x = torch.randn(2, 8, 32, 32)
        K, H, kappa = cm(x)
        assert K.shape == (2, 1, 32, 32)
        assert H.shape == (2, 1, 32, 32)
        assert kappa.shape == (2, 1, 32, 32)

    def test_learnable_parameters(self):
        """α and β should be learnable parameters."""
        cm = CurvatureMap(alpha_init=1.0, beta_init=0.5)
        param_names = [name for name, _ in cm.named_parameters()]
        assert "alpha" in param_names, "alpha should be a learnable parameter"
        assert "beta" in param_names, "beta should be a learnable parameter"

    def test_kappa_non_negative(self):
        """κ should be non-negative since κ = α|K| + β|H| with default positive weights."""
        cm = CurvatureMap()
        x = torch.randn(1, 4, 16, 16)
        _, _, kappa = cm(x)
        assert (kappa >= -1e-6).all(), "κ should be non-negative with positive α, β"

    def test_gradient_flow(self):
        """Verify end-to-end gradient flow through CurvatureMap."""
        cm = CurvatureMap()
        x = torch.randn(1, 4, 16, 16, requires_grad=True)
        K, H, kappa = cm(x)
        loss = kappa.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through CurvatureMap"
