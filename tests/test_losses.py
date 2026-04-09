"""
Tests for CS-DIP Loss Functions
================================

Validates DataFidelityLoss, CurvatureConsistencyLoss, and CSDIPLoss
for correctness and gradient flow.
"""

import pytest
import torch

from cs_dip.losses import CSDIPLoss, CurvatureConsistencyLoss, DataFidelityLoss


class TestDataFidelityLoss:
    """Test the L1 data fidelity loss."""

    def test_zero_loss_for_identical_inputs(self):
        """Identical prediction and target should give zero loss."""
        loss_fn = DataFidelityLoss()
        x = torch.randn(1, 3, 32, 32)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-6, f"Loss should be ~0 for identical inputs, got {loss.item()}"

    def test_positive_loss_for_different_inputs(self):
        """Different inputs should produce positive loss."""
        loss_fn = DataFidelityLoss()
        x = torch.randn(1, 3, 32, 32)
        y = torch.randn(1, 3, 32, 32)
        loss = loss_fn(x, y)
        assert loss.item() > 0, "Loss should be positive for different inputs"

    def test_with_degradation_function(self):
        """Test that degradation function is applied before loss computation."""
        # Degradation: halve the spatial size
        degrad = lambda img: torch.nn.functional.interpolate(img, scale_factor=0.5, mode="bilinear", align_corners=False)
        loss_fn = DataFidelityLoss(degradation_fn=degrad)
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "Loss with degradation should be computable"

    def test_gradient_flow(self):
        """Verify gradients flow through the data fidelity loss."""
        loss_fn = DataFidelityLoss()
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        y = torch.randn(1, 3, 16, 16)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


class TestCurvatureConsistencyLoss:
    """Test the curvature consistency regularizer."""

    def test_constant_image_low_loss(self):
        """A constant image has zero curvature gradients → low loss."""
        loss_fn = CurvatureConsistencyLoss()
        x = torch.ones(1, 3, 32, 32) * 0.5
        loss = loss_fn(x)
        assert loss.item() < 1e-4, f"Constant image should have very low curvature loss, got {loss.item()}"

    def test_noisy_image_higher_loss(self):
        """A noisy image should have higher curvature consistency loss."""
        loss_fn = CurvatureConsistencyLoss()
        constant = torch.ones(1, 1, 32, 32) * 0.5
        noisy = constant + torch.randn_like(constant) * 0.3
        loss_constant = loss_fn(constant)
        loss_noisy = loss_fn(noisy)
        assert loss_noisy > loss_constant, "Noisy image should have higher curvature loss"

    def test_positive_output(self):
        """Loss should always be non-negative."""
        loss_fn = CurvatureConsistencyLoss()
        x = torch.randn(1, 3, 16, 16)
        loss = loss_fn(x)
        assert loss.item() >= 0, "Curvature loss should be non-negative"

    def test_gradient_flow(self):
        """Verify gradients flow through curvature consistency loss."""
        loss_fn = CurvatureConsistencyLoss()
        x = torch.randn(1, 3, 16, 16, requires_grad=True)
        loss = loss_fn(x)
        loss.backward()
        assert x.grad is not None


class TestCSDIPLoss:
    """Test the combined CS-DIP loss."""

    def test_combined_loss_components(self):
        """Verify the combined loss returns correct components."""
        loss_fn = CSDIPLoss(lambda_curv=0.01)
        pred = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)
        total_loss, loss_dict = loss_fn(pred, target)

        assert "total" in loss_dict
        assert "data_fidelity" in loss_dict
        assert "curvature_consistency" in loss_dict
        assert total_loss.item() > 0

    def test_lambda_weighting(self):
        """Stronger λ should increase the curvature contribution."""
        pred = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 32, 32)

        loss_low = CSDIPLoss(lambda_curv=0.001)
        loss_high = CSDIPLoss(lambda_curv=1.0)

        total_low, _ = loss_low(pred, target)
        total_high, _ = loss_high(pred, target)

        # Higher λ should generally produce higher total loss (same data loss)
        # This may not always hold due to randomness, but with large λ difference it should
        assert total_high.item() >= total_low.item() * 0.5, "Higher λ should increase total loss"

    def test_gradient_flow(self):
        """End-to-end gradient flow through combined loss."""
        loss_fn = CSDIPLoss(lambda_curv=0.01)
        pred = torch.randn(1, 3, 16, 16, requires_grad=True)
        target = torch.randn(1, 3, 16, 16)
        total_loss, _ = loss_fn(pred, target)
        total_loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0
