"""
CS-DIP Loss Functions
=====================

Implements the composite objective for CS-DIP optimization:

.. math::

    \\mathcal{L} = \\mathcal{L}_{data} + \\lambda \\, \\mathcal{L}_{curv}

Where :math:`\\mathcal{L}_{data}` is the L1 data fidelity term and
:math:`\\mathcal{L}_{curv}` is the sparse curvature consistency regularizer
that penalizes high-frequency oscillations in the curvature field.

Reference:
    Section 3.2 of the CS-DIP paper.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.curvature import (
    SobelDerivatives,
    compute_gaussian_curvature,
    compute_mean_curvature,
)


class DataFidelityLoss(nn.Module):
    """L1 data fidelity loss between degraded prediction and observation.

    .. math::

        \\mathcal{L}_{data} = \\| \\mathcal{D}(f_\\theta(z)) - y \\|_1

    Args:
        degradation_fn: Optional callable that applies the forward
            degradation operator :math:`\\mathcal{D}` to the predicted
            image. If ``None``, identity is assumed (denoising).
    """

    def __init__(self, degradation_fn: Optional[Callable] = None):
        super().__init__()
        self.degradation_fn = degradation_fn

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute data fidelity loss.

        Args:
            prediction: Network output :math:`f_\\theta(z)`, shape
                ``(B, C, H, W)``.
            target: Degraded observation :math:`y`, shape
                ``(B, C, H', W')``.

        Returns:
            Scalar L1 loss.
        """
        if self.degradation_fn is not None:
            prediction = self.degradation_fn(prediction)
        return F.l1_loss(prediction, target)


class CurvatureConsistencyLoss(nn.Module):
    """Sparse curvature consistency regularizer.

    Penalizes high-frequency oscillations in the Gaussian and Mean
    curvature fields of the output image to suppress noise artifacts
    while allowing smooth geometric structures.

    .. math::

        \\mathcal{L}_{curv} = \\int_\\Omega
            (\\|\\nabla K\\|_2^2 + \\|\\nabla H\\|_2^2) \\, d\\Omega

    This is implemented as the mean squared gradient magnitude of the
    curvature fields computed via Sobel derivative operators.
    """

    def __init__(self):
        super().__init__()
        self.sobel = SobelDerivatives()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute curvature consistency loss on the output image.

        Args:
            image: Predicted image :math:`f_\\theta(z)`, shape
                ``(B, C, H, W)``.

        Returns:
            Scalar curvature consistency loss.
        """
        # Compute derivatives of the image
        I_x, I_y, I_xx, I_xy, I_yy = self.sobel(image)

        # Compute curvature fields
        K = compute_gaussian_curvature(I_x, I_y, I_xx, I_xy, I_yy)
        H = compute_mean_curvature(I_x, I_y, I_xx, I_xy, I_yy)

        # Compute spatial gradients of curvature fields
        # Re-use Sobel for ∇K and ∇H (only first-order needed)
        K_x, K_y, _, _, _ = self.sobel(K)
        H_x, H_y, _, _, _ = self.sobel(H)

        # Squared gradient magnitudes, averaged over spatial dimensions
        grad_K_sq = K_x ** 2 + K_y ** 2
        grad_H_sq = H_x ** 2 + H_y ** 2

        return (grad_K_sq + grad_H_sq).mean()


class CSDIPLoss(nn.Module):
    """Combined CS-DIP optimization objective.

    .. math::

        \\mathcal{L} = \\mathcal{L}_{data} + \\lambda \\, \\mathcal{L}_{curv}

    Args:
        lambda_curv: Weight for the curvature consistency regularizer.
            Default: ``0.01``.
        degradation_fn: Optional degradation operator for data fidelity.
    """

    def __init__(
        self,
        lambda_curv: float = 0.01,
        degradation_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.lambda_curv = lambda_curv
        self.data_loss = DataFidelityLoss(degradation_fn)
        self.curv_loss = CurvatureConsistencyLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined CS-DIP loss.

        Args:
            prediction: Network output :math:`f_\\theta(z)`, shape
                ``(B, C, H, W)``.
            target: Degraded observation :math:`y`.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where ``loss_dict``
            contains the individual loss components for logging.
        """
        l_data = self.data_loss(prediction, target)
        l_curv = self.curv_loss(prediction)
        total = l_data + self.lambda_curv * l_curv

        loss_dict = {
            "total": total.item(),
            "data_fidelity": l_data.item(),
            "curvature_consistency": l_curv.item(),
        }
        return total, loss_dict
