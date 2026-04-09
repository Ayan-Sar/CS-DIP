"""
Differential Geometry Utilities for CS-DIP
==========================================

Implements curvature computation on image manifolds using separable Sobel
convolution kernels. Treats an image I(x,y) as a 2D surface in R^3 and
computes the Gaussian curvature K and Mean curvature H from the first
and second fundamental forms.

All operations are fully differentiable via PyTorch autograd.

Reference:
    Section 2.1–2.2 of the CS-DIP paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelDerivatives(nn.Module):
    """Compute spatial derivatives using fixed (non-trainable) Sobel kernels.

    Computes first-order (I_x, I_y) and second-order (I_xx, I_xy, I_yy)
    partial derivatives of the input feature map via separable convolution.

    The input feature map is averaged across channels before derivative
    computation, producing a single-channel curvature field per spatial
    location.

    Args:
        padding_mode: Padding mode for convolutions. Default: ``'replicate'``.
    """

    def __init__(self, padding_mode: str = "replicate"):
        super().__init__()
        self.padding_mode = padding_mode

        # 1D Sobel kernels for separable convolution
        # Derivative kernel:  [-1, 0, 1] / 2
        # Smoothing kernel:   [1, 2, 1] / 4
        sobel_deriv = torch.tensor([-1.0, 0.0, 1.0]) / 2.0
        sobel_smooth = torch.tensor([1.0, 2.0, 1.0]) / 4.0

        # Register as buffers (non-trainable, moved with .to(device))
        # Shape for F.conv2d: (out_ch, in_ch, kH, kW) — we use groups=1
        # Horizontal derivative kernel (I_x): smooth vertically, diff horizontally
        kx = sobel_smooth.unsqueeze(1) * sobel_deriv.unsqueeze(0)  # (3, 3)
        self.register_buffer("kx", kx.unsqueeze(0).unsqueeze(0))  # (1,1,3,3)

        # Vertical derivative kernel (I_y): diff vertically, smooth horizontally
        ky = sobel_deriv.unsqueeze(1) * sobel_smooth.unsqueeze(0)  # (3, 3)
        self.register_buffer("ky", ky.unsqueeze(0).unsqueeze(0))  # (1,1,3,3)

        # Second-order kernels via composition
        # I_xx: derivative of I_x in x-direction
        kxx = F.conv2d(
            kx.unsqueeze(0).unsqueeze(0),
            kx.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze()
        # Fallback: use direct second derivative kernel
        # d^2/dx^2 ≈ [1, -2, 1]
        deriv2 = torch.tensor([1.0, -2.0, 1.0])
        kxx = sobel_smooth.unsqueeze(1) * deriv2.unsqueeze(0)
        self.register_buffer("kxx", kxx.unsqueeze(0).unsqueeze(0))

        kyy = deriv2.unsqueeze(1) * sobel_smooth.unsqueeze(0)
        self.register_buffer("kyy", kyy.unsqueeze(0).unsqueeze(0))

        # I_xy: derivative of I_x in y-direction
        kxy = sobel_deriv.unsqueeze(1) * sobel_deriv.unsqueeze(0)
        self.register_buffer("kxy", kxy.unsqueeze(0).unsqueeze(0))

    def _pad_and_conv(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply padding and convolution with a fixed kernel.

        Args:
            x: Input tensor of shape ``(B, 1, H, W)``.
            kernel: Convolution kernel of shape ``(1, 1, kH, kW)``.

        Returns:
            Convolved output of shape ``(B, 1, H, W)``.
        """
        pad_h = kernel.shape[2] // 2
        pad_w = kernel.shape[3] // 2
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
        return F.conv2d(x_padded, kernel)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute spatial derivatives of the input feature map.

        The input is averaged across channels to produce a single scalar
        field, then first- and second-order derivatives are computed.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            Tuple of ``(I_x, I_y, I_xx, I_xy, I_yy)``, each of shape
            ``(B, 1, H, W)``.
        """
        # Average across channels → (B, 1, H, W)
        x_mean = x.mean(dim=1, keepdim=True)

        I_x = self._pad_and_conv(x_mean, self.kx)
        I_y = self._pad_and_conv(x_mean, self.ky)
        I_xx = self._pad_and_conv(x_mean, self.kxx)
        I_xy = self._pad_and_conv(x_mean, self.kxy)
        I_yy = self._pad_and_conv(x_mean, self.kyy)

        return I_x, I_y, I_xx, I_xy, I_yy


def compute_gaussian_curvature(
    I_x: torch.Tensor,
    I_y: torch.Tensor,
    I_xx: torch.Tensor,
    I_xy: torch.Tensor,
    I_yy: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Compute Gaussian curvature K of the image manifold.

    .. math::

        K = \frac{I_{xx} I_{yy} - I_{xy}^2}{(1 + I_x^2 + I_y^2)^2}

    Args:
        I_x, I_y: First-order spatial derivatives, shape ``(B, 1, H, W)``.
        I_xx, I_xy, I_yy: Second-order derivatives, shape ``(B, 1, H, W)``.
        eps: Small constant for numerical stability.

    Returns:
        Gaussian curvature map K, shape ``(B, 1, H, W)``.
    """
    numerator = I_xx * I_yy - I_xy ** 2
    denominator = (1.0 + I_x ** 2 + I_y ** 2) ** 2 + eps
    return numerator / denominator


def compute_mean_curvature(
    I_x: torch.Tensor,
    I_y: torch.Tensor,
    I_xx: torch.Tensor,
    I_xy: torch.Tensor,
    I_yy: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Compute Mean curvature H of the image manifold.

    .. math::

        H = \frac{(1+I_x^2)I_{yy} - 2 I_x I_y I_{xy} + (1+I_y^2)I_{xx}}
            {2 (1 + I_x^2 + I_y^2)^{3/2}}

    Args:
        I_x, I_y: First-order spatial derivatives, shape ``(B, 1, H, W)``.
        I_xx, I_xy, I_yy: Second-order derivatives, shape ``(B, 1, H, W)``.
        eps: Small constant for numerical stability.

    Returns:
        Mean curvature map H, shape ``(B, 1, H, W)``.
    """
    numerator = (
        (1.0 + I_x ** 2) * I_yy
        - 2.0 * I_x * I_y * I_xy
        + (1.0 + I_y ** 2) * I_xx
    )
    denominator = 2.0 * (1.0 + I_x ** 2 + I_y ** 2) ** 1.5 + eps
    return numerator / denominator


class CurvatureMap(nn.Module):
    """Compute combined curvature magnitude from a feature map.

    Produces a scalar curvature field :math:`\\kappa = \\alpha |K| + \\beta |H|`
    with learnable mixing weights :math:`\\alpha, \\beta`.

    Args:
        alpha_init: Initial value for the Gaussian curvature weight.
            Default: ``1.0``.
        beta_init: Initial value for the Mean curvature weight.
            Default: ``1.0``.
    """

    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0):
        super().__init__()
        self.sobel = SobelDerivatives()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute curvature maps from an input feature map.

        Args:
            x: Input feature map of shape ``(B, C, H, W)``.

        Returns:
            Tuple of ``(K, H, kappa)`` where:

            - **K** — Gaussian curvature, shape ``(B, 1, H, W)``
            - **H** — Mean curvature, shape ``(B, 1, H, W)``
            - **kappa** — Combined curvature magnitude, shape ``(B, 1, H, W)``
        """
        I_x, I_y, I_xx, I_xy, I_yy = self.sobel(x)
        K = compute_gaussian_curvature(I_x, I_y, I_xx, I_xy, I_yy)
        H = compute_mean_curvature(I_x, I_y, I_xx, I_xy, I_yy)
        kappa = self.alpha * K.abs() + self.beta * H.abs()
        return K, H, kappa
