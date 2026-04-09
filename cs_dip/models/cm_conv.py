"""
Curvature-Modulated Convolution (CM-Conv) Layer
================================================

The core architectural contribution of CS-DIP. CM-Conv dynamically modulates
convolutional filter responses based on the local Gaussian and Mean curvatures
of the image manifold, creating a geometry-aware inductive bias.

In high-curvature regions (edges, corners):  σ(κ) → 1  →  curvature path active
In low-curvature regions (flat, smooth):     σ(κ) → 0  →  structure path active

Reference:
    Section 2.2 of the CS-DIP paper.
"""

import torch
import torch.nn as nn

from .curvature import CurvatureMap


class CMConv(nn.Module):
    """Curvature-Modulated Convolution layer.

    Implements dual-path convolution where the effective kernel is modulated
    by the local curvature of the feature map:

    .. math::

        Y = \\sigma(\\kappa) \\odot (W_c * X) + (1 - \\sigma(\\kappa)) \\odot (W_s * X)

    where :math:`W_s` is the structure path (active in flat regions),
    :math:`W_c` is the curvature path (active at edges/textures), and
    :math:`\\sigma(\\kappa)` is the sigmoid-normalized curvature magnitude.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel. Default: ``3``.
        use_bn: Whether to apply BatchNorm after modulation. Default: ``True``.
        activation: Activation function. Default: ``'leaky_relu'``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        padding = kernel_size // 2

        # Dual convolution paths
        self.conv_structure = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=not use_bn
        )
        self.conv_curvature = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=not use_bn
        )

        # Curvature estimation module
        self.curvature_map = CurvatureMap()

        # Normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Activation
        if activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CM-Conv.

        Args:
            x: Input feature map of shape ``(B, C_in, H, W)``.

        Returns:
            Tuple of ``(output, kappa)`` where:

            - **output** — Modulated feature map, shape ``(B, C_out, H, W)``
            - **kappa** — Curvature magnitude map, shape ``(B, 1, H, W)``
        """
        # Compute curvature from input features
        K, H, kappa = self.curvature_map(x)

        # Sigmoid normalization → [0, 1]
        gate = torch.sigmoid(kappa)  # (B, 1, H, W)

        # Dual-path convolution
        y_structure = self.conv_structure(x)   # (B, C_out, H, W)
        y_curvature = self.conv_curvature(x)   # (B, C_out, H, W)

        # Curvature-modulated blending (broadcast gate across channels)
        y = gate * y_curvature + (1.0 - gate) * y_structure

        # Normalize and activate
        y = self.act(self.bn(y))

        return y, kappa


class CMConvBlock(nn.Module):
    """Block of two stacked CM-Conv layers with residual connection.

    The block applies two CM-Conv layers sequentially and adds a learned
    residual (1×1 projection) from input to output. The curvature map
    from the last CM-Conv is returned for skip-connection transfer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_bn: Whether to apply BatchNorm. Default: ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
    ):
        super().__init__()
        self.cm_conv1 = CMConv(in_channels, out_channels, use_bn=use_bn)
        self.cm_conv2 = CMConv(out_channels, out_channels, use_bn=use_bn)

        # 1×1 projection for residual if channel dimensions differ
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CM-Conv block.

        Args:
            x: Input feature map of shape ``(B, C_in, H, W)``.

        Returns:
            Tuple of ``(output, kappa)`` where:

            - **output** — Output feature map, shape ``(B, C_out, H, W)``
            - **kappa** — Curvature map from the last CM-Conv, shape
              ``(B, 1, H, W)``
        """
        residual = self.residual_proj(x)

        y, _ = self.cm_conv1(x)
        y, kappa = self.cm_conv2(y)

        # Residual connection
        y = y + residual

        return y, kappa
