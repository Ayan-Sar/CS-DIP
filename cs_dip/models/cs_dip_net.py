"""
CS-DIP Network Architecture
============================

Full Curvature-Steered Deep Image Prior network based on the U-Net encoder-
decoder structure. Standard convolution blocks are replaced with CM-Conv
blocks, and skip connections transfer both feature maps and curvature maps
from encoder to decoder for geometric consistency across scales.

The network takes a fixed random noise tensor as input and outputs a
restored image. No external training data is required — the network
weights themselves serve as the image prior.

Reference:
    Section 2.3 of the CS-DIP paper.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cm_conv import CMConvBlock


@dataclass
class CSDIPNetConfig:
    """Configuration for the CS-DIP network.

    Attributes:
        in_channels: Number of input noise channels.
        out_channels: Number of output image channels (1 for grayscale,
            3 for RGB).
        encoder_channels: Channel counts at each encoder scale.
        use_bn: Whether to use BatchNorm in CM-Conv blocks.
        upsample_mode: Upsampling mode for the decoder
            (``'bilinear'`` or ``'nearest'``).
    """

    in_channels: int = 32
    out_channels: int = 3
    encoder_channels: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 512]
    )
    use_bn: bool = True
    upsample_mode: str = "bilinear"


class DownBlock(nn.Module):
    """Encoder down-sampling block: strided conv + CM-Conv block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_bn: Whether to use BatchNorm.
    """

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        # Strided convolution for spatial downsampling (2×)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cm_block = CMConvBlock(in_channels, out_channels, use_bn=use_bn)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Tuple of ``(output, kappa)`` with spatially halved output.
        """
        x = self.downsample(x)
        return self.cm_block(x)


class UpBlock(nn.Module):
    """Decoder up-sampling block: upsample + concat skip + CM-Conv block.

    Concatenates both the feature skip and curvature skip from the
    encoder, giving the decoder access to geometric information.

    Args:
        in_channels: Channels from previous decoder layer.
        skip_channels: Channels from the encoder skip connection.
        out_channels: Output channels after the CM-Conv block.
        use_bn: Whether to use BatchNorm.
        upsample_mode: Upsampling interpolation mode.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_bn: bool = True,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.upsample_mode = upsample_mode
        # +1 for the curvature skip channel
        self.cm_block = CMConvBlock(
            in_channels + skip_channels + 1, out_channels, use_bn=use_bn
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_feat: torch.Tensor,
        skip_kappa: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with skip connections.

        Args:
            x: Input from previous decoder layer, shape
                ``(B, C_in, H', W')``.
            skip_feat: Feature skip from encoder, shape
                ``(B, C_skip, H, W)``.
            skip_kappa: Curvature skip from encoder, shape
                ``(B, 1, H, W)``.

        Returns:
            Tuple of ``(output, kappa)`` with spatially doubled output.
        """
        # Upsample to match skip spatial dimensions
        x = F.interpolate(
            x,
            size=skip_feat.shape[2:],
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode == "bilinear" else None,
        )
        # Concatenate feature skip + curvature skip
        x = torch.cat([x, skip_feat, skip_kappa], dim=1)
        return self.cm_block(x)


class CSDIPNet(nn.Module):
    """Curvature-Steered Deep Image Prior network.

    A U-Net encoder-decoder architecture where all convolution blocks are
    replaced with Curvature-Modulated Convolution (CM-Conv) blocks. The
    skip connections carry both features and curvature maps to preserve
    geometric consistency across scales.

    The network takes a fixed random noise input ``z`` and optimizes its
    weights to produce a clean image, using the network structure as an
    implicit regularizer.

    Args:
        config: Network configuration. If ``None``, uses default config.

    Example::

        >>> config = CSDIPNetConfig(out_channels=1)
        >>> net = CSDIPNet(config)
        >>> z = torch.randn(1, 32, 256, 256)
        >>> output = net(z)
        >>> output.shape
        torch.Size([1, 1, 256, 256])
    """

    def __init__(self, config: Optional[CSDIPNetConfig] = None):
        super().__init__()
        if config is None:
            config = CSDIPNetConfig()
        self.config = config
        ch = config.encoder_channels

        # ---- Input projection ----
        self.input_block = CMConvBlock(
            config.in_channels, ch[0], use_bn=config.use_bn
        )

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, len(ch)):
            self.encoder_blocks.append(
                DownBlock(ch[i - 1], ch[i], use_bn=config.use_bn)
            )

        # ---- Bottleneck ----
        self.bottleneck = CMConvBlock(ch[-1], ch[-1], use_bn=config.use_bn)

        # ---- Decoder ----
        self.decoder_blocks = nn.ModuleList()
        decoder_ch = list(reversed(ch))
        for i in range(len(decoder_ch) - 1):
            self.decoder_blocks.append(
                UpBlock(
                    in_channels=decoder_ch[i],
                    skip_channels=decoder_ch[i + 1],
                    out_channels=decoder_ch[i + 1],
                    use_bn=config.use_bn,
                    upsample_mode=config.upsample_mode,
                )
            )

        # ---- Output head ----
        self.output_conv = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch[0], config.out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate an image from noise input.

        Args:
            z: Random noise input of shape
                ``(B, in_channels, H, W)``.

        Returns:
            Restored image of shape ``(B, out_channels, H, W)`` in ``[0, 1]``.
        """
        # Input projection
        x, kappa = self.input_block(z)

        # Encoder — store skips (features + curvature)
        skips_feat = [x]
        skips_kappa = [kappa]
        for enc_block in self.encoder_blocks:
            x, kappa = enc_block(x)
            skips_feat.append(x)
            skips_kappa.append(kappa)

        # Bottleneck
        x, _ = self.bottleneck(x)

        # Decoder — consume skips in reverse (skip the last, it's bottleneck input)
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_idx = len(skips_feat) - 2 - i
            x, _ = dec_block(x, skips_feat[skip_idx], skips_kappa[skip_idx])

        # Output head
        return self.output_conv(x)

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
