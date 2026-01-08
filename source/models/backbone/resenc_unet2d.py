# source/models/backbone/resenc_unet2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit, UpSample


class ResEncUNet2D(nn.Module):
    """
    nnU-Net-ish Residual Encoder U-Net (2D), designed to be UDA/FFT-friendly.

    Key properties:
    - Clean residual encoder stages (good for disentangle/anatomy features)
    - Modular stages (easy to insert FFT/wavelet branches at input/enc0/enc1/enc2)
    - Returns multi-scale features for UDA/disentangle hooks

    Forward returns:
        logits: [B, num_classes, H, W]
        feats:  dict with encoder/decoder features
        bottleneck: deepest feature map
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels=(32, 64, 128, 256, 512),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        dropout: float | None = None,
        deep_supervision: bool = False,   # kept for future extension; not used now
    ):
        super().__init__()
        assert len(channels) == 5, "Expect channels=(c0,c1,c2,c3,c4)"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.num_res_units = num_res_units
        self.norm = norm
        self.dropout = dropout
        self.deep_supervision = deep_supervision

        c0, c1, c2, c3, c4 = channels

        # -------------------------
        # Stem + Encoder
        # -------------------------
        # "Stem" conv helps stabilize very early features (common nnU-Net style)
        self.stem = Convolution(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=c0,
            kernel_size=3,
            strides=1,
            adn_ordering="NDA",
            norm=norm,
            dropout=dropout,
        )
        self.enc0 = self._make_stage(c0, c0, num_res_units, norm, dropout)

        self.down1 = Convolution(
            spatial_dims=2, in_channels=c0, out_channels=c1,
            kernel_size=3, strides=2, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.enc1 = self._make_stage(c1, c1, num_res_units, norm, dropout)

        self.down2 = Convolution(
            spatial_dims=2, in_channels=c1, out_channels=c2,
            kernel_size=3, strides=2, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.enc2 = self._make_stage(c2, c2, num_res_units, norm, dropout)

        self.down3 = Convolution(
            spatial_dims=2, in_channels=c2, out_channels=c3,
            kernel_size=3, strides=2, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.enc3 = self._make_stage(c3, c3, num_res_units, norm, dropout)

        self.down4 = Convolution(
            spatial_dims=2, in_channels=c3, out_channels=c4,
            kernel_size=3, strides=2, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.bottleneck = self._make_stage(c4, c4, num_res_units, norm, dropout)

        # -------------------------
        # Decoder (upsample -> concat skip -> residual refine)
        # -------------------------
        self.up3 = UpSample(spatial_dims=2, in_channels=c4, out_channels=c3, scale_factor=2, mode="deconv")
        self.dec3 = self._make_stage(c3 + c3, c3, num_res_units, norm, dropout)

        self.up2 = UpSample(spatial_dims=2, in_channels=c3, out_channels=c2, scale_factor=2, mode="deconv")
        self.dec2 = self._make_stage(c2 + c2, c2, num_res_units, norm, dropout)

        self.up1 = UpSample(spatial_dims=2, in_channels=c2, out_channels=c1, scale_factor=2, mode="deconv")
        self.dec1 = self._make_stage(c1 + c1, c1, num_res_units, norm, dropout)

        self.up0 = UpSample(spatial_dims=2, in_channels=c1, out_channels=c0, scale_factor=2, mode="deconv")
        self.dec0 = self._make_stage(c0 + c0, c0, num_res_units, norm, dropout)

        # segmentation head
        self.head = nn.Conv2d(c0, num_classes, kernel_size=1)

    @staticmethod
    def _make_stage(in_ch, out_ch, num_res_units, norm, dropout):
        """
        Residual stage: stack of ResidualUnit blocks.
        """
        blocks = []
        blocks.append(
            ResidualUnit(
                spatial_dims=2,
                in_channels=in_ch,
                out_channels=out_ch,
                strides=1,
                kernel_size=3,
                subunits=1,
                norm=norm,
                dropout=dropout,
            )
        )
        for _ in range(num_res_units - 1):
            blocks.append(
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=out_ch,
                    out_channels=out_ch,
                    strides=1,
                    kernel_size=3,
                    subunits=1,
                    norm=norm,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*blocks)

    @staticmethod
    def _cat(skip, up):
        # robust for odd sizes
        if skip.shape[-2:] != up.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, in_channels, H, W]
        """
        # ---- Stem + Encoder ----
        s0 = self.stem(x)         # [B,c0,H,W]
        e0 = self.enc0(s0)        # [B,c0,H,W]

        e1 = self.enc1(self.down1(e0))  # [B,c1,H/2,W/2]
        e2 = self.enc2(self.down2(e1))  # [B,c2,H/4,W/4]
        e3 = self.enc3(self.down3(e2))  # [B,c3,H/8,W/8]
        b  = self.bottleneck(self.down4(e3))  # [B,c4,H/16,W/16]

        # ---- Decoder ----
        u3 = self.up3(b)
        d3 = self.dec3(self._cat(e3, u3))

        u2 = self.up2(d3)
        d2 = self.dec2(self._cat(e2, u2))

        u1 = self.up1(d2)
        d1 = self.dec1(self._cat(e1, u1))

        u0 = self.up0(d1)
        d0 = self.dec0(self._cat(e0, u0))

        logits = self.head(d0)

        feats = {
            "stem": s0,
            "enc0": e0,
            "enc1": e1,
            "enc2": e2,
            "enc3": e3,
            "bottleneck": b,
            "up3": u3, "dec3": d3,
            "up2": u2, "dec2": d2,
            "up1": u1, "dec1": d1,
            "up0": u0, "dec0": d0,
        }
        return logits, feats, b