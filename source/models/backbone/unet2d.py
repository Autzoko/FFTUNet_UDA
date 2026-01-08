import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit, UpSample

class UNet2D(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 2,
            channels: tuple = (32, 64, 128, 256, 512),
            num_res_units: int = 2,
            norm: str = "INSTANCE",
            dropout: float | None = None,
    ):
        super().__init__()

        assert len(channels) == 5, "Channels tuple must have length 5."
        c0, c1, c2, c3, c4 = channels

        self.enc0 = self._make_stage(in_channels, c0, num_res_units, norm, dropout)

        # Downsample blocks (stride-2 conv) + residual stages
        self.down1 = Convolution(
            spatial_dims=2, in_channels=c0, out_channels=c1,
            strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.enc1 = self._make_stage(c1, c1, num_res_units, norm, dropout)

        self.down2 = Convolution(
            spatial_dims=2, in_channels=c1, out_channels=c2,
            strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.enc2 = self._make_stage(c2, c2, num_res_units, norm, dropout)

        self.down3 = Convolution(
            spatial_dims=2, in_channels=c2, out_channels=c3,
            strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.enc3 = self._make_stage(c3, c3, num_res_units, norm, dropout)

        self.down4 = Convolution(
            spatial_dims=2, in_channels=c3, out_channels=c4,
            strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout
        )
        self.bottleneck = self._make_stage(c4, c4, num_res_units, norm, dropout)

        # ---- Decoder ----
        # Upsample + concat skip + residual refine
        self.up3 = UpSample(spatial_dims=2, in_channels=c4, out_channels=c3, scale_factor=2, mode="deconv")
        self.dec3 = self._make_stage(c3 + c3, c3, num_res_units, norm, dropout)

        self.up2 = UpSample(spatial_dims=2, in_channels=c3, out_channels=c2, scale_factor=2, mode="deconv")
        self.dec2 = self._make_stage(c2 + c2, c2, num_res_units, norm, dropout)

        self.up1 = UpSample(spatial_dims=2, in_channels=c2, out_channels=c1, scale_factor=2, mode="deconv")
        self.dec1 = self._make_stage(c1 + c1, c1, num_res_units, norm, dropout)

        self.up0 = UpSample(spatial_dims=2, in_channels=c1, out_channels=c0, scale_factor=2, mode="deconv")
        self.dec0 = self._make_stage(c0 + c0, c0, num_res_units, norm, dropout)

        # Segmentation head (logits)
        self.head = nn.Conv2d(c0, num_classes, kernel_size=1)

        
    @staticmethod
    def _make_stage(in_ch, out_ch, num_res_units, norm, dropout):
        blocks = []
        blocks.append(
            ResidualUnit(
                spatial_dims=2, in_channels=in_ch, out_channels=out_ch,
                strides=1, kernel_size=3, subunits=1, norm=norm, dropout=dropout
            )
        )

        for _ in range(num_res_units - 1):
            blocks.append(
                ResidualUnit(
                    spatial_dims=2, in_channels=out_ch, out_channels=out_ch,
                    strides=1, kernel_size=3, subunits=1, norm=norm, dropout=dropout
                )
            )

        return nn.Sequential(*blocks)
    
    @staticmethod
    def _cat(skip, up):
        if skip.shape[-2:] != up.shape[-2:]:
            up = torch.nn.functional.interpolate(
                up, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([skip, up], dim=1)
    
    def forward(self, x):
        """
        x: [B, 1, H, W]
        """
        # ---- Encoder ----
        e0 = self.enc0(x)                 # [B, c0, H, W]
        e1 = self.enc1(self.down1(e0))    # [B, c1, H/2, W/2]
        e2 = self.enc2(self.down2(e1))    # [B, c2, H/4, W/4]
        e3 = self.enc3(self.down3(e2))    # [B, c3, H/8, W/8]
        b  = self.bottleneck(self.down4(e3))  # [B, c4, H/16, W/16]

        # ---- Decoder ----
        d3 = self.dec3(self._cat(e3, self.up3(b)))   # [B, c3, H/8, W/8]
        d2 = self.dec2(self._cat(e2, self.up2(d3)))  # [B, c2, H/4, W/4]
        d1 = self.dec1(self._cat(e1, self.up1(d2)))  # [B, c1, H/2, W/2]
        d0 = self.dec0(self._cat(e0, self.up0(d1)))  # [B, c0, H, W]

        logits = self.head(d0)  # [B, num_classes, H, W]

        feats = {
            "enc0": e0,
            "enc1": e1,
            "enc2": e2,
            "enc3": e3,
            "bottleneck": b,
            "dec3": d3,
            "dec2": d2,
            "dec1": d1,
            "dec0": d0,
        }
        return logits, feats, b
