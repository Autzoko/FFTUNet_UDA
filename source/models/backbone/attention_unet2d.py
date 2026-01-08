import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit, UpSample


class AttentionGate(nn.Module):
    """
    Attention gate for UNet skip connection.
    enc_feat: skip feature from encoder (higher resolution)
    dec_feat: gating feature from decoder (same resolution as enc_feat in recommended usage)
    """
    def __init__(self, in_ch_enc: int, in_ch_dec: int, inter_ch: int):
        super().__init__()
        inter_ch = max(1, int(inter_ch))  # safety

        # project encoder feat
        self.W_g = nn.Conv2d(in_ch_enc, inter_ch, kernel_size=1, bias=True)
        # project decoder gating feat
        self.W_x = nn.Conv2d(in_ch_dec, inter_ch, kernel_size=1, bias=True)

        # attention coefficient (1 channel)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(enc_feat)
        x1 = self.W_x(dec_feat)

        # safety: align spatial sizes if ever mismatched (e.g., odd input sizes)
        if g1.shape[-2:] != x1.shape[-2:]:
            x1 = F.interpolate(x1, size=g1.shape[-2:], mode="bilinear", align_corners=False)

        psi = self.sig(self.psi(self.act(g1 + x1)))  # [B,1,H,W]
        return enc_feat * psi


class AttentionUNet2D(nn.Module):
    """
    Attention U-Net 2D with attention gates on skip connections.

    Returns:
        logits: [B, num_classes, H, W]
        feats:  dict of intermediate features (enc/dec + upsampled)
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
    ):
        super().__init__()
        assert len(channels) == 5, "Expect channels=(c0,c1,c2,c3,c4)"
        c0, c1, c2, c3, c4 = channels

        # --- Encoder ---
        self.enc0 = self._make_stage(in_channels, c0, num_res_units, norm, dropout)

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

        # --- Decoder upsample blocks ---
        self.up3 = UpSample(spatial_dims=2, in_channels=c4, out_channels=c3, scale_factor=2, mode="deconv")
        self.up2 = UpSample(spatial_dims=2, in_channels=c3, out_channels=c2, scale_factor=2, mode="deconv")
        self.up1 = UpSample(spatial_dims=2, in_channels=c2, out_channels=c1, scale_factor=2, mode="deconv")
        self.up0 = UpSample(spatial_dims=2, in_channels=c1, out_channels=c0, scale_factor=2, mode="deconv")

        # --- Attention Gates (recommended: gate with SAME-SCALE decoder features u*) ---
        self.att3 = AttentionGate(in_ch_enc=c3, in_ch_dec=c3, inter_ch=c3 // 2)  # gate e3 with u3
        self.att2 = AttentionGate(in_ch_enc=c2, in_ch_dec=c2, inter_ch=c2 // 2)  # gate e2 with u2
        self.att1 = AttentionGate(in_ch_enc=c1, in_ch_dec=c1, inter_ch=c1 // 2)  # gate e1 with u1
        self.att0 = AttentionGate(in_ch_enc=c0, in_ch_dec=c0, inter_ch=c0 // 2)  # gate e0 with u0

        # --- Decoder refine stages (after concat) ---
        self.dec3 = self._make_stage(c3 + c3, c3, num_res_units, norm, dropout)
        self.dec2 = self._make_stage(c2 + c2, c2, num_res_units, norm, dropout)
        self.dec1 = self._make_stage(c1 + c1, c1, num_res_units, norm, dropout)
        self.dec0 = self._make_stage(c0 + c0, c0, num_res_units, norm, dropout)

        # seg head
        self.head = nn.Conv2d(c0, num_classes, kernel_size=1)

    @staticmethod
    def _make_stage(in_ch, out_ch, num_res_units, norm, dropout):
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
        # keep robust for odd input sizes
        if skip.shape[-2:] != up.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x):
        # --- Encoder ---
        e0 = self.enc0(x)                 # [B,c0,H,W]
        e1 = self.enc1(self.down1(e0))    # [B,c1,H/2,W/2]
        e2 = self.enc2(self.down2(e1))    # [B,c2,H/4,W/4]
        e3 = self.enc3(self.down3(e2))    # [B,c3,H/8,W/8]
        b  = self.bottleneck(self.down4(e3))  # [B,c4,H/16,W/16]

        # --- Decoder with SAME-SCALE gating (recommended) ---

        # level 3: b -> u3 (H/8), gate e3 with u3
        u3 = self.up3(b)                  # [B,c3,H/8,W/8]
        a3 = self.att3(e3, u3)            # [B,c3,H/8,W/8]
        d3 = self.dec3(self._cat(a3, u3)) # [B,c3,H/8,W/8]

        # level 2: d3 -> u2 (H/4), gate e2 with u2
        u2 = self.up2(d3)                 # [B,c2,H/4,W/4]
        a2 = self.att2(e2, u2)            # [B,c2,H/4,W/4]
        d2 = self.dec2(self._cat(a2, u2)) # [B,c2,H/4,W/4]

        # level 1: d2 -> u1 (H/2), gate e1 with u1
        u1 = self.up1(d2)                 # [B,c1,H/2,W/2]
        a1 = self.att1(e1, u1)            # [B,c1,H/2,W/2]
        d1 = self.dec1(self._cat(a1, u1)) # [B,c1,H/2,W/2]

        # level 0: d1 -> u0 (H), gate e0 with u0
        u0 = self.up0(d1)                 # [B,c0,H,W]
        a0 = self.att0(e0, u0)            # [B,c0,H,W]
        d0 = self.dec0(self._cat(a0, u0)) # [B,c0,H,W]

        logits = self.head(d0)

        feats = {
            "enc0": e0,
            "enc1": e1,
            "enc2": e2,
            "enc3": e3,
            "bottleneck": b,

            "up3": u3, "att3": a3, "dec3": d3,
            "up2": u2, "att2": a2, "dec2": d2,
            "up1": u1, "att1": a1, "dec1": d1,
            "up0": u0, "att0": a0, "dec0": d0,
        }
        return logits, feats, b