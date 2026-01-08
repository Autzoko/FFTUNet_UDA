import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit, UpSample


# -------------------------
# FFT Branch
# -------------------------
class FFTFeatureBlock(nn.Module):
    """
    FFT feature extractor:
    x -> FFT2 -> log amplitude -> small conv stack -> f_fft (C channels)
    """
    def __init__(self, out_channels: int, hidden_channels: int = 16, use_fftshift: bool = False):
        super().__init__()
        self.use_fftshift = use_fftshift

        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    @staticmethod
    def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] (real)
        # shift zero-freq to center by rolling H/2, W/2
        h, w = x.shape[-2], x.shape[-1]
        return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] float
        returns f_fft: [B,out_channels,H,W]
        """
        # FFT expects float/complex; torch.fft works on cuda/cpu. (MPS: depends on torch build)
        # Use orthonormal FFT for stability.
        X = torch.fft.fft2(x, norm="ortho")                # complex
        amp = torch.abs(X)                                 # [B,1,H,W], real
        amp = torch.log1p(amp)                              # log amplitude

        if self.use_fftshift:
            amp = self._fftshift2d(amp)

        # normalize per-sample to reduce scale drift (optional but helps)
        # (B,1,H,W) -> subtract mean, divide std
        mean = amp.mean(dim=(-2, -1), keepdim=True)
        std = amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        amp_n = (amp - mean) / std

        f_fft = self.net(amp_n)
        return f_fft


class FFTFusion(nn.Module):
    """
    Fuse spatial feat and fft feat at same resolution:
      delta = Conv1x1([spatial, fft])
      out = spatial + alpha * delta
    alpha is learnable and initialized to 0 for stability.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))  # start with "no FFT effect"

    def forward(self, spatial: torch.Tensor, fft_feat: torch.Tensor) -> torch.Tensor:
        if spatial.shape[-2:] != fft_feat.shape[-2:]:
            fft_feat = F.interpolate(fft_feat, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.proj(torch.cat([spatial, fft_feat], dim=1))
        return spatial + self.alpha * delta


# -------------------------
# Attention Gate
# -------------------------
class AttentionGate(nn.Module):
    """
    Attention gate for UNet skip connection.
    enc_feat: skip feature from encoder (higher resolution)
    dec_feat: gating feature from decoder (same resolution as enc_feat in recommended usage)
    """
    def __init__(self, in_ch_enc: int, in_ch_dec: int, inter_ch: int):
        super().__init__()
        inter_ch = max(1, int(inter_ch))  # safety

        self.W_g = nn.Conv2d(in_ch_enc, inter_ch, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(in_ch_dec, inter_ch, kernel_size=1, bias=True)

        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(enc_feat)
        x1 = self.W_x(dec_feat)

        if g1.shape[-2:] != x1.shape[-2:]:
            x1 = F.interpolate(x1, size=g1.shape[-2:], mode="bilinear", align_corners=False)

        psi = self.sig(self.psi(self.act(g1 + x1)))  # [B,1,H,W]
        return enc_feat * psi


# -------------------------
# Attention UNet + FFT
# -------------------------
class AttentionUNet2D(nn.Module):
    """
    Attention U-Net 2D with attention gates on skip connections + optional FFT branch injection.

    FFT design:
      - compute log amplitude spectrum from input x
      - shallow conv to produce fft features at full resolution
      - fuse into enc0 output with residual + learnable alpha (starts at 0)

    Returns:
        logits: [B, num_classes, H, W]
        feats:  dict of intermediate features
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
        # --- FFT options ---
        use_fft: bool = True,
        fft_hidden: int = 16,
        fft_shift: bool = False,
    ):
        super().__init__()
        assert len(channels) == 5, "Expect channels=(c0,c1,c2,c3,c4)"
        c0, c1, c2, c3, c4 = channels

        self.use_fft = use_fft

        # --- FFT branch (inject at enc0 resolution) ---
        if self.use_fft:
            self.fft_block = FFTFeatureBlock(out_channels=c0, hidden_channels=fft_hidden, use_fftshift=fft_shift)
            self.fft_fuse0 = FFTFusion(channels=c0)

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

        # --- Attention Gates (gate with SAME-SCALE decoder features u*) ---
        self.att3 = AttentionGate(in_ch_enc=c3, in_ch_dec=c3, inter_ch=c3 // 2)
        self.att2 = AttentionGate(in_ch_enc=c2, in_ch_dec=c2, inter_ch=c2 // 2)
        self.att1 = AttentionGate(in_ch_enc=c1, in_ch_dec=c1, inter_ch=c1 // 2)
        self.att0 = AttentionGate(in_ch_enc=c0, in_ch_dec=c0, inter_ch=c0 // 2)

        # --- Decoder refine stages ---
        self.dec3 = self._make_stage(c3 + c3, c3, num_res_units, norm, dropout)
        self.dec2 = self._make_stage(c2 + c2, c2, num_res_units, norm, dropout)
        self.dec1 = self._make_stage(c1 + c1, c1, num_res_units, norm, dropout)
        self.dec0 = self._make_stage(c0 + c0, c0, num_res_units, norm, dropout)

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
        if skip.shape[-2:] != up.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x):
        # --- Encoder ---
        e0 = self.enc0(x)  # [B,c0,H,W]

        # ✅ FFT injection at shallow level (enc0 output)
        fft_amp = None
        fft_feat0 = None
        if self.use_fft:
            # FFT from input x (recommended: use augmented x if training pipeline augments x)
            fft_feat0 = self.fft_block(x)      # [B,c0,H,W]
            e0 = self.fft_fuse0(e0, fft_feat0) # residual fusion with learnable alpha

        e1 = self.enc1(self.down1(e0))              # [B,c1,H/2,W/2]
        e2 = self.enc2(self.down2(e1))              # [B,c2,H/4,W/4]
        e3 = self.enc3(self.down3(e2))              # [B,c3,H/8,W/8]
        b  = self.bottleneck(self.down4(e3))        # [B,c4,H/16,W/16]

        # --- Decoder with SAME-SCALE gating ---
        u3 = self.up3(b)
        a3 = self.att3(e3, u3)
        d3 = self.dec3(self._cat(a3, u3))

        u2 = self.up2(d3)
        a2 = self.att2(e2, u2)
        d2 = self.dec2(self._cat(a2, u2))

        u1 = self.up1(d2)
        a1 = self.att1(e1, u1)
        d1 = self.dec1(self._cat(a1, u1))

        u0 = self.up0(d1)
        a0 = self.att0(e0, u0)
        d0 = self.dec0(self._cat(a0, u0))

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

        if self.use_fft:
            feats.update({
                "fft_feat0": fft_feat0,           # frequency branch feature at full res
                "fft_alpha0": self.fft_fuse0.alpha.detach().clone(),
            })

        return logits, feats, b