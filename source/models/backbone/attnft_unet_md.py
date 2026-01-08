import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit, UpSample


# -------------------------
# Helpers: frequency masks
# -------------------------
def _radial_frequency_mask(h: int, w: int, r_low: float, r_high: float, device):
    """
    Build a radial band-pass mask in frequency domain (centered).
    r_low, r_high are normalized radii in [0, 0.5] roughly (Nyquist at 0.5).
    """
    yy = torch.linspace(-0.5, 0.5, steps=h, device=device).view(h, 1).expand(h, w)
    xx = torch.linspace(-0.5, 0.5, steps=w, device=device).view(1, w).expand(h, w)
    rr = torch.sqrt(xx * xx + yy * yy)  # [H,W], 0..~0.707
    mask = (rr >= r_low) & (rr <= r_high)
    return mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    # x: [B,1,H,W] or [B,C,H,W]
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)


def _ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=-(h // 2), dims=-2), shifts=-(w // 2), dims=-1)


# -------------------------
# FFT Branch (band-limited)
# -------------------------
class FFTFeatureBlock(nn.Module):
    """
    x -> FFT2 -> band-pass on amplitude -> log amp -> conv -> f_fft
    NOTE: Uses centered FFT via shift, then applies radial band mask to keep low/mid frequencies.
    """
    def __init__(
        self,
        out_channels: int,
        in_channels: int = 1,
        hidden_channels: int = 16,
        r_low: float = 0.02,
        r_high: float = 0.20,
    ):
        super().__init__()
        self.r_low = float(r_low)
        self.r_high = float(r_high)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] in [0,1]
        """
        B, C, H, W = x.shape
        device = x.device

        # complex FFT
        X = torch.fft.fft2(x, norm="ortho")  # [B,1,H,W] complex
        Xc = _fftshift2d(X)                  # center

        amp = torch.abs(Xc)                  # [B,1,H,W]
        # band-pass keep low/mid
        mask = _radial_frequency_mask(H, W, self.r_low, self.r_high, device=device)
        amp = amp * mask

        amp = torch.log1p(amp)

        # per-sample normalize (stabilize across scanners)
        mean = amp.mean(dim=(-2, -1), keepdim=True)
        std = amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        amp = (amp - mean) / std

        return self.net(amp)  # [B,out_ch,H,W]


# -------------------------
# Light Cross-Attention (Spatial queries, FFT kv)
# -------------------------
class LiteCrossAttention(nn.Module):
    """
    Very lightweight cross-attention:
      Q from spatial feat (H*W tokens),
      K,V from fft feat pooled into small grid (gh*gw tokens).
    This avoids heavy quadratic attention.

    Output is projected back and used as residual delta.
    """
    def __init__(self, channels: int, heads: int = 4, grid: int = 8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.grid = grid
        self.dim_head = max(8, channels // heads)

        inner = self.heads * self.dim_head

        self.to_q = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_k = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_v = nn.Conv2d(channels, inner, 1, bias=False)

        self.proj = nn.Conv2d(inner, channels, 1, bias=True)

    def forward(self, spatial: torch.Tensor, fft_feat: torch.Tensor) -> torch.Tensor:
        """
        spatial: [B,C,H,W]
        fft_feat:[B,C,H,W] (same scale expected; if not, caller interpolates)
        returns attn_out: [B,C,H,W]
        """
        B, C, H, W = spatial.shape
        gh = min(self.grid, H)
        gw = min(self.grid, W)

        q = self.to_q(spatial)  # [B,inner,H,W]
        k = self.to_k(F.adaptive_avg_pool2d(fft_feat, (gh, gw)))  # [B,inner,gh,gw]
        v = self.to_v(F.adaptive_avg_pool2d(fft_feat, (gh, gw)))  # [B,inner,gh,gw]

        # reshape to heads
        inner = q.shape[1]
        q = q.view(B, self.heads, self.dim_head, H * W).transpose(2, 3)  # [B,heads,HW,dh]
        k = k.view(B, self.heads, self.dim_head, gh * gw)                # [B,heads,dh,G]
        v = v.view(B, self.heads, self.dim_head, gh * gw).transpose(2, 3)# [B,heads,G,dh]

        # attention: (HW x G)
        scale = self.dim_head ** -0.5
        attn = torch.softmax(torch.matmul(q, k) * scale, dim=-1)  # [B,heads,HW,G]
        out = torch.matmul(attn, v)                               # [B,heads,HW,dh]
        out = out.transpose(2, 3).contiguous().view(B, inner, H, W)  # [B,inner,H,W]

        return self.proj(out)  # [B,C,H,W]


# -------------------------
# FFT Fusion: gated residual + cross-attn
# -------------------------
class FFTFusion(nn.Module):
    """
    spatial <- spatial + gate * (proj([spatial, fft]) + beta * cross_attn(spatial, fft))
    gate is learnable scalar alpha (sigmoid) so it starts small but not dead.
    """
    def __init__(self, channels: int, use_xattn: bool = True, heads: int = 4, grid: int = 8):
        super().__init__()
        self.use_xattn = use_xattn

        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        if self.use_xattn:
            self.xattn = LiteCrossAttention(channels=channels, heads=heads, grid=grid)
            self.beta = nn.Parameter(torch.tensor(0.5))  # scale xattn residual
        else:
            self.xattn = None
            self.beta = None

        # alpha in logit space -> sigmoid(alpha) in (0,1)
        # init -2 => sigmoid ~ 0.12 (not 0, so FFT won't be ignored forever)
        self.alpha_logit = nn.Parameter(torch.tensor(-2.0))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, spatial: torch.Tensor, fft_feat: torch.Tensor) -> torch.Tensor:
        if spatial.shape[-2:] != fft_feat.shape[-2:]:
            fft_feat = F.interpolate(fft_feat, size=spatial.shape[-2:], mode="bilinear", align_corners=False)

        delta = self.proj(torch.cat([spatial, fft_feat], dim=1))  # [B,C,H,W]

        if self.use_xattn:
            delta2 = self.xattn(spatial, fft_feat)
            delta = delta + self.beta * delta2

        return spatial + self.alpha() * delta


# -------------------------
# Attention Gate
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, in_ch_enc: int, in_ch_dec: int, inter_ch: int):
        super().__init__()
        inter_ch = max(1, int(inter_ch))
        self.W_g = nn.Conv2d(in_ch_enc, inter_ch, 1, bias=True)
        self.W_x = nn.Conv2d(in_ch_dec, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(enc_feat)
        x1 = self.W_x(dec_feat)
        if g1.shape[-2:] != x1.shape[-2:]:
            x1 = F.interpolate(x1, size=g1.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.sig(self.psi(self.act(g1 + x1)))
        return enc_feat * psi


# -------------------------
# Boundary-friendly refinement
# -------------------------
class BoundaryRefineBlock(nn.Module):
    """
    A small boundary refinement head:
      d0 -> (dilated convs) -> residual refine
    This tends to sharpen edges without exploding params.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.refine(x)


# -------------------------
# Attention UNet + Multi-scale FFT + Deep Supervision + Boundary head
# -------------------------
class AttentionUNet2D(nn.Module):
    """
    Returns (by default):
      logits, feats, bottleneck

    Deep supervision:
      feats["ds_logits"] = [logits_d3, logits_d2, logits_d1]  (if enable_deep_supervision)

    Boundary:
      feats["edge_logits"] = edge prediction (optional)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels=(32, 64, 128, 256, 512),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        dropout: float | None = None,

        # FFT options
        use_fft: bool = True,
        fft_hidden: int = 16,
        fft_r_low: float = 0.02,
        fft_r_high: float = 0.20,
        fft_scales=("enc0", "enc1", "enc2"),  # multi-scale injection points
        fft_use_xattn: bool = True,

        # Deep supervision / boundary
        enable_deep_supervision: bool = True,
        enable_boundary_refine: bool = True,
        enable_edge_head: bool = True,
    ):
        super().__init__()
        assert len(channels) == 5
        c0, c1, c2, c3, c4 = channels

        self.use_fft = use_fft
        self.fft_scales = set(list(fft_scales))
        self.enable_deep_supervision = enable_deep_supervision
        self.enable_boundary_refine = enable_boundary_refine
        self.enable_edge_head = enable_edge_head

        # --- FFT branches for multiple scales ---
        if self.use_fft:
            # We always compute FFT from input x, then downsample by interp to match scale.
            self.fft0 = FFTFeatureBlock(out_channels=c0, in_channels=1, hidden_channels=fft_hidden,
                                        r_low=fft_r_low, r_high=fft_r_high)
            self.fuse0 = FFTFusion(channels=c0, use_xattn=fft_use_xattn, heads=4, grid=8)

            self.fft1 = FFTFeatureBlock(out_channels=c1, in_channels=1, hidden_channels=fft_hidden,
                                        r_low=fft_r_low, r_high=fft_r_high)
            self.fuse1 = FFTFusion(channels=c1, use_xattn=fft_use_xattn, heads=4, grid=8)

            self.fft2 = FFTFeatureBlock(out_channels=c2, in_channels=1, hidden_channels=fft_hidden,
                                        r_low=fft_r_low, r_high=fft_r_high)
            self.fuse2 = FFTFusion(channels=c2, use_xattn=fft_use_xattn, heads=4, grid=8)

        # --- Encoder ---
        self.enc0 = self._make_stage(in_channels, c0, num_res_units, norm, dropout)
        self.down1 = Convolution(2, c0, c1, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.enc1 = self._make_stage(c1, c1, num_res_units, norm, dropout)

        self.down2 = Convolution(2, c1, c2, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.enc2 = self._make_stage(c2, c2, num_res_units, norm, dropout)

        self.down3 = Convolution(2, c2, c3, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.enc3 = self._make_stage(c3, c3, num_res_units, norm, dropout)

        self.down4 = Convolution(2, c3, c4, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.bottleneck = self._make_stage(c4, c4, num_res_units, norm, dropout)

        # --- Decoder ---
        self.up3 = UpSample(2, c4, c3, scale_factor=2, mode="deconv")
        self.up2 = UpSample(2, c3, c2, scale_factor=2, mode="deconv")
        self.up1 = UpSample(2, c2, c1, scale_factor=2, mode="deconv")
        self.up0 = UpSample(2, c1, c0, scale_factor=2, mode="deconv")

        self.att3 = AttentionGate(c3, c3, c3 // 2)
        self.att2 = AttentionGate(c2, c2, c2 // 2)
        self.att1 = AttentionGate(c1, c1, c1 // 2)
        self.att0 = AttentionGate(c0, c0, c0 // 2)

        self.dec3 = self._make_stage(c3 + c3, c3, num_res_units, norm, dropout)
        self.dec2 = self._make_stage(c2 + c2, c2, num_res_units, norm, dropout)
        self.dec1 = self._make_stage(c1 + c1, c1, num_res_units, norm, dropout)
        self.dec0 = self._make_stage(c0 + c0, c0, num_res_units, norm, dropout)

        # boundary refinement (on d0)
        if self.enable_boundary_refine:
            self.boundary_refine = BoundaryRefineBlock(c0)
        else:
            self.boundary_refine = None

        # main head
        self.head = nn.Conv2d(c0, num_classes, 1)

        # deep supervision heads (tap decoder features)
        if self.enable_deep_supervision:
            self.ds3 = nn.Conv2d(c3, num_classes, 1)
            self.ds2 = nn.Conv2d(c2, num_classes, 1)
            self.ds1 = nn.Conv2d(c1, num_classes, 1)

        # edge head (optional): predicts boundary map (1 channel)
        if self.enable_edge_head:
            self.edge_head = nn.Sequential(
                nn.Conv2d(c0, c0 // 2, 3, padding=1, bias=False),
                nn.InstanceNorm2d(c0 // 2, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(c0 // 2, 1, 1, bias=True),
            )

    @staticmethod
    def _make_stage(in_ch, out_ch, num_res_units, norm, dropout):
        blocks = []
        blocks.append(ResidualUnit(2, in_ch, out_ch, strides=1, kernel_size=3, subunits=1, norm=norm, dropout=dropout))
        for _ in range(num_res_units - 1):
            blocks.append(ResidualUnit(2, out_ch, out_ch, strides=1, kernel_size=3, subunits=1, norm=norm, dropout=dropout))
        return nn.Sequential(*blocks)

    @staticmethod
    def _cat(skip, up):
        if skip.shape[-2:] != up.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def _inject_fft(self, x_in: torch.Tensor, feat: torch.Tensor, level: str):
        """
        x_in: original input [B,1,H,W] (already normalized/augmented by your pipeline)
        feat: spatial feature at certain scale
        """
        if not self.use_fft:
            return feat, None, None

        if level == "enc0" and "enc0" in self.fft_scales:
            f = self.fft0(x_in)
            feat = self.fuse0(feat, f)
            return feat, f, float(self.fuse0.alpha().detach().cpu().item())

        if level == "enc1" and "enc1" in self.fft_scales:
            x_ds = F.interpolate(x_in, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            f = self.fft1(x_ds)
            feat = self.fuse1(feat, f)
            return feat, f, float(self.fuse1.alpha().detach().cpu().item())

        if level == "enc2" and "enc2" in self.fft_scales:
            x_ds = F.interpolate(x_in, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            f = self.fft2(x_ds)
            feat = self.fuse2(feat, f)
            return feat, f, float(self.fuse2.alpha().detach().cpu().item())

        return feat, None, None

    def forward(self, x):
        # --- Encoder ---
        e0 = self.enc0(x)
        e0, f0, a0 = self._inject_fft(x, e0, "enc0")

        e1 = self.enc1(self.down1(e0))
        e1, f1, a1 = self._inject_fft(x, e1, "enc1")

        e2 = self.enc2(self.down2(e1))
        e2, f2, a2 = self._inject_fft(x, e2, "enc2")

        e3 = self.enc3(self.down3(e2))
        b  = self.bottleneck(self.down4(e3))

        # --- Decoder ---
        u3 = self.up3(b)
        a3s = self.att3(e3, u3)
        d3 = self.dec3(self._cat(a3s, u3))

        u2 = self.up2(d3)
        a2s = self.att2(e2, u2)
        d2 = self.dec2(self._cat(a2s, u2))

        u1 = self.up1(d2)
        a1s = self.att1(e1, u1)
        d1 = self.dec1(self._cat(a1s, u1))

        u0 = self.up0(d1)
        a0s = self.att0(e0, u0)
        d0 = self.dec0(self._cat(a0s, u0))

        # boundary refine (edge sharpening)
        if self.boundary_refine is not None:
            d0 = self.boundary_refine(d0)

        logits = self.head(d0)

        feats = {
            "enc0": e0, "enc1": e1, "enc2": e2, "enc3": e3, "bottleneck": b,
            "up3": u3, "att3": a3s, "dec3": d3,
            "up2": u2, "att2": a2s, "dec2": d2,
            "up1": u1, "att1": a1s, "dec1": d1,
            "up0": u0, "att0": a0s, "dec0": d0,
        }

        # FFT debug info
        if self.use_fft:
            feats.update({
                "fft_feat0": f0, "fft_alpha0": a0,
                "fft_feat1": f1, "fft_alpha1": a1,
                "fft_feat2": f2, "fft_alpha2": a2,
            })

        # deep supervision logits
        if self.enable_deep_supervision:
            ds3 = self.ds3(d3)
            ds2 = self.ds2(d2)
            ds1 = self.ds1(d1)

            # upsample to full-res for easy loss computation
            ds3 = F.interpolate(ds3, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            ds2 = F.interpolate(ds2, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            ds1 = F.interpolate(ds1, size=logits.shape[-2:], mode="bilinear", align_corners=False)

            feats["ds_logits"] = [ds3, ds2, ds1]

        # edge head
        if self.enable_edge_head:
            feats["edge_logits"] = self.edge_head(d0)  # [B,1,H,W]

        return logits, feats, b