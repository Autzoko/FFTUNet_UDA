import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit, UpSample


# -------------------------
# GroupNorm helper (only for 1x1-safe place)
# -------------------------
def GN(ch: int, num_groups: int = 32, affine: bool = True) -> nn.GroupNorm:
    g = min(int(num_groups), int(ch))
    while g > 1 and (ch % g != 0):
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch, affine=affine)


# -------------------------
# Helpers: frequency masks
# -------------------------
def _radial_frequency_mask(h: int, w: int, r_low: float, r_high: float, device):
    yy = torch.linspace(-0.5, 0.5, steps=h, device=device).view(h, 1).expand(h, w)
    xx = torch.linspace(-0.5, 0.5, steps=w, device=device).view(1, w).expand(h, w)
    rr = torch.sqrt(xx * xx + yy * yy)
    mask = (rr >= r_low) & (rr <= r_high)
    return mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)


# -------------------------
# FFT Branch (band-limited)  ✅ back to IN
# -------------------------
class FFTFeatureBlock(nn.Module):
    """
    x -> FFT2 -> center -> band-pass amplitude -> log amp -> conv -> f_fft
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
        B, C, H, W = x.shape
        device = x.device

        X = torch.fft.fft2(x, norm="ortho")
        Xc = _fftshift2d(X)

        amp = torch.abs(Xc)
        mask = _radial_frequency_mask(H, W, self.r_low, self.r_high, device=device)
        amp = amp * mask

        amp = torch.log1p(amp)

        mean = amp.mean(dim=(-2, -1), keepdim=True)
        std = amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        amp = (amp - mean) / std

        return self.net(amp)


# -------------------------
# Light Cross-Attention
# -------------------------
class LiteCrossAttention(nn.Module):
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
        B, C, H, W = spatial.shape
        gh = min(self.grid, H)
        gw = min(self.grid, W)

        q = self.to_q(spatial)
        fft_p = F.adaptive_avg_pool2d(fft_feat, (gh, gw))
        k = self.to_k(fft_p)
        v = self.to_v(fft_p)

        inner = q.shape[1]
        q = q.view(B, self.heads, self.dim_head, H * W).transpose(2, 3)          # [B,h,HW,dh]
        k = k.view(B, self.heads, self.dim_head, gh * gw)                        # [B,h,dh,G]
        v = v.view(B, self.heads, self.dim_head, gh * gw).transpose(2, 3)        # [B,h,G,dh]

        scale = self.dim_head ** -0.5
        attn = torch.softmax(torch.matmul(q, k) * scale, dim=-1)                 # [B,h,HW,G]
        out = torch.matmul(attn, v)                                              # [B,h,HW,dh]
        out = out.transpose(2, 3).contiguous().view(B, inner, H, W)
        return self.proj(out)


# -------------------------
# FFT Fusion (fix alpha naming collision)
# -------------------------
class FFTFusion(nn.Module):
    """
    spatial <- spatial + sigmoid(alpha_logit) * (proj([spatial, fft]) + beta * xattn)

    Notes:
      - alpha_logit is the only learnable gate param.
      - keep backward compat by exposing .alpha_logit and .alpha_param
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
            self.beta = nn.Parameter(torch.tensor(0.5))
        else:
            self.xattn = None
            self.beta = None

        self.alpha_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid ~ 0.12

    def gate(self):
        return torch.sigmoid(self.alpha_logit)

    @property
    def alpha_param(self):
        return self.alpha_logit

    def forward(self, spatial: torch.Tensor, fft_feat: torch.Tensor) -> torch.Tensor:
        if spatial.shape[-2:] != fft_feat.shape[-2:]:
            fft_feat = F.interpolate(fft_feat, size=spatial.shape[-2:], mode="bilinear", align_corners=False)

        delta = self.proj(torch.cat([spatial, fft_feat], dim=1))

        if self.use_xattn:
            delta2 = self.xattn(spatial, fft_feat)
            delta = delta + self.beta * delta2

        return spatial + self.gate() * delta


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
# Boundary-friendly refinement ✅ back to IN
# -------------------------
class BoundaryRefineBlock(nn.Module):
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
# C) Multi-scale context: ASPP
#   ✅ only global_pool uses GN (方案A)
# -------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 6, 12, 18), dropout=0.0, gn_groups: int = 32):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.InstanceNorm2d(out_ch, affine=True),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                        nn.InstanceNorm2d(out_ch, affine=True),
                        nn.ReLU(inplace=True),
                    )
                )

        # ✅ critical fix:
        # After AdaptiveAvgPool2d(1), tensor is [B,C,1,1].
        # InstanceNorm2d will crash in training. Use GN or Identity here.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            GN(out_ch, gn_groups, affine=True),   # <- 方案A1：GN（推荐）
            # nn.Identity(),                      # <- 方案A2：如果你更想“纯IN味道”，就用这个替换上一行
            nn.ReLU(inplace=True),
        )

        cat_ch = out_ch * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(cat_ch, out_ch, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(H, W), mode="bilinear", align_corners=False)
        feats.append(gp)
        x = torch.cat(feats, dim=1)
        return self.project(x)


# -------------------------
# A) SDM head ✅ back to IN
# -------------------------
class SDMHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# -------------------------
# Attention UNet + Multi-scale FFT + ASPP + DS + Boundary + SDM
# -------------------------
class AttentionUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels=(32, 64, 128, 256, 512),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        dropout: float | None = None,

        use_fft: bool = True,
        fft_hidden: int = 16,
        fft_r_low: float = 0.02,
        fft_r_high: float = 0.20,
        fft_scales=("enc0", "enc1", "enc2"),
        fft_use_xattn: bool = True,

        enable_aspp: bool = True,
        aspp_rates=(1, 6, 12, 18),
        aspp_dropout: float = 0.0,

        enable_deep_supervision: bool = True,
        enable_boundary_refine: bool = True,
        enable_edge_head: bool = True,

        enable_sdm_head: bool = True,
        sdm_mid_ch: int = 32,

        enable_topo_aux: bool = True,

        # only used by ASPP.global_pool GN
        aspp_gn_groups: int = 32,
    ):
        super().__init__()
        assert len(channels) == 5
        c0, c1, c2, c3, c4 = channels

        self.use_fft = use_fft
        self.fft_scales = set(list(fft_scales))
        self.enable_deep_supervision = enable_deep_supervision
        self.enable_boundary_refine = enable_boundary_refine
        self.enable_edge_head = enable_edge_head
        self.enable_aspp = enable_aspp
        self.enable_sdm_head = enable_sdm_head
        self.enable_topo_aux = enable_topo_aux

        if self.use_fft:
            self.fft0 = FFTFeatureBlock(c0, 1, fft_hidden, fft_r_low, fft_r_high)
            self.fuse0 = FFTFusion(c0, use_xattn=fft_use_xattn, heads=4, grid=8)

            self.fft1 = FFTFeatureBlock(c1, 1, fft_hidden, fft_r_low, fft_r_high)
            self.fuse1 = FFTFusion(c1, use_xattn=fft_use_xattn, heads=4, grid=8)

            self.fft2 = FFTFeatureBlock(c2, 1, fft_hidden, fft_r_low, fft_r_high)
            self.fuse2 = FFTFusion(c2, use_xattn=fft_use_xattn, heads=4, grid=8)

        # Encoder
        self.enc0 = self._make_stage(in_channels, c0, num_res_units, norm, dropout)
        self.down1 = Convolution(2, c0, c1, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.enc1 = self._make_stage(c1, c1, num_res_units, norm, dropout)

        self.down2 = Convolution(2, c1, c2, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.enc2 = self._make_stage(c2, c2, num_res_units, norm, dropout)

        self.down3 = Convolution(2, c2, c3, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.enc3 = self._make_stage(c3, c3, num_res_units, norm, dropout)

        self.down4 = Convolution(2, c3, c4, strides=2, kernel_size=3, adn_ordering="NDA", norm=norm, dropout=dropout)
        self.bottleneck = self._make_stage(c4, c4, num_res_units, norm, dropout)

        # ASPP
        if self.enable_aspp:
            self.aspp = ASPP(in_ch=c4, out_ch=c4, rates=aspp_rates, dropout=aspp_dropout, gn_groups=aspp_gn_groups)
        else:
            self.aspp = None

        # Decoder
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

        self.boundary_refine = BoundaryRefineBlock(c0) if self.enable_boundary_refine else None
        self.head = nn.Conv2d(c0, num_classes, 1)

        if self.enable_deep_supervision:
            self.ds3 = nn.Conv2d(c3, num_classes, 1)
            self.ds2 = nn.Conv2d(c2, num_classes, 1)
            self.ds1 = nn.Conv2d(c1, num_classes, 1)

        if self.enable_edge_head:
            self.edge_head = nn.Sequential(
                nn.Conv2d(c0, c0 // 2, 3, padding=1, bias=False),
                nn.InstanceNorm2d(c0 // 2, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(c0 // 2, 1, 1, bias=True),
            )

        if self.enable_sdm_head:
            self.sdm_head = SDMHead(in_ch=c0, mid_ch=sdm_mid_ch)

    @staticmethod
    def _make_stage(in_ch, out_ch, num_res_units, norm, dropout):
        blocks = [ResidualUnit(2, in_ch, out_ch, strides=1, kernel_size=3, subunits=1, norm=norm, dropout=dropout)]
        for _ in range(num_res_units - 1):
            blocks.append(ResidualUnit(2, out_ch, out_ch, strides=1, kernel_size=3, subunits=1, norm=norm, dropout=dropout))
        return nn.Sequential(*blocks)

    @staticmethod
    def _cat(skip, up):
        if skip.shape[-2:] != up.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def _inject_fft(self, x_in: torch.Tensor, feat: torch.Tensor, level: str):
        if not self.use_fft:
            return feat, None, None

        if level == "enc0" and "enc0" in self.fft_scales:
            f = self.fft0(x_in)
            feat = self.fuse0(feat, f)
            a = float(torch.sigmoid(self.fuse0.alpha_logit).detach().cpu().item())
            return feat, f, a

        if level == "enc1" and "enc1" in self.fft_scales:
            x_ds = F.interpolate(x_in, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            f = self.fft1(x_ds)
            feat = self.fuse1(feat, f)
            a = float(torch.sigmoid(self.fuse1.alpha_logit).detach().cpu().item())
            return feat, f, a

        if level == "enc2" and "enc2" in self.fft_scales:
            x_ds = F.interpolate(x_in, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            f = self.fft2(x_ds)
            feat = self.fuse2(feat, f)
            a = float(torch.sigmoid(self.fuse2.alpha_logit).detach().cpu().item())
            return feat, f, a

        return feat, None, None

    def forward(self, x):
        e0 = self.enc0(x)
        e0, f0, a0 = self._inject_fft(x, e0, "enc0")

        e1 = self.enc1(self.down1(e0))
        e1, f1, a1 = self._inject_fft(x, e1, "enc1")

        e2 = self.enc2(self.down2(e1))
        e2, f2, a2 = self._inject_fft(x, e2, "enc2")

        e3 = self.enc3(self.down3(e2))
        b  = self.bottleneck(self.down4(e3))

        if self.aspp is not None:
            b = self.aspp(b)
            aspp_feat = b
        else:
            aspp_feat = None

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

        if aspp_feat is not None:
            feats["aspp_feat"] = aspp_feat

        if self.use_fft:
            feats.update({
                "fft_feat0": f0, "fft_alpha0": a0,
                "fft_feat1": f1, "fft_alpha1": a1,
                "fft_feat2": f2, "fft_alpha2": a2,
            })

        if self.enable_deep_supervision:
            ds3 = F.interpolate(self.ds3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False)
            ds1 = F.interpolate(self.ds1(d1), size=logits.shape[-2:], mode="bilinear", align_corners=False)
            feats["ds_logits"] = [ds3, ds2, ds1]

        if self.enable_edge_head:
            feats["edge_logits"] = self.edge_head(d0)

        if self.enable_sdm_head:
            feats["sdm_logits"] = self.sdm_head(d0)

        if self.enable_topo_aux:
            feats["topo_aux"] = torch.softmax(logits, dim=1)[:, 1:2]

        return logits, feats, b