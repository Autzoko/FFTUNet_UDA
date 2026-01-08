import argparse
from pathlib import Path
import random
import math
import copy

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from monai.losses import DiceLoss

# ---- Backbones ----
from source.models.backbone.unet2d import UNet2D
from source.models.backbone.attention_unet2d import AttentionUNet2D
from source.models.backbone.resenc_unet2d import ResEncUNet2D
from source.models.backbone.attention_fft_unet2d import AttentionUNet2D as AttentionFFTUNet2D
from source.models.backbone.attnft_unet_md import AttentionUNet2D as AttentionFFTUNet2D_MD
from source.models.backbone.attnft_unet_md_2 import AttentionUNet2D as AttentionFFTUNet2D_MD2


# -------------------------
# Utils
# -------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (m > 0).astype(np.uint8)


def resize_np(img: np.ndarray, size: int, is_mask: bool):
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


# -------------------------
# Model output adapters (兼容所有旧/新模型)
# -------------------------
def parse_model_output(model_out):
    """
    Returns:
      logits: Tensor [B,C,H,W]
      feats:  dict or None
      extra:  anything else (e.g., bottleneck)
    Supports:
      - logits
      - (logits, feats, b) / (logits, feats) / (logits, ...)
      - {"logits":..., "feats":...} or arbitrary dict containing tensors
    """
    logits, feats, extra = None, None, None

    if isinstance(model_out, dict):
        if "logits" in model_out and torch.is_tensor(model_out["logits"]):
            logits = model_out["logits"]
        else:
            # fallback: first tensor value
            for v in model_out.values():
                if torch.is_tensor(v):
                    logits = v
                    break
        feats = model_out.get("feats", None)
        extra = model_out.get("extra", None)
        if logits is None:
            raise ValueError("Model returned dict but no tensor logits found.")
        return logits, feats, extra

    if isinstance(model_out, (tuple, list)):
        if len(model_out) == 0:
            raise ValueError("Model returned empty tuple/list.")
        logits = model_out[0]
        feats = model_out[1] if len(model_out) >= 2 and isinstance(model_out[1], dict) else None
        extra = model_out[2] if len(model_out) >= 3 else None
        return logits, feats, extra

    # plain tensor
    logits = model_out
    return logits, None, None


# -------------------------
# Augmentations
# -------------------------
def _rand_uniform(a, b):
    return a + (b - a) * random.random()


def aug_geometric_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    rot_deg: float = 15.0,
    translate: float = 0.10,
    scale: tuple = (0.9, 1.1),
    hflip_p: float = 0.5,
):
    """
    Geometric augmentation applied to BOTH image and mask.
    x: [1,H,W] float
    y: [H,W] long 0/1
    """
    x_b = x.unsqueeze(0)  # [1,1,H,W]
    y_b = y.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

    if random.random() < hflip_p:
        x_b = torch.flip(x_b, dims=[3])
        y_b = torch.flip(y_b, dims=[3])

    angle = math.radians(_rand_uniform(-rot_deg, rot_deg))
    s = _rand_uniform(scale[0], scale[1])
    tx = _rand_uniform(-translate, translate)
    ty = _rand_uniform(-translate, translate)

    a = s * math.cos(angle)
    b = -s * math.sin(angle)
    c = s * math.sin(angle)
    d = s * math.cos(angle)
    theta = torch.tensor([[a, b, tx],
                          [c, d, ty]], dtype=torch.float32, device=x.device).unsqueeze(0)

    grid = F.affine_grid(theta, size=x_b.size(), align_corners=False)
    x_warp = F.grid_sample(x_b, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_warp = F.grid_sample(y_b, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    x_out = x_warp.squeeze(0)  # [1,H,W]
    y_out = (y_warp.squeeze(0).squeeze(0) > 0.5).long()  # [H,W]
    return x_out, y_out


def _make_lowfreq_field(device, H, W, grid=32, blur=7):
    gH = min(grid, H)
    gW = min(grid, W)
    field = torch.randn(1, 1, gH, gW, device=device)
    field = F.interpolate(field, size=(H, W), mode="bilinear", align_corners=False)
    if blur and blur >= 3:
        k = int(blur)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        field = F.avg_pool2d(field, kernel_size=k, stride=1, padding=pad)
    field = field.squeeze(0)  # [1,H,W]
    field = field - field.mean()
    field = field / (field.std() + 1e-6)
    field = torch.tanh(field)
    return field


def aug_ultrasound_intensity_torch(
    x: torch.Tensor,
    p_gain_field: float = 0.35,
    gain_strength: tuple = (0.05, 0.25),
    p_speckle: float = 0.45,
    speckle_sigma: tuple = (0.03, 0.12),
    p_gamma: float = 0.6,
    gamma_range: tuple = (0.75, 1.35),
    p_brightness_contrast: float = 0.5,
    brightness: float = 0.08,
    contrast: float = 0.15,
    p_blur: float = 0.12,
):
    x = torch.clamp(x, 0.0, 1.0)

    if random.random() < p_gain_field:
        H, W = x.shape[-2:]
        field = _make_lowfreq_field(x.device, H, W, grid=32, blur=7)
        s = _rand_uniform(gain_strength[0], gain_strength[1])
        x = x * (1.0 + s * field)

    if random.random() < p_brightness_contrast:
        if brightness > 0:
            b = _rand_uniform(-brightness, brightness)
            x = x + b
        if contrast > 0:
            c = _rand_uniform(1.0 - contrast, 1.0 + contrast)
            x = x * c

    x = torch.clamp(x, 0.0, 1.0)

    if random.random() < p_gamma:
        g = _rand_uniform(gamma_range[0], gamma_range[1])
        x = torch.clamp(x, 0.0, 1.0) ** g

    if random.random() < p_speckle:
        sigma = _rand_uniform(speckle_sigma[0], speckle_sigma[1])
        n = torch.randn_like(x) * sigma
        x = x * (1.0 + n)

    if random.random() < p_blur:
        x_b = x.unsqueeze(0)
        x_b = F.avg_pool2d(x_b, kernel_size=3, stride=1, padding=1)
        x = x_b.squeeze(0)

    x = torch.clamp(x, 0.0, 1.0)
    return x


# -------------------------
# Dataset
# -------------------------
class FlatSegDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 256, augment: bool = False):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment

        ids_path = self.root / "splits" / f"{split}.txt"
        assert ids_path.exists(), f"Missing split file: {ids_path}"
        self.ids = [l.strip() for l in ids_path.read_text().splitlines() if l.strip()]

        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"
        assert self.img_dir.exists() and self.msk_dir.exists(), "Missing images/ or masks/"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        ip = self.img_dir / f"{sid}.png"
        mp = self.msk_dir / f"{sid}.png"
        assert ip.exists(), f"Missing image: {ip}"
        assert mp.exists(), f"Missing mask: {mp}"

        img = load_gray(ip)
        msk = load_mask(mp)

        img = resize_np(img, self.image_size, is_mask=False)
        msk = resize_np(msk, self.image_size, is_mask=True)

        img = img / 255.0

        x = torch.from_numpy(img).unsqueeze(0).float()  # [1,H,W]
        y = torch.from_numpy(msk).long()                # [H,W]

        if self.augment:
            x, y = aug_geometric_torch(x, y)
            x = aug_ultrasound_intensity_torch(x)

        return x, y, sid


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def dice_fg_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    inter = ((pred == 1) & (y == 1)).sum().float()
    denom = (pred == 1).sum().float() + (y == 1).sum().float()
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return float(dice.item())


# -------------------------
# Backbone factory
# -------------------------
def build_model(backbone: str):
    b = backbone.lower()
    if b in ["unet", "unet2d"]:
        return UNet2D(in_channels=1, num_classes=2)
    if b in ["attunet", "attentionunet", "attention_unet", "attention_unet2d"]:
        return AttentionUNet2D(in_channels=1, num_classes=2)
    if b in ["resenc_unet2d", "resenc", "nnunet"]:
        return ResEncUNet2D(in_channels=1, num_classes=2)
    if b in ["attention_fft_unet2d", "attention_fft_unet"]:
        return AttentionFFTUNet2D(in_channels=1, num_classes=2)
    if b in ["attention_fft_unet2d_md", "attnft_unet_md", "md"]:
        return AttentionFFTUNet2D_MD(in_channels=1, num_classes=2)
    if b in ["attention_fft_unet2d_md2", "attnft_unet_md2"]:
        return AttentionFFTUNet2D_MD2(in_channels=1, num_classes=2)
    raise ValueError(f"Unknown backbone: {backbone}.")


# -------------------------
# FFT alpha helpers (robust: only read alpha_logit / alpha tensor)
# -------------------------
def get_fft_alphas(model: nn.Module):
    """
    Robust alpha extractor:
      - if module has alpha_logit: report sigmoid(alpha_logit)
      - else if has alpha tensor: report alpha
    Does NOT attempt to call alpha() to avoid property/method collisions.
    """
    alphas = {}

    def _read_alpha(m):
        if hasattr(m, "alpha_logit") and torch.is_tensor(getattr(m, "alpha_logit")):
            return float(torch.sigmoid(getattr(m, "alpha_logit")).detach().cpu().item())
        if hasattr(m, "alpha") and torch.is_tensor(getattr(m, "alpha")):
            return float(getattr(m, "alpha").detach().cpu().item())
        return None

    # common module names first
    for name in ["fuse0", "fuse1", "fuse2", "fft_fuse0", "fft_fuse1", "fft_fuse2"]:
        if hasattr(model, name):
            v = _read_alpha(getattr(model, name))
            if v is not None:
                alphas[name] = v

    # fallback scan
    if len(alphas) == 0:
        for n, m in model.named_modules():
            v = _read_alpha(m)
            if v is not None:
                alphas[n] = v

    return alphas


def set_fft_alpha_trainable(model: nn.Module, trainable: bool):
    """
    Freeze/unfreeze FFT alpha parameters for warmup.
    Supports alpha (tensor) and alpha_logit (tensor).
    """
    # common names
    for name in ["fuse0", "fuse1", "fuse2", "fft_fuse0", "fft_fuse1", "fft_fuse2"]:
        if hasattr(model, name):
            m = getattr(model, name)
            if hasattr(m, "alpha") and torch.is_tensor(getattr(m, "alpha")):
                getattr(m, "alpha").requires_grad_(trainable)
            if hasattr(m, "alpha_logit") and torch.is_tensor(getattr(m, "alpha_logit")):
                getattr(m, "alpha_logit").requires_grad_(trainable)

    # fallback scan
    for _, m in model.named_modules():
        if hasattr(m, "alpha") and torch.is_tensor(getattr(m, "alpha")):
            getattr(m, "alpha").requires_grad_(trainable)
        if hasattr(m, "alpha_logit") and torch.is_tensor(getattr(m, "alpha_logit")):
            getattr(m, "alpha_logit").requires_grad_(trainable)


# -------------------------
# Losses
# -------------------------
class DiceCE(nn.Module):
    def __init__(self, dice_w=1.0, ce_w=1.0, label_smoothing=0.0):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_w = float(dice_w)
        self.ce_w = float(ce_w)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, y):
        dl = self.dice(logits, y.unsqueeze(1))
        ce = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        return self.dice_w * dl + self.ce_w * ce


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce

        if self.alpha is not None:
            if not torch.is_tensor(self.alpha):
                alpha = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            else:
                alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            a = alpha.gather(0, y.view(-1)).view_as(y).float()
            focal = focal * a

        return focal.mean()


class DiceFocal(nn.Module):
    def __init__(self, dice_w=1.0, focal_w=1.0, gamma=2.0, label_smoothing=0.0, alpha=None):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
        self.dice_w = float(dice_w)
        self.focal_w = float(focal_w)

    def forward(self, logits, y):
        dl = self.dice(logits, y.unsqueeze(1))
        fl = self.focal(logits, y)
        return self.dice_w * dl + self.focal_w * fl


# -------------------------
# Deep supervision + edge + SDM + hole penalty helpers
# -------------------------
@torch.no_grad()
def make_edge_target(y: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    y: [B,H,W] {0,1}
    returns edge map [B,1,H,W] in {0,1}
    Uses morphological gradient approx: dilate - erode.
    """
    y1 = y.float().unsqueeze(1)  # [B,1,H,W]
    pad = k // 2
    dil = F.max_pool2d(y1, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-y1, kernel_size=k, stride=1, padding=pad)
    edge = (dil - ero).clamp(0, 1)
    return edge


@torch.no_grad()
def make_sdm_target_approx(y: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """
    Approx Signed Distance Map using iterative dilate/erode counts.
    y: [B,H,W] {0,1}
    returns: sdm in [-1,1], shape [B,1,H,W]
    """
    y1 = y.float().unsqueeze(1)  # [B,1,H,W]

    outside = 1.0 - y1
    dist_out = torch.zeros_like(y1)
    cur = y1.clone()
    for i in range(1, iters + 1):
        cur = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
        newly = (cur > 0.5) & (dist_out == 0) & (outside > 0.5)
        dist_out[newly] = float(i)

    dist_in = torch.zeros_like(y1)
    cur = y1.clone()
    for i in range(1, iters + 1):
        cur = -F.max_pool2d(-cur, kernel_size=3, stride=1, padding=1)  # erode
        newly = (cur < 0.5) & (dist_in == 0) & (y1 > 0.5)
        dist_in[newly] = float(i)

    sdm = dist_in - dist_out
    sdm = torch.clamp(sdm, -float(iters), float(iters)) / float(iters)
    return sdm


def sdm_regression_loss(sdm_logits: torch.Tensor, sdm_gt: torch.Tensor, loss_type: str = "smoothl1") -> torch.Tensor:
    """
    sdm_logits: [B,1,H,W] raw regression
    sdm_gt:     [B,1,H,W] in [-1,1]
    """
    pred = torch.tanh(sdm_logits)
    if loss_type == "l1":
        return F.l1_loss(pred, sdm_gt)
    return F.smooth_l1_loss(pred, sdm_gt)


def hole_penalty(p_fg: torch.Tensor, k_close: int = 7) -> torch.Tensor:
    """
    p_fg: [B,1,H,W] in [0,1]
    closing(p) fills holes; penalize close(p) - p
    """
    k = int(k_close)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    dil = F.max_pool2d(p_fg, kernel_size=k, stride=1, padding=pad)
    clo = -F.max_pool2d(-dil, kernel_size=k, stride=1, padding=pad)
    return (clo - p_fg).clamp_min(0.0).mean()


def compute_total_loss(
    logits: torch.Tensor,
    feats: dict | None,
    y: torch.Tensor,
    main_loss_fn: nn.Module,
    ds_w: tuple = (0.5, 0.25, 0.125),
    edge_w: float = 0.15,

    # A) SDM
    sdm_w: float = 0.10,
    sdm_iters: int = 12,
    sdm_loss_type: str = "smoothl1",

    # B) hole penalty
    hole_w: float = 0.03,
    hole_k: int = 7,
) -> tuple[torch.Tensor, dict]:
    """
    Supports:
      - feats["ds_logits"]   = list of logits at full-res
      - feats["edge_logits"] = [B,1,H,W]
      - feats["sdm_logits"]  = [B,1,H,W] (optional; if absent, SDM loss skipped)
      - feats["topo_aux"]    = [B,1,H,W] prob (optional; else from logits)
    """
    logs = {}

    # main
    loss_main = main_loss_fn(logits, y)
    loss = loss_main
    logs["loss_main"] = float(loss_main.detach().cpu().item())

    # deep supervision (FIXED: keep tensor)
    if feats is not None and isinstance(feats, dict) and feats.get("ds_logits", None) is not None:
        ds_list = feats["ds_logits"]
        if isinstance(ds_list, (list, tuple)) and len(ds_list) > 0:
            ds_loss_sum = logits.new_tensor(0.0)
            for wi, ds_logits in zip(ds_w, ds_list):
                if ds_logits is None:
                    continue
                l = main_loss_fn(ds_logits, y)
                loss = loss + float(wi) * l
                ds_loss_sum = ds_loss_sum + float(wi) * l
            logs["loss_ds"] = float(ds_loss_sum.detach().cpu().item())

    # edge loss
    if feats is not None and isinstance(feats, dict) and feats.get("edge_logits", None) is not None and edge_w > 0:
        edge_logits = feats["edge_logits"]
        y_edge = make_edge_target(y, k=3)
        le = F.binary_cross_entropy_with_logits(edge_logits, y_edge)
        loss = loss + float(edge_w) * le
        logs["loss_edge"] = float(le.detach().cpu().item())

    # A) SDM loss (only if model provides sdm_logits)
    if feats is not None and isinstance(feats, dict) and feats.get("sdm_logits", None) is not None and sdm_w > 0:
        sdm_logits = feats["sdm_logits"]
        sdm_gt = make_sdm_target_approx(y, iters=int(sdm_iters))
        ls = sdm_regression_loss(sdm_logits, sdm_gt, loss_type=sdm_loss_type)
        loss = loss + float(sdm_w) * ls
        logs["loss_sdm"] = float(ls.detach().cpu().item())

    # B) hole penalty
    if hole_w > 0:
        if feats is not None and isinstance(feats, dict) and feats.get("topo_aux", None) is not None:
            p_fg = feats["topo_aux"]
        else:
            p_fg = torch.softmax(logits, dim=1)[:, 1:2]  # [B,1,H,W]
        lh = hole_penalty(p_fg, k_close=int(hole_k))
        loss = loss + float(hole_w) * lh
        logs["loss_hole"] = float(lh.detach().cpu().item())

    return loss, logs


# -------------------------
# EMA
# -------------------------
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)
    ema_model.load_state_dict(esd, strict=False)


def copy_model(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


# -------------------------
# Train / Val
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", type=str, default="./checkpoints_busi")

    ap.add_argument(
        "--backbone",
        type=str,
        default="attention_unet2d",
        choices=[
            "unet2d",
            "attention_unet2d",
            "resenc_unet2d",
            "attention_fft_unet2d",
            "attention_fft_unet2d_md",
            
        ],
    )

    # regularization / training strategy
    ap.add_argument("--weight_decay", type=float, default=2e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--drop_last", action="store_true")

    # FFT warm-up
    ap.add_argument("--fft_warmup_epochs", type=int, default=10)

    # loss choices
    ap.add_argument("--loss", type=str, default="dice_focal", choices=["dice_ce", "dice_focal"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_w", type=float, default=1.0)
    ap.add_argument("--dice_w", type=float, default=1.0)

    # deep supervision + edge weights (only used if model provides them)
    ap.add_argument("--ds_w3", type=float, default=0.5)
    ap.add_argument("--ds_w2", type=float, default=0.25)
    ap.add_argument("--ds_w1", type=float, default=0.125)
    ap.add_argument("--edge_w", type=float, default=0.15)

    # A) SDM supervision
    ap.add_argument("--sdm_w", type=float, default=0.10)
    ap.add_argument("--sdm_iters", type=int, default=12)
    ap.add_argument("--sdm_loss_type", type=str, default="smoothl1", choices=["smoothl1", "l1"])

    # B) hole penalty
    ap.add_argument("--hole_w", type=float, default=0.03)
    ap.add_argument("--hole_k", type=int, default=7)

    # EMA settings
    ap.add_argument("--use_ema", action="store_true", help="Use EMA teacher for validation/checkpointing.")
    ap.add_argument("--ema_decay", type=float, default=0.99)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = get_device()
    print("Device:", device)
    print("Backbone:", args.backbone)
    print("Augment:", (not args.no_aug))
    print("Loss:", args.loss)
    print(f"SDM: w={args.sdm_w} iters={args.sdm_iters} type={args.sdm_loss_type}")
    print(f"Hole: w={args.hole_w} k={args.hole_k}")

    save_dir = Path(args.save_dir) / args.backbone
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FlatSegDataset(args.data_root, "train", image_size=args.image_size, augment=(not args.no_aug))
    val_ds   = FlatSegDataset(args.data_root, "val",   image_size=args.image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=bool(args.drop_last),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.backbone).to(device)

    # EMA teacher (optional)
    ema_model = copy_model(model).to(device) if args.use_ema else None

    # loss fn
    if args.loss == "dice_ce":
        main_loss_fn = DiceCE(dice_w=args.dice_w, ce_w=1.0, label_smoothing=args.label_smoothing)
    else:
        alpha = None  # binary [bg, fg] 可选：例如 [0.3, 0.7]
        main_loss_fn = DiceFocal(
            dice_w=args.dice_w,
            focal_w=args.focal_w,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            alpha=alpha
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=8, threshold=1e-4, verbose=True
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = 0.0
    bad_epochs = 0

    ds_w = (args.ds_w3, args.ds_w2, args.ds_w1)

    for epoch in range(1, args.epochs + 1):
        # FFT alpha warm-up（只会影响带 FFT 的模型；其他模型无事发生）
        if args.fft_warmup_epochs > 0:
            set_fft_alpha_trainable(model, trainable=(epoch > args.fft_warmup_epochs))

        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_dice = 0.0
        n_tr = 0

        tr_logs_sum = {"loss_main": 0.0, "loss_ds": 0.0, "loss_edge": 0.0, "loss_sdm": 0.0, "loss_hole": 0.0}
        tr_logs_cnt = {"loss_main": 0,   "loss_ds": 0,   "loss_edge": 0,   "loss_sdm": 0,   "loss_hole": 0}

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            else:
                autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

            with autocast_ctx:
                out = model(x)
                logits, feats, _ = parse_model_output(out)

                loss, logs = compute_total_loss(
                    logits=logits,
                    feats=feats,
                    y=y,
                    main_loss_fn=main_loss_fn,
                    ds_w=ds_w,
                    edge_w=args.edge_w,
                    sdm_w=args.sdm_w,
                    sdm_iters=args.sdm_iters,
                    sdm_loss_type=args.sdm_loss_type,
                    hole_w=args.hole_w,
                    hole_k=args.hole_k,
                )

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(opt)
            scaler.update()

            if ema_model is not None:
                ema_update(ema_model, model, decay=args.ema_decay)

            bs = x.size(0)
            tr_loss += float(loss.item()) * bs
            tr_dice += dice_fg_from_logits(logits.detach(), y) * bs
            n_tr += bs

            for k in ["loss_main", "loss_ds", "loss_edge", "loss_sdm", "loss_hole"]:
                if k in logs:
                    tr_logs_sum[k] += float(logs[k]) * bs
                    tr_logs_cnt[k] += bs

        tr_loss /= max(1, n_tr)
        tr_dice /= max(1, n_tr)

        tr_logs = {}
        for k in ["loss_main", "loss_ds", "loss_edge", "loss_sdm", "loss_hole"]:
            if tr_logs_cnt[k] > 0:
                tr_logs[k] = tr_logs_sum[k] / tr_logs_cnt[k]
            else:
                tr_logs[k] = None

        # ---- val ----
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()

        va_loss = 0.0
        va_dice = 0.0
        n_va = 0

        va_logs_sum = {"loss_main": 0.0, "loss_ds": 0.0, "loss_edge": 0.0, "loss_sdm": 0.0, "loss_hole": 0.0}
        va_logs_cnt = {"loss_main": 0,   "loss_ds": 0,   "loss_edge": 0,   "loss_sdm": 0,   "loss_hole": 0}

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                out = eval_model(x)
                logits, feats, _ = parse_model_output(out)

                loss, logs = compute_total_loss(
                    logits=logits,
                    feats=feats,
                    y=y,
                    main_loss_fn=main_loss_fn,
                    ds_w=ds_w,
                    edge_w=args.edge_w,
                    sdm_w=args.sdm_w,
                    sdm_iters=args.sdm_iters,
                    sdm_loss_type=args.sdm_loss_type,
                    hole_w=args.hole_w,
                    hole_k=args.hole_k,
                )

                bs = x.size(0)
                va_loss += float(loss.item()) * bs
                va_dice += dice_fg_from_logits(logits, y) * bs
                n_va += bs

                for k in ["loss_main", "loss_ds", "loss_edge", "loss_sdm", "loss_hole"]:
                    if k in logs:
                        va_logs_sum[k] += float(logs[k]) * bs
                        va_logs_cnt[k] += bs

        va_loss /= max(1, n_va)
        va_dice /= max(1, n_va)

        va_logs = {}
        for k in ["loss_main", "loss_ds", "loss_edge", "loss_sdm", "loss_hole"]:
            if va_logs_cnt[k] > 0:
                va_logs[k] = va_logs_sum[k] / va_logs_cnt[k]
            else:
                va_logs[k] = None

        # FFT alpha print
        alphas = get_fft_alphas(model)
        alpha_str = ""
        if len(alphas) > 0:
            keys = list(alphas.keys())[:4]
            alpha_str = " | " + ", ".join([f"{k}={alphas[k]:.4f}" for k in keys if alphas[k] is not None])

        lr_now = opt.param_groups[0]["lr"]

        def _fmt(v):
            return "NA" if v is None else f"{v:.4f}"

        msg = (
            f"[Epoch {epoch:03d}] lr={lr_now:.2e} "
            f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} "
            f"(main={_fmt(tr_logs['loss_main'])}, ds={_fmt(tr_logs['loss_ds'])}, edge={_fmt(tr_logs['loss_edge'])}, "
            f"sdm={_fmt(tr_logs['loss_sdm'])}, hole={_fmt(tr_logs['loss_hole'])}) | "
            f"val_loss={va_loss:.4f} val_dice={va_dice:.4f} "
            f"(main={_fmt(va_logs['loss_main'])}, ds={_fmt(va_logs['loss_ds'])}, edge={_fmt(va_logs['loss_edge'])}, "
            f"sdm={_fmt(va_logs['loss_sdm'])}, hole={_fmt(va_logs['loss_hole'])})"
        )
        if ema_model is not None:
            msg += f" | ema={args.ema_decay:.3f}"
        msg += alpha_str
        print(msg)

        scheduler.step(va_dice)

        # save best
        if va_dice > best_val + 1e-6:
            best_val = va_dice
            bad_epochs = 0
            ckpt = {
                "epoch": epoch,
                "best_val_dice": best_val,
                "model": model.state_dict(),
                "ema_model": (ema_model.state_dict() if ema_model is not None else None),
                "opt": opt.state_dict(),
                "image_size": args.image_size,
                "backbone": args.backbone,
                "alphas": alphas,
                "loss": args.loss,
                "ds_w": ds_w,
                "edge_w": args.edge_w,
                "sdm_w": args.sdm_w,
                "sdm_iters": args.sdm_iters,
                "sdm_loss_type": args.sdm_loss_type,
                "hole_w": args.hole_w,
                "hole_k": args.hole_k,
            }
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ✓ saved best.pt (best_val_dice={best_val:.4f})")
        else:
            bad_epochs += 1

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()}, save_dir / f"epoch_{epoch:03d}.pt")

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {bad_epochs} epochs).")
            break

    print("Done. Best val dice:", best_val)


if __name__ == "__main__":
    main()