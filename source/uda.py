# uda_step1_ema_pseudo.py
# Step-1 UDA: Teacher-EMA + pseudo-label consistency (NO ROI, NO CutMix, NO disentangle yet)
# - Source: supervised loss
# - Target: EMA teacher generates pseudo labels; student learns with filtering
# - + Added: Optional Target validation metrics (global pixel-wise + per-case) for debugging
# - + Added: Optional "drop GT-empty (normal)" for target train/val, and "lesion-only" target val metrics
#
# Notes:
# - Target val is ONLY for monitoring/debugging (do NOT use it to pick best/early stop in "pure" UDA).
# - By default we evaluate EMA(teacher) because it's more stable; you can also evaluate student.
# - For BUSBRA(source) -> BUSI(target): BUSBRA has no normal; you may want to drop BUSI GT-empty from target train/val.

import argparse
from pathlib import Path
import random
import math
import copy
from dataclasses import dataclass

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

try:
    from source.models.backbone.attnft_unet_md_2 import AttentionUNet2D as AttentionFFTUNet2D_MD2
except Exception:
    AttentionFFTUNet2D_MD2 = None


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
    return np.array(Image.open(path).convert("L"), dtype=np.float32)  # 0..255


def load_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (m > 0).astype(np.uint8)


def resize_np(img: np.ndarray, size: int, is_mask: bool):
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


# -------------------------
# Model output adapters
# -------------------------
def parse_model_output(model_out):
    """
    Returns:
      logits: Tensor [B,C,H,W]
      feats:  dict or None
      extra:  anything else
    """
    logits, feats, extra = None, None, None

    if isinstance(model_out, dict):
        if "logits" in model_out and torch.is_tensor(model_out["logits"]):
            logits = model_out["logits"]
        else:
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
        if not torch.is_tensor(logits):
            raise ValueError("Model returned tuple/list but first element is not a tensor logits.")
        return logits, feats, extra

    if torch.is_tensor(model_out):
        return model_out, None, None

    raise ValueError(f"Unsupported model output type: {type(model_out)}")


# -------------------------
# Augmentations (same as your BUSI training)
# -------------------------
def _rand_uniform(a, b):
    return a + (b - a) * random.random()


def aug_geometric_torch(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    rot_deg: float = 15.0,
    translate: float = 0.10,
    scale: tuple = (0.9, 1.1),
    hflip_p: float = 0.5,
):
    """
    Apply same geometric transform to x and y (if y is provided).
    x: [1,H,W]
    y: [H,W] or None
    """
    x_b = x.unsqueeze(0)  # [1,1,H,W]
    y_b = None
    if y is not None:
        y_b = y.unsqueeze(0).unsqueeze(0).float()

    if random.random() < hflip_p:
        x_b = torch.flip(x_b, dims=[3])
        if y_b is not None:
            y_b = torch.flip(y_b, dims=[3])

    angle = math.radians(_rand_uniform(-rot_deg, rot_deg))
    s = _rand_uniform(scale[0], scale[1])
    tx = _rand_uniform(-translate, translate)
    ty = _rand_uniform(-translate, translate)

    a = s * math.cos(angle)
    b = -s * math.sin(angle)
    c = s * math.sin(angle)
    d = s * math.cos(angle)
    theta = torch.tensor([[a, b, tx], [c, d, ty]], dtype=torch.float32, device=x.device).unsqueeze(0)

    grid = F.affine_grid(theta, size=x_b.size(), align_corners=False)
    x_warp = F.grid_sample(x_b, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x_out = x_warp.squeeze(0)

    if y_b is None:
        return x_out, None

    y_warp = F.grid_sample(y_b, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    y_out = (y_warp.squeeze(0).squeeze(0) > 0.5).long()
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
# Datasets
# -------------------------
class FlatSegDatasetLabeled(Dataset):
    """
    root/
      images/{id}.png
      masks/{id}.png
      splits/{split}.txt
    """
    def __init__(self, root: str, split: str, image_size: int = 256, augment: bool = False):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.augment = bool(augment)

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

        img = (img / 255.0).astype(np.float32)

        x = torch.from_numpy(img).unsqueeze(0).float()  # [1,H,W]
        y = torch.from_numpy(msk).long()                # [H,W]

        if self.augment:
            x, y = aug_geometric_torch(x, y)
            x = aug_ultrasound_intensity_torch(x)

        return x, y, sid


class FlatSegDatasetUnlabeled(Dataset):
    """
    root/
      images/{id}.png
      splits/{split}.txt

    Optional (for filtering normal):
      masks/{id}.png   (used ONLY to decide GT-empty and drop it; NOT used as supervision)
    """
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        drop_gt_empty: bool = False,
        gt_empty_thr: int = 0,
        mask_dir_name: str = "masks",
        verbose_filter: bool = True,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.augment = bool(augment)

        ids_path = self.root / "splits" / f"{split}.txt"
        assert ids_path.exists(), f"Missing split file: {ids_path}"
        ids = [l.strip() for l in ids_path.read_text().splitlines() if l.strip()]

        self.img_dir = self.root / "images"
        assert self.img_dir.exists(), "Missing images/"

        self.drop_gt_empty = bool(drop_gt_empty)
        self.gt_empty_thr = int(gt_empty_thr)
        self.mask_dir = self.root / mask_dir_name

        if self.drop_gt_empty:
            assert self.mask_dir.exists(), (
                f"drop_gt_empty=True requires masks to exist for filtering, but not found: {self.mask_dir}"
            )
            kept = []
            dropped = 0
            for sid in ids:
                mp = self.mask_dir / f"{sid}.png"
                if not mp.exists():
                    raise FileNotFoundError(f"Mask not found for filtering GT-empty: {mp}")
                m = load_mask(mp)  # uint8 0/1
                if int(m.sum()) <= self.gt_empty_thr:
                    dropped += 1
                    continue
                kept.append(sid)
            self.ids = kept
            if verbose_filter:
                print(
                    f"[TargetFilter] split={split} drop_gt_empty=True thr={self.gt_empty_thr} "
                    f"kept={len(self.ids)} dropped={dropped} (from {len(ids)})"
                )
        else:
            self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        ip = self.img_dir / f"{sid}.png"
        assert ip.exists(), f"Missing image: {ip}"

        img = load_gray(ip)
        img = resize_np(img, self.image_size, is_mask=False)
        img = (img / 255.0).astype(np.float32)
        x = torch.from_numpy(img).unsqueeze(0).float()

        if self.augment:
            x, _ = aug_geometric_torch(x, None)
            x = aug_ultrasound_intensity_torch(x)

        return x, sid


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
    if b in ["attention_fft_unet2d_md2", "attnft_unet_md2", "md2"]:
        if AttentionFFTUNet2D_MD2 is None:
            raise ValueError("Backbone md2 requested but AttentionFFTUNet2D_MD2 import failed.")
        return AttentionFFTUNet2D_MD2(in_channels=1, num_classes=2)
    raise ValueError(f"Unknown backbone: {backbone}.")


# -------------------------
# Pseudo-label filtering (core)
# -------------------------
@dataclass
class PLFilterCfg:
    conf_thr: float = 0.85
    fg_min: float = 0.002
    fg_max: float = 0.20


@torch.no_grad()
def make_pseudo_from_teacher(logits_t: torch.Tensor):
    """
    logits_t: [B,2,H,W]
    returns:
      y_hat: [B,H,W] long 0/1
      conf_map: [B,H,W] max prob
      conf_mean: [B] mean conf per image
      fg_ratio: [B] predicted fg ratio per image
    """
    p = torch.softmax(logits_t, dim=1)           # [B,2,H,W]
    conf_map, y_hat = torch.max(p, dim=1)        # [B,H,W], [B,H,W]
    conf_mean = conf_map.flatten(1).mean(dim=1)  # [B]
    fg_ratio = (y_hat == 1).float().flatten(1).mean(dim=1)
    return y_hat.long(), conf_map, conf_mean, fg_ratio


def filter_mask_from_stats(conf_mean: torch.Tensor, fg_ratio: torch.Tensor, cfg: PLFilterCfg):
    """
    returns keep: [B] bool
    """
    keep = (conf_mean >= cfg.conf_thr) & (fg_ratio >= cfg.fg_min) & (fg_ratio <= cfg.fg_max)
    return keep


# -------------------------
# Evaluation (global pixel-wise + per-case)  [matches your eval_cross_dataset style]
# -------------------------
@torch.no_grad()
def batch_confusion(pred01: torch.Tensor, y01: torch.Tensor):
    """
    pred01, y01: [B,H,W] values in {0,1}
    returns tp, fp, fn, tn as float tensors (scalar)
    """
    pred = pred01.bool()
    y = y01.bool()
    tp = (pred & y).sum().float()
    fp = (pred & (~y)).sum().float()
    fn = ((~pred) & y).sum().float()
    tn = ((~pred) & (~y)).sum().float()
    return tp, fp, fn, tn


def metrics_from_totals(tp, fp, fn, tn):
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou  = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    prec = (tp + 1e-6) / (tp + fp + 1e-6)
    rec  = (tp + 1e-6) / (tp + fn + 1e-6)
    acc  = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(prec.item()),
        "recall": float(rec.item()),
        "accuracy": float(acc.item()),
    }


@torch.no_grad()
def per_case_dice(pred01: torch.Tensor, y01: torch.Tensor, empty_policy: str = "skip") -> list[float]:
    """
    pred01, y01: [B,H,W] {0,1}
    empty_policy:
      - "skip": exclude GT-empty cases from the returned list
      - "one":  if GT empty and Pred empty -> dice=1 else 0
      - "zero": if GT empty -> dice=0 regardless
    """
    assert empty_policy in ["skip", "one", "zero"]
    B = pred01.size(0)
    out = []
    for i in range(B):
        p = pred01[i].bool()
        y = y01[i].bool()
        p_sum = int(p.sum().item())
        y_sum = int(y.sum().item())

        if y_sum == 0:
            if empty_policy == "skip":
                continue
            if empty_policy == "zero":
                out.append(0.0)
                continue
            out.append(1.0 if p_sum == 0 else 0.0)
            continue

        tp = (p & y).sum().float()
        fp = (p & (~y)).sum().float()
        fn = ((~p) & y).sum().float()
        d = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        out.append(float(d.item()))
    return out


def summarize_list(vals: list[float]) -> dict:
    if len(vals) == 0:
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "p10": None, "p25": None, "p75": None, "p90": None
        }
    arr = np.array(vals, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


@torch.no_grad()
def eval_labeled_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    empty_policy: str = "skip",
    prefix: str = "VAL",
):
    """
    Returns dict with:
      - loss (DiceCE)
      - global metrics (dice/iou/precision/recall/accuracy)
      - pred_fg_ratio
      - gt_empty_fp_rate
      - per-case stats (mean/median/std/p10/p25/p75/p90, n)
      - bookkeeping (skipped empty)
    """
    model.eval()
    loss_fn = DiceCE(dice_w=1.0, ce_w=1.0, label_smoothing=0.0)

    tp = torch.tensor(0.0)
    fp = torch.tensor(0.0)
    fn = torch.tensor(0.0)
    tn = torch.tensor(0.0)

    gt_empty_total = 0
    gt_empty_fp = 0

    pred_fg_sum = 0.0
    pred_fg_count = 0

    percase_dices: list[float] = []
    percase_skipped_empty = 0

    tot_loss = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        logits, _, _ = parse_model_output(out)

        loss = loss_fn(logits, y)
        bs = x.size(0)
        tot_loss += float(loss.item()) * bs
        n += bs

        pred = torch.argmax(logits, dim=1)  # [B,H,W]

        # global confusion
        btp, bfp, bfn, btn = batch_confusion(pred, y)
        tp += btp.cpu()
        fp += bfp.cpu()
        fn += bfn.cpu()
        tn += btn.cpu()

        # pred fg ratio (per-image average)
        pred_fg_sum += float(pred.float().mean().item()) * bs
        pred_fg_count += bs

        # GT-empty FP rate
        for bi in range(bs):
            yi = y[bi]
            pi = pred[bi]
            if yi.sum().item() == 0:
                gt_empty_total += 1
                if pi.sum().item() > 0:
                    gt_empty_fp += 1

        # per-case
        percase_dices.extend(per_case_dice(pred, y, empty_policy=empty_policy))
        if empty_policy == "skip":
            for bi in range(bs):
                if int(y[bi].sum().item()) == 0:
                    percase_skipped_empty += 1

    scalars = metrics_from_totals(tp, fp, fn, tn)
    pred_fg_ratio = (pred_fg_sum / max(1, pred_fg_count))
    fp_rate_empty = (gt_empty_fp / gt_empty_total) if gt_empty_total > 0 else None
    percase_summary = summarize_list(percase_dices)

    return {
        "prefix": prefix,
        "loss": (tot_loss / max(1, n)),
        "global": scalars,
        "pred_fg_ratio": float(pred_fg_ratio),
        "gt_empty_total": int(gt_empty_total),
        "gt_empty_fp": int(gt_empty_fp),
        "gt_empty_fp_rate": None if fp_rate_empty is None else float(fp_rate_empty),
        "percase": percase_summary,
        "percase_skipped_empty": int(percase_skipped_empty),
        "empty_policy": empty_policy,
    }


def format_eval_line(tag: str, d: dict) -> str:
    g = d["global"]
    pc = d["percase"]
    fp_rate = d["gt_empty_fp_rate"]
    fp_str = "N/A" if fp_rate is None else f"{fp_rate:.4f} (empty={d['gt_empty_total']}, fp_on_empty={d['gt_empty_fp']})"
    if pc["n"] == 0:
        pc_str = "percase=N/A"
    else:
        pc_str = (
            f"percase_mean={pc['mean']:.4f} med={pc['median']:.4f} "
            f"p25={pc['p25']:.4f} p75={pc['p75']:.4f} n={pc['n']} "
            f"empty_policy={d['empty_policy']} skipped_empty={d['percase_skipped_empty']}"
        )
    return (
        f"{tag}: loss={d['loss']:.4f} "
        f"global_dice={g['dice']:.4f} iou={g['iou']:.4f} "
        f"prec={g['precision']:.4f} rec={g['recall']:.4f} acc={g['accuracy']:.4f} "
        f"pred_fg%={d['pred_fg_ratio']*100:.2f}% gt_empty_fp_rate={fp_str} | {pc_str}"
    )


# -------------------------
# Main training (mixed source+target per step)
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--source_root", type=str, required=True, help="Source flat root (labeled)")
    ap.add_argument("--target_root", type=str, required=True, help="Target flat root (unlabeled for training)")
    ap.add_argument("--image_size", type=int, default=256)

    # train
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", type=str, default="./checkpoints_uda_step1")

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
            "attention_fft_unet2d_md2",
        ],
    )

    # regularization
    ap.add_argument("--weight_decay", type=float, default=2e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_aug", action="store_true")

    # teacher-ema & pseudo-label
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--lambda_pl", type=float, default=0.3)
    ap.add_argument("--pl_conf_thr", type=float, default=0.85)
    ap.add_argument("--pl_fg_min", type=float, default=0.002)
    ap.add_argument("--pl_fg_max", type=float, default=0.20)

    # monitoring/eval
    ap.add_argument("--dice_empty", type=str, default="skip", choices=["skip", "one", "zero"],
                    help="Per-case dice policy for GT-empty samples (monitoring only).")
    ap.add_argument("--eval_target", action="store_true",
                    help="If set, evaluate target val (requires target masks exist).")
    ap.add_argument("--eval_student", action="store_true",
                    help="If set, also evaluate student model (in addition to EMA teacher).")

    # NEW: drop target GT-empty (normal) for training/val, and lesion-only target reporting
    ap.add_argument("--tgt_drop_gt_empty_train", action="store_true",
                    help="If set, drop GT-empty cases from TARGET TRAIN by reading masks for filtering (not supervision).")
    ap.add_argument("--tgt_drop_gt_empty_val", action="store_true",
                    help="If set, drop GT-empty cases from TARGET VAL dataset (monitoring only).")
    ap.add_argument("--tgt_gt_empty_thr", type=int, default=0,
                    help="Mask sum <= thr is treated as GT-empty for dropping.")
    ap.add_argument("--tgt_val_lesion_only", action="store_true",
                    help="If set, print an extra Target Val line computed on lesion-only (GT-non-empty) via empty_policy=skip.")

    # checkpoint init (optional): load pretrained and start UDA
    ap.add_argument("--init_ckpt", type=str, default="", help="Optional: pretrained best.pt to initialize student weights")
    ap.add_argument("--init_strict", action="store_true", help="strict load for init_ckpt")

    args = ap.parse_args()
    seed_everything(args.seed)

    device = get_device()
    print("Device:", device)
    print("Backbone:", args.backbone)
    print("Source:", args.source_root)
    print("Target:", args.target_root)
    print("Augment:", (not args.no_aug))
    print(f"EMA decay={args.ema_decay}  lambda_pl={args.lambda_pl}")
    print(f"PL filter: conf>={args.pl_conf_thr} fg in [{args.pl_fg_min},{args.pl_fg_max}]")
    print(f"Eval: dice_empty={args.dice_empty} eval_target={bool(args.eval_target)} eval_student={bool(args.eval_student)}")
    print(
        "TargetFilter:",
        f"drop_train={bool(args.tgt_drop_gt_empty_train)} drop_val={bool(args.tgt_drop_gt_empty_val)}",
        f"thr={args.tgt_gt_empty_thr} tgt_val_lesion_only={bool(args.tgt_val_lesion_only)}",
    )

    save_dir = Path(args.save_dir) / args.backbone
    save_dir.mkdir(parents=True, exist_ok=True)

    # datasets
    src_train = FlatSegDatasetLabeled(args.source_root, "train", image_size=args.image_size, augment=(not args.no_aug))
    src_val   = FlatSegDatasetLabeled(args.source_root, "val",   image_size=args.image_size, augment=False)

    tgt_train = FlatSegDatasetUnlabeled(
        args.target_root, "train",
        image_size=args.image_size,
        augment=(not args.no_aug),
        drop_gt_empty=bool(args.tgt_drop_gt_empty_train),
        gt_empty_thr=int(args.tgt_gt_empty_thr),
        mask_dir_name="masks",
        verbose_filter=True,
    )

    # optional (monitoring) target val uses masks
    tgt_val = None
    if args.eval_target:
        tgt_val = FlatSegDatasetLabeled(args.target_root, "val", image_size=args.image_size, augment=False)
        if args.tgt_drop_gt_empty_val:
            # drop GT-empty from target val by reusing the unlabeled filter logic on the val split,
            # then wrap it as a labeled dataset by keeping only ids that are non-empty.
            # simplest: filter ids here and overwrite tgt_val.ids
            ids_all = tgt_val.ids
            kept = []
            dropped = 0
            msk_dir = Path(args.target_root) / "masks"
            assert msk_dir.exists(), f"tgt_drop_gt_empty_val=True but masks not found: {msk_dir}"
            for sid in ids_all:
                mp = msk_dir / f"{sid}.png"
                if not mp.exists():
                    raise FileNotFoundError(f"Mask not found for filtering target val: {mp}")
                m = load_mask(mp)
                if int(m.sum()) <= int(args.tgt_gt_empty_thr):
                    dropped += 1
                    continue
                kept.append(sid)
            tgt_val.ids = kept
            print(
                f"[TargetValFilter] drop_gt_empty_val=True thr={args.tgt_gt_empty_thr} "
                f"kept={len(kept)} dropped={dropped} (from {len(ids_all)})"
            )

    src_loader = DataLoader(
        src_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    tgt_loader = DataLoader(
        tgt_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    src_val_loader = DataLoader(
        src_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    tgt_val_loader = None
    if tgt_val is not None:
        tgt_val_loader = DataLoader(
            tgt_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

    # model
    model = build_model(args.backbone).to(device)

    # optional init from checkpoint
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        sd = ckpt.get("ema_model", None) or ckpt.get("model", None) or ckpt
        if sd is None or (not isinstance(sd, dict)):
            raise ValueError("init_ckpt missing a usable state_dict ('model'/'ema_model' or directly).")
        missing, unexpected = model.load_state_dict(sd, strict=bool(args.init_strict))
        print(f"[Init] loaded from {args.init_ckpt} strict={bool(args.init_strict)}")
        if not args.init_strict:
            print(f"  missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) > 0: print("  first missing:", missing[:10])
            if len(unexpected) > 0: print("  first unexpected:", unexpected[:10])

    # teacher EMA
    ema_model = copy_model(model).to(device)

    # losses
    sup_loss_fn = DiceCE(dice_w=1.0, ce_w=1.0, label_smoothing=args.label_smoothing)
    pl_loss_fn = nn.CrossEntropyLoss(reduction="none")  # per-pixel

    pl_cfg = PLFilterCfg(conf_thr=args.pl_conf_thr, fg_min=args.pl_fg_min, fg_max=args.pl_fg_max)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=6, threshold=1e-4, verbose=True
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_src_val = 0.0

    # iterators
    tgt_iter = iter(tgt_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ema_model.eval()

        # stats
        n_steps = 0
        sup_loss_sum = 0.0
        pl_loss_sum = 0.0
        pl_keep_sum = 0
        pl_total_sum = 0
        pl_conf_sum = 0.0
        pl_fg_sum = 0.0

        for (xs, ys, _) in src_loader:
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, _ = next(tgt_iter)

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            else:
                autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

            with autocast_ctx:
                # ---- source supervised ----
                out_s = model(xs)
                logits_s, _, _ = parse_model_output(out_s)
                loss_sup = sup_loss_fn(logits_s, ys)

                # ---- target pseudo-label (from EMA teacher) ----
                with torch.no_grad():
                    out_t_teacher = ema_model(xt)
                    logits_t_teacher, _, _ = parse_model_output(out_t_teacher)
                    y_hat, _, conf_mean, fg_ratio = make_pseudo_from_teacher(logits_t_teacher)
                    keep = filter_mask_from_stats(conf_mean, fg_ratio, pl_cfg)  # [B]

                out_t_student = model(xt)
                logits_t_student, _, _ = parse_model_output(out_t_student)  # [B,2,H,W]

                # CE per-pixel -> per-image so we can mask by keep
                ce_pix = pl_loss_fn(logits_t_student, y_hat)     # [B,H,W]
                ce_img = ce_pix.flatten(1).mean(dim=1)           # [B]

                if keep.any():
                    loss_pl = (ce_img[keep]).mean()
                else:
                    loss_pl = loss_sup.new_tensor(0.0)

                loss = loss_sup + float(args.lambda_pl) * loss_pl

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # EMA update
            ema_update(ema_model, model, decay=float(args.ema_decay))

            # stats
            n_steps += 1
            sup_loss_sum += float(loss_sup.item())
            pl_loss_sum += float(loss_pl.item()) if keep.any() else 0.0
            pl_keep_sum += int(keep.sum().item())
            pl_total_sum += int(keep.numel())
            pl_conf_sum += float(conf_mean.mean().item())
            pl_fg_sum += float(fg_ratio.mean().item())

        # ---- eval (monitoring) ----
        lr_now = opt.param_groups[0]["lr"]
        keep_rate = pl_keep_sum / max(1, pl_total_sum)

        # teacher/EMA eval on SOURCE
        src_eval_t = eval_labeled_full(
            ema_model, src_val_loader, device, empty_policy=args.dice_empty, prefix="SRC_VAL"
        )
        print(
            f"[Epoch {epoch:03d}] lr={lr_now:.2e} "
            f"sup_loss={sup_loss_sum/max(1,n_steps):.4f} "
            f"pl_loss={pl_loss_sum/max(1,n_steps):.4f} "
            f"keep={keep_rate*100:.1f}% "
            f"t_conf={pl_conf_sum/max(1,n_steps):.3f} "
            f"t_fg%={pl_fg_sum/max(1,n_steps)*100:.2f}"
        )
        print("  " + format_eval_line("EMA " + src_eval_t["prefix"], src_eval_t))

        # teacher/EMA eval on TARGET (monitoring only)
        if tgt_val_loader is not None:
            tgt_eval_t = eval_labeled_full(
                ema_model, tgt_val_loader, device, empty_policy=args.dice_empty, prefix="TGT_VAL"
            )
            print("  " + format_eval_line("EMA " + tgt_eval_t["prefix"], tgt_eval_t))

            # Extra: lesion-only target metrics (always empty_policy=skip)
            if args.tgt_val_lesion_only:
                tgt_eval_t_les = eval_labeled_full(
                    ema_model, tgt_val_loader, device, empty_policy="skip", prefix="TGT_VAL_LESION_ONLY"
                )
                print("  " + format_eval_line("EMA " + tgt_eval_t_les["prefix"], tgt_eval_t_les))

        # optional student eval (often noisier)
        if args.eval_student:
            src_eval_s = eval_labeled_full(
                model, src_val_loader, device, empty_policy=args.dice_empty, prefix="SRC_VAL"
            )
            print("  " + format_eval_line("STU " + src_eval_s["prefix"], src_eval_s))
            if tgt_val_loader is not None:
                tgt_eval_s = eval_labeled_full(
                    model, tgt_val_loader, device, empty_policy=args.dice_empty, prefix="TGT_VAL"
                )
                print("  " + format_eval_line("STU " + tgt_eval_s["prefix"], tgt_eval_s))
                if args.tgt_val_lesion_only:
                    tgt_eval_s_les = eval_labeled_full(
                        model, tgt_val_loader, device, empty_policy="skip", prefix="TGT_VAL_LESION_ONLY"
                    )
                    print("  " + format_eval_line("STU " + tgt_eval_s_les["prefix"], tgt_eval_s_les))

        # scheduler + best ckpt by SOURCE EMA dice (protect source performance)
        scheduler.step(src_eval_t["global"]["dice"])

        if src_eval_t["global"]["dice"] > best_src_val + 1e-6:
            best_src_val = float(src_eval_t["global"]["dice"])
            ckpt = {
                "epoch": epoch,
                "best_src_val_dice": best_src_val,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "opt": opt.state_dict(),
                "backbone": args.backbone,
                "image_size": args.image_size,
                "ema_decay": args.ema_decay,
                "lambda_pl": args.lambda_pl,
                "pl_cfg": vars(pl_cfg),
                "dice_empty": args.dice_empty,
                "eval_target": bool(args.eval_target),
                "tgt_drop_gt_empty_train": bool(args.tgt_drop_gt_empty_train),
                "tgt_drop_gt_empty_val": bool(args.tgt_drop_gt_empty_val),
                "tgt_gt_empty_thr": int(args.tgt_gt_empty_thr),
                "tgt_val_lesion_only": bool(args.tgt_val_lesion_only),
            }
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ✓ saved best.pt (best_src_val_dice={best_src_val:.4f})")

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "ema_model": ema_model.state_dict()},
                save_dir / f"epoch_{epoch:03d}.pt",
            )

    print("Done. Best source val dice:", best_src_val)


if __name__ == "__main__":
    main()