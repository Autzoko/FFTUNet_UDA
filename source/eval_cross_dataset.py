# eval_cross_dataset.py
# Cross-dataset evaluation for BUSI-trained checkpoint on BUSBRA (or any flat dataset)
# Upgrades (this version):
# 1) Global (pixel-wise) metrics via accumulated confusion (same as your current)
# 2) Per-case Dice (image-wise) + mean/median/std + percentiles
#    - Handles GT-empty cases explicitly:
#      * dice_empty="skip": exclude GT-empty images from per-case mean (recommended for BUSBRA no-normal)
#      * dice_empty="one":  if GT empty and Pred empty -> dice=1 else 0
#      * dice_empty="zero": if GT empty -> dice=0 (harsh; usually not desired)
# 3) Keeps your EMA preference, robust output parsing, TTA, and visualization behavior

import argparse
from pathlib import Path
import random
import math

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Backbones (same as training)
# -------------------------
from source.models.backbone.unet2d import UNet2D
from source.models.backbone.attention_unet2d import AttentionUNet2D
from source.models.backbone.resenc_unet2d import ResEncUNet2D
from source.models.backbone.attention_fft_unet2d import AttentionUNet2D as AttentionFFTUNet2D
from source.models.backbone.attnft_unet_md import AttentionUNet2D as AttentionFFTUNet2D_MD

try:
    # optional (if you trained with MD2)
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
    # returns float32 0..255
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    # returns uint8 0/1
    m = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (m > 0).astype(np.uint8)


def resize_np(img: np.ndarray, size: int, is_mask: bool):
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


def parse_model_output(model_out):
    """
    Returns logits: Tensor [B,C,H,W]
    Supports:
      - logits tensor
      - (logits, feats, ...) tuple/list
      - {"logits":..., ...} or dict with any tensor value
    """
    if isinstance(model_out, dict):
        if "logits" in model_out and torch.is_tensor(model_out["logits"]):
            return model_out["logits"]
        for v in model_out.values():
            if torch.is_tensor(v):
                return v
        raise ValueError("Model returned dict but no tensor logits found.")
    if isinstance(model_out, (tuple, list)):
        if len(model_out) == 0:
            raise ValueError("Model returned empty tuple/list.")
        if not torch.is_tensor(model_out[0]):
            raise ValueError("Model returned tuple/list but first element is not a tensor.")
        return model_out[0]
    if torch.is_tensor(model_out):
        return model_out
    raise ValueError(f"Unsupported model output type: {type(model_out)}")


def build_model(backbone: str, in_channels=1, num_classes=2):
    b = backbone.lower()
    if b in ["unet", "unet2d"]:
        return UNet2D(in_channels=in_channels, num_classes=num_classes)
    if b in ["attunet", "attentionunet", "attention_unet", "attention_unet2d"]:
        return AttentionUNet2D(in_channels=in_channels, num_classes=num_classes)
    if b in ["resenc_unet2d", "resenc", "nnunet"]:
        return ResEncUNet2D(in_channels=in_channels, num_classes=num_classes)
    if b in ["attention_fft_unet2d", "attention_fft_unet"]:
        return AttentionFFTUNet2D(in_channels=in_channels, num_classes=num_classes)
    if b in ["attnft_unet_md", "attention_fft_unet2d_md", "attention_fft_md", "md"]:
        return AttentionFFTUNet2D_MD(in_channels=in_channels, num_classes=num_classes)
    if b in ["attention_fft_unet2d_md2", "attnft_unet_md2", "md2"]:
        if AttentionFFTUNet2D_MD2 is None:
            raise ValueError("Backbone md2 requested but AttentionFFTUNet2D_MD2 import failed.")
        return AttentionFFTUNet2D_MD2(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown backbone: {backbone}")


def pick_state_dict_from_ckpt(ckpt: dict, prefer_ema: bool = True) -> tuple[dict, str]:
    """
    Prefer EMA weights if present and non-empty.
    Returns (state_dict, which)
    """
    if prefer_ema and ckpt.get("ema_model", None) is not None:
        ema_sd = ckpt["ema_model"]
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            return ema_sd, "ema_model"
    return ckpt["model"], "model"


# -------------------------
# Dataset (flat)
# -------------------------
class FlatSegDataset(Dataset):
    """
    Expected structure:
      root/
        images/{id}.png
        masks/{id}.png
        splits/{split}.txt
    """
    def __init__(self, root: str, split: str, image_size: int = 256):
        self.root = Path(root)
        self.image_size = int(image_size)

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

        img = load_gray(ip)          # 0..255 float32
        msk = load_mask(mp)          # 0/1 uint8

        img = resize_np(img, self.image_size, is_mask=False)
        msk = resize_np(msk, self.image_size, is_mask=True)

        img = (img / 255.0).astype(np.float32)

        x = torch.from_numpy(img).unsqueeze(0).float()  # [1,H,W]
        y = torch.from_numpy(msk).long()                # [H,W]
        return x, y, sid


# -------------------------
# Metrics
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
    returns list of dice floats, one per selected case
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
            # empty_policy == "one"
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


# -------------------------
# Visualization
# -------------------------
def make_overlay_rgb(gray01: np.ndarray, gt01: np.ndarray, pr01: np.ndarray,
                     alpha_gt: float = 0.35, alpha_pr: float = 0.35):
    """
    gray01: H,W in [0,1]
    gt01, pr01: H,W in {0,1}
    GT: green, Pred: red
    """
    g = (np.clip(gray01, 0, 1) * 255.0).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)

    gt = (gt01 > 0).astype(np.float32)[..., None]
    pr = (pr01 > 0).astype(np.float32)[..., None]

    green = np.array([0, 255, 0], dtype=np.float32)[None, None, :]
    red   = np.array([255, 0, 0], dtype=np.float32)[None, None, :]

    rgb = rgb * (1 - alpha_gt * gt) + green * (alpha_gt * gt)
    rgb = rgb * (1 - alpha_pr * pr) + red * (alpha_pr * pr)

    return np.clip(rgb, 0, 255).astype(np.uint8)


@torch.no_grad()
def save_vis_samples(out_dir: Path, x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, sids):
    out_dir.mkdir(parents=True, exist_ok=True)
    x_np = x.detach().cpu().numpy()      # [B,1,H,W]
    y_np = y.detach().cpu().numpy()      # [B,H,W]
    p_np = pred.detach().cpu().numpy()   # [B,H,W]

    for i, sid in enumerate(sids):
        gray = x_np[i, 0]
        gt = y_np[i]
        pr = p_np[i]
        img = make_overlay_rgb(gray, gt, pr, alpha_gt=0.35, alpha_pr=0.35)
        Image.fromarray(img).save(out_dir / f"{sid}_overlay.png")


# -------------------------
# Inference helpers
# -------------------------
@torch.no_grad()
def predict_probs(model, x: torch.Tensor, tta_flip: bool = False):
    """
    Returns probabilities [B,2,H,W] (float32) for stability.
    """
    out = model(x)
    logits = parse_model_output(out)
    p = F.softmax(logits, dim=1)

    if not tta_flip:
        return p

    x_f = torch.flip(x, dims=[3])
    out_f = model(x_f)
    logits_f = parse_model_output(out_f)
    logits_f = torch.flip(logits_f, dims=[3])
    p_f = F.softmax(logits_f, dim=1)

    return 0.5 * (p + p_f)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt from training")
    ap.add_argument("--data_root", type=str, required=True, help="Target dataset flat root")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="", help="Optional override: cuda/mps/cpu")
    ap.add_argument("--tta_flip", action="store_true", help="Test-time augmentation: hflip + average probs")
    ap.add_argument("--prefer_ema", action="store_true", help="Prefer EMA weights if checkpoint contains them")
    ap.add_argument("--strict", action="store_true", help="Use strict=True when loading state_dict")
    ap.add_argument("--save_vis", action="store_true", help="Save overlay visualizations")
    ap.add_argument("--vis_n", type=int, default=24, help="How many images to visualize (random sample)")
    ap.add_argument("--out_dir", type=str, default="eval_vis", help="Root output dir for visualizations")
    ap.add_argument("--dice_empty", type=str, default="skip", choices=["skip", "one", "zero"],
                    help="Per-case dice policy for GT-empty samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    device = get_device()
    if args.device:
        device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    backbone = ckpt.get("backbone", None)
    image_size = int(ckpt.get("image_size", 256))
    if backbone is None:
        raise ValueError("Checkpoint missing 'backbone'. Please save it in training ckpt.")

    model = build_model(backbone, in_channels=1, num_classes=2).to(device)

    state_dict, which = pick_state_dict_from_ckpt(ckpt, prefer_ema=bool(args.prefer_ema))
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(args.strict))
    model.eval()

    print(f"Loaded ckpt: {ckpt_path}")
    print(f"  backbone={backbone}  image_size={image_size}  weights={which}  strict={bool(args.strict)}")
    if not args.strict:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("    (first 10 missing keys):", missing[:10])
        if len(unexpected) > 0:
            print("    (first 10 unexpected keys):", unexpected[:10])

    print(f"Eval on: {args.data_root} split={args.split} device={device} tta_flip={bool(args.tta_flip)}")
    print(f"Per-case dice: dice_empty_policy={args.dice_empty}")

    ds = FlatSegDataset(args.data_root, args.split, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Global confusion totals (pixel-wise)
    tp = torch.tensor(0.0)
    fp = torch.tensor(0.0)
    fn = torch.tensor(0.0)
    tn = torch.tensor(0.0)

    # GT-empty FP rate (image-wise)
    gt_empty_total = 0
    gt_empty_fp = 0

    # Predicted FG ratio (pixel-wise), averaged per-image
    pred_fg_sum = 0.0
    pred_fg_count = 0

    # Per-case dice list
    percase_dices: list[float] = []
    percase_included = 0
    percase_skipped_empty = 0

    # Visualization sampling (sample indices in dataset space)
    want_vis = bool(args.save_vis)
    vis_dir = Path(args.out_dir) / f"{Path(args.data_root).name}_{args.split}"
    vis_indices = set()
    if want_vis:
        n = len(ds)
        take = min(int(args.vis_n), n)
        vis_indices = set(random.sample(range(n), k=take))

    idx_base = 0
    with torch.no_grad():
        for x, y, sids in loader:
            x = x.to(device, non_blocking=True)  # [B,1,H,W]
            y = y.to(device, non_blocking=True)  # [B,H,W]

            p = predict_probs(model, x, tta_flip=bool(args.tta_flip))  # [B,2,H,W]
            pred = torch.argmax(p, dim=1)  # [B,H,W], 0/1

            # accumulate global confusion
            btp, bfp, bfn, btn = batch_confusion(pred, y)
            tp += btp.cpu()
            fp += bfp.cpu()
            fn += bfn.cpu()
            tn += btn.cpu()

            # pred fg ratio (per-image average)
            pred_fg_sum += float(pred.float().mean().item()) * x.size(0)
            pred_fg_count += x.size(0)

            # GT-empty FP rate (per-sample)
            for bi in range(x.size(0)):
                yi = y[bi]
                pi = pred[bi]
                if yi.sum().item() == 0:
                    gt_empty_total += 1
                    if pi.sum().item() > 0:
                        gt_empty_fp += 1

            # per-case dice
            before = len(percase_dices)
            percase_dices.extend(per_case_dice(pred, y, empty_policy=args.dice_empty))
            after = len(percase_dices)

            # bookkeeping for "skip"
            if args.dice_empty == "skip":
                # count how many GT-empty were skipped in this batch
                for bi in range(x.size(0)):
                    if int(y[bi].sum().item()) == 0:
                        percase_skipped_empty += 1
                percase_included += (after - before)
            else:
                percase_included += x.size(0)

            # visualization: save selected samples by global dataset index
            if want_vis:
                selected = []
                selected_sids = []
                for bi in range(x.size(0)):
                    gidx = idx_base + bi
                    if gidx in vis_indices:
                        selected.append(bi)
                        selected_sids.append(sids[bi])
                if selected:
                    xb = x[selected]
                    yb = y[selected]
                    pb = pred[selected]
                    save_vis_samples(vis_dir, xb, yb, pb, selected_sids)

            idx_base += x.size(0)

    # final global metrics
    scalars = metrics_from_totals(tp, fp, fn, tn)
    pred_fg_ratio = (pred_fg_sum / max(1, pred_fg_count))
    fp_rate_empty = (gt_empty_fp / gt_empty_total) if gt_empty_total > 0 else None

    # per-case summary
    percase_summary = summarize_list(percase_dices)

    print("\n===== Cross-dataset evaluation (GLOBAL pixel-wise) =====")
    print(f"Dice(FG):   {scalars['dice']:.4f}")
    print(f"IoU:        {scalars['iou']:.4f}")
    print(f"Precision:  {scalars['precision']:.4f}")
    print(f"Recall:     {scalars['recall']:.4f}")
    print(f"Accuracy:   {scalars['accuracy']:.4f}")
    print(f"Pred FG %:  {pred_fg_ratio*100:.2f}%")
    if fp_rate_empty is not None:
        print(f"GT-empty FP rate: {fp_rate_empty:.4f}  (empty={gt_empty_total}, fp_on_empty={gt_empty_fp})")
    else:
        print("GT-empty FP rate: N/A (no empty GT samples in this split)")

    print("\n===== Per-case Dice (image-wise) =====")
    print(f"Policy (GT-empty): {args.dice_empty}")
    if args.dice_empty == "skip":
        print(f"Included cases: {percase_summary['n']}  | Skipped GT-empty: {percase_skipped_empty}")
    else:
        print(f"Included cases: {percase_summary['n']}")

    if percase_summary["n"] == 0:
        print("Per-case Dice: N/A (no included cases under current policy)")
    else:
        print(f"Mean:   {percase_summary['mean']:.4f}")
        print(f"Median: {percase_summary['median']:.4f}")
        print(f"Std:    {percase_summary['std']:.4f}")
        print(f"P10/P25/P75/P90: {percase_summary['p10']:.4f} / {percase_summary['p25']:.4f} / "
              f"{percase_summary['p75']:.4f} / {percase_summary['p90']:.4f}")

    if want_vis:
        print(f"\nSaved overlays to: {vis_dir.resolve()}")


if __name__ == "__main__":
    main()