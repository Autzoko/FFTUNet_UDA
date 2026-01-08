import argparse
import csv
import random
import re
from pathlib import Path
from typing import Tuple, List, Dict

from PIL import Image, ImageChops


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def looks_like_mask(name: str) -> bool:
    # 常见 mask 命名：*_mask.png, *_mask_1.png, mask_*.png 等
    n = name.lower()
    return ("mask" in n) or ("seg" in n) or ("label" in n)


def strip_mask_tokens(stem: str) -> str:
    """
    把 stem 里的常见 mask 后缀去掉，得到“基础 id”
    e.g. "benign (12)_mask" -> "benign (12)"
         "xxx_mask_1" -> "xxx"
    """
    s = stem
    s = re.sub(r"(_mask(_\d+)?)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(mask(_\d+)?)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(_seg(_\d+)?)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(_label(_\d+)?)$", "", s, flags=re.IGNORECASE)
    return s


def binarize_mask_to_uint8(mask_path: Path) -> Image.Image:
    """
    读取 mask 并二值化为 0/255，返回 PIL Image(L)
    """
    m = Image.open(mask_path).convert("L")
    # 二值化：>0 -> 255
    m = m.point(lambda x: 255 if x > 0 else 0)
    return m


def union_masks(mask_paths: List[Path]) -> Image.Image:
    """
    将多个 mask 做 OR 合并，返回 0/255 的 PIL(L).
    - 自动把所有 mask 二值化
    - 若尺寸不一致：直接报错（避免悄悄对齐导致错误）
    """
    assert len(mask_paths) > 0, "union_masks got empty list"

    base = binarize_mask_to_uint8(mask_paths[0])
    w, h = base.size

    for mp in mask_paths[1:]:
        m = binarize_mask_to_uint8(mp)
        if m.size != (w, h):
            raise ValueError(f"Mask size mismatch: {mp} has {m.size}, expected {(w, h)}")
        # OR 合并（0/255）
        base = ImageChops.lighter(base, m)

    return base


def find_pairs_in_folder(folder: Path) -> List[Tuple[Path, List[Path]]]:
    """
    在一个类别文件夹里找 (image, [mask1,mask2,...]) pairs。
    策略：
      - 先分 image candidates / mask candidates（按文件名含 mask）
      - mask 根据 stem 去掉 mask token 后，与 image stem 匹配
      - 若同一 image 对多个 mask（_mask_1, _mask_2），全部收集并 union
    """
    files = [p for p in folder.rglob("*") if p.is_file() and is_image_file(p)]
    img_candidates = [p for p in files if not looks_like_mask(p.name)]
    mask_candidates = [p for p in files if looks_like_mask(p.name)]

    # 建 mask 映射：base_stem -> [mask_paths...]
    mask_map: Dict[str, List[Path]] = {}
    for mp in mask_candidates:
        base = strip_mask_tokens(mp.stem)
        mask_map.setdefault(base, []).append(mp)

    # 排序保证稳定
    for k in list(mask_map.keys()):
        mask_map[k] = sorted(mask_map[k])

    pairs: List[Tuple[Path, List[Path]]] = []
    for ip in img_candidates:
        base = ip.stem

        # 1) 优先直接匹配
        if base in mask_map and len(mask_map[base]) > 0:
            pairs.append((ip, mask_map[base]))
            continue

        # 2) 保守 fallback：包含关系匹配（可能存在空格/括号差异）
        matched_key = None
        for k in mask_map.keys():
            if k == base:
                continue
            if (k in base) or (base in k):
                matched_key = k
                break

        if matched_key is not None and len(mask_map[matched_key]) > 0:
            pairs.append((ip, mask_map[matched_key]))

    return pairs


def safe_id(s: str) -> str:
    # 生成可作为文件名的一部分
    s = s.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\(\)]", "_", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi_root", type=str, required=True,
                    help="Path to BUSI root folder containing benign/malignant/normal")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory (nnUNet-like flattened structure)")
    ap.add_argument("--include_normal", action="store_true", default=True,
                    help="Include normal class (default: True). If you want to EXCLUDE normal, pass --no_normal")
    ap.add_argument("--no_normal", action="store_true",
                    help="Exclude normal class")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    busi_root = Path(args.busi_root)
    out_dir = Path(args.out_dir)
    include_normal = (not args.no_normal)  # 默认要 normal

    # 类别目录
    class_dirs = []
    for cname in ["benign", "malignant", "normal"]:
        p = busi_root / cname
        if p.exists() and p.is_dir():
            if cname == "normal" and not include_normal:
                continue
            class_dirs.append((cname, p))

    assert class_dirs, f"No class folders found under {busi_root}. Expect benign/malignant/normal."

    # 输出目录
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    split_out = out_dir / "splits"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)
    split_out.mkdir(parents=True, exist_ok=True)

    all_records = []  # for meta.csv
    all_ids = []

    # 收集 pairs 并写出
    for cname, cdir in class_dirs:
        pairs = find_pairs_in_folder(cdir)
        if len(pairs) == 0:
            print(f"[WARN] No pairs found in {cdir}")
            continue

        for ip, mps in pairs:
            if len(mps) == 0:
                continue

            # 统一 sample id：类别前缀 + 原始 stem
            sid = safe_id(f"busi_{cname}_{ip.stem}")
            img_dst = img_out / f"{sid}.png"
            msk_dst = msk_out / f"{sid}.png"

            # 保存 image（转 L）
            img = Image.open(ip).convert("L")
            img.save(img_dst)

            # ✅ 保存 union mask（二值 0/255）
            um = union_masks(mps)
            # 安全检查：确保尺寸与 image 一致（不一致就报错，避免 silent bug）
            if um.size != img.size:
                raise ValueError(f"Image/mask size mismatch: image={ip} {img.size}, union_mask={um.size}")
            um.save(msk_dst)

            all_ids.append(sid)
            all_records.append({
                "id": sid,
                "class": cname,
                "image_src": str(ip),
                # 把所有 mask 源路径用 '|' 连接写入，便于追溯
                "mask_src": "|".join([str(p) for p in mps]),
                "image_dst": str(img_dst),
                "mask_dst": str(msk_dst),
                "num_masks_merged": str(len(mps)),
            })

    # 划分 train/val
    random.seed(args.seed)
    random.shuffle(all_ids)

    n_val = int(round(len(all_ids) * args.val_ratio))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]

    (split_out / "train.txt").write_text("\n".join(train_ids) + "\n")
    (split_out / "val.txt").write_text("\n".join(val_ids) + "\n")

    # 写 meta.csv
    with (out_dir / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "class", "image_src", "mask_src", "num_masks_merged", "image_dst", "mask_dst"]
        )
        w.writeheader()
        for r in all_records:
            w.writerow(r)

    print(f"Done. Total={len(all_ids)} train={len(train_ids)} val={len(val_ids)} include_normal={include_normal}")
    print(f"Output: {out_dir}")
    print(f"  images/: {img_out}")
    print(f"  masks/ : {msk_out}")
    print(f"  splits/: {split_out}")
    print(f"  meta.csv: {out_dir / 'meta.csv'}")


if __name__ == "__main__":
    main()