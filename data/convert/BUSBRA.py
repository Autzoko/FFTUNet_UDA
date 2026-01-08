import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def binarize_mask_to_uint8(mask_path: Path) -> Image.Image:
    """
    Read mask and binarize to 0/255 (PIL L).
    """
    m = Image.open(mask_path).convert("L")
    m = m.point(lambda x: 255 if x > 0 else 0)
    return m


def safe_id(s: str) -> str:
    s = s.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\(\)]", "_", s)
    return s


def bus_to_mask_name(bus_name: str) -> str:
    """
    bus_0001-l.png -> mask_0001-l.png
    """
    if bus_name.startswith("bus_"):
        return "mask_" + bus_name[len("bus_"):]
    # fallback: replace first occurrence
    return bus_name.replace("bus", "mask", 1)


def extract_case_id(stem: str) -> str:
    """
    From "bus_0001-l" get "0001".
    From "bus_0123_r" get "0123".
    We try a few patterns; fallback to whole stem.
    """
    # common patterns: bus_0001-l / bus_0001_r / bus_0001s
    m = re.search(r"bus_(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def find_pairs(images_dir: Path, masks_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """
    Return:
      - pairs: list of (image_path, mask_path) where mask exists
      - missing_masks: list of image_path where mask not found
    """
    imgs = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])
    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []

    for ip in imgs:
        mask_name = bus_to_mask_name(ip.name)
        mp = masks_dir / mask_name
        if mp.exists():
            pairs.append((ip, mp))
        else:
            missing.append(ip)

    return pairs, missing


def split_by_case(all_sids: List[str], sid_to_case: Dict[str, str], val_ratio: float, seed: int):
    """
    Split by unique case_id to avoid leakage between train/val.
    """
    cases = sorted(list(set(sid_to_case[sid] for sid in all_sids)))
    random.seed(seed)
    random.shuffle(cases)

    n_val_cases = int(round(len(cases) * val_ratio))
    val_cases = set(cases[:n_val_cases])

    train_ids, val_ids = [], []
    for sid in all_sids:
        if sid_to_case[sid] in val_cases:
            val_ids.append(sid)
        else:
            train_ids.append(sid)
    return train_ids, val_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busbra_root", type=str, required=True, help="Path to BUSBRA root containing Images/ and Masks/")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (flat structure: images/masks/splits)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_by_case", action="store_true", default=True,
                    help="Split by case id (bus_0001) to avoid leakage. Default True.")
    ap.add_argument("--no_split_by_case", action="store_true",
                    help="Disable split-by-case; will split by image (not recommended).")
    args = ap.parse_args()

    busbra_root = Path(args.busbra_root)
    images_dir = busbra_root / "Images"
    masks_dir = busbra_root / "Masks"
    assert images_dir.exists() and masks_dir.exists(), f"Expect Images/ and Masks/ under {busbra_root}"

    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    split_out = out_dir / "splits"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)
    split_out.mkdir(parents=True, exist_ok=True)

    pairs, missing = find_pairs(images_dir, masks_dir)

    if len(pairs) == 0:
        raise RuntimeError(f"No (image,mask) pairs found. Check naming rules in {images_dir} / {masks_dir}")

    all_records = []
    all_ids = []
    sid_to_case: Dict[str, str] = {}

    for ip, mp in pairs:
        # Build sid: busbra_<stem>  (keep -l/-r/-s etc.)
        sid = safe_id(f"busbra_{ip.stem}")
        img_dst = img_out / f"{sid}.png"
        msk_dst = msk_out / f"{sid}.png"

        img = Image.open(ip).convert("L")
        img.save(img_dst)

        m = binarize_mask_to_uint8(mp)
        if m.size != img.size:
            raise ValueError(f"Image/mask size mismatch: image={ip} {img.size}, mask={mp} {m.size}")
        m.save(msk_dst)

        all_ids.append(sid)
        sid_to_case[sid] = extract_case_id(ip.stem)

        all_records.append({
            "id": sid,
            "case_id": sid_to_case[sid],
            "image_src": str(ip),
            "mask_src": str(mp),
            "image_dst": str(img_dst),
            "mask_dst": str(msk_dst),
        })

    # Split
    random.seed(args.seed)
    do_split_by_case = (not args.no_split_by_case)

    if do_split_by_case:
        train_ids, val_ids = split_by_case(all_ids, sid_to_case, args.val_ratio, args.seed)
    else:
        ids = all_ids[:]
        random.shuffle(ids)
        n_val = int(round(len(ids) * args.val_ratio))
        val_ids = ids[:n_val]
        train_ids = ids[n_val:]

    (split_out / "train.txt").write_text("\n".join(train_ids) + "\n")
    (split_out / "val.txt").write_text("\n".join(val_ids) + "\n")

    # meta.csv
    with (out_dir / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "case_id", "image_src", "mask_src", "image_dst", "mask_dst"])
        w.writeheader()
        for r in all_records:
            w.writerow(r)

    print(f"Done. Total={len(all_ids)} train={len(train_ids)} val={len(val_ids)}")
    if do_split_by_case:
        print(f"Split mode: BY_CASE (unique cases={len(set(sid_to_case.values()))})")
    else:
        print("Split mode: BY_IMAGE (not recommended)")
    if missing:
        print(f"[WARN] Missing masks for {len(missing)} images. Example: {missing[0]}")
    print(f"Output: {out_dir}")
    print(f"  images/: {img_out}")
    print(f"  masks/ : {msk_out}")
    print(f"  splits/: {split_out}")
    print(f"  meta.csv: {out_dir / 'meta.csv'}")


if __name__ == "__main__":
    main()