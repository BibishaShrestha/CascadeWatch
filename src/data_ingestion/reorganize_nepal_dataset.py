"""Reorganize the raw nepal-landslide-dataset/ (messy, inconsistent split
naming) into datasets/nepal_landslide_seg/{train,validation,test}/{images,masks}
matching the same convention as datasets/landslide_seg/.

Raw layout quirks handled here:
- Training: images in `image/` prefixed `b_`, masks in `label/` suffixed
  `_mask`, matched by stripping both (e.g. b_COMP_0_11.tiff <-> COMP_0_11_mask.png).
- Validation/Test: images/masks share a plain numeric stem directly
  (e.g. 0.tiff <-> 0.png), but folder names differ (images/masks vs image/mask).
- Masks are single-channel palette PNGs valued {0, 1}, not {0, 255} - are
  converted to {0, 255} here so downstream code doesn't need to special-case
  this dataset's threshold.
- Images are per-landslide crops of individually varying size (not a fixed
  grid), converted from TIFF to PNG for consistency with landslide_seg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nepal-landslide-dataset"
DST_ROOT = REPO_ROOT / "datasets" / "nepal_landslide_seg"


def convert_mask(src_path: Path, dst_path: Path) -> None:
    arr = np.array(Image.open(src_path).convert("L"))
    binary = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(dst_path)


def convert_image(src_path: Path, dst_path: Path) -> None:
    Image.open(src_path).convert("RGB").save(dst_path)


def reorganize_training() -> int:
    img_dir = SRC_ROOT / "Training" / "image"
    label_dir = SRC_ROOT / "Training" / "label"
    dst_img_dir = DST_ROOT / "train" / "images"
    dst_mask_dir = DST_ROOT / "train" / "masks"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path in sorted(img_dir.glob("b_*.tiff")):
        core = img_path.stem[len("b_"):]
        label_path = label_dir / f"{core}_mask.png"
        if not label_path.exists():
            continue
        convert_image(img_path, dst_img_dir / f"image_{core}.png")
        convert_mask(label_path, dst_mask_dir / f"mask_{core}.png")
        n += 1
    return n


def reorganize_simple(split_src: str, img_subdir: str, mask_subdir: str, dst_split: str) -> int:
    img_dir = SRC_ROOT / split_src / img_subdir
    mask_dir = SRC_ROOT / split_src / mask_subdir
    dst_img_dir = DST_ROOT / dst_split / "images"
    dst_mask_dir = DST_ROOT / dst_split / "masks"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path in sorted(img_dir.glob("*.tiff")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            continue
        convert_image(img_path, dst_img_dir / f"image_{stem}.png")
        convert_mask(mask_path, dst_mask_dir / f"mask_{stem}.png")
        n += 1
    return n


def main() -> None:
    n_train = reorganize_training()
    n_val = reorganize_simple("Validation", "images", "masks", "validation")
    n_test = reorganize_simple("Test", "image", "mask", "test")
    print(f"train: {n_train} pairs, validation: {n_val} pairs, test: {n_test} pairs")
    print(f"written to {DST_ROOT}")


if __name__ == "__main__":
    main()
