"""Redraw converted YOLO-seg polygons onto their source images and save
side-by-side (original mask | polygon overlay) visualizations, to visually
confirm the mask->polygon conversion is correct before training.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "datasets" / "landslide_seg"
YOLO_ROOT = REPO_ROOT / "datasets" / "landslide_seg_yolo"
OUT_DIR = REPO_ROOT / "outputs" / "figures" / "yolo_conversion_sanity_check"

SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}
N_PER_SPLIT = 4


def load_label_polygons(label_path: Path, w: int, h: int) -> list[list[tuple[float, float]]]:
    polygons = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        coords = [float(x) for x in parts[1:]]
        pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
        polygons.append(pts)
    return polygons


def main() -> None:
    random.seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for src_split, dst_split in SPLIT_MAP.items():
        img_dir = YOLO_ROOT / "images" / dst_split
        label_dir = YOLO_ROOT / "labels" / dst_split
        mask_dir = SRC_ROOT / src_split / "masks"

        candidates = [p for p in sorted(img_dir.glob("*.png"))
                      if (label_dir / (p.stem + ".txt")).stat().st_size > 0]
        sample = random.sample(candidates, min(N_PER_SPLIT, len(candidates)))

        for img_path in sample:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            polygons = load_label_polygons(label_dir / (img_path.stem + ".txt"), w, h)

            overlay = img.copy()
            draw = ImageDraw.Draw(overlay)
            for poly in polygons:
                draw.polygon(poly, outline=(255, 0, 0), width=1)

            suffix = img_path.stem.split("_", 1)[1]
            mask_path = mask_dir / f"mask_{suffix}.png"
            mask_img = Image.open(mask_path).convert("RGB")

            combo = Image.new("RGB", (w * 3, h), (0, 0, 0))
            combo.paste(img, (0, 0))
            combo.paste(mask_img, (w, 0))
            combo.paste(overlay, (w * 2, 0))
            combo = combo.resize((w * 3 * 3, h * 3), Image.NEAREST)

            out_path = OUT_DIR / f"{dst_split}_{img_path.stem}.png"
            combo.save(out_path)
            print("saved", out_path, f"({len(polygons)} polygons)")


if __name__ == "__main__":
    main()
