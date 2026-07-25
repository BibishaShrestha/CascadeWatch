"""Convert a raster-mask segmentation dataset into YOLOv8-seg polygon label
format, preserving its existing train/validation/test split.

Point this at any dataset shaped like
<dataset-dir>/{train,validation,test}/{images,masks}/*.png and it produces
<dataset-dir>_yolo/ the same way.

Input:  <dataset-dir>/{train,validation,test}/{images,masks}/*.png
Output: <dataset-dir>_yolo/{images,labels}/{train,val,test}/... + data.yaml

Each foreground blob in a mask becomes its own polygon instance (masks can
contain multiple disjoint landslide regions). Contours smaller than
MIN_CONTOUR_AREA_PX are dropped as noise. Coordinates are simplified with
approxPolyDP (light touch, ~0.5% of perimeter) to avoid label files full of
raw pixel-staircase points, then normalized to [0, 1].
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from yolo_seg_utils import mask_to_polygons, write_data_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

CLASS_ID = 0


def convert_split(src_root: Path, dst_root: Path, src_split: str, dst_split: str, stats: dict) -> None:
    img_dir = src_root / src_split / "images"
    mask_dir = src_root / src_split / "masks"
    dst_img_dir = dst_root / "images" / dst_split
    dst_label_dir = dst_root / "labels" / dst_split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_instances = 0
    n_empty = 0

    for img_path in sorted(img_dir.glob("*.png")):
        suffix = img_path.name.split("_", 1)[1]
        mask_path = mask_dir / f"mask_{suffix}"
        if not mask_path.exists():
            continue

        polygons = mask_to_polygons(mask_path)

        Image.open(img_path).convert("RGB").save(dst_img_dir / img_path.name)

        label_path = dst_label_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            for poly in polygons:
                coords = " ".join(f"{c:.6f}" for c in poly)
                f.write(f"{CLASS_ID} {coords}\n")

        n_images += 1
        n_instances += len(polygons)
        if not polygons:
            n_empty += 1

    stats[dst_split] = {"images": n_images, "instances": n_instances, "empty_labels": n_empty}


def convert_dataset(dataset_dir: Path) -> Path:
    dst_root = dataset_dir.parent / f"{dataset_dir.name}_yolo"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    stats: dict[str, dict] = {}
    for src_split, dst_split in SPLIT_MAP.items():
        convert_split(dataset_dir, dst_root, src_split, dst_split, stats)
    write_data_yaml(dst_root)

    print(f"YOLO-seg dataset written to: {dst_root}")
    for split, s in stats.items():
        print(f"  {split}: {s['images']} images, {s['instances']} polygon instances, "
              f"{s['empty_labels']} images with zero instances")
    return dst_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to a dataset shaped like <dir>/{train,validation,test}/{images,masks}/*.png "
             "(absolute, or relative to datasets/)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / "datasets" / dataset_dir
    convert_dataset(dataset_dir)


if __name__ == "__main__":
    main()
