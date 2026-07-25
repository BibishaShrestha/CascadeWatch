"""Shared raster-mask -> YOLOv8-seg polygon conversion, used by
prepare_dataset.py for any dataset shaped like
<dataset-dir>/{train,validation,test}/{images,masks}/*.png.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MIN_CONTOUR_AREA_PX = 10.0
APPROX_EPSILON_FRAC = 0.005


def mask_to_polygons(mask_path: Path) -> list[list[float]]:
    """Return a list of normalized polygons (each a flat [x1,y1,x2,y2,...] list in 0..1).

    Each disjoint foreground blob becomes its own polygon instance. Contours
    smaller than MIN_CONTOUR_AREA_PX are dropped as noise. Coordinates are
    simplified with approxPolyDP (~0.5% of perimeter) before normalizing.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = mask.shape
    binary = (mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA_PX:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = APPROX_EPSILON_FRAC * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3:
            simplified = contour
        if len(simplified) < 3:
            continue
        flat = []
        for point in simplified.reshape(-1, 2):
            x, y = point
            flat.append(round(x / w, 6))
            flat.append(round(y / h, 6))
        polygons.append(flat)
    return polygons


def write_data_yaml(dst_root: Path, class_names: list[str] = None) -> None:
    names = class_names or ["landslide"]
    yaml_content = f"""\
path: {dst_root}
train: images/train
val: images/val
test: images/test
nc: {len(names)}
names: {names}
"""
    (dst_root / "data.yaml").write_text(yaml_content)
