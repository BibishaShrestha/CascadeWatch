"""Informal one-off evaluation of Model A (combined_yolov8n_seg, production
weights) against 2 CRS-verified, held-out Sen12Landslides Nepal patches
(nepal_155, nepal_218) - not part of the formal nepal_landslide_seg
train/val/test split, and not added to the Streamlit demo dropdown.

Computes IoU (shapely) between Model A's detected polygon(s) and the
provided ground-truth mask, in pixel space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.inference_api import predict_landslide

DATA_DIR = REPO_ROOT / "nepal-dataset-with-crs"
HOLDOUT_IDS = ["nepal_155", "nepal_218"]


def mask_to_shapely_polygons(mask_path: Path) -> list[Polygon]:
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        poly = Polygon(contour.reshape(-1, 2))
        if poly.is_valid and poly.area > 0:
            polygons.append(poly)
    return polygons


def compute_iou(pred_polygons: list[list[list[float]]], gt_polygons: list[Polygon]) -> float:
    pred_shapes = [Polygon(p) for p in pred_polygons if len(p) >= 3]
    pred_shapes = [p.buffer(0) if not p.is_valid else p for p in pred_shapes]
    pred_shapes = [p for p in pred_shapes if p.area > 0]

    if not pred_shapes or not gt_polygons:
        return 0.0

    pred_union = unary_union(pred_shapes)
    gt_union = unary_union(gt_polygons)
    intersection = pred_union.intersection(gt_union).area
    union = pred_union.union(gt_union).area
    return intersection / union if union > 0 else 0.0


def main() -> None:
    print(f"Model A weights in use: {__import__('models.inference_api', fromlist=['LANDSLIDE_MODEL_WEIGHTS']).LANDSLIDE_MODEL_WEIGHTS}\n")

    for chip_id in HOLDOUT_IDS:
        image_path = DATA_DIR / "png-images-filtered" / f"{chip_id}.png"
        mask_path = DATA_DIR / "png-images-segmentation-mask-filtered" / f"{chip_id}_mask.png"

        detections = predict_landslide(str(image_path))
        gt_polygons = mask_to_shapely_polygons(mask_path)
        gt_area_px = sum(p.area for p in gt_polygons)

        print(f"=== {chip_id} ===")
        print(f"Ground truth: {len(gt_polygons)} region(s), {gt_area_px:.0f} px^2 total")
        if not detections:
            print("Model A: 0 detections at conf>=0.25")
        else:
            for i, det in enumerate(detections):
                print(f"Model A detection {i}: confidence={det['confidence']:.4f}, "
                      f"class={det['class']}, polygon_pts={len(det['mask_polygon'])}")
            iou = compute_iou([d["mask_polygon"] for d in detections], gt_polygons)
            print(f"IoU (all detections union vs. real mask): {iou:.4f}")
        print()


if __name__ == "__main__":
    main()
