"""Visual check for the NDWI water detector against Jure's post-breach
Landsat imagery - RGB | NDWI heatmap | detected polygon overlay, side by
side, so "does this plausibly find the river?" can be checked by eye, not
just a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.water_detection import compute_ndwi, detect_water, read_green_nir

TIF_PATH = REPO_ROOT / "data" / "study_areas" / "sunkoshi" / "validation" / "jure" / "post_breach.tif"
RGB_PREVIEW = REPO_ROOT / "outputs" / "figures" / "validation_previews" / "sunkoshi_jure_post_breach.png"
OUT_PATH = REPO_ROOT / "outputs" / "figures" / "ndwi_jure_post_breach_evaluation.png"
SCALE = 4


def ndwi_to_heatmap(ndwi: np.ndarray) -> Image.Image:
    clipped = np.clip(ndwi, -1, 1)
    normalized = (clipped + 1) / 2
    r = ((1 - normalized) * 255).astype(np.uint8)
    b = (normalized * 255).astype(np.uint8)
    g = np.zeros_like(r)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def main() -> None:
    green, nir = read_green_nir(TIF_PATH)
    ndwi = compute_ndwi(green, nir)
    detections = detect_water(TIF_PATH)

    cell = green.shape[0] * SCALE
    pad = 6
    label_h = 16
    sheet = Image.new("RGB", (3 * cell + 4 * pad, cell + label_h + 2 * pad), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    rgb = Image.open(RGB_PREVIEW).convert("RGB").resize((cell, cell), Image.NEAREST)
    sheet.paste(rgb, (pad, pad))
    draw.text((pad, pad + cell + 1), "real RGB (post-breach)", fill=(255, 255, 0))

    heatmap = ndwi_to_heatmap(ndwi).resize((cell, cell), Image.NEAREST)
    sheet.paste(heatmap, (2 * pad + cell, pad))
    draw.text((2 * pad + cell, pad + cell + 1), "NDWI heatmap (blue=water-like)", fill=(255, 255, 0))

    overlay = rgb.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    for det in detections:
        poly = [(x * SCALE, y * SCALE) for x, y in det["mask_polygon"]]
        overlay_draw.polygon(poly, outline=(0, 255, 255), width=2)
    sheet.paste(overlay, (3 * pad + 2 * cell, pad))
    draw.text((3 * pad + 2 * cell, pad + cell + 1),
              f"NDWI detection ({len(detections)} polygon, thr=0.0)", fill=(255, 255, 0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"NDWI range: [{ndwi.min():.3f}, {ndwi.max():.3f}], mean={ndwi.mean():.3f}")
    for d in detections:
        print(f"  detection: confidence={d['confidence']:.3f}, points={len(d['mask_polygon'])}")


if __name__ == "__main__":
    main()
