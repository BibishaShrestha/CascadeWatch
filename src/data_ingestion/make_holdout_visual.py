"""Contact sheet for the Sen12Landslides held-out evaluation - image |
ground-truth mask | Model A detections, side by side, for visual
inspection of held-out detection quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.inference_api import predict_landslide

DATA_DIR = REPO_ROOT / "nepal-dataset-with-crs"
HOLDOUT_IDS = ["nepal_155", "nepal_218"]
OUT_PATH = REPO_ROOT / "outputs" / "figures" / "sen12_holdout_evaluation.png"
SCALE = 3


def main() -> None:
    cell = 128 * SCALE
    pad = 6
    label_h = 16
    sheet = Image.new("RGB", (3 * cell + 4 * pad, len(HOLDOUT_IDS) * (cell + label_h + pad) + pad), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for row, chip_id in enumerate(HOLDOUT_IDS):
        image_path = DATA_DIR / "png-images-filtered" / f"{chip_id}.png"
        mask_path = DATA_DIR / "png-images-segmentation-mask-filtered" / f"{chip_id}_mask.png"
        y0 = row * (cell + label_h + pad) + pad

        img = Image.open(image_path).convert("RGB").resize((cell, cell), Image.NEAREST)
        sheet.paste(img, (pad, y0))
        draw.text((pad, y0 + cell + 1), f"{chip_id}: input", fill=(255, 255, 0))

        mask_rgb = Image.open(mask_path).convert("L").resize((cell, cell), Image.NEAREST).convert("RGB")
        sheet.paste(mask_rgb, (2 * pad + cell, y0))
        draw.text((2 * pad + cell, y0 + cell + 1), "real ground-truth mask", fill=(255, 255, 0))

        det_img = img.copy()
        det_draw = ImageDraw.Draw(det_img)
        detections = predict_landslide(str(image_path))
        for det in detections:
            poly = [(x * SCALE, y * SCALE) for x, y in det["mask_polygon"]]
            det_draw.polygon(poly, outline=(255, 0, 0), width=2)
        sheet.paste(det_img, (3 * pad + 2 * cell, y0))
        draw.text((3 * pad + 2 * cell, y0 + cell + 1),
                  f"Model A detections ({len(detections)})", fill=(255, 255, 0))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
