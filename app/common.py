"""Shared helpers for the CascadeWatch Streamlit pages.

Kept deliberately tiny - each page owns its own pipeline logic. This module
only holds the bits that are genuinely identical across pages (drawing
detections, rendering a verdict as a badge) so the pages stay readable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.inference_api import load_image_array

VERDICT_BADGE = {
    "stable": ("green", ":material/check_circle:"),
    "uncertain": ("orange", ":material/warning:"),
    "breach_risk": ("red", ":material/dangerous:"),
}


def annotate_detections(image_path, detections, scale: int = 4) -> Image.Image:
    """Draw red scar-polygon outlines + scarN labels (collision-avoided) on
    an upscaled copy of the image.

    Uses load_image_array() (not plain PIL.Image.open) so real GeoTIFF
    uploads render correctly - PIL's TIFF plugin can't identify real
    int16/GeoTIFF-tagged rasters (raises UnidentifiedImageError), a real
    crash found via live-demo testing with an uploaded real GeoTIFF.
    """
    img = Image.fromarray(load_image_array(str(image_path)))
    annotated = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(annotated)

    def boxes_overlap(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    placed_label_boxes = []
    for i, det in enumerate(detections):
        poly = [(x * scale, y * scale) for x, y in det["mask_polygon"]]
        draw.polygon(poly, outline=(255, 0, 0), width=2)
        label = f"scar{i}"
        xs, ys = zip(*poly)
        x0, y0 = min(xs), max(min(ys) - 14, 0)
        for _ in range(20):
            bbox = draw.textbbox((x0, y0), label)
            if not any(boxes_overlap(bbox, placed) for placed in placed_label_boxes):
                break
            y0 += (bbox[3] - bbox[1]) + 2
        placed_label_boxes.append(bbox)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x0, y0), label, fill=(255, 255, 0))
    return annotated


def verdict_badge(verdict: str) -> None:
    color, icon = VERDICT_BADGE.get(verdict, ("gray", ":material/help:"))
    st.badge(verdict, icon=icon, color=color)
