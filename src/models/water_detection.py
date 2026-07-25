"""NDWI (Normalized Difference Water Index) water/flood detection.

    NDWI = (Green - NIR) / (Green + NIR)   [McFeeters 1996]

An established remote-sensing formula, not a trained model - this is
Model B's implementation. It needs a multi-band raster with a Green band
and a NIR band present; the demo's RGB-only PNGs/previews don't qualify,
only a GeoTIFF source with NIR does. The only such source in this project
is the Landsat 8 validation imagery (`pull_validation_imagery.py`, bands
["SR_B4","SR_B3","SR_B2","SR_B5"] = Red, Green, Blue, NIR), so
DEFAULT_GREEN_BAND/DEFAULT_NIR_BAND below match that band order
(1-indexed, rasterio convention). A Sentinel-2 source would need
green_band/nir_band passed explicitly (its usual order is
B02,B03,B04,B08 = Blue,Green,Red,NIR - different from ours).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio

NDWI_THRESHOLD = 0.0
MIN_CONTOUR_AREA_PX = 10.0
APPROX_EPSILON_FRAC = 0.005

DEFAULT_GREEN_BAND = 2
DEFAULT_NIR_BAND = 4


class MissingBandError(Exception):
    """Raised when the input raster doesn't have the bands NDWI needs."""


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    green = green.astype(np.float64)
    nir = nir.astype(np.float64)
    denom = green + nir
    ndwi = np.zeros_like(green)
    valid = denom != 0
    ndwi[valid] = (green[valid] - nir[valid]) / denom[valid]
    return ndwi


def read_green_nir(
    tif_path: Path, green_band: int = DEFAULT_GREEN_BAND, nir_band: int = DEFAULT_NIR_BAND
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(tif_path) as src:
        if src.count < max(green_band, nir_band):
            raise MissingBandError(
                f"{tif_path} has {src.count} band(s) - need at least band "
                f"{max(green_band, nir_band)} for NDWI (green={green_band}, nir={nir_band})"
            )
        green = src.read(green_band)
        nir = src.read(nir_band)
    return green, nir


def ndwi_mask_to_polygons(binary_mask: np.ndarray, ndwi: np.ndarray) -> list[dict]:
    """binary_mask: boolean array (ndwi > threshold). Returns
    [{"mask_polygon": [[x,y],...], "confidence": float}, ...] in PIXEL
    coordinates (point pairs) - matching predict_landslide()'s real-branch
    output shape, which is what the frozen predict() interface actually
    requires.

    NOT reusing yolo_seg_utils.mask_to_polygons(): that utility reads a mask
    PNG from a file path and returns normalized [0,1] FLAT coordinate lists
    (built for writing YOLO label files) - a different coordinate
    convention than the frozen interface needs, and it can't take an
    in-memory array. This reuses the same underlying cv2 contour-finding
    approach directly instead, producing the correct shape natively.
    """
    binary_u8 = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = []
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

        contour_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        mean_ndwi = float(ndwi[contour_mask.astype(bool)].mean())
        confidence = round(max(0.0, min(1.0, mean_ndwi)), 4)

        polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in simplified.reshape(-1, 2)]
        results.append({"mask_polygon": polygon, "confidence": confidence, "class": "flood"})
    return results


def detect_water(
    tif_path: Path,
    green_band: int = DEFAULT_GREEN_BAND,
    nir_band: int = DEFAULT_NIR_BAND,
    threshold: float = NDWI_THRESHOLD,
) -> list[dict]:
    """NDWI water detection on a multi-band GeoTIFF. Raises MissingBandError
    if the file doesn't have enough bands - callers decide how to handle
    that (predict_flood() catches it and falls back to an empty result)."""
    green, nir = read_green_nir(tif_path, green_band, nir_band)
    ndwi = compute_ndwi(green, nir)
    binary = ndwi > threshold
    return ndwi_mask_to_polygons(binary, ndwi)
