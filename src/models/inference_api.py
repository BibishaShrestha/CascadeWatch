"""Shared inference interface for both CV models (Model A: landslide scar
segmentation, Model B: flood extent detection).

    def predict(image_path_or_array) -> list[dict]
        [{"mask_polygon": [[x, y], ...], "confidence": float, "class": str}, ...]

Both predict_landslide and predict_flood return exactly this shape, whether
backed by a trained model (Model A) or a non-trained algorithm (Model B:
NDWI, see models/water_detection.py), so nothing downstream (physics
engine, validation routine, Streamlit app) needs to know which is which.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from models.model_config import DEFAULT_GSD_M_PER_PX, read_model_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMBINED_WEIGHTS_PATH = _REPO_ROOT / "outputs" / "models" / "combined_yolov8n_seg" / "weights" / "best.pt"
_LANDSLIDE_ONLY_WEIGHTS_PATH = _REPO_ROOT / "outputs" / "models" / "landslide_yolov8n_seg" / "weights" / "best.pt"
_LANDSLIDE_WEIGHTS_PATH = _COMBINED_WEIGHTS_PATH if _COMBINED_WEIGHTS_PATH.exists() else _LANDSLIDE_ONLY_WEIGHTS_PATH

LANDSLIDE_MODEL_WEIGHTS = str(_LANDSLIDE_WEIGHTS_PATH) if _LANDSLIDE_WEIGHTS_PATH.exists() else None
FLOOD_MODEL_WEIGHTS = None


def default_gsd_m_per_px() -> float:
    """The currently-loaded landslide model's training GSD, read from the
    model_config.json its training run wrote next to best.pt - falls back
    to DEFAULT_GSD_M_PER_PX if no trained model (or no model_config.json
    for it) is present, e.g. a fresh checkout before training has run.
    """
    if LANDSLIDE_MODEL_WEIGHTS is None:
        return DEFAULT_GSD_M_PER_PX
    config = read_model_config(LANDSLIDE_MODEL_WEIGHTS)
    if config is None:
        return DEFAULT_GSD_M_PER_PX
    return config["gsd_m_per_px"]


@lru_cache(maxsize=None)
def _load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def _read_tif_as_rgb_uint8(path: Path) -> np.ndarray:
    """GeoTIFFs (int16, multi-band, GeoTIFF tags OpenCV/PIL don't understand)
    fail or silently misread as 1-channel through both PIL
    (Image.open raises UnidentifiedImageError) and Ultralytics' own
    OpenCV-based loader (produces a 1-channel array, crashing the model's
    first conv layer). rasterio (already a core dependency, used throughout
    terrain/) reads these correctly; this stretches the first 3 bands to a
    normal 8-bit RGB view for the model, the same 2nd-98th percentile
    approach used for the Landsat validation previews.
    """
    import rasterio

    with rasterio.open(path) as src:
        n_bands = min(3, src.count)
        bands = [src.read(i + 1).astype(np.float64) for i in range(n_bands)]
    if len(bands) < 3:
        bands = (bands * 3)[:3]

    stretched = []
    for band in bands:
        lo, hi = np.percentile(band, [2, 98])
        stretched.append(np.clip((band - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8))
    return np.stack(stretched, axis=-1)


def load_image_array(image_path_or_array) -> np.ndarray:
    if isinstance(image_path_or_array, str):
        path = Path(image_path_or_array)
        if path.suffix.lower() in {".tif", ".tiff"}:
            return _read_tif_as_rgb_uint8(path)
        return np.array(Image.open(image_path_or_array).convert("RGB"))
    return np.asarray(image_path_or_array)


def polygon_area_px(polygon: list[list[float]]) -> float:
    """Shoelace formula, absolute area in px^2."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_area_m2(polygon: list[list[float]], gsd_m_per_px: float | None = None) -> float:
    """gsd_m_per_px defaults to the currently-loaded model's training GSD
    (see default_gsd_m_per_px()); pass an explicit value to override, e.g.
    validate_jure.py's 30m Landsat case.
    """
    if gsd_m_per_px is None:
        gsd_m_per_px = default_gsd_m_per_px()
    return polygon_area_px(polygon) * (gsd_m_per_px ** 2)


def predict_landslide(image_path_or_array, conf: float = 0.25) -> list[dict]:
    """Model A: landslide scar segmentation.

    Runs YOLOv8n-seg inference when trained weights are found at
    outputs/models/landslide_yolov8n_seg/weights/best.pt; falls back to a
    deterministic stub polygon otherwise, e.g. in a fresh checkout before
    training has run.
    """
    if LANDSLIDE_MODEL_WEIGHTS is not None:
        return _predict_landslide_real(image_path_or_array, conf=conf)
    return _predict_landslide_stub(image_path_or_array)


def _predict_landslide_real(image_path_or_array, conf: float = 0.25) -> list[dict]:
    img_rgb = load_image_array(image_path_or_array)
    img_bgr = img_rgb[:, :, ::-1]
    model = _load_yolo_model(LANDSLIDE_MODEL_WEIGHTS)
    result = model.predict(img_bgr, conf=conf, verbose=False)[0]

    detections = []
    if result.masks is not None:
        polygons = result.masks.xy
        confidences = result.boxes.conf.tolist()
        class_ids = result.boxes.cls.tolist()
        for polygon, confidence, class_id in zip(polygons, confidences, class_ids):
            if len(polygon) < 3:
                continue
            detections.append(
                {
                    "mask_polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in polygon],
                    "confidence": round(float(confidence), 4),
                    "class": result.names[int(class_id)],
                }
            )
    return detections


def _predict_landslide_stub(image_path_or_array) -> list[dict]:
    """Deterministic stub detection, used when no trained weights exist yet."""
    img = load_image_array(image_path_or_array)
    h, w = img.shape[0], img.shape[1]
    cx, cy = w * 0.55, h * 0.5
    rx, ry = w * 0.22, h * 0.16
    polygon = [
        [cx - rx, cy - ry * 0.3],
        [cx - rx * 0.4, cy - ry],
        [cx + rx * 0.5, cy - ry * 0.6],
        [cx + rx, cy + ry * 0.2],
        [cx + rx * 0.3, cy + ry],
        [cx - rx * 0.6, cy + ry * 0.7],
    ]
    return [
        {
            "mask_polygon": [[round(x, 1), round(y, 1)] for x, y in polygon],
            "confidence": 0.87,
            "class": "landslide",
        }
    ]


def predict_flood(image_path_or_array, green_band: int | None = None, nir_band: int | None = None) -> list[dict]:
    """Model B: NDWI-based water/flood detection (McFeeters 1996), not a
    trained segmentation model - see models/water_detection.py.

    Needs a multi-band raster with a Green band and a NIR band present.
    Most of this project's imagery (RGB PNGs, the rendered previews under
    outputs/figures/validation_previews/) doesn't qualify - only a GeoTIFF
    source with NIR does, currently just the Landsat 8 validation imagery
    under data/study_areas/*/validation/.

    Given an input that doesn't qualify (not a .tif/.tiff path, or a raster
    missing the needed bands), returns [] and prints why rather than
    returning a fake polygon.
    """
    if FLOOD_MODEL_WEIGHTS is not None:
        raise NotImplementedError(
            "Trained Model B inference not implemented - NDWI (water_detection.py) "
            "is the current Model B; a trained segmentation model would be a "
            "separate future upgrade, not this code path"
        )

    from models.water_detection import MissingBandError, detect_water

    path = image_path_or_array if isinstance(image_path_or_array, (str, Path)) else None
    if path is None or Path(path).suffix.lower() not in {".tif", ".tiff"}:
        print(
            f"predict_flood: NDWI needs a multi-band GeoTIFF with Green+NIR bands - "
            f"{image_path_or_array!r} isn't one. Returning no detections."
        )
        return []

    kwargs = {}
    if green_band is not None:
        kwargs["green_band"] = green_band
    if nir_band is not None:
        kwargs["nir_band"] = nir_band
    try:
        return detect_water(Path(path), **kwargs)
    except MissingBandError as e:
        print(f"predict_flood: {e} Returning no detections.")
        return []
