"""Ground-sample-distance (GSD) reader for georeferenced imagery.

Unlike terrain/process_dem.py's `degrees * 111_320` approximation (only
valid for that module's Nepal-DEM, geographic-CRS use case), this handles
any CRS - geographic or projected - via a geodesic corner-to-corner distance
(pyproj.Geod).

Neither Model A training dataset (landslide_seg, nepal_landslide_seg)
carries per-image CRS - both are plain PNGs with a fixed 10m/px GSD. This
module matters once a georeferenced training set is added; until then every
call falls back to the caller-supplied default.
"""
from __future__ import annotations

import math
from pathlib import Path

RASTER_EXTENSIONS = {".tif", ".tiff"}


def get_gsd_m_per_px(image_path, fallback_gsd: float) -> tuple[float, str]:
    """Return (gsd_m_per_px, source), source is "geotiff_crs" or "fallback"."""
    path = Path(image_path)
    if path.suffix.lower() not in RASTER_EXTENSIONS:
        return fallback_gsd, "fallback"

    try:
        import rasterio
        from pyproj import Geod, Transformer

        with rasterio.open(path) as src:
            if src.crs is None:
                return fallback_gsd, "fallback"
            width, height = src.width, src.height
            (x0, y0) = src.transform * (0, 0)
            (x1, y1) = src.transform * (width, height)
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon0, lat0 = transformer.transform(x0, y0)
            lon1, lat1 = transformer.transform(x1, y1)
    except Exception:
        return fallback_gsd, "fallback"

    geod = Geod(ellps="WGS84")
    _, _, diagonal_m = geod.inv(lon0, lat0, lon1, lat1)
    diagonal_px = math.hypot(width, height)
    if diagonal_px == 0 or diagonal_m == 0:
        return fallback_gsd, "fallback"
    return diagonal_m / diagonal_px, "geotiff_crs"


def get_center_latlon(image_path) -> tuple[float, float] | None:
    """Embedded center (lat, lon) for a GeoTIFF with a CRS, or None if the
    file isn't a raster or has no CRS (plain PNG/JPG, or a GeoTIFF stripped
    of georeferencing). Used to catch a class of self-inflicted error:
    nothing stops a user from uploading an image and asserting it's at a
    location it manifestly isn't - this lets the app catch that specific
    case (a real GeoTIFF whose own embedded coordinates disagree with the
    region/coordinate the user selected) rather than silently running a
    "real" physics chain against the wrong place.
    """
    path = Path(image_path)
    if path.suffix.lower() not in RASTER_EXTENSIONS:
        return None
    try:
        import rasterio
        from pyproj import Transformer

        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            cx, cy = src.transform * (src.width / 2, src.height / 2)
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(cx, cy)
    except Exception:
        return None
    return lat, lon
