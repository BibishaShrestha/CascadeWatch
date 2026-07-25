"""Render viewable 8-bit RGB previews from the real Landsat validation
GeoTIFFs (surface reflectance, ~0-1 range) via a 2nd-98th percentile
contrast stretch per band.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def stretch_to_uint8(band: np.ndarray) -> np.ndarray:
    valid = band[np.isfinite(band)]
    lo, hi = np.percentile(valid, [2, 98])
    stretched = np.clip((band - lo) / (hi - lo + 1e-9), 0, 1)
    return (stretched * 255).astype(np.uint8)


def make_preview(tif_path: Path, out_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        arr = src.read()
    rgb = np.stack([stretch_to_uint8(arr[i]) for i in range(3)], axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def main() -> None:
    validation_dir = REPO_ROOT / "data" / "study_areas"
    out_dir = REPO_ROOT / "outputs" / "figures" / "validation_previews"
    for tif_path in sorted(validation_dir.glob("*/validation/*/*.tif")):
        region, _, case, stage_file = tif_path.parts[-4:]
        stage = stage_file.replace(".tif", "")
        out_path = out_dir / f"{region}_{case}_{stage}.png"
        make_preview(tif_path, out_path)
        print(f"{tif_path} -> {out_path}")


if __name__ == "__main__":
    main()
