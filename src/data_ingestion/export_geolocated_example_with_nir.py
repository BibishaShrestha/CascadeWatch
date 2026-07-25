"""One-off export: re-derive a geolocated demo example's GeoTIFF to include
the NIR band (B08) alongside R/G/B, sourced from the original
Sen12Landslides .nc patch (data_harmonized/s2/) rather than the RGB-only
tiff-images-filtered/ export used for the PNG demo asset.

Only run for the chosen default demo example (nepal_241). Additive: does
not touch the existing 3-band tif, PNG, or mask.

Band order matches models/water_detection.py's Landsat convention (Red,
Green, Blue, NIR - green_band=2, nir_band=4) so detect_water()'s defaults
work unchanged on this file too.

NDWI's (Green-NIR)/(Green+NIR) ratio is invariant to any positive
multiplicative constant, and this .nc's bands carry no CF scale_factor/
add_offset - so raw int16 DN values are used directly, no reflectance
rescaling needed for NDWI to be mathematically correct.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

NC_PATH = Path("/home/bishal/ml-codespace/temp/staging/sen12landslides_nepal_check/data_harmonized/s2/nepal_s2_241.nc")
OUT_PATH = REPO_ROOT / "datasets" / "geolocated_examples" / "nepal_241_rgbnir.tif"
EXISTING_RGB_TIF = REPO_ROOT / "nepal-dataset-with-crs" / "tiff-images-filtered" / "nepal_241.tif"

BANDS = ["B04", "B03", "B02", "B08"]


def main() -> None:
    ds = xr.open_dataset(NC_PATH)
    pre_post_dates = ds.attrs["pre_post_dates"]
    if isinstance(pre_post_dates, str):
        pre_post_dates = ast.literal_eval(pre_post_dates)
    post_idx = pre_post_dates["post"]
    print(f"post-event time index: {post_idx} ({ds.time.values[post_idx]}), "
          f"event_date attr: {ds.attrs['event_date']}")

    with rasterio.open(EXISTING_RGB_TIF) as src:
        existing_bounds = src.bounds
        existing_crs = src.crs
    print(f"existing 3-band tif bounds: {existing_bounds}, crs: {existing_crs}")

    x_res = float(ds.x.values[1] - ds.x.values[0])
    y_res = float(ds.y.values[1] - ds.y.values[0])
    transform = from_origin(
        float(ds.x.values[0]) - x_res / 2,
        float(ds.y.values[0]) - y_res / 2,
        x_res,
        -y_res,
    )
    print(f"derived transform: {transform}")

    bands = np.stack([ds[b].isel(time=post_idx).values for b in BANDS], axis=0).astype(np.int16)
    print(f"exported array shape: {bands.shape}, band means: "
          f"{[round(float(bands[i].mean()), 1) for i in range(4)]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUT_PATH, "w", driver="GTiff",
        height=bands.shape[1], width=bands.shape[2], count=4,
        dtype="int16", crs="EPSG:32645", transform=transform,
    ) as dst:
        dst.write(bands)
        dst.descriptions = ("Red (B04)", "Green (B03)", "Blue (B02)", "NIR (B08)")

    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
