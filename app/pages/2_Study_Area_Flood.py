"""Study Area Flood - ADD-ON module.

Chain: image -> Model A (real) -> real DEM-derived terrain lookup -> DBI
physics (real) -> real OSM-derived downstream exposure -> asset ranking.

Only lights up once at least one study area is registered under
data/study_areas/<name>/ (a real DEM with CRS, at minimum) - a
drop-a-folder-in operation, no code changes.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = APP_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(APP_DIR))

from common import annotate_detections, verdict_badge
from models.gsd import get_center_latlon
from models.inference_api import polygon_area_m2, predict_landslide
from physics.dbi import assess_scar
from physics.exposure import propagate_exposure, rank_assets
from study_areas.registry import get_bbox, list_study_areas, locate_study_area
from terrain.lookup import lookup_terrain_at

st.title(":material/water_drop: Study Area Flood")

areas = list_study_areas()
if not areas:
    st.info(
        "**No study areas registered yet.** This module needs a real DEM (with CRS) to compute "
        "drainage area and downstream exposure - drop one in under `data/study_areas/<name>/` "
        "and it shows up here automatically."
    )
    st.stop()

st.caption(
    "Model A and DBI physics are real, same as Landslide Risk. Terrain and downstream exposure "
    "here are also real - derived from an actual DEM drainage graph and OSM roads/bridges/POIs "
    "for the study area **you** pick below. If you upload a real GeoTIFF with its own embedded "
    "coordinates, the study area and coordinate boxes auto-fill from it. A plain PNG/JPG has no "
    "such metadata, so those stay a manual assertion - this page cross-checks whatever you upload "
    "against whatever the boxes say, and warns you on a mismatch either way."
)

with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Satellite image tile for this location", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an image above to continue.")
        st.stop()
    upload_dir = Path(tempfile.gettempdir()) / "cascadewatch_uploads"
    upload_dir.mkdir(exist_ok=True)
    image_choice = upload_dir / uploaded.name
    image_choice.write_bytes(uploaded.getvalue())
    image_path = str(image_choice)

    embedded_latlon = get_center_latlon(image_path)
    auto_region = locate_study_area(*embedded_latlon) if embedded_latlon is not None else None

    area_options = sorted(areas)
    default_region_index = area_options.index(auto_region) if auto_region in area_options else 0
    region = st.selectbox("Study area", options=area_options, index=default_region_index, key=f"region_{uploaded.file_id}")

    lon_min, lat_min, lon_max, lat_max = get_bbox(region)
    if embedded_latlon is not None and auto_region == region:
        default_lat, default_lon = embedded_latlon
        st.caption(":material/check_circle: Latitude/Longitude auto-filled from this upload's real embedded coordinates.")
    else:
        default_lat, default_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
        if embedded_latlon is None:
            st.caption(":material/info: No embedded coordinates in this upload (plain PNG/JPG) - defaulted to the region's center. Set the real location manually if you know it.")

    lat = st.number_input(
        "Latitude", value=round(default_lat, 4), format="%.4f",
        min_value=lat_min, max_value=lat_max, key=f"lat_{uploaded.file_id}",
    )
    lon = st.number_input(
        "Longitude", value=round(default_lon, 4), format="%.4f",
        min_value=lon_min, max_value=lon_max, key=f"lon_{uploaded.file_id}",
    )
    st.caption(f"{region} bbox: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")

    run = st.button("Run full chain", type="primary")

col_img, col_results = st.columns([1, 1.4])

if not run:
    st.info("Pick a study area and coordinate, upload an image, then click **Run full chain**.")
    st.stop()

POINT_MISMATCH_KM = 1.0

if embedded_latlon is not None:
    embedded_lat, embedded_lon = embedded_latlon
    embedded_region = locate_study_area(embedded_lat, embedded_lon)
    if embedded_region is None:
        st.warning(
            f"This upload has real embedded coordinates ({embedded_lat:.4f}, {embedded_lon:.4f}) "
            f"that fall **outside every registered study area** - not inside **{region}**, which you "
            "selected. The result below is real terrain/exposure for your selected region+coordinate, "
            "not for where this image actually is."
        )
    elif embedded_region != region:
        st.error(
            f"**Location mismatch**: this upload's real embedded coordinates "
            f"({embedded_lat:.4f}, {embedded_lon:.4f}) are actually inside **{embedded_region}**, not "
            f"**{region}**, which you selected. The result below is real terrain/exposure for "
            f"{region} at ({lat:.4f}, {lon:.4f}) - a real result, but for the wrong place. "
            f"Switch the study area dropdown to **{embedded_region}** to get a result that actually "
            "matches this image."
        )
    else:
        from pyproj import Geod

        _, _, point_distance_m = Geod(ellps="WGS84").inv(embedded_lon, embedded_lat, lon, lat)
        point_distance_km = point_distance_m / 1000
        if point_distance_km > POINT_MISMATCH_KM:
            st.warning(
                f"This upload's real embedded coordinates ({embedded_lat:.4f}, {embedded_lon:.4f}) are "
                f"inside **{region}** (matching your region selection), but they're "
                f"**{point_distance_km:.1f} km away** from the ({lat:.4f}, {lon:.4f}) you entered in "
                "the Latitude/Longitude boxes below - likely still at their default (region bbox "
                "center), not updated to match this image. The terrain/exposure result below is real, "
                "but for that different point, not for where this image actually is. Update the "
                "Latitude/Longitude boxes to this image's real coordinates to fix this."
            )
        else:
            st.success(
                f"This upload's real embedded coordinates ({embedded_lat:.4f}, {embedded_lon:.4f}) "
                f"confirm it's really inside **{region}**, {point_distance_km*1000:.0f}m from the "
                "coordinate you entered - matching your selection."
            )

detections = predict_landslide(image_path)

with col_img:
    st.subheader("Model A detections")
    annotated = annotate_detections(image_path, detections)
    st.image(annotated, caption=f"{image_choice.name} (red = detected scar polygon)", width="stretch")

real_terrain = lookup_terrain_at(region, lat, lon)

scar_results = []
n_skipped_degenerate = 0
for i, det in enumerate(detections):
    area_m2 = polygon_area_m2(det["mask_polygon"])
    if area_m2 <= 0:
        n_skipped_degenerate += 1
        continue
    scar = assess_scar(
        scar_id=f"{image_choice.stem}-scar{i}",
        area_m2=area_m2,
        upstream_area_km2=real_terrain["upstream_area_km2"],
        local_relief_m=real_terrain["local_relief_m"],
        graph_node_id=str(real_terrain["nearest_channel_node"]),
    )
    scar["confidence"] = det["confidence"]
    scar_results.append(scar)

with col_results:
    if n_skipped_degenerate:
        st.warning(f"Skipped {n_skipped_degenerate} degenerate detection(s) with a zero-area polygon.")

    relief_note = (
        f"real local relief {real_terrain['local_relief_m']:.0f}m (1km DEM window)"
        if real_terrain.get("local_relief_is_real")
        else f"local relief fell back to a {real_terrain['local_relief_m']:.0f}m placeholder (DEM window lookup failed)"
    )
    st.success(
        f"Terrain: real DEM-derived upstream area ({real_terrain['upstream_area_km2']:.2f} km²) at "
        f"({lat:.4f}, {lon:.4f}) - nearest real channel node is "
        f"{real_terrain['distance_to_channel_km']:.2f} km away; {relief_note}."
    )

    st.subheader("DBI verdicts")
    if not scar_results:
        st.info("No landslide scars detected in this image - nothing for the physics engine to assess.")
    else:
        for scar in scar_results:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{scar['scar_id']}**")
                    st.caption(f"confidence {scar['confidence']:.2f} · area {scar['area_m2']:.0f} m² · DBI {scar['dbi']:.2f}")
                with c2:
                    verdict_badge(scar["verdict"])

        with st.expander("Full DBI table"):
            st.dataframe(
                pd.DataFrame(scar_results)[
                    ["scar_id", "confidence", "area_m2", "volume_m3", "dam_height_m", "dbi", "verdict", "graph_node_id"]
                ],
                width="stretch",
            )

        st.subheader("Downstream exposure - real OSM assets along the real channel path")
        for scar in scar_results:
            st.markdown(f"**{scar['scar_id']}** — verdict: ")
            verdict_badge(scar["verdict"])
            exposure = propagate_exposure(scar, region=region, lat=lat, lon=lon)
            ranked = rank_assets(exposure)
            if ranked:
                st.dataframe(pd.DataFrame(ranked), width="stretch")
            else:
                st.caption("No OSM assets found within range of the downstream path from this scar.")

st.success("Full chain ran end to end - real detection, real terrain, real exposure ranking.")
