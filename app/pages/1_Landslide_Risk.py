"""Landslide Risk - CORE module.

Chain: image -> Model A (real trained YOLOv8n-seg) -> DBI physics (real math).

Deliberately has NO concept of regions, coordinates, or "real vs stub"
terrain. Terrain (upstream_area_km2, local_relief_m) is a pair of plain,
user-adjustable number inputs applied to every detected scar in the image -
you're telling the physics engine what valley this scar sits in, same as a
field surveyor would.

Training data for Model A is fully swappable.
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
from data_ingestion.geolocated_examples import all_examples as all_geolocated_examples
from data_ingestion.geolocated_examples import image_path as geolocated_image_path
from models.inference_api import polygon_area_m2, predict_landslide
from physics.dbi import assess_scar

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
NEPAL_ROOT = DATASETS_DIR / "nepal_landslide_seg"
GLOBAL_ROOT = DATASETS_DIR / "landslide_seg"

NEPAL_IMAGES = [
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_9_9.png",
    NEPAL_ROOT / "train" / "images" / "image_validation_23.png",
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_17_7.png",
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_11_71.png",
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_14_7.png",
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_5_65.png",
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_11_68.png",
    NEPAL_ROOT / "train" / "images" / "image_validation_10.png",
    NEPAL_ROOT / "validation" / "images" / "image_train_COMP_0_1.png",
    NEPAL_ROOT / "train" / "images" / "image_train_COMP_0_3.png",
]
GLOBAL_IMAGES = [
    GLOBAL_ROOT / "test" / "images" / "image_825.png",
    GLOBAL_ROOT / "test" / "images" / "image_3580.png",
    GLOBAL_ROOT / "test" / "images" / "image_781.png",
    GLOBAL_ROOT / "test" / "images" / "image_373.png",
    GLOBAL_ROOT / "test" / "images" / "image_1150.png",
    GLOBAL_ROOT / "test" / "images" / "image_702.png",
    GLOBAL_ROOT / "test" / "images" / "image_1094.png",
    GLOBAL_ROOT / "test" / "images" / "image_456.png",
    GLOBAL_ROOT / "test" / "images" / "image_1177.png",
    GLOBAL_ROOT / "test" / "images" / "image_1356.png",
]
GEOLOCATED_IMAGES = [ex["path"] for ex in all_geolocated_examples()]

SAMPLE_IMAGES = (
    [(p, "Nepal") for p in NEPAL_IMAGES]
    + [(p, "Global") for p in GLOBAL_IMAGES]
    + [(p, "Real-GEE") for p in GEOLOCATED_IMAGES]
)

DEFAULT_EXAMPLE_PATH = geolocated_image_path("nepal_241")
try:
    DEFAULT_EXAMPLE_INDEX = next(
        i for i, (p, tag) in enumerate(SAMPLE_IMAGES) if p == DEFAULT_EXAMPLE_PATH
    )
except StopIteration:
    DEFAULT_EXAMPLE_INDEX = 0

st.title(":material/landslide: Landslide Risk")
st.caption(
    "Model A is a real trained YOLOv8n-seg model. DBI physics is real math. "
    "Terrain below is whatever you tell it - adjust the sliders to match the valley you're assessing."
)

with st.sidebar:
    st.header("Inputs")
    if not SAMPLE_IMAGES:
        st.error(f"No sample images found under {DATASETS_DIR}")
        st.stop()

    image_source = st.radio("Image source", ["Curated examples", "Upload your own"], horizontal=True)

    if image_source == "Upload your own":
        uploaded = st.file_uploader("Upload a satellite image tile", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded is None:
            st.info("Upload an image above to continue, or switch back to curated examples.")
            st.stop()
        upload_dir = Path(tempfile.gettempdir()) / "cascadewatch_uploads"
        upload_dir.mkdir(exist_ok=True)
        image_choice = upload_dir / uploaded.name
        image_choice.write_bytes(uploaded.getvalue())
        image_tag = "Upload"
    else:
        image_choice, image_tag = st.selectbox(
            "Example image",
            options=SAMPLE_IMAGES,
            index=DEFAULT_EXAMPLE_INDEX,
            format_func=lambda item: f"[{item[1]}] {item[0].name}",
        )

    st.divider()
    st.subheader("Terrain")
    st.caption("Applied to every scar detected in this image.")
    upstream_area_km2 = st.number_input(
        "Upstream drainage area (km²)", min_value=0.01, value=5.0, step=0.5,
        help="How much of the watershed drains through this point - bigger river, bigger area.",
    )
    local_relief_m = st.number_input(
        "Local relief (m)", min_value=1.0, value=500.0, step=50.0,
        help="Valley wall height near the scar - caps how tall a debris dam can physically pile up.",
    )

    run = st.button("Run detection + DBI", type="primary")

col_img, col_results = st.columns([1, 1.4])

if not run:
    st.info("Pick an image and terrain values in the sidebar, then click **Run detection + DBI**.")
    st.stop()

image_path = str(image_choice)
detections = predict_landslide(image_path)

with col_img:
    st.subheader("Model A detections")
    annotated = annotate_detections(image_path, detections)
    st.image(
        annotated,
        caption=f"[{image_tag}] {image_choice.name} (red = detected scar polygon, label = scar id)",
        width="stretch",
    )

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
        upstream_area_km2=upstream_area_km2,
        local_relief_m=local_relief_m,
    )
    scar["confidence"] = det["confidence"]
    scar_results.append(scar)

with col_results:
    if n_skipped_degenerate:
        st.warning(
            f"Skipped {n_skipped_degenerate} degenerate detection(s) with a zero-area polygon "
            "(self-intersecting mask contour)."
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
                    st.caption(
                        f"confidence {scar['confidence']:.2f} · area {scar['area_m2']:.0f} m² · "
                        f"dam height {scar['dam_height_m']:.1f} m · DBI {scar['dbi']:.2f}"
                    )
                with c2:
                    verdict_badge(scar["verdict"])

        dbi_df = pd.DataFrame(scar_results)[
            ["scar_id", "confidence", "area_m2", "volume_m3", "dam_height_m", "dbi", "verdict"]
        ]
        with st.expander("Full table"):
            st.dataframe(dbi_df, width="stretch")

st.success("Detection + DBI physics ran end to end (both real).")
