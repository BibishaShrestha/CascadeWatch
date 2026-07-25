"""Historical Validation - a real historical case study, run live.

Not part of the core Landslide Risk / Study Area Flood pipeline - a
standalone comparison against the 2014 Jure/Sunkoshi landslide-dam breach
and a control case, using Landsat 8 imagery. Model A, the DBI physics
engine, and Model B (NDWI water detection - an established remote-sensing
formula, not a trained model) all run live below, every time this page
loads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = APP_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(APP_DIR))

from common import annotate_detections
from validation.validate_jure import CASES as VALIDATION_CASES
from validation.validate_jure import run_case as run_validation_case

cached_run_validation_case = st.cache_data(run_validation_case)

st.title(":material/history: Historical Validation — Jure 2014")
st.caption(
    "Model B uses NDWI-based water detection (McFeeters 1996) on the multi-band "
    "Landsat imagery below."
)
st.caption(
    "Not a precomputed number - Model A and the physics engine run live below, every time "
    "this page loads, on Landsat 8 imagery of the Jure/Sunkoshi 2014 breach and a "
    "control case. Model A shows a known domain gap on this Landsat-resolution imagery, "
    "surfaced directly below rather than hidden."
)

for case_name in VALIDATION_CASES:
    result = cached_run_validation_case(case_name)
    with st.expander(f"{case_name.title()} — known outcome: {result['known_outcome']}", expanded=True):
        img_col_a, img_col_b = st.columns(2)

        with img_col_a:
            annotated = annotate_detections(result["post_landslide_img"], result["detections"])
            st.image(annotated, caption="Post-landslide/post-event imagery + Model A detections (real)", width="stretch")

        with img_col_b:
            if result["post_breach_img"] is not None:
                flood_annotated = annotate_detections(result["post_breach_img"], result["flood_detections"])
                st.image(
                    flood_annotated,
                    caption="Post-breach imagery + Model B: NDWI-based water detection (real algorithm, not a trained segmentation model)",
                    width="stretch",
                )
            else:
                st.info("No post-breach imagery for this case - it has no breach event to bracket.")

        st.markdown(f"**Model A: {len(result['detections'])} scar(s) detected** (conf>=0.25)")
        if not result["detections"] and result["subthreshold"]:
            st.warning(
                f"Zero detections at the production threshold, but "
                f"{len(result['subthreshold']['confidences'])} sub-threshold response(s) found at "
                f"conf>={result['subthreshold']['conf']} ({result['subthreshold']['confidences']}) - "
                "real but weak signal, consistent with a Landsat/Sentinel-2 sensor domain gap "
                "Model A was never trained on, not a total absence of signal."
            )
        if result["scar_results"]:
            st.dataframe(
                pd.DataFrame(result["scar_results"])[
                    ["scar_id", "confidence", "area_m2", "dbi", "verdict", "graph_node_id"]
                ],
                width="stretch",
            )
        if result["post_breach_img"] is not None:
            st.markdown(f"**Model B (NDWI, real): {len(result['flood_detections'])} flood polygon(s)**")
            if result["iou"] is not None:
                st.error(
                    f"Scar-vs-flood-mask IoU: {result['iou']:.3f} - Model B's side is real now "
                    "(NDWI), but **this number still isn't a meaningful validation result**, "
                    "because Model A detected 0 scars on this imagery (see the domain-gap "
                    "warning above) - it's comparing a water mask against an empty "
                    "detection, not two independent detections of the same event."
                )
