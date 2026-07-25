"""CascadeWatch - landing page.

Two independent modules, use either on its own:

1. Landslide Risk (core) - always works. Training dataset is swappable.
2. Study Area Flood (add-on) - only lights up once a DEM+CRS study area is
   registered under data/study_areas/<name>/.

Historical Validation is a third, standalone page for the Jure 2014 case
study - not part of the core pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from study_areas.registry import list_study_areas

st.set_page_config(page_title="CascadeWatch", page_icon=":material/landslide:", layout="wide")

st.title("CascadeWatch")
st.caption("Landslide scar detection + landslide-dam breach risk, built as two independent modules.")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader(":material/landslide: Landslide Risk", anchor=False)
        st.markdown(
            "**Always works.** Upload or pick a satellite image tile, run the real trained "
            "detection model, and get a DBI (Dimensionless Blockage Index) breach-risk verdict "
            "per detected scar. Terrain inputs are plain, adjustable numbers - no region or "
            "coordinate setup needed."
        )
        st.page_link("pages/1_Landslide_Risk.py", label="Open Landslide Risk", icon=":material/arrow_forward:")

with col2:
    with st.container(border=True):
        st.subheader(":material/water_drop: Study Area Flood", anchor=False)
        areas = list_study_areas()
        if areas:
            st.markdown(
                f"**Add-on, active.** {len(areas)} study area(s) registered "
                f"({', '.join(sorted(areas))}) with a real DEM, drainage graph, and OSM asset "
                "index. Pick a location to get real downstream exposure ranking, not a guess."
            )
        else:
            st.markdown(
                "**Add-on, not active yet.** This lights up once a real DEM (with CRS) is "
                "registered as a study area under `data/study_areas/<name>/`."
            )
        st.page_link(
            "pages/2_Study_Area_Flood.py",
            label="Open Study Area Flood",
            icon=":material/arrow_forward:",
            disabled=not areas,
        )

st.divider()
st.page_link(
    "pages/3_Historical_Validation.py",
    label="Historical Validation (Jure 2014) — real historical case study, run live",
    icon=":material/history:",
)
