# CascadeWatch

**Landslide detection and cascading flood risk prediction from satellite imagery.**

Built for IIMS Hackathon 3.0, Track 3 (Satellite Imaging & Remote Sensing).

A landslide in steep terrain doesn't just remove a hillside — the debris
can dam the river below it, and when that dam fails, the resulting flood
hits everything downstream with no warning. CascadeWatch tries to close
that gap: detect the landslide scar from satellite imagery, run the
physics on whether the resulting dam is likely to breach, and rank which
roads, bridges, schools, and settlements downstream are actually at risk.

## How it works

```
satellite image
      │
      ▼
Model A (YOLOv8n-seg)  →  landslide scar mask + area
      │
      ▼
DBI physics (Dimensionless Blockage Index)
      │
      ▼
stable / uncertain / breach_risk
      │
      ▼ (if a study area with terrain data is registered)
drainage-graph lookup → downstream OSM assets → exposure ranking
```

The app is split into two independent modules plus a validation page:

- **Landslide Risk** (core) — always works. Upload or pick a satellite
  tile, run detection, get a breach-risk verdict. Terrain inputs are
  plain adjustable numbers, so no region setup is required.
- **Study Area Flood** (add-on) — activates once a DEM-backed study area
  is registered. Adds a real drainage-graph terrain lookup and
  OpenStreetMap-derived downstream exposure ranking on top of the same
  detection + physics pipeline. Trishuli and Sunkoshi (Nepal) are
  registered as the first two study areas.
- **Historical Validation** — runs the whole pipeline live against the
  2014 Jure/Sunkoshi landslide-dam breach and a non-breach control case,
  using Landsat 8 imagery from before/after each event.

## The physics

The breach-risk verdict comes from the Dimensionless Blockage Index (DBI),
following Larsen et al. (2010) for scar-volume scaling:

```
scar area → deposit volume → dam height (capped by local relief)
DBI = log10(upstream_drainage_area_km² × dam_height_m / deposit_volume_Mm³)
```

Higher DBI means a bigger dam sitting on a bigger river — more likely to
fail. Thresholds split the result into `stable`, `uncertain`, and
`breach_risk`.

## Data

- **Model A training data**: a combined pool of the Landslide4Sense 2022
  benchmark (Ghorbanzadeh et al., arXiv:2206.00515) and a Nepal-specific
  landslide dataset, converted to YOLO-seg format.
- **Terrain**: Copernicus GLO-30 DEM, processed with `pysheds` into a
  drainage network (`networkx` DiGraph) for the Trishuli and Sunkoshi
  corridors.
- **Infrastructure**: OpenStreetMap extracts (via Geofabrik + `pyrosm`) —
  roads, bridges, schools, settlements, health facilities.
- **Historical validation imagery**: Landsat 8 (Earth Engine), pre/post
  imagery for the Jure 2014 breach and a control case.
- **Flood detection (Model B)**: NDWI (McFeeters 1996) computed from
  Green/NIR bands — a standard remote-sensing water index, not a trained
  model, since no flood-extent training set was available in the hackathon
  window.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Two optional credentials are only needed if you want to re-run data
ingestion (not required to run the demo — the processed data is already
committed):

- `OPENTOPOGRAPHY_API_KEY` in a `.env` file at the repo root, for DEM
  downloads (`src/data_ingestion/download_dem.py`).
- An Earth Engine session (`earthengine authenticate`), for pulling
  historical validation imagery.

## Run it

```bash
source .venv/bin/activate
streamlit run app/Home.py
```

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Project layout

```
app/                    Streamlit pages
src/models/              Model A (YOLOv8-seg) + inference interface
src/physics/              DBI physics + exposure/asset ranking
src/terrain/               DEM processing + drainage-graph lookup
src/study_areas/            Study-area registry (DEM/CRS-backed regions)
src/validation/              Historical validation routine (Jure 2014)
src/data_ingestion/           DEM/OSM/imagery download + dataset prep scripts
```



https://github.com/user-attachments/assets/8e415601-b74c-41cd-a18a-ccc8d84a46f8


## Team

Built during IIMS Hackathon 3.0 by Team Cascadia.
