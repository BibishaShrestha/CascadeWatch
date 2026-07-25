"""DEM -> drainage graph pipeline, per registered study area.

Steps: fill pits/depressions, resolve flats, D8 flow direction, flow
accumulation, threshold to a channel mask, then build a networkx DiGraph
over channel cells only (not every DEM cell - a 1-degree GLO-30 tile is
~13M cells, but the channel network within it is a few percent of that).
Each node carries its real upstream drainage area (accumulation x cell
area), which is exactly what src/physics/dbi.py needs.

Requires data/study_areas/<name>/config.json to already exist, and a DEM
already downloaded to data/study_areas/<name>/dem_glo30.tif
(src/data_ingestion/download_dem.py, or supply your own).

Usage:
    python src/terrain/process_dem.py [--region <name>|all] [--threshold 1000]
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

np.in1d = np.isin

import networkx as nx
import rasterio
from pysheds.grid import Grid

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from study_areas import registry

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
DIR_OFFSETS = {
    64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
    4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1),
}
METERS_PER_DEG_LAT = 111_320
DEFAULT_CHANNEL_THRESHOLD = 1000


def process_region(region: str, threshold: int) -> dict:
    dem_path = registry.paths(region)["dem"]
    if not dem_path.exists():
        raise FileNotFoundError(f"{dem_path} not found - run download_dem.py first")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = Grid.from_raster(str(dem_path))
        dem = grid.read_raster(str(dem_path))

    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)

    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs

    px_deg = abs(transform.a)
    py_deg = abs(transform.e)
    mean_lat = grid.bbox[1] + (grid.bbox[3] - grid.bbox[1]) / 2
    cell_size_y_m = py_deg * METERS_PER_DEG_LAT
    cell_size_x_m = px_deg * METERS_PER_DEG_LAT * math.cos(math.radians(mean_lat))
    cell_area_km2 = (cell_size_x_m * cell_size_y_m) / 1e6

    channel_mask = np.asarray(acc) > threshold
    fdir_arr = np.asarray(fdir)
    acc_arr = np.asarray(acc)
    rows, cols = np.nonzero(channel_mask)

    graph = nx.DiGraph()
    for r, c in zip(rows.tolist(), cols.tolist()):
        lon, lat = transform * (c + 0.5, r + 0.5)
        graph.add_node(
            (r, c),
            lat=lat,
            lon=lon,
            upstream_area_km2=float(acc_arr[r, c] * cell_area_km2),
        )

    for r, c in zip(rows.tolist(), cols.tolist()):
        direction = int(fdir_arr[r, c])
        offset = DIR_OFFSETS.get(direction)
        if offset is None:
            continue
        nr, nc = r + offset[0], c + offset[1]
        in_bounds = 0 <= nr < channel_mask.shape[0] and 0 <= nc < channel_mask.shape[1]
        if not in_bounds or not channel_mask[nr, nc]:
            continue
        is_diagonal = direction in (128, 2, 8, 32)
        dist_m = math.hypot(cell_size_x_m, cell_size_y_m) if is_diagonal else (
            cell_size_y_m if offset[1] == 0 else cell_size_x_m
        )
        graph.add_edge((r, c), (nr, nc), distance_m=dist_m)

    assert nx.is_directed_acyclic_graph(graph), f"{region}: drainage graph is not a DAG"

    area_paths = registry.paths(region)
    area_paths["processed"].mkdir(parents=True, exist_ok=True)

    channel_mask_path = area_paths["channel_mask"]
    with rasterio.open(dem_path) as src:
        profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(channel_mask_path, "w", **profile) as dst:
        dst.write(channel_mask.astype(np.uint8), 1)

    graph_path = area_paths["drainage_graph"]
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    return {
        "region": region,
        "threshold": threshold,
        "cell_area_km2": cell_area_km2,
        "n_channel_cells": len(rows),
        "n_graph_nodes": graph.number_of_nodes(),
        "n_graph_edges": graph.number_of_edges(),
        "max_upstream_area_km2": max(nx.get_node_attributes(graph, "upstream_area_km2").values()),
        "channel_mask_path": str(channel_mask_path),
        "graph_path": str(graph_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=registry.list_study_areas() + ["all"], default="all")
    parser.add_argument("--threshold", type=int, default=DEFAULT_CHANNEL_THRESHOLD)
    args = parser.parse_args()

    regions = registry.list_study_areas() if args.region == "all" else [args.region]
    for region in regions:
        stats = process_region(region, args.threshold)
        print(f"\n[{region}] done:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
