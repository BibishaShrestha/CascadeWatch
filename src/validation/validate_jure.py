"""Historical validation routine: run Model A + DBI physics (with the
terrain lookup) on historical imagery for the Jure/Sunkoshi 2014 breach
case and a non-breach-outcome-unconfirmed control case, then compare
against Model B's output on post-event imagery.

Scope note: the exposure *zone* doesn't exist as spatial geometry -
physics/exposure.py returns a list of named assets with a distance-decay
score, not a polygon, so there's nothing to compute a geometric IoU
against on that side. This script instead compares Model A's detected
scar footprint against Model B's NDWI water detection (McFeeters 1996 -
see models/water_detection.py) as an overlap metric. That IoU number
isn't a meaningful validation result on its own, because Model A detects
0 scars on this imagery (see the domain-gap note below): one side of the
comparison (water mask) is populated, the other (scar detection) is
empty/sub-threshold.

Note: these Landsat 8 chips are natively 30m/px, not the ~10m/px scale
Model A was trained on (Landslide4Sense/Sentinel-2-derived) - area_m2 here
uses the 30m GSD, and detection quality is subject to a train/test
resolution domain shift.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.inference_api import polygon_area_m2, predict_flood, predict_landslide
from physics.dbi import assess_scar
from terrain.lookup import lookup_terrain_at

LANDSAT8_GSD_M_PER_PX = 30.0

CASES = {
    "jure": {
        "lat": 27.76767, "lon": 85.87099, "region": "sunkoshi",
        "post_landslide_img": REPO_ROOT / "outputs" / "figures" / "validation_previews" / "sunkoshi_jure_post_landslide.png",
        "post_breach_img": REPO_ROOT / "outputs" / "figures" / "validation_previews" / "sunkoshi_jure_post_breach.png",
        "post_breach_tif": REPO_ROOT / "data" / "study_areas" / "sunkoshi" / "validation" / "jure" / "post_breach.tif",
        "known_outcome": "breach (2014-09-07, historical record)",
    },
    "control": {
        "lat": 28.21898, "lon": 85.48355, "region": "trishuli",
        "post_landslide_img": REPO_ROOT / "outputs" / "figures" / "validation_previews" / "trishuli_control_post_event.png",
        "post_breach_img": None,
        "post_breach_tif": None,
        "known_outcome": "unconfirmed (no reported breach disaster found - not a verified non-breach ground truth)",
    },
}


def polygon_iou(poly_a: list[list[float]], poly_b: list[list[float]]) -> float:
    """IoU between two polygons via shapely - see the module docstring for
    why the scar-vs-flood-mask comparison it's used for isn't yet
    meaningful on its own."""
    from shapely.geometry import Polygon

    a = Polygon(poly_a).buffer(0)
    b = Polygon(poly_b).buffer(0)
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def run_case(name: str, verbose: bool = True) -> dict:
    """Run Model A + physics on the case's post-landslide/post-event image,
    and Model B's NDWI water detection on its post-breach .tif if it has
    one.

    Returns a dict (detections, scar_results, sub-threshold diagnostic,
    flood detections, scar-vs-flood-mask IoU) so callers - the CLI `main()`
    below, or the Streamlit live validation panel - can both drive off the
    same computation rather than duplicating it.
    """
    case = CASES[name]

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n=== {name} (known outcome: {case['known_outcome']}) ===")

    detections = predict_landslide(str(case["post_landslide_img"]))
    log(f"Model A: {len(detections)} scar(s) detected in post-landslide imagery (conf>=0.25, production default)")

    subthreshold = None
    if not detections:
        for conf in (0.05, 0.01):
            weak = predict_landslide(str(case["post_landslide_img"]), conf=conf)
            if weak:
                subthreshold = {"conf": conf, "confidences": [round(d["confidence"], 3) for d in weak]}
                log(f"  [diagnostic] {len(weak)} sub-threshold detection(s) at conf>={conf}: "
                    f"{subthreshold['confidences']} - weak signal, "
                    "consistent with a Landsat/Sentinel-2 sensor domain gap, not zero signal.")
                break

    scar_results = []
    for i, det in enumerate(detections):
        area_m2 = polygon_area_m2(det["mask_polygon"], gsd_m_per_px=LANDSAT8_GSD_M_PER_PX)
        if area_m2 <= 0:
            continue
        terrain = lookup_terrain_at(case["region"], case["lat"], case["lon"])
        scar = assess_scar(
            scar_id=f"{name}-scar{i}",
            area_m2=area_m2,
            upstream_area_km2=terrain["upstream_area_km2"],
            local_relief_m=terrain["local_relief_m"],
            graph_node_id=str(terrain["nearest_channel_node"]),
        )
        scar["confidence"] = det["confidence"]
        scar["mask_polygon_px"] = det["mask_polygon"]
        scar_results.append(scar)
        log(f"  scar{i}: area={area_m2:.0f} m2, upstream_area={terrain['upstream_area_km2']:.2f} km2, "
            f"dbi={scar['dbi']:.3f}, verdict={scar['verdict']}")

    flood_detections = []
    iou = None
    if case["post_breach_tif"] is not None:
        flood_detections = predict_flood(str(case["post_breach_tif"]))
        log(f"Model B (NDWI, not a trained model): {len(flood_detections)} flood polygon(s)")
        if scar_results and flood_detections:
            iou = polygon_iou(scar_results[0]["mask_polygon_px"], flood_detections[0]["mask_polygon"])
            log(f"  scar-vs-flood-mask IoU: {iou:.3f} - Model A detected 0 scars on this "
                f"imagery (see the domain-gap note above), so this isn't a meaningful "
                f"validation result yet - it's comparing a water mask against an "
                f"empty/sub-threshold scar detection, not two independent detections "
                f"of the same event.")

    return {
        "case": name,
        "known_outcome": case["known_outcome"],
        "region": case["region"],
        "lat": case["lat"],
        "lon": case["lon"],
        "post_landslide_img": case["post_landslide_img"],
        "post_breach_img": case["post_breach_img"],
        "detections": detections,
        "subthreshold": subthreshold,
        "scar_results": scar_results,
        "n_scars": len(scar_results),
        "verdicts": [s["verdict"] for s in scar_results],
        "flood_detections": flood_detections,
        "iou": iou,
    }


def main() -> None:
    results = [run_case(name) for name in CASES]
    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
