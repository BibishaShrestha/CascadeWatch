"""Dimensionless Blockage Index (DBI) physics engine.

Landslide-dam breach risk from scar geometry, following Larsen et al. (2010)
for scar-volume scaling and the standard DBI blockage-index formulation.

Reference worked example (from project spec):
    area_m2=2e5 -> volume~1.6e6 m^3 -> dam height~117m -> upstream_area_km2=250
    -> DBI~4.26 -> "breach_risk"
"""
import math

DBI_STABLE_MAX = 2.75
DBI_UNCERTAIN_MAX = 3.08


def scar_volume_m3(area_m2: float) -> float:
    """Landslide volume from planform scar area (Larsen et al. 2010)."""
    if area_m2 <= 0:
        raise ValueError(f"area_m2 must be positive, got {area_m2}")
    return 0.146 * area_m2 ** 1.332


def dam_height_m(volume_m3: float, local_relief_m: float | None = None) -> float:
    """Dam height from deposited volume, capped at the valley's local relief.

    The uncapped height assumes the debris pile could grow arbitrarily tall;
    in reality it can't exceed the valley wall height (local relief), so a
    dam blocked in a shallow valley saturates in height rather than dbi
    growing unbounded.
    """
    if volume_m3 <= 0:
        raise ValueError(f"volume_m3 must be positive, got {volume_m3}")
    h = volume_m3 ** (1.0 / 3.0)
    if local_relief_m is not None:
        h = min(h, local_relief_m)
    return h


def compute_dbi(upstream_area_km2: float, dam_height_m_: float, volume_m3: float) -> float:
    """Dimensionless Blockage Index.

    DBI = log10( upstream_drainage_area_km2 * dam_height_m / deposit_volume_Mm3 )
    Higher DBI = larger dam relative to the debris volume that built it,
    i.e. a big blockage sitting on a big river -> more likely to fail/breach.
    """
    if upstream_area_km2 <= 0:
        raise ValueError(f"upstream_area_km2 must be positive, got {upstream_area_km2}")
    if dam_height_m_ <= 0:
        raise ValueError(f"dam_height_m must be positive, got {dam_height_m_}")
    if volume_m3 <= 0:
        raise ValueError(f"volume_m3 must be positive, got {volume_m3}")
    volume_mm3 = volume_m3 / 1e6
    return math.log10(upstream_area_km2 * dam_height_m_ / volume_mm3)


def classify_dbi(dbi: float) -> str:
    if dbi < DBI_STABLE_MAX:
        return "stable"
    if dbi <= DBI_UNCERTAIN_MAX:
        return "uncertain"
    return "breach_risk"


def assess_scar(
    scar_id: str,
    area_m2: float,
    upstream_area_km2: float,
    local_relief_m: float | None = None,
    graph_node_id: str | None = None,
) -> dict:
    """Run the full DBI pipeline for one landslide scar.

    Returns the frozen physics-engine output schema (scar_id, area_m2, dbi,
    verdict, graph_node_id), plus additive debug fields (volume_m3,
    dam_height_m) for display/logging — downstream consumers should only
    rely on the four frozen keys.
    """
    volume = scar_volume_m3(area_m2)
    height = dam_height_m(volume, local_relief_m=local_relief_m)
    dbi = compute_dbi(upstream_area_km2, height, volume)
    verdict = classify_dbi(dbi)
    return {
        "scar_id": scar_id,
        "area_m2": area_m2,
        "dbi": dbi,
        "verdict": verdict,
        "graph_node_id": graph_node_id,
        "volume_m3": volume,
        "dam_height_m": height,
        "upstream_area_km2": upstream_area_km2,
    }
