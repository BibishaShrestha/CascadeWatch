"""Writes/reads model_config.json - the real-GSD provenance record a
training run leaves next to its best.pt, so inference_api.py never has to
guess a model's GSD from a blind module constant.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_GSD_M_PER_PX = 10.0

MODEL_CONFIG_FILENAME = "model_config.json"


def _source_dataset_dir(yolo_data_yaml: Path) -> Path:
    """<repo>/datasets/<name>_yolo/data.yaml -> <repo>/datasets/<name>/"""
    yolo_dir = yolo_data_yaml.parent
    name = yolo_dir.name
    if name.endswith("_yolo"):
        name = name[: -len("_yolo")]
    return yolo_dir.parent / name


def write_model_config(weights_dir: Path, data_yaml_paths: list[Path]) -> dict:
    """Read gsd_m_per_px from each dataset's dataset_config.json (sibling of
    the raw dataset dir a yolo-converted data.yaml was built from) and write
    a merged model_config.json into weights_dir (the folder holding best.pt).

    Raises ValueError if the datasets used for one training run disagree on
    GSD - training on a silently mismatched GSD would make
    inference_api.py's area_m2 calculation wrong for whichever dataset
    didn't match the reported value.
    """
    gsds: dict[str, float] = {}
    sources = []
    for data_yaml in data_yaml_paths:
        dataset_dir = _source_dataset_dir(Path(data_yaml))
        config_path = dataset_dir / "dataset_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        gsd = config.get("gsd_m_per_px")
        if gsd is not None:
            gsds[dataset_dir.name] = gsd
        sources.append({"dataset": dataset_dir.name, **config})

    if not gsds:
        gsd_m_per_px = DEFAULT_GSD_M_PER_PX
    elif len(set(gsds.values())) > 1:
        raise ValueError(
            f"Datasets disagree on gsd_m_per_px: {gsds} - resolve this before "
            "training a model on their combination."
        )
    else:
        gsd_m_per_px = next(iter(gsds.values()))

    model_config = {"gsd_m_per_px": gsd_m_per_px, "sources": sources}
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / MODEL_CONFIG_FILENAME).write_text(json.dumps(model_config, indent=2))
    return model_config


def read_model_config(weights_path: Path) -> dict | None:
    """weights_path is best.pt - looks for model_config.json in the same directory."""
    config_path = Path(weights_path).parent / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())
