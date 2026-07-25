"""Train YOLOv8n-seg on the converted landslide_seg_yolo dataset (Model A).

Reusable for Model B (flood_seg) later: just point --data/--name at the
flood dataset once it exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from model_config import write_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "datasets" / "landslide_seg_yolo" / "data.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="landslide_yolov8n_seg")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(OUTPUT_DIR),
        name=args.name,
        device=args.device,
        exist_ok=True,
    )

    print("\n=== VAL split metrics ===")
    val_metrics = model.val(
        data=args.data, split="val", project=str(OUTPUT_DIR / args.name), name="val_metrics"
    )
    print(val_metrics.results_dict)

    print("\n=== TEST split metrics ===")
    test_metrics = model.val(
        data=args.data, split="test", project=str(OUTPUT_DIR / args.name), name="test_metrics"
    )
    print(test_metrics.results_dict)

    model_config = write_model_config(OUTPUT_DIR / args.name / "weights", [Path(args.data)])
    print(f"\nWrote model_config.json: gsd_m_per_px={model_config['gsd_m_per_px']}")


if __name__ == "__main__":
    main()
