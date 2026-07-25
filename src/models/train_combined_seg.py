"""Train YOLOv8n-seg on the combined landslide_seg + nepal_landslide_seg
training pools (Model A, Nepal-grounded version).

Trains on the union of both datasets' train+val splits (see
datasets/combined_seg_yolo_data.yaml), then reports test-set metrics
separately on:
  - landslide_seg's own test split (global, Landslide4Sense-derived)
  - nepal_landslide_seg's own test split (small, only 10 images, but the
    only Nepal-imagery held-out set - the real answer to "does this work
    on Nepal imagery")
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from model_config import write_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINED_DATA = REPO_ROOT / "datasets" / "combined_seg_yolo_data.yaml"
LANDSLIDE_SEG_DATA = REPO_ROOT / "datasets" / "landslide_seg_yolo" / "data.yaml"
NEPAL_DATA = REPO_ROOT / "datasets" / "nepal_landslide_seg_yolo" / "data.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="combined_yolov8n_seg")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(COMBINED_DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(OUTPUT_DIR),
        name=args.name,
        device=args.device,
        exist_ok=True,
    )

    print("\n=== landslide_seg TEST split metrics (global/Landslide4Sense-derived) ===")
    global_test = model.val(
        data=str(LANDSLIDE_SEG_DATA), split="test",
        project=str(OUTPUT_DIR / args.name), name="global_test_metrics",
    )
    print(global_test.results_dict)

    print(
        "\n=== nepal_landslide_seg TEST split metrics (real Nepal imagery, "
        "n=10, indicative not conclusive - not statistically robust) ==="
    )
    nepal_test = model.val(
        data=str(NEPAL_DATA), split="test",
        project=str(OUTPUT_DIR / args.name), name="nepal_test_metrics",
    )
    print(nepal_test.results_dict)
    print("^ (n=10, indicative not conclusive - not statistically robust)")

    model_config = write_model_config(
        OUTPUT_DIR / args.name / "weights", [LANDSLIDE_SEG_DATA, NEPAL_DATA]
    )
    print(f"\nWrote model_config.json: gsd_m_per_px={model_config['gsd_m_per_px']}")


if __name__ == "__main__":
    main()
