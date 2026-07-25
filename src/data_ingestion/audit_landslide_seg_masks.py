"""Stratified random audit of landslide_seg mask quality.

Flags masks whose foreground touches the tile border - a signal the scar
was likely cut off by the tile boundary rather than fully captured, one
concrete, checkable form of the "partial annotation" issue in this
dataset. Read-only: samples and reports, does not modify the dataset.
Saves a contact sheet of the flagged pairs for visual confirmation.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "landslide_seg"
SPLITS = ["train", "validation", "test"]
SAMPLE_SIZE = 30
SEED = 20260725
OUT_DIR = REPO_ROOT / "outputs" / "figures"
CONTACT_SHEET_PATH = OUT_DIR / "landslide_seg_mask_audit_flagged.png"


def list_pairs(split: str) -> list[tuple[Path, Path]]:
    img_dir = DATASET_DIR / split / "images"
    mask_dir = DATASET_DIR / split / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.png")):
        suffix = img_path.name.split("_", 1)[1]
        mask_path = mask_dir / f"mask_{suffix}"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def stratified_sample(rng: random.Random) -> list[tuple[str, Path, Path]]:
    split_pairs = {split: list_pairs(split) for split in SPLITS}
    total = sum(len(p) for p in split_pairs.values())
    sample: list[tuple[str, Path, Path]] = []
    remaining = SAMPLE_SIZE
    splits = list(split_pairs.keys())
    for i, split in enumerate(splits):
        if i == len(splits) - 1:
            n = remaining
        else:
            n = round(SAMPLE_SIZE * len(split_pairs[split]) / total)
        n = min(n, len(split_pairs[split]))
        chosen = rng.sample(split_pairs[split], n)
        sample.extend((split, img, mask) for img, mask in chosen)
        remaining -= n
    return sample


def analyze_mask(mask_path: Path) -> dict:
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = mask > 127
    h, w = binary.shape

    labeled, n_components = ndimage.label(binary)
    positive_pixels = int(binary.sum())

    border = np.zeros_like(binary)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    touches_border = bool(np.any(binary & border))

    return {
        "positive_pixels": positive_pixels,
        "foreground_frac": positive_pixels / (h * w),
        "n_components": int(n_components),
        "touches_border": touches_border,
    }


def make_contact_sheet(flagged: list[dict], out_path: Path) -> None:
    if not flagged:
        return
    cell = 128
    pad = 4
    label_h = 14
    cols = min(4, len(flagged))
    rows = (len(flagged) + cols - 1) // cols
    sheet_w = cols * (2 * cell + pad * 3)
    sheet_h = rows * (cell + label_h + pad * 2)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(flagged):
        row, col = divmod(i, cols)
        x0 = col * (2 * cell + pad * 3) + pad
        y0 = row * (cell + label_h + pad * 2) + pad

        img = Image.open(item["image_path"]).convert("RGB").resize((cell, cell), Image.NEAREST)
        mask = Image.open(item["mask_path"]).convert("L").resize((cell, cell), Image.NEAREST).convert("RGB")
        sheet.paste(img, (x0, y0))
        sheet.paste(mask, (x0 + cell + pad, y0))
        label = f"{item['split']}/{item['image_path'].name} ({item['n_components']}c)"
        draw.text((x0, y0 + cell + 1), label[:40], fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    rng = random.Random(SEED)
    sample = stratified_sample(rng)
    print(f"Sampled {len(sample)} masks (seed={SEED}): "
          f"{ {s: sum(1 for x in sample if x[0] == s) for s in SPLITS} }")

    results = []
    for split, img_path, mask_path in sample:
        stats = analyze_mask(mask_path)
        stats.update({"split": split, "image_path": img_path, "mask_path": mask_path})
        results.append(stats)

    flagged = [r for r in results if r["touches_border"]]

    print(f"\n{'split':<12}{'image':<24}{'pos_px':>8}{'frac':>8}{'components':>12}{'border?':>10}")
    for r in sorted(results, key=lambda r: (r["split"], r["image_path"].name)):
        print(
            f"{r['split']:<12}{r['image_path'].name:<24}{r['positive_pixels']:>8}"
            f"{r['foreground_frac']:>8.3f}{r['n_components']:>12}"
            f"{'YES' if r['touches_border'] else '':>10}"
        )

    print(f"\nFlagged (foreground touches tile border): {len(flagged)}/{len(results)} "
          f"({100 * len(flagged) / len(results):.1f}%)")
    for r in flagged:
        print(f"  {r['split']}/{r['image_path'].name} - {r['n_components']} component(s), "
              f"{r['positive_pixels']}px foreground")

    make_contact_sheet(flagged, CONTACT_SHEET_PATH)
    if flagged:
        print(f"\nContact sheet saved: {CONTACT_SHEET_PATH}")
    else:
        print("\nNo flagged masks - no contact sheet written.")


if __name__ == "__main__":
    main()
