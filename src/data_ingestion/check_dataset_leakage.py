"""Perceptual-hash (phash) audit for train/val/test leakage in Model A's
training datasets.

Checks two things, neither of which the original dataset packaging
documented:
  1. Within each dataset, does any image (or a near-duplicate crop of it)
     appear across a train/test or train/val boundary? If so, reported
     test-set metrics are partly measuring memorization, not generalization.
  2. Do the two datasets (landslide_seg, nepal_landslide_seg) share any
     images? They were sourced independently, but nothing before this
     script actually checked.

Read-only - reports file paths, does not modify or delete anything.
"""
from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "landslide_seg": REPO_ROOT / "datasets" / "landslide_seg",
    "nepal_landslide_seg": REPO_ROOT / "datasets" / "nepal_landslide_seg",
}
SPLITS = ["train", "validation", "test"]
HAMMING_THRESHOLD = 5


def hash_dataset(dataset_dir: Path) -> dict[str, dict[Path, imagehash.ImageHash]]:
    result: dict[str, dict[Path, imagehash.ImageHash]] = {}
    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        if not img_dir.exists():
            continue
        split_hashes = {}
        for img_path in sorted(img_dir.glob("*.png")):
            try:
                split_hashes[img_path] = imagehash.phash(Image.open(img_path))
            except Exception as e:
                print(f"WARN: failed to hash {img_path}: {e}")
        result[split] = split_hashes
    return result


def find_matches(items_a, items_b, threshold=HAMMING_THRESHOLD):
    """items_a/items_b: list of (label, path, hash). Returns list of
    (label_a, path_a, label_b, path_b, hamming_distance)."""
    matches = []
    for label_a, path_a, hash_a in items_a:
        for label_b, path_b, hash_b in items_b:
            dist = hash_a - hash_b
            if dist <= threshold:
                matches.append((label_a, path_a, label_b, path_b, dist))
    return matches


def flatten(hashes_by_split: dict[str, dict[Path, imagehash.ImageHash]]):
    return [(split, path, h) for split, d in hashes_by_split.items() for path, h in d.items()]


def main() -> None:
    all_hashes = {}
    for name, dataset_dir in DATASETS.items():
        print(f"Hashing {name} ({dataset_dir})...")
        all_hashes[name] = hash_dataset(dataset_dir)
        counts = {split: len(d) for split, d in all_hashes[name].items()}
        print(f"  {counts}")

    print("\n=== Within-dataset cross-split leakage ===")
    any_within = False
    for name, hashes_by_split in all_hashes.items():
        splits_present = list(hashes_by_split.keys())
        for i, split_a in enumerate(splits_present):
            for split_b in splits_present[i + 1 :]:
                items_a = [(split_a, p, h) for p, h in hashes_by_split[split_a].items()]
                items_b = [(split_b, p, h) for p, h in hashes_by_split[split_b].items()]
                matches = find_matches(items_a, items_b)
                if matches:
                    any_within = True
                    print(f"\n{name}: {split_a} <-> {split_b} - {len(matches)} match(es)")
                    for label_a, path_a, label_b, path_b, dist in matches:
                        exact = " [EXACT]" if dist == 0 else ""
                        print(f"  dist={dist}{exact}  {path_a}  <->  {path_b}")
    if not any_within:
        print("None found (checked all train/val, train/test, val/test pairs in both datasets).")

    print("\n=== Cross-dataset duplicates (landslide_seg <-> nepal_landslide_seg) ===")
    items_a = flatten(all_hashes["landslide_seg"])
    items_b = flatten(all_hashes["nepal_landslide_seg"])
    items_a = [(f"landslide_seg/{split}", p, h) for split, p, h in items_a]
    items_b = [(f"nepal_landslide_seg/{split}", p, h) for split, p, h in items_b]
    cross_matches = find_matches(items_a, items_b)
    if cross_matches:
        for label_a, path_a, label_b, path_b, dist in cross_matches:
            exact = " [EXACT]" if dist == 0 else ""
            print(f"  dist={dist}{exact}  {path_a}: {path_a.name}  <->  {label_b}: {path_b.name}")
    else:
        print(f"None found ({len(items_a)} x {len(items_b)} comparisons).")


if __name__ == "__main__":
    main()
