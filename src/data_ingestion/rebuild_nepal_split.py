"""Rebuild nepal_landslide_seg's train/val/test split to fix the leakage
found by check_dataset_leakage.py (94% of validation and up to 30% of test
were exact duplicates of train images, inherited from the raw dataset's own
pre-existing split).

Validation/Test's original tile identity was discarded (renamed to plain
sequential integers in the raw download), so source-image grouping can't
be recovered from filenames. Instead: hash every image (phash), union
near-duplicates (hamming distance <= 5) into groups, then assign whole
groups to splits so no group's members can land in more than one split -
preserving the original ~83.6/12.7/3.6% train/val/test ratio as closely as
group sizes allow.

The original (leaky) organized copy is renamed aside, not deleted - the
raw source (datasets/nepal_landslide_seg_raw/) is the backup.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import imagehash
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "nepal_landslide_seg"
BACKUP_DIR = REPO_ROOT / "datasets" / "nepal_landslide_seg_leaky_v1_backup"
OLD_SPLITS = ["train", "validation", "test"]
HAMMING_THRESHOLD = 5
SEED = 20260726

ORIGINAL_COUNTS = {"train": 230, "validation": 35, "test": 10}
TARGET_FRACTIONS = {k: v / sum(ORIGINAL_COUNTS.values()) for k, v in ORIGINAL_COUNTS.items()}


def list_pairs(base_dir: Path, split: str) -> list[tuple[Path, Path]]:
    img_dir = base_dir / split / "images"
    mask_dir = base_dir / split / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.png")):
        suffix = img_path.name.split("_", 1)[1]
        mask_path = mask_dir / f"mask_{suffix}"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_groups(all_pairs: list[tuple[str, Path, Path]]) -> list[list[tuple[str, Path, Path]]]:
    """all_pairs: (old_split, image_path, mask_path). Returns clusters of
    near-duplicate items (hamming distance <= threshold), each treated as
    one indivisible group for splitting."""
    hashes = {}
    for old_split, img_path, mask_path in all_pairs:
        hashes[img_path] = imagehash.phash(Image.open(img_path))

    uf = UnionFind([img_path for _, img_path, _ in all_pairs])
    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] <= HAMMING_THRESHOLD:
                uf.union(paths[i], paths[j])

    groups: dict[Path, list[tuple[str, Path, Path]]] = {}
    for old_split, img_path, mask_path in all_pairs:
        root = uf.find(img_path)
        groups.setdefault(root, []).append((old_split, img_path, mask_path))
    return list(groups.values())


def assign_groups_to_splits(groups: list[list]) -> dict[str, list[list]]:
    """Greedy group-stratified split: repeatedly assign the largest
    remaining group to whichever split is currently furthest below its
    target proportion of already-assigned items."""
    groups_sorted = sorted(groups, key=len, reverse=True)
    assigned: dict[str, list[list]] = {k: [] for k in TARGET_FRACTIONS}
    counts = {k: 0 for k in TARGET_FRACTIONS}
    total_assigned = 0

    for group in groups_sorted:
        if total_assigned == 0:
            deficits = {k: TARGET_FRACTIONS[k] for k in TARGET_FRACTIONS}
        else:
            deficits = {
                k: TARGET_FRACTIONS[k] - counts[k] / total_assigned
                for k in TARGET_FRACTIONS
            }
        target_split = max(deficits, key=deficits.get)
        assigned[target_split].append(group)
        counts[target_split] += len(group)
        total_assigned += len(group)

    return assigned


def write_split(assigned: dict[str, list[list]]) -> dict[str, int]:
    """img_path/mask_path in each group were captured before DATASET_DIR was
    moved aside to BACKUP_DIR - remap them to their new (still-real) location
    under BACKUP_DIR before reading.

    Old validation/test used generic sequential names (image_0.png ...
    image_9.png) that COLLIDE across old splits despite being different
    content - confirmed via `comm` on the two directories. Destination
    filenames are prefixed with the old split name to guarantee uniqueness;
    without this, same-named-different-content files silently overwrote
    each other on the first attempt (lost 6 of 275 pairs before this fix).
    """
    counts = {}
    for split, groups in assigned.items():
        dst_img_dir = DATASET_DIR / split / "images"
        dst_mask_dir = DATASET_DIR / split / "masks"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_mask_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for group in groups:
            for old_split, img_path, mask_path in group:
                real_img_path = BACKUP_DIR / old_split / "images" / img_path.name
                real_mask_path = BACKUP_DIR / old_split / "masks" / mask_path.name
                new_stem = f"{old_split}_{img_path.stem[len('image_'):]}"
                dst_img_path = dst_img_dir / f"image_{new_stem}.png"
                dst_mask_path = dst_mask_dir / f"mask_{new_stem}.png"
                assert not dst_img_path.exists(), f"collision: {dst_img_path}"
                dst_img_path.write_bytes(real_img_path.read_bytes())
                dst_mask_path.write_bytes(real_mask_path.read_bytes())
                n += 1
        counts[split] = n
    return counts


def main() -> None:
    if DATASET_DIR.exists():
        print("Backing up old (leaky) organized copy...")
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.move(str(DATASET_DIR), str(BACKUP_DIR))
    else:
        print(f"{DATASET_DIR} already moved aside (resuming after a prior "
              f"interrupted run) - using existing {BACKUP_DIR} as source.")
        assert BACKUP_DIR.exists(), f"{BACKUP_DIR} must exist if {DATASET_DIR} doesn't"

    print("Loading all existing pairs across old train/validation/test (from backup)...")
    all_pairs = []
    for split in OLD_SPLITS:
        for img_path, mask_path in list_pairs(BACKUP_DIR, split):
            all_pairs.append((split, img_path, mask_path))
    print(f"Total pairs: {len(all_pairs)}")

    print("Hashing + clustering near-duplicates (hamming <= 5)...")
    groups = build_groups(all_pairs)
    group_sizes = sorted((len(g) for g in groups), reverse=True)
    n_multi = sum(1 for s in group_sizes if s > 1)
    print(f"{len(groups)} groups total ({n_multi} groups have >1 member, "
          f"largest group has {group_sizes[0]} members)")

    print("Assigning groups to splits (group-stratified, preserving ~83.6/12.7/3.6% ratio)...")
    assigned = assign_groups_to_splits(groups)

    print("Writing new split...")
    new_counts = write_split(assigned)

    old_config = BACKUP_DIR / "dataset_config.json"
    if old_config.exists():
        shutil.copy(old_config, DATASET_DIR / "dataset_config.json")
        print(f"Carried forward dataset_config.json from {old_config}")

    total = sum(new_counts.values())
    print(f"\nOriginal counts: {ORIGINAL_COUNTS} (ratio "
          f"{ORIGINAL_COUNTS['train']/275:.3f}/{ORIGINAL_COUNTS['validation']/275:.3f}/{ORIGINAL_COUNTS['test']/275:.3f})")
    print(f"New counts:      {new_counts} (ratio "
          f"{new_counts['train']/total:.3f}/{new_counts['validation']/total:.3f}/{new_counts['test']/total:.3f})")
    print(f"\nOld (leaky) organized copy preserved at: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
