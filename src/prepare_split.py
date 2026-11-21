import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    CLASS_NAMES,
    MIN_INSTANCES,
    SEED,
    TARGET_TOTAL,
    data_dir,
    read_labels,
    save_json,
)

POOL = data_dir() / "_pool"
MAX_GROUP_FOR_SMALL_SPLITS = 30
SPLITS = ["train", "val", "test"]


def pool_records():
    import json

    manifest = json.loads((POOL / "pool_manifest.json").read_text(encoding="utf-8"))
    return manifest["records"]


def class_counts_of_groups(records):
    counts = defaultdict(Counter)
    sizes = Counter()
    for rec in records:
        stem = Path(rec["file"]).stem
        boxes = read_labels(POOL / "labels" / (stem + ".txt"))
        counts[rec["group"]].update(b[0] for b in boxes)
        sizes[rec["group"]] += 1
    return counts, sizes


def main():
    records = pool_records()
    counts, sizes = class_counts_of_groups(records)
    groups = sorted(counts.keys())
    rng = random.Random(SEED)

    ratio = {"train": 0.70, "val": 0.15, "test": 0.15}
    desired = {s: round(TARGET_TOTAL * ratio[s]) for s in ratio}
    assignment = {s: [] for s in SPLITS}
    images_in = {s: 0 for s in SPLITS}
    inst_in = {s: Counter() for s in SPLITS}
    groups_left = set(groups)
    assigned = {}

    def take(g, split):
        assignment[split].append(g)
        assigned[g] = split
        groups_left.discard(g)
        images_in[split] += sizes[g]
        inst_in[split].update(counts[g])

    for split in ["test", "val", "train"]:
        for cls in sorted(range(len(CLASS_NAMES)), key=lambda c: sum(counts[g][c] for g in groups)):
            need = MIN_INSTANCES[split]
            cap = {"train": 60, "val": MAX_GROUP_FOR_SMALL_SPLITS, "test": MAX_GROUP_FOR_SMALL_SPLITS}[split]
            while inst_in[split][cls] < need:
                cands = [g for g in groups_left if counts[g][cls] > 0 and sizes[g] <= cap]
                if not cands:
                    cands = [g for g in groups_left if counts[g][cls] > 0]
                if not cands:
                    break
                g = max(cands, key=lambda g: (counts[g][cls], -sizes[g]))
                take(g, split)

    small_left = sorted(groups_left, key=lambda g: (sizes[g], rng.random()))
    for g in small_left:
        deficits = {s: desired[s] - images_in[s] for s in SPLITS}
        if all(d <= 0 for d in deficits.values()):
            break
        max_size = {"train": 60, "val": MAX_GROUP_FOR_SMALL_SPLITS, "test": MAX_GROUP_FOR_SMALL_SPLITS}
        cands = {s: deficits[s] for s in SPLITS if deficits[s] > 0 and sizes[g] <= max_size[s]}
        if not cands:
            continue
        take(g, max(cands, key=lambda s: cands[s]))

    if 0 < desired["train"] - images_in["train"] <= 100:
        top_up = sorted(groups_left, key=lambda g: (sizes[g], g))
        for g in top_up:
            if images_in["train"] >= desired["train"]:
                break
            take(g, "train")

    data = data_dir()
    for split in SPLITS:
        (data / "images" / split).mkdir(parents=True, exist_ok=True)
        (data / "labels" / split).mkdir(parents=True, exist_ok=True)
        for old in (data / "images" / split).iterdir():
            old.unlink()
        for old in (data / "labels" / split).iterdir():
            old.unlink()
    manifest = {s: {} for s in SPLITS}
    used = 0
    for rec in records:
        split = assigned.get(rec["group"])
        if split is None:
            continue
        used += 1
        stem = Path(rec["file"]).stem
        shutil.copy2(POOL / "images" / rec["file"], data / "images" / split / rec["file"])
        shutil.copy2(POOL / "labels" / (stem + ".txt"), data / "labels" / split / (stem + ".txt"))
        manifest[split][rec["file"]] = rec["group"]
    print(f"used {used} of {len(records)} pool images")

    print(f"{'split':6} {'images':>6} {'groups':>6} {'target':>6}")
    for s in SPLITS:
        print(f"{s:6} {images_in[s]:>6} {len(assignment[s]):>6} {desired[s]:>6}")
    print(f"total {sum(images_in.values())} (target {TARGET_TOTAL})")
    for s in SPLITS:
        missing = [CLASS_NAMES[c] for c in range(len(CLASS_NAMES)) if inst_in[s][c] == 0]
        low = {CLASS_NAMES[c]: inst_in[s][c] for c in range(len(CLASS_NAMES)) if 0 < inst_in[s][c] < MIN_INSTANCES[s]}
        print(f"{s}: missing classes: {missing or 'none'}; below min: {low or 'none'}")
    save_json(
        {
            "seed": SEED,
            "target_total": TARGET_TOTAL,
            "desired": desired,
            "group_rule": "source prefix before extension marker, trailing frame index collapsed, video series kept together",
            "assignment": manifest,
        },
        data_dir() / "split_manifest.json",
    )


if __name__ == "__main__":
    main()
