import sys
from collections import defaultdict
from pathlib import Path

from common import data_dir, group_key, label_for_image, list_images, md5, repo_root, save_json

SPLITS = ["train", "val", "test"]


def main():
    data = data_dir()
    problems = []
    stats = {}
    seen_names = {}
    seen_hashes = {}
    seen_groups = {}
    for split in SPLITS:
        images = list_images(data / "images" / split)
        hashes = {}
        groups = defaultdict(int)
        for img in images:
            if img.name in seen_names:
                problems.append(f"duplicate filename across {seen_names[img.name]} and {split}: {img.name}")
            seen_names[img.name] = split
            h = md5(img)
            hashes[img.name] = h
            if h in seen_hashes:
                problems.append(f"duplicate content across {seen_hashes[h]} and {split}: {img.name}")
            seen_hashes[h] = split
            g = group_key(img.stem)
            groups[g] += 1
            if g in seen_groups and seen_groups[g] != split:
                problems.append(f"group {g} in {seen_groups[g]} and {split}")
            seen_groups[g] = split
            lbl = label_for_image(img)
            if not lbl.exists():
                problems.append(f"missing label: {split}/{img.stem}.txt")
        stats[split] = {"images": len(images), "groups": len(groups), "images_per_group": sorted(groups.values())}

    report = {"ok": not problems, "problems": problems[:50], "problem_count": len(problems), "stats": stats}
    save_json(report, repo_root() / "reports" / "leak_check.json")
    for s in SPLITS:
        st = stats[s]
        print(f"{s}: {st['images']} images, {st['groups']} groups")
    if problems:
        for p in problems[:20]:
            print("PROBLEM:", p)
        sys.exit(1)
    print("leak check: OK")


if __name__ == "__main__":
    main()
