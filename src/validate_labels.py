from collections import Counter
from pathlib import Path

import cv2

from common import CLASS_NAMES, data_dir, label_for_image, list_images, read_labels, repo_root, save_json

SPLITS = ["train", "val", "test"]
NC = len(CLASS_NAMES)


def main():
    data = data_dir()
    problems = []
    stats = {}
    for split in SPLITS:
        images = list_images(data / "images" / split)
        per_class = Counter()
        boxes_total = 0
        empty = 0
        widths = []
        for img in images:
            boxes = read_labels(label_for_image(img))
            if not boxes:
                empty += 1
                continue
            for cls, x, y, w, h in boxes:
                boxes_total += 1
                per_class[cls] += 1
                if not (0 <= cls < NC):
                    problems.append(f"{split}/{img.name}: class {cls} out of range")
                if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                    problems.append(f"{split}/{img.name}: coords out of [0,1]: {x},{y},{w},{h}")
                if w <= 0 or h <= 0:
                    problems.append(f"{split}/{img.name}: non-positive box size")
                if x - w / 2 < -0.001 or y - h / 2 < -0.001 or x + w / 2 > 1.001 or y + h / 2 > 1.001:
                    problems.append(f"{split}/{img.name}: box exceeds image: {x},{y},{w},{h}")
            widths.append(cv2.imread(str(img)).shape[1])
        stats[split] = {
            "images": len(images),
            "boxes": boxes_total,
            "empty_label_images": empty,
            "boxes_per_class": {CLASS_NAMES[c]: per_class.get(c, 0) for c in range(NC)},
            "mean_width_px": round(sum(widths) / len(widths), 1) if widths else 0,
        }
    report = {"ok": not problems, "problems": problems[:50], "problem_count": len(problems), "stats": stats}
    save_json(report, repo_root() / "reports" / "label_validation.json")
    save_json({"splits": stats}, data_dir() / "dataset_stats.json")
    for s in SPLITS:
        print(s, stats[s])
    if problems:
        for p in problems[:20]:
            print("PROBLEM:", p)
        raise SystemExit(1)
    print("label validation: OK")


if __name__ == "__main__":
    main()
