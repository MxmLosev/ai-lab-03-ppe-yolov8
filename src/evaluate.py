import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO, settings

from common import CLASS_NAMES, data_dir, label_for_image, list_images, read_labels, repo_root, save_json

SPLITS = ["train", "val", "test"]


def test_class_counts(names):
    counts = Counter()
    for img in list_images(data_dir() / "images" / "test"):
        for box in read_labels(label_for_image(img)):
            counts[int(box[0])] += 1
    return {names[c]: counts.get(c, 0) for c in range(len(names))}


def evaluate(weights: Path, imgsz: int, tag: str):
    model = YOLO(str(weights))
    metrics = model.val(data="configs/ppe.yaml", split="test", imgsz=imgsz, plots=True, name=tag, exist_ok=True, project="runs/ppe")
    names = metrics.names
    per_class = {}
    present = list(metrics.box.ap_class_index)
    for j, c in enumerate(present):
        per_class[names[c]] = {
            "precision": float(metrics.box.p[j]),
            "recall": float(metrics.box.r[j]),
            "ap50": float(metrics.box.ap50[j]),
            "ap50_95": float(metrics.box.ap[j]),
            "test_instances": test_class_counts(names)[names[c]],
        }
    for name in CLASS_NAMES:
        if name not in per_class:
            per_class[name] = {"precision": None, "recall": None, "ap50": None, "ap50_95": None, "test_instances": test_class_counts(names)[name]}
    report = {
        "weights": str(weights),
        "imgsz": imgsz,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
        "test_instances_per_class": test_class_counts(names),
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
    }
    cm = np.asarray(metrics.confusion_matrix.matrix)
    np.save(repo_root() / "reports" / f"confusion_{tag}.npy", cm)
    save_json(report, repo_root() / "reports" / f"metrics_{tag}.json")
    print(f"[{tag}] mAP50={report['overall']['map50']:.3f} mAP50-95={report['overall']['map50_95']:.3f} P={report['overall']['precision']:.3f} R={report['overall']['recall']:.3f}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    settings.update({"datasets_dir": str(repo_root())})
    evaluate(Path(args.weights), args.imgsz, args.tag)


if __name__ == "__main__":
    main()
