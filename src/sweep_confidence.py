import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, settings

from common import CLASS_NAMES, data_dir, greedy_match, label_for_image, list_images, read_labels, repo_root, save_json, xywh_to_xyxy

CRITICAL_CLASS = "NO-Hardhat"


def norm_xyxy_from_pixels(xyxy, shape):
    h, w = shape[:2]
    return [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h]


def collect_predictions(weights, imgsz, conf):
    model = YOLO(weights)
    items = []
    for img_path in list_images(data_dir() / "images" / "test"):
        r = model(str(img_path), imgsz=imgsz, conf=conf, iou=0.7, verbose=False)[0]
        preds = []
        for cls, c, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
            preds.append({"cls": int(cls), "conf": float(c), "box": norm_xyxy_from_pixels([float(v) for v in xyxy], r.orig_shape)})
        gts = [{"cls": int(b[0]), "box": list(xywh_to_xyxy(b))} for b in read_labels(label_for_image(img_path))]
        items.append({"image": img_path, "preds": preds, "gts": gts})
    return model, items


def threshold_sweep(items, iou_thr=0.5):
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        tp, fp, fn = 0, 0, 0
        for it in items:
            preds = [p for p in it["preds"] if p["conf"] >= t]
            gts = it["gts"]
            pairs, up, ug = greedy_match(preds, gts, iou_thr)
            tp += len(pairs)
            fp += len(preds) - len(pairs)
            fn += len(gts) - len(pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": round(float(t), 2), "precision": precision, "recall": recall, "f1": f1})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--conf", type=float, default=0.01)
    args = parser.parse_args()
    settings.update({"datasets_dir": str(repo_root())})
    _, items = collect_predictions(args.weights, args.imgsz, args.conf)
    rows = threshold_sweep(items)
    best = max(rows, key=lambda r: r["f1"])
    figures = repo_root() / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [r["threshold"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ts, [r["precision"] for r in rows], marker="o", label="precision")
    plt.plot(ts, [r["recall"] for r in rows], marker="s", label="recall")
    plt.plot(ts, [r["f1"] for r in rows], marker="^", label="F1")
    plt.axvline(best["threshold"], ls="--", c="gray", label=f"best F1 @ {best['threshold']:.2f}")
    plt.xlabel("Порог уверенности")
    plt.ylabel("Значение")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"Зависимость от порога уверенности ({args.tag})")
    plt.tight_layout()
    plt.savefig(figures / f"conf_threshold_{args.tag}.png", dpi=130)
    save_json({"rows": rows, "best_f1": best}, repo_root() / "reports" / f"conf_sweep_{args.tag}.json")
    print(f"best F1={best['f1']:.3f} at conf={best['threshold']}")


if __name__ == "__main__":
    main()
