import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import settings

from common import CLASS_NAMES, data_dir, greedy_match, iou_xyxy, repo_root, save_json
from sweep_confidence import collect_predictions

CELL_W = 420
CAPTION_H = 64
COLS = 4
MAX_CELLS = 8
CRITICAL = "NO-Hardhat"

GT_COLOR = (80, 220, 80)
PRED_COLOR = (60, 60, 230)
MATCH_COLOR = (60, 200, 230)


def draw_box(img, box, color, label):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    ty = max(14, y1 - 5)
    cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def make_cell(img_path, gts, preds, lines, crop=None):
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((480, CELL_W, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if crop is not None:
        cx1, cy1, cx2, cy2 = crop
        pad_w, pad_h = (cx2 - cx1) * 1.2, (cy2 - cy1) * 1.2
        x1, y1 = int(max(0, (cx1 + cx2) / 2 - pad_w / 2) * w), int(max(0, (cy1 + cy2) / 2 - pad_h / 2) * h)
        x2, y2 = int(min(1, (cx1 + cx2) / 2 + pad_w / 2) * w), int(min(1, (cy1 + cy2) / 2 + pad_h / 2) * h)
        x2, y2 = max(x2, x1 + 40), max(y2, y1 + 40)
        img = img[y1:y2, x1:x2]
    h, w = img.shape[:2]
    scale = CELL_W / w
    img = cv2.resize(img, (CELL_W, int(h * scale)))
    ih, iw = img.shape[:2]
    for g in gts:
        draw_box(img, g["box"], GT_COLOR, f"G:{CLASS_NAMES[g['cls']]}")
    for p in preds:
        draw_box(img, p["box"], PRED_COLOR, f"P:{CLASS_NAMES[p['cls']]} {p['conf']:.2f}")
    cell = np.zeros((ih + CAPTION_H, CELL_W, 3), dtype=np.uint8)
    cell[:ih] = img
    name = Path(img_path).name
    name = name.split(".rf.")[0][:52]
    for i, line in enumerate([name] + lines):
        y = ih + 18 + 15 * i
        cv2.putText(cell, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def render_gallery(cells, out_path):
    cell_h = max(c.shape[0] for c in cells)
    rows = math.ceil(len(cells) / COLS)
    sheet = np.zeros((rows * cell_h, COLS * CELL_W, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, COLS)
        sheet[r * cell_h : r * cell_h + c.shape[0], col * CELL_W : col * CELL_W + c.shape[1]] = c
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return sheet


def analyze(items, conf_thr):
    fns, fps, low_iou, confusions, small, gt_err = [], [], [], [], [], []
    for it in items:
        preds = [p for p in it["preds"] if p["conf"] >= conf_thr]
        gts = it["gts"]
        pairs, up, ug = greedy_match(preds, gts, 0.5)
        agnostic_pairs, _, _ = greedy_match(preds, gts, 0.5, class_agnostic=True)
        img = it["image"]
        top_fp = max((preds[pi] for pi in range(len(preds)) if pi not in up), key=lambda p: p["conf"], default=None)
        if top_fp is not None and top_fp["conf"] >= 0.5:
            fps.append((top_fp["conf"], img, top_fp, [f"FP {CLASS_NAMES[top_fp['cls']]} conf={top_fp['conf']:.2f}"]))
        missed = [gts[gi] for gi in range(len(gts)) if gi not in ug and gts[gi]["cls"] == CLASS_NAMES.index(CRITICAL)]
        if missed:
            fns.append((len(missed), img, missed, [f"miss {CRITICAL} x{len(missed)}"]))
        worst = min((p for p in pairs if p[2] < 0.7), key=lambda p: p[2], default=None)
        if worst is not None:
            pi, gi, i = worst
            low_iou.append((i, img, [preds[pi]], [f"IoU={i:.2f}", f"P:{CLASS_NAMES[preds[pi]['cls']]} vs G:{CLASS_NAMES[gts[gi]['cls']]}"]))
        for pi, gi, i in agnostic_pairs:
            if preds[pi]["cls"] != gts[gi]["cls"]:
                confusions.append((preds[pi]["conf"], img, preds[pi], gts[gi], [f"G:{CLASS_NAMES[gts[gi]['cls']]} -> P:{CLASS_NAMES[preds[pi]['cls']]} {preds[pi]['conf']:.2f}"]))
                break
        small_gts = [g for g in gts if g["box"][2] * g["box"][3] < 0.002]
        if small_gts:
            g = min(small_gts, key=lambda g: g["box"][2] * g["box"][3])
            small.append((g["box"][2] * g["box"][3], img, g, [f"small GT area={g['box'][2]*g['box'][3]:.4f}"]))
        for a in range(len(gts)):
            for b in range(a + 1, len(gts)):
                if gts[a]["cls"] == gts[b]["cls"] and iou_xyxy(gts[a]["box"], gts[b]["box"]) >= 0.85:
                    gt_err.append((1.0, img, gts[a], [f"duplicate GT {CLASS_NAMES[gts[a]['cls']]} IoU>=0.85"]))
                    break
            else:
                continue
            break
        if not gt_err or gt_err[-1][1] != img:
            if gts and (gts[0]["box"][2] * gts[0]["box"][3]) > 0.9:
                gt_err.append((0.9, img, gts[0], [f"huge GT box area>0.9 {CLASS_NAMES[gts[0]['cls']]}"]))
    return fps, fns, low_iou, confusions, small, gt_err


def build_galleries(items, conf_thr, out_dir):
    fps, fns, low_iou, confusions, small, gt_err = analyze(items, conf_thr)
    galleries = {}
    fps.sort(key=lambda t: -t[0])
    cells = [make_cell(img, it["gts"], [p], lines) for _, img, p, lines in fps[:MAX_CELLS] for it in [next(x for x in items if x["image"] == img)]]
    galleries["1_high_conf_false_positives"] = cells
    fns.sort(key=lambda t: (-t[0],))
    cells = [make_cell(img, [g for g in it["gts"] if g["cls"] == CLASS_NAMES.index(CRITICAL)], [p for p in it["preds"] if p["conf"] >= conf_thr and p["cls"] == CLASS_NAMES.index(CRITICAL)], lines) for _, img, g, lines in fns[:MAX_CELLS] for it in [next(x for x in items if x["image"] == img)]]
    galleries["2_missed_critical"] = cells
    low_iou.sort(key=lambda t: t[0])
    cells = [make_cell(img, [g for g in it["gts"]], [p for p in it["preds"] if p["conf"] >= conf_thr], lines) for _, img, p, lines in low_iou[:MAX_CELLS] for it in [next(x for x in items if x["image"] == img)]]
    galleries["3_low_iou_matches"] = cells
    confusions.sort(key=lambda t: -t[0])
    cells = [
        make_cell(img, [g] if not isinstance(g, list) else g, [p], lines)
        for _, img, p, g, lines in confusions[:MAX_CELLS]
        for it in [next(x for x in items if x["image"] == img)]
    ]
    galleries["4_class_confusion"] = cells
    small.sort(key=lambda t: t[0])
    cells = []
    for _, img, g, lines in small[:MAX_CELLS]:
        it = next(x for x in items if x["image"] == img)
        preds_near = [p for p in it["preds"] if p["conf"] >= conf_thr and iou_xyxy(p["box"], g["box"]) > 0.1]
        cells.append(make_cell(img, [g], preds_near, lines, crop=g["box"]))
    galleries["5_small_and_occluded"] = cells
    seen = set()
    cells = []
    for _, img, g, lines in gt_err[:MAX_CELLS]:
        if img in seen:
            continue
        seen.add(img)
        it = next(x for x in items if x["image"] == img)
        cells.append(make_cell(img, it["gts"], [p for p in it["preds"] if p["conf"] >= conf_thr], lines))
    galleries["6_suspected_gt_errors"] = cells
    titles = {
        "1_high_conf_false_positives": "False positives with high confidence",
        "2_missed_critical": f"Missed objects of critical class ({CRITICAL})",
        "3_low_iou_matches": "Matched boxes with low IoU",
        "4_class_confusion": "Confusion between similar classes",
        "5_small_and_occluded": "Small and occluded objects",
        "6_suspected_gt_errors": "Suspected ground truth annotation errors",
    }
    sizes = {}
    for name, cells in galleries.items():
        sheet = render_gallery(cells, out_dir / f"{name}.png")
        sizes[name] = {"cells": len(cells), "size_px": [int(sheet.shape[1]), int(sheet.shape[0])], "title": titles[name]}
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    settings.update({"datasets_dir": str(repo_root())})
    _, items = collect_predictions(args.weights, args.imgsz, 0.01)
    out_dir = repo_root() / "galleries"
    sizes = build_galleries(items, args.conf, out_dir)
    save_json({"weights": args.weights, "conf": args.conf, "galleries": sizes}, out_dir / "index_data.json")
    print("galleries:", {k: v["cells"] for k, v in sizes.items()})


if __name__ == "__main__":
    main()
