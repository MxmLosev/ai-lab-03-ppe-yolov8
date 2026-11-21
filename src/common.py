import hashlib
import json
import re
from pathlib import Path

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGET_TOTAL = 800
MIN_INSTANCES = {"train": 25, "val": 10, "test": 10}
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return repo_root() / "data"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_key(stem: str) -> str:
    base = re.split(r"\.rf\.|_jpg|_jpeg|_png|_mp4|_mov", stem)[0]
    base = base.rstrip("-_")
    stripped = re.sub(r"[-_]\d{1,5}$", "", base)
    if stripped:
        base = stripped
    return base.lower()


def read_labels(path: Path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:])
        boxes.append((cls, x, y, w, h))
    return boxes


def write_labels(path: Path, boxes):
    lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def xywh_to_xyxy(box):
    _, x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def sanitize_boxes(boxes, min_wh=0.001):
    out, n_clipped, n_dropped = [], 0, 0
    for cls, x, y, w, h in boxes:
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        if x2 - x1 < min_wh or y2 - y1 < min_wh:
            n_dropped += 1
            continue
        if (x1, y1, x2, y2) != (x - w / 2, y - h / 2, x + w / 2, y + h / 2):
            n_clipped += 1
        out.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out, n_clipped, n_dropped


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, iou_thr=0.5, class_agnostic=False):
    cand = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if not class_agnostic and p[0] != g[0]:
                continue
            i = iou_xyxy(p[1:5], g[1:5])
            if i >= iou_thr:
                cand.append((i, pi, gi))
    cand.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for i, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, i))
    return pairs, used_p, used_g


def list_images(split_dir: Path):
    return sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_for_image(img_path: Path) -> Path:
    return img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
