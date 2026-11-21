import random
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_NAMES, data_dir, label_for_image, list_images, read_labels, repo_root

SEED = 42
SHEET_W = 5

COLORS = [
    (56, 56, 255),
    (151, 157, 255),
    (56, 255, 56),
    (255, 56, 157),
    (255, 157, 56),
    (56, 255, 217),
    (157, 56, 255),
    (56, 217, 255),
    (255, 56, 56),
    (255, 255, 56),
]


def draw_gt(img_path: Path, cell_w=360):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    scale = cell_w / w
    img = cv2.resize(img, (cell_w, int(h * scale)))
    ih, iw = img.shape[:2]
    for cls, x, y, bw, bh in read_labels(label_for_image(img_path)):
        x1, y1 = int((x - bw / 2) * iw), int((y - bh / 2) * ih)
        x2, y2 = int((x + bw / 2) * iw), int((y + bh / 2) * ih)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        ty = max(12, y1 - 4)
        cv2.putText(img, CLASS_NAMES[cls], (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def make_sheet(images, out_path: Path, cols=SHEET_W):
    cells = [draw_gt(p) for p in images]
    cell_h = max(c.shape[0] for c in cells)
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * cells[0].shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        y0, x0 = r * cell_h, col * c.shape[1]
        sheet[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]] = c
    cv2.imwrite(str(out_path), sheet)
    print(f"saved {out_path} ({len(cells)} cells)")


def main():
    rng = random.Random(SEED)
    test_dir = data_dir() / "images" / "test"
    figures = repo_root() / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    images = list_images(test_dir)
    sample30 = rng.sample(images, 30)
    make_sheet(sample30, figures / "contact_sheet_30.png")

    scored = []
    for p in images:
        boxes = read_labels(label_for_image(p))
        if not boxes:
            continue
        import cv2 as cv

        h, w = cv.imread(str(p)).shape[:2]
        smallest = min(b[3] * b[4] for b in boxes)
        scored.append((smallest, len(boxes), p))
    scored.sort(key=lambda t: (t[0], -t[1]))
    small10 = [p for _, _, p in scored[:10]]
    make_sheet(small10, figures / "contact_sheet_small10.png")


if __name__ == "__main__":
    main()
