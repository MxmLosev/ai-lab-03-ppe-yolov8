import sys
from pathlib import Path

import cv2
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import CLASS_NAMES, group_key, iou_xyxy, label_for_image, list_images, md5, read_labels  # noqa: E402

SPLITS = ["train", "val", "test"]


@pytest.fixture(scope="session", autouse=True)
def _ultralytics_settings():
    from ultralytics import settings as usettings

    usettings.update({"datasets_dir": str(ROOT)})


@pytest.fixture(scope="session")
def ppe_yaml():
    return yaml.safe_load((ROOT / "configs" / "ppe.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def split_images(ppe_yaml):
    base = ROOT / ppe_yaml["path"]
    return {s: list_images(base / ppe_yaml[s]) for s in SPLITS}


def test_yaml_image_paths_exist_and_labels_readable(ppe_yaml, split_images):
    base = ROOT / ppe_yaml["path"]
    for s in SPLITS:
        assert (base / ppe_yaml[s]).is_dir()
    for s in SPLITS:
        assert len(split_images[s]) > 0
        for img in split_images[s]:
            lbl = label_for_image(img)
            assert lbl.exists(), f"missing label {lbl}"
            for box in read_labels(lbl):
                assert len(box) == 5


def test_label_schema_valid(ppe_yaml, split_images):
    nc = len(ppe_yaml["names"])
    for s in SPLITS:
        for img in split_images[s]:
            for cls, x, y, w, h in read_labels(label_for_image(img)):
                assert 0 <= cls < nc, f"class {cls} out of range in {img.name}"
                assert all(0.0 <= v <= 1.0 for v in (x, y, w, h)), f"coord out of [0,1] in {img.name}"
                assert x - w / 2 >= -0.001 and x + w / 2 <= 1.001, f"box x exceeds image in {img.name}"
                assert y - h / 2 >= -0.001 and y + h / 2 <= 1.001, f"box y exceeds image in {img.name}"


def test_no_split_leaks(split_images):
    names = {}
    hashes = {}
    groups = {}
    for s in SPLITS:
        for img in split_images[s]:
            assert img.name not in names, f"same name in {names.get(img.name)} and {s}: {img.name}"
            names[img.name] = s
            h = md5(img)
            assert h not in hashes, f"same content in {hashes.get(h)} and {s}: {img.name}"
            hashes[h] = s
            g = group_key(img.stem)
            assert groups.get(g, s) == s, f"group {g} in {groups.get(g)} and {s}"
            groups[g] = s


def test_visualizer_output_size(tmp_path):
    import numpy as np

    import error_galleries as eg

    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    path = tmp_path / "fake.jpg"
    cv2.imwrite(str(path), img)
    gts = [{"cls": 0, "box": [0.2, 0.2, 0.4, 0.4]}]
    preds = [{"cls": 0, "conf": 0.9, "box": [0.25, 0.25, 0.4, 0.4]}]
    cell = eg.make_cell(path, gts, preds, ["test line"])
    out = eg.render_gallery([cell] * (eg.COLS + 1), tmp_path / "gallery.png")
    assert cell.shape[1] == eg.CELL_W
    assert out.shape[1] == eg.COLS * eg.CELL_W
    assert out.shape[0] == 2 * cell.shape[0]


def test_model_output_schema(ppe_yaml, split_images):
    from ultralytics import YOLO

    weights = ROOT / "models" / "ppe_best.pt"
    assert weights.exists()
    nc = len(ppe_yaml["names"])
    model = YOLO(str(weights))
    img = split_images["test"][0]
    r = model(str(img), verbose=False)[0]
    h, w = r.orig_shape
    assert len(r.boxes.cls) >= 1, "no detections on control image"
    for cls, conf, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
        assert 0 <= int(cls) < nc
        assert 0.0 <= float(conf) <= 1.0
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        assert -1 <= x1 < x2 <= w + 1
        assert -1 <= y1 < y2 <= h + 1
