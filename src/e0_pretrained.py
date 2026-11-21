import random

import cv2
from ultralytics import YOLO, settings

from common import CLASS_NAMES, data_dir, list_images, repo_root, save_json

N_IMAGES = 10
SEED = 42
CONF = 0.25


def main():
    settings.update({"datasets_dir": str(repo_root())})
    random.seed(SEED)
    test_dir = data_dir() / "images" / "test"
    sample = random.sample(list_images(test_dir), N_IMAGES)
    model = YOLO("yolov8n.pt")
    out_dir = repo_root() / "reports" / "e0"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for img_path in sample:
        result = model(str(img_path), conf=CONF, verbose=False)[0]
        detections = []
        for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            detections.append({"class": model.names[int(cls)], "conf": round(float(conf), 3), "xyxy": [round(float(v), 1) for v in xyxy]})
        cv2.imwrite(str(out_dir / img_path.name), result.plot())
        records.append({"image": img_path.name, "detections": detections})
    save_json({"weights": "yolov8n.pt", "source": "COCO pretrained", "conf": CONF, "images": records}, out_dir / "e0_predictions.json")
    total = sum(len(r["detections"]) for r in records)
    by_class = {}
    for r in records:
        for d in r["detections"]:
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
    print(f"E0: {total} detections on {N_IMAGES} images: {by_class}")
    print(f"saved annotated images and e0_predictions.json in {out_dir}")


if __name__ == "__main__":
    main()
