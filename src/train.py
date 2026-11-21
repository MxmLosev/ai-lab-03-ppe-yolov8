import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    args = parser.parse_args()
    params = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    settings.update({"datasets_dir": str(ROOT)})
    model = YOLO(args.weights)
    model.train(
        data="configs/ppe.yaml",
        project="runs/ppe",
        exist_ok=True,
        **params,
    )


if __name__ == "__main__":
    main()
