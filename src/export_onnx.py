import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False, device="cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, args.out)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
