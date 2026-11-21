import argparse
import json
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression

from common import data_dir, list_images, repo_root, save_json

CONF = 0.25
IOU_NMS = 0.7
WARMUP = 10


def letterbox_tensor(img, imgsz, device):
    lb = LetterBox(imgsz, auto=False, center=True)
    im = lb(image=img)
    im = im.transpose(2, 0, 1)[::-1]
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im).unsqueeze(0).float().to(device) / 255.0


def timed(fn, n):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ts), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    images = list_images(data_dir() / "images" / "test")[: args.n]
    raw = [cv2.imread(str(p)) for p in images]

    net = YOLO(args.weights).model.cpu().eval()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    diffs, det_stats = [], []
    for img in raw:
        t = letterbox_tensor(img, args.imgsz, "cpu")
        with torch.no_grad():
            p_torch = net(t)
        if isinstance(p_torch, tuple):
            p_torch = p_torch[0]
        p_ort = torch.from_numpy(sess.run(None, {input_name: t.numpy()})[0])
        diffs.append(float((p_torch - p_ort).abs().max()))
        d_t = non_max_suppression(p_torch, conf_thres=CONF, iou_thres=IOU_NMS)[0]
        d_o = non_max_suppression(p_ort, conf_thres=CONF, iou_thres=IOU_NMS)[0]
        det_stats.append((len(d_t), len(d_o)))

    def torch_pipe(img):
        t = letterbox_tensor(img, args.imgsz, "cpu")
        with torch.no_grad():
            p = net(t)
        if isinstance(p, tuple):
            p = p[0]
        non_max_suppression(p, conf_thres=CONF, iou_thres=IOU_NMS)

    def ort_pipe(img):
        t = letterbox_tensor(img, args.imgsz, "cpu")
        p = sess.run(None, {input_name: t.numpy()})[0]
        non_max_suppression(torch.from_numpy(p), conf_thres=CONF, iou_thres=IOU_NMS)

    report = {
        "weights": args.weights,
        "onnx": args.onnx,
        "imgsz": args.imgsz,
        "n_images": len(raw),
        "device": "CPU",
        "raw_output_max_abs_diff": round(max(diffs), 6),
        "raw_output_mean_abs_diff": round(float(np.mean(diffs)), 6),
        "nms_det_count_pytorch": [a for a, _ in det_stats],
        "nms_det_count_onnx": [b for _, b in det_stats],
        "det_count_mismatches": sum(1 for a, b in det_stats if a != b),
        "latency_ms_median": {
            "pytorch_cpu_full": timed(lambda i=0: torch_pipe(raw[i]), len(raw)),
            "onnx_cpu_full": timed(lambda i=0: ort_pipe(raw[i]), len(raw)),
        },
    }
    save_json(report, repo_root() / "reports" / "onnx_compare.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
