import argparse
import json
import statistics
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO, settings
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.torch_utils import select_device

from common import data_dir, list_images, repo_root, save_json

WARMUP = 30
N_TIMED = 150
CONF = 0.25
IOU_NMS = 0.7


def percentile(values, q):
    return float(np.percentile(values, q))


def summarize(times_ms):
    return {
        "mean_ms": round(statistics.fmean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(percentile(times_ms, 95), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    settings.update({"datasets_dir": str(repo_root())})
    device = select_device("")
    model = YOLO(args.weights)
    net = model.model.to(device).eval()
    names = model.names
    nc = len(names)

    images = list_images(data_dir() / "images" / "test")[:10]
    letterbox = LetterBox(args.imgsz, auto=False, center=True)

    raw = [cv2.imread(str(p)) for p in images]
    lb_shapes = [letterbox(image=img).shape for img in raw]

    def decode():
        return cv2.imread(str(images[0]))

    def preprocess(idx):
        im = letterbox(image=raw[idx])
        im = im.transpose(2, 0, 1)[::-1]
        im = np.ascontiguousarray(im)
        t = torch.from_numpy(im).to(device).float() / 255.0
        return t.unsqueeze(0)

    def infer(t):
        with torch.no_grad():
            return net(t)

    def postprocess(pred, idx):
        det = non_max_suppression(pred, conf_thres=CONF, iou_thres=IOU_NMS, max_det=300)[0]
        if len(det):
            scale_boxes(lb_shapes[idx], det[:, :4], raw[idx].shape)
        return det

    def full(idx):
        t0 = time.perf_counter()
        img = cv2.imread(str(images[idx]))
        t1 = time.perf_counter()
        lb_img = letterbox(image=img)
        im = lb_img.transpose(2, 0, 1)[::-1]
        im = np.ascontiguousarray(im)
        t = torch.from_numpy(im).to(device).float().unsqueeze(0) / 255.0
        t2 = time.perf_counter()
        with torch.no_grad():
            pred = net(t)
        if device.type != "cpu":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        det = non_max_suppression(pred, conf_thres=CONF, iou_thres=IOU_NMS, max_det=300)[0]
        if len(det):
            scale_boxes(lb_img.shape, det[:, :4], img.shape)
        t4 = time.perf_counter()
        return [t0, t1, t2, t3, t4]

    idx_cycle = [i % len(images) for i in range(N_TIMED)]
    for i in range(WARMUP):
        full(i % len(images))

    t_decode, t_pre, t_infer, t_post, t_full = [], [], [], [], []
    for idx in idx_cycle:
        t0, t1, t2, t3, t4 = full(idx)
        t_decode.append((t1 - t0) * 1000)
        t_pre.append((t2 - t1) * 1000)
        t_infer.append((t3 - t2) * 1000)
        t_post.append((t4 - t3) * 1000)
        t_full.append((t4 - t0) * 1000)

    torch.cuda.reset_peak_memory_stats()
    _ = preprocess(0)
    infer(preprocess(0))
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1 << 20) if device.type != "cpu" else 0

    single = []
    for idx in idx_cycle[:50]:
        t = preprocess(idx)
        single.append(t)

    report = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "device": str(device),
        "precision": "fp32",
        "conf": CONF,
        "nms_iou": IOU_NMS,
        "n_images": len(images),
        "n_timed": N_TIMED,
        "warmup": WARMUP,
        "stages": {
            "decode": summarize(t_decode),
            "preprocess_letterbox_to_gpu": summarize(t_pre),
            "inference": summarize(t_infer),
            "postprocess_nms_scale": summarize(t_post),
            "full_call": summarize(t_full),
        },
        "throughput_fps_from_median": round(1000 / statistics.median(t_full), 2),
        "gpu_peak_memory_mb": round(peak_mem, 1),
    }
    save_json(report, repo_root() / "reports" / f"latency_{args.tag}.json")
    print(json.dumps(report["stages"], indent=2))
    print(f"throughput ~{report['throughput_fps_from_median']} fps, gpu peak {report['gpu_peak_memory_mb']} MB")


if __name__ == "__main__":
    main()
