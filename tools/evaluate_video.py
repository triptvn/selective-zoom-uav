"""Z_S (selective zoom) on video with one detector call per frame.

The zoom window of frame t is chosen from the detections of frame t-1, so it can be batched with the
full view of frame t. Compared on consecutive UAVDT test frames:
  single     full view only
  zs         window chosen from the current frame (two calls)
  zs_prev    window chosen from the previous frame, batched with the full view (one call, batch 2)

  python tools/evaluate_video.py --weights yfzt_uavdt.pt --root /path/to/UAVDT-DET-ig --seqs 20 --frames 150
(the paper uses all test sequences: 5,932 frames)
"""
import argparse
import json
import time
from pathlib import Path

import _path  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--root", required=True, help="UAVDT-DET-ig folder")
ap.add_argument("--split", default="test")
ap.add_argument("--seqs", type=int, default=20)
ap.add_argument("--frames", type=int, default=150, help="first N consecutive frames of each sequence")
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--lat-n", type=int, default=200)
ap.add_argument("--out", default=None)
args = ap.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from tinyuav import coco_eval, zoom  # noqa: E402
from tinyuav.data import UAVDT_NAMES  # noqa: E402

root = Path(args.root)
seqs = {}
for q in sorted(root.glob(f"{args.split}/**/*.jpg")):
    seqs.setdefault(q.stem.split("_")[0], []).append(q)
frames = []
for s in sorted(seqs)[:args.seqs]:
    frames += sorted(seqs[s], key=lambda q: int(q.stem.split("img")[-1]))[:args.frames]

images, anns = [], []
for i, q in enumerate(frames, 1):
    h, w = cv2.imread(str(q)).shape[:2]
    images.append({"id": i, "file_name": str(q), "width": w, "height": h})
    s = coco_eval.BUCKET_FRAME / max(w, h)
    lab = root / args.split / "gt_whole" / (q.stem + ".txt")
    for line in (lab.read_text().splitlines() if lab.exists() else []):
        v = [float(x) for x in line.split(",")[:5]]
        if len(v) == 5 and v[3] > 0 and v[4] > 0:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(v[0]) - 1, "bbox": v[1:5],
                         "area": v[3] * s * v[4] * s, "iscrowd": 0})
gt = {"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for i, n in enumerate(UAVDT_NAMES)]}

model = YOLO(args.weights)
cfg = zoom.ZoomConfig(imgsz=args.imgsz)
modes = {"single": [], "zs": [], "zs_prev": []}
calls = dict.fromkeys(modes, 0)
prev, prev_seq = None, None
for im, q in zip(images, frames):
    img = cv2.imread(im["file_name"])
    h, w = img.shape[:2]
    g = zoom.predict(model, img, args.imgsz)
    seq = q.stem.split("_")[0]
    now = zoom.select_windows(g[0], g[1], w, h, cfg)
    before = zoom.select_windows(prev[0], prev[1], w, h, cfg) if prev is not None and seq == prev_seq else []
    for m, wins, n_calls in (("single", [], 1), ("zs", now, 1 + len(now)), ("zs_prev", before, 1)):
        modes[m] += coco_eval.to_coco(im["id"], zoom.fuse(g, zoom.run_windows(model, img, wins, args.imgsz)))
        calls[m] += n_calls
    prev, prev_seq = g, seq

res = {"weights": args.weights, "n_frames": len(images), "modes": {}}
for m, dets in modes.items():
    r = coco_eval.evaluate(gt, dets)
    res["modes"][m] = {"mAP50": r["mAP50"], "mAP50_95": r["mAP50_95"], "calls_per_frame": calls[m] / len(images)}
    print(f"{m:8s} AP50 {r['mAP50']:.2f}  calls/frame {calls[m] / len(images):.2f}")

# latency per frame: full view, full view + window (two calls), full view and window in one batch
cuda = torch.cuda.is_available()
sync = torch.cuda.synchronize if cuda else (lambda: None)
kw = dict(imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=500, verbose=False)
lat = {"single": [], "z1_two_calls": [], "z1_prev_batch2": []}
prev = None
for n, im in enumerate(images[:args.lat_n + 10]):
    img = cv2.imread(im["file_name"])
    h, w = img.shape[:2]
    wins = zoom.select_windows(prev[0], prev[1], w, h, cfg) if prev is not None else []
    x0, y0, x1, y1 = wins[0] if wins else (0, 0, w * 0.6, h * 0.6)
    crop = img[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
    sync(); a = time.perf_counter()
    r = model.predict(img, **kw)[0]
    sync(); b = time.perf_counter()
    model.predict(crop, **kw)
    sync(); c = time.perf_counter()
    model.predict([img, crop], **kw)
    sync(); d = time.perf_counter()
    bx = r.boxes
    prev = (bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy()) if bx is not None and len(bx) else (np.zeros((0, 4)), np.zeros(0))
    if n >= 10:  # warm-up
        lat["single"].append((b - a) * 1e3)
        lat["z1_two_calls"].append((c - a) * 1e3)
        lat["z1_prev_batch2"].append((d - c) * 1e3)
res["latency_ms"] = {k: float(np.mean(v)) for k, v in lat.items()}
print("latency (ms/frame):", {k: round(v, 1) for k, v in res["latency_ms"].items()})
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
