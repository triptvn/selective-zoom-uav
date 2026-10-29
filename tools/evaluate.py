"""Image-wise evaluation of inference modes (the protocol behind all accuracy tables).

Modes (comma-separated in --modes):
  single   one pass at --imgsz (use --imgsz 1088/1216/... for whole-image upscaling)
  zs       Z_S, one zoom window (K=1)            [old name: z1]
  zs_k2    Z_S, up to two windows                [old name: z1k2]
  sahi     2x2 tiles with 20% overlap + full view (five calls)
  random / center / oracle   window-selection study: same number of windows as Z_S per image,
           placed at random / in the centre, or chosen from ground-truth boxes

  python tools/evaluate.py --weights yfzt_s0.pt --data data/VisDrone/data.yaml --split test --modes single,zs,sahi
Writes one JSON with COCO AP50, AP50:95, P, R, size buckets, per-class AP and calls per image.
"""
import argparse
import json
import time
from pathlib import Path

import _path  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help=".pt / exported model (.engine, *_ncnn_model) / D-FINE .pth")
ap.add_argument("--data", required=True, help="data.yaml of a prepared dataset")
ap.add_argument("--split", default="test")
ap.add_argument("--modes", default="single,zs,sahi")
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--limit", type=int, default=0, help="first N images only (quick check)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tinyuav import coco_eval, zoom  # noqa: E402
from tinyuav.data import load_names  # noqa: E402
from tinyuav.dfine import load_detector  # noqa: E402

root = Path(args.data).parent
img_dir = root / "images" / args.split
gt = coco_eval.build_ground_truth(img_dir, root / "labels" / args.split, load_names(args.data))
if args.limit:
    gt["images"] = gt["images"][:args.limit]
    ids = {im["id"] for im in gt["images"]}
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in ids]
gt_boxes = {}
for a in gt["annotations"]:
    x, y, w, h = a["bbox"]
    gt_boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])

model = load_detector(args.weights)
cfg = zoom.ZoomConfig(imgsz=args.imgsz)
modes = list(dict.fromkeys(zoom.canonical_mode(m) for m in args.modes.split(",")))
dets = {m: [] for m in modes}
calls = {m: 0 for m in modes}
rng = np.random.default_rng(0)


def study_windows(kind, n, w, h, image_id):
    ww, wh = cfg.window * w, cfg.window * h
    if kind == "oracle":
        b = np.array(gt_boxes.get(image_id, []), np.float32).reshape(-1, 4)
        return zoom.select_windows(b, np.ones(len(b), np.float32), w, h, cfg, tau=0.0)
    if kind == "random":
        return [(x, y, x + ww, y + wh) for x, y in zip(rng.uniform(0, w - ww, n), rng.uniform(0, h - wh, n))]
    return [((w - ww) / 2, (h - wh) / 2, (w + ww) / 2, (h + wh) / 2)][:n]


t0 = time.time()
for i, im in enumerate(gt["images"]):
    img = cv2.imread(str(img_dir / im["file_name"]))
    h, w = img.shape[:2]
    g = zoom.predict(model, img, args.imgsz)
    zs = zoom.select_windows(g[0], g[1], w, h, cfg, k=2 if "zs_k2" in modes else 1)
    zs_dets = zoom.run_windows(model, img, zs, args.imgsz)
    for m in modes:
        if m == "single":
            wd = []
        elif m == "zs":
            wd = zs_dets[:1]
        elif m == "zs_k2":
            wd = zs_dets
        elif m == "sahi":
            wd = zoom.run_windows(model, img, zoom.tile_windows(w, h), args.imgsz)
        else:
            wd = zoom.run_windows(model, img, study_windows(m, min(len(zs), 1), w, h, im["id"]), args.imgsz)
        calls[m] += 1 + len(wd)
        dets[m] += coco_eval.to_coco(im["id"], zoom.fuse(g, wd))
    if i % 200 == 0:
        print(f"{i}/{len(gt['images'])} images, {time.time() - t0:.0f}s", flush=True)

res = {"weights": str(args.weights), "data": str(args.data), "split": args.split, "imgsz": args.imgsz,
       "n_images": len(gt["images"]), "modes": {}}
print(f"\n{'mode':<8}{'calls':>7}{'AP50':>8}{'AP50:95':>9}{'tiny':>7}{'P':>7}{'R':>7}")
for m in modes:
    r = {**coco_eval.evaluate(gt, dets[m]), "calls_per_img": calls[m] / len(gt["images"])}
    res["modes"][m] = r
    print(f"{m:<8}{r['calls_per_img']:7.2f}{r['mAP50']:8.2f}{r['mAP50_95']:9.2f}{r.get('tiny_AP50', 0):7.2f}{r['P']:7.1f}{r['R']:7.1f}")
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
