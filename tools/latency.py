"""Full-pipeline latency at batch 1: pre-processing, detector calls, window selection and fusion.
Image decoding is excluded. Each config is <mode>@<imgsz> with mode in {single, zs, zs_k2, sahi}
(zs = Z_S, the selective zoom; the old names z1 and z1k2 are accepted).

  # T4, PyTorch FP32 (zoom windows batched into one call)
  python tools/latency.py --weights yfzt.pt --images data/VisDrone/images/test --configs single@896,single@1216,zs@896,sahi@896 --batched
  # T4, TensorRT FP16: engines from tools/export_trt.py
  python tools/latency.py --weights yfzt.pt --engines engines.json --images ... --configs single@896,zs@896 --batched
  # Raspberry Pi 5, NCNN (yolo export format=ncnn imgsz=896), sequential windows
  python tools/latency.py --weights yfzt_ncnn_model --images ... --n 100 --warmup 10 --device cpu --configs single@896,zs@896
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import _path  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--configs", default="single@896,zs@896,zs_k2@896,sahi@896")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--batched", action="store_true", help="run all windows of an image in one detector call")
ap.add_argument("--half", action="store_true")
ap.add_argument("--device", default=None)
ap.add_argument("--engines", default=None, help='JSON {"896": "yfzt_896.engine", ...}')
ap.add_argument("--out", default=None)
args = ap.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from tinyuav import zoom  # noqa: E402
from tinyuav.dfine import load_detector  # noqa: E402

cuda = torch.cuda.is_available() and args.device != "cpu"
device = args.device or ("0" if cuda else "cpu")
model = load_detector(args.weights)
engines = {int(k): YOLO(v, task="detect") for k, v in json.loads(Path(args.engines).read_text()).items()} if args.engines else {}
kw = {"device": device, **({"half": True} if args.half else {})}
imgs = [cv2.imread(str(p)) for p in sorted(Path(args.images).glob("*.jpg"))[:args.n + args.warmup]]


def sync():
    if cuda:
        torch.cuda.synchronize()


def run(cfg, img):
    mode, sz = cfg.split("@")
    sz = int(sz)
    if args.engines and sz not in engines:
        raise RuntimeError(f"no engine for {sz}")
    m = engines.get(sz, model)
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    g = zoom.predict(m, img, sz, **kw)
    if mode == "single":
        sync()
        return time.perf_counter() - t0, 0
    zc = zoom.ZoomConfig(imgsz=sz, k=2 if mode == "zs_k2" else 1)
    wins = zoom.tile_windows(w, h) if mode == "sahi" else zoom.select_windows(g[0], g[1], w, h, zc)
    wd = zoom.run_windows(m, img, wins, sz, batched=args.batched, **kw)
    zoom.fuse(g, wd)
    sync()
    return time.perf_counter() - t0, len(wins)


res = {"weights": args.weights, "device": torch.cuda.get_device_name(0) if cuda else "cpu", "half": args.half,
       "batched": args.batched, "engines": sorted(engines), "n": len(imgs) - args.warmup, "configs": {}, "errors": {}}
for cfg in args.configs.split(","):
    m, sz = cfg.split("@")
    cfg = f"{zoom.canonical_mode(m)}@{sz}"
    try:
        for img in imgs[:args.warmup]:
            run(cfg, img)
        rows = [run(cfg, img) for img in imgs[args.warmup:]]
    except Exception as e:  # a broken engine should not stop the other configs
        res["errors"][cfg] = repr(e)[:300]
        print(f"{cfg:14s} failed: {repr(e)[:150]}")
        continue
    ms = [t * 1000 for t, _ in rows]
    res["configs"][cfg] = {"mean_ms": st.mean(ms), "median_ms": st.median(ms), "p90_ms": float(np.percentile(ms, 90)),
                           "windows_per_img": st.mean(n for _, n in rows)}
    r = res["configs"][cfg]
    print(f"{cfg:14s} {r['mean_ms']:8.1f} ms  (median {r['median_ms']:.1f}, p90 {r['p90_ms']:.1f})  windows/img {r['windows_per_img']:.2f}", flush=True)
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
