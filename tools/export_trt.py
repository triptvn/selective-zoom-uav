"""Export TensorRT FP16 engines, one per input size, and write {imgsz: engine} for tools/latency.py.

  python tools/export_trt.py --weights a1zt.pt --sizes 896 1344 --out engines.json

Each engine is test-run with batch 1 and 2 after export; if that fails, simpler export settings are
tried in turn, ending with a fixed batch-1 engine (enough for single pass and Z1* with K=1).
Note: for the P5-free model, engines at 1088-1536 px passed export but failed at run time in our
setup (TensorRT through Ultralytics 8.4.131 on a T4), so the paper reports TensorRT only at 896 and 1344.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--sizes", type=int, nargs="+", default=[896, 1088, 1216, 1344])
ap.add_argument("--dir", default="engines")
ap.add_argument("--out", required=True)
args = ap.parse_args()

ATTEMPTS = [dict(dynamic=True, batch=4), dict(dynamic=True, batch=4, simplify=False),
            dict(dynamic=True, batch=2, simplify=False), dict(dynamic=False, batch=1)]


def works(engine, sz, two):
    try:
        m = YOLO(engine, task="detect")
        im = np.zeros((sz, sz, 3), np.uint8)
        m.predict(im, imgsz=sz, verbose=False)
        if two:
            m.predict([im, im], imgsz=sz, verbose=False)
        return True
    except Exception as e:
        print(f"engine {sz} failed at run time: {repr(e)[:200]}")
        return False


d = Path(args.dir)
d.mkdir(parents=True, exist_ok=True)
out = {}
for sz in args.sizes:
    for i, kw in enumerate(ATTEMPTS):
        w = d / f"{Path(args.weights).stem}_{sz}_{i}.pt"   # separate copy: export writes next to the weights
        shutil.copy2(args.weights, w)
        try:
            e = str(YOLO(str(w)).export(format="engine", imgsz=sz, half=True, device=0, verbose=False, **kw))
        except Exception as ex:
            print(f"export {sz} {kw} failed: {repr(ex)[:200]}")
            continue
        if works(e, sz, kw["dynamic"]):
            out[sz] = e
            break
Path(args.out).write_text(json.dumps(out, indent=1))
print(out)
