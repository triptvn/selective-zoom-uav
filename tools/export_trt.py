"""Export TensorRT FP16 engines, one per input size, and write {imgsz: engine} for tools/latency.py.

  python tools/export_trt.py --weights yfzt.pt --sizes 896 1344 --out engines.json

Each engine is test-run after export (square and letterboxed inputs, batch 1 and 2); if that fails,
simpler export settings are tried in turn, ending with a fixed batch-1 engine.

P5-free models: Ultralytics stores the largest head stride (16) in the engine metadata, so letterboxed
inputs are padded to multiples of 16 (e.g. 1088 x 624). The backbone still has a stride-32 stage, and
heights that are not multiples of 32 break the top-down concat inside the engine. PyTorch inference is
unaffected because Ultralytics always pads .pt models to multiples of 32. We rewrite the stride in the
engine metadata to 32 after export.
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


def fix_stride(engine):
    d = Path(engine).read_bytes()
    n = int.from_bytes(d[:4], "little")
    meta = json.loads(d[4:4 + n])
    if int(meta.get("stride", 32)) < 32:
        meta["stride"] = 32
        m = json.dumps(meta).encode()
        Path(engine).write_bytes(len(m).to_bytes(4, "little") + m + d[4 + n:])


def works(engine, sz, two):
    try:
        m = YOLO(engine, task="detect")
        im = np.zeros((sz, sz, 3), np.uint8)
        m.predict(im, imgsz=sz, verbose=False)
        m.predict(np.zeros((sz * 9 // 16, sz, 3), np.uint8), imgsz=sz, verbose=False)
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
        fix_stride(e)
        if works(e, sz, kw["dynamic"]):
            out[sz] = e
            break
Path(args.out).write_text(json.dumps(out, indent=1))
print(out)
