"""Train a detector with the protocol of the paper.

Defaults: 896 px, batch 8 (nominal batch 64), COCO initialisation, MuSGD with a constant lr of 0.01.
The optimizer is fixed on purpose: Ultralytics' "auto" switches to AdamW for runs under 10k steps,
which would change the recipe between the 50- and 100-epoch schedules.

  # proposed detector (Y_F + Z_T; called A1 + ZT during the experiments), step-matched to a 100-epoch baseline
  python tools/train.py --model configs/yolo26n-p2-noP5.yaml --data data/VisDrone_ZT/data.yaml --epochs 54 --seed 0
  # baseline
  python tools/train.py --model yolo26n-p2.yaml --data data/VisDrone/data.yaml --epochs 100 --seed 0
"""
import argparse
from pathlib import Path

import _path  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="model YAML (Ultralytics name or a file in configs/)")
ap.add_argument("--weights", default=None, help="initial weights; default: matching COCO checkpoint, e.g. yolo26n.pt")
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--stal-floor", type=int, default=None, help="16 = P2-aware STAL (ablation only)")
ap.add_argument("--project", default="runs/train")
ap.add_argument("--name", default=None)
args = ap.parse_args()

from ultralytics import YOLO  # noqa: E402


def coco_weights(model):
    """yolo26n-p2.yaml / yolo26n-p2-noP5.yaml -> yolo26n.pt, yolov8n-p2.yaml -> yolov8n.pt, ..."""
    stem = Path(model).stem
    return stem.split("-")[0] + ".pt"


model = YOLO(args.model)
if args.stal_floor:
    from tinyuav.stal import stal_floor_callback
    model.add_callback("on_train_batch_start", stal_floor_callback(args.stal_floor))
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
            seed=args.seed, deterministic=True, pretrained=args.weights or coco_weights(args.model),
            optimizer="MuSGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0, close_mosaic=min(10, args.epochs),
            project=args.project, name=args.name or f"{Path(args.model).stem}_s{args.seed}", exist_ok=True, plots=False)
