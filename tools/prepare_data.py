"""Prepare VisDrone2019-DET, the UAVDT subset and the zoom-aware (ZT) variants.

  python tools/prepare_data.py visdrone --root data/VisDrone
  python tools/prepare_data.py uavdt --src /path/to/UAVDT-DET-ig --root data/UAVDT
  python tools/prepare_data.py zt --src data/VisDrone --root data/VisDrone_ZT --base-epochs 100
"""
import argparse

import _path  # noqa: F401
from tinyuav import data

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["visdrone", "uavdt", "zt"])
ap.add_argument("--root", required=True, help="output directory")
ap.add_argument("--src", help="UAVDT-DET-ig folder (uavdt) or a prepared dataset (zt)")
ap.add_argument("--splits", default="train,val,test")
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--base-epochs", type=int, default=100, help="zt: epochs of the baseline to step-match")
args = ap.parse_args()

if args.what == "visdrone":
    print(data.prepare_visdrone(args.root, args.splits.split(",")))
elif args.what == "uavdt":
    print(data.prepare_uavdt(args.src, args.root))
else:
    n_img, n_crop = data.make_zt_dataset(args.src, args.root, imgsz=args.imgsz)
    e = data.step_matched_epochs(args.base_epochs, n_img, n_crop)
    print(f"{n_img} images + {n_crop} crops -> train for {e} epochs to match {args.base_epochs} baseline epochs")
