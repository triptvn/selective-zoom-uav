#!/usr/bin/env bash
# UAVDT frame subset (50-epoch schedule; Z_T step-matched to 34 epochs). UAVDT is sensitive to the
# training seed, so report every variant as a mean over seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
UAVDT_RAW=${UAVDT_RAW:?set UAVDT_RAW to the UAVDT-DET-ig folder}

python tools/prepare_data.py uavdt --src "$UAVDT_RAW" --root data/UAVDT
python tools/prepare_data.py zt --src data/UAVDT --root data/UAVDT_ZT --base-epochs 50

for s in 0 1 2; do
  python tools/train.py --model yolo26n-p2.yaml              --data data/UAVDT/data.yaml    --epochs 50 --seed $s --name uv_base_s$s
  python tools/train.py --model yolo26n-p2.yaml              --data data/UAVDT_ZT/data.yaml --epochs 34 --seed $s --name uv_zt_s$s
  python tools/train.py --model configs/yolo26n-p2-noP5.yaml --data data/UAVDT_ZT/data.yaml --epochs 34 --seed $s --name uv_yfzt_s$s
done
for run in runs/train/uv_*_s?; do
  python tools/evaluate.py --weights "$run/weights/best.pt" --data data/UAVDT/data.yaml --split test \
    --modes single,zs,zs_k2,sahi --out results/uavdt/$(basename "$run")_test.json
done
# upscaling only interpolates on UAVDT
for sz in 1088 1216 1344 1536; do
  python tools/evaluate.py --weights runs/train/uv_base_s0/weights/best.pt --data data/UAVDT/data.yaml --split test \
    --imgsz $sz --modes single --out results/uavdt/uv_base_s0_test_$sz.json
done
python tools/evaluate_video.py --weights runs/train/uv_yfzt_s0/weights/best.pt --root "$UAVDT_RAW" --seqs 1000 --frames 100000 \
  --out results/uavdt/video.json
