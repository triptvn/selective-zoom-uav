#!/usr/bin/env bash
# Raspberry Pi 5 (CPU only): export to NCNN on the device, then latency on 100 test-dev images and
# NCNN accuracy on the full test-dev. NCNN uses a fixed square input, which costs every detector
# one to two AP50 points compared with rectangular inference, without changing their ranking.
# Let the board cool down between long runs; we logged temperature and kept it below the soft-throttle limit.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:?usage: edge_pi5.sh weights.pt}
n=$(basename "$W" .pt)

for sz in 896 1088; do
  yolo export model="$W" format=ncnn imgsz=$sz
  mv "${W%.pt}_ncnn_model" "${n}_${sz}_ncnn_model"
done
python tools/latency.py --weights ${n}_896_ncnn_model --images data/VisDrone/images/test --n 100 --warmup 10 --device cpu \
  --configs single@896,z1@896,sahi@896 --out results/pi5/${n}_896.json
python tools/latency.py --weights ${n}_1088_ncnn_model --images data/VisDrone/images/test --n 100 --warmup 10 --device cpu \
  --configs single@1088,z1@1088 --out results/pi5/${n}_1088.json
python tools/evaluate.py --weights ${n}_896_ncnn_model --data data/VisDrone/data.yaml --split test --modes single,z1 \
  --out results/pi5/${n}_896_acc.json
