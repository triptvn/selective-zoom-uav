#!/usr/bin/env bash
# Accuracy-latency frontier on a GPU: PyTorch FP32 and TensorRT FP16, full pipeline, batch 1.
# Measure all detectors you want to compare in the same session.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:?usage: latency_t4.sh weights.pt}
IMG=data/VisDrone/images/test
n=$(basename "$W" .pt)

python tools/latency.py --weights "$W" --images $IMG --batched --out results/latency/${n}_pt.json \
  --configs single@896,single@1088,single@1216,single@1344,z1@896,z1k2@896,z1@1088,sahi@896
python tools/export_trt.py --weights "$W" --sizes 896 1088 1216 1344 --out engines/${n}.json
python tools/latency.py --weights "$W" --engines engines/${n}.json --images $IMG --batched --out results/latency/${n}_trt.json \
  --configs single@896,single@1088,single@1216,single@1344,z1@896,z1k2@896,z1@1088,sahi@896
