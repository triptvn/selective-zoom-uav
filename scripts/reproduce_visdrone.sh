#!/usr/bin/env bash
# VisDrone: baseline, ablation arms and the proposed detector, three seeds each, then the evaluation
# behind the comparison, ablation and significance tables. One T4 needs roughly 8-9 h per 100-epoch run.
set -euo pipefail
cd "$(dirname "$0")/.."

python tools/prepare_data.py visdrone --root data/VisDrone
python tools/prepare_data.py zt --src data/VisDrone --root data/VisDrone_ZT --base-epochs 100   # -> 54 epochs

for s in 0 1 2; do
  python tools/train.py --model yolo26n-p2.yaml               --data data/VisDrone/data.yaml    --epochs 100 --seed $s --name base_s$s
  python tools/train.py --model configs/yolo26n-p2-noP5.yaml  --data data/VisDrone/data.yaml    --epochs 100 --seed $s --name a1_s$s
  python tools/train.py --model yolo26n-p2.yaml               --data data/VisDrone_ZT/data.yaml --epochs 54  --seed $s --name zt_s$s
  python tools/train.py --model configs/yolo26n-p2-noP5.yaml  --data data/VisDrone_ZT/data.yaml --epochs 54  --seed $s --name a1zt_s$s
done
# other detectors in the comparison use the same command, e.g. --model yolov10n.yaml / yolo11n.yaml / configs/yolo11n-p2.yaml

for run in runs/train/*_s?; do
  n=$(basename "$run")
  for split in val test; do
    python tools/evaluate.py --weights "$run/weights/best.pt" --data data/VisDrone/data.yaml --split $split \
      --modes single,z1,z1k2,sahi --out results/visdrone/${n}_${split}.json
  done
done

python tools/stats.py welch "results/visdrone/a1zt_s?_test.json" "results/visdrone/base_s?_test.json" --mode single
python tools/stats.py welch "results/visdrone/a1zt_s?_test.json" "results/visdrone/base_s?_test.json" --mode z1
python tools/stats.py paired "results/visdrone/*_test.json" --mode z1 --ref single

# whole-image upscaling at equal cost (no retraining)
for sz in 1088 1216 1344 1536; do
  python tools/evaluate.py --weights runs/train/a1zt_s0/weights/best.pt --data data/VisDrone/data.yaml --split test \
    --imgsz $sz --modes single,z1 --out results/visdrone/a1zt_s0_test_$sz.json
done
# window-selection study
python tools/evaluate.py --weights runs/train/base_s0/weights/best.pt --data data/VisDrone/data.yaml --split test \
  --modes single,z1,random,center,oracle --out results/visdrone/base_s0_test_windows.json
