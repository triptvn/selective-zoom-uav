# Selective Zoom for Lightweight NMS-free UAV Tiny-Object Detection

Code for the paper *"Selective Zoom as Test-Time Scale Adaptation for Lightweight NMS-free UAV
Tiny-Object Detection: an Equal-Cost and Equal-Latency Study"* (IEEE Access, under review).

Most objects in drone images are tiny and packed into a few dense regions. With a small compute
budget, a detector can enlarge the whole image, slice it into tiles, zoom into selected regions or
switch to a larger network. This repository contains the pipeline we used to compare these options
at equal FLOPs and equal measured latency:

- **Z<sub>S</sub> (self-guided selective zoom).** One full-view pass, then at most one zoom pass on the
  densest window of the detector's own confident small detections. No extra parameters and no
  retraining, so it works on top of any detector (we tested ten, including D-FINE).
- **Z<sub>T</sub> (zoom-aware training).** Ground-truth dense crops chosen with the same rule as Z<sub>S</sub> are added
  to the training set, with the number of epochs reduced so that the optimizer steps match the baseline.
- **Y<sub>F</sub>.** YOLO26n-P2 without the stride-32 head: 1.74 M parameters, 28% fewer than YOLO26n-P2.

### Notation

The experiments used shorter working names, which still appear in some file names, run names and
command-line options. The paper uses the names on the left.

| Paper | Name during the experiments | In this code |
|---|---|---|
| Y<sub>F</sub> (P5-free YOLO26n-P2) | A1 | `configs/yolo26n-p2-noP5.yaml` |
| Z<sub>S</sub> (selective zoom) | Z1\* | mode `zs` (old name `z1`); `zs_k2` for two windows (old `z1k2`) |
| Z<sub>T</sub> (zoom-aware training) | ZT | `prepare_data.py zt` |
| Y<sub>F</sub> + Z<sub>T</sub> (proposed detector) | A1 + ZT | run and weight names `yfzt*` (`a1zt*` in our logs) |

The tools accept the old mode names, so commands and result files from before the renaming still work.

## Main results

VisDrone2019-DET test-dev, COCO AP50 (max. 500 detections, one label per box, image-wise), mean ± std
over three seeds. Latency: full pipeline, batch 1, Tesla T4.

| Detector | Params | Single pass | + Z<sub>S</sub> | Calls / image | PyTorch ms (single / Z<sub>S</sub>) |
|---|---|---|---|---|---|
| YOLOv10n | 2.27 M | 34.34 ± 0.09 | 38.62 | 1.85 | 10.8 / 23.2 |
| YOLO26n-P2 (baseline) | 2.40 M | 35.00 ± 0.14 | 38.74 ± 0.06 | 1.87 | 14.5 / 31.4 |
| **Y<sub>F</sub> + Z<sub>T</sub> (proposed)** | **1.74 M** | **35.49 ± 0.15** | **40.02 ± 0.10** | 1.87 | 13.5 / 28.6 |

- The proposed detector with a 1216 px full view and Z<sub>S</sub> reaches 43.11 AP50.
- Z<sub>S</sub> adds 3.0 to 5.3 AP50 to each of the nine other detectors we tested. It improved all 54
  training runs on VisDrone and UAVDT.
- On a Raspberry Pi 5 (CPU, NCNN), the proposed pipeline with Z<sub>S</sub> reaches 38.92 AP50 in 415 ms.
  A single pass of YOLO26s-P2 (9.41 M parameters) reaches 37.89 in 528 ms.

## Installation

```bash
git clone https://github.com/triptvn/selective-zoom-uav.git && cd selective-zoom-uav
pip install -r requirements.txt
```

The results in the paper were obtained with these versions:

| Setup | Software |
|---|---|
| Training (Tesla T4, 16 GB) | Python 3.13, PyTorch 2.11 + CUDA 12.8, Ultralytics 8.4.131 |
| Second GPU (RTX 3060) | Python 3.12, same PyTorch and Ultralytics versions |
| Raspberry Pi 5 | PyTorch 2.14.1 (CPU), NCNN 1.0, ONNX Runtime 1.31 |

Keep `ultralytics==8.4.131`, since YOLO26 training defaults can change between releases.

## Data

```bash
# VisDrone2019-DET (downloaded automatically): 6,471 train / 548 val / 1,610 test-dev images
python tools/prepare_data.py visdrone --root data/VisDrone

# UAVDT: download UAVDT-DET with ignored regions blacked out, then build the frame subset
#   train = every 5th frame of the training sequences except the last five
#   val   = every 10th frame of those five sequences, test = every 10th frame of all test sequences
python tools/prepare_data.py uavdt --src /path/to/UAVDT-DET-ig --root data/UAVDT

# zoom-aware training sets; the script prints the step-matched number of epochs
python tools/prepare_data.py zt --src data/VisDrone --root data/VisDrone_ZT --base-epochs 100   # 54 epochs
python tools/prepare_data.py zt --src data/UAVDT   --root data/UAVDT_ZT   --base-epochs 50      # 34 epochs
```

## Training

All detectors use one recipe: 896 px, batch 8, COCO initialisation, and MuSGD with a constant
learning rate of 0.01. We train for 100 epochs on VisDrone and 50 on UAVDT.

```bash
# proposed detector (Y_F + Z_T)
python tools/train.py --model configs/yolo26n-p2-noP5.yaml --data data/VisDrone_ZT/data.yaml --epochs 54 --seed 0
# baseline and other detectors
python tools/train.py --model yolo26n-p2.yaml --data data/VisDrone/data.yaml --epochs 100 --seed 0
python tools/train.py --model yolov10n.yaml   --data data/VisDrone/data.yaml --epochs 100 --seed 0
# P2-aware STAL ablation (16 px floor for tiny boxes; not used by the final model)
python tools/train.py --model yolo26n-p2.yaml --data data/VisDrone/data.yaml --epochs 50 --stal-floor 16
```

## Using Z<sub>S</sub> in your own code

```python
import cv2
from ultralytics import YOLO
from tinyuav import SelectiveZoom, ZoomConfig

zoom = SelectiveZoom(YOLO("yfzt.pt"), ZoomConfig(imgsz=896))   # defaults are the paper's settings
boxes, scores, classes = zoom(cv2.imread("frame.jpg"))         # original-image pixels
```

`ZoomConfig` holds the settings selected on the validation split:

| Setting | Value |
|---|---|
| Window size | 60% × 60% of the image |
| Grid | 32 × 32 |
| Small object | score ≥ 0.10 and √(wh) < 32 px in the 896 frame |
| Gate | m = 8 confident small detections in the window |
| Fusion | class-wise NMS at IoU 0.6, at most 500 boxes |

Raising `min_count` skips more zoom passes and lowers the cost. Use `k=2` for two windows.

## Evaluation

```bash
# accuracy of inference modes on one split (writes AP50, AP50:95, P, R, size buckets, per-class AP)
python tools/evaluate.py --weights yfzt.pt --data data/VisDrone/data.yaml --split test --modes single,zs,zs_k2,sahi --out results/yfzt_test.json
# whole-image upscaling
python tools/evaluate.py --weights yfzt.pt --data data/VisDrone/data.yaml --imgsz 1216 --modes single,zs
# window-selection study (random / centre / ground-truth oracle windows)
python tools/evaluate.py --weights base.pt --data data/VisDrone/data.yaml --modes zs,random,center,oracle
# significance over seeds
python tools/stats.py welch "results/yfzt_s?_test.json" "results/base_s?_test.json" --mode zs
```

We report COCO AP50 with up to 500 detections per image and one label per box, and do not rank
models with the default Ultralytics validation. That validation runs NMS with `multi_label=True`,
which lets one box score for several classes. Detectors that use NMS gain about 2.6 points from this.

## Latency

```bash
scripts/latency_t4.sh yfzt.pt     # PyTorch FP32 and TensorRT FP16 on a GPU
scripts/edge_pi5.sh yfzt.pt       # NCNN on a Raspberry Pi 5, run on the board
python tools/evaluate_video.py --weights yfzt_uavdt.pt --root /path/to/UAVDT-DET-ig   # window from the previous frame
```

Latency covers pre-processing, detector calls, window selection and fusion. It excludes image
decoding. Compare detectors only within one session on the same machine.

## Reproducing the paper

`scripts/reproduce_visdrone.sh` and `scripts/reproduce_uavdt.sh` list every training and evaluation
run behind the tables. Expect small differences between GPUs:

- AP50 of the same checkpoint agrees within about 0.02 between a T4 and an RTX 3060.
- P and R at the best-F1 threshold can move by one to two points, because the F1 curve is flat near
  its maximum.

For D-FINE-N we used the official repository (commit `956d170`), adapted to 896 px and ten classes.
`tinyuav/dfine.py` wraps a trained checkpoint so the same evaluation and zoom code applies.

## Repository layout

```
configs/      Y_F model (yolo26n-p2-noP5.yaml) and the YOLO11n-P2 baseline
tinyuav/      zoom.py (Z_S, tiles, fusion), data.py (datasets, Z_T crops), coco_eval.py,
              stal.py (P2-aware STAL), dfine.py (D-FINE wrapper)
tools/        prepare_data, train, evaluate, evaluate_video, latency, export_trt, stats
scripts/      end-to-end commands for VisDrone, UAVDT, GPU latency and Raspberry Pi 5
```

## Citation

```bibtex
@article{tinyuav2026selectivezoom,
  title   = {Selective Zoom as Test-Time Scale Adaptation for Lightweight NMS-free UAV Tiny-Object Detection:
             an Equal-Cost and Equal-Latency Study},
  author  = {...},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

AGPL-3.0, the same license as Ultralytics, which this code builds on.
