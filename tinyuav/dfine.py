"""D-FINE (official repository) behind the same `predict` interface as an Ultralytics model.

Needs a clone of https://github.com/Peterande/D-FINE (we used commit 956d170) and its config file:
    DFine("best.pth", repo="/path/to/D-FINE", cfg="/path/to/dfine_hgnetv2_n_visdrone.yml")
Images are resized to a square input without letterboxing, as in the official inference tool, and at
most `num_top_queries` (300) boxes are returned per image.
"""
import os
import sys

import cv2
import numpy as np
import torch


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class DFine:
    def __init__(self, ckpt, repo=None, cfg=None, device=None):
        repo, cfg = repo or os.environ["DFINE_REPO"], cfg or os.environ["DFINE_CFG"]
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.core import YAMLConfig

        c = YAMLConfig(cfg, resume=ckpt)
        if "HGNetv2" in c.yaml_cfg:
            c.yaml_cfg["HGNetv2"]["pretrained"] = False
        state = torch.load(ckpt, map_location="cpu")
        c.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = c.model.deploy().to(self.device).eval()
        self.post = c.postprocessor.deploy().eval()

    @torch.no_grad()
    def predict(self, imgs, imgsz=896, conf=0.001, max_det=500, half=False, **_):
        imgs = imgs if isinstance(imgs, list) else [imgs]
        x = np.stack([cv2.resize(im[:, :, ::-1], (imgsz, imgsz), interpolation=cv2.INTER_LINEAR) for im in imgs])
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float().div_(255)
        sizes = torch.tensor([[im.shape[1], im.shape[0]] for im in imgs], device=self.device)
        with torch.autocast("cuda", enabled=half and self.device != "cpu"):
            labels, boxes, scores = self.post(self.model(x), sizes)
        out = []
        for lb, bx, sc in zip(labels, boxes, scores):
            k = sc >= conf
            o = sc[k].float().argsort(descending=True)[:max_det]
            out.append(_Result(_Boxes(bx[k].float()[o], sc[k].float()[o], lb[k][o].float())))
        return out


def load_detector(weights):
    """`.pth` -> D-FINE (DFINE_REPO / DFINE_CFG environment variables), anything else -> ultralytics.YOLO."""
    if str(weights).endswith(".pth"):
        return DFine(str(weights))
    from ultralytics import YOLO

    return YOLO(weights)
