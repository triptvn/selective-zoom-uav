"""Self-guided selective zoom (Z1*) and uniform slicing (SAHI-style) for any detector.

A detector is anything with an Ultralytics-like `predict(images, imgsz=..., conf=..., max_det=...)`
that returns results with `.boxes.xyxy / .conf / .cls`. Ultralytics models (PyTorch, TensorRT, NCNN,
ONNX) work as they are; D-FINE goes through `tinyuav.dfine.DFine`.

All boxes are in original-image pixels. A detection set is a tuple (xyxy [N,4], score [N], cls [N]).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import batched_nms

EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


@dataclass
class ZoomConfig:
    imgsz: int = 896          # input size of both passes
    window: float = 0.6       # window side as a fraction of the image side
    tau: float = 0.10         # score threshold for "confident" small detections
    small_px: float = 32.0    # sqrt(w*h) below this (in the imgsz frame) counts as small
    min_count: int = 8        # gate m: zoom only if the window holds at least this many
    grid: int = 32
    k: int = 1                # number of windows (K=2 in the paper's high-accuracy variant)
    max_iou: float = 0.3      # max overlap between windows when k > 1
    nms_iou: float = 0.6      # class-wise NMS used to fuse the views
    max_det: int = 500
    edge_px: float = 3.0      # window boxes this close to an inner border are dropped


def _to_numpy(result):
    b = result.boxes
    if b is None or len(b) == 0:
        return EMPTY
    return (b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(np.int64))


def predict(model, img, imgsz, max_det=500, **kw):
    r = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.7, max_det=max_det, verbose=False, **kw)[0]
    return _to_numpy(r)


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def select_windows(xyxy, score, img_w, img_h, cfg: ZoomConfig, k=None, tau=None):
    """Densest window(s) of confident small detections, or [] when the gate is closed.

    Scores of small boxes are accumulated on a grid at their centres; window sums come from an
    integral image, so every window position is scored in O(1).
    """
    k = cfg.k if k is None else k
    tau = cfg.tau if tau is None else tau
    if k <= 0 or len(xyxy) == 0:
        return []
    s = cfg.imgsz / max(img_w, img_h)
    wh = (xyxy[:, 2:] - xyxy[:, :2]) * s
    keep = (score >= tau) & (np.sqrt(np.clip(wh[:, 0] * wh[:, 1], 0, None)) < cfg.small_px)
    if keep.sum() < cfg.min_count:
        return []
    g = cfg.grid
    gx = np.clip(((xyxy[keep, 0] + xyxy[keep, 2]) / 2 / img_w * g).astype(int), 0, g - 1)
    gy = np.clip(((xyxy[keep, 1] + xyxy[keep, 3]) / 2 / img_h * g).astype(int), 0, g - 1)

    def window_sums(values):
        m = np.zeros((g, g), np.float32)
        np.add.at(m, (gy, gx), values)
        ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return ii[n:, n:] - ii[:-n, n:] - ii[n:, :-n] + ii[:-n, :-n]

    n = max(1, int(round(cfg.window * g)))
    sums = window_sums(score[keep])
    counts = window_sums(np.ones(int(keep.sum()), np.float32))
    chosen = []
    for y, x in zip(*np.unravel_index(np.argsort(-sums, axis=None), sums.shape)):
        if len(chosen) >= k or counts[y, x] < cfg.min_count:
            break
        box = (x / g * img_w, y / g * img_h, (x + n) / g * img_w, (y + n) / g * img_h)
        if all(_box_iou(box, c) <= cfg.max_iou for c in chosen):
            chosen.append(box)
    return chosen


def tile_windows(img_w, img_h, n=2, overlap=0.2):
    """Uniform n x n tiles with overlap (SAHI-style slicing)."""
    tw, th = img_w / (n - (n - 1) * overlap), img_h / (n - (n - 1) * overlap)
    xs = [i * tw * (1 - overlap) for i in range(n)]
    ys = [j * th * (1 - overlap) for j in range(n)]
    return [(x, y, min(img_w, x + tw), min(img_h, y + th)) for y in ys for x in xs]


def _clean(dets, win, img_w, img_h, edge_px):
    """Drop boxes cut by an inner window border and shift the rest to image coordinates."""
    b, s, c = dets
    if len(b) == 0:
        return dets
    x0, y0, x1, y1 = win
    bad = np.zeros(len(b), bool)
    if x0 > 0:
        bad |= b[:, 0] <= edge_px
    if y0 > 0:
        bad |= b[:, 1] <= edge_px
    if x1 < img_w:
        bad |= b[:, 2] >= (x1 - x0) - edge_px
    if y1 < img_h:
        bad |= b[:, 3] >= (y1 - y0) - edge_px
    return b[~bad] + np.array([x0, y0, x0, y0], np.float32), s[~bad], c[~bad]


def run_windows(model, img, windows, imgsz, max_det=500, edge_px=3.0, batched=False, **kw):
    """Detect inside each window, cropped from the original image and resized to imgsz."""
    if not windows:
        return []
    h, w = img.shape[:2]
    boxes = [(int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))) for x0, y0, x1, y1 in windows]
    crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    if batched:
        res = [_to_numpy(r) for r in model.predict(crops, imgsz=imgsz, conf=0.001, iou=0.7, max_det=max_det,
                                                   verbose=False, **kw)]
    else:
        res = [predict(model, c, imgsz, max_det, **kw) for c in crops]
    return [_clean(r, box, w, h, edge_px) for r, box in zip(res, boxes)]


def fuse(global_dets, window_dets, nms_iou=0.6, max_det=500):
    """Union of the full view and the window views, class-wise NMS, top max_det by score."""
    parts = [global_dets] + list(window_dets)
    B = torch.from_numpy(np.concatenate([p[0] for p in parts]).astype(np.float32))
    S = torch.from_numpy(np.concatenate([p[1] for p in parts]).astype(np.float32))
    C = torch.from_numpy(np.concatenate([p[2] for p in parts]).astype(np.int64))
    if len(B) == 0:
        return EMPTY
    keep = batched_nms(B, S, C, nms_iou)[:max_det]
    return B[keep].numpy(), S[keep].numpy(), C[keep].numpy()


class SelectiveZoom:
    """Z1*: one full-view pass, then at most k zoom passes on dense small-object windows.

    >>> z = SelectiveZoom(YOLO('a1zt.pt'))
    >>> boxes, scores, classes = z(cv2.imread('img.jpg'))
    `z.calls` counts detector calls since construction (1 + number of opened windows per image).
    """

    def __init__(self, model, cfg: ZoomConfig | None = None, batched=False, **predict_kw):
        self.model, self.cfg, self.batched, self.kw = model, cfg or ZoomConfig(), batched, predict_kw
        self.calls = 0

    def __call__(self, img, return_windows=False):
        c = self.cfg
        h, w = img.shape[:2]
        g = predict(self.model, img, c.imgsz, c.max_det, **self.kw)
        wins = select_windows(g[0], g[1], w, h, c)
        wd = run_windows(self.model, img, wins, c.imgsz, c.max_det, c.edge_px, self.batched, **self.kw)
        self.calls += 1 + len(wins)
        out = fuse(g, wd, c.nms_iou, c.max_det)
        return (out, wins) if return_windows else out
