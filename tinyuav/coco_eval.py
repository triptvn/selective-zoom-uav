"""COCO evaluation used for every number in the paper.

- AP50 / AP50:95 with at most 500 detections per image and one label per box.
- Size buckets (tiny < 16 px, small 16-32 px, medium 32-96 px) are measured after resizing the
  image so that its long side is 640 px, i.e. roughly the size the network sees.
- P and R are taken at the confidence threshold that maximises the class-mean F1 at IoU 0.5
  (same definition as Ultralytics).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BUCKET_FRAME = 640
SIZE_BUCKETS = [("all", 0.0, 1e10), ("tiny", 0.0, 16.0 ** 2), ("small", 16.0 ** 2, 32.0 ** 2),
                ("medium", 32.0 ** 2, 96.0 ** 2), ("large", 96.0 ** 2, 1e10)]


def build_ground_truth(images_dir, labels_dir, names):
    """COCO ground truth from YOLO labels. Boxes stay in image pixels; `area` is in the 640 frame."""
    images, anns = [], []
    for image_id, p in enumerate(sorted(Path(images_dir).glob("*.jpg")), start=1):
        w, h = Image.open(p).size
        images.append({"id": image_id, "file_name": p.name, "width": w, "height": h})
        lab = Path(labels_dir) / f"{p.stem}.txt"
        if not lab.exists():
            continue
        s = BUCKET_FRAME / max(w, h)
        for line in lab.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            cx, cy, bw, bh = (float(x) for x in v[1:5])
            bw, bh = bw * w, bh * h
            if bw <= 0 or bh <= 0:
                continue
            anns.append({"id": len(anns) + 1, "image_id": image_id, "category_id": int(float(v[0])),
                         "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh], "area": bw * s * bh * s, "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}


def to_coco(image_id, dets):
    b, s, c = dets
    return [{"image_id": image_id, "category_id": int(k), "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
             "score": float(sc)} for (x0, y0, x1, y1), sc, k in zip(b, s, c)]


def _coco(gt):
    g = COCO()
    g.dataset = gt
    g.createIndex()
    return g


def _mean(a):
    a = a[a > -1]
    return float(a.mean() * 100) if a.size else float("nan")


def evaluate(gt, dets, max_det=500):
    """-> dict with mAP50, mAP50_95, P, R, per_class and tiny/small/medium AP."""
    if not dets:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "P": 0.0, "R": 0.0, "per_class": {}}
    g = _coco(gt)
    ev = COCOeval(g, g.loadRes(list(dets)), iouType="bbox")
    ev.params.maxDets = [max_det]
    ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
    ev.evaluate()
    ev.accumulate()
    pr = ev.eval["precision"]  # [iou, recall, class, area, maxdet]
    i50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    res = {"mAP50": _mean(pr[i50, :, :, 0, 0]), "mAP50_95": _mean(pr[:, :, :, 0, 0]),
           "per_class": {names.get(cid, str(cid)): {"AP50": _mean(pr[i50, :, k, 0, 0]), "AP50_95": _mean(pr[:, :, k, 0, 0])}
                         for k, cid in enumerate(ev.params.catIds)}}
    res.update(_precision_recall(ev, i50))
    res.update(_size_buckets(gt, dets, max_det))
    return res


def _precision_recall(ev, i50):
    per = {cid: ([], [], 0) for cid in ev.params.catIds}
    for e in ev.evalImgs:
        if e is None:
            continue
        sc, tp, npos = per[e["category_id"]]
        if len(e["dtScores"]):
            keep = ~np.asarray(e["dtIgnore"])[i50].astype(bool)
            sc += list(np.asarray(e["dtScores"])[keep])
            tp += list((np.asarray(e["dtMatches"])[i50] > 0)[keep])
        per[e["category_id"]] = (sc, tp, npos + int((np.asarray(e["gtIgnore"]) == 0).sum()))
    th = np.linspace(0, 1, 1001)
    P, R = [], []
    for sc, tp, npos in per.values():
        if npos == 0:
            continue
        sc, tp = np.asarray(sc), np.asarray(tp, float)
        o = np.argsort(-sc)
        sc, ctp = sc[o], np.cumsum(tp[o])
        n = np.searchsorted(-sc, -th, side="right")  # detections with score >= threshold
        tpc = np.where(n > 0, ctp[np.maximum(n - 1, 0)], 0.0)
        P.append(np.where(n > 0, tpc / np.maximum(n, 1), 1.0))
        R.append(tpc / npos)
    P, R = np.array(P), np.array(R)
    i = int((2 * P * R / np.maximum(P + R, 1e-16)).mean(0).argmax())
    return {"P": float(P[:, i].mean() * 100), "R": float(R[:, i].mean() * 100), "conf_F1": float(th[i])}


def _size_buckets(gt, dets, max_det):
    g = _coco(gt)
    d = g.loadRes(list(dets))
    # loadRes computes detection areas in image pixels; move them to the bucket frame as well,
    # otherwise unmatched detections land in the wrong bucket.
    scale = {im["id"]: (BUCKET_FRAME / max(im["width"], im["height"])) ** 2 for im in gt["images"]}
    for a in d.anns.values():
        a["area"] *= scale[a["image_id"]]
    ev = COCOeval(g, d, iouType="bbox")
    ev.params.areaRng = [[lo, hi] for _, lo, hi in SIZE_BUCKETS]
    ev.params.areaRngLbl = [n for n, _, _ in SIZE_BUCKETS]
    ev.params.maxDets = [max_det]
    ev.evaluate()
    ev.accumulate()
    pr = ev.eval["precision"]
    i50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    out = {}
    for a, (name, _, _) in enumerate(SIZE_BUCKETS):
        if name in ("tiny", "small", "medium"):
            out[f"{name}_AP50"] = _mean(pr[i50, :, :, a, 0])
            out[f"{name}_AP50_95"] = _mean(pr[:, :, :, a, 0])
    return out
