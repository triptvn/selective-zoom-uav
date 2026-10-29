"""Dataset preparation: VisDrone2019-DET, the UAVDT frame subset, and zoom-aware training (Z_T, formerly ZT) crops.

Every dataset ends up in the same YOLO layout:
    <root>/images/{train,val,test}/*.jpg
    <root>/labels/{train,val,test}/*.txt
    <root>/data.yaml
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
UAVDT_NAMES = ["car", "truck", "bus"]
VISDRONE_ZIPS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def write_yaml(root, names):
    root = Path(root)
    lines = [f"path: {root.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n")
    return root / "data.yaml"


def load_names(data_yaml):
    import yaml
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


# ----------------------------------------------------------------------------------------- VisDrone
def _visdrone_split(src, images_out, labels_out):
    """Raw annotation: left, top, width, height, score, category, truncation, occlusion.

    Rows with score 0 (ignored regions), category 0 (ignored) and 11 (others) are dropped,
    boxes are clipped to the image and degenerate ones removed. Categories 1..10 -> classes 0..9.
    """
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for p in sorted((src / "images").glob("*.jpg")):
        w, h = Image.open(p).size
        rows, ann = [], src / "annotations" / f"{p.stem}.txt"
        for line in (ann.read_text(errors="ignore").strip().splitlines() if ann.exists() else []):
            v = [x for x in line.replace(" ", "").split(",") if x]
            if len(v) < 6:
                continue
            try:
                x, y, bw, bh = (int(float(t)) for t in v[:4])
                score, cat = int(float(v[4])), int(float(v[5]))
            except ValueError:
                continue
            if score == 0 or not 1 <= cat <= 10 or bw <= 0 or bh <= 0:
                continue
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            rows.append(f"{cat - 1} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        if not (images_out / p.name).exists():
            shutil.move(str(p), images_out / p.name)
        (labels_out / f"{p.stem}.txt").write_text("\n".join(rows))
        n_img, n_box = n_img + 1, n_box + len(rows)
    return n_img, n_box


def prepare_visdrone(root, splits=("train", "val", "test")):
    """Download (Ultralytics mirror of the official archives) and convert VisDrone2019-DET."""
    from ultralytics.utils.downloads import download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        img_dir = root / "images" / split
        if img_dir.exists() and any(img_dir.glob("*.jpg")):
            continue
        src = root / VISDRONE_ZIPS[split]
        if not src.exists():
            download([f"{ASSETS_URL}/{VISDRONE_ZIPS[split]}.zip"], dir=root, threads=1, unzip=True, delete=True)
        n_img, n_box = _visdrone_split(src, img_dir, root / "labels" / split)
        if n_img == 0:
            raise RuntimeError(f"no images found for split {split} in {src}")
        print(f"[visdrone] {split}: {n_img} images, {n_box} boxes")
        shutil.rmtree(src, ignore_errors=True)
    return write_yaml(root, VISDRONE_NAMES)


# ----------------------------------------------------------------------------------------- UAVDT
def prepare_uavdt(src, dst):
    """UAVDT-DET with ignored regions blacked out (folders train/ and test/, labels in gt_whole/).

    train: every 5th frame of all training sequences except the last five
    val:   every 10th frame of those last five sequences
    test:  every 10th frame of all test sequences (1,665 images)
    Label rows are "class(1 car, 2 truck, 3 bus), x, y, w, h, id". Images are symlinked.
    """
    src, dst = Path(src), Path(dst)
    frame = lambda q: int(Path(q).stem.split("img")[-1])
    seq = lambda q: Path(q).stem.split("_")[0]
    tr = sorted(str(q) for q in src.glob("train/**/*.jpg"))
    te = sorted(str(q) for q in src.glob("test/**/*.jpg"))
    val_seqs = set(sorted({seq(q) for q in tr})[-5:])
    splits = {"train": [q for q in tr if seq(q) not in val_seqs and frame(q) % 5 == 1],
              "val": [q for q in tr if seq(q) in val_seqs and frame(q) % 10 == 1],
              "test": [q for q in te if frame(q) % 10 == 1]}
    for split, files in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_box = 0
        for q in files:
            link = dst / "images" / split / Path(q).name
            if not link.exists():
                os.symlink(q, link)
            w, h = Image.open(q).size
            gt = src / ("test" if split == "test" else "train") / "gt_whole" / (Path(q).stem + ".txt")
            rows = []
            for line in (gt.read_text().splitlines() if gt.exists() else []):
                v = [float(x) for x in line.split(",")[:5]]
                if len(v) < 5 or v[3] <= 0 or v[4] <= 0:
                    continue
                x0, y0, x1, y1 = max(v[1], 0), max(v[2], 0), min(v[1] + v[3], w), min(v[2] + v[4], h)
                if x1 <= x0 or y1 <= y0:
                    continue
                rows.append(f"{int(v[0]) - 1} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            n_box += len(rows)
            (dst / "labels" / split / (Path(q).stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"[uavdt] {split}: {len(files)} images, {n_box} boxes")
    return write_yaml(dst, UAVDT_NAMES)


# ----------------------------------------------------------------------------------------- Z_T (zoom-aware training)
def make_zt_dataset(src, dst, imgsz=896, window=0.6, min_count=8, min_visible=0.5):
    """Zoom-aware training set (Z_T): all training images plus one dense crop per image where possible.

    The crop is chosen from ground-truth boxes with the same rule as Z_S at test time (60% window,
    at least eight small objects), cut from the original image and saved as a new training sample;
    boxes keeping at least half of their area are retained. val/test are symlinked unchanged.
    Returns (n_images, n_crops); train the result for E_base * n_images / (n_images + n_crops) epochs.
    """
    import cv2

    from .zoom import ZoomConfig, select_windows

    src, dst = Path(src), Path(dst)
    cfg = ZoomConfig(imgsz=imgsz, window=window, min_count=min_count, k=1)
    for sub in ("images/train", "labels/train"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        for kind in ("images", "labels"):
            link = dst / kind / split
            if not link.exists() and (src / kind / split).exists():
                os.symlink((src / kind / split).resolve(), link)
    n_img = n_crop = 0
    for p in sorted((src / "images" / "train").glob("*.jpg")):
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            s, d = src / kind / "train" / f"{p.stem}{ext}", dst / kind / "train" / f"{p.stem}{ext}"
            if s.exists() and not d.exists():
                os.symlink(s.resolve(), d)
        n_img += 1
        lab = src / "labels" / "train" / f"{p.stem}.txt"
        rows = np.loadtxt(lab, ndmin=2) if lab.exists() else np.zeros((0, 5))
        if rows.size == 0:
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        cls, cx, cy, bw, bh = rows[:, 0], rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
        for j, (x0, y0, x1, y1) in enumerate(select_windows(xyxy, np.ones(len(xyxy), np.float32), w, h, cfg, tau=0.5)):
            x0, y0, x1, y1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
            crop = img[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            ix0, iy0 = np.clip(xyxy[:, 0], x0, x1), np.clip(xyxy[:, 1], y0, y1)
            ix1, iy1 = np.clip(xyxy[:, 2], x0, x1), np.clip(xyxy[:, 3], y0, y1)
            visible = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None) / np.clip(bw * bh, 1e-6, None)
            keep = visible >= min_visible
            if not keep.any():
                continue
            out = np.stack([cls, ((ix0 + ix1) / 2 - x0) / cw, ((iy0 + iy1) / 2 - y0) / ch, (ix1 - ix0) / cw, (iy1 - iy0) / ch], 1)[keep]
            cv2.imwrite(str(dst / "images" / "train" / f"{p.stem}_zoom{j}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.savetxt(dst / "labels" / "train" / f"{p.stem}_zoom{j}.txt", out, fmt="%d %.6f %.6f %.6f %.6f")
            n_crop += 1
    names = load_names(src / "data.yaml")
    write_yaml(dst, names)
    return n_img, n_crop


def step_matched_epochs(base_epochs, n_images, n_crops):
    return round(base_epochs * n_images / (n_images + n_crops))
