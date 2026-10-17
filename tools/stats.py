"""Significance tests over seeds, as in the paper's significance table.

Inputs are JSON files written by tools/evaluate.py; a glob pattern selects the seeds of one arm.

  # two arms, Welch t-test (unequal variances) on AP50 of a given mode
  python tools/stats.py welch "results/a1zt_s*_test.json" "results/base_s*_test.json" --mode z1
  # same trained models, Z1* versus single pass, paired t-test over all files
  python tools/stats.py paired "results/*_test.json" --mode z1 --ref single
"""
import argparse
import glob
import json

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("test", choices=["welch", "paired"])
ap.add_argument("a")
ap.add_argument("b", nargs="?")
ap.add_argument("--mode", default="single")
ap.add_argument("--ref", default="single", help="paired: mode to compare against")
ap.add_argument("--metric", default="mAP50")
args = ap.parse_args()


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    return [json.load(open(f))["modes"] for f in files]


if args.test == "welch":
    a = np.array([r[args.mode][args.metric] for r in load(args.a)])
    b = np.array([r[args.mode][args.metric] for r in load(args.b)])
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"A {a.mean():.2f}±{a.std(ddof=1):.2f} (n={len(a)})  B {b.mean():.2f}±{b.std(ddof=1):.2f} (n={len(b)})")
    print(f"delta {a.mean() - b.mean():+.2f}  t={t:.2f}  p={p:.2g}")
else:
    runs = load(args.a)
    x = np.array([r[args.mode][args.metric] for r in runs])
    y = np.array([r[args.ref][args.metric] for r in runs])
    t, p = stats.ttest_rel(x, y)
    print(f"{len(runs)} runs: delta {np.mean(x - y):+.2f} (min {np.min(x - y):+.2f}, max {np.max(x - y):+.2f})  t={t:.2f}  p={p:.2g}")
