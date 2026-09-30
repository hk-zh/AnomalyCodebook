#!/usr/bin/env python3
"""Offline study of image-level read-outs, from dumped pixel anomaly maps.

Our pixel maps match or beat every baseline, but the image score derived from
them trails MultiADS by 2 points on average and by 5 to 7 on MPDD and MAD-Real.
The image-level deficit is therefore a read-out problem, not a representation
problem, and a read-out can be studied without touching the GPU: dump the maps
once, then score them every way we can think of.

The current read-out is the mean of the top rho*HW pixels. Its known failure is
that it is an absolute statistic, so an image whose normal pixels are uniformly
warm outranks an image with a small genuine defect. Most alternatives here are
contrast statistics, which cancel the per-image offset.

Usage:
    python3 tools/readout_study.py --dump_root results/readout/visa/dump
    python3 tools/readout_study.py --dump_root ... --json out.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def load_dump(dump_root):
    """-> {cls_name: (maps [N,H,W] float32, labels [N] int)}"""
    out = {}
    for cls_name in sorted(os.listdir(dump_root)):
        cls_dir = os.path.join(dump_root, cls_name)
        scores_p = os.path.join(cls_dir, "scores.json")
        if not os.path.isfile(scores_p):
            continue
        recs = json.load(open(scores_p))
        maps, labels = [], []
        for r in recs:
            p = os.path.join(cls_dir, r["defect_cls"], r["stem"] + ".npy")
            if not os.path.isfile(p):
                continue
            maps.append(np.load(p).astype(np.float32))
            labels.append(int(r["anomaly"]))
        if maps:
            out[cls_name] = (np.stack(maps), np.asarray(labels))
    return out


# ---------------------------------------------------------------- read-outs
# Each takes maps [N, H, W] and returns one score per image. Every one of these
# is label-free and has no per-dataset constant unless named otherwise.

def _topk_mean(flat, rho):
    k = max(1, int(round(rho * flat.shape[1])))
    if k == 1:
        return flat.max(axis=1)
    idx = np.argpartition(flat, -k, axis=1)[:, -k:]
    return np.take_along_axis(flat, idx, axis=1).mean(axis=1)


def topk(rho):
    return lambda flat: _topk_mean(flat, rho)


def topk_minus_median(rho):
    """Contrast form: how far the hottest region sits above the image's own bulk."""
    return lambda flat: _topk_mean(flat, rho) - np.median(flat, axis=1)


def topk_over_median(rho):
    return lambda flat: _topk_mean(flat, rho) / (np.median(flat, axis=1) + 1e-6)


def topk_z(rho):
    """Top-k mean expressed in per-image standard deviations."""
    def f(flat):
        mu = flat.mean(axis=1)
        sd = flat.std(axis=1) + 1e-6
        return (_topk_mean(flat, rho) - mu) / sd
    return f


def quantile_gap(qhi, qlo):
    return lambda flat: (np.quantile(flat, qhi, axis=1)
                         - np.quantile(flat, qlo, axis=1))


def adaptive_topk(z):
    """Label-free k: pool exactly the pixels lying more than z sigma above the
    image's own mean, which lets the pooled area follow the defect size instead
    of being fixed by a hand-set rho. Falls back to the max if none qualify."""
    def f(flat):
        mu = flat.mean(axis=1, keepdims=True)
        sd = flat.std(axis=1, keepdims=True) + 1e-6
        thr = mu + z * sd
        mask = flat >= thr
        num = (flat * mask).sum(axis=1)
        den = mask.sum(axis=1)
        out = np.where(den > 0, num / np.maximum(den, 1), flat.max(axis=1))
        return (out - mu[:, 0]) / sd[:, 0]
    return f


def logsumexp_pool(t):
    """Smooth maximum; t -> 0 approaches the max, large t approaches the mean."""
    def f(flat):
        m = flat.max(axis=1, keepdims=True)
        return (m[:, 0] + t * np.log(np.exp((flat - m) / t).mean(axis=1)))
    return f


READOUTS = {}
for _r in (0.001, 0.01, 0.05, 0.1):
    READOUTS[f"topk@{_r}"] = topk(_r)
    READOUTS[f"topk-med@{_r}"] = topk_minus_median(_r)
    READOUTS[f"topk/med@{_r}"] = topk_over_median(_r)
    READOUTS[f"topk-z@{_r}"] = topk_z(_r)
for _q in (0.999, 0.99, 0.95):
    READOUTS[f"q{_q}-q50"] = quantile_gap(_q, 0.5)
for _z in (2.0, 3.0, 4.0):
    READOUTS[f"adaptive-k@{_z}sd"] = adaptive_topk(_z)
for _t in (0.01, 0.05):
    READOUTS[f"lse@{_t}"] = logsumexp_pool(_t)


def evaluate(data, fn):
    """Per-class AUROC/AP, then the unweighted class mean, as the paper reports."""
    aurocs, aps = [], []
    for _cls, (maps, labels) in data.items():
        if labels.min() == labels.max():
            continue
        s = fn(maps.reshape(maps.shape[0], -1))
        aurocs.append(roc_auc_score(labels, s))
        aps.append(average_precision_score(labels, s))
    if not aurocs:
        return float("nan"), float("nan")
    return 100 * float(np.mean(aurocs)), 100 * float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", required=True,
                    help="a <save_path>/dump directory written by test.py --dump_maps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    data = load_dump(args.dump_root)
    n = sum(len(v[1]) for v in data.values())
    print(f"=== {args.tag or args.dump_root}: {len(data)} classes, {n} images")
    print(f"{'read-out':<20}{'img AUROC':>11}{'img AP':>9}")

    rows = {}
    for name, fn in READOUTS.items():
        auroc, apv = evaluate(data, fn)
        rows[name] = {"auroc": auroc, "ap": apv}
        print(f"{name:<20}{auroc:>11.1f}{apv:>9.1f}")

    best = max(rows.items(), key=lambda kv: kv[1]["auroc"])
    print(f"\nbest by AUROC: {best[0]}  {best[1]['auroc']:.1f} / {best[1]['ap']:.1f}")

    if args.json:
        json.dump({"tag": args.tag, "dump_root": args.dump_root, "rows": rows},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
