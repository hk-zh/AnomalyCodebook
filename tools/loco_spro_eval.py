# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Standalone MVTec-LOCO evaluator. Reads per-image anomaly maps dumped by a
# detector (test.py --dump_maps, or the DAPO dumper) together with the raw
# multi-valued ground truth and each object's defects_config.json, and reports:
#
#   * image-level AUROC, split into logical / structural (good images are the
#     shared negatives) and their mean, following the LOCO convention
#     "det. 90.7 (L:95.8 / S:85.5)";
#   * AU-sPRO at FPR limits 0.05 (headline) and 0.30, split logical / structural
#     and their mean, via tools/spro.py.
#
# Predictions are upsampled to the raw GT resolution because absolute
# saturation thresholds in defects_config.json are counts of full-res pixels.

import argparse
import json
import os

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from spro import build_saturation_lookup, au_spro_from_scores


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
LOGICAL = "logical_anomalies"
STRUCTURAL = "structural_anomalies"
GOOD = "good"


def load_map(npy_path, gt_hw):
    """Load a dumped anomaly map and upsample it to the GT (H, W)."""
    m = np.load(npy_path).astype(np.float32)
    if m.shape == gt_hw:
        return m
    # Bilinear upsample to full GT resolution.
    im = Image.fromarray(m, mode="F").resize((gt_hw[1], gt_hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def region_masks_for_image(gt_dir):
    """Return the list of per-region multi-valued masks for one image.

    LOCO stores one PNG per annotated region under <gt_dir>; each PNG's non-zero
    pixels carry the region's pixel_value. Missing dir -> no regions (good).
    """
    if not os.path.isdir(gt_dir):
        return []
    masks = []
    for n in sorted(os.listdir(gt_dir)):
        if n.lower().endswith(IMG_EXTS):
            masks.append(np.array(Image.open(os.path.join(gt_dir, n))))
    return masks


def class_resolution(dataset_root, cls_name):
    """(H, W) shared by every image in a LOCO class, read from any GT mask."""
    gt_root = os.path.join(dataset_root, cls_name, "ground_truth")
    for state in (LOGICAL, STRUCTURAL):
        sdir = os.path.join(gt_root, state)
        if not os.path.isdir(sdir):
            continue
        for stem in sorted(os.listdir(sdir)):
            masks = region_masks_for_image(os.path.join(sdir, stem))
            if masks:
                return masks[0].shape  # (H, W)
    raise RuntimeError(f"no ground-truth masks found for class {cls_name}")


def evaluate_class(dataset_root, dump_root, cls_name):
    """Gather (scores, gt_masks, records) for one object category."""
    with open(os.path.join(dump_root, cls_name, "scores.json")) as f:
        records = json.load(f)

    cfg_path = os.path.join(dataset_root, cls_name, "defects_config.json")
    with open(cfg_path) as f:
        sat_lut = build_saturation_lookup(json.load(f))

    # All images in a class share one resolution; derive it once so good images
    # (which have no GT) need not resolve their possibly-relative img_path.
    cls_hw = class_resolution(dataset_root, cls_name)

    scores, gt_masks, metas = [], [], []
    for rec in records:
        state = rec["defect_cls"]
        stem = rec["stem"]

        if state == GOOD:
            masks = []
        else:
            gt_dir = os.path.join(dataset_root, cls_name, "ground_truth", state, stem)
            masks = region_masks_for_image(gt_dir)

        npy = os.path.join(dump_root, cls_name, state, stem + ".npy")
        sc = load_map(npy, cls_hw)
        scores.append(sc)
        gt_masks.append(masks)
        metas.append(rec)

    return scores, gt_masks, metas, sat_lut


def image_auroc(metas, subset_state):
    """AUROC of good (neg) vs one anomaly state (pos), pooled over classes."""
    y, s = [], []
    for m in metas:
        if m["defect_cls"] == GOOD:
            y.append(0); s.append(m["image_score"])
        elif m["defect_cls"] == subset_state:
            y.append(1); s.append(m["image_score"])
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def build_curves(metas, all_scores, all_gt, merged_lut, fpr_limits):
    """Compute AU-sPRO for logical, structural, and their mean.

    For each subset we keep every image's pixels for the FPR denominator but
    only count regions whose image belongs to that subset as positives.
    """
    out = {}
    for subset in (LOGICAL, STRUCTURAL):
        gt_subset = []
        for i, rec in enumerate(metas):
            if rec["defect_cls"] == subset:
                gt_subset.append(all_gt[i])          # keep its regions
            else:
                gt_subset.append([])                 # good + other type: bg only
        res, _ = au_spro_from_scores(all_scores, gt_subset, merged_lut,
                                     fpr_limits=fpr_limits)
        out[subset] = res
    out["mean"] = {
        lim: np.mean([out[LOGICAL][lim], out[STRUCTURAL][lim]])
        for lim in fpr_limits
    }
    return out


def main():
    ap = argparse.ArgumentParser("MVTec-LOCO AU-sPRO evaluator")
    ap.add_argument("--dataset_root", required=True,
                    help="raw LOCO root, e.g. ./data/MMAD/MVTec-LOCO")
    ap.add_argument("--dump_root", required=True,
                    help="<save_path>/dump produced by --dump_maps")
    ap.add_argument("--fpr_limits", type=float, nargs="+", default=[0.05, 0.3])
    ap.add_argument("--tag", default="", help="label printed with the results")
    args = ap.parse_args()

    fpr_limits = tuple(args.fpr_limits)
    classes = sorted(
        d for d in os.listdir(args.dump_root)
        if os.path.isdir(os.path.join(args.dump_root, d))
    )

    all_scores, all_gt, all_metas = [], [], []
    sat_lut_by_cls = {}
    for cls_name in classes:
        sc, gt, metas, lut = evaluate_class(args.dataset_root, args.dump_root, cls_name)
        all_scores.extend(sc)
        all_gt.extend(gt)
        all_metas.extend(metas)
        sat_lut_by_cls[cls_name] = lut

    merged_lut = {}
    for lut in sat_lut_by_cls.values():
        merged_lut.update(lut)

    # Image-level AUROC (pooled over all classes).
    img_log = image_auroc(all_metas, LOGICAL)
    img_str = image_auroc(all_metas, STRUCTURAL)
    img_mean = np.nanmean([img_log, img_str])

    # AU-sPRO.
    spro = build_curves(all_metas, all_scores, all_gt, merged_lut, fpr_limits)

    print(f"\n===== LOCO evaluation {args.tag} =====")
    print(f"images: {len(all_metas)}  classes: {len(classes)}")
    print(f"\nImage-AUROC  mean {img_mean*100:.1f}  "
          f"(L:{img_log*100:.1f} / S:{img_str*100:.1f})")
    for lim in fpr_limits:
        m = spro['mean'][lim] * 100
        l = spro[LOGICAL][lim] * 100
        s = spro[STRUCTURAL][lim] * 100
        print(f"AU-sPRO@{lim:<4}  mean {m:.1f}  (L:{l:.1f} / S:{s:.1f})")

    # Machine-readable line for scripting.
    summary = {
        "tag": args.tag,
        "image_auroc": {"mean": img_mean, "logical": img_log, "structural": img_str},
        "au_spro": {str(l): spro_for_json(spro, l) for l in fpr_limits},
    }
    print("\nJSON " + json.dumps(summary))


def spro_for_json(spro, lim):
    return {
        "mean": spro["mean"][lim],
        "logical": spro[LOGICAL][lim],
        "structural": spro[STRUCTURAL][lim],
    }


if __name__ == "__main__":
    main()
