# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Generate a DAPO-format meta.json for an MMAD-layout dataset (MVTec-LOCO or
# GoodsAD), which ship no meta.json. DAPO's Dataset reads
# meta.json[mode][cls] -> list of {img_path, mask_path, cls_name, anomaly,
# specie_name}, all paths relative to the dataset root.
#
#   * specie_name carries the state subdir (good / logical_anomalies /
#     structural_anomalies for LOCO; the defect folder for GoodsAD) so the
#     dump-based AU-sPRO evaluator can split logical vs structural.
#   * For LOCO, mask_path points at the first region PNG only. DAPO's internal
#     per-pixel metric is therefore NOT meaningful for LOCO (a logical image
#     has several regions); LOCO is scored with AU-sPRO from the dumped maps
#     against the raw multi-valued GT instead. Run DAPO with --metrics
#     image_level on LOCO. GoodsAD masks are single-file, so its pixel metric
#     is valid there.

import argparse
import json
import os

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
GOOD = "good"


def is_img(name):
    return name.lower().endswith(IMG_EXTS)


def loco_mask(root, cls, state, stem):
    """First region PNG for a LOCO image (relative path), or '' if none."""
    gt_dir = os.path.join(root, cls, "ground_truth", state, stem)
    if not os.path.isdir(gt_dir):
        return ""
    regions = sorted(n for n in os.listdir(gt_dir) if is_img(n))
    if not regions:
        return ""
    return os.path.join(cls, "ground_truth", state, stem, regions[0])


def goodsad_mask(root, cls, state, stem):
    """Single mask for a GoodsAD image (relative path), or '' if none."""
    gt_dir = os.path.join(root, cls, "ground_truth", state)
    for ext in IMG_EXTS:
        cand = os.path.join(cls, "ground_truth", state, stem + ext)
        if os.path.exists(os.path.join(root, cand)):
            return cand
    return ""


def build(root, dataset, split):
    mask_fn = loco_mask if dataset == "loco" else goodsad_mask
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, split))
    )
    meta = {split: {}}
    counts = {}
    for cls in classes:
        split_dir = os.path.join(root, cls, split)
        entries = []
        for state in sorted(os.listdir(split_dir)):
            state_dir = os.path.join(split_dir, state)
            if not os.path.isdir(state_dir):
                continue
            anomaly = 0 if state == GOOD else 1
            for name in sorted(n for n in os.listdir(state_dir) if is_img(n)):
                stem = os.path.splitext(name)[0]
                mask_path = "" if anomaly == 0 else mask_fn(root, cls, state, stem)
                entries.append({
                    "img_path": os.path.join(cls, split, state, name),
                    "mask_path": mask_path,
                    "cls_name": cls,
                    "anomaly": anomaly,
                    "specie_name": state,
                })
        meta[split][cls] = entries
        counts[cls] = (len(entries),
                       sum(1 for e in entries if e["anomaly"] == 1))
    return meta, counts


def main():
    ap = argparse.ArgumentParser("DAPO meta.json generator")
    ap.add_argument("--root", required=True, help="dataset root (MMAD layout)")
    ap.add_argument("--dataset", required=True, choices=["loco", "goodsad"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="defaults to <root>/meta.json")
    args = ap.parse_args()

    meta, counts = build(args.root, args.dataset, args.split)
    out = args.out or os.path.join(args.root, "meta.json")
    with open(out, "w") as f:
        json.dump(meta, f)

    total = sum(c[0] for c in counts.values())
    anom = sum(c[1] for c in counts.values())
    missing = 0
    for cls, entries in meta[args.split].items():
        for e in entries:
            if e["anomaly"] == 1 and not e["mask_path"]:
                missing += 1
    print(f"wrote {out}")
    print(f"classes: {len(counts)}  images: {total}  anomalous: {anom}")
    for cls, (n, a) in counts.items():
        print(f"  {cls:22s} {n:5d} imgs  {a:5d} anom")
    if missing:
        print(f"WARNING: {missing} anomalous images have no mask_path")


if __name__ == "__main__":
    main()
