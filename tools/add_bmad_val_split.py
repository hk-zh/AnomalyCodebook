# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# Index the BMAD validation folders into each subset's meta.json under a
# uniform "valid" key, so hyperparameters (e.g. the top-k pooling ratio)
# can be tuned off-test.
#
# The shipped BMAD folders are inconsistent: the split dir is "valid" for
# BraTS/Liver but "val" for RESC, and the anomalous dir is "Ungood" for
# BraTS but "ungood" for Liver/RESC. We resolve both case-insensitively.

import json
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "BMAD")

# subset dir -> cls_name used by the prompts / codebook
SUBSETS = {
    "BraTS2021_slice": "brain",
    "Liver": "liver",
    "RESC": "resc",
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_dir(parent, *candidates):
    """Case-insensitive lookup of a child directory."""
    if not os.path.isdir(parent):
        return None
    listing = {name.lower(): name for name in os.listdir(parent)}
    for cand in candidates:
        hit = listing.get(cand.lower())
        if hit and os.path.isdir(os.path.join(parent, hit)):
            return os.path.join(parent, hit)
    return None


def index_split(subset_root, cls_name):
    val_dir = find_dir(subset_root, "valid", "val")
    if val_dir is None:
        raise RuntimeError(f"no valid/val dir under {subset_root}")

    entries = []
    for state, anomaly in (("good", 0), ("ungood", 1)):
        state_dir = find_dir(val_dir, state)
        if state_dir is None:
            print(f"  [warn] no '{state}' dir in {val_dir}, skipping")
            continue

        img_dir = find_dir(state_dir, "img") or state_dir
        label_dir = find_dir(state_dir, "label")

        names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(IMG_EXTS))
        for name in names:
            img_path = os.path.relpath(os.path.join(img_dir, name), subset_root)

            # Normal images carry no mask, matching the convention in the
            # shipped test split (mask_path == "").
            mask_path = ""
            if anomaly == 1 and label_dir is not None:
                cand = os.path.join(label_dir, name)
                if not os.path.exists(cand):
                    stem = os.path.splitext(name)[0]
                    for ext in IMG_EXTS:
                        alt = os.path.join(label_dir, stem + ext)
                        if os.path.exists(alt):
                            cand = alt
                            break
                if os.path.exists(cand):
                    mask_path = os.path.relpath(cand, subset_root)
                else:
                    print(f"  [warn] no mask for {img_path}")

            entries.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "cls_name": cls_name,
                "specie_name": "",
                "anomaly": anomaly,
            })

    return entries


def main():
    dry_run = "--apply" not in sys.argv
    for subset, cls_name in SUBSETS.items():
        subset_root = os.path.join(ROOT, subset)
        meta_path = os.path.join(subset_root, "meta.json")
        if not os.path.isfile(meta_path):
            print(f"[skip] {meta_path} missing")
            continue

        entries = index_split(subset_root, cls_name)
        n_anom = sum(e["anomaly"] for e in entries)
        n_mask = sum(1 for e in entries if e["mask_path"])
        print(f"{subset:20s} valid: {len(entries):4d} imgs "
              f"({n_anom} anomalous, {len(entries) - n_anom} normal, {n_mask} masks)")

        if dry_run:
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if "valid" not in meta:
            shutil.copy2(meta_path, meta_path + ".bak")

        meta["valid"] = {cls_name: entries}
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"  -> wrote 'valid' split to {meta_path}")

    if dry_run:
        print("\ndry run; re-run with --apply to write meta.json")


if __name__ == "__main__":
    main()
