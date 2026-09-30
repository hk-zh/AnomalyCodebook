#!/usr/bin/env python3
"""Score a dumped set of anomaly maps with the same evaluator we use for ours.

The transfer sections compare against baselines that were never published on
BMAD or MVTec-LOCO, so their numbers have to be produced here. Running each
baseline's own forward pass in its own repo and then scoring every method with
this one script keeps the metric code identical across methods, which is the
only part of the protocol that must not differ.

Expects the dump layout that test.py --dump_maps writes:

    <dump_root>/<cls_name>/<defect_cls>/<stem>.npy     float16 anomaly map
    <dump_root>/<cls_name>/scores.json                 per-image records

Ground-truth masks are read from the dataset itself rather than from the dump,
so a baseline cannot accidentally be scored against its own resized masks.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import BMADDataset, MVTecLOCODataset          # noqa: E402
from test import evaluate_metrics                          # noqa: E402


def build_dataset(name, data_path, img_size, split):
    target_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    if name == "bmad":
        return BMADDataset(root=data_path, transform=preprocess,
                           target_transform=target_transform, mode=split)
    if name == "loco":
        return MVTecLOCODataset(root=data_path, transform=preprocess,
                                target_transform=target_transform, mode=split)
    raise ValueError("unsupported dataset for dump eval: %s" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", required=True)
    ap.add_argument("--dataset", required=True, choices=["bmad", "loco"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval_workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ds = build_dataset(args.dataset, args.data_path, args.image_size, args.split)

    # Index the dump by (cls_name, stem) so it is matched against the dataset
    # order rather than assumed to share it.
    dumped = {}
    for cls_name in sorted(os.listdir(args.dump_root)):
        sj = os.path.join(args.dump_root, cls_name, "scores.json")
        if not os.path.isfile(sj):
            continue
        with open(sj) as fh:
            for rec in json.load(fh):
                npy = os.path.join(args.dump_root, cls_name,
                                   rec["defect_cls"], rec["stem"] + ".npy")
                dumped[(cls_name, rec["stem"])] = (npy, rec["image_score"])

    results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [],
               "gt_sp": [], "pr_sp": []}
    missing = 0

    for i in range(len(ds)):
        item = ds[i]
        cls_name = item["cls_name"]
        stem = os.path.splitext(os.path.basename(item["img_path"]))[0]
        key = (cls_name, stem)
        if key not in dumped:
            missing += 1
            continue
        npy, score = dumped[key]
        amap = np.load(npy).astype(np.float32)

        mask = item["img_mask_b"]
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        mask = np.squeeze(mask)
        mask = (mask > 0.5).astype(np.uint8)

        if amap.shape != mask.shape:
            amap = np.array(
                torch.nn.functional.interpolate(
                    torch.from_numpy(amap)[None, None], size=mask.shape,
                    mode="bilinear", align_corners=False
                )[0, 0]
            )

        results["cls_names"].append(cls_name)
        results["imgs_masks"].append(mask)
        results["anomaly_maps"].append(amap)
        results["gt_sp"].append(int(item["anomaly"]))
        results["pr_sp"].append(float(score))

    if missing:
        sys.stderr.write(
            "WARNING: %d of %d dataset images had no dumped map\n" % (missing, len(ds)))
    if not results["cls_names"]:
        raise SystemExit("no images matched between dataset and dump")

    print("scored %d images over %d classes  %s" % (
        len(results["cls_names"]), len(set(results["cls_names"])), args.tag))
    print(evaluate_metrics(results=results,
                           obj_list=sorted(set(results["cls_names"])),
                           num_workers=args.eval_workers))


if __name__ == "__main__":
    main()
