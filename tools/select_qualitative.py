#!/usr/bin/env python3
"""Pick qualitative examples by a stated rule, from a random pool.

The pool (tools/make_qualitative.py compute --per_dataset N) holds N random
anomalous test images per dataset with our maps; tools/qual_baselines.py adds the
baselines' maps for the same images. For every image and method this computes the
per-image pixel AP (or AUROC) against the ground truth over the interior (a 12 px border
is left out for every method, because our smoothing zero-pads it, which would
otherwise favour us). Per dataset it selects the image where our score exceeds
the best baseline's by the most (or the --rank-th most, a hand pick among the top
candidates that the caption has to state), among images whose defect covers at least
--min_area of the image, and writes that subset in the format tools/make_qualitative.py render reads. It
also prints how often we beat the best baseline over the whole pool, the context
the selected examples should be read against. If the pool has our uncalibrated maps
(tools/make_qualitative.py nocali), they are carried into the subset and the share of
images where calibration raises our AP is printed too.

Usage:
    python tools/select_qualitative.py --pool results/qualitative/pool/s42.npz \\
        --out_dir results/qualitative/selected
    python tools/select_qualitative.py --rank MPDD=2 "Brain MRI=3"   # hand pick, see tools/qual_candidates.py
"""
import argparse
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = ["anomalyclip", "dapo", "multiads"]
B = 12


def pix_score(amap, mask, metric):
	"""Per-image pixel AP (default: rewards the highest values lying on the defect) or AUROC."""
	m = mask[B:-B, B:-B].ravel()
	if m.min() == m.max():
		return np.nan
	f = average_precision_score if metric == "ap" else roc_auc_score
	return f(m, amap.astype(np.float32)[B:-B, B:-B].ravel())


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pool", default="results/qualitative/pool/s42.npz")
	ap.add_argument("--out_dir", default="results/qualitative/selected")
	ap.add_argument("--metric", choices=["ap", "auroc"], default="ap")
	ap.add_argument("--min_ours", type=float, default=0.0, help="minimum per-image score of ours for a candidate")
	ap.add_argument("--min_area", type=float, default=0.002, help="minimum defect area (fraction of the image), so it is visible at figure size")
	ap.add_argument("--rank", nargs="*", default=[], metavar="DATASET=R", help="take the image with the R-th largest margin for these datasets instead of the largest")
	args = ap.parse_args()
	rank = {k: int(v) for k, v in (r.rsplit("=", 1) for r in args.rank)}
	d = dict(np.load(args.pool, allow_pickle=False))
	base = {m: dict(np.load(os.path.join(os.path.dirname(args.pool), m + ".npz"))) for m in METHODS}
	nc = os.path.join(os.path.dirname(args.pool), "nocali.npz")
	if os.path.isfile(nc):
		base["nocali"] = dict(np.load(nc))
	n = len(d["label"])
	auc = {"ours": np.array([pix_score(d["amap"][i], d["mask"][i], args.metric) for i in range(n)])}
	for m in base:
		assert list(base[m]["label"]) == list(d["label"])
		auc[m] = np.array([pix_score(base[m]["amap"][i], d["mask"][i], args.metric) for i in range(n)])
	best_base = np.nanmax(np.stack([auc[m] for m in METHODS]), axis=0)
	gain = auc["ours"] - best_base

	labels = list(dict.fromkeys(d["label"]))
	pick = []
	cali = "nocali" in base
	print(f"{'dataset':12s} {'pool':>4s} {'ours>best':>9s} {'mean ours':>9s} " + " ".join(f"{m:>11s}" for m in METHODS)
		  + (f" {'nocali':>8s} {'cali>noc':>8s}" if cali else "") + "   selected: ours / best baseline")
	for lab in labels:
		idx = np.array([i for i in range(n) if d["label"][i] == lab and not np.isnan(gain[i])])
		area = np.array([d["mask"][i].mean() for i in idx])
		ok = idx[(auc["ours"][idx] >= args.min_ours) & (area >= args.min_area)]
		cand = ok if len(ok) else idx[area >= args.min_area] if (area >= args.min_area).any() else idx
		order = cand[np.argsort(-gain[cand], kind="stable")]
		r = min(rank.get(lab, 1), len(order))
		j = int(order[r - 1])
		pick.append(j)
		print(f"{lab:12s} {len(idx):4d} {np.mean(gain[idx] > 0):9.0%} {np.nanmean(auc['ours'][idx]):9.3f} "
			  + " ".join(f"{np.nanmean(auc[m][idx]):11.3f}" for m in METHODS)
			  + (f" {np.nanmean(auc['nocali'][idx]):8.3f} {np.mean(auc['ours'][idx] > auc['nocali'][idx]):8.0%}" if cali else "")
			  + f"   {auc['ours'][j]:.3f} / {best_base[j]:.3f}  (rank {r}, {os.path.basename(d['img_path'][j])})")
	os.makedirs(args.out_dir, exist_ok=True)
	sub = {k: v[pick] for k, v in d.items()}
	sub["pix_score"] = auc["ours"][pick]
	np.savez_compressed(os.path.join(args.out_dir, "s42.npz"), **sub)
	for m in base:
		np.savez_compressed(os.path.join(args.out_dir, m + ".npz"), amap=base[m]["amap"][pick],
							label=base[m]["label"][pick], pix_score=auc[m][pick])
	print(f"[saved] {args.out_dir}: {len(pick)} examples")


if __name__ == "__main__":
	main()
