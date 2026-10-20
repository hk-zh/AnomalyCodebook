#!/usr/bin/env python3
"""Candidate sheets for the qualitative figure: the top --k images per dataset of the
random pool by our per-image pixel AP margin over the best baseline (same score and
area floor as tools/select_qualitative.py), written in the format
tools/make_qualitative.py render reads, one subset per dataset, columns labelled
"<dataset> <rank>". If the pool has uncalibrated maps (make_qualitative.py nocali),
they are carried along and their AP is printed next to ours.

Usage:
    python tools/qual_candidates.py --k 5
    python tools/make_qualitative.py render --inp results/qualitative/candidates/MPDD/s42.npz \\
        --layout compact --vmin_pct 80 --png /tmp/cand_MPDD.png
"""
import argparse
import os

import numpy as np

from select_qualitative import METHODS, pix_score


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pool", default="results/qualitative/pool/s42.npz")
	ap.add_argument("--out_dir", default="results/qualitative/candidates")
	ap.add_argument("--k", type=int, default=5)
	ap.add_argument("--min_area", type=float, default=0.002)
	args = ap.parse_args()
	root = os.path.dirname(args.pool)
	d = dict(np.load(args.pool, allow_pickle=False))
	base = {m: dict(np.load(os.path.join(root, m + ".npz"))) for m in METHODS}
	nc = os.path.join(root, "nocali.npz")
	if os.path.isfile(nc):
		base["nocali"] = dict(np.load(nc))
	n = len(d["label"])
	score = {"ours": np.array([pix_score(d["amap"][i], d["mask"][i], "ap") for i in range(n)])}
	for m in base:
		score[m] = base[m]["pix_score"] if "pix_score" in base[m] else \
			np.array([pix_score(base[m]["amap"][i], d["mask"][i], "ap") for i in range(n)])
	best = np.nanmax(np.stack([score[m] for m in METHODS]), axis=0)
	gain = score["ours"] - best
	for lab in dict.fromkeys(d["label"]):
		idx = np.array([i for i in range(n) if d["label"][i] == lab and not np.isnan(gain[i])
						and d["mask"][i].mean() >= args.min_area])
		top = idx[np.argsort(-gain[idx])][:args.k]
		names = np.array([f"{lab} {r + 1}" for r in range(len(top))])
		out = os.path.join(args.out_dir, lab)
		os.makedirs(out, exist_ok=True)
		sub = {k: v[top] for k, v in d.items()}
		sub["label"], sub["pix_score"] = names, score["ours"][top]
		np.savez_compressed(os.path.join(out, "s42.npz"), **sub)
		for m in base:
			np.savez_compressed(os.path.join(out, m + ".npz"), amap=base[m]["amap"][top], label=names,
								pix_score=score[m][top])
		print(f"{lab:12s} " + "  ".join(
			f"#{r + 1} ours {score['ours'][j]:.2f} best {best[j]:.2f}"
			+ (f" nocali {score['nocali'][j]:.2f}" if "nocali" in score else "") for r, j in enumerate(top)))


if __name__ == "__main__":
	main()
