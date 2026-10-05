#!/usr/bin/env python3
"""Emit per-class LaTeX tables for the appendix from test.py result logs.

The logs end with a markdown table (objects x metrics) written by
evaluate_metrics(). This turns each one into a booktabs tabular, keeping only
the four metrics the paper reports and bolding the mean row.

The image columns of the logs are the old single-rho map read-out. With
--readout_b they are recomputed for the adopted read-out instead (multi-scale
top-k mean of the map, fused with 0.75 * frozen CLIP G at 336 px, T=0.1,
flip-averaged and averaged with its most anomalous 3x3 tile) from the per-image
scores of tools/image_score_study.py, the fp32 global scores of
tools/global_native_scores.py and the tile scores of
tools/global_crop_scores.py; pixel columns always come
from the logs, which the read-out does not affect.

Usage:
    python tools/make_perclass_tex.py --readout_b > latex/sec/7_appendix_perclass.tex
"""

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# (label, path to log.txt) in the order the appendix should present them.
RUNS = [
	("VisA",           "results/stage2_final_stab42/visa/log.txt"),
	("MPDD",           "results/stage2_final_stab42/mpdd/log.txt"),
	("MAD-Sim",        "results/stage2_final_stab42/mad_sim/log.txt"),
	("MAD-Real",       "results/stage2_final_stab42/mad_real/log.txt"),
	("Real-IAD",       "results/stage2_final_stab42/real_iad/log.txt"),
	("BMAD brain MRI", "results/stage2_final_stab42/bmad_brain/log.txt"),
	("BMAD liver CT",  "results/stage2_final_stab42/bmad_liver/log.txt"),
	("BMAD RESC OCT",  "results/stage2_final_stab42/bmad_resc/log.txt"),
	("MVTec-LOCO",     "results/stage2_final_stab42/loco/log.txt"),
]

# markdown column -> (paper header, keep?)
KEEP = [("auroc_px", "Px AUROC"), ("aupro", "AUPRO"), ("auroc_sp", "Img AUROC"), ("ap_sp", "Img AP")]


def parse(path):
	"""Return (header, rows) from the trailing markdown table of a log file."""
	rows = []
	header = None
	with open(path) as f:
		for line in f:
			line = line.strip()
			if not line.startswith("|"):
				continue
			cells = [c.strip() for c in line.strip("|").split("|")]
			if cells[0].startswith(":") or set(cells[0]) <= set("-: "):
				continue
			if cells[0] == "objects":
				header = cells
				rows = []
			elif header is not None:
				rows.append(cells)
	return header, rows


def esc(name):
	return name.replace("_", r"\_")


def anomaly_prob(cos, T):
	"""cos [..., C] (NaN past a product's class count), class 0 = normal."""
	z = np.where(np.isnan(cos), -np.inf, cos / T)
	z = z - z.max(axis=-1, keepdims=True)
	e = np.exp(z)
	return 1.0 - e[..., 0] / e.sum(axis=-1)


def readout_b(path, args):
	"""{class: (img AUROC, img AP)} of the adopted read-out for the run of this log."""
	ds = os.path.basename(os.path.dirname(path))
	d = np.load(os.path.join(args.score_root, ds + ".npz"))
	g = np.load(os.path.join(args.global_root, ds + ".npz"))
	assert (g["img_path"] == d["img_path"]).all(), f"image order differs: {ds}"
	ms = d["map_topk"][:, list(d["sigmas"]).index(4.0), :].mean(axis=1)
	glob = (anomaly_prob(g["g_cos"], 0.1) + anomaly_prob(g["g_cos_flip"], 0.1)) / 2
	if args.crop_root:		# most anomalous of the 3x3 tiles, averaged in (tools/global_crop_scores.py)
		c = np.load(os.path.join(args.crop_root, ds + ".npz"))
		assert (c["img_path"] == d["img_path"]).all(), f"image order differs: {ds} tiles"
		tile = (anomaly_prob(c["g_crop_cos"], 0.1) + anomaly_prob(c["g_crop_cos_flip"], 0.1)) / 2
		glob = (glob + tile.max(axis=1)) / 2
	score = 0.25 * ms + 0.75 * glob
	y, cls = d["label"].astype(int), d["cls_name"]
	return {c: (100 * roc_auc_score(y[cls == c], score[cls == c]),
				100 * average_precision_score(y[cls == c], score[cls == c])) for c in sorted(set(cls))}


def emit(label, path, args):
	header, rows = parse(path)
	if header is None:
		print(f"% MISSING: {path}", file=sys.stderr)
		return
	idx = {h: header.index(h) for h, _ in KEEP if h in header}
	cols = [(h, disp) for h, disp in KEEP if h in idx]
	if args.readout_b:		# replace the logged image columns, mean row included
		img = readout_b(path, args)
		assert {c[0] for c in rows if c[0] != "mean"} == set(img), f"class names differ: {path}"
		img["mean"] = tuple(np.mean([v[i] for v in img.values()]) for i in (0, 1))
		for cells in rows:
			cells[idx["auroc_sp"]], cells[idx["ap_sp"]] = (f"{v:.1f}" for v in img[cells[0]])
		readout = "adopted read-out of Eq.~\\mref{eq:fuse}"
	else:
		readout = r"$\rho{=}0.001$"

	print(r"\begin{table}[t]")
	print(r"\centering")
	print(r"\footnotesize")
	print(rf"\caption{{Per-class results on {label} (seed~42, $K{{=}}150$, epoch~1, "
		  rf"{readout}).}}")
	print(rf"\label{{tab:perclass-{os.path.basename(os.path.dirname(path))}}}")
	print(r"\begin{tabular}{l" + "c" * len(cols) + "}")
	print(r"\toprule")
	print("Class & " + " & ".join(d for _, d in cols) + r" \\")
	print(r"\midrule")
	for cells in rows:
		name = cells[0]
		vals = [f"{float(cells[idx[h]]):.1f}" for h, _ in cols]		# the logs drop a trailing .0
		if name == "mean":
			print(r"\midrule")
			print(r"\textbf{Mean} & " + " & ".join(rf"\textbf{{{v}}}" for v in vals) + r" \\")
		else:
			print(f"{esc(name)} & " + " & ".join(vals) + r" \\")
	print(r"\bottomrule")
	print(r"\end{tabular}")
	print(r"\end{table}")
	print()


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--readout_b", action="store_true", help="image columns from the adopted read-out")
	ap.add_argument("--score_root", default="results/imgscore/stab42")
	ap.add_argument("--global_root", default="results/globalnative_fp32full/336")
	ap.add_argument("--crop_root", default="results/globalcrop/g3", help="3x3 tile scores of the adopted read-out; '' for read-out B without tiles")
	args = ap.parse_args()
	print("% Generated by tools/make_perclass_tex.py" + (" --readout_b" if args.readout_b else "")
		  + "; do not edit by hand.")
	print()
	for label, path in RUNS:
		if os.path.exists(path):
			emit(label, path, args)
		else:
			print(f"% MISSING RUN: {path}")
