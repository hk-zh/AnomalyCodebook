#!/usr/bin/env python3
"""Full-metric comparison of image-score read-outs over seeds.

The read-outs compared here differ only in the image score, so pixel AUROC,
AUPRO and LOCO AU-sPRO are the same for every row and come from the test.py
logs; image AUROC and AP are recomputed per read-out from the per-image scores
of tools/image_score_study.py (joined with the global scores, as in
tools/image_score_fusion.py). Image metrics are per-class means as in test.py,
except LOCO, which pools classes like tools/loco_spro_eval.py (the evaluator of
the paper's LOCO table). The "current" row is checked against the logs.

Usage:
    python tools/full_results_table.py --runs stab42:42 stab1:1 stab2:2 \\
        --variant "current=M@0.001/s4" --variant "frozen G=M+0.75*G336f@0.05"
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_score_fusion import candidates, load_scores, per_class_metric   # noqa: E402

MAIN = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad"]
BMAD = ["bmad_brain", "bmad_liver", "bmad_resc"]
LOCO_SUBSETS = (("L", "logical_anomalies"), ("S", "structural_anomalies"))


def log_row(path):
	"""-> (px, pro, img, ap) from the '| mean' row of a test.py log; None while the run is unfinished."""
	if not os.path.isfile(path):
		return None
	rows = [l for l in open(path, errors="ignore") if "| mean" in l]
	if not rows:
		return None
	f = rows[-1].split("|")
	return tuple(float(f[i]) for i in (2, 5, 6, 8))


def loco_json(tag):
	"""The last tools/loco_spro_eval.py JSON line printed for this tag in bsub_log/."""
	hit = None
	for p in sorted(glob.glob("bsub_log/*.stdout"), key=os.path.getmtime):
		for l in open(p, errors="ignore"):
			if l.startswith("JSON ") and f'"tag": "{tag}"' in l:
				hit = json.loads(l[5:])
	return hit


def loco_image(d, score):
	"""Pooled good-vs-subset AUROC per LOCO subset, as in tools/loco_spro_eval.py."""
	y, dc = d["label"].astype(int), d["defect_cls"]
	out = {}
	for k, sub in LOCO_SUBSETS:
		m = (dc == "good") | (dc == sub)
		out[k] = 100 * roc_auc_score(y[m], score[m])
	return out


def fmt(v):
	"""mean±std over the seeds that have a value; '*' marks cells missing some seeds."""
	v = np.asarray(v, dtype=float)
	ok = v[~np.isnan(v)]
	if not len(ok):
		return "n/a"
	return (f"{ok.mean():.1f}" + (f"±{ok.std(ddof=1):.1f}" if len(ok) > 1 else "")
			+ ("*" if len(ok) < len(v) else ""))


def table(title, header, rows):
	print(f"\n### {title}\n")
	print("| " + " | ".join(header) + " |")
	print("|" + "---|" * len(header))
	for r in rows:
		print("| " + " | ".join(r) + " |")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--runs", nargs="+", default=["stab42:42", "stab1:1", "stab2:2"],
					help="tag:seed; tag names the checkpoint, seed the results/multiseed/s<seed> dir")
	ap.add_argument("--variant", action="append", required=True,
					help="label=candidate name of tools/image_score_fusion.py")
	ap.add_argument("--root", default="results/imgscore")
	ap.add_argument("--globaltok_root", default="results/globaltok")
	ap.add_argument("--native_root", default="results/globalnative")
	ap.add_argument("--head_root", default="results/globalhead")
	ap.add_argument("--wise_root", default="results/globalhead_wise")
	ap.add_argument("--main_log", default="results/multiseed/s{seed}/{ds}/log.txt",
					help="test.py log of the industrial targets; cleanrun.bsub writes results/stage2_final_{tag}/{ds}/log.txt")
	args = ap.parse_args()
	variants = [v.split("=", 1) for v in args.variant]
	runs = [r.split(":") for r in args.runs]

	# pix[ds] -> list over seeds of (px, pro, img_log, ap_log); img[label][ds] -> list of (auroc, ap)
	pix = {ds: [] for ds in MAIN + BMAD}
	img = {lab: {ds: [] for ds in MAIN + BMAD} for lab, _ in variants}
	loco_pix, loco_img = [], {lab: [] for lab, _ in variants}
	for tag, seed in runs:
		for ds in MAIN + BMAD:
			log = (args.main_log.format(seed=seed, tag=tag, ds=ds) if ds in MAIN
				   else f"results/stage2_final_{tag}/{ds}/log.txt")
			d = load_scores(args, tag, ds)
			row = log_row(log)
			if row is None:		# pixel metrics need the log; image metrics only the scores
				print(f"[missing] {tag}/{ds} log")
			pix[ds].append(row if row is not None else (np.nan,) * 4)
			if d is None:
				print(f"[missing] {tag}/{ds} scores")
				for lab, _ in variants:
					img[lab][ds].append((np.nan, np.nan))
				continue
			c = candidates(d)
			for lab, name in variants:
				img[lab][ds].append(per_class_metric(d["label"].astype(int), d["cls_name"], c[name]))
		j = loco_json(f"final_{tag}")
		d = load_scores(args, tag, "loco")
		if j is None or d is None:
			print(f"[missing] {tag}/loco")
			loco_pix.append(None)
			for lab, _ in variants:
				loco_img[lab].append(None)
			continue
		loco_pix.append(j)
		c = candidates(d)
		for lab, name in variants:
			loco_img[lab].append(loco_image(d, c[name]))

	# the first variant must be the map-only read-out test.py logs: check it
	lab0, _ = variants[0]
	for ds in MAIN + BMAD:
		dev = [max(abs(a - p[2]), abs(b - p[3])) for (a, b), p in zip(img[lab0][ds], pix[ds])]
		print(f"[check] {ds:<11} max |recomputed - logged| image AUROC/AP for '{lab0}': {np.nanmax(dev):.2f}")
	for li, j in zip(loco_img[lab0], loco_pix):
		if j is not None:
			print(f"[check] loco        recomputed L/S {li['L']:.1f}/{li['S']:.1f} vs logged "
				  f"{100 * j['image_auroc']['logical']:.1f}/{100 * j['image_auroc']['structural']:.1f}")

	n = len(runs)
	names = {"visa": "VisA", "mpdd": "MPDD", "mad_sim": "MAD-Sim", "mad_real": "MAD-Real",
			 "real_iad": "Real-IAD", "bmad_brain": "Brain", "bmad_liver": "Liver", "bmad_resc": "RESC"}

	def ds_cells(lab, ds):
		p = np.array(pix[ds])
		a = np.array(img[lab][ds])
		return [fmt(p[:, 0]), fmt(p[:, 1]), fmt(a[:, 0]), fmt(a[:, 1])]

	hdr = ["Read-out"] + [f"{names[ds]} {m}" for ds in MAIN for m in ("Px", "PRO", "Img", "AP")]
	table(f"Industrial targets (mean±std over {n} seeds)", hdr,
		  [[lab] + sum((ds_cells(lab, ds) for ds in MAIN), []) for lab, _ in variants])

	def bmad_mean(lab):
		p = np.nanmean([np.array(pix[ds]) for ds in BMAD], axis=0)
		a = np.nanmean([np.array(img[lab][ds]) for ds in BMAD], axis=0)
		return [fmt(p[:, 0]), fmt(p[:, 1]), fmt(a[:, 0]), fmt(a[:, 1])]

	hdr = ["Read-out"] + [f"{names[ds]} {m}" for ds in BMAD for m in ("Px", "PRO", "Img", "AP")] \
		+ [f"Mean {m}" for m in ("Px", "PRO", "Img", "AP")]
	table(f"BMAD (mean±std over {n} seeds)", hdr,
		  [[lab] + sum((ds_cells(lab, ds) for ds in BMAD), []) + bmad_mean(lab) for lab, _ in variants])

	ok = [i for i, j in enumerate(loco_pix) if j is not None]

	def spro(fpr, k):
		key = {"Mean": "mean", "L": "logical", "S": "structural"}[k]
		return fmt([100 * loco_pix[i]["au_spro"][fpr][key] for i in ok])

	rows = []
	for lab, _ in variants:
		li = [loco_img[lab][i] for i in ok]
		rows.append([lab, fmt([(x["L"] + x["S"]) / 2 for x in li]), fmt([x["L"] for x in li]),
					 fmt([x["S"] for x in li])] + [spro(f, k) for f in ("0.05", "0.3") for k in ("Mean", "L", "S")])
	hdr = ["Read-out", "Img Mean", "Img L", "Img S", "sPRO.05 Mean", "sPRO.05 L", "sPRO.05 S",
		   "sPRO.30 Mean", "sPRO.30 L", "sPRO.30 S"]
	table(f"MVTec-LOCO (mean±std over {len(ok)} seeds)", hdr, rows)


if __name__ == "__main__":
	main()
