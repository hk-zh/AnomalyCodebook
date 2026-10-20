#!/usr/bin/env python3
"""The paper's image-level numbers from test.py's own per-image scores.

regen.bsub runs test.py (default setting, 3x3 tile read-out) on every target with
--dump_image_scores. Each image_scores.npz holds, per image, the fused score
(Adopted), the map term (Map) and the global term; the whole-image global score
alone comes from the same fp32 file test.py loads, so Whole = 0.25 * Map + 0.75 *
G_whole. Prints what tools/readout_paper_numbers.py prints, but with test.py's map
term, and checks every recomputed Adopted AUROC/AP against test.py's log.

Usage:
    python tools/paper_numbers_from_dumps.py --root results/regen [--seeds 42 1 2 3]

--seeds sets the seeds every mean and standard deviation (sample, n-1) is taken
over; the paper reports all four since the 2026-10-10 revision.
"""
import argparse
import os
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DATASETS = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad", "bmad_brain", "bmad_liver", "bmad_resc", "loco"]
SEEDS = ["42", "1", "2", "3"]
W, T = 0.75, 0.1


def anomaly_prob(cos):
	z = np.where(np.isnan(cos), -np.inf, cos / T)
	z = z - z.max(axis=-1, keepdims=True)
	e = np.exp(z)
	return 1.0 - e[..., 0] / e.sum(axis=-1)


def per_class(y, cls, s):
	aucs = [100 * roc_auc_score(y[cls == k], s[cls == k]) for k in np.unique(cls)]
	aps = [100 * average_precision_score(y[cls == k], s[cls == k]) for k in np.unique(cls)]
	return np.mean(aucs), np.mean(aps)


def loco_split(paths):
	return np.array(["logical_anomalies" if "/logical_anomalies/" in p else
					 "structural_anomalies" if "/structural_anomalies/" in p else "good" for p in paths])


def metric(ds, d, s):
	y, cls = d["label"].astype(int), d["cls_name"]
	if ds == "loco":		# pooled good-vs-subset AUROC, as tools/loco_spro_eval.py
		dc = loco_split(d["img_path"])
		ls = [100 * roc_auc_score(y[(dc == "good") | (dc == sub)], s[(dc == "good") | (dc == sub)])
			  for sub in ("logical_anomalies", "structural_anomalies")]
		return np.mean(ls), np.nan, ls[0], ls[1]
	a, p = per_class(y, cls, s)
	return a, p, np.nan, np.nan


def logged(path):
	rows = [l for l in open(path, errors="ignore") if "| mean" in l]
	f = rows[-1].split("|")
	return float(f[6]), float(f[8])


def logged_pixel(path):
	rows = [l for l in open(path, errors="ignore") if "| mean" in l]
	f = rows[-1].split("|")
	return float(f[2]), float(f[5])


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--root", default="results/regen")
	ap.add_argument("--global_root", default="results/globalnative_fp32full/336")
	ap.add_argument("--seeds", nargs="+", default=SEEDS)
	args = ap.parse_args()

	R, madsim, missing = {}, {}, []
	for ds in DATASETS:
		g = np.load(os.path.join(args.global_root, ds + ".npz"))
		whole = dict(zip(g["img_path"].tolist(), ((anomaly_prob(g["g_cos"]) + anomaly_prob(g["g_cos_flip"])) / 2).tolist()))
		for seed in SEEDS:
			p = os.path.join(args.root, f"s{seed}", ds, "image_scores.npz")
			if not os.path.isfile(p):
				missing.append(f"s{seed}/{ds}")
				continue
			d = dict(np.load(p, allow_pickle=False))
			gw = np.array([whole[q] for q in d["img_path"]])
			s = {"Map": d["map_score"], "Whole": (1 - W) * d["map_score"] + W * gw, "Adopted": d["score"]}
			for name, v in s.items():
				R.setdefault(ds, {}).setdefault(name, {})[seed] = metric(ds, d, v)
			if ds != "loco":
				la, lp = logged(os.path.join(args.root, f"s{seed}", ds, "log.txt"))
				a, pp = R[ds]["Adopted"][seed][:2]
				if abs(a - la) > 0.051 or abs(pp - lp) > 0.051:
					print(f"[check] MISMATCH s{seed}/{ds}: recomputed {a:.2f}/{pp:.2f} vs log {la}/{lp}")
			if ds == "mad_sim" and seed in args.seeds:
				y, cls = d["label"].astype(int), d["cls_name"]
				dc = np.array([q.split("/")[-2] for q in d["img_path"]])
				for name, v in s.items():
					for t in ("Missing", "Burrs", "Stains"):
						aucs = [100 * roc_auc_score(y[(cls == k) & ((dc == "good") | (dc == t))], v[(cls == k) & ((dc == "good") | (dc == t))])
								for k in np.unique(cls) if len(np.unique(y[(cls == k) & ((dc == "good") | (dc == t))])) == 2]
						madsim.setdefault(name, {}).setdefault(seed, {})[t] = np.mean(aucs)
	if missing:
		print("missing dumps:", " ".join(missing))

	three = args.seeds
	n = len(three)
	def ms(v): v = np.asarray(v, float); return f"{v.mean():.1f}+-{v.std(ddof=1):.1f}"
	print(f"== 0. {n}-seed pixel AUROC / AUPRO (test.py logs, rounded per seed)")
	for ds in DATASETS[:-1]:
		px = [logged_pixel(os.path.join(args.root, f"s{s}", ds, "log.txt")) for s in three]
		print(f"{ds:11s} Px {ms([p[0] for p in px])}  PRO {ms([p[1] for p in px])}   per seed {[p for p in px]}")
	print(f"\n== 1. {n}-seed image AUROC / AP (Adopted), Map and Whole AUROC")
	for ds in R:
		if not all(s in R[ds]["Adopted"] for s in three):
			continue
		a = R[ds]["Adopted"]
		print(f"{ds:11s} Adopted {ms([a[s][0] for s in three])} / {ms([a[s][1] for s in three]) if ds != 'loco' else '-'}"
			  f"   Map {np.mean([R[ds]['Map'][s][0] for s in three]):.1f}  Whole {np.mean([R[ds]['Whole'][s][0] for s in three]):.1f}"
			  + (f"   L {np.mean([a[s][2] for s in three]):.1f} S {np.mean([a[s][3] for s in three]):.1f}" if ds == "loco" else ""))
	print(f"\n== 2. Delta (Adopted - Map) per seed ({', '.join(three)}) and 4-seed spread")
	for ds in R:
		if not all(s in R[ds]["Adopted"] for s in SEEDS):
			continue
		dl = [R[ds]["Adopted"][s][0] - R[ds]["Map"][s][0] for s in three]
		sp = {n: max(R[ds][n][s][0] for s in SEEDS) - min(R[ds][n][s][0] for s in SEEDS) for n in ("Map", "Adopted")}
		print(f"{ds:11s} Delta {min(dl):+.1f} .. {max(dl):+.1f}   spread Map {sp['Map']:.1f}  Adopted {sp['Adopted']:.1f}")
	print("\n== 3. seed 42 (transfer tables)")
	bm = [ds for ds in ("bmad_brain", "bmad_liver", "bmad_resc") if ds in R and "42" in R[ds]["Adopted"]]
	for ds in bm:
		print(f"{ds:11s} AUROC {R[ds]['Adopted']['42'][0]:.1f}  AP {R[ds]['Adopted']['42'][1]:.1f}")
	if len(bm) == 3:
		print(f"BMAD mean   AUROC {np.mean([R[ds]['Adopted']['42'][0] for ds in bm]):.1f}  AP {np.mean([R[ds]['Adopted']['42'][1] for ds in bm]):.1f}")
		dl = [np.mean([R[ds]['Adopted'][s][0] - R[ds]['Map'][s][0] for ds in bm]) for s in three]
		print(f"BMAD mean   Delta per seed {[round(x, 1) for x in dl]}   {n}-seed AUROC {ms([np.mean([R[ds]['Adopted'][s][0] for ds in bm]) for s in three])}")
	if "loco" in R and "42" in R["loco"]["Adopted"]:
		m, _, L, S = R["loco"]["Adopted"]["42"]
		print(f"loco        mean {m:.1f}  L {L:.1f}  S {S:.1f}")
	if madsim:
		print(f"\n== 4. MAD-Sim per defect type, {n}-seed mean")
		for name in madsim:
			print(f"{name:8s} " + "  ".join(f"{t} {np.mean([madsim[name][s][t] for s in three]):.1f}" for t in ("Missing", "Burrs", "Stains")))


if __name__ == "__main__":
	main()
