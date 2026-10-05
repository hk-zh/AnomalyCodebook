#!/usr/bin/env python3
"""Every image-level number the paper quotes for the adopted read-out.

Read-outs (tools/image_score_fusion.py names): Single = M@0.001/s4, Map =
MS/s4, B = MS+0.75*G336f@0.1 (whole image only), Adopted =
MS+0.75*G336fT3@0.1 (whole image and its most anomalous 3x3 tile). Prints
  1. 3-seed mean +- std, image AUROC and AP, per dataset (Table 1, Supp. MAD table)
  2. per-seed Delta (Adopted - Map) and 4-seed spreads (Supp. read-out table)
  3. seed-42 values for the transfer tables (BMAD, MVTec-LOCO L/S)
  4. MAD-Sim per defect type, per-class mean AUROC of each type against the normal
     images, 3-seed mean (Supp. read-out section)

Usage:
    python tools/readout_paper_numbers.py
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_results_table import loco_image                        # noqa: E402
from image_score_fusion import candidates, load_scores, per_class_metric, anomaly_prob   # noqa: E402

DATASETS = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad", "bmad_brain", "bmad_liver", "bmad_resc", "loco"]
READOUTS = {"Single": "M@0.001/s4", "Map": "MS/s4", "B": "MS+0.75*G336f@0.1", "Adopted": "MS+0.75*G336fT3@0.1"}


def metric(ds, d, s):
	"""(AUROC, AP) per-class mean; MVTec-LOCO: mean of the pooled L/S AUROCs and (L, S)."""
	if ds == "loco":
		li = loco_image(d, s)
		return (li["L"] + li["S"]) / 2, np.nan, li["L"], li["S"]
	a, p = per_class_metric(d["label"].astype(int), d["cls_name"], s)
	return a, p, np.nan, np.nan


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--root", default="results/imgscore")
	ap.add_argument("--native_root", default="results/globalnative_fp32full")
	ap.add_argument("--crop_root", default="results/globalcrop")
	ap.add_argument("--globaltok_root", default="")
	ap.add_argument("--head_root", default="")
	ap.add_argument("--wise_root", default="")
	args = ap.parse_args()
	seeds = ["stab42", "stab1", "stab2", "stab3"]

	R = {}		# R[ds][readout][seed] = (auroc, ap, L, S)
	madsim = {}	# madsim[readout][seed][type] = per-class mean AUROC vs good
	for ds in DATASETS:
		for seed in seeds:
			d = load_scores(args, seed, ds)
			c = candidates(d)
			c["G whole"] = c["G336f@0.1"]
			c["G whole+tile"] = c["G336fT3@0.1"]
			for name, key in READOUTS.items():
				R.setdefault(ds, {}).setdefault(name, {})[seed] = metric(ds, d, c[key])
			if ds == "mad_sim" and seed != "stab3":
				y, cls, dc = d["label"].astype(int), d["cls_name"], d["defect_cls"]
				for name, key in list(READOUTS.items()) + [("G whole", "G whole"), ("G whole+tile", "G whole+tile")]:
					for t in ("Missing", "Burrs", "Stains"):
						aucs = []
						for k in np.unique(cls):
							m = (cls == k) & ((dc == "good") | (dc == t))
							if len(np.unique(y[m])) == 2:
								aucs.append(100 * roc_auc_score(y[m], c[key][m]))
						madsim.setdefault(name, {}).setdefault(seed, {})[t] = np.mean(aucs)

	three = seeds[:3]
	def ms(v): v = np.asarray(v, float); return f"{v.mean():.1f}+-{v.std(ddof=1):.1f}"
	print("== 1. 3-seed image AUROC / AP")
	for ds in DATASETS:
		print(f"{ds:11s} " + "  ".join(
			f"{n}: {ms([R[ds][n][s][0] for s in three])} / {ms([R[ds][n][s][1] for s in three]) if ds != 'loco' else '-'}"
			for n in READOUTS))
	print("\n== 2. Delta (Adopted - Map) per seed (42, 1, 2) and 4-seed spread (max - min)")
	for ds in DATASETS:
		dl = [R[ds]["Adopted"][s][0] - R[ds]["Map"][s][0] for s in three]
		sp = {n: max(R[ds][n][s][0] for s in seeds) - min(R[ds][n][s][0] for s in seeds) for n in ("Map", "B", "Adopted")}
		print(f"{ds:11s} Delta min {min(dl):+.1f} max {max(dl):+.1f} | spread Map {sp['Map']:.1f}  B {sp['B']:.1f}  Adopted {sp['Adopted']:.1f}"
			  f" | 3-seed means Single {np.mean([R[ds]['Single'][s][0] for s in three]):.1f} Map {np.mean([R[ds]['Map'][s][0] for s in three]):.1f}"
			  f" B {np.mean([R[ds]['B'][s][0] for s in three]):.1f} Adopted {np.mean([R[ds]['Adopted'][s][0] for s in three]):.1f}")
	print("\n== 3. seed 42 (transfer tables)")
	for ds in ("bmad_brain", "bmad_liver", "bmad_resc"):
		a, p, _, _ = R[ds]["Adopted"]["stab42"]
		print(f"{ds:11s} Adopted AUROC {a:.1f} AP {p:.1f}")
	a = [R[ds]["Adopted"]["stab42"][0] for ds in ("bmad_brain", "bmad_liver", "bmad_resc")]
	p = [R[ds]["Adopted"]["stab42"][1] for ds in ("bmad_brain", "bmad_liver", "bmad_resc")]
	print(f"BMAD mean   Adopted AUROC {np.mean(a):.1f} AP {np.mean(p):.1f}")
	m, _, L, S = R["loco"]["Adopted"]["stab42"]
	print(f"loco        Adopted mean {m:.1f}  L {L:.1f}  S {S:.1f}")
	print("\n== 4. MAD-Sim per defect type, 3-seed mean")
	for name in madsim:
		print(f"{name:13s} " + "  ".join(f"{t} {np.mean([madsim[name][s][t] for s in three]):.1f}" for t in ("Missing", "Burrs", "Stains")))


if __name__ == "__main__":
	main()
