#!/usr/bin/env python3
"""Offline analysis of the candidate image scores from tools/image_score_study.py.

Selection protocol: the candidate family is enumerated below in full, every
member is scored on every dataset and seed, and the one we adopt is the member
with the best VisA image AUROC averaged over seeds (image AP breaks ties). VisA
is the declared validation target of the paper; every other column is
report-only. (The weight grid and the map bases of the fusions were widened
after a first look at seed 42, where the VisA optimum sat on the grid edge. The
native-resolution global scores G336 and G336f were added after the four-seed
run showed the 518 px global score to be weak on its own. The trained head
H336/H336f of tools/train_global_head.py was added after that, and after it,
once the head was seen to hurt the targets far from MVTec, its WiSE-FT
interpolations HW<lam>f toward frozen CLIP (tools/global_head_wise.py) and the
plain average GHf of G336f and H336f. Last came the multi-scale map read-out
MS, the top-k mean averaged over the rho grid image_score_study.py stores,
after a per-defect-type look at MPDD showed small rho failing on large defects.) Nothing here uses target labels to
set a constant, and no score is normalized with statistics of the target test
set (fusions are convex combinations of quantities that are already
probabilities, or products of them).

Usage:
    python tools/image_score_fusion.py --root results/imgscore --tags stab42 stab1 stab2 stab3
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

DATASETS = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad",
			"bmad_brain", "bmad_liver", "bmad_resc", "loco"]
TEMPS = (0.01, 0.02, 0.05, 0.1)
WEIGHTS = (0.1, 0.25, 0.5, 0.6, 0.75, 0.9)
FUSE_RHOS = (0.0005, 0.001, 0.005, 0.01, 0.05)
NATIVE_SIZES = (336,)					# tools/global_native_scores.py resolutions
FROZEN_G = re.compile(r"^(G(\d+f?)?|H\d+f?|HW[\d.]+f|GHf)@")	# global scores: frozen G@, G336@, G336f@;
															# trained head H336@, H336f@, HW<lam>f@; average GHf@


def anomaly_prob(cos, T):
	"""cos [N, C] (NaN past a product's class count), class 0 = normal."""
	z = np.where(np.isnan(cos), -np.inf, cos / T)
	z = z - z.max(axis=1, keepdims=True)
	e = np.exp(z)
	return 1.0 - e[:, 0] / e.sum(axis=1)


def candidates(d):
	"""-> {name: [N] score}. Every entry is label-free."""
	rhos = list(d["rhos"])
	sig = list(d["sigmas"])
	lr = list(d["layer_rhos"])
	c = {}
	for si, s in enumerate(sig):
		for ri, r in enumerate(rhos):
			c[f"M@{r}/s{int(s)}"] = d["map_topk"][:, si, ri]
		# multi-scale read-out: top-k mean averaged over every stored rho, so no rho is chosen
		c[f"MS/s{int(s)}"] = d["map_topk"][:, si, :].mean(axis=1)
	c["Mmean"] = d["map_mean"]
	nL = d["layer_topk"].shape[1]
	for ri, r in enumerate(lr):
		for l in range(nL):
			c[f"L{l}@{r}"] = d["layer_topk"][:, l, ri]
		c[f"Lmax@{r}"] = d["layer_topk"][:, :, ri].max(axis=1)
	for T in TEMPS:
		c[f"G@{T}"] = anomaly_prob(d["g_cos"], T)
		for key, nm in (("cls_cos", "CLS"), ("clsq_cos", "CLSQ"), ("meanq_cos", "MQ")):
			per = np.stack([anomaly_prob(d[key][:, l], T) for l in range(nL)], 1)
			for l in range(nL):
				c[f"{nm}{l}@{T}"] = per[:, l]
			c[f"{nm}avg@{T}"] = per.mean(axis=1)
	for sz in NATIVE_SIZES:
		if f"g{sz}_cos" in d:
			for T in TEMPS:
				p = anomaly_prob(d[f"g{sz}_cos"], T)
				c[f"G{sz}@{T}"] = p
				c[f"G{sz}f@{T}"] = (p + anomaly_prob(d[f"g{sz}_cos_flip"], T)) / 2
	if "h_cos" in d:
		for T in TEMPS:
			p = anomaly_prob(d["h_cos"], T)
			c[f"H336@{T}"] = p
			c[f"H336f@{T}"] = (p + anomaly_prob(d["h_cos_flip"], T)) / 2
			if "g336_cos" in d:
				c[f"GHf@{T}"] = (c[f"G336f@{T}"] + c[f"H336f@{T}"]) / 2
	for k in [k for k in d if k.startswith("hw") and k.endswith("_cos")]:
		lam = k[2:-4]
		for T in TEMPS:
			c[f"HW{lam}f@{T}"] = (anomaly_prob(d[k], T) + anomaly_prob(d[k + "_flip"], T)) / 2
	for key, nm in (("ga_cos", "GA"), ("gq_cos", "GQ")):
		if key in d:
			for T in TEMPS:
				c[f"{nm}@{T}"] = anomaly_prob(d[key], T)
	for key in ("nov_max", "nov_top1pct", "nov_mean", "ent_max", "ent_top1pct", "ent_mean"):
		for l in range(nL):
			c[f"{key}{l}"] = d[key][:, l]
		c[f"{key}_avg"] = d[key].mean(axis=1)

	# fusions of the headline map score with every global-type signal
	base = {"M": c["M@0.001/s4"]}
	base.update({f"M{r}": c[f"M@{r}/s4"] for r in FUSE_RHOS if r != 0.001})
	base["MS"] = c["MS/s4"]
	glob = [k for k in c if FROZEN_G.match(k) or k.startswith(("GA@", "GQ@", "CLSavg@", "CLSQavg@", "MQavg@"))]
	glob += [f"CLS{l}@{T}" for l in range(nL) for T in TEMPS]
	for bn, b in base.items():
		for g in glob:
			if bn != "M" and not FROZEN_G.match(g):
				continue
			x = c[g]
			for w in WEIGHTS:
				c[f"{bn}+{w}*{g}"] = (1 - w) * b + w * x
			c[f"{bn}*{g}"] = np.sqrt(np.clip(b, 0, None) * np.clip(x, 0, None))
	return c


def per_class_metric(labels, cls, score, defect=None, keep=None):
	aucs, aps = [], []
	for k in sorted(set(cls)):
		m = cls == k
		if keep is not None:
			m = m & keep
		y = labels[m]
		if len(set(y)) < 2:
			continue
		aucs.append(roc_auc_score(y, score[m]))
		aps.append(average_precision_score(y, score[m]))
	return (100 * np.mean(aucs), 100 * np.mean(aps)) if aucs else (np.nan, np.nan)


def evaluate(d, score, dataset):
	labels = d["label"].astype(int)
	cls = d["cls_name"]
	if dataset == "loco" and "defect_cls" in d:
		dc = d["defect_cls"]
		lo = per_class_metric(labels, cls, score, keep=(dc == "good") | (dc == "logical_anomalies"))
		st = per_class_metric(labels, cls, score, keep=(dc == "good") | (dc == "structural_anomalies"))
		return ((lo[0] + st[0]) / 2, (lo[1] + st[1]) / 2)
	return per_class_metric(labels, cls, score)


def load_scores(args, tag, ds):
	"""imgscore .npz of one checkpoint and dataset, joined with every optional
	global-score output found under args.*_root. None if the imgscore file is missing."""
	p = os.path.join(args.root, tag, ds + ".npz")
	if not os.path.isfile(p):
		return None
	d = dict(np.load(p, allow_pickle=False))
	pg = os.path.join(args.globaltok_root, tag, ds + ".npz")
	if os.path.isfile(pg):
		g = np.load(pg, allow_pickle=False)
		assert (g["img_path"] == d["img_path"]).all(), f"image order differs: {pg}"
		d["ga_cos"], d["gq_cos"] = g["ga_cos"], g["gq_cos"]
	ph = os.path.join(args.head_root, tag, ds + ".npz")
	if os.path.isfile(ph):
		g = np.load(ph, allow_pickle=False)
		assert (g["img_path"] == d["img_path"]).all(), f"image order differs: {ph}"
		d["h_cos"], d["h_cos_flip"] = g["h_cos"], g["h_cos_flip"]
	pw = os.path.join(args.wise_root, tag, ds + ".npz")
	if os.path.isfile(pw):
		g = np.load(pw, allow_pickle=False)
		assert (g["img_path"] == d["img_path"]).all(), f"image order differs: {pw}"
		d.update({k: g[k] for k in g.files if k.startswith("hw")})
	for sz in NATIVE_SIZES + (518,):
		pn = os.path.join(args.native_root, str(sz), ds + ".npz")
		if not os.path.isfile(pn):
			continue
		g = np.load(pn, allow_pickle=False)
		assert (g["img_path"] == d["img_path"]).all(), f"image order differs: {pn}"
		if sz == 518:	# same resolution as image_score_study.py: must reproduce g_cos
			dev = np.nanmax(np.abs(g["g_cos"] - d["g_cos"]))
			print(f"[check] {tag}/{ds}: 518 px g_cos max |diff| vs imgscore = {dev:.2e}")
			continue
		d[f"g{sz}_cos"], d[f"g{sz}_cos_flip"] = g["g_cos"], g["g_cos_flip"]
	return d


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--root", default="results/imgscore")
	ap.add_argument("--tags", nargs="+", required=True)
	ap.add_argument("--globaltok_root", default="results/globaltok",
					help="optional tools/global_token_scores.py output, joined on img_path")
	ap.add_argument("--native_root", default="results/globalnative",
					help="optional tools/global_native_scores.py output (seed-independent), joined on img_path")
	ap.add_argument("--head_root", default="results/globalhead",
					help="optional tools/train_global_head.py output, one head per tag, joined on img_path")
	ap.add_argument("--wise_root", default="results/globalhead_wise",
					help="optional tools/global_head_wise.py output, joined on img_path")
	ap.add_argument("--select_on", default="visa")
	ap.add_argument("--show", type=int, default=40, help="rows to print, by the selection column")
	ap.add_argument("--grep", default=None, help="also print every candidate whose name contains this")
	ap.add_argument("--json", default="")
	args = ap.parse_args()

	# res[cand][dataset] -> list over tags of (auroc, ap)
	res = defaultdict(lambda: defaultdict(list))
	present = defaultdict(list)
	for tag in args.tags:
		for ds in DATASETS:
			d = load_scores(args, tag, ds)
			if d is None:
				continue
			present[ds].append(tag)
			for name, s in candidates(d).items():
				res[name][ds].append(evaluate(d, s, ds))

	cols = [ds for ds in DATASETS if present[ds]]
	print("seeds per dataset:", {ds: len(present[ds]) for ds in cols})

	def mean(name, ds, i=0):
		v = res[name][ds]
		return np.mean([x[i] for x in v]) if v else np.nan

	def bmad(name, i=0):
		v = [mean(name, b, i) for b in ("bmad_brain", "bmad_liver", "bmad_resc")]
		return np.mean(v) if not any(np.isnan(v)) else np.nan

	show_cols = [c for c in cols if not c.startswith("bmad_")]
	has_bmad = any(c.startswith("bmad_") for c in cols)
	hdr = f"{'candidate':<32}" + "".join(f"{c[:8]:>9}" for c in show_cols)
	hdr += f"{'bmad':>9}" if has_bmad else ""
	hdr += f"{'mean5':>9}"

	def row(name):
		vals = [mean(name, c) for c in show_cols]
		s = f"{name:<32}" + "".join(f"{v:>9.1f}" for v in vals)
		if has_bmad:
			s += f"{bmad(name):>9.1f}"
		core = [mean(name, c) for c in ("visa", "mpdd", "mad_sim", "mad_real", "real_iad") if c in cols]
		s += f"{np.mean(core):>9.1f}"
		return s

	# rank only candidates scored on every seed of the selection dataset
	full = [n for n in res if len(res[n][args.select_on]) == len(present[args.select_on])]
	names = sorted(full, key=lambda n: (-mean(n, args.select_on), -mean(n, args.select_on, 1)))
	print("\n=== headline read-out")
	print(hdr)
	print(row("M@0.001/s4"))
	print(f"\n=== top {args.show} by {args.select_on} image AUROC (selection column)")
	print(hdr)
	for n in names[:args.show]:
		print(row(n))
	if args.grep is not None:
		print(f"\n=== candidates matching '{args.grep}'")
		print(hdr)
		for n in names:
			if args.grep in n:
				print(row(n))

	win = names[0]
	print(f"\nSELECTED on {args.select_on}: {win}")
	for c in cols:
		v = res[win][c]
		a = np.array(v)
		print(f"  {c:<11} AUROC {a[:, 0].mean():5.1f} +- {a[:, 0].std(ddof=1) if len(a) > 1 else 0:.1f}"
			  f"   AP {a[:, 1].mean():5.1f} +- {a[:, 1].std(ddof=1) if len(a) > 1 else 0:.1f}   (n={len(a)})")
	if args.json:
		json.dump({n: {c: res[n][c] for c in cols} for n in res}, open(args.json, "w"))


if __name__ == "__main__":
	main()
