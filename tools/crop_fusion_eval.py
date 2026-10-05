#!/usr/bin/env python3
"""Does a tile-based global score improve read-out B? Selected on VisA only.

Read-out B is 0.25 * s_map + 0.75 * G, with G the frozen CLIP anomaly
probability of the whole image at 336 px (T=0.1, flip-averaged). The candidates
replace G by an aggregate of the same probability over g x g tiles
(tools/global_crop_scores.py), or average the two; weight and temperature stay
those of B. The candidate set is fixed here, before any result is seen:

  grid in {2, 3}  x  tile aggregate in {mean, max}  x  {replace G, average with G}

The VisA pick is reported on every target next to B, mean +- std over seeds.

Usage:
    python tools/crop_fusion_eval.py --seeds stab42 stab1 stab2
"""
import argparse
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DATASETS = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad", "bmad_brain", "bmad_liver", "bmad_resc", "loco"]
T = 0.1
W = 0.75


def anomaly_prob(cos, temp):
	"""cos [..., C] (NaN past a product's class count), class 0 = normal."""
	z = np.where(np.isnan(cos), -np.inf, cos / temp)
	z = z - z.max(axis=-1, keepdims=True)
	e = np.exp(z)
	return 1.0 - e[..., 0] / e.sum(axis=-1)


def metrics(ds, d, score):
	y, cls = d["label"].astype(int), d["cls_name"]
	if ds == "loco":		# pooled over classes, good vs each subset, as in the paper's LOCO table
		dc = d["defect_cls"]
		out = []
		for sub in ("logical_anomalies", "structural_anomalies"):
			m = (dc == "good") | (dc == sub)
			out.append(100 * roc_auc_score(y[m], score[m]))
		return np.mean(out), np.nan
	aucs = [100 * roc_auc_score(y[cls == c], score[cls == c]) for c in np.unique(cls)]
	aps = [100 * average_precision_score(y[cls == c], score[cls == c]) for c in np.unique(cls)]
	return np.mean(aucs), np.mean(aps)


def candidates(d, crops):
	ms = d["map_topk"][:, list(d["sigmas"]).index(4.0), :].mean(axis=1)
	g_full = (anomaly_prob(d["g_cos336"], T) + anomaly_prob(d["g_cos336_flip"], T)) / 2
	out = {"B": (1 - W) * ms + W * g_full}
	for grid, c in crops.items():
		p = (anomaly_prob(c["g_crop_cos"], T) + anomaly_prob(c["g_crop_cos_flip"], T)) / 2		# [N, tiles]
		for agg, a in (("mean", p.mean(axis=1)), ("max", p.max(axis=1))):
			out[f"g{grid} {agg} replace"] = (1 - W) * ms + W * a
			out[f"g{grid} {agg} avg"] = (1 - W) * ms + W * (g_full + a) / 2
	return out


def load(args, seed, ds):
	p = os.path.join(args.score_root, seed, ds + ".npz")
	if not os.path.isfile(p):
		return None, None
	d = dict(np.load(p, allow_pickle=False))
	g = np.load(os.path.join(args.global_root, ds + ".npz"))
	assert (g["img_path"] == d["img_path"]).all(), f"image order differs: {ds} global"
	d["g_cos336"], d["g_cos336_flip"] = g["g_cos"], g["g_cos_flip"]
	crops = {}
	for grid in (2, 3):
		pc = os.path.join(args.crop_root, f"g{grid}", ds + ".npz")
		if os.path.isfile(pc):
			c = np.load(pc)
			assert (c["img_path"] == d["img_path"]).all(), f"image order differs: {ds} crop g{grid}"
			crops[grid] = {k: c[k] for k in ("g_crop_cos", "g_crop_cos_flip")}
	return d, crops


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--seeds", nargs="+", default=["stab42", "stab1", "stab2"])
	ap.add_argument("--score_root", default="results/imgscore")
	ap.add_argument("--global_root", default="results/globalnative_fp32full/336")
	ap.add_argument("--crop_root", default="results/globalcrop")
	args = ap.parse_args()

	res = {}		# res[ds][name] -> list over seeds of (auroc, ap)
	for ds in DATASETS:
		for seed in args.seeds:
			d, crops = load(args, seed, ds)
			if d is None:
				continue
			for name, s in candidates(d, crops).items():
				res.setdefault(ds, {}).setdefault(name, []).append(metrics(ds, d, s))

	def fmt(v):
		v = np.asarray(v, dtype=float)
		return f"{v.mean():.1f}+-{v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"

	print("VisA image AUROC (selection set), mean +- std over seeds:")
	visa = res["visa"]
	for name, v in sorted(visa.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]])):
		print(f"  {name:18s} {fmt([x[0] for x in v])}   AP {fmt([x[1] for x in v])}")
	full = [n for n in visa if n != "B" and len(visa[n]) == len(visa["B"])]
	if not full:
		return
	pick = max(full, key=lambda n: np.mean([x[0] for x in visa[n]]))
	b = np.array([x[0] for x in visa["B"]])
	gain = np.mean([x[0] for x in visa[pick]]) - b.mean()
	print(f"\nVisA pick: {pick}  gain {gain:+.2f}  (2 x std of B = {2 * b.std(ddof=1):.2f})")
	print(f"\n{'dataset':11s} {'B AUROC':>12s} {'pick AUROC':>12s} {'B AP':>12s} {'pick AP':>12s}")
	for ds in DATASETS:
		if ds in res and pick in res[ds]:
			r = res[ds]
			print(f"{ds:11s} {fmt([x[0] for x in r['B']]):>12s} {fmt([x[0] for x in r[pick]]):>12s} "
				  f"{fmt([x[1] for x in r['B']]):>12s} {fmt([x[1] for x in r[pick]]):>12s}")


if __name__ == "__main__":
	main()
