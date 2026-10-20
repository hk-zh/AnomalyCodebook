#!/usr/bin/env python3
"""Numbers for the 2026-10-10 revision, from revision.bsub's outputs.

  quantized/<variant>   temperature controls on the adopted checkpoints
                        (results/rev/ctrl_s<seed>/<ds>/variants.json); sched is the reported model
  nocb/fixed:0.01       the codebook removed, no calibration (results/rev/nocb_s<seed>/<ds>)

Every value is a per-product mean, then mean +- sample SD over the seeds found.

Usage:
    python tools/revision_tables.py [--seeds 42 1 2 3] [--datasets visa mpdd mad_sim mad_real real_iad]
"""
import argparse
import json
import os

import numpy as np

ROOT = "results/rev"
KEYS = ("auroc_px", "aupro", "auroc_sp", "ap_sp")


def variants(tag, seed, ds):
	p = os.path.join(ROOT, f"{tag}_s{seed}", ds, "variants.json")
	return json.load(open(p)) if os.path.isfile(p) else None


def fmt(vals):
	v = np.array(vals, float)
	if len(v) == 0:
		return "   --      "
	sd = v.std(ddof=1) if len(v) > 1 else float("nan")
	return f"{v.mean():5.1f}+-{sd:3.1f}({len(v)})"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--seeds", nargs="+", default=["42", "1", "2", "3"])
	ap.add_argument("--datasets", nargs="+", default=["visa", "mpdd", "mad_sim", "mad_real", "real_iad"])
	ap.add_argument("--keys", nargs="+", default=list(KEYS))
	args = ap.parse_args()

	rows = {}		# rows[name][ds][key] = [per-seed values]
	def add(name, ds, rec):
		if rec is None:
			return
		for k in KEYS:
			rows.setdefault(name, {}).setdefault(ds, {}).setdefault(k, []).append(rec[k])

	for ds in args.datasets:
		for s in args.seeds:
			for tag, label in (("ctrl", "quantized"), ("nocb", "nocb")):
				for name, rec in (variants(tag, s, ds) or {}).items():
					add(f"{label}/{name}", ds, rec)

	for key in args.keys:
		print(f"\n== {key}   (mean+-sd(n seeds))")
		print(f"{'variant':28s} " + " ".join(f"{ds:>15s}" for ds in args.datasets) + f" {'mean of means':>14s}")
		for name in rows:
			cells = [rows[name].get(ds, {}).get(key, []) for ds in args.datasets]
			full = all(len(c) for c in cells)
			mm = f"{np.mean([np.mean(c) for c in cells]):14.2f}" if full else f"{'':14s}"
			print(f"{name:28s} " + " ".join(f"{fmt(c):>15s}" for c in cells) + mm)


if __name__ == "__main__":
	main()
