#!/usr/bin/env python3
"""VisA results of the improvement attempts (improve.bsub) against the baseline.

Every arm is the adopted read-out B on VisA, seeds 42, 1, 2. A gain counts only
if it exceeds twice the baseline's seed standard deviation, the rule of
Table 1; '*' marks such a difference, in either direction.
"""
import os

import numpy as np

R = "results/improve"
SEEDS = ["42", "1", "2"]
ARMS = [("baseline", "base/s{s}/visa"), ("flip-averaged map", "tta/s{s}/visa"),
		("672 px input", "res672/s{s}/visa"), ("896 px input", "res896/s{s}/visa"),
		("tile read-out (adopted)", "tiles/s{s}/visa"), ("tiles + flip-averaged map", "tiles_tta/s{s}/visa"),
		("0.50 epoch", "steps/s{s}/step_216/visa"), ("0.75 epoch", "steps/s{s}/step_324/visa"),
		("1.00 epoch (retrained)", "steps/s{s}/step_432/visa"), ("1.25 epochs", "steps/s{s}/step_540/visa"),
		("1.50 epochs", "steps/s{s}/step_648/visa"), ("per-layer codebooks", "perlayer/s{s}/visa")]
METRICS = ("Px", "PRO", "Img", "AP")


def row(path):
	"""(px, pro, img, ap) from the '| mean' row of a test.py log; None while unfinished."""
	p = os.path.join(R, path, "log.txt")
	if not os.path.isfile(p):
		return None
	rows = [l for l in open(p, errors="ignore") if "| mean" in l]
	if not rows:
		return None
	f = rows[-1].split("|")
	return tuple(float(f[i]) for i in (2, 5, 6, 8))


def main():
	base = np.array([r for r in (row(ARMS[0][1].format(s=s)) for s in SEEDS) if r is not None])
	print(f"{'arm':24s} " + " ".join(f"{m:>12s}" for m in METRICS) + "  seeds")
	for name, tmpl in ARMS:
		v = [row(tmpl.format(s=s)) for s in SEEDS]
		v = np.array([x for x in v if x is not None])
		if not len(v):
			print(f"{name:24s} pending")
			continue
		cells = []
		for k in range(4):
			c = f"{v[:, k].mean():.1f}" + (f"+-{v[:, k].std(ddof=1):.1f}" if len(v) > 1 else "")
			if name != "baseline" and len(base) > 1:
				d = v[:, k].mean() - base[:, k].mean()
				c += "*" if abs(d) > 2 * base[:, k].std(ddof=1) else " "
			cells.append(c)
		print(f"{name:24s} " + " ".join(f"{c:>12s}" for c in cells) + f"  {len(v)}")
	if len(base) > 1:
		print("\n'*': differs from the baseline by more than 2 x its seed std ("
			  + ", ".join(f"{m} {2 * base[:, k].std(ddof=1):.2f}" for k, m in enumerate(METRICS)) + ")")


if __name__ == "__main__":
	main()
