#!/usr/bin/env python3
"""Collect the calibration-path hybrid and residual-quantization results.

Prints, for every dataset and every variant, all seven metrics and the delta
against the paired baseline run from the same job. Also pulls the residual
diagnostic out of the bsub logs, since it is logged rather than tabulated.

Usage:  python3 tools/collect_hybrid2.py [results_root] [bsub_log_glob]
"""
import glob
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "./results/hybrid2_stab42"
LOGS = sys.argv[2] if len(sys.argv) > 2 else "./bsub_log/hybrid2.*.stderr"

METRICS = ["px_auroc", "px_f1", "px_ap", "aupro", "img_auroc", "img_f1", "img_ap"]
DATASETS = ["visa", "mpdd", "mad_sim", "mad_real", "real_iad"]
# baseline first, then the variants in the order they are run
VARIANTS = [
    "base",
    "semcal_t0.05", "semcal_t0.1", "semcal_t0.2", "semcal_t0.4",
    "ctrl_t0.1", "ctrl_t0.4",
    "resid",
]


def read_mean(path):
    """Return the seven metrics from the '| mean |' row of a results log."""
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for line in fh:
            if line.startswith("| mean"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                vals = []
                for c in cells[1:]:
                    try:
                        vals.append(float(c))
                    except ValueError:
                        pass
                if len(vals) == len(METRICS):
                    return vals
    return None


def residual_lines():
    """Map (variant, dataset) -> residual diagnostic string from the job logs."""
    out = {}
    cur_ds = None
    pat_hdr = re.compile(r"^=====\s+(\S+)\s+(\S+)\s+rho=")
    pat_res = re.compile(r"residual diagnostic: (.+)$")
    for lg in sorted(glob.glob(LOGS)):
        with open(lg, errors="ignore") as fh:
            for line in fh:
                m = pat_hdr.search(line)
                if m:
                    cur_ds = (m.group(1), m.group(2))
                m = pat_res.search(line)
                if m and cur_ds:
                    out[cur_ds] = m.group(1).strip()
    return out


def main():
    resid = residual_lines()
    hdr = f"{'variant':14s}" + "".join(f"{m:>11s}" for m in METRICS)

    for ds in DATASETS:
        base = read_mean(os.path.join(ROOT, "base", ds, "log.txt"))
        if base is None:
            continue
        print(f"\n=== {ds} " + "=" * (len(hdr) - len(ds) - 5))
        print(hdr)
        for v in VARIANTS:
            vals = read_mean(os.path.join(ROOT, v, ds, "log.txt"))
            if vals is None:
                print(f"{v:14s}" + f"{'(pending)':>11s}")
                continue
            if v == "base":
                print(f"{v:14s}" + "".join(f"{x:11.1f}" for x in vals))
            else:
                row = f"{v:14s}"
                for x, b in zip(vals, base):
                    d = x - b
                    row += f"{x:7.1f}{d:+4.1f}"
                print(row)
        for (v, d), txt in sorted(resid.items()):
            if d == ds:
                print(f"  residual [{v}]: {txt}")

    print("\nEach cell: value then delta vs the paired baseline in the same job.")
    print("ctrl_* uses random directions of the same count, so semcal must beat")
    print("ctrl, not merely baseline, for the gain to be about semantics.")


if __name__ == "__main__":
    main()
